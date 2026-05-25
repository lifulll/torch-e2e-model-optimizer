"""
inductor_pass_compile_test_template.py
--------------------------------------
Template for validating an Inductor pass under torch.compile.

Use this when the model code stays the same, but an Inductor pass rewrites a
matched graph pattern. For example, the model code is:

    return a.add(b) - c

and the pass replaces that pattern with a custom generated path. The template
compares:

    baseline path: pass disabled by monkeypatch
    pass path:     pass enabled by monkeypatch

Both paths run with default torch.compile mode, the same fixed inputs, dtype-aware
accuracy tolerances, and steady-state benchmark timing. It intentionally does not
tune AMP, precision, batch size, or microbatching; keep those fixed for attribution.

Default tolerances are dtype-aware:

    fp32/other: atol=1e-6, rtol=1e-5
    fp16:       atol=1e-3, rtol=1e-3
    bf16:       atol=2e-2, rtol=2e-2

Usage:
    1. Copy this file into the target repo's test or benchmark area.
    2. Replace model_impl(), make_inputs(), disable_pass(), and enable_pass().
    3. Run:
        python inductor_pass_compile_test_template.py --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch._dynamo


@dataclass
class AccuracyResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float


@dataclass
class BenchmarkResult:
    name: str
    compile_first_call_s: float
    p50_s: float
    p90_s: float
    mean_s: float
    runs: int


DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

DEFAULT_TOLERANCE: tuple[float, float] = (1e-6, 1e-5)


def model_impl(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Replace with model code that should trigger the Inductor pass pattern."""
    return a.add(b) - c


def make_inputs(device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Create one fixed representative input set for both pass states."""
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    c = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    return (a, b, c), {}


def disable_pass() -> contextlib.AbstractContextManager[None]:
    """
    Monkeypatch context that disables the pass under test.

    Replace this with project-specific monkeypatching. Examples:

        @contextlib.contextmanager
        def disable_pass():
            import torch._inductor.fx_passes.post_grad as post_grad
            old = post_grad.fuse_my_pass
            post_grad.fuse_my_pass = lambda graph: None
            try:
                yield
            finally:
                post_grad.fuse_my_pass = old

    or patch a config flag that gates your pass.
    """
    return contextlib.nullcontext()


def enable_pass() -> contextlib.AbstractContextManager[None]:
    """
    Monkeypatch context that enables the pass under test.

    Replace this with project-specific monkeypatching. If the pass is enabled by
    default, this can stay as nullcontext().
    """
    return contextlib.nullcontext()


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        out: list[torch.Tensor] = []
        for item in value:
            out.extend(flatten_tensors(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten_tensors(value[key]))
        return out
    raise TypeError(f"Unsupported output type for comparison: {type(value)!r}")


def compare_outputs(
    expected: Any,
    actual: Any,
    *,
    atol: float,
    rtol: float,
) -> AccuracyResult:
    expected_tensors = flatten_tensors(expected)
    actual_tensors = flatten_tensors(actual)
    if len(expected_tensors) != len(actual_tensors):
        raise AssertionError(
            f"Output tensor count mismatch: {len(expected_tensors)} vs {len(actual_tensors)}"
        )

    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for ref, new in zip(expected_tensors, actual_tensors):
        if ref.shape != new.shape:
            raise AssertionError(f"Output shape mismatch: {ref.shape} vs {new.shape}")
        diff = (ref - new).detach().abs()
        abs_err = float(diff.max().item()) if diff.numel() else 0.0
        denom = ref.detach().abs().clamp_min(1e-12)
        rel_err = float((diff / denom).max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        passed = passed and torch.allclose(ref, new, atol=atol, rtol=rtol)

    return AccuracyResult(
        passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
    )


def infer_tolerance(outputs: Any) -> tuple[float, float]:
    tensors = flatten_tensors(outputs)
    atol, rtol = DEFAULT_TOLERANCE
    for tensor in tensors:
        candidate = DTYPE_TOLERANCES.get(tensor.dtype)
        if candidate is None:
            continue
        atol = max(atol, candidate[0])
        rtol = max(rtol, candidate[1])
    return atol, rtol


def compile_callable(fn):
    return torch.compile(fn, mode="default", backend="inductor")


def reset_compile_state() -> None:
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_once(fn, args, kwargs):
    with torch.inference_mode():
        return fn(*args, **kwargs)


def compiled_output(
    patch_context: contextlib.AbstractContextManager[None],
    args,
    kwargs,
) -> Any:
    reset_compile_state()
    with patch_context:
        compiled = compile_callable(model_impl)
        return run_once(compiled, args, kwargs)


def benchmark(
    name: str,
    patch_context_factory,
    args,
    kwargs,
    *,
    device: torch.device,
    warmup: int,
    runs: int,
) -> BenchmarkResult:
    reset_compile_state()
    with patch_context_factory():
        compiled = compile_callable(model_impl)

        sync_if_needed(device)
        t0 = time.perf_counter()
        run_once(compiled, args, kwargs)
        sync_if_needed(device)
        first_call = time.perf_counter() - t0

        for _ in range(warmup):
            run_once(compiled, args, kwargs)
        sync_if_needed(device)

        times = []
        for _ in range(runs):
            sync_if_needed(device)
            start = time.perf_counter()
            run_once(compiled, args, kwargs)
            sync_if_needed(device)
            times.append(time.perf_counter() - start)

    sorted_times = sorted(times)
    p90_idx = min(int(0.9 * len(sorted_times)), len(sorted_times) - 1)
    return BenchmarkResult(
        name=name,
        compile_first_call_s=first_call,
        p50_s=statistics.median(times),
        p90_s=sorted_times[p90_idx],
        mean_s=statistics.mean(times),
        runs=runs,
    )


def format_report(
    accuracy: AccuracyResult,
    baseline_bench: BenchmarkResult,
    pass_bench: BenchmarkResult,
) -> str:
    speedup = baseline_bench.p50_s / pass_bench.p50_s if pass_bench.p50_s else 0.0
    lines = [
        "Inductor Pass torch.compile Test",
        "=" * 72,
        f"Accuracy passed: {accuracy.passed}",
        f"max_abs_error:   {accuracy.max_abs_error:.6g}  (atol={accuracy.atol:g})",
        f"max_rel_error:   {accuracy.max_rel_error:.6g}  (rtol={accuracy.rtol:g})",
        "",
        f"{'Path':<12} {'first_call(s)':>14} {'p50(s)':>12} {'p90(s)':>12} {'mean(s)':>12}",
        "-" * 72,
    ]
    for result in [baseline_bench, pass_bench]:
        lines.append(
            f"{result.name:<12} {result.compile_first_call_s:>14.6f} "
            f"{result.p50_s:>12.6f} {result.p90_s:>12.6f} {result.mean_s:>12.6f}"
        )
    lines.extend(
        [
            "-" * 72,
            f"pass speedup vs baseline, p50: {speedup:.4f}x",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--json-out", default="")
    args_ns = parser.parse_args()

    device = torch.device(args_ns.device)
    call_args, call_kwargs = make_inputs(device)

    baseline_output = compiled_output(disable_pass(), call_args, call_kwargs)
    pass_output = compiled_output(enable_pass(), call_args, call_kwargs)
    default_atol, default_rtol = infer_tolerance(baseline_output)
    atol = default_atol if args_ns.atol is None else args_ns.atol
    rtol = default_rtol if args_ns.rtol is None else args_ns.rtol
    accuracy = compare_outputs(baseline_output, pass_output, atol=atol, rtol=rtol)

    baseline_bench = benchmark(
        "baseline",
        disable_pass,
        call_args,
        call_kwargs,
        device=device,
        warmup=args_ns.warmup,
        runs=args_ns.runs,
    )
    pass_bench = benchmark(
        "pass",
        enable_pass,
        call_args,
        call_kwargs,
        device=device,
        warmup=args_ns.warmup,
        runs=args_ns.runs,
    )

    print(format_report(accuracy, baseline_bench, pass_bench))

    if args_ns.json_out:
        payload = {
            "accuracy": asdict(accuracy),
            "baseline": asdict(baseline_bench),
            "pass": asdict(pass_bench),
            "speedup_p50": baseline_bench.p50_s / pass_bench.p50_s
            if pass_bench.p50_s
            else 0.0,
        }
        out = Path(args_ns.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if accuracy.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
