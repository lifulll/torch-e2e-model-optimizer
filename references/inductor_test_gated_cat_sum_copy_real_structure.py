import os
import statistics
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


REAL_MIXER_SHAPES = (
    pytest.param(32, id="seq32"),
    pytest.param(64, id="seq64"),
    pytest.param(128, id="seq128"),
    pytest.param(256, id="seq256"),
)


def _make_real_inputs(device, seq):
    # Shapes and strides mirror the debug graph around the four real fused sites:
    # seq=32/64/128/256, channels=16, hidden=256, copy hidden=512.
    batch = _env_int("MIXER_GATED_CAT_BATCH", 150)
    channels = _env_int("MIXER_GATED_CAT_CHANNELS", 16)
    hidden = _env_int("MIXER_GATED_CAT_HIDDEN", 256)
    out_features = _env_int("MIXER_GATED_CAT_BMM_N", 128)

    torch.manual_seed(20260518 + seq)

    base_storage = torch.empty_strided(
        (channels, batch * seq, hidden),
        (batch * seq * hidden, hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    projected = torch.empty_strided(
        (batch, seq, channels, hidden * 2),
        (seq * hidden, hidden, batch * seq * hidden * 2, 1),
        device=device,
        dtype=torch.float32,
    )
    base_storage.normal_(0.0, 0.1)
    projected.normal_(0.0, 0.1)

    gate = projected[..., :hidden]
    value = projected[..., hidden:]
    left_weight = torch.randn(
        channels,
        out_features,
        batch * seq,
        device=device,
        dtype=torch.bfloat16,
    )
    right_weight = torch.randn(
        channels,
        hidden * 2,
        out_features,
        device=device,
        dtype=torch.bfloat16,
    )
    return base_storage, value, gate, left_weight, right_weight


def _real_gated_cat_sum_copy_model(
    base_storage, value, gate, left_weight, right_weight
):
    batch, seq, channels, hidden = value.shape
    base = base_storage.as_strided(
        (batch, seq, channels, hidden),
        (seq * hidden, hidden, batch * seq * hidden, 1),
    )
    sig = torch.sigmoid(gate)
    silu = gate * sig
    right = base.to(torch.float32) * silu

    value_grad = base.to(torch.float32) * value
    gate_grad = gate * (1.0 - sig)
    gate_grad = sig * (gate_grad + 1.0)
    left = value_grad * gate_grad

    cat = torch.cat([left, right], dim=3)
    copy = cat.to(torch.bfloat16)
    summed = cat.sum((0, 1), keepdim=True)

    bmm_in = (
        copy.view(batch, seq, channels, hidden * 2, 1)
        .permute(2, 0, 1, 4, 3)
        .view(channels, batch * seq, hidden * 2)
    )
    bmm_left = torch.bmm(left_weight, bmm_in)
    bmm_right = torch.bmm(bmm_in, right_weight)
    return copy, summed, bmm_left, bmm_right


def _install_pass_counter(monkeypatch):
    import torch._inductor.fx_passes.gated_cat_sum_copy_scheduler as pass_mod

    calls = []
    original = pass_mod.fuse_cat_copy_for_sum_copy_users

    def counted(nodes):
        out = original(nodes)
        calls.append(
            sum(
                1
                for node in out
                if "gated_cat_sum_copy" in getattr(node, "get_name", lambda: "")()
            )
        )
        return out

    monkeypatch.setattr(pass_mod, "fuse_cat_copy_for_sum_copy_users", counted)
    return calls


def _compile_and_run(monkeypatch, enabled, inputs):
    import torch._inductor.config as inductor_config
    import torch._inductor.utils as inductor_utils

    torch._dynamo.reset()
    inductor_utils.clear_inductor_caches()
    monkeypatch.setattr(inductor_config, "force_disable_caches", True)
    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    if enabled:
        monkeypatch.setenv("CAT_SUM_COPY_SCHEDULER_ENABLE", "1")
        calls = _install_pass_counter(monkeypatch)
    else:
        monkeypatch.delenv("CAT_SUM_COPY_SCHEDULER_ENABLE", raising=False)
        calls = []

    compiled = torch.compile(
        _real_gated_cat_sum_copy_model,
        backend="inductor",
        fullgraph=True,
    )
    outputs = compiled(*inputs)
    torch.cuda.synchronize()
    return compiled, outputs, calls


def _assert_close_outputs(pass_outputs, baseline_outputs):
    tolerances = [
        # copy output is bf16 materialization of the same cat expression.
        dict(atol=2e-2, rtol=2e-2),
        # summed is fp32, but reduction order differs after fusion.
        dict(atol=3e-1, rtol=3e-2),
        dict(atol=3e-1, rtol=3e-2),
        dict(atol=3e-1, rtol=3e-2),
    ]
    for actual, expected, kwargs in zip(pass_outputs, baseline_outputs, tolerances):
        torch.testing.assert_close(actual, expected, **kwargs)


def _benchmark(compiled, inputs):
    warmup = _env_int("MIXER_GATED_CAT_WARMUP", 3)
    iters = _env_int("MIXER_GATED_CAT_ITERS", 10)
    repeats = _env_int("MIXER_GATED_CAT_REPEATS", 3)

    for _ in range(warmup):
        compiled(*inputs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            compiled(*inputs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
        time.sleep(0.05)
    return statistics.median(samples)


def _profile_target_ms(compiled, inputs):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        compiled(*inputs)
    torch.cuda.synchronize()

    target_ms = 0.0
    for event in prof.key_averages():
        if event.device_type != torch.profiler.DeviceType.CUDA:
            continue
        name = event.key
        if (name.startswith("triton_") or name.startswith("inductor_gated_")) and "bmm" not in name:
            target_ms += event.device_time_total / 1000.0
    return target_ms


@pytest.mark.parametrize("seq", REAL_MIXER_SHAPES)
def test_real_gated_cat_sum_copy_precision_and_perf(monkeypatch, seq):
    inputs = _make_real_inputs("cuda", seq)

    baseline_compiled, baseline_outputs, baseline_calls = _compile_and_run(
        monkeypatch, False, inputs
    )
    baseline_ms = _benchmark(baseline_compiled, inputs)
    baseline_target_ms = _profile_target_ms(baseline_compiled, inputs)

    pass_compiled, pass_outputs, pass_calls = _compile_and_run(monkeypatch, True, inputs)
    pass_ms = _benchmark(pass_compiled, inputs)
    pass_target_ms = _profile_target_ms(pass_compiled, inputs)

    assert baseline_calls == []
    assert pass_calls and max(pass_calls) >= 1
    _assert_close_outputs(pass_outputs, baseline_outputs)

    speedup = baseline_ms / pass_ms
    target_speedup = baseline_target_ms / pass_target_ms

    print(
        f"cat/sum/to_copy kernels seq={seq}: baseline={baseline_target_ms:.3f} ms "
        f"pass={pass_target_ms:.3f} ms speedup={target_speedup:.3f}x"
    )
    print(
        f"end-to-end seq={seq}: baseline={baseline_ms:.3f} ms "
        f"pass={pass_ms:.3f} ms speedup={speedup:.3f}x"
    )
    if _env_flag("MIXER_GATED_CAT_REQUIRE_SPEEDUP"):
        assert pass_target_ms < baseline_target_ms
    else:
        assert baseline_ms > 0.0
        assert pass_ms > 0.0
