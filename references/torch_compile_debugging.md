# torch.compile Debugging

Use this reference when a workload already uses `torch.compile` but speedup is weak, unstable, or shape-dependent.

## Log Commands

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" python <entrypoint> 2>&1 | tee artifacts/torch_logs/compile.log
TORCH_TRACE=artifacts/traces/torch_trace python <entrypoint>
TORCH_COMPILE_DEBUG=1 python <entrypoint>
```

Use `scripts/analyze_torch_logs.py` for a quick count, then inspect exact sites manually.

## Graph Break Triage

Common causes:

- `.item()` or tensor value entering Python control flow.
- `print`, logging, progress bars, assertions, timers in compiled functions.
- Python container mutation in the hot path.
- Third-party library calls inside forward.
- Data-dependent branching that can be expressed as tensor operations.
- CPU/GPU synchronization from `.cpu()`, `.numpy()`, or scalar extraction.

Fix order:

1. Move side effects outside compiled code.
2. Rewrite scalar Python control as tensor operations.
3. Compile a smaller stable region if only one submodule is problematic.
4. Mark unsupported cold code with `torch.compiler.disable`.

## Recompilation Triage

Common causes:

- changing batch or sequence shapes,
- changing dtype, device, layout, or stride,
- Python bools or constants changing inside compiled calls,
- mutable module attributes,
- lists/dicts with varying length or keys,
- dropout/train/eval changes crossing a compiled boundary.

Fix order:

1. Pad or bucket input shapes.
2. Keep Python constants outside compiled function arguments.
3. Use precise `torch._dynamo.mark_dynamic` on only the dimensions that need it.
4. Try `dynamic=True` only after targeted fixes fail.

## Compile Modes

Compare one at a time:

```python
model = torch.compile(model)
model = torch.compile(model, mode="reduce-overhead")
model = torch.compile(model, mode="max-autotune")
```

Record compile time, warm steady-state performance, peak memory, and correctness. `max-autotune` can improve steady-state while increasing compile time. `reduce-overhead` can reduce launch overhead while increasing memory pressure.
