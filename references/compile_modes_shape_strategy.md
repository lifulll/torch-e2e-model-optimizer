# torch.compile Modes And Shape Strategy

Use this reference when choosing `torch.compile` mode, debugging CUDA graph applicability, or controlling recompilation caused by variable shapes.

## Mode Selection

Start with `mode="default"` unless evidence points elsewhere.

```text
Start: mode="default"
  -> If Python/launch overhead dominates and shapes are fixed: try "reduce-overhead"
  -> If long production run amortizes compile time: try "max-autotune"
  -> If max-autotune helps but CUDA graphs fail: try "max-autotune-no-cudagraphs"
  -> If debugging graph breaks: use default plus fullgraph=True as a diagnostic
  -> If isolating codegen bugs: use backend="aot_eager"
```

| Scenario | Mode |
|---|---|
| Development or first attempt | `default` |
| Variable sequence lengths | `default` + bucketing |
| Fixed-shape, overhead-sensitive inference | `reduce-overhead` |
| Long fixed-shape production training | `max-autotune` |
| Long variable-shape production training | `max-autotune-no-cudagraphs` |
| Graph break debugging | `default` + `fullgraph=True` |
| Codegen isolation | `backend="aot_eager"` |

## Mode Notes

- `default`: fastest compile time, good first baseline, tolerant of graph breaks.
- `reduce-overhead`: uses CUDA graph capture to reduce Python/dispatcher overhead; requires stable shapes and memory addresses; may increase memory.
- `max-autotune`: autotunes generated kernels for steady-state speed; compile can take minutes but may amortize over long runs.
- `max-autotune-no-cudagraphs`: keeps autotuning while avoiding CUDA graph constraints.
- `aot_eager`: debugging backend for Dynamo/AOTAutograd capture without Inductor codegen; not a speedup path.

## CUDA Graph Constraints

CUDA graph capture needs:

- static shapes inside the captured region,
- no `.item()`, `.numpy()`, or tensor-to-Python coercions inside the region,
- no in-place mutation of graph inputs that corrupts replay,
- no tensor-value-dependent Python control flow,
- no CPU-side operations interleaved with captured GPU work.

Diagnose applicability:

```bash
TORCH_LOGS=perf_hints python <entrypoint>
```

## Dynamic Shape Modes

- `dynamic=None` default: starts static, then may generalize dimensions after guard failures. Usually the best default.
- `dynamic=False`: fully static; best when shapes are truly fixed. Shape variation causes recompilation.
- `dynamic=True`: all dimensions symbolic; useful for debugging or highly irregular shapes, but often slower. Avoid as a production default.

## Shape Stabilization Strategies

Prefer this order:

1. Fixed shapes or `drop_last=True` when the workload naturally allows it.
2. Bucketing/padding to a small number of representative shapes.
3. Precise `torch._dynamo.mark_dynamic` for dimensions that truly vary.
4. `dynamic=True` only when targeted strategies fail.

For LLM-style variable sequence lengths, bucketing is usually the best performance/overhead tradeoff:

```text
length <= 256  -> pad to 256
length <= 512  -> pad to 512
length <= 1024 -> pad to 1024
length <= 2048 -> pad to 2048
```

Use `mark_dynamic` before invoking compiled code, not inside `forward`:

```python
torch._dynamo.mark_dynamic(input_ids, dim=1, min=1, max=max_seq_len)
torch._dynamo.mark_dynamic(attention_mask, dim=1, min=1, max=max_seq_len)
out = compiled_model(input_ids=input_ids, attention_mask=attention_mask)
```

Only mark dimensions that actually vary. Batch size often should stay fixed.

## Monitoring Recompilation

```bash
TORCH_LOGS=recompiles,dynamic python <entrypoint>
```

Track:

- number of unique graphs,
- guard failure reasons,
- shape/dtype/device/stride changes,
- whether Dynamo gives up after the cache size limit.

Useful test expectations:

- bucketed shapes should compile once per bucket, not once per batch,
- `mark_dynamic` on the intended dimension should reduce repeated recompiles,
- all chosen modes must produce outputs close to eager.
