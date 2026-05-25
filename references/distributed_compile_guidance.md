# Distributed torch.compile Guidance

Use this reference when optimizing training with DDP, FSDP, gradient checkpointing, or distributed communication.

## DDP

For DDP, compile the inner model before wrapping with `DistributedDataParallel`:

```python
model = torch.compile(model, mode="default")
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
)
```

Avoid compiling the DDP wrapper first; that captures wrapper boilerplate instead of the model compute graph.

DDP notes:

- Prefer `find_unused_parameters=False` unless unused parameters are real.
- Test on actual multi-GPU runs; single-GPU tests do not cover allreduce or DDP graph breaks.
- `reduce-overhead` can conflict with allreduce/CUDA graph capture; use `TORCH_LOGS=perf_hints` and fall back to `default` if needed.
- `static_graph=True` can help when parameter usage/order is truly static.
- Gradient clipping normally happens outside the compiled graph and does not need special handling.

## FSDP

FSDP support depends on API/version.

- FSDP2/composable `fully_shard` is generally the preferred path for compile compatibility.
- FSDP1 can be more fragile; selective block compilation is often safer.

Selective block compilation pattern:

```python
for i, block in enumerate(model.transformer.blocks):
    model.transformer.blocks[i] = torch.compile(block, mode="default")

model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
)
```

Tradeoffs:

- more reliable with wrappers left eager,
- still captures attention/MLP-heavy compute,
- leaves Python overhead at block boundaries,
- loses cross-block fusion opportunities.

## Gradient Checkpointing

When using `torch.utils.checkpoint.checkpoint`, prefer:

```python
checkpoint(fn, *args, use_reentrant=False)
```

The reentrant checkpoint path uses Python-level control flow that is more likely to cause graph breaks.

## Checkpoint And State Dict Boundaries

Keep checkpoint save/load and state-dict operations outside compiled regions. Use `torch.compiler.disable` or an eager-only helper when needed:

```python
@torch.compiler.disable
def save_checkpoint(model, path):
    state = model.state_dict()
    torch.save(state, path)
```

## Distributed Validation Checklist

- Run `TORCH_LOGS=graph_breaks` on the actual distributed command.
- Run `TORCH_LOGS=recompiles` for enough steps to catch shape and wrapper instability.
- Compare eager versus compiled outputs or loss on the same seed.
- Compare gradient norms across eager and compiled runs.
- Measure peak memory on all ranks.
- Check rank skew and communication overlap in profiler traces.
- Verify checkpoint save/load outside compiled regions.
- For DDP, test with at least two ranks.
