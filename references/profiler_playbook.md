# Profiler Playbook

Use this when step time, throughput, or latency alone cannot explain the bottleneck.

## Minimal PyTorch Profiler Shape

Use warmup and active windows. Export a trace and record key averages:

```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    schedule=schedule(wait=2, warmup=3, active=5, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    on_trace_ready=tensorboard_trace_handler("./torch_e2e_opt_trace"),
) as prof:
    for step, batch in enumerate(loader_or_inputs):
        run_one_step(batch)
        prof.step()
        if step >= 12:
            break
```

## What to Inspect

| Symptom | Inspect | Likely Next Step |
|---|---|---|
| Accelerator idle gaps before kernels | CPU dataloader, preprocessing, H2D copy, graph breaks | runtime data path or compile graph break |
| Many short kernels | compiled region coverage, small op chains, graph breaks | model rewrite or compile region |
| One large generated kernel dominates | generated Triton, layout, fusion group | Inductor/Triton step |
| Backward dominates | AOTAutograd graph, activation memory, optimizer | backward/optimizer step |
| Optimizer dominates | foreach/fused support, parameter count, dtype | optimizer method |
| Memory spikes | temporary materialization, clone/contiguous, activation storage | layout/model rewrite |
| Distributed scaling poor | rank skew, allreduce, overlap, buckets | distributed path |
| Compile time dominates but warm path good | trace cache, regional compile, compile scope | compile scope/cache |

## Required Numbers

Record these in the step summary:

- p50/p90 step time or latency.
- samples/sec or tokens/sec.
- peak memory.
- compile time and first-token/first-step time when relevant.
- graph break count and recompilation count when using `torch.compile`.
- top CPU operators and top accelerator kernels by self time.
- any profiler caveats: missing counters, synthetic data, short active window, remote tool failure.
