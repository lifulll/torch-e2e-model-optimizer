# Optimization Ladder

Use this reference for the detailed step order. Do not enter the next step until the current step has no remaining plausible optimization point, unless profiler evidence directly identifies a deeper compiler or kernel bottleneck.

## Step 0: Baseline and Environment

Write `env.md` and `baseline.md`.

Capture:

- full launch command and git diff status,
- Python, PyTorch, Triton, CUDA/ROCm/DTK versions,
- hardware and visible devices,
- `torch.compile` settings,
- input shapes and data source,
- eager versus compiled comparison when feasible,
- compile time, steady-state time, throughput, latency, memory, and correctness.

Do not compare first-iteration compiled latency with eager steady-state latency.

## Step 1: Model Code

Write `step01_model_code.md`.

Optimize model-level tensor code:

- remove Python-side work from hot tensor paths,
- replace small op chains with equivalent PyTorch fused/library ops,
- use `torch.nn.functional.scaled_dot_product_attention` or mature attention/norm/loss implementations when applicable,
- reduce unnecessary `clone`, `contiguous`, `permute`, `transpose`, dtype casts, device copies, and host synchronization,
- simplify data-dependent Python control flow into tensor-friendly operations,
- move logging, metrics formatting, timers, and debug prints outside compiled regions,
- isolate unsupported third-party calls outside compiled functions.

Retain a model-code change only if E2E performance improves and correctness remains aligned.

## Step 2: PyTorch Runtime and Data Path

Write `step02_pytorch_runtime.md`.

Optimize framework-level execution:

- use `torch.inference_mode()` or `torch.no_grad()` for inference,
- use AMP/bf16/fp16 where numerically valid,
- set `torch.set_float32_matmul_precision("high")` for matmul-heavy workloads when appropriate,
- try channels-last for convolutional vision models,
- use `optimizer.zero_grad(set_to_none=True)`,
- try fused or foreach optimizers when supported,
- tune `DataLoader` workers, `pin_memory`, `persistent_workers`, `prefetch_factor`, CPU transforms, caching, and non-blocking copies,
- remove accidental synchronization from `.item()`, `.cpu()`, printing tensors, or per-step metrics.

Training benchmarks must cover forward, loss, backward, optimizer step, zero grad, data fetch, and synchronization. Inference benchmarks must separate cold compile, warm path, and serving batch behavior.

## Step 3: torch.compile, Dynamo, and Graph Breaks

Write `step03_compile_dynamo.md`.

Treat `torch.compile` as the default path:

- verify the correct module or function is compiled,
- compare default mode, `reduce-overhead`, and `max-autotune` one at a time,
- use regional compilation when full-model compilation causes unacceptable compile time or graph instability,
- inspect graph breaks and move side effects or unsupported code out of compiled regions,
- reduce recompilation from dynamic shapes, changing Python constants, guards, container structure, or mutable module state,
- prefer padding/bucketing or precise `torch._dynamo.mark_dynamic` before broad `dynamic=True`,
- use `torch.compiler.disable` only to isolate code that clearly harms compilation.

Record graph break count, recompilation count, compile latency, cache behavior, and compiled-region coverage.

## Step 4: Backward, Optimizer, and Distributed Path

Write `step04_aot_backward.md`.

Use this step for training or fine-tuning:

- profile backward separately from forward,
- identify AOTAutograd-generated graph behavior,
- check optimizer step and gradient synchronization cost,
- evaluate fused optimizers, foreach variants, compiled optimizer step, activation checkpointing, and gradient accumulation,
- for DDP/FSDP/ZeRO, measure rank skew, communication overlap, bucket sizing, sharding effects, and scaling efficiency,
- distinguish memory-saving changes from speed changes.

Do not accept a change that improves one microphase but worsens total step time.

## Step 5: TorchInductor Scheduling and Generated Code

Write `step05_inductor.md`.

Enter this layer only when evidence points to generated code, fusion, scheduling, layout, or launch overhead.

Collect:

```bash
TORCH_COMPILE_DEBUG=1 python <entrypoint>
TORCH_LOGS="output_code,kernel_code,schedule,perf_hints,fusion" python <entrypoint>
```

Inspect:

- FX graph and compiled graph boundaries,
- Inductor fusion decisions,
- layout and stride choices,
- generated code for hotspot kernels,
- autotune decisions and selected configs,
- intermediate materialization,
- small kernels or over-fused kernels,
- dynamic shape specialization.

Actions:

- adjust model expression boundaries to encourage better fusion,
- split over-fused regions only when profiler supports it,
- combine small pointwise patterns when launch overhead dominates,
- choose library ops that Inductor lowers well,
- apply targeted `.contiguous()` only when copy cost is justified,
- tune supported Inductor config only after verifying the current PyTorch version.

## Step 6: Generated Triton Kernel Analysis

Write `step06_triton_kernel.md`.

Analyze generated Triton only after end-to-end and Inductor evidence identifies stable hotspot kernels.

Classify each hotspot as pointwise, reduction, matmul-like, attention-like, normalization, scatter/gather, memory-bound, launch-bound, compute-bound, or occupancy/register-pressure limited.

Inspect grid, block size, num warps, num stages, memory coalescing, stride/index math, masks, vectorization, reduction dimensions, shared memory/LDS usage, register pressure, spills, occupancy, and achieved bandwidth or FLOPs.

Prefer influencing Triton generation through model or Inductor-level changes.

## Step 7: Custom Triton/CUDA/HIP Kernel Escape Hatch

Write `step07_custom_kernel.md` only if this layer is reached.

Use a custom kernel only when all are true:

- profiler shows a stable hotspot,
- correctness contract is clear,
- PyTorch library ops and model rewrites are insufficient,
- Inductor scheduling/config options are insufficient,
- expected E2E gain justifies maintenance cost,
- fallback implementation remains available.

For every custom kernel, add correctness tests, microbenchmarks, E2E benchmarks, supported-shape documentation, and fallback behavior.
