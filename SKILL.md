---
name: torch-e2e-model-optimizer
description: Optimize PyTorch AI model training or inference end to end when the model already uses torch.compile or should be treated as torch.compile-first. Use for performance work that must progress from model code to PyTorch runtime, TorchDynamo/AOTAutograd, TorchInductor scheduling/codegen, and generated Triton kernels, with iterative benchmarking, correctness checks, profiler evidence, per-step summary files, code-change attribution, and stop-only-when-no-more-performance-gain behavior.
---

# Torch E2E Model Optimizer

Use this skill to optimize PyTorch model training and inference end to end. Assume the target workload already uses `torch.compile` unless evidence proves otherwise. Work from easy, low-risk changes toward deeper compiler and kernel changes:

`model code -> PyTorch runtime -> torch.compile/TorchDynamo -> AOTAutograd -> TorchInductor -> generated Triton kernel -> custom kernel only if justified`

The job is not complete after one suggestion or one profile. Iterate until benchmark evidence shows no material performance gain remains, a correctness or engineering boundary is reached, or the user explicitly stops the run.

## Basic Principles

This skill inherits the core principles of `torch-compile-model-optimization` and makes them mandatory for end-to-end work:

- Treat "optimization" as improving accelerator-side kernel efficiency and effective device throughput, not generic CPU cleanup. Valid optimizations should improve device kernel execution, reduce device idle time, improve fusion/layout/memory access/occupancy, raise device utilization, or increase end-to-end device throughput.
- Count CPU-side work as an optimization only when it directly improves device-side performance or end-to-end accelerator throughput, such as fixing dataloader stalls, host-device synchronization, kernel launch gaps, recompilation from unstable shapes, or Python logic that blocks compiled graphs.
- Start from a reproducible baseline. Do not optimize without a runnable command, representative input shape or shape distribution, and a correctness signal.
- If the launch command, model code, or environment facts are missing, infer them from the repository first; ask the user only when they cannot be recovered safely.
- Separate compile latency, warmup behavior, and steady-state performance. Never judge `torch.compile` from the first compiled iteration.
- Treat end-to-end throughput or latency as the primary score. Kernel microbenchmarks are supporting evidence, not the final result.
- Change one primary variable per iteration unless the codebase already couples the changes.
- Keep every retained optimization backed by before/after measurements and correctness checks.
- Revert or quarantine changes that do not improve end-to-end performance beyond noise.
- When changing tensor math semantics, layout, dtype, shape handling, or operator ordering, run an equivalence test against baseline outputs, loss, metric, or accepted tolerance.
- Prefer low-risk framework/configuration changes before model rewrites, and prefer model/PyTorch rewrites before compiler internals or custom Triton/CUDA/HIP kernels.
- Record performance and correctness together. Faster but wrong is not an optimization.
- Actively look for hidden bottlenecks: data loading, CPU preprocessing, logging, synchronization, optimizer step, distributed communication, graph breaks, and recompilation.
- If running on Hygon DCU, ROCm, HIP, or DTK, also use the relevant Hygon/DCU profiling and kernel optimization skills.

## Target State

Define the target state before selecting changes. If the user does not provide one, use this default:

- Performance: maximize stable steady-state throughput or minimize stable latency for the current hardware, current model, current precision, and representative workload. Pick the metric that matches the workload: samples/sec, tokens/sec, step time, p50/p90 latency, memory, compile time, serving p99, or cost.
- Correctness: preserve baseline behavior. Loss, logits/output error, validation metrics, generated text quality, or task-specific acceptance checks must not regress beyond the agreed tolerance.
- Maintainability: prefer reproducible, bounded, reversible changes that are robust to future shape variation and project configuration changes.
- Deployment fitness: report compile-time cost, memory impact, cold-start behavior, and any version/device constraints so the fastest path is not accidentally unusable.

If measurements show the workload is close to a practical ceiling, explain what limits further optimization: compute, memory bandwidth, memory capacity, communication, input pipeline, compiler capability, launch overhead, or the workload algorithm itself.

## Iteration Protocol

When the user asks to optimize model performance, execute an iterative loop rather than a one-shot recommendation:

1. Define the target state: primary metric, secondary constraints, correctness tolerance, and acceptable compile-time/memory tradeoff.
2. Establish baseline: record eager versus `torch.compile` when feasible, and separate compile time from steady-state performance.
3. Identify the current dominant bottleneck layer: data path, graph break, recompilation, optimizer/backward, model structure, distributed communication, Inductor scheduling, or generated Triton kernel.
4. Choose the shallowest layer that can plausibly remove the bottleneck.
5. Change one most-promising variable or create a controlled branch set for that one variable family.
6. Re-run correctness and benchmark with the same measurement protocol.
7. Record performance, correctness, files changed, evidence, side effects, and retain/revert decision in the step summary and `iteration_table.md`.
8. If the target is not reached, use the new evidence to move to the next bottleneck or deeper layer. Do not fall back to generic advice.
9. Escalate to user decision only after two consecutive no-gain iterations, a clear numerical/correctness boundary, an engineering tradeoff, or an out-of-scope bottleneck.

Maintain the experiment sequence as:

`baseline -> change -> verify -> record -> next bottleneck`

Every iteration must state the current conclusion, the next hypothesis, and why failed attempts failed.

## Required Inputs

Collect or infer these before starting:

- Workload type: training, inference, fine-tuning, serving, prefill/decode, or mixed.
- Entry command: exact script, config, environment variables, and dataset or synthetic input mode.
- Model and shapes: batch size, sequence length, image size, vocab size, dynamic dimensions, and representative distributions.
- Precision and layout: fp32, tf32, bf16, fp16, quantization, channels-last, or custom layout.
- Hardware: GPU/DCU/CPU model, accelerator count, interconnect, driver/runtime, PyTorch, Triton, CUDA/ROCm/DTK versions.
- Current `torch.compile` usage: target module/function, mode, dynamic setting, backend, fullgraph, compile cache, and disabled regions.
- Optimization objective: throughput, latency, step time, tokens/sec, samples/sec, memory, compile time, serving p99, cost, or stability.

If required inputs cannot be inferred from the repository or launch scripts, ask once briefly. Otherwise proceed.

## Run Directory

Create one run directory near the benchmark or under the repository root:

```text
torch_e2e_opt_YYYYMMDD_HHMMSS/
|-- env.md
|-- baseline.md
|-- iteration_table.md
|-- final_summary.md
|-- step01_model_code.md
|-- step02_pytorch_runtime.md
|-- step03_compile_dynamo.md
|-- step04_aot_backward.md
|-- step05_inductor.md
|-- step06_triton_kernel.md
|-- step07_custom_kernel.md
|-- artifacts/
|   |-- profiler/
|   |-- torch_logs/
|   |-- traces/
|   `-- generated_code/
`-- patches/
```

Every optimization step must write or update its step summary file, even when the conclusion is "no change retained".

Fast path:

```bash
python <skill>/scripts/collect_env.py --out /tmp/torch_env.json --markdown /tmp/torch_env.md
python <skill>/scripts/init_run.py --root . --entry-cmd "<benchmark command>" --objective "<primary metric>" --env-json /tmp/torch_env.json
```

## Report Table Contract

Maintain `iteration_table.md` as the authoritative ledger. Use this table shape:

```markdown
| Iter | Layer | Hypothesis | Code/config changed | Files changed | Correctness | E2E metric before | E2E metric after | Delta | Retained | Evidence |
|---:|---|---|---|---|---|---:|---:|---:|---|---|
| 0 | baseline | establish reference | none | none | pass | 123 tok/s | 123 tok/s | 0% | yes | baseline.md |
```

Each step summary file must include:

- What was optimized.
- Why this layer was selected.
- Exact commands used.
- Files and code regions changed.
- Correctness result and tolerance.
- End-to-end performance before and after.
- Profiler or compiler evidence.
- Whether the change was retained, reverted, or left as an optional branch.
- Next bottleneck and next hypothesis.

Use percent improvement consistently:

```text
throughput_gain = (after / before - 1) * 100
latency_gain = (before / after - 1) * 100
```

## Measurement Protocol

Before every comparison:

1. Pin the same model revision, config, seed, input shape distribution, precision, batch size, and device placement.
2. Warm up enough iterations to exclude compile and cache effects.
3. Report compile time separately from steady-state.
4. Use at least median and p90, or enough repeated runs to expose noise.
5. Synchronize accelerator timing only at measurement boundaries.
6. Record peak memory and any OOM, recompilation, graph break, or fallback.
7. Run correctness checks before accepting faster results.

Useful starting commands:

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" python <entrypoint>
TORCH_TRACE=./torch_e2e_trace python <entrypoint>
TORCH_COMPILE_DEBUG=1 python <entrypoint>
```

Use PyTorch Profiler or the platform profiler when simple timing cannot explain the bottleneck. Inspect compiled regions, kernel gaps, CPU launch overhead, memory allocation, communication, optimizer time, dataloader stalls, and generated kernels.

## Execution Loop Details

Use these execution details when applying the iteration protocol:

1. Measure current best end-to-end performance.
2. Identify the highest-impact bottleneck layer.
3. Choose the earliest layer in the optimization chain that can plausibly fix it.
4. Make a bounded code or configuration change.
5. Validate correctness.
6. Re-benchmark with the same protocol.
7. Update the step summary and `iteration_table.md`.
8. Keep the change only if it improves the primary E2E metric beyond noise without violating correctness or maintainability.
9. Move deeper only when shallower layers are exhausted or evidence points directly to compiler/kernel behavior.

Default no-gain stopping rule:

- Stop after two consecutive retained-or-tested iterations fail to improve the primary E2E metric by at least 1-2% beyond benchmark noise, or after the profiler shows the remaining bottleneck is outside the requested optimization scope.
- If performance is near a clear hardware, memory bandwidth, communication, or algorithmic ceiling, stop and explain the ceiling with evidence.
- Do not stop merely because one optimization failed. Convert failed attempts into the next bottleneck hypothesis.

## Optimization Ladder

### Step 0: Baseline and Environment

Write `env.md` and `baseline.md`.

Capture:

- full launch command and git diff status,
- Python, PyTorch, Triton, CUDA/ROCm/DTK versions,
- hardware and visible devices,
- `torch.compile` settings,
- input shapes and data source,
- eager versus compiled comparison when feasible,
- compile time, steady-state time, throughput, latency, memory, and correctness.

Do not compare first-iteration compiled latency against eager steady-state latency. That confuses compilation cost with runtime performance.

### Step 1: Model Code

Write `step01_model_code.md`.

Optimize model-level tensor code before reaching into compiler internals:

- remove Python-side work from hot tensor paths,
- replace small op chains with equivalent PyTorch fused/library ops,
- use `torch.nn.functional.scaled_dot_product_attention` or mature attention/norm/loss implementations when applicable,
- reduce unnecessary `clone`, `contiguous`, `permute`, `transpose`, dtype casts, device copies, and host synchronization,
- simplify data-dependent Python control flow into tensor-friendly operations,
- stabilize shapes through padding or bucketing,
- move logging, metrics formatting, timers, and debug prints outside compiled regions,
- isolate unsupported third-party calls outside the compiled function.

Retain a model-code change only if E2E performance improves and correctness remains aligned.

### Step 2: PyTorch Runtime and Data Path

Write `step02_pytorch_runtime.md`.

Optimize framework-level execution:

- use `torch.inference_mode()` or `torch.no_grad()` for inference,
- use AMP/bf16/fp16 where numerically valid,
- set `torch.set_float32_matmul_precision("high")` for matmul-heavy workloads when appropriate,
- try channels-last for convolutional vision models,
- tune batch size, gradient accumulation, and microbatching,
- use `optimizer.zero_grad(set_to_none=True)`,
- try fused or foreach optimizers when supported,
- tune `DataLoader` workers, `pin_memory`, `persistent_workers`, `prefetch_factor`, CPU transforms, caching, and non-blocking copies,
- remove accidental synchronization from `.item()`, `.cpu()`, printing tensors, or per-step metrics.

Training benchmarks must include forward, loss, backward, optimizer step, zero grad, data fetch, and any distributed synchronization. Inference benchmarks must separate cold compile, warm path, and serving batch behavior.

### Step 3: torch.compile, Dynamo, and Graph Breaks

Write `step03_compile_dynamo.md`.

Treat `torch.compile` as the default path:

- verify the correct module or function is compiled,
- compare default mode, `reduce-overhead`, and `max-autotune` one at a time,
- use regional compilation when full-model compilation causes unacceptable compile time or graph instability,
- inspect graph breaks and move side effects or unsupported code out of compiled regions,
- reduce recompilation from dynamic shapes, changing Python constants, guards, container structure, or mutable module state,
- prefer padding/bucketing or precise `torch._dynamo.mark_dynamic` before broad `dynamic=True`,
- use `torch.compiler.disable` only to isolate code that clearly harms compilation.

Record graph break count, recompilation count, compile latency, cache behavior, and the compiled-region coverage before and after.

### Step 4: Backward, Optimizer, and Distributed Path

Write `step04_aot_backward.md`.

Use this step for training or fine-tuning:

- profile backward separately from forward,
- identify AOTAutograd-generated graph behavior,
- check optimizer step and gradient synchronization cost,
- evaluate fused optimizers, foreach variants, compiled optimizer step, activation checkpointing, and gradient accumulation,
- for DDP/FSDP/ZeRO, measure rank skew, communication overlap, bucket sizing, sharding effects, and scaling efficiency,
- distinguish memory-saving changes from speed changes.

Do not accept a training optimization that improves one microphase but worsens total step time.

### Step 5: TorchInductor Scheduling and Generated Code

Write `step05_inductor.md`.

Enter this layer only when evidence points to compiler-generated code, fusion, scheduling, layout, or launch overhead.

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
- excessive intermediate materialization,
- too many small kernels or over-fused kernels with high register pressure,
- dynamic shape specialization that blocks good schedules.

Optimization actions:

- adjust model expression boundaries to encourage better fusion,
- split over-fused regions with explicit boundaries only when profiler supports it,
- combine small pointwise patterns when launch overhead dominates,
- choose library ops that Inductor lowers well,
- apply targeted `.contiguous()` only when copy cost is lower than kernel loss,
- tune compile modes and supported Inductor config only after verifying the option exists in the current PyTorch version.

Record generated file paths or trace artifact paths in `step05_inductor.md`.

### Step 6: Generated Triton Kernel Analysis

Write `step06_triton_kernel.md`.

Analyze generated Triton only after end-to-end and Inductor evidence identifies one or a few hotspot kernels.

Classify each hotspot:

- pointwise,
- reduction,
- matmul-like,
- attention-like,
- normalization,
- scatter/gather,
- memory-bound fused kernel,
- launch-bound small kernel,
- compute-bound kernel,
- occupancy/register-pressure limited kernel.

Inspect:

- grid, block size, num warps, num stages,
- memory coalescing and stride/index math,
- mask density and boundary checks,
- vectorization,
- reduction dimensions,
- shared memory/LDS usage when applicable,
- register pressure, spills, occupancy, and achieved bandwidth or FLOPs.

Prefer influencing Triton generation through model or Inductor-level changes. Only edit or replace kernels when the generated kernel is proven to be the bottleneck and shallower fixes are exhausted.

### Step 7: Custom Triton/CUDA/HIP Kernel Escape Hatch

Write `step07_custom_kernel.md` only if this layer is reached.

Use a custom kernel only when all are true:

- profiler shows a stable hotspot,
- correctness contract is clear,
- PyTorch library ops and model rewrites are insufficient,
- Inductor scheduling/config options are insufficient,
- expected E2E gain justifies maintenance cost,
- fallback implementation remains available.

For every custom kernel:

- write a correctness test against the PyTorch reference,
- benchmark micro and end-to-end performance,
- document supported shapes, dtypes, layouts, devices, and fallback behavior,
- record why the generated Inductor/Triton kernel was not enough.

## Bottleneck Decision Guide

- Accelerator idle with high CPU time: inspect data loading, graph breaks, launch overhead, synchronization, and Python side effects.
- Frequent recompilation: stabilize shapes, constants, guards, and compiled call signatures.
- Many small kernels: reduce graph breaks, compile a larger stable region, or refactor small op chains.
- One large slow kernel: inspect operator choice, layout, fusion shape, generated code, and Triton kernel resource use.
- Memory pressure: inspect activation size, batch size, optimizer state, checkpointing, precision, sharding, and temporary materialization.
- Poor distributed scaling: inspect communication, overlap, bucket sizes, rank skew, and data imbalance.
- Compile time too high but warm path good: consider regional compilation, caching, static input contracts, and reduced compile scope.
- Kernel speed improves but E2E does not: look for a new bottleneck, CPU gaps, communication, data path, or measurement error.

## Final Summary

Write `final_summary.md` and include:

- best retained command/config,
- total E2E improvement from baseline,
- final latency/throughput/memory,
- compile-time impact,
- correctness status,
- table of all iterations,
- retained changes and files modified,
- reverted or rejected changes with reasons,
- remaining bottleneck and why optimization stopped,
- risks, version constraints, and follow-up options.

Final user-facing answer should summarize the same facts briefly and point to the run directory.

## Bundled Resources

Use bundled resources instead of rewriting the same scaffolding each time.

### scripts/

- `collect_env.py`: collect Python, PyTorch, Triton, CUDA/HIP, visible devices, and profiler-tool availability into JSON/Markdown.
- `init_run.py`: create the `torch_e2e_opt_YYYYMMDD_HHMMSS/` run directory, summary files, artifact folders, and `state.json`.
- `analyze_torch_logs.py`: count common `torch.compile` log signals such as graph breaks, recompilation, dynamic-shape mentions, Inductor, and Triton.
- `record_iteration.py`: append a measured iteration to `iteration_table.md` and update `state.json` with computed percent improvement.
- `summarize.py`: generate `final_summary.md` from `state.json` and `iteration_table.md`.

### references/

- `optimization_catalog.md`: method registry by layer, trigger, action, skip rule, and evidence.
- `profiler_playbook.md`: profiler setup and symptom-to-next-step mapping.
- `torch_compile_debugging.md`: graph break, recompilation, dynamic-shape, and compile-mode triage.
- `inductor_triton_playbook.md`: generated-code, fusion, layout, and Triton hotspot analysis.
- `branch_attribution_protocol.md`: branch/select/ablation protocol adapted from kernel optimization for torch E2E variants.
- `reporting_contract.md`: exact report contents expected from each step and final summary.
- `method_registry.json`: structured method IDs for automated or semi-automated selection.
- `inductor_gated_cat_sum_copy_scheduler.py`, `inductor_test_gated_cat_sum_copy_real_structure.py`, `inductor_mixer_sparse_posfb.py`: migrated Inductor references from the local `torch-compile-model-optimization` skill. Load only for Inductor pass or scheduler-level work.

### templates/ and examples/

- `templates/step_summary.md`: copy this shape for every `stepXX_*.md` file.
- `templates/final_summary.md`: final report skeleton.
- `templates/iteration_row.md`: canonical iteration table header and first row.
- `examples/walkthrough.md`: end-to-end example showing environment collection, baseline, iteration recording, and finalization.

## Avoid

- Treating first-run compile latency as steady-state runtime.
- Reporting a kernel microbenchmark as success when E2E performance did not improve.
- Skipping correctness because the change "only affects performance".
- Changing multiple major variables and then claiming attribution.
- Jumping directly to custom Triton, CUDA, or HIP kernels before model, PyTorch, compile, and Inductor evidence supports it.
- Keeping an optimization whose gain is within noise and has no clear profiler support.
- Ending with only recommendations when the repository and environment allow another optimization iteration.
