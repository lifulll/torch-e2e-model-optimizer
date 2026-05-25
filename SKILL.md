---
name: torch-e2e-model-optimizer
description: Optimize PyTorch AI model training or inference end to end when the model already uses torch.compile or should be treated as torch.compile-first. Use for performance work that must progress from model code to PyTorch runtime, TorchDynamo/AOTAutograd, TorchInductor scheduling/codegen, and generated Triton kernels, with iterative benchmarking, correctness checks, profiler evidence, per-step summary files, code-change attribution, and stop-only-when-no-more-performance-gain behavior.
---

# Torch E2E Model Optimizer

Optimize PyTorch training or inference end to end. Assume the workload already uses `torch.compile` unless evidence proves otherwise.

Primary chain:

`model code -> PyTorch runtime -> torch.compile/TorchDynamo -> AOTAutograd -> TorchInductor -> generated Triton kernel -> Inductor pass-inserted custom kernel only if justified`

Do not finish after one suggestion or one profile. Iterate until no material gain remains, a correctness or engineering boundary is reached, or the user stops the run.

## Required Inputs

Before starting, confirm these two inputs are available:

1. Model execution command, for example:

```bash
cd /xxx/xxx/
sh torch_start_1card.sh run_with_numerous.py
```

2. Path to the model-generated profile output or trace directory.

Also collect or infer workload type, representative shapes, batch size, precision, hardware/runtime versions, current `torch.compile` settings, and the optimization objective.

If either required input is missing and cannot be inferred, ask once briefly, then proceed with what can be measured.

## Hard Rules

- Optimize accelerator-side kernel efficiency and effective device throughput, not generic CPU cleanup.
- Count CPU-side work only when it directly improves device throughput, such as fixing dataloader stalls, synchronization, launch gaps, recompilation, or Python logic that blocks compiled graphs.
- Establish a reproducible baseline before changing code.
- Separate compile latency, warmup behavior, and steady-state performance.
- Use end-to-end throughput or latency as the primary score. Kernel microbenchmarks are supporting evidence only.
- Change one primary variable per iteration, or one controlled branch family.
- Keep changes only when they improve the primary E2E metric beyond noise and pass correctness checks.
- Run equivalence tests when changing tensor math, layout, dtype, shape handling, or operator ordering.
- Prefer framework/configuration changes before model rewrites, and model/PyTorch rewrites before compiler internals or pass-inserted custom kernels.
- Do not use AMP/mixed-precision changes or batch-size/microbatch sweeps as optimization directions for this skill. Keep existing precision and batch size fixed for attribution unless the user explicitly requests otherwise.
- If running on Hygon DCU, ROCm, HIP, or DTK, also use the relevant Hygon/DCU profiling and kernel optimization skills.

## Target State

Default target when the user does not provide one:

- Performance: best stable steady-state throughput or latency for the current hardware, model, precision, and representative workload.
- Correctness: baseline behavior does not regress beyond accepted tolerance.
- Maintainability: changes are reproducible, bounded, reversible, and robust to expected shape variation.
- Deployment fitness: compile-time cost, memory impact, cold-start behavior, and version/device constraints are reported.

If close to a practical ceiling, name the limiter: compute, bandwidth, memory capacity, communication, input pipeline, compiler capability, launch overhead, or algorithm.

## Loop At A Glance

Follow this loop exactly:

```text
0. check_env          -> env.md
1. init run folder    -> run_YYYYMMDD_HHMMSS/
2. establish baseline -> baseline.md + iteration_table.md
3. for step in step01..step07:
     a. profile current best for this step
     b. while this step still has plausible optimization points:
          - if no method can beat benchmark noise by 1-2%, stop this step
          - Codex picks one method or one controlled branch family
          - Codex writes code/config
          - run unit/equivalence tests and benchmark
          - update stepXX_*.md and iteration_table.md
     c. enter the next step only after this step has no remaining optimization point
4. emit final_summary.md
```

Maintain `baseline -> change -> verify -> record -> next bottleneck`. Every iteration must state the conclusion, next hypothesis, and failed-attempt reason.

## Run Setup

Use the bundled scripts instead of hand-creating run files:

```bash
python <skill>/scripts/collect_env.py --out /tmp/torch_env.json --markdown /tmp/torch_env.md
python <skill>/scripts/init_run.py --root . --entry-cmd "<benchmark command>" --objective "<primary metric>" --env-json /tmp/torch_env.json
```

Read `references/workflow_contract.md` for the run directory layout, measurement protocol, `iteration_table.md`, per-step summaries, and `final_summary.md` contract.

## Optimization Ladder

Use `references/optimization_ladder.md` for step details. The required order is:

0. Baseline and environment
1. Model code
2. PyTorch runtime and data path
3. `torch.compile`, Dynamo, and graph breaks
4. Backward, optimizer, and distributed path
5. TorchInductor scheduling and generated code
6. Generated Triton kernel analysis
7. Custom kernel through an Inductor pass

Do not enter the next step until the current step has no remaining plausible optimization point, unless profiler evidence directly identifies a deeper compiler or kernel bottleneck.

## When To Load References

- `references/workflow_contract.md`: always load before creating or auditing run outputs.
- `references/optimization_ladder.md`: load before selecting or executing step-level optimizations.
- `references/optimization_catalog.md`: load when choosing methods from evidence.
- `references/branch_attribution_protocol.md`: load for compile-mode sweeps, batch/layout variants, Inductor boundary variants, or any noisy branch comparison.
- `references/profiler_playbook.md`: load when timing alone cannot explain the bottleneck.
- `references/torch_compile_debugging.md`: load for graph breaks, recompilation, dynamic shapes, or compile-mode triage.
- `references/compile_modes_shape_strategy.md`: load when selecting compile modes, CUDA graph strategy, bucketing, `mark_dynamic`, or dynamic-shape policy.
- `references/distributed_compile_guidance.md`: load for DDP/FSDP, distributed graph breaks, gradient checkpointing, checkpoint boundaries, or rank-level validation.
- `references/inductor_triton_playbook.md`: load only when generated code, fusion, layout, or Triton hotspots are implicated.
- `references/inductor_triton_hop_development_flow.md`: load when replacing an FX subgraph with hand-written Triton kernels through `triton_kernel_wrapper_functional` HOP in the Inductor post-grad pipeline.
- `references/reporting_contract.md`: load when finalizing or checking output completeness.

## Scripts

- `collect_env.py`: collect Python/PyTorch/Triton/CUDA/HIP/tool availability into JSON/Markdown.
- `init_run.py`: create `run_YYYYMMDD_HHMMSS/`, step files, artifact folders, and `state.json`.
- `analyze_torch_logs.py`: summarize `torch.compile` log signals.
- `record_iteration.py`: append one measured result to `iteration_table.md` and `state.json`.
- `summarize.py`: generate `final_summary.md`.

## Templates

- `templates/model_code_change_compile_test_template.py`: copy when validating a model-code rewrite such as `a.add(b)` to `a + b`; compares original vs rewritten implementations under default `torch.compile` for accuracy and benchmark timing while keeping precision and batch size fixed.

## Avoid

- Treating first-run compile latency as steady-state runtime.
- Reporting a kernel microbenchmark as success when E2E performance did not improve.
- Skipping correctness because a change "only affects performance".
- Changing multiple major variables and claiming attribution.
- Jumping to custom Triton/CUDA/HIP before model, PyTorch, compile, and Inductor evidence supports it.
- Keeping gains within noise without profiler support.
- Ending with recommendations when another measured iteration is possible.
