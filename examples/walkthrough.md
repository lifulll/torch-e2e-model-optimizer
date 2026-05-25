# Walkthrough

This example shows the expected workflow for a repository with `train.py` or `infer.py` already using `torch.compile`.

## 0. Confirm Required Inputs

Have these before starting:

- model execution command,
- model-generated profile output or trace directory,
- kill command for stale model processes,
- objective and correctness constraint.

Example:

```bash
ENTRY_CMD="python train.py --config configs/model.yaml"
PROFILE_DIR="./profiles/latest"
CLEANUP_CMD="pkill -f 'train.py --config configs/model.yaml' || true"
OBJECTIVE="maximize steady-state samples/sec with unchanged validation loss"
```

Keep precision and batch size fixed for attribution. Do not use AMP, mixed precision changes, batch-size sweeps, or microbatch sweeps unless the user explicitly asks.

## 1. Create Run Directory

```bash
python /home/torch-e2e-model-optimizer/scripts/collect_env.py \
  --out /tmp/torch_env.json \
  --markdown /tmp/torch_env.md

RUN_DIR=$(python /home/torch-e2e-model-optimizer/scripts/init_run.py \
  --root . \
  --entry-cmd "$ENTRY_CMD" \
  --cleanup-cmd "$CLEANUP_CMD" \
  --objective "$OBJECTIVE" \
  --env-json /tmp/torch_env.json \
  --env-md /tmp/torch_env.md)
```

Read `references/workflow_contract.md` before auditing or editing run outputs.

## 2. Establish Baseline

Run the existing benchmark with warmup. Record compile time separately from steady-state.

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" \
python /home/torch-e2e-model-optimizer/scripts/run_with_cleanup.py \
  --cleanup-cmd "$CLEANUP_CMD" \
  --entry-cmd "$ENTRY_CMD" \
  2>&1 | tee "$RUN_DIR/artifacts/torch_logs/baseline_compile.log"

python /home/torch-e2e-model-optimizer/scripts/analyze_torch_logs.py \
  "$RUN_DIR/artifacts/torch_logs/baseline_compile.log" \
  --markdown "$RUN_DIR/artifacts/torch_logs/baseline_compile_summary.md"
```

Fill `baseline.md` with:

- command and profile path,
- shapes, precision, and fixed batch size,
- `torch.compile` settings,
- p50/p90 step time,
- samples/sec or tokens/sec,
- compile time,
- peak memory,
- correctness result.

Seed `iteration_table.md` using `templates/iteration_row.md` if creating it by hand.

## 3. Iterate Step By Step

Load `references/optimization_ladder.md`, then work through steps 1-7. Do not enter the next step until the current step has no remaining material optimization point, unless profiler evidence directly identifies a deeper compiler or kernel bottleneck.

Use 2% E2E improvement as the default materiality threshold. If benchmark noise is larger than 2%, use the measured noise as the threshold. Skip smaller cleanups unless they unblock a larger optimization, and note them briefly in the step summary.

For each step:

```text
profile current best
while this step still has material optimization points:
  pick one method or one controlled branch family
  change code/config
  run cleanup command, then unit/equivalence tests and benchmark
  update stepXX_*.md
  append iteration_table.md
stop this step when remaining ideas cannot plausibly beat the materiality threshold
```

Example Step 1 model-code iteration:

```bash
python /home/torch-e2e-model-optimizer/scripts/run_with_cleanup.py \
  --cleanup-cmd "$CLEANUP_CMD" \
  --entry-cmd "$ENTRY_CMD"

python /home/torch-e2e-model-optimizer/scripts/record_iteration.py \
  --run-dir "$RUN_DIR" \
  --iter 1 \
  --layer model_code \
  --hypothesis "removing scalar sync from compiled path reduces graph breaks" \
  --changed "moved metric .item() outside compiled forward" \
  --files "model.py" \
  --correctness "pass: loss diff < 1e-5" \
  --before 120.0 \
  --after 132.0 \
  --unit "samples/sec" \
  --direction throughput \
  --materiality-threshold 2.0 \
  --noise-percent 0.8 \
  --retained yes \
  --evidence "step01_model_code.md; artifacts/torch_logs/iter1_compile_summary.md"
```

Use `templates/step_summary.md` for each `stepXX_*.md`.

For local model-code rewrites such as `a.add(b)` to `a + b`, copy `templates/model_code_change_compile_test_template.py` and compare accuracy plus benchmark timing under default `torch.compile`.

For Inductor pass work, copy `templates/inductor_pass_compile_test_template.py` and monkeypatch the pass disabled/enabled paths to compare accuracy and performance.

## 4. Reference Selection

Use references only when relevant:

- `references/optimization_catalog.md`: choose methods from evidence.
- `references/profiler_playbook.md`: timing alone cannot explain the bottleneck.
- `references/torch_compile_debugging.md`: graph breaks, recompilation, or dynamic shapes.
- `references/compile_modes_shape_strategy.md`: compile modes, CUDA graphs, bucketing, or `mark_dynamic`.
- `references/distributed_compile_guidance.md`: DDP/FSDP, checkpoint boundaries, or rank-level validation.
- `references/inductor_triton_playbook.md`: generated code, fusion, layout, or Triton hotspots.
- `references/inductor_triton_hop_development_flow.md`: Step 7 custom kernel through an Inductor pass using `triton_kernel_wrapper_functional`.

Do not move to Step 7 until end-to-end profiler evidence shows a stable generated-kernel bottleneck and easier layers are exhausted. Step 7 should use an Inductor FX pass to insert hand-written Triton kernels through HOP when possible.

## 5. Finalize

When two consecutive iterations show no material improvement, only sub-threshold ideas remain, or a real boundary is reached:

```bash
python /home/torch-e2e-model-optimizer/scripts/summarize.py \
  --run-dir "$RUN_DIR" \
  --stop-reason "two no-gain iterations; remaining bottleneck is outside current scope"
```

Use `templates/final_summary.md` for the final report shape. The final user-facing answer should report best retained command/config, total E2E speedup, correctness, files changed, remaining bottleneck, and the run directory.
