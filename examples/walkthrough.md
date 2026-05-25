# Walkthrough

This example shows the intended workflow for a repository with `train.py` or `infer.py` already using `torch.compile`.

## 1. Create a Run Directory

```bash
python /home/torch-e2e-model-optimizer/scripts/collect_env.py \
  --out /tmp/torch_env.json \
  --markdown /tmp/torch_env.md

python /home/torch-e2e-model-optimizer/scripts/init_run.py \
  --root . \
  --entry-cmd "python train.py --config configs/model.yaml" \
  --objective "maximize steady-state samples/sec with unchanged validation loss" \
  --env-json /tmp/torch_env.json
```

Copy or move `/tmp/torch_env.md` content into the run directory `env.md` if useful.

## 2. Establish Baseline

Run the existing benchmark with warmup. Record compile time separately from steady-state:

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" \
python train.py --config configs/model.yaml 2>&1 | tee run_*/artifacts/torch_logs/baseline_compile.log

python /home/torch-e2e-model-optimizer/scripts/analyze_torch_logs.py \
  run_*/artifacts/torch_logs/baseline_compile.log \
  --markdown run_*/artifacts/torch_logs/baseline_compile_summary.md
```

Fill `baseline.md` with:

- command,
- shapes,
- precision,
- `torch.compile` settings,
- p50/p90 step time,
- samples/sec or tokens/sec,
- compile time,
- peak memory,
- correctness result.

## 3. First Iteration: Shallowest Plausible Layer

If the compile log shows graph breaks from logging or `.item()` in `forward`, fix model code first. After the change:

```bash
python train.py --config configs/model.yaml

python /home/torch-e2e-model-optimizer/scripts/record_iteration.py \
  --run-dir run_YYYYMMDD_HHMMSS \
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
  --retained yes \
  --evidence "step01_model_code.md; artifacts/torch_logs/iter1_compile_summary.md"
```

Update `step01_model_code.md` using `templates/step_summary.md`.

## 4. Continue Down the Ladder

Use references in this order:

1. `references/optimization_catalog.md` to pick the next method.
2. `references/profiler_playbook.md` when timing alone is unclear.
3. `references/torch_compile_debugging.md` for graph breaks and recompilation.
4. `references/inductor_triton_playbook.md` only when generated code is implicated.

Do not move to custom Triton/CUDA/HIP until end-to-end profiler evidence shows a stable generated-kernel bottleneck and easier layers are exhausted.

## 5. Finalize

When two consecutive iterations show no material improvement, or a real boundary is reached:

```bash
python /home/torch-e2e-model-optimizer/scripts/summarize.py \
  --run-dir run_YYYYMMDD_HHMMSS \
  --stop-reason "two no-gain iterations; remaining time is dataloader outside requested scope"
```

The final user-facing answer should report the best retained command/config, total E2E speedup, correctness, files changed, and the run directory.
