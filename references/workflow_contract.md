# Workflow Contract

Use this reference for run setup, measurement, loop control, and required output files.

## Run Directory

Create one run directory near the benchmark or repository root:

```text
run_YYYYMMDD_HHMMSS/
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

Fast path:

```bash
python <skill>/scripts/collect_env.py --out /tmp/torch_env.json --markdown /tmp/torch_env.md
python <skill>/scripts/init_run.py --root . --entry-cmd "<benchmark command>" --cleanup-cmd "<kill command>" --objective "<primary metric>" --env-json /tmp/torch_env.json --env-md /tmp/torch_env.md
```

Every optimization step must update its `stepXX_*.md`, even when no change is retained.

## Measurement Protocol

Before every comparison:

1. Run the user-provided kill/cleanup command to remove stale model processes.
2. Pin the same model revision, config, seed, shape distribution, precision, batch size, and device placement.
3. Warm up enough iterations to exclude compile and cache effects.
4. Report compile time separately from steady-state.
5. Use median and p90, or repeated runs sufficient to expose noise.
6. Synchronize accelerator timing only at measurement boundaries.
7. Record peak memory and any OOM, recompilation, graph break, or fallback.
8. Run correctness checks before accepting faster results.

Use the helper when possible:

```bash
python <skill>/scripts/run_with_cleanup.py --cleanup-cmd "<kill command>" --entry-cmd "<benchmark command>"
```

Useful commands:

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" python <entrypoint>
TORCH_TRACE=./run_trace python <entrypoint>
TORCH_COMPILE_DEBUG=1 python <entrypoint>
```

Use PyTorch Profiler or platform profiler when timing alone cannot explain the bottleneck.

## Materiality Threshold

Only benchmark and retain optimization candidates that are expected to improve the primary E2E metric by at least 2%, or by more than the measured benchmark noise when noise is larger than 2%. Small cleanups below that threshold can be noted in the step summary as skipped, but they do not block moving to the next layer.

## Iteration Table

Maintain `iteration_table.md` as the authoritative ledger:

```markdown
| Iter | Layer | Hypothesis | Code/config changed | Files changed | Correctness | E2E metric before | E2E metric after | Delta | Retained | Evidence |
|---:|---|---|---|---|---|---:|---:|---:|---|---|
| 0 | baseline | establish reference | none | none | pass | 123 tok/s | 123 tok/s | 0% | yes | baseline.md |
```

Use percent improvement consistently:

```text
throughput_gain = (after / before - 1) * 100
latency_gain = (before / after - 1) * 100
```

Use `scripts/record_iteration.py --materiality-threshold 2.0 --noise-percent <measured_noise>` when recording benchmarked candidates. The script writes materiality fields into `state.json`; `scripts/summarize.py` reports sub-threshold and trailing no-gain counts.

## Per-Step Summary

Each `stepXX_*.md` must include:

- what was optimized and why this layer was selected,
- exact commands,
- cleanup command and result,
- files and code regions changed,
- correctness result and tolerance,
- end-to-end performance before and after,
- profiler or compiler evidence,
- retained/reverted/optional decision,
- skipped sub-threshold ideas, if any,
- next bottleneck and next material hypothesis.

## Final Summary

Write `final_summary.md` with:

- best retained command/config,
- total E2E improvement from baseline,
- final latency/throughput/memory,
- compile-time impact,
- correctness status,
- table of all iterations,
- retained and rejected changes,
- remaining bottleneck and stop reason,
- risks, version constraints, and follow-up options.
