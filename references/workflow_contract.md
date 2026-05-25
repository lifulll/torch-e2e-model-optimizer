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
python <skill>/scripts/init_run.py --root . --entry-cmd "<benchmark command>" --objective "<primary metric>" --env-json /tmp/torch_env.json
```

Every optimization step must update its `stepXX_*.md`, even when no change is retained.

## Measurement Protocol

Before every comparison:

1. Pin the same model revision, config, seed, shape distribution, precision, batch size, and device placement.
2. Warm up enough iterations to exclude compile and cache effects.
3. Report compile time separately from steady-state.
4. Use median and p90, or repeated runs sufficient to expose noise.
5. Synchronize accelerator timing only at measurement boundaries.
6. Record peak memory and any OOM, recompilation, graph break, or fallback.
7. Run correctness checks before accepting faster results.

Useful commands:

```bash
TORCH_LOGS="graph_breaks,recompiles,dynamic" python <entrypoint>
TORCH_TRACE=./run_trace python <entrypoint>
TORCH_COMPILE_DEBUG=1 python <entrypoint>
```

Use PyTorch Profiler or platform profiler when timing alone cannot explain the bottleneck.

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

## Per-Step Summary

Each `stepXX_*.md` must include:

- what was optimized and why this layer was selected,
- exact commands,
- files and code regions changed,
- correctness result and tolerance,
- end-to-end performance before and after,
- profiler or compiler evidence,
- retained/reverted/optional decision,
- next bottleneck and next hypothesis.

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
