# Reporting Contract

Use this as the output completeness checklist. `workflow_contract.md` owns run directory layout, measurement protocol, and loop control.

## Per-Step Summary Checklist

Every `stepXX_*.md` file must include:

- Objective and hypothesis.
- Evidence that selected this layer.
- Exact command and environment differences.
- Cleanup command and result before model runs.
- Code/config files changed.
- Patch archive path in `patches/`.
- Correctness test and tolerance.
- Before/after E2E metric.
- Secondary metrics: memory, compile time, graph breaks, recompiles, top kernels.
- Decision: retained, reverted, optional, or blocked.
- Skipped sub-threshold ideas, if any.
- Next bottleneck.

Use `templates/step_summary.md` for the file shape.

## Iteration Table Checklist

Keep `iteration_table.md` in this exact shape:

```markdown
| Iter | Layer | Hypothesis | Code/config changed | Files changed | Correctness | E2E metric before | E2E metric after | Delta | Retained | Evidence |
|---:|---|---|---|---|---|---:|---:|---:|---|---|
```

Use `scripts/record_iteration.py` when possible.
Use `templates/iteration_row.md` when creating the table by hand.

## Final Summary Checklist

`final_summary.md` must include:

- baseline and final E2E metric,
- total percent gain,
- retained changes,
- rejected changes and why,
- files modified,
- patch files generated under `patches/`,
- cleanup command used before model runs,
- correctness status,
- compile-time and memory impact,
- remaining bottleneck,
- reason for stopping.

Do not call the optimization finished if no final summary exists.
Use `templates/final_summary.md` for the final report shape.
