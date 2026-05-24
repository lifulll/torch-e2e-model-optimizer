# Reporting Contract

Every run must be auditable from files, not memory.

## Per-Step Summary

Each `stepXX_*.md` file must include:

- Objective and hypothesis.
- Evidence that selected this layer.
- Exact command and environment differences.
- Code/config files changed.
- Correctness test and tolerance.
- Before/after E2E metric.
- Secondary metrics: memory, compile time, graph breaks, recompiles, top kernels.
- Decision: retained, reverted, optional, or blocked.
- Next bottleneck.

## Iteration Table

Keep `iteration_table.md` in this exact shape:

```markdown
| Iter | Layer | Hypothesis | Code/config changed | Files changed | Correctness | E2E metric before | E2E metric after | Delta | Retained | Evidence |
|---:|---|---|---|---|---|---:|---:|---:|---|---|
```

Use `scripts/record_iteration.py` when possible.

## Final Summary

`final_summary.md` must include:

- baseline and final E2E metric,
- total percent gain,
- retained changes,
- rejected changes and why,
- files modified,
- correctness status,
- compile-time and memory impact,
- remaining bottleneck,
- reason for stopping.

Do not call the optimization finished if no final summary exists.
