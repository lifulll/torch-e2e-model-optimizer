# Branch and Attribution Protocol

Use this when an optimization layer has multiple plausible implementations and benchmark noise makes single-branch decisions weak. This adapts the branch/select/ablation pattern from kernel optimization to torch end-to-end model work.

## When to Use

Use branch exploration for:

- compile mode comparisons,
- batch or microbatch sweeps,
- dynamic-shape strategy choices,
- layout variants,
- dataloader settings,
- optimizer variants,
- Inductor fusion boundary rewrites,
- Triton/custom kernel variants.

Do not use it for trivial one-line fixes with obvious correctness and low benchmark noise.

## Branch Rules

Create branches that vary one family of decisions:

| Family | Valid Branch Examples |
|---|---|
| compile mode | default, reduce-overhead, max-autotune |
| shape strategy | no padding, bucketed padding, precise mark_dynamic, dynamic=True |
| layout | baseline, channels-last, targeted contiguous, boundary-only conversion |
| optimizer | baseline, foreach, fused, compiled optimizer |
| dataloader | baseline workers, more workers, persistent workers, cached transforms |
| Inductor influence | baseline expression, split boundary, fused expression, library op |

Each branch must record:

- exact code/config difference,
- correctness result,
- compile time,
- steady-state E2E metric,
- memory,
- graph breaks/recompiles if relevant,
- profiler evidence for why it won or lost.

## Selection Rule

Keep the fastest correct branch only when:

1. improvement exceeds the noise threshold,
2. it improves the primary E2E metric,
3. secondary costs are acceptable,
4. the reason is explainable by logs, profiler evidence, or generated code.

If a branch improves a microphase but worsens total E2E, reject it.

## Ablation Rule

When a champion includes multiple changes, remove one change at a time and remeasure. Mark a method as effective only when removing it measurably hurts the champion beyond noise. If attribution is unclear, record the branch as optional or split it into smaller iterations.

## Reporting

Add a row per branch or a compact branch table in the relevant `stepXX_*.md`:

```markdown
| Branch | Change | Correctness | Compile time | E2E metric | Delta vs baseline | Decision | Evidence |
|---|---|---|---:|---:|---:|---|---|
```

Then add only the retained champion to `iteration_table.md`.
