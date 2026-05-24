# Inductor and Triton Playbook

Use this only after model, runtime, graph break, and recompilation issues are no longer the main bottleneck.

## Evidence Collection

```bash
TORCH_COMPILE_DEBUG=1 python <entrypoint>
TORCH_LOGS="output_code,kernel_code,schedule,perf_hints,fusion" python <entrypoint> 2>&1 | tee artifacts/torch_logs/inductor.log
```

Save generated files, logs, and traces under the run directory. Link them from `step05_inductor.md` or `step06_triton_kernel.md`.

## Inductor Questions

- Which FX nodes feed the hotspot kernel?
- Is the kernel from pointwise fusion, reduction, matmul/conv template, attention, scatter/gather, or mixed fusion?
- Is fusion insufficient, or is an over-fused kernel reducing occupancy?
- Are there repeated layout conversions or materialized temporaries?
- Are dynamic shapes preventing specialization?
- Did autotune run, and what config was selected?

## Triton Kernel Questions

- What are the grid, block size, num warps, and num stages?
- Are loads/stores contiguous and coalesced?
- Are masks dense or mostly wasted?
- Does index math look more expensive than useful work?
- Is the kernel memory-bound, compute-bound, launch-bound, or register-pressure limited?
- Does a library op or model-level rewrite generate a better kernel?

## Action Order

1. Prefer model rewrites that let Inductor produce a better schedule.
2. Adjust expression boundaries to improve fusion or split harmful fusion.
3. Try supported compile/autotune options for the current PyTorch version.
4. Use targeted layout fixes only when copy cost is justified.
5. Add custom Triton only when generated code is a stable E2E bottleneck and fallback/tests are in place.

## Imported Inductor References

The following files are migrated from the local `torch-compile-model-optimization` skill for deeper Inductor pass work:

- `inductor_gated_cat_sum_copy_scheduler.py`
- `inductor_test_gated_cat_sum_copy_real_structure.py`
- `inductor_mixer_sparse_posfb.py`

Read them only when modifying Inductor passes, scheduler behavior, or real-structure tests. They are not needed for ordinary model-level optimization.
