# Optimization Catalog

Use this catalog to select the next optimization method from measured evidence. Prefer earlier layers unless profiler evidence points deeper.

## Method Registry

| ID | Layer | Trigger | Action | Skip When | Evidence |
|---|---|---|---|---|---|
| model-remove-side-effects | model code | Graph breaks or CPU gaps inside forward | Move logging, timers, prints, metrics, `.item()`, and host sync outside compiled path | Side effects are outside hot path | Graph break log, profiler CPU timeline |
| model-stabilize-shapes | model code | Recompilation or dynamic guards | Pad, bucket, or fix batch/sequence/image shapes | Padding cost exceeds compile/runtime gain | `TORCH_LOGS=recompiles,dynamic` |
| model-rewrite-op-chain | model code | Many tiny pointwise ops or poor fusion | Replace hand-written op chains with PyTorch functional/library ops | Equivalent library op is unavailable or slower | profiler, generated code |
| model-layout-cleanup | model code | Hidden copies, bad stride, transpose-heavy path | Move layout conversion to boundaries; remove redundant clone/contiguous/permute | Copy cost dominates | profiler memory ops, tensor stride audit |
| runtime-inference-mode | PyTorch runtime | Inference path includes autograd overhead | Use `torch.inference_mode()` or `torch.no_grad()` | Training or gradients required | profiler autograd events |
| runtime-dataloader | PyTorch runtime | Accelerator idle before compute | Tune workers, prefetch, pin memory, persistent workers, caching, transforms | Synthetic benchmark already excludes data | profiler gaps and data timing |
| compile-region | torch.compile | Compiled coverage too small or too broad | Compile stable tensor-heavy region; use regional compile for repeated blocks | Full model compile is already stable and fast | graph/module coverage |
| compile-mode | torch.compile | Warm path may benefit from mode changes | Compare default, `reduce-overhead`, `max-autotune` one at a time | Compile time or memory budget forbids it | steady-state and compile-time table |
| compile-graph-break | torch.compile | Graph breaks split hot path | Rewrite unsupported Python into tensor ops or isolate disabled region | Unsupported op is cold | graph break log |
| compile-recompile | torch.compile | Guards repeatedly fail | Stabilize shape, dtype, stride, Python constants, module state; use precise `mark_dynamic` | True dynamic workload makes specialization impossible | recompile log |
| backward-optimizer | AOT/backward | Optimizer or backward dominates step | Try fused/foreach optimizer, compile optimizer, checkpointing, accumulation | Optimizer is not a bottleneck | profiler step breakdown |
| distributed-overlap | distributed | Scaling efficiency poor | Tune bucket size, accumulation, overlap, sharding, rank balance | Single-rank bottleneck not understood | rank traces and comm timings |
| inductor-fusion | Inductor | Too many small kernels or over-fused slow kernel | Adjust expression boundaries to improve fusion or split over-fused group | E2E bottleneck is elsewhere | schedule/fusion logs |
| inductor-layout | Inductor | Generated code has expensive index math or copies | Change layout/stride at model boundaries or targeted `.contiguous()` | Copy is more expensive than saved kernel time | generated code and profiler |
| inductor-autotune | Inductor | Matmul/conv/reduction kernel config looks weak | Try compile mode/config supported by current PyTorch | Internal option not present or unstable | output code and selected config |
| triton-hotspot | Triton | One or few generated Triton kernels dominate | Inspect grid/block/warps/stages/masks/strides; influence generation first | Hotspot is not stable across inputs | profiler and generated kernel |
| custom-kernel-pass | Inductor pass | Generated/library kernel remains proven bottleneck and target is a recognizable FX pattern | Insert hand-written Triton kernel through `triton_kernel_wrapper_functional` HOP, with fallback and tests | Maintenance cost, shape coverage, or matcher safety is unjustified | FX rewrite test, generated code, microbench plus E2E |

## Combining Rules

- Keep precision and batch size fixed for attribution. Do not select AMP/mixed precision, batch-size sweeps, or microbatch sweeps as optimization methods unless explicitly requested by the user.
- Do not combine compile mode changes with model rewrites in the same attribution iteration.
- Do not tune Triton or add pass-inserted custom kernels before graph breaks and recompilation are under control.
- If a change improves kernel time but not E2E time, keep it only when it is required for a later retained change.

## No-Gain Interpretation

Treat an iteration as no-gain when the primary E2E metric improves less than the measured noise threshold, usually 1-2%, or when the improvement is not reproducible. Two consecutive no-gain iterations should trigger a deeper bottleneck re-check or stopping summary.
