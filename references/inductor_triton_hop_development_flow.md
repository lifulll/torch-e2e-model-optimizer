# Inductor Triton HOP Pass Tutorial

Use this tutorial when an Inductor pass should replace a recognizable FX subgraph with one or more hand-written Triton kernels using `torch.ops.higher_order.triton_kernel_wrapper_functional`.

The intended path is:

```text
FX pattern rewrite
  -> insert triton_kernel_wrapper_functional
  -> Inductor lowers to ir.UserDefinedTritonKernel
  -> generated Inductor code launches the Triton kernel
```

This is usually simpler than adding a Python custom op or writing a new lowering, as long as the replacement is a normal Triton kernel launch.

## When To Use

Use this approach when:

- a slow FX subgraph is easy to recognize structurally,
- a hand-written Triton kernel can replace that subgraph,
- the generated Inductor code should launch the Triton kernel directly,
- one or more kernel launches are enough to express the replacement,
- no new Inductor IR node or scheduler behavior is required.

Do not use this path when the operation needs custom scheduler semantics, a new IR node, or complex aliasing behavior that the functional HOP cannot express.

## Files To Touch

Typical pass module:

```text
torch/_inductor/fx_passes/<your_pass>.py
```

The pass module should contain:

- Triton kernel definitions,
- FX matcher helpers,
- a Triton HOP insertion helper,
- an exported pass function such as `fuse_<your_pattern>(graph)`.

Wire the pass into:

```text
torch/_inductor/fx_passes/post_grad.py
```

Use `GraphTransformObserver` when registering the pass:

```python
GraphTransformObserver(gm, "<your_pass_name>").apply_graph_pass(
    fuse_<your_pattern>
)
```

Place the pass before later post-grad passes destroy the target pattern, and before `decompose_triton_kernel_wrapper_functional`.

## Step 1: Write And Validate The Triton Kernel

Write a normal `@triton.jit` kernel first and validate it independently.

Rules for HOP integration:

- kernel argument names must match the keys used in HOP `kwargs`,
- `tl.constexpr` values and launch options such as `num_warps` belong in `constant_args`,
- output tensors are passed as pointer arguments,
- written output names must be listed in `tensors_to_clone`.

## Step 2: Match The FX Subgraph

Write small helpers to recognize nodes:

```python
def _is_call(node, target):
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == target
    )
```

Then write structural matchers that:

- start from stable anchor nodes such as final `view`, `sum`, `where`, or `index_put`,
- verify each required producer and consumer,
- recover every tensor, scalar, shape, and constant needed by the Triton kernel,
- return `None` when any structure does not match.

Keep matchers conservative. A missed optimization is safer than rewriting the wrong graph.

## Step 3: Insert `triton_kernel_wrapper_functional`

Use `kernel_side_table` because FX graphs store IDs, not Python Triton kernel objects.

Minimal helper:

```python
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

triton_kernel_wrapper_functional = (
    torch.ops.higher_order.triton_kernel_wrapper_functional
)


def _triton_functional(
    graph,
    kernel,
    grid,
    kwargs,
    tensors_to_clone,
    constant_args=None,
):
    kernel_idx = kernel_side_table.add_kernel(kernel)
    constant_args_idx = kernel_side_table.add_constant_args(constant_args or {})
    return graph.call_function(
        triton_kernel_wrapper_functional,
        (),
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": {},
            "kwargs": kwargs,
            "tensors_to_clone": tensors_to_clone,
        },
    )
```

Field notes:

- `grid` may contain FX nodes or symbolic shape expressions.
- `kwargs` contains runtime tensor/scalar arguments.
- `constant_args` contains compile-time Triton constants and launch metadata.
- `tensors_to_clone` names the output tensors written by the kernel.

## Step 4: Allocate Outputs Functionally

Triton kernels often mutate output pointers. The functional HOP pattern is:

1. create output buffers in the FX graph,
2. pass them in `kwargs`,
3. list them in `tensors_to_clone`,
4. recover returned outputs with `operator.getitem`.

Skeleton:

```python
out = graph.call_function(
    aten.empty.memory_format,
    (shape,),
    alloc_kwargs,
)

hop_result = _triton_functional(
    graph,
    _your_kernel,
    grid,
    {"x": x, "out": out},
    tensors_to_clone=["out"],
    constant_args={"BLOCK": 1024, "num_warps": 4},
)

out = graph.call_function(operator.getitem, (hop_result, "out"))
```

For multiple kernels, insert several HOP calls in sequence and feed `getitem` outputs from one stage into the next.

## Step 5: Replace The Original Subgraph

Insert the replacement before the original output node, replace uses, and run dead-code elimination:

```python
with graph.inserting_before(old_output):
    new_output = _insert_your_triton_hop(graph, ...)

old_output.replace_all_uses_with(new_output)
graph.eliminate_dead_code()
```

In tests, call `graph.lint()` after the rewrite.

Key rules:

- Replace the final observable output nodes, not random internal nodes.
- Preserve any numerical post-processing from the original path.
- Do not leave the old dense path alive after replacement.
- Keep rewrites local and conservative.

## Step 6: Test The Pass

Cover two levels.

### FX Rewrite Test

Build or capture a small FX graph with the target pattern. After the pass, assert:

- the expected number of `triton_kernel_wrapper_functional` nodes exists,
- the original slow subgraph was removed by DCE,
- outputs use the HOP results,
- `graph.lint()` passes.

### torch.compile End-To-End Test

Compile a small function that triggers the pattern:

```python
compiled = torch.compile(fn, backend="inductor", fullgraph=True)
actual = compiled(...)
```

Compare against eager or a dense reference path with absolute and relative tolerances. Confirm generated `output_code.py` launches the Triton kernel and no longer contains the original slow path.

## Common Pitfalls

- Mismatched kernel argument names between Triton signature and HOP `kwargs`.
- Forgetting to put constexpr values in `constant_args`.
- Forgetting to include written outputs in `tensors_to_clone`.
- Using the wrong `operator.getitem` key for the returned output dictionary.
- Inserting the pass after another pass has destroyed the pattern.
- Rewriting a partial match that is not semantically equivalent.
- Forgetting `graph.eliminate_dead_code()`, leaving the original slow path in the graph.

## Checklist

- [ ] Identify the exact FX subgraph to replace.
- [ ] Validate the Triton kernel independently.
- [ ] Write a conservative matcher and recover all kernel inputs.
- [ ] Insert `triton_kernel_wrapper_functional` with correct `kernel_idx`, `constant_args_idx`, `grid`, `kwargs`, and `tensors_to_clone`.
- [ ] Allocate outputs in the FX graph and recover them with `operator.getitem`.
- [ ] Replace original output uses and run DCE.
- [ ] Wire the pass into post-grad at the right location.
- [ ] Add FX rewrite tests.
- [ ] Add a `torch.compile` end-to-end test.
- [ ] Inspect generated code to confirm the Triton launch path is used.
