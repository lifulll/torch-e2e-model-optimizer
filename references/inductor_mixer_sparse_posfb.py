# mypy: allow-untyped-defs
from __future__ import annotations

import math
import operator
import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.fx import Graph, Node
from torch.fx.experimental.symbolic_shapes import hint_int
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch._inductor.runtime.triton_helpers import math as tl_math


aten = torch.ops.aten
triton_kernel_wrapper_functional = torch.ops.higher_order.triton_kernel_wrapper_functional

_STAGE1_TARGET_ACTIVE_COLUMNS = 256
_REDUCE_BLOCK_LIMIT = 1024
_DEFAULT_NUM_WARPS = 8
_LOG_ENV = "CUSTOM_FUSION_PASS_LOG"


@dataclass(frozen=True)
class _SparsePosfbShape:
    batch: object
    patches: int
    patch_size: int
    dense_size: int
    patches_per_partial: int
    partials_per_batch: int
    stage1_block: int
    stage2_block: int


def _log_enabled() -> bool:
    return os.getenv(_LOG_ENV, "").lower() in {"1", "true", "yes", "on"}


def _node_desc(node: object) -> str:
    if isinstance(node, Node):
        return f"{node.name}:{node.op}:{node.target}"
    return repr(node)


def _log(stage: str, message: str, **kwargs) -> None:
    if not _log_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in kwargs.items())
    suffix = f" {details}" if details else ""
    print(f"[mixer_sparse_posfb][{stage}] {message}{suffix}")


@triton.jit
def _mixer_posfb_sparse_partials(
    diag_blocks,
    time_dense,
    gamma_ptr,
    beta_ptr,
    alpha_ptr,
    partial0,
    partial1,
    P: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    PATCHES_PER_PARTIAL: tl.constexpr,
    PARTIALS_PER_BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)[None, :]

    batch = pid // PARTIALS_PER_BATCH
    chunk_in_batch = pid % PARTIALS_PER_BATCH
    gamma = tl.load(gamma_ptr + 0)
    beta = tl.load(beta_ptr + 0)
    alpha = tl.load(alpha_ptr + 0)
    log_gamma = tl_math.log(gamma)
    zero = tl.full([1, BLOCK], 0.0, tl.float32)

    chunk_start = chunk_in_batch * PATCHES_PER_PARTIAL * K
    chunk_end = tl.minimum((chunk_in_batch + 1) * PATCHES_PER_PARTIAL * K, H)
    col = chunk_start + lane
    active_col = col < chunk_end
    patch = col // K
    jj = col % K
    lane_sum0 = zero
    lane_sum1 = zero
    for ii in tl.static_range(0, K):
        row = patch * K + ii
        diag_offset = batch * P * K * K + patch * K * K + ii * K + jj
        dense_offset = batch * H * H + row * H + col

        v = tl.load(diag_blocks + diag_offset, active_col, other=0.0)
        t = tl.load(time_dense + dense_offset, active_col, other=1.0)

        log_t = tl_math.log(t)
        p4 = tl_math.exp(log_t * beta)
        p5 = tl_math.exp(log_gamma * p4)

        term0 = v * p5
        d_gamma = tl.where((gamma == 0.0) & (p4 >= 0.0), zero, p5 * log_gamma)
        d_time = tl.where((t == 0.0) & (beta >= 0.0), zero, p4 * log_t)
        term1 = v * alpha * d_gamma * d_time

        lane_sum0 += tl.where(active_col, term0, zero)
        lane_sum1 += tl.where(active_col, term1, zero)

    tl.store(
        partial0 + pid + tl.full([1, 1], 0, tl.int64),
        tl.sum(lane_sum0, axis=1)[:, None],
    )
    tl.store(
        partial1 + pid + tl.full([1, 1], 0, tl.int64),
        tl.sum(lane_sum1, axis=1)[:, None],
    )


@triton.jit
def _mixer_posfb_reduce_pair_stage(
    in0,
    in1,
    out0,
    out1,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    zero = tl.full([BLOCK], 0.0, tl.float32)
    v0 = tl.load(in0 + offsets, mask, other=0.0)
    v1 = tl.load(in1 + offsets, mask, other=0.0)
    tl.store(out0 + pid, tl.sum(tl.where(mask, v0, zero), axis=0))
    tl.store(out1 + pid, tl.sum(tl.where(mask, v1, zero), axis=0))


def _is_call(node: object, target: object) -> bool:
    return isinstance(node, Node) and node.op == "call_function" and node.target == target

def _is_index_put(node: Node) -> bool:
    return _is_call(node, aten.index_put.default) or _is_call(node, aten.index_put_.default)

def _is_sum_keepdim_012(node: Node) -> bool:
    if not _is_call(node, aten.sum.dim_IntList):
        return False
    return len(node.args) >= 3 and list(node.args[1]) == [0, 1, 2] and node.args[2] is True


def _is_view_or_reshape(node: object) -> bool:
    return _is_call(node, aten.view.default) or _is_call(node, aten.reshape.default)


def _is_view_1_of_sum(node: Node) -> bool:
    if not _is_view_or_reshape(node):
        return False
    return len(node.args) >= 2 and list(node.args[1]) == [1] and _is_sum_keepdim_012(node.args[0])


def _find_pow_chain(root: object) -> Optional[tuple[Node, Node, Node, Node]]:
    if _is_call(root, aten.pow.Tensor_Tensor):
        base, exp = root.args
        if _is_call(exp, aten.pow.Tensor_Tensor):
            time_dense, beta = exp.args
            if (
                isinstance(base, Node)
                and isinstance(time_dense, Node)
                and isinstance(beta, Node)
            ):
                return base, exp, time_dense, beta
    if isinstance(root, Node):
        for arg in root.args:
            found = _find_pow_chain(arg)
            if found is not None:
                return found
    if isinstance(root, (tuple, list)):
        for arg in root:
            found = _find_pow_chain(arg)
            if found is not None:
                return found
    return None


def _trace_diag_blocks_from_dense_index_put(index_put_2: Node) -> Optional[Node]:
    if not _is_index_put(index_put_2) or len(index_put_2.args) < 3:
        _log(
            "trace_diag",
            "dense scatter is not index_put with value arg",
            node=_node_desc(index_put_2),
        )
        return None
    view_1160 = index_put_2.args[2]
    if not _is_view_or_reshape(view_1160):
        _log("trace_diag", "index_put_2 value is not view/reshape", value=_node_desc(view_1160))
        return None
    clone_229 = view_1160.args[0]
    if not _is_call(clone_229, aten.clone.default):
        _log("trace_diag", "view_1160 input is not clone", value=_node_desc(clone_229))
        return None
    permute_1312 = clone_229.args[0]
    if not _is_call(permute_1312, aten.permute.default):
        _log("trace_diag", "clone_229 input is not permute", value=_node_desc(permute_1312))
        return None
    index_put_1 = permute_1312.args[0]
    if not _is_index_put(index_put_1) or len(index_put_1.args) < 3:
        _log(
            "trace_diag",
            "permute_1312 input is not index_put with value arg",
            value=_node_desc(index_put_1),
        )
        return None
    view_1158 = index_put_1.args[2]
    if not _is_view_or_reshape(view_1158):
        _log("trace_diag", "index_put_1 value is not view/reshape", value=_node_desc(view_1158))
        return None
    clone_228 = view_1158.args[0]
    if not _is_call(clone_228, aten.clone.default):
        _log("trace_diag", "view_1158 input is not clone", value=_node_desc(clone_228))
        return None
    permute_1311 = clone_228.args[0]
    if not _is_call(permute_1311, aten.permute.default):
        _log("trace_diag", "clone_228 input is not permute", value=_node_desc(permute_1311))
        return None
    diag_blocks = permute_1311.args[0]
    if not _is_index_put(diag_blocks):
        _log("trace_diag", "permute_1311 input is not diag index_put", value=_node_desc(diag_blocks))
        return None
    _log("trace_diag", "found diagonal blocks", diag_blocks=_node_desc(diag_blocks))
    return diag_blocks


def _shape_as_int(value: object) -> Optional[int]:
    if isinstance(value, Node):
        value = value.meta.get("val")
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        try:
            return hint_int(value)
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _choose_reduce_block(numel: int) -> int:
    return min(_REDUCE_BLOCK_LIMIT, _next_power_of_2(max(1, numel)))


def _shape_from_meta(node: Node) -> Optional[tuple[int, ...]]:
    example = node.meta.get("val")
    if example is None or not hasattr(example, "shape"):
        return None
    shape = tuple(_shape_as_int(dim) for dim in example.shape)
    if any(dim is None for dim in shape):
        return None
    return shape


def _infer_sparse_posfb_shape(
    diag_blocks: Node,
    time_dense: Node,
) -> Optional[_SparsePosfbShape]:
    full_zero = diag_blocks.args[0]
    if not _is_call(full_zero, aten.full.default):
        _log("shape", "diag_blocks base is not aten.full.default", base=_node_desc(full_zero))
        return None

    diag_shape = full_zero.args[0]
    if not isinstance(diag_shape, (list, tuple)) or len(diag_shape) != 5:
        _log("shape", "diag_blocks shape is not 5D list/tuple", shape=diag_shape)
        return None

    batch = diag_shape[0]
    patches0 = _shape_as_int(diag_shape[1])
    patches1 = _shape_as_int(diag_shape[2])
    patch0 = _shape_as_int(diag_shape[3])
    patch1 = _shape_as_int(diag_shape[4])
    if (
        patches0 is None
        or patches1 is None
        or patch0 is None
        or patch1 is None
        or patches0 != patches1
        or patch0 != patch1
    ):
        _log(
            "shape",
            "diag_blocks shape is not [B,P,P,K,K] with static P/K",
            shape=diag_shape,
            patches0=patches0,
            patches1=patches1,
            patch0=patch0,
            patch1=patch1,
        )
        return None

    patches = patches0
    patch_size = patch0
    if patches <= 0 or patch_size <= 0:
        _log("shape", "patches or patch_size is non-positive", patches=patches, patch_size=patch_size)
        return None

    dense_size = patches * patch_size
    time_shape = _shape_from_meta(time_dense)
    if time_shape is not None:
        if len(time_shape) != 3 or time_shape[1] != time_shape[2]:
            _log("shape", "time_dense meta shape is not [B,H,H]", time_shape=time_shape)
            return None
        if time_shape[1] != dense_size:
            _log(
                "shape",
                "time_dense H does not match P*K",
                time_shape=time_shape,
                dense_size=dense_size,
            )
            return None

    patches_per_partial = min(
        patches,
        max(1, _STAGE1_TARGET_ACTIVE_COLUMNS // patch_size),
    )
    partials_per_batch = math.ceil(patches / patches_per_partial)
    stage1_block = _next_power_of_2(patches_per_partial * patch_size)
    stage2_block = _choose_reduce_block(dense_size)

    _log(
        "shape",
        "inferred sparse posfb shape",
        batch=batch,
        patches=patches,
        patch_size=patch_size,
        dense_size=dense_size,
        patches_per_partial=patches_per_partial,
        partials_per_batch=partials_per_batch,
        stage1_block=stage1_block,
        stage2_block=stage2_block,
    )
    return _SparsePosfbShape(
        batch=batch,
        patches=patches,
        patch_size=patch_size,
        dense_size=dense_size,
        patches_per_partial=patches_per_partial,
        partials_per_batch=partials_per_batch,
        stage1_block=stage1_block,
        stage2_block=stage2_block,
    )


def _extract_alloc_kwargs_from_diag_blocks(diag_blocks: Node) -> dict[str, object]:
    full_zero = diag_blocks.args[0]
    return {
        "dtype": full_zero.kwargs.get("dtype", torch.float32),
        "layout": full_zero.kwargs.get("layout", torch.strided),
        "device": full_zero.kwargs.get("device", torch.device("cuda")),
        "pin_memory": full_zero.kwargs.get("pin_memory", False),
    }


def _ceildiv(graph: Graph, numerator: object, denominator: int) -> object:
    if isinstance(numerator, int):
        return (numerator + denominator - 1) // denominator
    plus = graph.call_function(operator.add, (numerator, denominator - 1))
    numerator_meta = numerator.meta.get("val") if isinstance(numerator, Node) else numerator
    plus_meta = None
    if numerator_meta is not None:
        plus_meta = numerator_meta + denominator - 1
        _set_meta_val(plus, plus_meta)
    div = graph.call_function(operator.floordiv, (plus, denominator))
    if plus_meta is not None:
        _set_meta_val(div, plus_meta // denominator)
    return div


def _mul_int(graph: Graph, lhs: object, rhs: int) -> object:
    if isinstance(lhs, int):
        return lhs * rhs
    node = graph.call_function(operator.mul, (lhs, rhs))
    lhs_meta = lhs.meta.get("val") if isinstance(lhs, Node) else lhs
    if lhs_meta is not None:
        _set_meta_val(node, lhs_meta * rhs)
    return node


def _new_empty_meta(example_node: Node, numel: object):
    example = example_node.meta.get("val")
    return None if example is None or numel is None else example.new_empty((numel,))


def _empty_1d_like(
    graph: Graph,
    numel: object,
    alloc_kwargs: dict[str, object],
    example_node: Node,
    meta_numel: object,
) -> tuple[Node, object]:
    node = graph.call_function(aten.empty.memory_format, ([numel],), alloc_kwargs)
    meta = _new_empty_meta(example_node, meta_numel)
    _set_meta_val(node, meta)
    return node, meta


def _getitem_with_meta(graph: Graph, source: Node, key: str, meta) -> Node:
    node = graph.call_function(operator.getitem, (source, key))
    _set_meta_val(node, meta)
    return node


def _node_meta_shape0(node: Node) -> Optional[int]:
    example = node.meta.get("val")
    if example is None:
        return None
    return example.shape[0]


def _set_meta_val(node: object, value) -> None:
    if isinstance(node, Node) and value is not None:
        node.meta["val"] = value


def _triton_functional(
    graph: Graph,
    kernel,
    grid: list[tuple[object, ...]],
    kwargs: dict[str, object],
    tensors_to_clone: list[str],
    constant_args: Optional[dict[str, object]] = None,
) -> Node:
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


def _insert_sparse_posfb_triton_hop(
    graph: Graph,
    diag_blocks: Node,
    time_dense: Node,
    gamma: Node,
    beta: Node,
    alpha: Node,
) -> Optional[tuple[Node, Node]]:
    shape = _infer_sparse_posfb_shape(diag_blocks, time_dense)
    if shape is None:
        _log("insert", "shape inference failed", diag_blocks=_node_desc(diag_blocks))
        return None

    diag_values = diag_blocks.args[2]
    if not isinstance(diag_values, Node):
        _log("insert", "diag index_put value is not an FX node", value=_node_desc(diag_values))
        return None

    meta_batch = _node_meta_shape0(diag_values)
    meta_nblocks = (
        meta_batch * shape.partials_per_batch if meta_batch is not None else None
    )
    meta_qblocks = (
        (meta_nblocks + shape.stage2_block - 1) // shape.stage2_block
        if meta_nblocks is not None
        else None
    )
    if meta_qblocks is not None and meta_qblocks > _REDUCE_BLOCK_LIMIT:
        _log(
            "insert",
            "final reduction would exceed reduce block limit",
            meta_qblocks=meta_qblocks,
            limit=_REDUCE_BLOCK_LIMIT,
        )
        return None

    alloc_kwargs = _extract_alloc_kwargs_from_diag_blocks(diag_blocks)
    nblocks = _mul_int(graph, shape.batch, shape.partials_per_batch)
    _log(
        "insert",
        "inserting sparse posfb triton HOPs",
        diag_values=_node_desc(diag_values),
        nblocks=nblocks,
        meta_nblocks=meta_nblocks,
        meta_qblocks=meta_qblocks,
    )
    if isinstance(nblocks, Node):
        _set_meta_val(nblocks, meta_nblocks)
    partial0, partial0_meta = _empty_1d_like(
        graph, nblocks, alloc_kwargs, diag_values, meta_nblocks
    )
    partial1, partial1_meta = _empty_1d_like(
        graph, nblocks, alloc_kwargs, diag_values, meta_nblocks
    )

    stage1 = _triton_functional(
        graph,
        _mixer_posfb_sparse_partials,
        [(nblocks, 1, 1)],
        {
            "diag_blocks": diag_values,
            "time_dense": time_dense,
            "gamma_ptr": gamma,
            "beta_ptr": beta,
            "alpha_ptr": alpha,
            "partial0": partial0,
            "partial1": partial1,
        },
        tensors_to_clone=["partial0", "partial1"],
        constant_args={
            "P": shape.patches,
            "K": shape.patch_size,
            "H": shape.dense_size,
            "PATCHES_PER_PARTIAL": shape.patches_per_partial,
            "PARTIALS_PER_BATCH": shape.partials_per_batch,
            "BLOCK": shape.stage1_block,
            "num_warps": _DEFAULT_NUM_WARPS,
        },
    )
    _set_meta_val(stage1, {"partial0": partial0_meta, "partial1": partial1_meta})
    partial0 = _getitem_with_meta(graph, stage1, "partial0", partial0_meta)
    partial1 = _getitem_with_meta(graph, stage1, "partial1", partial1_meta)

    qblocks = _ceildiv(graph, nblocks, shape.stage2_block)
    _set_meta_val(qblocks, meta_qblocks)
    reduced0, reduced0_meta = _empty_1d_like(
        graph, qblocks, alloc_kwargs, diag_values, meta_qblocks
    )
    reduced1, reduced1_meta = _empty_1d_like(
        graph, qblocks, alloc_kwargs, diag_values, meta_qblocks
    )
    stage2 = _triton_functional(
        graph,
        _mixer_posfb_reduce_pair_stage,
        [(qblocks, 1, 1)],
        {
            "in0": partial0,
            "in1": partial1,
            "out0": reduced0,
            "out1": reduced1,
            "n": nblocks,
        },
        tensors_to_clone=["out0", "out1"],
        constant_args={"BLOCK": shape.stage2_block, "num_warps": _DEFAULT_NUM_WARPS},
    )
    _set_meta_val(stage2, {"out0": reduced0_meta, "out1": reduced1_meta})
    reduced0 = _getitem_with_meta(graph, stage2, "out0", reduced0_meta)
    reduced1 = _getitem_with_meta(graph, stage2, "out1", reduced1_meta)

    out0, out0_meta = _empty_1d_like(graph, 1, alloc_kwargs, diag_values, 1)
    out1, out1_meta = _empty_1d_like(graph, 1, alloc_kwargs, diag_values, 1)
    stage3_block = (
        _choose_reduce_block(meta_qblocks)
        if meta_qblocks is not None
        else shape.stage2_block
    )
    stage3 = _triton_functional(
        graph,
        _mixer_posfb_reduce_pair_stage,
        [(1, 1, 1)],
        {
            "in0": reduced0,
            "in1": reduced1,
            "out0": out0,
            "out1": out1,
            "n": qblocks,
        },
        tensors_to_clone=["out0", "out1"],
        constant_args={"BLOCK": stage3_block, "num_warps": _DEFAULT_NUM_WARPS},
    )
    _set_meta_val(stage3, {"out0": out0_meta, "out1": out1_meta})
    out0 = _getitem_with_meta(graph, stage3, "out0", out0_meta)
    out1 = _getitem_with_meta(graph, stage3, "out1", out1_meta)

    inf = graph.call_function(aten.full_like.default, (out0, math.inf))
    _set_meta_val(inf, out0_meta)
    out0 = graph.call_function(aten.nextafter.default, (out0, inf))
    _set_meta_val(out0, out0_meta)
    _log("insert", "inserted sparse posfb triton HOPs")
    return out0, out1


def _match_sum0(view_node: Node) -> Optional[tuple[Node, Node, Node, Node, Node]]:
    if not _is_view_1_of_sum(view_node):
        _log("match_sum0", "candidate is not reshape/view of target sum", node=_node_desc(view_node))
        return None
    mul = view_node.args[0].args[0]
    if not _is_call(mul, aten.mul.Tensor):
        _log("match_sum0", "sum input is not aten.mul.Tensor", node=_node_desc(view_node), sum_input=_node_desc(mul))
        return None

    lhs, rhs = mul.args
    index_put_2 = (
        lhs
        if _is_index_put(lhs)
        else rhs
        if _is_index_put(rhs)
        else None
    )
    pow_root = rhs if index_put_2 is lhs else lhs if index_put_2 is rhs else None
    if index_put_2 is None:
        _log(
            "match_sum0",
            "mul does not contain index_put dense scatter",
            node=_node_desc(view_node),
            lhs=_node_desc(lhs),
            rhs=_node_desc(rhs),
        )
        return None
    pow_chain = _find_pow_chain(pow_root)
    if pow_chain is None:
        _log("match_sum0", "could not find gamma ** (T ** beta) pow chain", pow_root=_node_desc(pow_root))
        return None
    gamma, _pow_time, time_dense, beta = pow_chain
    diag_blocks = _trace_diag_blocks_from_dense_index_put(index_put_2)
    if diag_blocks is None:
        _log("match_sum0", "could not trace diag_blocks from dense index_put", index_put_2=_node_desc(index_put_2))
        return None
    _log(
        "match_sum0",
        "matched alpha-grad branch",
        node=_node_desc(view_node),
        index_put_2=_node_desc(index_put_2),
        time_dense=_node_desc(time_dense),
        gamma=_node_desc(gamma),
        beta=_node_desc(beta),
    )
    return diag_blocks, index_put_2, time_dense, gamma, beta


def _match_sum1(
    view_node: Node,
    index_put_2: Node,
    time_dense: Node,
    gamma: Node,
    beta: Node,
) -> Optional[Node]:
    if not _is_view_1_of_sum(view_node):
        _log("match_sum1", "candidate is not reshape/view of target sum", node=_node_desc(view_node))
        return None
    reduced = view_node.args[0].args[0]
    # Be intentionally loose here: sum0 already anchors time/gamma/beta.  The
    # beta-grad expression is a product tree, so only walk mul nodes and avoid
    # descending into large pow/log/where branches just to prove containment.
    stack = [reduced]
    seen_ids: set[int] = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, Node):
            continue
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        if not _is_call(node, aten.mul.Tensor):
            continue

        a, b = node.args
        if a is index_put_2 and isinstance(b, Node):
            _log("match_sum1", "matched beta-grad branch", node=_node_desc(view_node), alpha=_node_desc(b))
            return b
        if b is index_put_2 and isinstance(a, Node):
            _log("match_sum1", "matched beta-grad branch", node=_node_desc(view_node), alpha=_node_desc(a))
            return a
        if isinstance(b, Node) and _is_call(b, aten.mul.Tensor):
            stack.append(b)
        if isinstance(a, Node) and _is_call(a, aten.mul.Tensor):
            stack.append(a)

    _log("match_sum1", "could not find mul directly pairing dense index_put", node=_node_desc(view_node))
    return None


def fuse_mixer_sparse_posfb(graph: Graph) -> None:
    views = [node for node in graph.nodes if _is_view_1_of_sum(node)]
    replacements = 0
    _log("pass", "starting fuse_mixer_sparse_posfb", candidate_views=len(views))

    for sum0_view in views:
        _log("pass", "trying alpha-grad candidate", node=_node_desc(sum0_view))
        matched = _match_sum0(sum0_view)
        if matched is None:
            _log("pass", "alpha-grad candidate did not match", node=_node_desc(sum0_view))
            continue
        diag_blocks, index_put_2, time_dense, gamma, beta = matched

        for sum1_view in views:
            if sum1_view is sum0_view:
                continue
            _log("pass", "trying beta-grad candidate", node=_node_desc(sum1_view))
            alpha = _match_sum1(sum1_view, index_put_2, time_dense, gamma, beta)
            if alpha is None:
                _log("pass", "beta-grad candidate did not match", node=_node_desc(sum1_view))
                continue

            with graph.inserting_before(sum0_view):
                replacement = _insert_sparse_posfb_triton_hop(
                    graph,
                    diag_blocks,
                    time_dense,
                    gamma,
                    beta,
                    alpha,
                )
                if replacement is None:
                    _log(
                        "pass",
                        "matched branches but HOP insertion declined",
                        sum0=_node_desc(sum0_view),
                        sum1=_node_desc(sum1_view),
                    )
                    continue
                out0, out1 = replacement

            sum0_view.replace_all_uses_with(out0)
            sum1_view.replace_all_uses_with(out1)
            replacements += 1
            _log(
                "pass",
                "replaced sparse posfb branches",
                sum0=_node_desc(sum0_view),
                sum1=_node_desc(sum1_view),
                replacements=replacements,
            )
            break

    if replacements:
        graph.eliminate_dead_code()
        _log("pass", "eliminated dead code after replacement", replacements=replacements)
    else:
        _log("pass", "no sparse posfb replacement applied")
