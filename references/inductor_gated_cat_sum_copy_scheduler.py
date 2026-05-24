# mypy: allow-untyped-defs
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.fx.experimental.symbolic_shapes import hint_int

from torch._inductor import ir


_LOG_ENV = "CUSTOM_FUSION_PASS_LOG"
_DEFAULT_XBLOCK = 128
_DEFAULT_RBLOCK = 8
_FAST_XBLOCK = 512
_FAST_RBLOCK = 8
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_STAGES = 1

_KERNEL_COUNTER = 0


@dataclass(frozen=True)
class _LogicalRank4Tensor:
    buffer: object
    stride: tuple[object, object, object, object]


@dataclass(frozen=True)
class _GatedCatMatch:
    cat: object
    copy: object
    sum: object
    base: _LogicalRank4Tensor
    value: _LogicalRank4Tensor
    gate: _LogicalRank4Tensor
    batch: object
    seq: object
    channels: object
    hidden: object


def _log_enabled() -> bool:
    return os.getenv(_LOG_ENV, "").lower() in {"1", "true", "yes", "on"}


def _log(message: str, **kwargs) -> None:
    if not _log_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in kwargs.items())
    suffix = f" {details}" if details else ""
    print(f"[gated_cat_sum_copy_scheduler] {message}{suffix}")


def _hinted_int(value):
    if isinstance(value, int):
        return value
    try:
        return int(hint_int(value))
    except Exception:
        pass
    try:
        from torch._inductor.virtualized import V

        return int(V.graph.sizevars.size_hint(value))
    except Exception:
        return None


def _ceildiv_expr(value, divisor):
    return (value + divisor - 1) // divisor


def _same_expr(lhs, rhs) -> bool:
    lhs_i = _hinted_int(lhs)
    rhs_i = _hinted_int(rhs)
    if lhs_i is not None and rhs_i is not None:
        return lhs_i == rhs_i
    try:
        return bool(lhs == rhs)
    except Exception:
        return str(lhs) == str(rhs)


def _targets(node) -> set[object]:
    if node.node is None:
        return set()
    return {origin.target for origin in node.node.get_origins()}


def _has_target(node, targets: Iterable[object]) -> bool:
    node_targets = _targets(node)
    return any(target in node_targets for target in targets)


def _is_cat_node(node) -> bool:
    return _has_target(node, (torch.ops.aten.cat.default,))


def _is_copy_node(node) -> bool:
    return (not node.is_reduction()) and _has_target(
        node,
        (
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten.to.dtype,
        ),
    )


def _is_sum_node(node) -> bool:
    return node.is_reduction() and _has_target(
        node,
        (
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.sum.default,
        ),
    )


def _reads_buffer(node, names: set[str]) -> bool:
    return any(dep.name in names for dep in node.unmet_dependencies)


def _has_bmm_consumer(nodes, producer) -> bool:
    names = set(producer.get_buffer_names())
    return any(
        node is not producer
        and _reads_buffer(node, names)
        and _has_target(node, (torch.ops.aten.bmm.default,))
        for node in nodes
    )


def _producer_targets(scheduler, name: str) -> set[object]:
    if name in scheduler.name_to_buf:
        producer = scheduler.name_to_buf[name].node
        if producer is not None:
            return _targets(producer)
    return set()


def _is_gated_silu_backward_targets(targets: set[object]) -> bool:
    return (
        torch.ops.aten.sigmoid.default in targets
        and torch.ops.aten.mul.Tensor in targets
        and (
            torch.ops.aten.add.Scalar in targets
            or torch.ops.aten.add.Tensor in targets
        )
        and torch.ops.aten.sub.Tensor in targets
    )


def _unwrap_storage_box(value):
    if hasattr(value, "data"):
        value = value.data
    if hasattr(value, "data"):
        value = value.data
    return value


def _as_logical_rank4(buf, batch, seq, channels, hidden):
    size = tuple(buf.get_size())
    stride = tuple(buf.get_stride())
    if len(size) == 4:
        if not (
            _same_expr(size[0], batch)
            and _same_expr(size[1], seq)
            and _same_expr(size[2], channels)
            and _same_expr(size[3], hidden)
        ):
            return None
        return _LogicalRank4Tensor(buf, stride)
    if len(size) == 3:
        if not (
            _same_expr(size[0], channels)
            and _same_expr(size[1], batch * seq)
            and _same_expr(size[2], hidden)
        ):
            return None
        logical_stride = (seq * stride[1], stride[1], stride[0], stride[2])
        return _LogicalRank4Tensor(buf, logical_stride)
    return None


def _candidate_tensors(scheduler, cat):
    from torch._inductor.virtualized import V

    names: list[str] = []
    for dep in cat.read_writes.reads:
        if dep.name not in names and (
            dep.name in scheduler.name_to_buf or dep.name in V.graph.graph_inputs
        ):
            names.append(dep.name)

    pairs = []
    for name in names:
        if name in scheduler.name_to_buf:
            buf = scheduler.name_to_buf[name].node
        else:
            buf = _unwrap_storage_box(V.graph.graph_inputs[name])
        if (
            hasattr(buf, "get_layout")
            and hasattr(buf, "get_size")
            and hasattr(buf, "get_stride")
            and hasattr(buf, "get_dtype")
            and buf.get_device() is not None
            and buf.get_device().type == "cuda"
        ):
            pairs.append((name, buf))
    return pairs


def _pick_gated_inputs(scheduler, cat):
    pairs = _candidate_tensors(scheduler, cat)
    base_pairs = [(name, buf) for name, buf in pairs if buf.get_dtype() == torch.bfloat16]
    fp32_pairs = [
        (name, buf)
        for name, buf in pairs
        if buf.get_dtype() == torch.float32 and len(buf.get_size()) == 4
    ]
    if len(base_pairs) != 1 or len(fp32_pairs) != 2:
        return None

    base_name, base_buf = base_pairs[0]
    del base_name
    (value_name, value_buf), (gate_name, gate_buf) = fp32_pairs
    if tuple(value_buf.get_size()) != tuple(gate_buf.get_size()):
        return None

    batch, seq, channels, hidden = value_buf.get_size()
    base = _as_logical_rank4(base_buf, batch, seq, channels, hidden)
    value = _as_logical_rank4(value_buf, batch, seq, channels, hidden)
    gate = _as_logical_rank4(gate_buf, batch, seq, channels, hidden)
    if base is None or value is None or gate is None:
        return None

    producer_targets = [
        _producer_targets(scheduler, value_name),
        _producer_targets(scheduler, gate_name),
    ]
    visible_producers = [targets for targets in producer_targets if targets]
    if visible_producers and not any(
        _is_gated_silu_backward_targets(targets) for targets in visible_producers
    ):
        return None

    return base, value, gate, batch, seq, channels, hidden


def _is_sum_output(sum_node, channels, hidden) -> bool:
    size = tuple(sum_node.node.get_size())
    return (
        sum_node.node.get_dtype() == torch.float32
        and len(size) in (4, 5)
        and _same_expr(size[0], 1)
        and _same_expr(size[1], 1)
        and _same_expr(size[2], channels)
        and _same_expr(size[3], hidden * 2)
    )


def _is_copy_output(copy_node, batch, seq, channels, hidden) -> bool:
    size = tuple(copy_node.node.get_size())
    return (
        len(size) == 4
        and copy_node.node.get_dtype() == torch.bfloat16
        and _same_expr(size[0], batch)
        and _same_expr(size[1], seq)
        and _same_expr(size[2], channels)
        and _same_expr(size[3], hidden * 2)
    )


def _match_cat_sum_copy(nodes, cat):
    cat_buffers = set(cat.get_buffer_names())
    copy_users = [
        node
        for node in nodes
        if node is not cat and _is_copy_node(node) and _reads_buffer(node, cat_buffers)
    ]
    sum_users = [
        node
        for node in nodes
        if node is not cat and _is_sum_node(node) and _reads_buffer(node, cat_buffers)
    ]
    if len(copy_users) != 1 or len(sum_users) != 1:
        return None

    copy = copy_users[0]
    sum_node = sum_users[0]
    if not _has_bmm_consumer(nodes, copy):
        _log("skip copy without bmm consumer", cat=cat.get_name(), copy=copy.get_name())
        return None

    picked = _pick_gated_inputs(cat.scheduler, cat)
    if picked is None:
        return None
    base, value, gate, batch, seq, channels, hidden = picked

    if not _is_copy_output(copy, batch, seq, channels, hidden):
        _log("skip copy layout", copy=copy.get_name(), size=copy.node.get_size())
        return None
    if not _is_sum_output(sum_node, channels, hidden):
        _log("skip sum layout", sum=sum_node.get_name(), size=sum_node.node.get_size())
        return None

    return _GatedCatMatch(
        cat=cat,
        copy=copy,
        sum=sum_node,
        base=base,
        value=value,
        gate=gate,
        batch=batch,
        seq=seq,
        channels=channels,
        hidden=hidden,
    )


def _meta_dict(kernel_name: str, device, *, signature, constants, num_load, num_reduction):
    from triton.compiler.compiler import AttrsDescriptor

    from torch._inductor.codegen.triton import TritonKernel
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties
    from torch._inductor.virtualized import V

    inductor_meta = {
        **triton_heuristics.FixedGrid.setup_grid_as_args(),
        "autotune_hints": set(),
        "kernel_name": kernel_name,
        "mutated_arg_names": [],
        "optimize_mem": V.graph.is_inference or V.graph.is_backward,
        "no_x_dim": False,
        "num_load": num_load,
        "num_reduction": num_reduction,
        **TritonKernel.inductor_meta_common(),
    }
    triton_meta = {
        "signature": signature,
        "device": DeviceProperties.create(device),
        "constants": constants,
        "configs": [
            AttrsDescriptor.from_dict(
                {
                    "arg_properties": {
                        "tt.divisibility": (),
                        "tt.equal_to": (),
                        "tt.pointer_range": (),
                    },
                    "cls": "HIPAttrsDescriptor" if torch.version.hip else "AttrsDescriptor",
                }
            )
        ],
    }
    return inductor_meta, triton_meta


def _triton_prelude(kernel_name: str, triton_meta, inductor_meta) -> str:
    return f"""
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties

triton_helpers.set_driver_to_gpu()

@triton_heuristics.user_autotune(
    configs=[{{'num_warps': {_DEFAULT_NUM_WARPS}, 'num_stages': {_DEFAULT_NUM_STAGES}}}],
    triton_meta={triton_meta!r},
    inductor_meta={inductor_meta!r},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
"""


def _kernel_tail(value_offset: str = "value_offset", gate_offset: str = "gate_offset") -> str:
    return f"""
    base = tl.load(base_ptr + base_offset, mask, eviction_policy="evict_last", other=0.0).to(tl.float32)
    value = tl.load(value_ptr + {value_offset}, mask, eviction_policy="evict_last", other=0.0)
    gate = tl.load(gate_ptr + {gate_offset}, mask, eviction_policy="evict_last", other=0.0)
    sig = tl.sigmoid(gate)

    left = (base * value) * (sig * (1.0 + gate * (1.0 - sig)))
    right = base * gate * sig

    tl.store(copy_out + copy_left_offset, left, mask)
    tl.store(copy_out + copy_right_offset, right, mask)

    left_reduced = tl.sum(tl.where(mask, left, 0.0), axis=0)
    right_reduced = tl.sum(tl.where(mask, right, 0.0), axis=0)
    split_idx = tl.program_id(1) % sum_split
    tl.atomic_add(sum_out + channel * sum_s2 + half_col * sum_s3 + split_idx * sum_s4, left_reduced[None, :], sem="relaxed", mask=xmask)
    tl.atomic_add(sum_out + channel * sum_s2 + (half_col + hidden) * sum_s3 + split_idx * sum_s4, right_reduced[None, :], sem="relaxed", mask=xmask)
"""


def _kernel_source(kernel_name: str, device, *, rblock: int, xblock: int) -> str:
    signature = {
        "base_ptr": "*bf16",
        "value_ptr": "*fp32",
        "gate_ptr": "*fp32",
        "copy_out": "*bf16",
        "sum_out": "*fp32",
        "rnumel": "i32",
        "seq": "i32",
        "hidden": "i32",
        "half_xnumel": "i32",
        "base_s0": "i64",
        "base_s1": "i64",
        "base_s2": "i64",
        "base_s3": "i64",
        "value_s0": "i64",
        "value_s1": "i64",
        "value_s2": "i64",
        "value_s3": "i64",
        "gate_s0": "i64",
        "gate_s1": "i64",
        "gate_s2": "i64",
        "gate_s3": "i64",
        "copy_s0": "i64",
        "copy_s1": "i64",
        "copy_s2": "i64",
        "copy_s3": "i64",
        "sum_split": "i32",
        "sum_s2": "i64",
        "sum_s3": "i64",
        "sum_s4": "i64",
    }
    constants = {
        "RBLOCK": rblock,
        "XBLOCK": xblock,
    }
    inductor_meta, triton_meta = _meta_dict(
        kernel_name,
        device,
        signature=signature,
        constants=constants,
        num_load=3,
        num_reduction=1,
    )
    return _triton_prelude(kernel_name, triton_meta, inductor_meta) + f"""def {kernel_name}(
    base_ptr,
    value_ptr,
    gate_ptr,
    copy_out,
    sum_out,
    rnumel,
    seq,
    hidden,
    half_xnumel,
    base_s0,
    base_s1,
    base_s2,
    base_s3,
    value_s0,
    value_s1,
    value_s2,
    value_s3,
    gate_s0,
    gate_s1,
    gate_s2,
    gate_s3,
    copy_s0,
    copy_s1,
    copy_s2,
    copy_s3,
    sum_split,
    sum_s2,
    sum_s3,
    sum_s4,
    RBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    r = tl.program_id(1) * RBLOCK + tl.arange(0, RBLOCK)[:, None]
    xmask = x < half_xnumel
    rmask = r < rnumel
    mask = xmask & rmask

    channel = x // hidden
    half_col = x - channel * hidden
    batch = r // seq
    seq_idx = r - batch * seq

    base_offset = batch * base_s0 + seq_idx * base_s1 + channel * base_s2 + half_col * base_s3
    value_offset = batch * value_s0 + seq_idx * value_s1 + channel * value_s2 + half_col * value_s3
    gate_offset = batch * gate_s0 + seq_idx * gate_s1 + channel * gate_s2 + half_col * gate_s3
    copy_left_offset = batch * copy_s0 + seq_idx * copy_s1 + channel * copy_s2 + half_col * copy_s3
    copy_right_offset = copy_left_offset + hidden * copy_s3

""" + _kernel_tail()


def _split_channel_major_kernel_source(
    kernel_name: str,
    device,
    *,
    hidden: int,
    half_xnumel: int,
    rblock: int,
    xblock: int,
) -> str:
    signature = {
        "base_ptr": "*bf16",
        "value_ptr": "*fp32",
        "gate_ptr": "*fp32",
        "copy_out": "*bf16",
        "sum_out": "*fp32",
        "rnumel": "i32",
        "base_s1": "i64",
        "base_s2": "i64",
        "value_s1": "i64",
        "value_s2": "i64",
        "gate_s1": "i64",
        "gate_s2": "i64",
        "copy_s1": "i64",
        "copy_s2": "i64",
        "sum_split": "i32",
        "sum_s2": "i64",
        "sum_s3": "i64",
        "sum_s4": "i64",
    }
    constants = {
        "hidden": hidden,
        "half_xnumel": half_xnumel,
        "RBLOCK": rblock,
        "XBLOCK": xblock,
    }
    inductor_meta, triton_meta = _meta_dict(
        kernel_name,
        device,
        signature=signature,
        constants=constants,
        num_load=3,
        num_reduction=1,
    )
    return _triton_prelude(kernel_name, triton_meta, inductor_meta) + f"""def {kernel_name}(
    base_ptr,
    value_ptr,
    gate_ptr,
    copy_out,
    sum_out,
    rnumel,
    base_s1,
    base_s2,
    value_s1,
    value_s2,
    gate_s1,
    gate_s2,
    copy_s1,
    copy_s2,
    sum_split,
    sum_s2,
    sum_s3,
    sum_s4,
    hidden: tl.constexpr,
    half_xnumel: tl.constexpr,
    RBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    r = tl.program_id(1) * RBLOCK + tl.arange(0, RBLOCK)[:, None]
    xmask = x < half_xnumel
    rmask = r < rnumel
    mask = xmask & rmask

    channel = x // hidden
    half_col = x - channel * hidden

    base_offset = r * base_s1 + channel * base_s2 + half_col
    value_offset = r * value_s1 + channel * value_s2 + half_col
    gate_offset = r * gate_s1 + channel * gate_s2 + half_col
    copy_left_offset = r * copy_s1 + channel * copy_s2 + half_col
    copy_right_offset = copy_left_offset + hidden
""" + _kernel_tail("value_offset", "gate_offset")


def _py_tuple(wrapper, values):
    return "(" + ", ".join(wrapper.val_to_arg_str(v) for v in values) + ("," if len(values) == 1 else "") + ")"


class GatedCatSumCopyTemplateBuffer(ir.TemplateBuffer):
    def __init__(self, match: _GatedCatMatch) -> None:
        copy_buf = match.copy.node
        sum_buf = match.sum.node
        super().__init__(
            layout=ir.MultiOutputLayout(device=copy_buf.get_device()),
            inputs=[match.base.buffer, match.value.buffer, match.gate.buffer],
            make_kernel_render=self._make_kernel_render,
        )
        self.match = match
        self.copy_buffer = ir.Buffer(name=copy_buf.get_name(), layout=copy_buf.layout)
        self.sum_buffer = ir.Buffer(name=sum_buf.get_name(), layout=sum_buf.layout)
        self.operation_name = f"{match.copy.get_name()}_{match.sum.get_name()}_gated_cat_sum_copy"

    def _make_kernel_render(self, _out_node):
        raise NotImplementedError("GatedCatSumCopyTemplateBuffer uses custom codegen")

    def get_outputs(self):
        return [self.copy_buffer, self.sum_buffer]

    def simplify_and_reorder(
        self,
        extra_indexing_constraints=None,
        recompute_sizes_body_func=None,
    ):
        return (([1], ()), None)

    def extract_read_writes(self, normalize):
        from torch._inductor import dependencies
        from torch.utils._ordered_set import OrderedSet

        reads = OrderedSet(dependencies.StarDep(inp.get_name()) for inp in self.inputs)
        writes = OrderedSet(
            dependencies.StarDep(buf.get_name()) for buf in self.get_outputs()
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )

    def codegen_template(self, wrapper) -> None:
        global _KERNEL_COUNTER

        kernel_name = f"inductor_gated_cat_sum_copy_{_KERNEL_COUNTER}"
        _KERNEL_COUNTER += 1

        match = self.match
        batch, seq, channels, hidden = (
            match.batch,
            match.seq,
            match.channels,
            match.hidden,
        )
        half_xnumel = channels * hidden
        rnumel = batch * seq
        hidden_i = _hinted_int(hidden)
        channels_i = _hinted_int(channels)
        use_fast_path = (
            hidden_i is not None
            and channels_i is not None
            and self._supports_split_channel_major_fast_path()
        )
        xblock = _FAST_XBLOCK if use_fast_path else _DEFAULT_XBLOCK
        rblock = _FAST_RBLOCK if use_fast_path else _DEFAULT_RBLOCK

        wrapper.write_triton_header_once()
        source = (
            _split_channel_major_kernel_source(
                kernel_name,
                self.get_device(),
                hidden=hidden_i,
                half_xnumel=channels_i * hidden_i,
                rblock=rblock,
                xblock=xblock,
            )
            if use_fast_path
            else _kernel_source(
                kernel_name,
                self.get_device(),
                rblock=rblock,
                xblock=xblock,
            )
        )
        wrapper.header.writeline(
            f"{kernel_name} = async_compile.triton({kernel_name!r}, '''{source}''', device_str='cuda')"
        )

        copy_ref = self.copy_buffer.codegen_reference()
        sum_ref = self.sum_buffer.codegen_reference()
        wrapper.writeline(
            f"{copy_ref} = empty_strided_cuda({_py_tuple(wrapper, self.copy_buffer.get_size())}, {_py_tuple(wrapper, self.copy_buffer.get_stride())}, torch.bfloat16)"
        )
        wrapper.writeline(
            f"{sum_ref} = empty_strided_cuda({_py_tuple(wrapper, self.sum_buffer.get_size())}, {_py_tuple(wrapper, self.sum_buffer.get_stride())}, torch.float32)"
        )
        wrapper.writeline(f"{sum_ref}.zero_()")

        copy_stride = self.copy_buffer.get_stride()
        sum_stride = self.sum_buffer.get_stride()
        sum_size = self.sum_buffer.get_size()
        if len(sum_size) == 5:
            sum_split = sum_size[4]
            sum_s4 = sum_stride[4]
        else:
            sum_split = 1
            sum_s4 = 0

        input_args = [
            match.base.buffer.codegen_reference(),
            match.value.buffer.codegen_reference(),
            match.gate.buffer.codegen_reference(),
            copy_ref,
            sum_ref,
            rnumel,
        ]
        copy_args = [copy_stride[0], copy_stride[1], copy_stride[2], copy_stride[3]]
        sum_args = [sum_split, sum_stride[2], sum_stride[3], sum_s4]
        grid_args = [
            rblock,
            xblock,
            _ceildiv_expr(half_xnumel, xblock),
            _ceildiv_expr(rnumel, rblock),
            1,
        ]

        if use_fast_path:
            call_args = [
                *input_args,
                match.base.stride[1],
                match.base.stride[2],
                match.value.stride[1],
                match.value.stride[2],
                match.gate.stride[1],
                match.gate.stride[2],
                copy_stride[1],
                copy_stride[2],
                *sum_args,
                hidden_i,
                channels_i * hidden_i,
                *grid_args,
            ]
        else:
            call_args = [
                *input_args,
                seq,
                hidden,
                half_xnumel,
                *match.base.stride,
                *match.value.stride,
                *match.gate.stride,
                *copy_args,
                *sum_args,
                *grid_args,
            ]
        wrapper.generate_kernel_call(kernel_name, call_args, device=self.get_device())

    def _supports_split_channel_major_fast_path(self) -> bool:
        match = self.match
        seq = match.seq
        base_stride = match.base.stride
        value_stride = match.value.stride
        gate_stride = match.gate.stride
        copy_stride = match.copy.node.get_stride()
        # Fast path works when batch and seq are already folded into the leading
        # stride relation, so the kernel can index with r = batch * seq + seq_idx
        # and keep the layout shape-generic.
        return (
            _same_expr(base_stride[0], seq * base_stride[1])
            and _same_expr(base_stride[3], 1)
            and _same_expr(value_stride[0], seq * value_stride[1])
            and _same_expr(value_stride[3], 1)
            and _same_expr(gate_stride[0], seq * gate_stride[1])
            and _same_expr(gate_stride[3], 1)
            and _same_expr(copy_stride[0], seq * copy_stride[1])
            and _same_expr(copy_stride[3], 1)
        )


def _make_scheduler_node(match: _GatedCatMatch):
    from torch._inductor.scheduler import SchedulerNode

    template = GatedCatSumCopyTemplateBuffer(match)
    native_node = SchedulerNode(match.cat.scheduler, template)
    native_node.min_order = min(
        match.cat.min_order, match.copy.min_order, match.sum.min_order
    )
    native_node.max_order = max(
        match.cat.max_order, match.copy.max_order, match.sum.max_order
    )
    native_node.ancestors = match.cat.ancestors | match.copy.ancestors | match.sum.ancestors
    for new_buf in native_node.get_outputs():
        old_buf = match.cat.scheduler.name_to_buf.get(new_buf.get_name())
        if old_buf is not None:
            new_buf.set_users(old_buf.users)
        match.cat.scheduler.name_to_buf[new_buf.get_name()] = new_buf
    return native_node


def fuse_gated_cat_sum_copy(nodes):
    """Fuse gated-SiLU cat feeding both sum and bf16 copy into one Triton kernel."""

    used = set()
    output = []
    replacements = {}
    matched = 0

    for node in nodes:
        if node in used:
            continue
        if not _is_cat_node(node):
            output.append(replacements.get(node, node))
            continue

        try:
            match = _match_cat_sum_copy(nodes, node)
        except Exception as exc:
            _log("skip after match error", cat=node.get_name(), error=exc)
            output.append(node)
            continue

        if match is None:
            output.append(node)
            continue

        native_node = _make_scheduler_node(match)
        for old in (match.cat, match.copy, match.sum):
            replacements[old] = native_node
            used.add(old)
            match.cat.scheduler.name_to_fused_node[old.get_name()] = native_node
        match.cat.scheduler.name_to_fused_node[native_node.get_name()] = native_node
        output.append(native_node)
        matched += 1
        _log(
            "fused gated cat/sum/copy",
            cat=match.cat.get_name(),
            copy=match.copy.get_name(),
            sum=match.sum.get_name(),
            batch=match.batch,
            seq=match.seq,
            channels=match.channels,
            hidden=match.hidden,
        )

    for node in nodes:
        if node in used:
            continue
        replacement = replacements.get(node)
        if replacement is not None and replacement not in output:
            output.append(replacement)
        elif node not in output:
            output.append(node)

    if matched:
        _log("summary", matched=matched)
    return output


def fuse_cat_copy_for_sum_copy_users(nodes):
    """Compatibility entrypoint used by the existing environment-variable hook."""

    return fuse_gated_cat_sum_copy(nodes)
