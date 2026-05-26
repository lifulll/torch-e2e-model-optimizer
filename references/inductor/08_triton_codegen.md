# 08 Triton Codegen

![mindmap](images/08_triton.png)

The Triton path generates GPU kernels for pointwise, reduction, and other general fused groups when templates or external libraries are not selected.

## What Triton Codegen Receives

It receives a scheduled node or fused node with loop ranges, indexing expressions, buffer layouts, dtype information, masks, reduction axes, and operation bodies. It emits Triton source and metadata for compilation and launch.

## Core Concepts

- Iteration space: symbolic loop ranges mapped to program ids and block dimensions.
- Indexing and masks: generated from layout, strides, dynamic shapes, and boundary conditions.
- CSE and dtype handling: reduce repeated expressions and control casts/accumulation.
- Reduction: more complex than pointwise because accumulation order, block reductions, and cooperative reduction may be involved.

## Kernel Quality Checklist

Check grid shape, block sizes, num warps, num stages, coalescing, masks, vectorization, register pressure, spills, occupancy, achieved bandwidth/FLOPs, and whether indexing is too complex.

## Template Boundary

Not all Triton kernels are produced by generic Triton codegen. Matmul, convolution, attention, and other structured paths may use template generation and autotuning, producing Triton through a different path.

## Optimization Intuition

Influence Triton generation through model code, lowering, layout, and scheduler first. Direct custom kernels are later-stage work and should be justified by stable E2E profiler evidence.
