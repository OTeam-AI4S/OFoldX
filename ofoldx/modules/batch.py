r"""Logical batch-axis tensor helpers."""

from __future__ import annotations

from danling.tensors import NestedTensor
from torch import Tensor


def repeat_batch(input: Tensor, repeats: int) -> Tensor:
    r"""Repeat complete logical batch elements in interleaved order.

    Dense tensors retain standard ``repeat_interleave(..., dim=0)`` semantics.
    DanLing ``NestedTensor`` inputs repeat packed sample segments and their
    tensor-backed topology without padding or per-element Python dispatch.
    """
    if repeats == 1:
        return input
    if isinstance(input, NestedTensor):
        return input.repeat_batch(repeats)
    return input.repeat_interleave(repeats, dim=0)
