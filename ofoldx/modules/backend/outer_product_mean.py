# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Backend-dispatched outer-product mean.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from ..._oops import OOpsUnsupportedError, oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_outer_product_mean = cast(Callable[..., Tensor], oops_kernel("outer_product_mean"))

_SUPPORTED = ("oops", "eager")


def outer_product_mean(
    msa: Tensor,
    msa_mask: Tensor | None,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    msa_norm_eps: float,
    denominator_eps: float,
    divide_before_projection: bool,
    clamp_denominator: bool,
    backend: str,
    left_bias: Tensor | None = None,
    right_bias: Tensor | None = None,
) -> Tensor:
    r"""
    Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
    projections and project it to a pair update, on the selected backend.

    Args:
        msa: ``(..., S, N, msa_size)`` MSA representation.
        msa_mask: Optional mask broadcastable to ``(..., S, N)``.
        msa_norm_weight: LayerNorm weight for the pre-normalization.
        msa_norm_bias: LayerNorm bias for the pre-normalization.
        left_weight: Left projection weight.
        right_weight: Right projection weight.
        output_weight: Output projection weight.
        output_bias: Output projection bias.
        msa_norm_eps: Epsilon used by the pre-normalization.
        denominator_eps: Epsilon added to the per-pair count in the divide-after-projection branch.
        divide_before_projection: When ``True`` divide by ``count.clamp(min=1)`` before projecting,
            otherwise project then divide by ``count + denominator_eps``.
        clamp_denominator: Whether to clamp the post-projection denominator to at least one.
        backend: Operator provider; see [ofoldx.modules.backend][].
        left_bias: Optional left projection bias; ``None`` keeps the projection bias-free.
        right_bias: Optional right projection bias; ``None`` keeps the projection bias-free.

    Returns:
        ``(..., N, N, pair_size)`` pair update, where ``pair_size`` is the row count of ``output_weight``.
    """

    if msa_mask is not None:
        coefficients = msa_mask[..., None]
        keep = (
            coefficients
            if coefficients.dtype == torch.bool
            else ((coefficients != 0) | torch.isfinite(msa).all(-1, keepdim=True))
        )
        msa = torch.where(keep, msa, 0)

    resolved = resolve_tensor_backend(
        backend,
        tensors=(msa, msa_mask),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_outer_product_mean,
    )
    if resolved == "oops":
        prefix = msa.shape[:-3]
        flat_msa, flat_mask = msa, msa_mask
        dense = not isinstance(msa, NestedTensor) and msa.dim() >= 3 and msa.dim() != 4
        if dense:
            flat_batch = math.prod(prefix)
            flat_msa = msa.reshape(flat_batch, *msa.shape[-3:])
            if msa_mask is not None:
                flat_mask = torch.broadcast_to(msa_mask, msa.shape[:-1]).reshape(flat_batch, *msa.shape[-3:-1])
        elif (
            not isinstance(msa, NestedTensor)
            and msa.dim() == 4
            and msa_mask is not None
            and msa_mask.shape != msa.shape[:-1]
        ):
            flat_mask = torch.broadcast_to(msa_mask, msa.shape[:-1])
        try:
            result = _oops_outer_product_mean(
                flat_msa,
                flat_mask,
                msa_norm_weight=msa_norm_weight,
                msa_norm_bias=msa_norm_bias,
                left_weight=left_weight,
                right_weight=right_weight,
                output_weight=output_weight,
                left_bias=left_bias,
                right_bias=right_bias,
                output_bias=output_bias,
                msa_norm_eps=msa_norm_eps,
                denominator_eps=denominator_eps,
                divide_before_projection=divide_before_projection,
                clamp_denominator=clamp_denominator,
            )
        except OOpsUnsupportedError:
            if backend != "auto":
                raise
        else:
            return result.reshape(*prefix, msa.shape[-2], msa.shape[-2], result.shape[-1]) if dense else result

    msa = layer_norm(
        msa,
        (msa.shape[-1],),
        msa_norm_weight,
        msa_norm_bias,
        msa_norm_eps,
        backend=backend,
    )
    left = torch.nn.functional.linear(msa, left_weight, left_bias)
    right = torch.nn.functional.linear(msa, right_weight, right_bias)
    if msa_mask is None:
        msa_mask = msa.new_ones(msa.shape[:-1])
    msa_mask = msa_mask.to(dtype=msa.dtype)
    left = left * msa_mask[..., None]
    right = right * msa_mask[..., None]

    outer = torch.einsum("...sih,...sjk->...ijhk", left, right)
    outer = outer.reshape(*outer.shape[:-2], outer.shape[-2] * outer.shape[-1])
    count = torch.einsum("...si,...sj->...ij", msa_mask, msa_mask).unsqueeze(-1)
    if divide_before_projection:
        return torch.nn.functional.linear(outer / count.clamp(min=1), output_weight, output_bias)
    denominator = count + denominator_eps
    if clamp_denominator:
        denominator = denominator.clamp_min(1)
    return torch.nn.functional.linear(outer, output_weight, output_bias) / denominator
