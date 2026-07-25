r"""
Backend-dispatched outer-product mean.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_outer_product_mean = oops_kernel("outer_product_mean")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")


def outer_product_mean(
    msa: Tensor,
    msa_mask: Tensor | None,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    proj_left_weight: Tensor,
    proj_right_weight: Tensor,
    proj_out_weight: Tensor,
    proj_out_bias: Tensor,
    msa_norm_eps: float,
    denom_eps: float,
    divide_before_projection: bool,
    clamp_denominator: bool,
    backend: str,
    proj_left_bias: Tensor | None = None,
    proj_right_bias: Tensor | None = None,
) -> Tensor:
    r"""
    Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
    projections and project it to a pair update, on the selected backend.

    Args:
        msa: ``(B, S, N, msa_size)`` MSA representation.
        msa_mask: Optional ``(B, S, N)`` mask zeroing invalid rows and counting valid pairs.
        msa_norm_weight: LayerNorm weight for the pre-normalization.
        msa_norm_bias: LayerNorm bias for the pre-normalization.
        proj_left_weight: Left projection weight.
        proj_right_weight: Right projection weight.
        proj_out_weight: Output projection weight.
        proj_out_bias: Output projection bias.
        msa_norm_eps: Epsilon used by the pre-normalization.
        denom_eps: Epsilon added to the per-pair count in the divide-after-projection branch.
        divide_before_projection: When ``True`` divide by ``count.clamp(min=1)`` before projecting,
            otherwise project then divide by ``count + denom_eps``.
        clamp_denominator: Whether to clamp the post-projection denominator to at least one.
        backend: Operator provider; see [ofoldx.modules.backend][].
        proj_left_bias: Optional left projection bias; ``None`` keeps the projection bias-free.
        proj_right_bias: Optional right projection bias; ``None`` keeps the projection bias-free.

    Returns:
        ``(B, N, N, pair_size)`` pair update, where ``pair_size`` is the row count of ``proj_out_weight``.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(msa, msa_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_outer_product_mean(
            msa,
            msa_mask,
            msa_norm_weight=msa_norm_weight,
            msa_norm_bias=msa_norm_bias,
            proj_left_weight=proj_left_weight,
            proj_right_weight=proj_right_weight,
            proj_out_weight=proj_out_weight,
            proj_left_bias=proj_left_bias,
            proj_right_bias=proj_right_bias,
            proj_out_bias=proj_out_bias,
            msa_norm_eps=msa_norm_eps,
            denom_eps=denom_eps,
            divide_before_projection=divide_before_projection,
            clamp_denominator=clamp_denominator,
        )

    msa = layer_norm(
        msa,
        (msa.shape[-1],),
        msa_norm_weight,
        msa_norm_bias,
        msa_norm_eps,
        backend=backend,
    )
    left = torch.nn.functional.linear(msa, proj_left_weight, proj_left_bias)
    right = torch.nn.functional.linear(msa, proj_right_weight, proj_right_bias)
    if msa_mask is None:
        msa_mask = msa.new_ones(msa.shape[:-1])
    msa_mask = msa_mask.to(dtype=msa.dtype)
    left = left * msa_mask[..., None]
    right = right * msa_mask[..., None]

    outer = torch.einsum("...sih,...sjk->...ijhk", left, right)
    outer = outer.reshape(*outer.shape[:-2], outer.shape[-2] * outer.shape[-1])
    count = torch.einsum("...si,...sj->...ij", msa_mask, msa_mask).unsqueeze(-1)
    if divide_before_projection:
        return torch.nn.functional.linear(outer / count.clamp(min=1), proj_out_weight, proj_out_bias)
    denominator = count + denom_eps
    if clamp_denominator:
        denominator = denominator.clamp_min(1)
    return torch.nn.functional.linear(outer, proj_out_weight, proj_out_bias) / denominator
