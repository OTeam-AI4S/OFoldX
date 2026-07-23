r"""
Backend dispatch for the outer-product-mean operator.

The dispatcher owns backend selection, availability, and calling-convention
adaptation. Modules pass the ordinary tensor arguments without a second
validation layer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_outer_product_mean = oops_kernel("outer_product_mean")

_SUPPORTED = ("oops", "eager")


def outer_product_mean(
    msa: Tensor,
    msa_mask: Tensor | None,
    *,
    norm_weight: Tensor,
    norm_bias: Tensor,
    proj_left_weight: Tensor,
    proj_right_weight: Tensor,
    proj_out_weight: Tensor,
    proj_out_bias: Tensor,
    norm_eps: float,
    denom_eps: float,
    divide_before_projection: bool,
    operator_backend: str,
    proj_left_bias: Tensor | None = None,
    proj_right_bias: Tensor | None = None,
) -> Tensor:
    r"""
    Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
    projections and project it to a pair update, on the selected backend.

    Args:
        msa: ``(B, S, N, msa_size)`` MSA representation.
        msa_mask: Optional ``(B, S, N)`` mask zeroing invalid rows and counting valid pairs.
        norm_weight: LayerNorm weight for the pre-normalization.
        norm_bias: LayerNorm bias for the pre-normalization.
        proj_left_weight: Left projection weight.
        proj_right_weight: Right projection weight.
        proj_out_weight: Output projection weight.
        proj_out_bias: Output projection bias.
        norm_eps: Epsilon used by the pre-normalization.
        denom_eps: Epsilon added to the per-pair count in the divide-after-projection branch.
        divide_before_projection: When ``True`` divide by ``count.clamp(min=1)`` before projecting,
            otherwise project then divide by ``count + denom_eps``.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.
        proj_left_bias: Optional left projection bias; ``None`` keeps the projection bias-free.
        proj_right_bias: Optional right projection bias; ``None`` keeps the projection bias-free.

    Returns:
        ``(B, N, N, pair_size)`` pair update, where ``pair_size`` is the row count of ``proj_out_weight``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.outer_product_mean import outer_product_mean
        >>> _ = torch.manual_seed(0)
        >>> msa = torch.randn(1, 2, 3, 4)
        >>> out = outer_product_mean(
        ...     msa,
        ...     None,
        ...     norm_weight=torch.ones(4),
        ...     norm_bias=torch.zeros(4),
        ...     proj_left_weight=torch.randn(2, 4),
        ...     proj_right_weight=torch.randn(2, 4),
        ...     proj_out_weight=torch.randn(5, 4),
        ...     proj_out_bias=torch.zeros(5),
        ...     norm_eps=1e-5,
        ...     denom_eps=1e-3,
        ...     divide_before_projection=False,
        ...     operator_backend="eager",
        ... )
        >>> out.shape
        torch.Size([1, 3, 3, 5])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(msa, msa_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_outer_product_mean(
            msa,
            msa_mask,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            proj_left_weight=proj_left_weight,
            proj_right_weight=proj_right_weight,
            proj_out_weight=proj_out_weight,
            proj_left_bias=proj_left_bias,
            proj_right_bias=proj_right_bias,
            proj_out_bias=proj_out_bias,
            eps=norm_eps,
            denom_eps=denom_eps,
            divide_before_projection=divide_before_projection,
        )

    msa = torch.nn.functional.layer_norm(msa, (msa.shape[-1],), norm_weight, norm_bias, norm_eps)
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
    return torch.nn.functional.linear(outer, proj_out_weight, proj_out_bias) / (count + denom_eps)
