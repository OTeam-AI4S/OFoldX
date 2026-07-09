r"""
Backend dispatch for the outer-product-mean operator.

The dispatcher owns everything backend-specific for this operator in one place: the supported
backends, ``"auto"`` resolution, the CUDA guard, and each backend's calling convention. Modules pass
their gathered parameters plus the requested ``operator_backend`` and stay backend-agnostic. New
backends (cuEquivariance / NVIDIA) are added as a branch here, not in every caller.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import OOpsUnsupportedError, oops_kernel
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
) -> Tensor:
    r"""
    Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
    projections and project it to a pair update, on the selected backend.

    Args:
        msa: ``(B, S, N, msa_size)`` MSA representation.
        msa_mask: Optional ``(B, S, N)`` mask zeroing invalid rows and counting valid pairs.
        norm_weight, norm_bias, norm_eps: LayerNorm parameters for the pre-normalization.
        proj_left_weight, proj_right_weight: Bias-free left/right projection weights.
        proj_out_weight, proj_out_bias: Output projection (affine).
        denom_eps: Epsilon added to the per-pair count in the divide-after-projection branch.
        divide_before_projection: When ``True`` divide by ``count.clamp(min=1)`` before projecting,
            otherwise project then divide by ``count + denom_eps``.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``(B, N, N, pair_size)`` pair update, where ``pair_size`` is the row count of ``proj_out_weight``.

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

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
        if not msa.is_cuda:
            raise OOpsUnsupportedError(
                "OuterProductMean backend 'oops' requires CUDA tensors; use operator_backend='eager'"
            )
        if msa_mask is not None and not msa_mask.is_cuda:
            raise OOpsUnsupportedError(
                "OuterProductMean backend 'oops' requires a CUDA msa_mask; use operator_backend='eager'"
            )
        return _oops_outer_product_mean(
            msa,
            msa_mask,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            proj_left_weight=proj_left_weight,
            proj_right_weight=proj_right_weight,
            proj_out_weight=proj_out_weight,
            proj_out_bias=proj_out_bias,
            eps=norm_eps,
            denom_eps=denom_eps,
            divide_before_projection=divide_before_projection,
        )

    msa = torch.nn.functional.layer_norm(msa, (msa.shape[-1],), norm_weight, norm_bias, norm_eps)
    left = torch.nn.functional.linear(msa, proj_left_weight)
    right = torch.nn.functional.linear(msa, proj_right_weight)
    if msa_mask is None:
        msa_mask = msa.new_ones(msa.shape[:-1])
    msa_mask = msa_mask.to(dtype=msa.dtype, device=msa.device)
    left = left * msa_mask[..., None]
    right = right * msa_mask[..., None]

    outer = torch.einsum("...sih,...sjk->...ijhk", left, right)
    outer = outer.reshape(*outer.shape[:-2], outer.shape[-2] * outer.shape[-1])
    count = torch.einsum("...si,...sj->...ij", msa_mask, msa_mask).unsqueeze(-1)
    if divide_before_projection:
        return torch.nn.functional.linear(outer / count.clamp(min=1), proj_out_weight, proj_out_bias)
    return torch.nn.functional.linear(outer, proj_out_weight, proj_out_bias) / (count + denom_eps)
