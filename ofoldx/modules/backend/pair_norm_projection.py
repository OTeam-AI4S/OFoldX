r"""
Backend-dispatched pair normalization and projection.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_backend

PairProjectionLayout = Literal["head_last", "head_major"]

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_oops_norm_projection = oops_kernel("norm_projection")


def pair_norm_projection(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    projection_weight: Tensor,
    eps: float = 1e-5,
    *,
    output_layout: PairProjectionLayout = "head_last",
    backend: str,
) -> Tensor:
    r"""
    Normalize pair features and project them to a small set of channels.

    ``pair`` has shape ``[..., Q, K, C]`` and ``projection_weight`` has shape
    ``[H, C]``. ``output_layout="head_last"`` returns ``[..., Q, K, H]``;
    ``"head_major"`` returns a contiguous ``[..., H, Q, K]`` tensor.

    The source-precision layout uses FP32 ``E[x**2] - E[x]**2`` statistics and
    affine arithmetic, explicitly rounds the normalized pair back to the input
    dtype, and then projects it. Other layouts use standard PyTorch LayerNorm
    semantics. OOps is an optional implementation of the same public equation.

    Args:
        pair: Pair representation ``[..., Q, K, C]``.
        norm_weight: LayerNorm scale ``[C]``.
        norm_bias: Optional LayerNorm shift ``[C]``.
        projection_weight: Bias-free projection weight ``[H, C]``.
        eps: Value added to the variance for numerical stability.
        output_layout: Position of the projected head dimension.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        Projected pair features in the requested layout.

    Raises:
        ValueError: If tensor shapes or ``output_layout`` are unsupported.
    """

    _check_shapes(pair, norm_weight, norm_bias, projection_weight)
    if output_layout not in ("head_last", "head_major"):
        raise ValueError("output_layout must be 'head_last' or 'head_major'")

    backend = resolve_tensor_backend(
        backend,
        tensors=(pair, norm_weight, norm_bias, projection_weight),
        supported=("oops", "eager"),
        public_default="eager",
    )
    if backend == "oops":
        return cast(
            Tensor,
            _oops_norm_projection(
                pair,
                norm_weight,
                norm_bias,
                projection_weight,
                eps,
                output_layout=output_layout,
            ),
        )

    normalized = (
        _source_precision_pair_norm(pair, norm_weight, norm_bias, eps)
        if _uses_source_precision(pair, norm_weight, norm_bias, projection_weight)
        else F.layer_norm(pair, (pair.shape[-1],), norm_weight, norm_bias, eps)
    )
    projected = F.linear(normalized, projection_weight)
    if output_layout == "head_last":
        return projected
    return projected.movedim(-1, -3).contiguous()


def _uses_source_precision(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    projection_weight: Tensor,
) -> bool:
    autocast_master_weight = (
        pair.dtype == torch.bfloat16
        and pair.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and projection_weight.dtype == torch.float32
    )
    return (
        pair.dtype in _HALF_DTYPES
        and norm_weight.dtype == torch.float32
        and (norm_bias is None or norm_bias.dtype == torch.float32)
        and (projection_weight.dtype == pair.dtype or autocast_master_weight)
    )


def _source_precision_pair_norm(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    eps: float,
) -> Tensor:
    pair_float = pair.float()
    mean = pair_float.mean(dim=-1, keepdim=True)
    variance = pair_float.square().mean(dim=-1, keepdim=True) - mean.square()
    normalized = norm_weight * torch.rsqrt(variance + eps) * (pair_float - mean)
    if norm_bias is not None:
        normalized = normalized + norm_bias
    return normalized.to(dtype=pair.dtype)


def _check_shapes(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    projection_weight: Tensor,
) -> None:
    if pair.dim() < 3:
        raise ValueError("pair must have shape (..., Q, K, C)")
    channels = pair.shape[-1]
    if norm_weight.shape != (channels,):
        raise ValueError(f"norm_weight must have shape {(channels,)}; got {tuple(norm_weight.shape)}")
    if norm_bias is not None and norm_bias.shape != (channels,):
        raise ValueError(f"norm_bias must have shape {(channels,)}; got {tuple(norm_bias.shape)}")
    if projection_weight.dim() != 2 or projection_weight.shape[1] != channels or projection_weight.shape[0] <= 0:
        raise ValueError(f"projection_weight must have shape (H, {channels}) with H > 0")
