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
Backend-dispatched pair normalization and projection.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from ..._oops import oops_kernel
from .normalization import layer_norm_linear
from .resolve import resolve_tensor_backend

PairProjectionLayout = Literal["head_last", "head_major"]

_oops_norm_projection = oops_kernel("norm_projection")

_SUPPORTED = ("oops", "eager")


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

    Both layouts normalize the channel axis with standard PyTorch LayerNorm
    semantics before the bias-free linear projection. OOps implements the same
    centered normalization and projection equation. Its fused specialization
    serves contiguous CUDA float16/bfloat16/float32 pairs with at most 256 pair
    channels and 32 projected channels. Other shapes use the canonical
    normalization/linear composition, including an explicit OOps preference.

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

    resolved = resolve_tensor_backend(
        backend,
        tensors=(pair, norm_weight, norm_bias, projection_weight),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_norm_projection,
    )
    fused = (
        pair.is_cuda
        and pair.shape[-1] <= 256
        and projection_weight.shape[0] <= 32
        and pair.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (
            projection_weight.dtype == pair.dtype
            or (
                torch.is_autocast_enabled("cuda")
                and projection_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
            )
        )
        and all(value is None or value.dtype in (pair.dtype, torch.float32) for value in (norm_weight, norm_bias))
        and (isinstance(pair, NestedTensor) or pair.is_contiguous())
        and all(value is None or value.is_contiguous() for value in (norm_weight, norm_bias, projection_weight))
    )
    if resolved == "oops" and fused:
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

    if "oops" in (backend, resolved):
        backend = "eager"
    projected = layer_norm_linear(
        pair, (pair.shape[-1],), norm_weight, norm_bias, projection_weight, eps=eps, backend=backend
    )
    if output_layout == "head_last":
        return projected
    return projected.movedim(-1, -3).contiguous()


def _check_shapes(
    pair: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    projection_weight: Tensor,
) -> None:
    if pair.dim() < 3:
        raise ValueError("pair must have shape (..., Q, K, C)")
    pair_size = pair.shape[-1]
    if norm_weight.shape != (pair_size,):
        raise ValueError(f"norm_weight must have shape {(pair_size,)}; got {tuple(norm_weight.shape)}")
    if norm_bias is not None and norm_bias.shape != (pair_size,):
        raise ValueError(f"norm_bias must have shape {(pair_size,)}; got {tuple(norm_bias.shape)}")
    if projection_weight.dim() != 2 or projection_weight.shape[1] != pair_size or projection_weight.shape[0] <= 0:
        raise ValueError(f"projection_weight must have shape (H, {pair_size}) with H > 0")
