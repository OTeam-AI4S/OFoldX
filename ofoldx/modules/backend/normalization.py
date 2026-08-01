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
Backend dispatch for LayerNorm.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..._oops import oops_kernel
from ._apex import apex_layer_norm
from ._deepspeed import deepspeed_layer_norm
from .resolve import resolve_config_backend, resolve_tensor_backend

_oops_layer_norm = oops_kernel("layer_norm")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")


def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    *,
    backend: str,
) -> Tensor:
    r"""
    Apply layer normalization with a selectable backend.

    Args:
        input: Tensor whose trailing dimensions are normalized.
        normalized_shape: Sizes of the trailing dimensions to normalize.
        weight: Optional affine scale with shape ``normalized_shape``.
        bias: Optional affine bias with shape ``normalized_shape``.
        eps: Value added to the variance for numerical stability.
        backend: Backend used for normalization.

    Returns:
        A normalized tensor with the same shape as ``input``.
    """

    automatic = backend == "auto"
    supported: tuple[str, ...] = _SUPPORTED
    if automatic:
        if input.dtype == torch.float64:
            supported = ("eager",)
        elif isinstance(input, NestedTensor):
            supported = ("oops", "eager")
        # CUDA autocast promotes lower-precision LayerNorm operands to float32,
        # while OOps returns the input dtype. Use native dispatch when needed.
        if (
            input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and torch.is_autocast_enabled("cuda")
            and (
                input.dtype in (torch.float16, torch.bfloat16)
                or any(value is not None and value.dtype != torch.float32 for value in (weight, bias))
            )
        ):
            supported = ("eager",)
    backend = resolve_tensor_backend(
        backend,
        tensors=(input, weight, bias),
        supported=supported,
        public_default="eager",
        oops=_oops_layer_norm,
    )
    if backend == "oops":
        return cast(Tensor, _oops_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps))
    if backend == "apex" and input.is_cuda:
        output = apex_layer_norm(input, normalized_shape, weight, bias, eps)
        if output is not None:
            return output
        if automatic:
            backend = resolve_tensor_backend(
                "auto", tensors=(input,), supported=("deepspeed", "eager"), public_default="eager"
            )
    needs_grad = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (input, weight, bias)
    )
    if backend == "deepspeed" and input.is_cuda and len(normalized_shape) == 1 and not needs_grad:
        return deepspeed_layer_norm(input, weight, bias, eps)
    return F.layer_norm(input, tuple(normalized_shape), weight, bias, eps)


def layer_norm_linear(
    input: Tensor,
    normalized_shape: Sequence[int],
    norm_weight: Tensor | None,
    norm_bias: Tensor | None,
    projection_weight: Tensor,
    projection_bias: Tensor | None = None,
    eps: float = 1e-5,
    *,
    backend: str,
) -> Tensor:
    r"""Apply centered LayerNorm followed by a linear projection on the last axis."""

    normalized = layer_norm(input, normalized_shape, norm_weight, norm_bias, eps, backend=backend)
    dtype = normalized.dtype
    for coefficient in (projection_weight, projection_bias):
        if coefficient is not None:
            dtype = torch.promote_types(dtype, coefficient.dtype)
    return F.linear(
        normalized.to(dtype=dtype),
        projection_weight.to(dtype=dtype),
        projection_bias.to(dtype=dtype) if projection_bias is not None else None,
    )


def layer_norm_projection(input: Tensor, *, norm: LayerNorm, projection: nn.Linear) -> Tensor:
    r"""Apply native LayerNorm followed by a projection with promoted affine operands."""

    norm_weight = norm.weight
    norm_bias = norm.bias
    return layer_norm_linear(
        input,
        norm.normalized_shape,
        norm_weight.to(dtype=torch.promote_types(input.dtype, norm_weight.dtype)) if norm_weight is not None else None,
        norm_bias.to(dtype=torch.promote_types(input.dtype, norm_bias.dtype)) if norm_bias is not None else None,
        projection.weight,
        projection.bias,
        norm.eps,
        backend=norm.backend,
    )


class LayerNorm(nn.LayerNorm):
    r"""
    Layer normalization with a selectable backend.

    Parameter layout and normalized dimensions match [`torch.nn.LayerNorm`][torch.nn.LayerNorm].

    Args:
        normalized_shape: Sizes of the trailing dimensions to normalize.
        eps: Value added to the variance for numerical stability.
        elementwise_affine: Whether to learn per-element affine parameters.
        bias: Whether to learn an additive bias when affine parameters are enabled.
        device: Device of newly created parameters.
        dtype: Dtype of newly created parameters.
        backend: Backend used for normalization.
    """

    backend: str

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.backend = resolve_config_backend(
            backend,
            supported=_SUPPORTED,
            public_default="eager",
        )

    def forward(self, input: Tensor) -> Tensor:
        r"""
        Normalize the configured trailing dimensions.

        Args:
            input: Tensor whose trailing dimensions match ``normalized_shape``.

        Returns:
            A normalized tensor with the same shape as ``input``.
        """

        return layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            backend=self.backend,
        )
