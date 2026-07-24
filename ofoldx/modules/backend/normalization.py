r"""
Backend dispatch for LayerNorm.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._apex import apex_layer_norm
from ._deepspeed import deepspeed_layer_norm
from ._oops import oops_kernel
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
        A normalized tensor with the same shape and dtype as ``input``.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(input, weight, bias),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return cast(
            Tensor,
            _oops_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps),
        )
    if backend == "apex" and input.is_cuda:
        output = apex_layer_norm(input, normalized_shape, weight, bias, eps)
        if output is not None:
            return output
    needs_grad = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (input, weight, bias)
    )
    if backend == "deepspeed" and input.is_cuda and len(normalized_shape) == 1 and not needs_grad:
        return deepspeed_layer_norm(input, weight, bias, eps)
    return F.layer_norm(input, tuple(normalized_shape), weight, bias, eps)


class LayerNorm(nn.LayerNorm):
    r"""
    Layer normalization with a selectable backend.

    Parameters and numerical semantics match [`torch.nn.LayerNorm`][torch.nn.LayerNorm].

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
            A normalized tensor with the same shape and dtype as ``input``.
        """

        return layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            backend=self.backend,
        )
