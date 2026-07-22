r"""
Backend dispatch for the adaptive (conditioned) layer-norm operator.

The dispatcher owns backend selection and calling-convention adaptation. Modules pass the ordinary
tensor arguments -- the fused ``conditioning_projection`` weight ``[2 * hidden_size, conditioning_size]``
and the ``[hidden_size]`` ``conditioning_scale_bias`` -- without a second validation layer. The
``"oops"`` backend delegates to the OOps-owned implementation; ``"eager"`` runs the
equivalent pure-PyTorch composition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_adaptive_layer_norm = oops_kernel("adaptive_layer_norm")

_SUPPORTED = ("oops", "eager")


def adaptive_layer_norm(
    hidden: Tensor,
    conditioning: Tensor,
    *,
    conditioning_norm_weight: Tensor,
    projection_weight: Tensor,
    scale_bias: Tensor,
    eps: float,
    operator_backend: str,
) -> Tensor:
    r"""
    Apply adaptive layer norm ``sigmoid(scale) * norm(hidden) + shift`` on the selected backend.

    ``hidden`` is layer-normed affine-free; ``conditioning`` is layer-normed with
    ``conditioning_norm_weight`` (no bias). The normed conditioning is projected once through the fused
    ``projection_weight`` and split into the raw ``scale`` and ``shift`` halves; ``scale_bias`` is added
    to the scale half before the sigmoid gate. The ``"oops"`` backend delegates to OOps dispatch;
    ``"eager"`` runs the equivalent pure-PyTorch composition.

    Args:
        hidden: ``[..., hidden_size]`` input activations.
        conditioning: ``[..., conditioning_size]`` conditioning activations.
        conditioning_norm_weight: ``[conditioning_size]`` LayerNorm scale for the conditioning norm.
        projection_weight: ``[2 * hidden_size, conditioning_size]`` fused bias-free scale/shift projection
            weight; the first ``hidden_size`` rows produce the scale, the remainder the shift.
        scale_bias: ``[hidden_size]`` additive bias applied to the scale half before the sigmoid.
        eps: LayerNorm epsilon shared by both norms.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., hidden_size]`` conditioned activations.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.adaptive_norm import adaptive_layer_norm
        >>> _ = torch.manual_seed(0)
        >>> hidden = torch.randn(2, 8)
        >>> conditioning = torch.randn(2, 4)
        >>> out = adaptive_layer_norm(
        ...     hidden,
        ...     conditioning,
        ...     conditioning_norm_weight=torch.ones(4),
        ...     projection_weight=torch.randn(16, 4),
        ...     scale_bias=torch.zeros(8),
        ...     eps=1e-5,
        ...     operator_backend="eager",
        ... )
        >>> out.shape
        torch.Size([2, 8])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(hidden, conditioning),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_adaptive_layer_norm(
            hidden,
            conditioning,
            projection_weight,
            scale_bias,
            conditioning_norm_weight=conditioning_norm_weight,
            eps=eps,
            conditioning_eps=eps,
        )
    hidden_normed = F.layer_norm(hidden, (hidden.shape[-1],), None, None, eps)
    conditioning_normed = F.layer_norm(conditioning, (conditioning.shape[-1],), conditioning_norm_weight, None, eps)
    scale, shift = F.linear(conditioning_normed, projection_weight).chunk(2, dim=-1)
    return torch.sigmoid(scale + scale_bias) * hidden_normed + shift
