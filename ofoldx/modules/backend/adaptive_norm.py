r"""
Backend-dispatched adaptive layer normalization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_operator_backend

_oops_adaptive_layer_norm = oops_kernel("adaptive_layer_norm")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")


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
        operator_backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., hidden_size]`` conditioned activations.
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
    hidden_normed = layer_norm(hidden, (hidden.shape[-1],), None, None, eps, operator_backend=backend)
    conditioning_normed = layer_norm(
        conditioning,
        (conditioning.shape[-1],),
        conditioning_norm_weight,
        None,
        eps,
        operator_backend=backend,
    )
    scale, shift = F.linear(conditioning_normed, projection_weight).chunk(2, dim=-1)
    return torch.sigmoid(scale + scale_bias) * hidden_normed + shift
