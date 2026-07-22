r"""
Backend dispatch for the position-wise transition (SwiGLU) operator.

The dispatcher owns backend selection and calling-convention adaptation. Modules
pass the ordinary tensor arguments without a second validation layer.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_transition = oops_kernel("transition")

_SUPPORTED = ("oops", "eager")


def transition(
    hidden: Tensor,
    *,
    norm_weight: Tensor,
    norm_bias: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    eps: float,
    operator_backend: str,
) -> Tensor:
    r"""
    Apply the SwiGLU transition ``down(silu(gate(norm(x))) * up(norm(x)))`` on the selected backend.

    Pre-normalizes ``hidden`` with LayerNorm, gates the up-projection by the SiLU of the gate-projection,
    and projects back down. The ``"oops"`` backend delegates to the OOps-owned implementation; ``"eager"`` runs the
    equivalent pure-PyTorch composition.

    Args:
        hidden: ``[..., hidden_size]`` input activations.
        norm_weight: ``[hidden_size]`` LayerNorm scale for the pre-normalization.
        norm_bias: ``[hidden_size]`` LayerNorm shift for the pre-normalization.
        gate_weight: ``[intermediate_size, hidden_size]`` bias-free gate projection weight.
        up_weight: ``[intermediate_size, hidden_size]`` bias-free up projection weight.
        down_weight: ``[hidden_size, intermediate_size]`` bias-free down projection weight.
        eps: LayerNorm epsilon.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., hidden_size]`` transformed activations.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.transition import transition
        >>> _ = torch.manual_seed(0)
        >>> hidden = torch.randn(2, 4)
        >>> out = transition(
        ...     hidden,
        ...     norm_weight=torch.ones(4),
        ...     norm_bias=torch.zeros(4),
        ...     gate_weight=torch.randn(6, 4),
        ...     up_weight=torch.randn(6, 4),
        ...     down_weight=torch.randn(4, 6),
        ...     eps=1e-5,
        ...     operator_backend="eager",
        ... )
        >>> out.shape
        torch.Size([2, 4])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(hidden,),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_transition(
            hidden,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            eps=eps,
        )
    normed = F.layer_norm(hidden, (hidden.shape[-1],), norm_weight, norm_bias, eps)
    return F.linear(F.silu(F.linear(normed, gate_weight)) * F.linear(normed, up_weight), down_weight)
