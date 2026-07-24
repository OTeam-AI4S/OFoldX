r"""
Backend-dispatched SwiGLU transition.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_transition = oops_kernel("transition")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")


def transition(
    hidden: Tensor,
    *,
    norm_weight: Tensor,
    norm_bias: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    eps: float,
    backend: str,
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
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., hidden_size]`` transformed activations.
    """

    backend = resolve_tensor_backend(
        backend,
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
    normed = layer_norm(hidden, (hidden.shape[-1],), norm_weight, norm_bias, eps, backend=backend)
    return F.linear(F.silu(F.linear(normed, gate_weight)) * F.linear(normed, up_weight), down_weight)
