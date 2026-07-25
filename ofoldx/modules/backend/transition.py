r"""
Backend-dispatched SwiGLU transition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_transition = oops_kernel("transition")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")
_HALF_DTYPES = (torch.float16, torch.bfloat16)


def transition(
    hidden: Tensor,
    *,
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    input_norm_eps: float,
    backend: str,
) -> Tensor:
    r"""
    Apply the SwiGLU transition ``down(silu(gate(norm(x))) * up(norm(x)))`` on the selected backend.

    Pre-normalizes ``hidden`` with LayerNorm, gates the up-projection by the SiLU
    of the gate-projection, and projects back down. The released mixed-precision
    layout (half activations/projections and float32 normalization parameters)
    keeps normalization statistics and the gated product in float32. The
    ``"oops"`` backend delegates to the OOps-owned implementation; other backends
    retain the same equation in PyTorch.

    Args:
        hidden: ``[..., hidden_size]`` input activations.
        input_norm_weight: ``[hidden_size]`` LayerNorm scale for the pre-normalization.
        input_norm_bias: ``[hidden_size]`` LayerNorm shift for the pre-normalization.
        gate_weight: ``[intermediate_size, hidden_size]`` bias-free gate projection weight.
        up_weight: ``[intermediate_size, hidden_size]`` bias-free up projection weight.
        down_weight: ``[hidden_size, intermediate_size]`` bias-free down projection weight.
        input_norm_eps: LayerNorm epsilon.
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
            input_norm_weight=input_norm_weight,
            input_norm_bias=input_norm_bias,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            input_norm_eps=input_norm_eps,
        )
    if (
        hidden.dtype in _HALF_DTYPES
        and input_norm_weight.dtype == torch.float32
        and input_norm_bias.dtype == torch.float32
        and gate_weight.dtype == hidden.dtype
        and up_weight.dtype == hidden.dtype
        and down_weight.dtype == hidden.dtype
    ):
        output_dtype = hidden.dtype
        normalized = hidden.float()
        mean = normalized.mean(dim=-1, keepdim=True)
        variance = normalized.square().mean(dim=-1, keepdim=True) - mean.square()
        inverse_scale = input_norm_weight * torch.rsqrt(variance + input_norm_eps)
        normalized = (inverse_scale * (normalized - mean) + input_norm_bias).to(dtype=output_dtype)
        gate = F.linear(normalized, gate_weight)
        value = F.linear(normalized, up_weight)
        activated = (F.silu(gate.float()) * value.float()).to(dtype=output_dtype)
        return F.linear(activated, down_weight)
    normed = layer_norm(
        hidden,
        (hidden.shape[-1],),
        input_norm_weight,
        input_norm_bias,
        input_norm_eps,
        backend=backend,
    )
    return F.linear(F.silu(F.linear(normed, gate_weight)) * F.linear(normed, up_weight), down_weight)
