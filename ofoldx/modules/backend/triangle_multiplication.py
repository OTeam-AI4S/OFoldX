r"""
Backend-dispatched gated triangle multiplication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from ._cueq import cueq_triangle_multiplication
from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_triangle_multiplication = cast(Callable[..., Tensor], oops_kernel("triangle_multiplication"))

_SUPPORTED = ("oops", "cueq", "apex", "deepspeed", "eager")


def triangle_multiplication(
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    outgoing: bool,
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    input_norm_eps: float,
    output_norm_eps: float,
    backend: str,
    left_value_bias: Tensor | None = None,
    left_gate_bias: Tensor | None = None,
    right_value_bias: Tensor | None = None,
    right_gate_bias: Tensor | None = None,
    output_bias: Tensor | None = None,
    output_gate_bias: Tensor | None = None,
) -> Tensor:
    r"""
    Apply the gated triangle multiplicative update to ``pair`` on the selected backend.

    Layer-norms the input, forms gated left/right factors ``(..., N, N, hidden_size)``, contracts them
    over the shared node (``ikh,jkh->ijh`` outgoing, else ``kih,kjh->ijh`` incoming), then norms, projects
    back to ``pair_size`` and gates by the normalized input. The ``"oops"`` backend delegates to the OOps-owned
    ``triangle_multiplication`` implementation; ``"eager"`` runs the equivalent pure-PyTorch composition.

    Args:
        pair: ``(..., N, N, pair_size)`` pair representation.
        pair_mask: Optional ``(..., N, N)`` mask; masked entries are zeroed in the factors before the
            contraction.
        outgoing: Contract over the outgoing edge (``ikh,jkh->ijh``) when ``True``, else the incoming
            edge (``kih,kjh->ijh``).
        input_norm_weight: Input LayerNorm weight.
        input_norm_bias: Input LayerNorm bias.
        left_value_weight: Left factor value-projection weight; ``hidden_size`` is
            ``left_value_weight.shape[0]``.
        left_gate_weight: Left factor gate-projection weight.
        right_value_weight: Right factor value-projection weight.
        right_gate_weight: Right factor gate-projection weight.
        output_norm_weight: Output LayerNorm weight applied to the contraction.
        output_norm_bias: Output LayerNorm bias applied to the contraction.
        output_weight: Projection weight mapping back to ``pair_size``.
        output_gate_weight: Gate-projection weight applied to the normalized input.
        input_norm_eps: LayerNorm epsilon for the pair input.
        output_norm_eps: LayerNorm epsilon for the contracted update.
        backend: Operator provider; see [ofoldx.modules.backend][].
        left_value_bias: Optional left value-projection bias.
        left_gate_bias: Optional left gate-projection bias.
        right_value_bias: Optional right value-projection bias.
        right_gate_bias: Optional right gate-projection bias.
        output_bias: Optional output-projection bias.
        output_gate_bias: Optional output gate-projection bias.

    Returns:
        ``(..., N, N, pair_size)`` pair update, matching the shape of ``pair``.
    """

    hidden_size = left_value_weight.shape[0]
    pair_size = pair.shape[-1]
    backend = resolve_tensor_backend(
        backend,
        tensors=(pair, pair_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_triangle_multiplication(
            pair,
            pair_mask,
            outgoing=outgoing,
            input_norm_weight=input_norm_weight,
            input_norm_bias=input_norm_bias,
            left_value_weight=left_value_weight,
            left_gate_weight=left_gate_weight,
            right_value_weight=right_value_weight,
            right_gate_weight=right_gate_weight,
            output_norm_weight=output_norm_weight,
            output_norm_bias=output_norm_bias,
            output_weight=output_weight,
            output_gate_weight=output_gate_weight,
            left_value_bias=left_value_bias,
            left_gate_bias=left_gate_bias,
            right_value_bias=right_value_bias,
            right_gate_bias=right_gate_bias,
            output_bias=output_bias,
            output_gate_bias=output_gate_bias,
            input_norm_eps=input_norm_eps,
            output_norm_eps=output_norm_eps,
        )
    if backend == "cueq" and pair.is_cuda and hidden_size == pair_size and input_norm_eps == output_norm_eps:
        input_bias = _concatenate_optional_biases(
            left_value_bias,
            right_value_bias,
            left_value_weight,
            right_value_weight,
        )
        input_gate_bias = _concatenate_optional_biases(
            left_gate_bias,
            right_gate_bias,
            left_gate_weight,
            right_gate_weight,
        )
        return cueq_triangle_multiplication(
            pair,
            direction="outgoing" if outgoing else "incoming",
            mask=pair_mask,
            input_norm_weight=input_norm_weight,
            input_norm_bias=input_norm_bias,
            input_weight=torch.cat((left_value_weight, right_value_weight)),
            input_bias=input_bias,
            input_gate_weight=torch.cat((left_gate_weight, right_gate_weight)),
            input_gate_bias=input_gate_bias,
            output_norm_weight=output_norm_weight,
            output_norm_bias=output_norm_bias,
            output_weight=output_weight,
            output_bias=output_bias,
            output_gate_weight=output_gate_weight,
            output_gate_bias=output_gate_bias,
            eps=input_norm_eps,
        )
    normalized = layer_norm(
        pair,
        (pair_size,),
        input_norm_weight,
        input_norm_bias,
        input_norm_eps,
        backend=backend,
    )
    left = torch.nn.functional.linear(normalized, left_value_weight, left_value_bias) * torch.sigmoid(
        torch.nn.functional.linear(normalized, left_gate_weight, left_gate_bias)
    )
    right = torch.nn.functional.linear(normalized, right_value_weight, right_value_bias) * torch.sigmoid(
        torch.nn.functional.linear(normalized, right_gate_weight, right_gate_bias)
    )
    left, right = _mask_triangle_factors(left, right, pair_mask)
    equation = "...ikh,...jkh->...ijh" if outgoing else "...kih,...kjh->...ijh"
    update = torch.einsum(equation, left, right)
    normed_update = layer_norm(
        update,
        (hidden_size,),
        output_norm_weight,
        output_norm_bias,
        output_norm_eps,
        backend=backend,
    )
    update = torch.nn.functional.linear(normed_update, output_weight, output_bias)
    return update * torch.sigmoid(torch.nn.functional.linear(normalized, output_gate_weight, output_gate_bias))


def _concatenate_optional_biases(
    first: Tensor | None,
    second: Tensor | None,
    first_weight: Tensor,
    second_weight: Tensor,
) -> Tensor | None:
    r"""
    Pack two optional projection biases for cuEquivariance's fused input projection.
    """

    if first is None and second is None:
        return None
    if first is None:
        first = first_weight.new_zeros(first_weight.shape[0])
    if second is None:
        second = second_weight.new_zeros(second_weight.shape[0])
    return torch.cat((first, second))


def _mask_triangle_factors(left: Tensor, right: Tensor, pair_mask: Tensor | None) -> tuple[Tensor, Tensor]:
    r"""
    Zero masked entries of the left/right factors before the triangle contraction.

    Args:
        left: ``(..., N, N, hidden_size)`` gated left factor.
        right: ``(..., N, N, hidden_size)`` gated right factor.
        pair_mask: Optional ``(..., N, N)`` mask; ``None`` returns the factors unchanged.

    Returns:
        The ``(left, right)`` pair with masked positions zeroed (or unchanged when ``pair_mask`` is
        ``None``).
    """
    if pair_mask is None:
        return left, right
    pair_mask = pair_mask[..., None].to(dtype=left.dtype)
    return left * pair_mask, right * pair_mask
