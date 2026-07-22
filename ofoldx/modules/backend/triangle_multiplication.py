r"""
Backend dispatch for the gated triangle multiplicative update.

The dispatcher owns backend selection and calling-convention adaptation.
[`TriangleMultiplication`][] passes the ordinary tensor arguments without a
second validation layer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_triangle_multiplication = oops_kernel("triangle_multiplication")

_SUPPORTED = ("oops", "eager")


def triangle_multiplication(
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    outgoing: bool,
    norm_input_weight: Tensor,
    norm_input_bias: Tensor,
    left_value_weight: Tensor,
    left_gate_weight: Tensor,
    right_value_weight: Tensor,
    right_gate_weight: Tensor,
    norm_output_weight: Tensor,
    norm_output_bias: Tensor,
    output_weight: Tensor,
    output_gate_weight: Tensor,
    eps: float,
    operator_backend: str,
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
        norm_input_weight, norm_input_bias: Input LayerNorm parameters.
        left_value_weight, left_gate_weight: Bias-free left factor value/gate projection weights;
            ``hidden_size`` is ``left_value_weight.shape[0]``.
        right_value_weight, right_gate_weight: Bias-free right factor value/gate projection weights.
        norm_output_weight, norm_output_bias: Output LayerNorm parameters applied to the contraction.
        output_weight: Bias-free projection back to ``pair_size``.
        output_gate_weight: Bias-free gate projection applied to the normalized input.
        eps: LayerNorm epsilon shared by the input and output normalizations.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.
        left_value_bias, left_gate_bias, right_value_bias, right_gate_bias, output_bias,
            output_gate_bias: Optional per-projection biases (AlphaFold 2 lineage). Default ``None`` ⇒
            the projection is applied bias-free, byte-identical to the AlphaFold 3 lineage.

    Returns:
        ``(..., N, N, pair_size)`` pair update, matching the shape of ``pair``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.triangle_multiplication import triangle_multiplication
        >>> _ = torch.manual_seed(0)
        >>> pair_size, hidden_size = 4, 5
        >>> pair = torch.randn(1, 3, 3, pair_size)
        >>> out = triangle_multiplication(
        ...     pair,
        ...     None,
        ...     outgoing=True,
        ...     norm_input_weight=torch.ones(pair_size),
        ...     norm_input_bias=torch.zeros(pair_size),
        ...     left_value_weight=torch.randn(hidden_size, pair_size),
        ...     left_gate_weight=torch.randn(hidden_size, pair_size),
        ...     right_value_weight=torch.randn(hidden_size, pair_size),
        ...     right_gate_weight=torch.randn(hidden_size, pair_size),
        ...     norm_output_weight=torch.ones(hidden_size),
        ...     norm_output_bias=torch.zeros(hidden_size),
        ...     output_weight=torch.randn(pair_size, hidden_size),
        ...     output_gate_weight=torch.randn(pair_size, pair_size),
        ...     eps=1e-5,
        ...     operator_backend="eager",
        ... )
        >>> out.shape
        torch.Size([1, 3, 3, 4])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(pair, pair_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_triangle_multiplication(
            pair,
            pair_mask,
            outgoing=outgoing,
            norm_input_weight=norm_input_weight,
            norm_input_bias=norm_input_bias,
            left_value_weight=left_value_weight,
            left_gate_weight=left_gate_weight,
            right_value_weight=right_value_weight,
            right_gate_weight=right_gate_weight,
            norm_output_weight=norm_output_weight,
            norm_output_bias=norm_output_bias,
            output_weight=output_weight,
            output_gate_weight=output_gate_weight,
            left_value_bias=left_value_bias,
            left_gate_bias=left_gate_bias,
            right_value_bias=right_value_bias,
            right_gate_bias=right_gate_bias,
            output_bias=output_bias,
            output_gate_bias=output_gate_bias,
            eps=eps,
        )
    hidden_size = left_value_weight.shape[0]
    pair_size = pair.shape[-1]
    normalized = torch.nn.functional.layer_norm(pair, (pair_size,), norm_input_weight, norm_input_bias, eps)
    left = torch.nn.functional.linear(normalized, left_value_weight, left_value_bias) * torch.sigmoid(
        torch.nn.functional.linear(normalized, left_gate_weight, left_gate_bias)
    )
    right = torch.nn.functional.linear(normalized, right_value_weight, right_value_bias) * torch.sigmoid(
        torch.nn.functional.linear(normalized, right_gate_weight, right_gate_bias)
    )
    left, right = _mask_triangle_factors(left, right, pair_mask)
    equation = "...ikh,...jkh->...ijh" if outgoing else "...kih,...kjh->...ijh"
    update = torch.einsum(equation, left, right)
    normed_update = torch.nn.functional.layer_norm(update, (hidden_size,), norm_output_weight, norm_output_bias, eps)
    update = torch.nn.functional.linear(normed_update, output_weight, output_bias)
    return update * torch.sigmoid(torch.nn.functional.linear(normalized, output_gate_weight, output_gate_bias))


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
