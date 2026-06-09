r"""
Backend dispatch for the gated triangle multiplicative update.

The dispatcher owns everything backend-specific for this operator in one place: the supported
backends, ``"auto"`` resolution, and each backend's calling convention. [`TriangleMultiplication`][]
gathers its parameters and passes them plus its requested ``operator_backend``, staying backend-agnostic.
New backends (cuEquivariance / NVIDIA) are added as a branch here, not in the module.
"""

from __future__ import annotations

import torch
from oops import OOpsUnsupportedError
from oops import triangle_multiplication as _oops_triangle_multiplication
from torch import Tensor

from .resolve import resolve_tensor_operator_backend

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
) -> Tensor:
    r"""
    Apply the gated triangle multiplicative update to ``pair`` on the selected backend.

    Layer-norms the input, forms gated left/right factors ``(..., N, N, hidden_size)``, contracts them
    over the shared node (``ikh,jkh->ijh`` outgoing, else ``kih,kjh->ijh`` incoming), then norms, projects
    back to ``pair_size`` and gates by the normalized input. The ``"oops"`` backend delegates to the fused
    ``triangle_multiplication`` kernel; ``"eager"`` runs the equivalent pure-PyTorch composition.

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

    Returns:
        ``(..., N, N, pair_size)`` pair update, matching the shape of ``pair``.

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA ``pair``/``pair_mask``.

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
        if not pair.is_cuda:
            raise OOpsUnsupportedError(
                "TriangleMultiplication backend 'oops' requires CUDA tensors; use operator_backend='sdpa'"
            )
        if pair_mask is not None and not pair_mask.is_cuda:
            raise OOpsUnsupportedError(
                "TriangleMultiplication backend 'oops' requires a CUDA pair_mask; use operator_backend='sdpa'"
            )
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
            eps=eps,
        )
    hidden_size = left_value_weight.shape[0]
    pair_size = pair.shape[-1]
    normalized = torch.nn.functional.layer_norm(pair, (pair_size,), norm_input_weight, norm_input_bias, eps)
    left = torch.nn.functional.linear(normalized, left_value_weight) * torch.sigmoid(
        torch.nn.functional.linear(normalized, left_gate_weight)
    )
    right = torch.nn.functional.linear(normalized, right_value_weight) * torch.sigmoid(
        torch.nn.functional.linear(normalized, right_gate_weight)
    )
    left, right = _mask_triangle_factors(left, right, pair_mask)
    equation = "...ikh,...jkh->...ijh" if outgoing else "...kih,...kjh->...ijh"
    update = torch.einsum(equation, left, right)
    normed_update = torch.nn.functional.layer_norm(update, (hidden_size,), norm_output_weight, norm_output_bias, eps)
    update = torch.nn.functional.linear(normed_update, output_weight)
    return update * torch.sigmoid(torch.nn.functional.linear(normalized, output_gate_weight))


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
    pair_mask = pair_mask[..., None].to(dtype=left.dtype, device=left.device)
    return left * pair_mask, right * pair_mask
