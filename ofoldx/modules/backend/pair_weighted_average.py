r"""
Backend dispatch for the MSA pair-weighted-averaging operator.

The dispatcher owns backend selection, availability, and calling-convention
adaptation. Modules pass the ordinary tensor arguments without a second
validation layer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_pair_weighted_average = oops_kernel("pair_weighted_average")

_SUPPORTED = ("oops", "eager")


def pair_weighted_average(
    msa: Tensor,
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    norm_msa_weight: Tensor,
    norm_msa_bias: Tensor,
    norm_pair_weight: Tensor,
    norm_pair_bias: Tensor,
    value_weight: Tensor,
    gate_weight: Tensor,
    pair_logits_weight: Tensor,
    output_weight: Tensor,
    num_heads: int,
    head_dim: int,
    norm_msa_eps: float,
    norm_pair_eps: float,
    operator_backend: str,
) -> Tensor:
    r"""
    Average MSA values along the query-residue axis with per-head weights softmaxed from the pair
    representation, then sigmoid-gate and project back, on the selected backend.

    Args:
        msa: ``(B, S, N, msa_size)`` MSA representation.
        pair: ``(B, N, N, pair_size)`` pair representation.
        pair_mask: Optional ``(B, N, N)`` mask over invalid keys.
        norm_msa_weight, norm_msa_bias, norm_msa_eps: MSA LayerNorm parameters.
        norm_pair_weight, norm_pair_bias, norm_pair_eps: Pair LayerNorm parameters.
        value_weight, gate_weight, pair_logits_weight, output_weight: Bias-free projection weights.
        num_heads, head_dim: Attention head count and per-head width.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``(B, S, N, msa_size)`` MSA update, matching the shape of ``msa``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.pair_weighted_average import pair_weighted_average
        >>> _ = torch.manual_seed(0)
        >>> num_heads, head_dim = 2, 3
        >>> msa = torch.randn(1, 2, 3, 4)
        >>> pair = torch.randn(1, 3, 3, 6)
        >>> out = pair_weighted_average(
        ...     msa,
        ...     pair,
        ...     None,
        ...     norm_msa_weight=torch.ones(4),
        ...     norm_msa_bias=torch.zeros(4),
        ...     norm_pair_weight=torch.ones(6),
        ...     norm_pair_bias=torch.zeros(6),
        ...     value_weight=torch.randn(num_heads * head_dim, 4),
        ...     gate_weight=torch.randn(num_heads * head_dim, 4),
        ...     pair_logits_weight=torch.randn(num_heads, 6),
        ...     output_weight=torch.randn(4, num_heads * head_dim),
        ...     num_heads=num_heads,
        ...     head_dim=head_dim,
        ...     norm_msa_eps=1e-5,
        ...     norm_pair_eps=1e-5,
        ...     operator_backend="eager",
        ... )
        >>> out.shape
        torch.Size([1, 2, 3, 4])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(msa, pair, pair_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_pair_weighted_average(
            msa,
            pair,
            pair_mask,
            norm_msa_weight=norm_msa_weight,
            norm_msa_bias=norm_msa_bias,
            norm_pair_weight=norm_pair_weight,
            norm_pair_bias=norm_pair_bias,
            value_weight=value_weight,
            gate_weight=gate_weight,
            pair_logits_weight=pair_logits_weight,
            output_weight=output_weight,
            num_heads=num_heads,
            norm_msa_eps=norm_msa_eps,
            norm_pair_eps=norm_pair_eps,
        )

    normalized = torch.nn.functional.layer_norm(msa, (msa.shape[-1],), norm_msa_weight, norm_msa_bias, norm_msa_eps)
    values = torch.nn.functional.linear(normalized, value_weight).reshape(*normalized.shape[:-1], num_heads, head_dim)
    pair_normalized = torch.nn.functional.layer_norm(
        pair, (pair.shape[-1],), norm_pair_weight, norm_pair_bias, norm_pair_eps
    )
    logits = torch.nn.functional.linear(pair_normalized, pair_logits_weight).movedim(-1, -3)
    if pair_mask is not None:
        logits = logits.masked_fill(~pair_mask.unsqueeze(-3).bool(), torch.finfo(logits.dtype).min)

    weights = torch.softmax(logits, dim=-1)
    weights = weights.to(dtype=values.dtype)
    averaged = torch.einsum("...hij,...sjhd->...sihd", weights, values)
    averaged = averaged.reshape(*normalized.shape[:-1], num_heads * head_dim)
    gated = averaged * torch.sigmoid(torch.nn.functional.linear(normalized, gate_weight))
    return torch.nn.functional.linear(gated, output_weight)
