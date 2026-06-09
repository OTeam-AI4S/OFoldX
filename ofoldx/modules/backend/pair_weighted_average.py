r"""
Backend dispatch for the MSA pair-weighted-averaging operator.

The dispatcher owns everything backend-specific for this operator in one place: the supported
backends, ``"auto"`` resolution, the CUDA guard, and each backend's calling convention. Modules pass
their gathered parameters plus the requested ``operator_backend`` and stay backend-agnostic. New
backends (cuEquivariance / NVIDIA) are added as a branch here, not in every caller.
"""

from __future__ import annotations

import torch
from oops import OOpsUnsupportedError
from oops import pair_weighted_average as _oops_pair_weighted_average
from torch import Tensor

from .resolve import resolve_tensor_operator_backend

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

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

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
        if not msa.is_cuda or not pair.is_cuda:
            raise OOpsUnsupportedError(
                "MsaPairWeightedAveraging backend 'oops' requires CUDA tensors; use operator_backend='eager'"
            )
        if pair_mask is not None and not pair_mask.is_cuda:
            raise OOpsUnsupportedError(
                "MsaPairWeightedAveraging backend 'oops' requires a CUDA pair_mask; use operator_backend='eager'"
            )
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
            eps=norm_msa_eps,
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
