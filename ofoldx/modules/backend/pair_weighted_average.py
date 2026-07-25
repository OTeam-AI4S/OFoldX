r"""
Backend-dispatched MSA pair-weighted averaging.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_pair_weighted_average = oops_kernel("pair_weighted_average")

_SUPPORTED = ("oops", "apex", "deepspeed", "eager")


def pair_weighted_average(
    msa: Tensor,
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    msa_norm_weight: Tensor,
    msa_norm_bias: Tensor,
    pair_norm_weight: Tensor,
    pair_norm_bias: Tensor,
    value_weight: Tensor,
    gate_weight: Tensor,
    pair_logits_weight: Tensor,
    output_weight: Tensor,
    num_attention_heads: int,
    head_dim: int,
    msa_norm_eps: float,
    pair_norm_eps: float,
    backend: str,
) -> Tensor:
    r"""
    Average MSA values along the query-residue axis with per-head weights softmaxed from the pair
    representation, then sigmoid-gate and project back, on the selected backend.

    Args:
        msa: ``(B, S, N, msa_size)`` MSA representation.
        pair: ``(B, N, N, pair_size)`` pair representation.
        pair_mask: Optional ``(B, N, N)`` mask over invalid keys.
        msa_norm_weight: MSA LayerNorm weight.
        msa_norm_bias: MSA LayerNorm bias.
        pair_norm_weight: Pair LayerNorm weight.
        pair_norm_bias: Pair LayerNorm bias.
        value_weight: Bias-free value projection weight.
        gate_weight: Bias-free gate projection weight.
        pair_logits_weight: Bias-free pair-logit projection weight.
        output_weight: Bias-free output projection weight.
        num_attention_heads: Number of attention heads.
        head_dim: Width of each attention head.
        msa_norm_eps: Epsilon used by the MSA LayerNorm.
        pair_norm_eps: Epsilon used by the pair LayerNorm.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``(B, S, N, msa_size)`` MSA update, matching the shape of ``msa``.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(msa, pair, pair_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_pair_weighted_average(
            msa,
            pair,
            pair_mask,
            msa_norm_weight=msa_norm_weight,
            msa_norm_bias=msa_norm_bias,
            pair_norm_weight=pair_norm_weight,
            pair_norm_bias=pair_norm_bias,
            value_weight=value_weight,
            gate_weight=gate_weight,
            pair_logits_weight=pair_logits_weight,
            output_weight=output_weight,
            num_heads=num_attention_heads,
            msa_norm_eps=msa_norm_eps,
            pair_norm_eps=pair_norm_eps,
        )

    normalized = layer_norm(msa, (msa.shape[-1],), msa_norm_weight, msa_norm_bias, msa_norm_eps, backend=backend)
    values = torch.nn.functional.linear(normalized, value_weight).reshape(
        *normalized.shape[:-1],
        num_attention_heads,
        head_dim,
    )
    pair_normalized = layer_norm(
        pair,
        (pair.shape[-1],),
        pair_norm_weight,
        pair_norm_bias,
        pair_norm_eps,
        backend=backend,
    )
    logits = torch.nn.functional.linear(pair_normalized, pair_logits_weight).movedim(-1, -3)
    if pair_mask is not None:
        logits = logits.masked_fill(~pair_mask.unsqueeze(-3).bool(), torch.finfo(logits.dtype).min)

    weights = torch.softmax(logits, dim=-1)
    weights = weights.to(dtype=values.dtype)
    averaged = torch.einsum("...hij,...sjhd->...sihd", weights, values)
    averaged = averaged.reshape(*normalized.shape[:-1], num_attention_heads * head_dim)
    gated = averaged * torch.sigmoid(torch.nn.functional.linear(normalized, gate_weight))
    return torch.nn.functional.linear(gated, output_weight)
