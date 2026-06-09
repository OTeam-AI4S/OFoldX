r"""
pTM / ipTM confidence scoring from predicted-aligned-error logits.

The confidence head emits per-token-pair PAE logits; this module turns them into the scalar pTM
(predicted TM-score) and ipTM (interface predicted TM-score) summaries that rank predictions. The
computation is a faithful port of the upstream AlphaFold3 TM-adjusted-PAE pipeline
(``model/network/confidence_head.py`` ``_get_tmscore_adjusted_pae`` and ``model/confidences.py``
``predicted_tm_score``): softmax the PAE logits into a distance-error distribution, weight each bin by
its TM-score term ``1 / (1 + (d / d0)^2)`` with the TM-score length scale ``d0(n)``, then take the
best per-row masked average. It adds no parameters and runs entirely on the head outputs, so it is an
opt-in post-processing summary rather than part of the network graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# TM-score length-scale constants (Yang & Skolnick 2004, eqn. 5); ``d0`` is undefined below 19 tokens.
_D0_SCALE = 1.24
_D0_OFFSET = 15.0
_D0_SHIFT = 1.8
_D0_MIN_TOKENS = 19
# ipTM contributes 0.8 and pTM 0.2 to the combined ranking score (upstream ``_IPTM_WEIGHT``).
_IPTM_WEIGHT = 0.8


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    r"""
    Scalar confidence summaries derived from PAE logits.

    Attributes:
        ptm: ``[...]`` predicted TM-score over all token pairs.
        iptm: ``[...]`` interface predicted TM-score over cross-chain pairs; ``NaN`` when the system has
            a single chain (no interface is defined).
        ranking_score: ``[...]`` combined ``0.8 * iptm + 0.2 * ptm`` ranking score, falling back to
            ``ptm`` where ``iptm`` is ``NaN``.

    Examples:
        >>> import torch
        >>> summary = ConfidenceSummary(
        ...     ptm=torch.tensor([0.8]), iptm=torch.tensor([0.6]), ranking_score=torch.tensor([0.64])
        ... )
        >>> summary.ranking_score.tolist()
        [0.6399999856948853]
    """

    ptm: Tensor
    iptm: Tensor
    ranking_score: Tensor


def pae_bin_centers(num_bins: int, max_error_bin: float = 31.0, *, device: torch.device | None = None) -> Tensor:
    r"""
    Return the PAE-bin center distances used to take the expected aligned error.

    The first ``num_bins - 1`` centers evenly tile ``[0, max_error_bin]`` at half-step offsets and a
    final catch-all bin is appended one step beyond, matching the upstream ``pae_breaks`` construction.

    Args:
        num_bins: Number of PAE logit bins.
        max_error_bin: Upper edge of the resolved (non-catch-all) bins in Angstrom.
        device: Optional device for the returned tensor.

    Returns:
        ``[num_bins]`` bin-center distances in Angstrom.

    Raises:
        ValueError: If ``num_bins`` is less than two.

    Examples:
        >>> import torch
        >>> centers = pae_bin_centers(4)
        >>> centers.shape
        torch.Size([4])
        >>> centers.round(decimals=3).tolist()
        [7.75, 23.25, 38.75, 54.25]
        >>> pae_bin_centers(1)
        Traceback (most recent call last):
            ...
        ValueError: num_bins must be at least 2, got 1
    """

    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step], dim=0)


def tm_adjusted_pae(
    pae_logits: Tensor,
    *,
    pair_mask: Tensor,
    asym_id: Tensor,
    interface: bool,
    max_error_bin: float = 31.0,
) -> Tensor:
    r"""
    Return the per-token-pair TM-adjusted PAE from PAE logits.

    Softmaxes the logits into a distance-error distribution, weights each bin by its TM-score term
    ``1 / (1 + (d / d0)^2)``, and sums over bins. The TM-score length scale ``d0`` uses the interface
    token count (both chains of the pair) when ``interface`` is set, else the global token count, so
    ipTM and pTM use the same tensor construction with different normalization counts (upstream
    ``tmscore_adjusted_pae_interface`` vs ``tmscore_adjusted_pae_global``).

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        asym_id: ``[..., N_token]`` chain (asym) id per token.
        interface: If ``True``, use the interface token count for ``d0``; else the global count.
        max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

    Returns:
        ``[..., N_token, N_token]`` TM-adjusted PAE.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> logits = torch.randn(1, 3, 3, 4)
        >>> mask = torch.ones(1, 3, dtype=torch.bool)
        >>> pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> tm_adjusted_pae(logits, pair_mask=pair_mask, asym_id=asym_id, interface=False).shape
        torch.Size([1, 3, 3])
    """

    num_bins = pae_logits.shape[-1]
    centers = pae_bin_centers(num_bins, max_error_bin, device=pae_logits.device).to(pae_logits.dtype)
    probs = torch.softmax(pae_logits.float(), dim=-1).to(pae_logits.dtype)
    pair_mask = pair_mask.to(pae_logits.dtype)

    same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
    if interface:
        num_chain_tokens = (same_chain.to(pae_logits.dtype) * pair_mask).sum(dim=-1)
        num_tokens = num_chain_tokens.unsqueeze(-1) + num_chain_tokens.unsqueeze(-2)
        # Don't double-count within a single chain (upstream ``num_interface_tokens`` correction).
        num_tokens = num_tokens - same_chain.to(pae_logits.dtype) * torch.floor(num_tokens / 2)
        num_tokens = num_tokens * pair_mask
    else:
        num_tokens = pair_mask.new_full(pair_mask.shape[:-2], float(pair_mask.shape[-1])).unsqueeze(-1).unsqueeze(-1)
        num_tokens = num_tokens.expand_as(pair_mask)

    clipped = torch.clamp(num_tokens, min=float(_D0_MIN_TOKENS))
    d0 = _D0_SCALE * (clipped - _D0_OFFSET) ** (1.0 / 3) - _D0_SHIFT
    tm_per_bin = 1.0 / (1.0 + (centers / d0.unsqueeze(-1)) ** 2)
    return (probs * tm_per_bin).sum(dim=-1)


def predicted_tm_score(adjusted_pae: Tensor, *, pair_mask: Tensor, asym_id: Tensor, interface: bool) -> Tensor:
    r"""
    Reduce a TM-adjusted PAE matrix to a scalar pTM or ipTM.

    Restricts to cross-chain pairs when ``interface`` is set, row-normalizes the mask, takes the masked
    per-row average of the adjusted PAE, and returns the best row. Returns ``NaN`` when the (interface)
    mask is empty, matching upstream ``predicted_tm_score`` (single-chain systems have no ipTM).

    Args:
        adjusted_pae: ``[..., N_token, N_token]`` TM-adjusted PAE.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        asym_id: ``[..., N_token]`` chain id per token.
        interface: If ``True``, compute ipTM over cross-chain pairs; else pTM over all pairs.

    Returns:
        ``[...]`` pTM or ipTM score.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> adjusted = torch.rand(1, 3, 3)
        >>> mask = torch.ones(1, 3, dtype=torch.bool)
        >>> pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> predicted_tm_score(adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=False).shape
        torch.Size([1])
    """

    mask = pair_mask.to(adjusted_pae.dtype)
    if interface:
        mask = mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).to(adjusted_pae.dtype)
    normed = mask / (1e-8 + mask.sum(dim=-1, keepdim=True))
    per_alignment = (adjusted_pae * normed).sum(dim=-1)
    score = per_alignment.max(dim=-1).values
    empty = mask.sum(dim=(-2, -1)) == 0
    return torch.where(empty, torch.full_like(score, float("nan")), score)


def confidence_summary(
    pae_logits: Tensor,
    *,
    frame_mask: Tensor,
    asym_id: Tensor,
    max_error_bin: float = 31.0,
) -> ConfidenceSummary:
    r"""
    Compute pTM, ipTM, and the combined ranking score from PAE logits.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        frame_mask: ``[..., N_token]`` boolean mask of tokens with a valid frame.
        asym_id: ``[..., N_token]`` chain (asym) id per token.
        max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

    Returns:
        Scalar pTM / ipTM / ranking-score summaries broadcast over the leading batch/sample axes.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> logits = torch.randn(1, 3, 3, 4)
        >>> frame_mask = torch.ones(1, 3, dtype=torch.bool)
        >>> asym_id = torch.tensor([[0, 0, 1]])
        >>> summary = confidence_summary(logits, frame_mask=frame_mask, asym_id=asym_id)
        >>> summary.ptm.shape
        torch.Size([1])
        >>> summary.ranking_score.shape
        torch.Size([1])
    """

    frame_mask = frame_mask.to(torch.bool)
    pair_mask = frame_mask.unsqueeze(-1) & frame_mask.unsqueeze(-2)
    global_adjusted = tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, asym_id=asym_id, interface=False, max_error_bin=max_error_bin
    )
    interface_adjusted = tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, asym_id=asym_id, interface=True, max_error_bin=max_error_bin
    )
    ptm = predicted_tm_score(global_adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=False)
    iptm = predicted_tm_score(interface_adjusted, pair_mask=pair_mask, asym_id=asym_id, interface=True)
    ranking_score = torch.where(torch.isnan(iptm), ptm, _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm)
    return ConfidenceSummary(ptm=ptm, iptm=iptm, ranking_score=ranking_score)
