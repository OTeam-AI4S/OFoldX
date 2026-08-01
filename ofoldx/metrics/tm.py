# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Predicted TM-score mathematics.
"""

from collections.abc import Iterable
from typing import cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from ..modules.functional import weighted_mean
from .confidence import categorical_expectation


def tm_length_scale(num_tokens: Tensor | float) -> Tensor | float:
    r"""Return ``d0(n) = 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8``."""

    if isinstance(num_tokens, Tensor):
        num_tokens = num_tokens.clamp(min=19)
    else:
        num_tokens = max(float(num_tokens), 19.0)
    return 1.24 * (num_tokens - 15.0) ** (1.0 / 3.0) - 1.8


def tm_bin_weights(bin_centers: Tensor, num_tokens: Tensor | float) -> Tensor:
    r"""Return the per-bin weight ``1 / (1 + (center / d0(n)) ** 2)``."""

    scale = tm_length_scale(num_tokens)
    if isinstance(scale, Tensor):
        scale = scale.unsqueeze(-1)
    return 1.0 / (1.0 + (bin_centers / scale) ** 2)


def tm_weight_from_pae(pae_logits: Tensor, num_tokens: Tensor | float, *, bin_centers: Tensor) -> Tensor:
    r"""Return the expected per-pair TM weight from PAE logits."""

    probability = pae_logits.softmax(dim=-1)
    return categorical_expectation(probability, tm_bin_weights(bin_centers, num_tokens))


def predicted_tm_score(
    tm_weights: Tensor,
    *,
    pair_weights: Tensor | None = None,
    reference_weights: Tensor | None = None,
    backend: str = "auto",
) -> Tensor:
    r"""
    Return the TM mean at the best eligible source reference frame.

    Target weights normalize each source row exactly. Reference weights choose
    the largest weighted row mean; the returned mean retains its original scale.
    Without explicit reference weights, only rows with positive target weight
    are eligible. An empty eligible-reference set scores zero. A source explicitly
    declared eligible with no target weight has an undefined mean (NaN).

    Args:
        tm_weights: Expected per-pair TM terms with shape ``(..., N_source, N_target)``.
        pair_weights: Nonnegative target-pair weights broadcastable to
            ``tm_weights``; ``None`` gives every target equal weight.
        reference_weights: Nonnegative source-reference selection weights with
            shape broadcastable to ``(..., N_source)``; zero excludes a source.

    Returns:
        Scores with shape ``(...)`` on the ``0--1`` TM scale.
    """

    if pair_weights is None:
        pair_weights = torch.ones((), device=tm_weights.device, dtype=tm_weights.dtype)
    weights = torch.broadcast_tensors(tm_weights, pair_weights.to(device=tm_weights.device))[1]
    references = (
        (weights > 0).any(dim=-1).to(dtype=tm_weights.dtype)
        if reference_weights is None
        else reference_weights.to(device=tm_weights.device)
    )
    eligible = references > 0
    alignment = torch.where(eligible, weighted_mean(tm_weights, weights, backend=backend), 0.0)
    alignment, references = torch.broadcast_tensors(alignment, references)
    eligible = references > 0
    selection = (alignment * references).masked_fill(~eligible, -torch.inf)
    # An identity candidate defines the empty reference domain without changing
    # ordinary max/argmax's empty-tensor contract or adding a ragged-only formula.
    identity = torch.zeros_like(alignment.sum(dim=-1, keepdim=True))
    alignment = torch.cat((alignment, identity), dim=-1)
    selection = torch.cat((selection, torch.full_like(identity, -torch.inf)), dim=-1)
    best_reference = selection.argmax(dim=-1, keepdim=True)
    # The choice is already one dense index per batch/sample. Its argmax was
    # taken over true ragged rows, including the identity row, so padded rows
    # cannot be selected. Lower only this final vector for the dense lookup.
    rows = alignment.tensor if isinstance(alignment, NestedTensor) else alignment
    score = rows.gather(-1, best_reference).squeeze(-1)
    return torch.where(eligible.any(dim=-1), score, 0.0)


def mean_partner_tm_scores(
    chain_pair_scores: Tensor, *, pair_mask: Tensor | None = None, backend: str = "auto"
) -> Tensor:
    r"""Average finite scores over distinct eligible partner chains once each.

    Rows identify the chain and columns its partners. Diagonal entries are
    excluded. ``pair_mask`` carries the owner's observation eligibility;
    an empty partner domain returns NaN with zero excluded-value gradients.
    """

    eligible = torch.isfinite(chain_pair_scores) & ~torch.eye(
        chain_pair_scores.shape[-1], device=chain_pair_scores.device, dtype=torch.bool
    )
    if pair_mask is not None:
        eligible = eligible & pair_mask
    return weighted_mean(chain_pair_scores, eligible, backend=backend)


def chain_pair_tm_scores(
    tm_weights: Tensor,
    chain_index: Tensor,
    chain_ids: Iterable[int],
    *,
    pair_weights: Tensor | None = None,
    backend: str = "auto",
) -> Tensor:
    r"""
    Return directed chain-pair scores from already calibrated TM expectations.

    Each matrix entry selects the best eligible frame in its source chain and
    averages over targets in its target chain. Diagonal entries are intra-chain
    scores; an empty source/target pair scores zero. Calibration belongs to the
    caller, so this function does not change the length scale.
    """

    chains = tuple(chain_ids)
    scores = tm_weights.new_empty((*tm_weights.shape[:-2], len(chains), len(chains)))
    for source_index, source_chain in enumerate(chains):
        source = chain_index == source_chain
        for target_index, target_chain in enumerate(chains):
            target = chain_index == target_chain
            pairs = source.unsqueeze(-1) & target.unsqueeze(-2)
            weights = pairs if pair_weights is None else pairs * pair_weights
            scores[..., source_index, target_index] = predicted_tm_score(
                tm_weights, pair_weights=weights, backend=backend
            )
    return scores


def chain_pair_mean_tm_scores(
    tm_weights: Tensor,
    chain_index: Tensor,
    chain_ids: Iterable[int],
    *,
    pair_weights: Tensor | None = None,
    backend: str = "auto",
) -> Tensor:
    r"""Average calibrated TM expectations over both directed chain-pair axes.

    Rows identify source chains and columns identify target chains in
    ``chain_ids`` order. Diagonal entries retain the intra-chain pair mean;
    empty observation domains return zero with zero gradients. Calibration
    belongs to the caller, and this reduction does not select a source frame.
    """

    chains = tuple(chain_ids)
    scores = tm_weights.new_empty((*tm_weights.shape[:-2], len(chains), len(chains)))
    for source_index, source_chain in enumerate(chains):
        source = chain_index == source_chain
        for target_index, target_chain in enumerate(chains):
            target = chain_index == target_chain
            pairs = source.unsqueeze(-1) & target.unsqueeze(-2)
            weights = pairs if pair_weights is None else pairs * pair_weights
            values, weights = torch.broadcast_tensors(tm_weights, weights)
            mean = weighted_mean(values.flatten(-2), weights.flatten(-2), backend=backend)
            scores[..., source_index, target_index] = torch.where((weights > 0).any(dim=(-2, -1)), mean, 0.0)
    return scores


def directional_ipsae(
    pae: Tensor,
    *,
    source_mask: Tensor,
    target_mask: Tensor,
    frame_mask: Tensor,
    token_mask: Tensor,
    cutoff: float = 15.0,
    d0_min: float = 1.0,
    backend: str = "auto",
) -> Tensor:
    r"""
    Compute directional iPSAE from expected pairwise aligned errors in Angstrom.

    Valid target pairs below the cutoff determine each source row's length scale.
    The score is the largest source-row mean TM weight. Frame validity applies
    to the source endpoint; token validity applies to both endpoints.

    Args:
        pae: Expected errors with shape ``(..., N, N)``.
        source_mask: Source-token selection with shape ``(..., N)``.
        target_mask: Target-token selection with shape ``(..., N)``.
        frame_mask: Valid source frames with shape ``(..., N)``.
        token_mask: Valid tokens with shape ``(..., N)``.
        cutoff: Strict upper bound on scored errors, in Angstrom.
        d0_min: Minimum row length scale, in Angstrom.

    Returns:
        One directional score per leading batch element, with shape ``(...)``.
    """

    pair_mask = (
        frame_mask.unsqueeze(-1)
        * token_mask.unsqueeze(-2)
        * token_mask.unsqueeze(-1)
        * source_mask.unsqueeze(-1).to(pae)
        * target_mask.unsqueeze(-2).to(pae)
    )
    valid_pae = (pae < cutoff).to(pae) * pair_mask
    num_valid = valid_pae.sum(dim=-1)
    scale = cast(Tensor, tm_length_scale(num_valid)).clamp_min(d0_min)
    terms = 1.0 / (1.0 + (pae / scale.unsqueeze(-1)).square())
    return predicted_tm_score(terms, pair_weights=valid_pae, backend=backend)
