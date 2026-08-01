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
Confidence expectations from categorical predictions.
"""

import torch
from torch import Tensor


def categorical_expectation(probabilities: Tensor, bin_values: Tensor) -> Tensor:
    r"""Return the probability-weighted bin values, reducing the final bin axis.

    ``bin_values`` broadcasts against ``probabilities``. Inputs retain ordinary
    PyTorch dtype promotion; callers prepare values in the metric's units.
    """

    return (probabilities * bin_values).sum(dim=-1)


def categorical_mean(logits: Tensor, *, start: float = 0.0, end: float = 1.0) -> Tensor:
    r"""
    Return the softmax expectation over uniform bin midpoints in ``[start, end]``.

    Args:
        logits: Categorical logits with shape ``(..., num_bins)``.
        start: Lower edge of the first bin, in the caller's units.
        end: Upper edge of the last bin, in the caller's units.

    Returns:
        Expected values with shape ``(...)``, in the logits' dtype.
    """

    centers = start + (torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) + 0.5) * (
        (end - start) / logits.shape[-1]
    )
    return categorical_expectation(logits.softmax(dim=-1), centers)


def categorical_entropy_from_logits(logits: Tensor) -> Tensor:
    r"""Return categorical entropy in nat over the final logit axis.

    Normalize in log space. Negative-infinity logits denote missing categories,
    whose zero-probability terms and gradients are zero. Empty categories or an
    all-missing support report NaN with zero input gradients. NaN and positive
    infinity remain invalid logits; dtype and device follow ordinary PyTorch.
    """

    missing = logits == -torch.inf
    present = (~missing).any(dim=-1, keepdim=True)
    # An absent support has no log-normalizer. Its zero-gradient identity is
    # evaluated on zeros before selecting the undefined entropy below.
    normalized_logits = torch.where(present, logits, torch.zeros_like(logits))
    log_probabilities = normalized_logits.log_softmax(dim=-1)
    probabilities = log_probabilities.exp().masked_fill(missing, 0.0)
    log_probabilities = log_probabilities.masked_fill(missing, 0.0)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    return torch.where(present.squeeze(-1), entropy, entropy.new_full((), float("nan")))


def plddt_from_logits(logits: Tensor) -> Tensor:
    r"""Return expected pLDDT on the 0--100 scale from confidence-bin logits."""

    return categorical_mean(logits, end=100.0)


def categorical_probability_below(logits: Tensor, bin_values: Tensor, cutoff: float) -> Tensor:
    r"""Return categorical probability for bin values strictly below ``cutoff``.

    Bin values broadcast against the final categorical axis. The caller owns
    their scientific units and the definition of the distance bins.
    """

    return (logits.softmax(dim=-1) * (bin_values < cutoff)).sum(dim=-1)


def resolved_probability_from_logits(logits: Tensor) -> Tensor:
    r"""Return resolved probability from canonical ``[unresolved, resolved]`` logits."""

    return logits.softmax(dim=-1)[..., 1]


def interface_confidence_weights(
    distances: Tensor,
    chain_index: Tensor,
    is_ligand: Tensor,
    valid: Tensor,
    *,
    ligand_weight: float = 2.0,
    interface_weight: float = 1.0,
    background_weight: float = 0.0,
    cutoff: float = 8.0,
) -> Tensor:
    r"""Weight confidence observations by valid inter-chain contact membership.

    Both endpoints must be valid before a pair can create an interface. Ligand
    observations take their own weight; nonligands use interface or background
    weight. Distances have trailing source/target axes, with metadata broadcasting
    over their shared observation axis. Output weights retain distance dtype.
    """

    selected = valid.bool()
    different_chain = chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)
    contacts = ((distances < cutoff) & different_chain & selected.unsqueeze(-1) & selected.unsqueeze(-2)).any(dim=-1)
    weight = torch.where(contacts, distances.new_tensor(interface_weight), distances.new_tensor(background_weight))
    weight = torch.where(is_ligand.bool(), distances.new_tensor(ligand_weight), weight)
    return weight.to(distances).masked_fill(~selected, 0.0)


def minimum_confidence_error(values: Tensor, mask: Tensor, *, dim: tuple[int, ...] = (-2, -1)) -> Tensor:
    r"""Return the smallest selected error; an absent observation domain is NaN."""

    values, selected = torch.broadcast_tensors(values, mask.bool())
    if any(values.shape[axis] == 0 for axis in dim):
        return values.sum(dim=dim) + values.new_full((), float("nan"))
    result = values.masked_fill(~selected, torch.inf).amin(dim=dim)
    return torch.where(selected.any(dim=dim), result, result.new_full((), float("nan")))
