r"""Structure-conditioned MPNN probability result carrier."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..data.molecular import ComponentReference
from ..data.structure import Structure


@dataclass(eq=False)
class MpnnProbabilityResult:
    r"""Repeated aligned probability output for one structure-conditioned MPNN call.

    Args:
        target: Canonical structure used for conditioning.
        component_references: Residue references aligned to the sequence axis. ``None`` marks a
            padded or unresolved row.
        sequence_tokens: Input token ids with shape ``[residues]``.
        residue_mask: Valid-residue mask with shape ``[residues]``.
        design_positions: Designed-position mask with shape ``[residues]``.
        logits: Vocabulary logits with shape ``[evaluations, residues, vocabulary]``.
        decoding_ranks: Autoregressive ranks with shape ``[evaluations, residues]``.
    """

    target: Structure
    component_references: tuple[ComponentReference | None, ...]
    sequence_tokens: Tensor
    residue_mask: Tensor
    design_positions: Tensor
    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""Return vocabulary log probabilities for every retained evaluation."""

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""Return vocabulary probabilities for every retained evaluation."""

        return self.logits.softmax(dim=-1)

    @property
    def mean_probabilities(self) -> Tensor:
        r"""Average vocabulary probabilities over the leading evaluation axis."""

        return self.probabilities.mean(dim=0)

    @property
    def probability_standard_deviation(self) -> Tensor:
        r"""Return the population standard deviation over repeated evaluations."""

        return self.probabilities.std(dim=0, correction=0)
