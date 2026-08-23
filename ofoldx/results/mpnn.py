r"""
Structure-conditioned MPNN probability result carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..data.molecular import ComponentReference
from ..data.structure import Structure


@dataclass(eq=False)
class MpnnProbabilityResult:
    r"""
    Probability evaluations from a structure-conditioned MPNN.

    Attributes:
        target: Structure used for conditioning.
        component_references: Component occurrence for each residue row; ``None`` marks
            padding or an unresolved row.
        sequence_tokens: Input token IDs with shape ``(num_residues,)``.
        residue_mask: Valid-residue mask with shape ``(num_residues,)``.
        design_mask: Designed-position mask with shape ``(num_residues,)``.
        logits: Vocabulary logits with shape
            ``(num_evaluations, num_residues, vocabulary_size)``.
        decoding_ranks: Autoregressive ranks with shape
            ``(num_evaluations, num_residues)`` for whole-sequence evaluations.
            Per-residue conditional and unconditional evaluations retain the
            effective order for every target and have shape
            ``(num_evaluations, num_target_residues, num_residues)``.
    """

    target: Structure
    component_references: tuple[ComponentReference | None, ...]
    sequence_tokens: Tensor
    residue_mask: Tensor
    design_mask: Tensor
    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Return log probabilities for every evaluation.

        Returns:
            A tensor with shape
            ``(num_evaluations, num_residues, vocabulary_size)``.
        """

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""
        Return probabilities for every evaluation.

        Returns:
            A tensor with shape
            ``(num_evaluations, num_residues, vocabulary_size)``.
        """

        return self.logits.softmax(dim=-1)

    @property
    def mean_probabilities(self) -> Tensor:
        r"""
        Return probabilities averaged over evaluations.

        Returns:
            A tensor with shape ``(num_residues, vocabulary_size)``.
        """

        return self.probabilities.mean(dim=0)

    @property
    def probability_standard_deviation(self) -> Tensor:
        r"""
        Return population standard deviations over evaluations.

        Returns:
            A tensor with shape ``(num_residues, vocabulary_size)``.
        """

        return self.probabilities.std(dim=0, correction=0)
