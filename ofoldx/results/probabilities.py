r"""
Autoregressive probability result carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..data.molecular import ComponentReference
from ..data.structure import Structure


@dataclass(eq=False)
class AutoregressiveProbabilityResult:
    r"""
    Probability evaluations from a structure-conditioned autoregressive model.

    Attributes:
        target: Structure used for conditioning.
        component_references: Component occurrence for each sequence row; ``None`` marks
            padding or an unresolved row.
        sequence_tokens: Input token IDs with shape ``(num_rows,)``.
        residue_mask: Valid-row mask with shape ``(num_rows,)``.
        design_positions: Designed-position mask with shape ``(num_rows,)``.
        logits: Vocabulary logits with shape
            ``(num_evaluations, num_rows, vocabulary_size)``.
        decoding_ranks: Autoregressive ranks with shape
            ``(num_evaluations, num_rows)`` for whole-sequence evaluations.
            Per-row conditional and unconditional evaluations retain the
            effective order for every target and have shape
            ``(num_evaluations, num_target_rows, num_rows)``.
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
        r"""Return log probabilities for every evaluation."""

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""Return probabilities for every evaluation."""

        return self.logits.softmax(dim=-1)

    @property
    def mean_probabilities(self) -> Tensor:
        r"""Return probabilities averaged over evaluations."""

        return self.probabilities.mean(dim=0)

    @property
    def probability_standard_deviation(self) -> Tensor:
        r"""Return population standard deviations over evaluations."""

        return self.probabilities.std(dim=0, correction=0)
