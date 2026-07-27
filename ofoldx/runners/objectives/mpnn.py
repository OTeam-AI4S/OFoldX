r"""
Sequence-recovery objectives for the MPNN inverse-folding families.

Each objective scores the model's teacher-forced logits for the observed sequence with the family's
own label-smoothed loss (ProteinMPNN's uniform smoothing, NA-MPNN's polymer-specific smoothing), then
averages over the residues that are both valid and eligible for design
(``residue_mask & design_positions`` -- the exact selection the models use internally).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...models.nampnn.objectives import smoothed_loss as nampnn_smoothed_loss
from ...models.proteinmpnn.objectives import smoothed_loss as proteinmpnn_smoothed_loss


def _masked_mean(per_residue: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(per_residue.dtype)
    return (per_residue * weights).sum() / weights.sum().clamp_min(torch.finfo(per_residue.dtype).tiny)


class ProteinMpnnObjective:
    r"""
    ProteinMPNN sequence-recovery loss with uniform label smoothing.

    Args:
        label_smoothing: Mass distributed uniformly over the vocabulary.
    """

    def __init__(self, *, label_smoothing: float = 0.1) -> None:
        self.label_smoothing = label_smoothing

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        output = model(batch)
        log_probabilities = F.log_softmax(output.logits, dim=-1)
        per_residue = proteinmpnn_smoothed_loss(
            batch.sequence_tokens,
            log_probabilities,
            weight=self.label_smoothing,
        )
        loss = _masked_mean(per_residue, batch.residue_mask & batch.design_positions)
        return output, loss


class NaMpnnObjective:
    r"""
    NA-MPNN sequence-recovery loss with polymer-specific label smoothing.

    Args:
        label_smoothing: Mass distributed over the tokens of each residue's polymer type.
        shared_nucleic_tokens: Whether RNA targets use the shared DNA token span; must match the
            processor that produced the batch.
    """

    def __init__(self, *, label_smoothing: float = 0.1, shared_nucleic_tokens: bool = True) -> None:
        self.label_smoothing = label_smoothing
        self.shared_nucleic_tokens = shared_nucleic_tokens

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        output = model(batch)
        log_probabilities = F.log_softmax(output.logits, dim=-1)
        per_residue = nampnn_smoothed_loss(
            batch.sequence_tokens,
            log_probabilities,
            batch.polymer_type_index,
            weight=self.label_smoothing,
            shared_nucleic_tokens=self.shared_nucleic_tokens,
        )
        loss = _masked_mean(per_residue, batch.residue_mask & batch.design_positions)
        return output, loss
