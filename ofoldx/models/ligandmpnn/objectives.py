r"""
LigandMPNN training objectives.

LigandMPNN is protein sequence design with ligand context; its recovery loss is the same uniform
label-smoothed cross entropy as ProteinMPNN.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .model import LigandMpnnInput, LigandMpnnOutput


def criterion(
    pred: LigandMpnnOutput,
    target: LigandMpnnInput,
    *,
    label_smoothing: float = 0.1,
) -> Tensor:
    r"""
    DanLing-shaped sequence-recovery loss over the designable residues.
    """

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    num_classes = log_probabilities.shape[-1]
    target_distribution = F.one_hot(target.sequence_tokens, num_classes).to(log_probabilities.dtype)
    target_distribution = target_distribution + label_smoothing / num_classes
    target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True)
    per_residue = -(target_distribution * log_probabilities).sum(dim=-1)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    scope = mask.sum()
    return (per_residue * mask).sum() / scope
