r"""
LigandMPNN training objectives.

LigandMPNN is protein sequence design with ligand context; its recovery loss is the same uniform
label-smoothed cross entropy as ProteinMPNN.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ...modules.mpnn import label_smoothed_nll_loss
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
    per_residue = label_smoothed_nll_loss(
        target.sequence_tokens,
        log_probabilities,
        weight=label_smoothing,
    )
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    scope = mask.sum()
    return (per_residue * mask).sum() / scope
