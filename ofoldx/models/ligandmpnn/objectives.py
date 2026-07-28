r"""
LigandMPNN training objectives.

LigandMPNN is protein sequence design with ligand context; its recovery loss is the same uniform
label-smoothed cross entropy as ProteinMPNN.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ...modules.mpnn import label_smoothed_nll_loss
from ...modules.mpnn import nll_loss as _nll_loss


def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor:
    r"""Return LigandMPNN's unreduced per-token negative log likelihood."""

    return _nll_loss(sequence_tokens, log_probabilities)


def smoothed_loss(sequence_tokens: Tensor, log_probabilities: Tensor, *, weight: float = 0.1) -> Tensor:
    r"""Return LigandMPNN's unreduced uniformly label-smoothed loss."""

    return label_smoothed_nll_loss(sequence_tokens, log_probabilities, weight=weight)


def criterion(pred: Any, target: Any, *, label_smoothing: float = 0.1) -> Tensor:
    r"""DanLing-shaped sequence-recovery loss over the designable residues."""

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    per_residue = smoothed_loss(target.sequence_tokens, log_probabilities, weight=label_smoothing)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    return (per_residue * mask).sum() / mask.sum()


def criterion_factory(model: Any, processor: Any) -> Any:
    r"""Runner criterion factory: LigandMPNN's loss needs no Model/Processor binding."""

    return criterion
