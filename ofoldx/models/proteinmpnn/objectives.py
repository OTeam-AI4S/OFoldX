r"""
ProteinMPNN training objectives.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ...modules.mpnn import label_smoothed_nll_loss
from .model import ProteinMpnnInput, ProteinMpnnOutput


def criterion(
    pred: ProteinMpnnOutput,
    target: ProteinMpnnInput,
    *,
    label_smoothing: float = 0.1,
) -> Tensor:
    r"""
    DanLing-shaped loss: score the model output's logits for the observed sequence and reduce over the
    designable residues (``residue_mask & design_positions``).

    Args:
        pred: The model output (``ProteinMpnnOutput`` with ``logits``).
        target: The batched ``ProteinMpnnInput`` carrying ``sequence_tokens`` and the masks.
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
