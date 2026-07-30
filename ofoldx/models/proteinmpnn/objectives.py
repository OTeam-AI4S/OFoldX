r"""
ProteinMPNN training objectives.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ...modules.mpnn import label_smoothed_nll_loss as smoothed_loss


def criterion(pred: Any, target: Any, *, label_smoothing: float = 0.1) -> Tensor:
    r"""
    DanLing-shaped loss: score the model output's logits for the observed sequence and reduce over the
    designable residues (``residue_mask & design_positions``).

    Args:
        pred: The model output (``ProteinMpnnOutput`` with ``logits``).
        target: The batched ``ProteinMpnnInput`` carrying ``sequence_tokens`` and the masks.

    Raises:
        ValueError: If the batch scores no designable residue.
    """

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    per_residue = smoothed_loss(target.sequence_tokens, log_probabilities, weight=label_smoothing)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    scope = mask.sum()
    # An empty scope would divide by zero; the resulting NaN reaches the optimizer and permanently
    # poisons both the weights and the optimizer state, so refuse the batch instead of scoring it.
    if not scope:
        raise ValueError(
            "ProteinMPNN loss has no designable residue to score: residue_mask & design_positions is "
            "empty for every structure in this batch"
        )
    return (per_residue * mask).sum() / scope


def criterion_factory(model: Any, processor: Any) -> Any:
    r"""Bind a ProteinMPNN Runner only when Model and Processor use the same graph."""

    if model.config.ca_only != processor.config.ca_only:
        raise ValueError("ProteinMPNN Model and Processor must use the same ca_only value")
    return criterion
