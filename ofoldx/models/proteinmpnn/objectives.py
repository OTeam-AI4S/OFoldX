r"""
ProteinMPNN training objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

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
    num_classes = log_probabilities.shape[-1]
    target_distribution = F.one_hot(target.sequence_tokens, num_classes).to(log_probabilities.dtype)
    target_distribution = target_distribution + label_smoothing / num_classes
    target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True)
    per_residue = -(target_distribution * log_probabilities).sum(dim=-1)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    # A batch with no designable residue has no loss to report, so the value is NaN. The NaN is selected
    # in rather than divided into: a literal 0/0 here would backpropagate NaN into every parameter
    # through the masked product, whereas dividing by a safe count and applying ``torch.where``
    # afterwards routes zero gradient into the empty branch.
    scope = mask.sum()
    value = (per_residue * mask).sum() / scope.clamp_min(1)
    return torch.where(scope > 0, value, value.new_full((), float("nan")))
