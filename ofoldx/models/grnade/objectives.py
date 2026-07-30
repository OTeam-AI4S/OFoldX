r"""
gRNAde training objective.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from .model import GRNADE_UNKNOWN_TOKEN


def sequence_criterion(
    pred: Any,
    target: Any,
    *,
    label_smoothing: float,
) -> Tensor:
    r"""Cross entropy over designable residues with canonical RNA identities."""

    known_design = target.design_mask & (target.sequence_tokens != GRNADE_UNKNOWN_TOKEN)
    return F.cross_entropy(
        pred.logits[known_design],
        target.sequence_tokens[known_design],
        label_smoothing=label_smoothing,
    )


def criterion_factory(model: Any, processor: Any) -> Any:
    r"""Use the released ``0.05`` training smoothing and unsmoothed evaluation loss."""

    del processor

    def criterion(pred: Any, target: Any) -> Tensor:
        return sequence_criterion(
            pred,
            target,
            label_smoothing=0.05 if model.training else 0.0,
        )

    return criterion
