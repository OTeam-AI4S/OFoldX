r"""
Autoregressive sequence-recovery objective for MPNN models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...modules.mpnn import label_smoothed_nll_loss
from ..registry import OBJECTIVES


@OBJECTIVES.register("mpnn")
class MpnnObjective:
    r"""
    Compute label-smoothed sequence-recovery loss.

    Args:
        label_smoothing: Weight assigned to the uniform vocabulary target.
    """

    def __init__(self, *, label_smoothing: float = 0.1) -> None:
        self.label_smoothing = label_smoothing

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Tensor, Tensor]:
        r"""
        Compute logits and sequence-recovery loss for one batch.
        """

        log_probabilities = model(batch).log_probabilities
        per_token = label_smoothed_nll_loss(
            batch.sequence_tokens,
            log_probabilities,
            weight=self.label_smoothing,
        )
        mask = (batch.valid_mask & batch.design_mask).to(dtype=per_token.dtype)
        loss = (per_token * mask).sum() / mask.sum().clamp_min(torch.finfo(per_token.dtype).tiny)
        return log_probabilities, loss
