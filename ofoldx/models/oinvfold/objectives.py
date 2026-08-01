r"""
OInvFold polymer training objective.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .model import OInvFoldOutput, OInvFoldPolymerInput


def criterion(pred: OInvFoldOutput, target: OInvFoldPolymerInput) -> Tensor:
    r"""Average released polymer negative log likelihood over designable nodes."""

    sequence_tokens = target.sequence_tokens
    if sequence_tokens is None:
        raise ValueError("OInvFold polymer training requires canonical sequence tokens")
    return F.nll_loss(
        pred.log_probabilities[target.design_mask],
        sequence_tokens[target.design_mask],
    )
