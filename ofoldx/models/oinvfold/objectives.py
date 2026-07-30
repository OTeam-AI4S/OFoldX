r"""
OInvFold polymer training objective.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor


def masked_nll_criterion(pred: Any, target: Any) -> Tensor:
    r"""Average released polymer negative log likelihood over designable nodes."""

    sequence_tokens = target.sequence_tokens
    if sequence_tokens is None:
        raise ValueError("OInvFold polymer training requires canonical sequence tokens")
    return F.nll_loss(
        pred.log_probabilities[target.design_mask],
        sequence_tokens[target.design_mask],
    )


def criterion_factory(model: Any, processor: Any) -> Any:
    r"""Return the polymer criterion; it has no Model- or Processor-bound values."""

    del model, processor
    return masked_nll_criterion
