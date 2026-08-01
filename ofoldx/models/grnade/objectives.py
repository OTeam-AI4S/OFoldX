r"""
gRNAde training objective.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .model import GRNADE_UNKNOWN_TOKEN, GrnadeInput, GrnadeOutput


def criterion(
    pred: GrnadeOutput,
    target: GrnadeInput,
    *,
    label_smoothing: float = 0.05,
) -> Tensor:
    r"""Return cross entropy over designable canonical RNA residues."""

    known_design = target.design_mask & (target.sequence_tokens != GRNADE_UNKNOWN_TOKEN)
    return F.cross_entropy(
        pred.logits[known_design],
        target.sequence_tokens[known_design],
        label_smoothing=label_smoothing,
    )
