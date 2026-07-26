r"""
ProteinMPNN training objectives.
"""

from __future__ import annotations

from torch import Tensor

from ...modules.mpnn import label_smoothed_nll_loss
from ...modules.mpnn import nll_loss as _nll_loss


def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor:
    r"""Return ProteinMPNN's unreduced per-token negative log likelihood."""

    return _nll_loss(sequence_tokens, log_probabilities)


def smoothed_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor:
    r"""Return ProteinMPNN's unreduced uniformly label-smoothed loss."""

    return label_smoothed_nll_loss(
        sequence_tokens,
        log_probabilities,
        weight=weight,
    )
