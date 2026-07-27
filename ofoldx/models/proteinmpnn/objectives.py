r"""
ProteinMPNN training objectives.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ...modules.collate import pad_and_collate
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


def criterion(pred: Any, target: Any, *, label_smoothing: float = 0.1) -> Tensor:
    r"""
    DanLing-shaped loss: score the model output's logits for the observed sequence and reduce over the
    designable residues (``residue_mask & design_positions``).

    Args:
        pred: The model output (``ProteinMpnnOutput`` with ``logits``).
        target: The batched ``ProteinMpnnInput`` carrying ``sequence_tokens`` and the masks.

    An empty design scope gives a NaN mean -- surfaced, not hidden by a fallback.
    """

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    per_residue = smoothed_loss(target.sequence_tokens, log_probabilities, weight=label_smoothing)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    return (per_residue * mask).sum() / mask.sum()


def loss_tokens(target: Any) -> int:
    r"""Local count of designable residues -- DanLing's per-token loss normalizer for one micro-batch."""

    return int((target.residue_mask & target.design_positions).sum())


def collate(batch: list[Any]) -> dict[str, Any]:
    r"""
    Collate to DanLing's ``{"input", "target", "loss_normalizer"}`` batch. ``input`` and ``target`` are
    the same batched ``ProteinMpnnInput`` (it carries both the graph features and the target sequence).
    """

    batched = pad_and_collate(batch, token_pad_id=0)
    return {"input": batched, "target": batched, "loss_normalizer": loss_tokens(batched)}
