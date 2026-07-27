r"""
LigandMPNN training objectives.

LigandMPNN is protein sequence design with ligand context; its recovery loss is the same uniform
label-smoothed cross entropy as ProteinMPNN.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from ...modules.collate import collate_single
from ...modules.mpnn import label_smoothed_nll_loss
from ...modules.mpnn import nll_loss as _nll_loss


def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor:
    r"""Return LigandMPNN's unreduced per-token negative log likelihood."""

    return _nll_loss(sequence_tokens, log_probabilities)


def smoothed_loss(sequence_tokens: Tensor, log_probabilities: Tensor, *, weight: float = 0.1) -> Tensor:
    r"""Return LigandMPNN's unreduced uniformly label-smoothed loss."""

    return label_smoothed_nll_loss(sequence_tokens, log_probabilities, weight=weight)


def criterion(pred: Any, target: Any, *, label_smoothing: float = 0.1) -> Tensor:
    r"""DanLing-shaped sequence-recovery loss over the designable residues."""

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    per_residue = smoothed_loss(target.sequence_tokens, log_probabilities, weight=label_smoothing)
    mask = (target.residue_mask & target.design_positions).to(per_residue.dtype)
    return (per_residue * mask).sum() / mask.sum()


def loss_tokens(target: Any) -> int:
    r"""Local count of designable residues -- DanLing's per-token loss normalizer."""

    return int((target.residue_mask & target.design_positions).sum())


def collate(batch: list[Any]) -> dict[str, Any]:
    r"""
    Collate to DanLing's ``{"input", "target", "loss_normalizer"}`` batch. ``LigandMpnnInput`` carries a
    single ``structure`` and its ligand context, so LigandMPNN trains at batch_size=1 (multi-structure
    batching via gradient accumulation weighted by the token count).
    """

    batched = collate_single(batch)
    return {"input": batched, "target": batched, "loss_normalizer": loss_tokens(batched)}
