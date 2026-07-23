r"""The MPNN autoregressive sequence-recovery objective.

Wires ProteinMPNN / LigandMPNN / NA-MPNN's in-repo, unreduced per-token losses
(:func:`ofoldx.models.proteinmpnn.model.smoothed_loss` /
:func:`~ofoldx.models.proteinmpnn.model.nll_loss`) into one scalar training loss
via the caller-owned masked mean (DESIGN:3613-3620): the model exposes the
unreduced primitive, the Runner-side objective owns the reduction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...models.proteinmpnn.model import smoothed_loss
from ..registry import OBJECTIVES


@OBJECTIVES.register("mpnn")
class MpnnObjective:
    r"""Label-smoothed cross-entropy of the recovered sequence, masked-meaned.

    Contract (``__call__(model, batch) -> (output, loss)``):

    - ``batch`` is a ``ProteinMpnnInput`` (or the NA-MPNN / LigandMPNN analogue)
      carrying the native ``sequence_tokens``, ``valid_mask``, and ``design_mask``.
    - Run the model's autoregressive graph to obtain per-position
      ``log_probabilities`` teacher-forced on ``sequence_tokens``.
    - ``loss`` = masked mean of ``smoothed_loss(sequence_tokens, log_probabilities,
      weight=...)`` over ``valid_mask & design_mask`` (positions actually scored).
    - Return ``(log_probabilities, loss)``.

    The label-smoothing weight is a construction-time hyperparameter on this objective,
    not a Model or Processor config field.
    """

    def __init__(self, *, label_smoothing: float = 0.1) -> None:
        self.label_smoothing = label_smoothing

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Tensor, Tensor]:
        r"""Reduce one batched MPNN input to its recovered-sequence loss.

        ``model(batch)`` runs the family's teacher-forced training graph: it
        encodes the backbone, draws a fixed-first causal decoding order, and
        returns sequence-conditioned logits over the random order. The vocabulary log probabilities feed the model's
        own unreduced label-smoothed cross entropy, which this objective
        masked-means over the positions that are both present and designed.
        """

        log_probabilities = model(batch).log_probabilities
        per_token = smoothed_loss(
            batch.sequence_tokens,
            log_probabilities,
            weight=self.label_smoothing,
        )
        mask = (batch.valid_mask & batch.design_mask).to(dtype=per_token.dtype)
        loss = (per_token * mask).sum() / mask.sum().clamp_min(torch.finfo(per_token.dtype).tiny)
        return log_probabilities, loss
