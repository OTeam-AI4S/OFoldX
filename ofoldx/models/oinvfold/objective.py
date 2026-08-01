# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
OInvFold polymer training objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import RequestRef, fail
from ..processing import model_log_probabilities
from ..targets import SequenceTarget
from .model import OInvFoldOutput


class OInvFoldLoss(nn.Module):
    r"""Average released polymer negative log likelihood over designable nodes."""

    def forward(self, pred: OInvFoldOutput, target: SequenceTarget, *, mask: Tensor | None = None) -> Tensor:
        r"""Score aligned sequence targets within an optional task domain.

        Args:
            pred: Predicted canonical-class logits.
            target: Aligned classes and their knownness mask.
            mask: Optional scoring domain, intersected with target knownness.

        Returns:
            Mean sequence loss over the selected rows.
        """
        mask = target.mask if mask is None else target.mask & mask.bool()
        sequence_tokens = target.tokens[mask]
        logits = pred.logits[mask]
        log_probabilities = model_log_probabilities(
            logits, torch.ones_like(sequence_tokens, dtype=torch.bool), feature="OInvFold scored logits"
        )
        scorable = torch.isfinite(log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1))).all()
        if torch.compiler.is_compiling():
            torch._assert_async(scorable, "OInvFold scored classes must be classes the model predicts")
        elif not bool(scorable):
            fail("model_capacity", RequestRef("target", "a scored class the model does not predict"))
        return F.nll_loss(log_probabilities, sequence_tokens)
