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
gRNAde training objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..targets import SequenceTarget
from .model import GRNADE_CLASSES, GrnadeOutput


class GrnadeLoss(nn.Module):
    r"""
    Cross entropy over the designable residues of a predicted class, on the four predicted classes (upstream's axis,
    so label smoothing spreads over them only).

    Args:
        label_smoothing: Uniform label-smoothing mass distributed over the predicted classes.
    """

    def __init__(self, *, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred: GrnadeOutput, target: SequenceTarget, *, mask: Tensor | None = None) -> Tensor:
        r"""Score aligned sequence targets within an optional task domain.

        Args:
            pred: Predicted canonical-class logits.
            target: Aligned classes and their knownness mask.
            mask: Optional scoring domain, intersected with target knownness.

        Returns:
            Mean sequence loss over the selected rows.
        """
        classes = torch.tensor(GRNADE_CLASSES, device=target.tokens.device)
        matches = target.tokens.unsqueeze(-1) == classes
        known_design = target.mask & matches.any(dim=-1)
        if mask is not None:
            known_design = known_design & mask.bool()
        return F.cross_entropy(
            pred.logits[known_design][:, classes],
            matches[known_design].long().argmax(dim=-1),
            label_smoothing=self.label_smoothing,
        )
