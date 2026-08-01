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
AlphaFold 2 training objectives.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vocabulary import masked_msa_targets


class AlphaFold2MaskedMsaLoss(nn.Module):
    r"""
    Score the released masked-MSA reconstruction head.

    AlphaFold 2 averages the token cross entropy within each example over the
    BERT mask, then averages the examples. The realized labels and mask belong to
    the output because MSA corruption is sampled inside the Model.
    """

    def forward(self, pred: Any, target: Any) -> Tensor:
        del target
        logits = pred.masked_msa_logits
        probabilities = F.embedding(pred.true_msa, masked_msa_targets(logits))
        mask = pred.bert_mask
        logits = torch.where(mask[..., None] != 0, logits, 0.0)
        per_token = F.cross_entropy(
            logits.movedim(-1, 1),
            probabilities.movedim(-1, 1),
            reduction="none",
        )
        mask = mask.to(torch.promote_types(per_token.dtype, mask.dtype))
        scope = mask.sum(dim=(-2, -1))
        denominator = torch.where(scope > 0, scope + 1e-8, 1.0)
        per_example = (per_token * mask).sum(dim=(-2, -1)) / denominator
        return per_example.mean()
