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

"""Behavior of ofoldx.models.grnade.objective."""

import torch
import torch.nn.functional as F

from ofoldx.data.tokenization import TOKENS
from ofoldx.models.grnade import GrnadeLoss, GrnadeOutput, GrnadeProcessor
from ofoldx.models.grnade.model import GRNADE_CLASSES


def test_loss_on_predicted_classes(rna):
    structure = rna(2)
    prepared, target = GrnadeProcessor(dtype=torch.float64).prepare(structure, target=structure)
    logits = torch.full((2, len(TOKENS)), -torch.inf, dtype=torch.float64)
    logits[:, list(GRNADE_CLASSES)] = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, 1.0]], dtype=torch.float64)
    loss = GrnadeLoss()(GrnadeOutput(logits=logits, batch_size=[]), target, mask=prepared.features.design_mask)
    expected = F.cross_entropy(logits[:, list(GRNADE_CLASSES)], torch.zeros(2, dtype=torch.long), label_smoothing=0.05)
    torch.testing.assert_close(loss, expected)
