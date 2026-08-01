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

"""Behavior of ofoldx.models.oinvfold.objective."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import UnsupportedError
from ofoldx.models.oinvfold import OInvFoldLoss, OInvFoldOutput, OInvFoldPolymerProcessor


@pytest.fixture
def target(peptide):
    structure = peptide("ALA", "GLY", "ALA")
    prepared, target = OInvFoldPolymerProcessor().prepare(structure, target=structure)
    mask = target.mask.clone()
    mask[0] = False
    return type(target)(target.tokens, mask)


def test_designed_rows_scored(target, polymer_model):
    trained = polymer_model.trained_classes
    logits = torch.linspace(-1.0, 1.0, len(TOKENS)).expand(3, -1).masked_fill(~trained, -torch.inf)
    loss = OInvFoldLoss()(OInvFoldOutput(logits=logits), target)
    picked = logits.log_softmax(-1)[[1, 2], target.tokens[1:]]
    torch.testing.assert_close(loss, -picked.mean())


def test_unpredicted_class_refused(target):
    logits = torch.zeros(3, len(TOKENS))
    logits[1, target.tokens[1]] = -torch.inf
    with pytest.raises(UnsupportedError, match="model_capacity"):
        OInvFoldLoss()(OInvFoldOutput(logits=logits), target)
