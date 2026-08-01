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

"""Behavior of ofoldx.models.oinvfold.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.models.oinvfold import OInvFoldPolymerConfig, OInvFoldPolymerModel, OInvFoldPolymerProcessor


def test_logits_on_released_columns(polymer_model, peptide):
    input = OInvFoldPolymerProcessor(dtype=torch.float64).prepare_input(peptide("ALA", "GLY", "ALA")).features
    with torch.no_grad():
        logits = polymer_model(input, mask_probability=0.0).logits
    assert logits.shape == (3, len(TOKENS))
    assert torch.equal(torch.isinf(logits), ~polymer_model.trained_classes.expand(3, -1))


@pytest.mark.parametrize(("modality", "shared"), [("protein", "RNA"), ("dna", "DNA"), ("rna", "RNA")])
def test_trained_classes_per_modality(modality, shared):
    config = OInvFoldPolymerConfig(modality=modality, hidden_size=8, decoder_hidden_size=8, num_encoder_blocks=1)
    trained = OInvFoldPolymerModel(config).trained_classes
    assert int(trained.sum()) == 25
    assert trained[[TOKEN_INDEX[f"{shared}_{letter}"] for letter in "ACG"]].all()
    assert trained[[TOKEN_INDEX["DNA_T"], TOKEN_INDEX["RNA_U"]]].all()
