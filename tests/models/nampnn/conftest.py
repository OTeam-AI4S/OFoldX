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

import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.models.nampnn import NaMpnnConfig, NaMpnnModel


@pytest.fixture
def complex_structure(library, placed):
    """Chain ``A``: RNA ``A A``; chain ``B``: protein ``ALA GLY``; placed."""
    entities = {
        "R": PolymerSequence("polyribonucleotide", ("A", "A")),
        "P": PolymerSequence("polypeptide(L)", ("ALA", "GLY")),
    }
    system, _, _ = build_system(Recipe(entities, {"A": "R", "B": "P"}), library)
    return placed(system)


@pytest.fixture
def model():
    torch.manual_seed(83)
    values = {"hidden_size": 8, "feature_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1}
    values.update(num_neighbors=3, dropout=0.0, positional_embedding_size=4, num_rbf=2, include_predicted_na_n=False)
    return NaMpnnModel(NaMpnnConfig(values)).eval()
