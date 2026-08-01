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

"""Recovery scores use only native identities available for comparison."""

from __future__ import annotations

import math

import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.molecular import ComponentRef
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.ligandmpnn import LigandMpnnConfig, LigandMpnnModel, LigandMpnnPipeline, LigandMpnnProcessor
from ofoldx.models.ligandmpnn.config import LigandMpnnProcessorConfig
from ofoldx.models.membranempnn import (
    MembraneMpnnConfig,
    MembraneMpnnModel,
    MembraneMpnnPipeline,
    MembraneMpnnProcessor,
    MembraneMpnnProcessorConfig,
)
from ofoldx.models.nampnn import NaMpnnConfig, NaMpnnModel, NaMpnnPipeline, NaMpnnProcessor
from ofoldx.models.nampnn.config import NaMpnnProcessorConfig
from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnModel, ProteinMpnnPipeline, ProteinMpnnProcessor


@pytest.fixture(params=("proteinmpnn", "ligandmpnn", "membranempnn", "nampnn"))
def designer(request):
    torch.manual_seed(83)
    small = {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1, "num_neighbors": 2, "dropout": 0.0}
    if request.param == "proteinmpnn":
        return ProteinMpnnPipeline(model=ProteinMpnnModel(ProteinMpnnConfig(small)), processor=ProteinMpnnProcessor())
    if request.param == "ligandmpnn":
        return LigandMpnnPipeline(
            model=LigandMpnnModel(LigandMpnnConfig(small)),
            processor=LigandMpnnProcessor(LigandMpnnProcessorConfig(num_context_atoms=2)),
        )
    if request.param == "membranempnn":
        return MembraneMpnnPipeline(
            model=MembraneMpnnModel(MembraneMpnnConfig({**small, "label_mode": "per_residue"})),
            processor=MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="per_residue")),
        )
    config = NaMpnnConfig({**small, "feature_size": 8, "positional_embedding_size": 4, "num_rbf": 2})
    return NaMpnnPipeline(model=NaMpnnModel(config), processor=NaMpnnProcessor())


def allowed(index, identity):
    return AllowedResidueIdentities((ComponentRef("A", index),), frozenset({identity}))


@pytest.mark.parametrize(
    ("native", "forced", "expected", "scope"),
    [
        (("UNK", "UNK"), ("ALA", "ALA"), None, ()),
        (("ALA", "UNK"), ("ALA", "ALA"), 1.0, (0,)),
        (("ALA", "GLY"), ("ALA", "ALA"), 0.5, (0, 1)),
        (("MSE", "UNK"), ("MET", "ALA"), 1.0, (0,)),
    ],
)
def test_recovery_is_scoped_to_comparable_native_identities(
    designer, peptide, library, native, forced, expected, scope
):
    conditions = tuple(allowed(index, identity) for index, identity in enumerate(forced))
    result = designer(peptide(*native), conditions=conditions, seed=7, library=library)
    score = result.candidates[0].scores["sequence_recovery"]
    assert score.scope == frozenset(ComponentRef("A", index) for index in scope)
    if expected is None:
        assert math.isnan(score.value)
        with pytest.raises(ValueError, match="NaN"):
            result.best_by("sequence_recovery")
    else:
        assert score.value == expected


@pytest.mark.parametrize(
    ("polymer", "components", "known_identity"),
    [
        ("polyribonucleotide", ("A", "N"), "A"),
        ("polydeoxyribonucleotide", ("DA", "DN"), "DA"),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
def test_nampnn_recovery_excludes_unknown_nucleic_identity(
    library, placed, polymer, components, known_identity, shared
):
    system, _, _ = build_system(Recipe({"P": PolymerSequence(polymer, components)}, {"A": "P"}), library)
    config = NaMpnnConfig(
        hidden_size=8,
        feature_size=8,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        num_neighbors=2,
        dropout=0.0,
        positional_embedding_size=4,
        num_rbf=2,
    )
    pipeline = NaMpnnPipeline(
        model=NaMpnnModel(config), processor=NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=shared))
    )
    result = pipeline(
        placed(system), conditions=(allowed(0, known_identity), allowed(1, known_identity)), seed=7, library=library
    )
    score = result.candidates[0].scores["sequence_recovery"]
    assert score.value == 1.0
    assert score.scope == frozenset({ComponentRef("A", 0)})
