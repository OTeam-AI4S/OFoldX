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

"""Behavior of ofoldx.models.nampnn.pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import ComponentRef
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.design import PositionSelection
from ofoldx.models.nampnn import NaMpnnPipeline, NaMpnnProcessor, NaMpnnScorePipeline


@pytest.fixture
def designer(model):
    return NaMpnnPipeline(model=model, processor=NaMpnnProcessor())


def only(instance: str, component: int, identity: str) -> AllowedResidueIdentities:
    return AllowedResidueIdentities((ComponentRef(instance, component),), frozenset({identity}))


def ids(structure) -> list[str]:
    return structure.system.text("component", structure.system.component_symbol).tolist()


class TestDesign:
    def test_protein_replacement(self, designer, complex_structure, library):
        selection = PositionSelection(components=(ComponentRef("B", 0),))
        result = designer.design(
            complex_structure, design=selection, conditions=(only("B", 0, "LYS"),), seed=19, library=library
        )
        assert ids(result.candidates[0].target)[2] == "LYS"
        assert result.records[0].replaced().tolist() == [2]

    def test_shared_rna_retained(self, designer, complex_structure, library):
        result = designer.design(
            complex_structure,
            nucleic_acid_only=True,
            conditions=(only("A", 0, "A"), only("A", 1, "A")),
            seed=7,
            library=library,
        )
        assert ids(result.candidates[0].target)[:2] == ["A", "A"]
        assert {position.basis for position in result.records[0].positions} == {"retained_unchanged"}

    def test_specificity_profile(self, designer, complex_structure):
        result = designer.specificity_profile(complex_structure, num_samples=2, seed=3)
        total = sum(metric.values for metric in result.metrics.values())
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)
        assert result.metrics["adenine_probability"].axis == (ComponentRef("A", 0), ComponentRef("A", 1))


class TestScore:
    def test_score_scopes(self, designer, complex_structure):
        scorer = NaMpnnScorePipeline(model=designer.model, processor=designer.processor)
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        result = scorer.score(complex_structure, design=selection, num_decoding_orders=2, seed=29)
        assert result.scores["designed_sequence_nll"].scope == frozenset({ComponentRef("A", 1)})
        assert len(result.scores["global_sequence_nll"].evaluations) == 2
