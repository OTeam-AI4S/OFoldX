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

"""Behavior of ofoldx.models.grnade.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.grnade import GrnadeProcessor, GrnadeV032Processor
from ofoldx.models.grnade.config import GrnadeV032ProcessorConfig

FEATURES = ("node_scalar_features", "node_vector_features", "edge_scalar_features", "edge_vector_features")


def translated(structure, shift):
    every = np.ones(len(structure.coordinates), dtype=bool)
    atoms = np.flatnonzero(structure.coordinate_mask)
    moved = structure.coordinates[atoms] + np.asarray(shift)
    return replace_coordinates(structure, atoms, moved, scope=every, evidence=CoordinateEvidence.OBSERVED)[0]


class TestPrepare:
    def test_rows_and_slots(self, rna):
        structure = rna(3)
        prepared = GrnadeProcessor(dtype=torch.float64).prepare_input(structure)
        names = structure.system.text("atom", structure.system.atom_symbol[prepared.context.slots.atom])
        assert names.tolist() == ["P", "C4'", "N9"] * 3
        assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX["RNA_A"]] * 3
        assert prepared.context.scope.rule == "none"

    def test_rigid_translation(self, rna):
        structure = rna(3)
        processor = GrnadeProcessor(dtype=torch.float64)
        first = processor.prepare_input(structure).features
        second = processor.prepare_input(translated(structure, [31.0, -17.0, 9.0])).features
        for name in FEATURES:
            torch.testing.assert_close(getattr(first, name), getattr(second, name), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(first.edge_index, second.edge_index)

    def test_secondary_structure_edge(self, rna):
        input = GrnadeProcessor(dtype=torch.float64).prepare_input(rna(3), secondary_structure="(.)").features
        edge = ((input.edge_index[0] == 0) & (input.edge_index[1] == 2)).nonzero().flatten()
        assert input.edge_scalar_features[edge, 0, :4].argmax(-1).tolist() == [3]

    def test_conformations_differ(self, rna):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            GrnadeProcessor().prepare_input((rna(3), rna(2)))


class TestV032:
    def test_unobserved_row_dropped(self, rna, unobserve):
        structure = rna(3)
        atom = int(structure.system.atom_index([AtomSite("A", 1, "C4'")])[0])
        processor = GrnadeV032Processor(GrnadeV032ProcessorConfig(max_num_conformations=2), dtype=torch.float64)
        prepared = processor.prepare_input(unobserve(structure, atom), generator=np.random.default_rng(0))
        assert prepared.features.component_index.tolist() == [0, 2]
        assert prepared.features.conformation_mask.shape == (2, 2)


class TestSequenceTarget:
    @pytest.mark.parametrize("processor", [GrnadeProcessor, GrnadeV032Processor])
    def test_truth_classes_do_not_replace_input(self, rna, polymers, processor):
        source = rna(3)
        truth = Structure.unobserved(polymers(("A", "R", ("G", "A", "U"))))
        processor = processor()
        baseline = processor.prepare_input(source)
        prepared, target = processor.prepare(source, target=truth)
        assert target.tokens.tolist() == [TOKEN_INDEX["RNA_" + token] for token in "GAU"]
        assert target.mask.all()
        torch.testing.assert_close(prepared.features.to_dict(), baseline.features.to_dict())
        assert prepared.context.structure is source
