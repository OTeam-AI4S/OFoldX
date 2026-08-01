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

"""Behavior of ofoldx.models.boltz1.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.boltz1.processor import Boltz1Processor


class TestPrepare:
    def test_prediction_scope(self, peptide):
        prepared = Boltz1Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert prepared.features.residue_type_features.shape[-1] == len(TOKENS)

    def test_designated_leaving_unaddressed(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = Boltz1Processor().prepare_input(structure)
        names = structure.system.text("atom", structure.system.atom_symbol[prepared.context.slots.atom]).tolist()
        assert "OXT" not in names
        assert prepared.features.atom_index.shape[-1] == len(names)

    def test_token_bonds(self, peptide):
        prepared = Boltz1Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        bonds = prepared.features.token_bonds[0]
        assert bonds[:2].sum() == 0  # no polymer junction, as upstream
        assert bonds[2:, 2:].any()  # the ligand's own bonds

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            Boltz1Processor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = Boltz1Processor(dtype=torch.float64)
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure, generator=np.random.default_rng(3))
        tensors = target
        assert tensors.coordinates.shape == (1, len(structure.coordinates), 3)
        assert tensors.chain_span.tolist() == [[[0, len(structure.coordinates)]]]
