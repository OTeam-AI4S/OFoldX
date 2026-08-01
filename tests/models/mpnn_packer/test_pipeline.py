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

"""Behavior of ofoldx.models.mpnn_packer.pipeline."""

from __future__ import annotations

import numpy as np
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.design import PositionSelection
from ofoldx.models.mpnn_packer import MpnnPackerPipeline


class TestPack:
    def test_mixture_count_survives_repository_roundtrip(self, tmp_path, small_packer, peptide):
        small_packer.save(tmp_path)
        restored = MpnnPackerPipeline.from_pretrained(tmp_path, backend="eager")
        assert restored.model.config.num_mixtures == 1
        input = restored.prepare(peptide("LYS", "LYS")).features
        coordinates = torch.zeros((1, 2, 14, 3))
        mask = torch.ones((1, 2, 14), dtype=torch.bool)
        output = restored.model(input, atom14_coordinates=coordinates, atom14_mask=mask)
        assert output.mean.shape == (1, 2, 4, 1)

    def test_side_chains_generated(self, small_packer, peptide):
        structure = peptide("LYS", "ALA")
        selection = PositionSelection(components=(ComponentRef("A", 0),))
        result = small_packer.pack(structure, positions=selection, num_denoising_steps=1, num_torsion_samples=1, seed=3)
        packed = result.predictions[0].structure
        names = structure.system.text("atom", structure.system.atom_symbol)
        lysine = np.arange(structure.system.component_atom_ptr[0], structure.system.component_atom_ptr[1])
        side_chain = lysine[np.isin(names[lysine], ["CB", "CG", "CD", "CE", "NZ"])]
        assert (packed.coordinate_evidence[side_chain] == CoordinateEvidence.GENERATED).all()
        backbone = lysine[np.isin(names[lysine], ["N", "CA", "C", "O"])]
        np.testing.assert_array_equal(packed.coordinates[backbone], structure.coordinates[backbone])
        assert "chi1_log_density" in result.predictions[0].metrics

    def test_nothing_packed(self, small_packer, peptide):
        structure = peptide("ALA")
        result = small_packer.pack(structure, positions=PositionSelection(indices=np.zeros(0, dtype=np.int64)))
        assert result.predictions[0].structure is structure
