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

"""Behavior of ofoldx.models.boltz2.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.boltz2 import (
    Boltz2AffinityConfig,
    Boltz2AffinityHeadConfig,
    Boltz2AffinityModel,
    Boltz2AffinityProcessor,
    Boltz2AffinityScorePipeline,
    Boltz2Model,
    Boltz2Pipeline,
    Boltz2Processor,
)


class TestFold:
    @pytest.mark.parametrize("designed", [(), (ComponentRef("A", 1),)])
    def test_decoded_predictions(self, peptide, boltz2_config, designed):
        processor = Boltz2Processor(dtype=torch.float64)
        prepared = processor.prepare_input(peptide("ALA", "GLY"), designed_positions=designed)
        pipeline = Boltz2Pipeline(model=Boltz2Model(boltz2_config()).double().eval(), processor=processor)
        output = pipeline.forward(
            prepared.features, num_samples=2, num_recycles=1, num_sampling_steps=2, generator=torch.Generator()
        )
        result = pipeline.postprocess(prepared, output)
        structure = result.predictions[1].structure
        np.testing.assert_array_equal(
            structure.coordinates[prepared.context.slots.atom], output.coordinates[0, 1].numpy()
        )
        assert (structure.coordinate_evidence[prepared.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
        assert result.predictions[0].metrics["plddt"].axis == (ComponentRef("A", 0), ComponentRef("A", 1))
        assert ("design_iptm" in result.predictions[0].scores) == bool(designed)


class TestAffinityScore:
    def test_scores_scoped_to_ligand(self, peptide, boltz2_config):
        config = Boltz2AffinityConfig(dict(boltz2_config()))
        head = {"num_pair_update_blocks": 1, "num_distogram_bins": 4, "num_conditioning_transitions": 1}
        head |= {"triangle_attention_head_size": 4, "num_triangle_attention_heads": 1, "dropout": 0.0}
        config.affinity_heads = (Boltz2AffinityHeadConfig(**head), Boltz2AffinityHeadConfig(**head))
        processor = Boltz2AffinityProcessor(dtype=torch.float64)
        prepared = processor.prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        pipeline = Boltz2AffinityScorePipeline(model=Boltz2AffinityModel(config).double().eval(), processor=processor)
        output = pipeline.forward(
            prepared.features, num_samples=1, num_recycles=1, num_sampling_steps=2, generator=torch.Generator()
        )
        result = pipeline.postprocess(prepared, output)
        assert len(result.scores) == 6 and all(score.scope == frozenset({"B"}) for score in result.scores.values())
        selected = output.structure.coordinates[0, int(output.selected_sample_index[0])].numpy()
        np.testing.assert_array_equal(result.target.coordinates[prepared.context.slots.atom], selected)
