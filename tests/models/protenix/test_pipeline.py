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

"""Behavior of ofoldx.models.protenix.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.protenix import (
    ProtenixConfig,
    ProtenixModel,
    ProtenixPipeline,
    ProtenixPipelineInput,
    ProtenixProcessor,
    ProtenixSamplingOutput,
)


@pytest.fixture
def decoded(peptide):
    processor = ProtenixProcessor(dtype=torch.float64)
    structure = peptide("ALA", "GLY", ligands=("ACT",))
    prepared = processor.prepare_input(structure)
    atoms, tokens, samples = len(prepared.context.slots.atom), len(prepared.context.slots.layout.token_component), 2
    coordinates = torch.arange(samples * atoms * 3, dtype=torch.float64).reshape(1, samples, atoms, 3)
    atom_logits = torch.zeros(1, samples, atoms, 4, dtype=torch.float64)
    pair_logits = torch.zeros(1, samples, tokens, tokens, 4, dtype=torch.float64)
    output = ProtenixSamplingOutput(
        coordinates=coordinates,
        distogram_logits=pair_logits[:, 0].clone(),
        plddt_logits=atom_logits,
        pae_logits=pair_logits,
        pde_logits=pair_logits.clone(),
        resolved_logits=atom_logits[..., :2],
        batch_size=[1],
    )
    with torch.device("meta"):
        model = ProtenixModel(ProtenixConfig(trunk={"num_pairformer_blocks": 0}))
    pipeline = ProtenixPipeline(model=model.set_backend("eager"), processor=processor)
    return structure, prepared, coordinates, pipeline.postprocess(ProtenixPipelineInput(prepared, None), output)


class TestPostprocess:
    def test_coordinates(self, decoded):
        _, prepared, coordinates, result = decoded
        structure = result.predictions[1].structure
        np.testing.assert_array_equal(structure.coordinates[prepared.context.slots.atom], coordinates[0, 1].numpy())
        assert (structure.coordinate_evidence[prepared.context.slots.atom] == CoordinateEvidence.PREDICTED).all()

    def test_axes(self, decoded):
        structure, prepared, _, result = decoded
        prediction = result.predictions[0]
        assert prediction.metrics["chain_plddt"].axis == ("A", "B")
        assert len(prediction.metrics["plddt"].axis) == len(prepared.context.slots.atom)
        tokens = prediction.pair_metrics["pae"].row_axis
        assert tokens[:2] == (ComponentRef("A", 0), ComponentRef("A", 1))
        assert all(isinstance(token, AtomSite) and token.instance == "B" for token in tokens[2:])

    def test_ranking_score(self, decoded):
        _, _, _, result = decoded
        assert set(result.predictions[0].scores) >= {"ranking_score", "ptm", "iptm", "has_clash"}
