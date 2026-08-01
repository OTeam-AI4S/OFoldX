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

"""Behavior of ofoldx.models.boltz1.pipeline."""

from __future__ import annotations

import numpy as np
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model, Boltz1Pipeline, Boltz1Processor
from ofoldx.models.boltz1.model import Boltz1ConfidenceOutput, Boltz1SamplingWithConfidenceOutput


def test_decoded_predictions(peptide):
    processor = Boltz1Processor(dtype=torch.float64)
    prepared = processor.prepare_input(peptide("ALA", "GLY"))
    atoms, tokens, samples = len(prepared.context.slots.atom), len(prepared.context.slots.layout.token_component), 2
    coordinates = torch.arange(samples * atoms * 3, dtype=torch.float64).reshape(1, samples, atoms, 3)
    score = torch.zeros(1, samples, dtype=torch.float64)
    confidence = Boltz1ConfidenceOutput(
        coordinates=coordinates,
        distogram_logits=torch.zeros(1, tokens, tokens, 4),
        plddt_logits=torch.zeros(1, samples, tokens, 4),
        pde_logits=torch.zeros(1, samples, tokens, tokens, 4),
        resolved_logits=torch.zeros(1, samples, tokens, 2),
        plddt=torch.full((1, samples, tokens), 50.0, dtype=torch.float64),
        pde=torch.full((1, samples, tokens, tokens), 16.0, dtype=torch.float64),
        mean_plddt=score,
        complex_iplddt=score,
        complex_pde=score,
        complex_ipde=score,
        confidence_score=score,
        batch_size=[1],
    )
    output = Boltz1SamplingWithConfidenceOutput(coordinates, confidence.distogram_logits, confidence)
    with torch.device("meta"):
        model = Boltz1Model(Boltz1Config(trunk={"num_pairformer_blocks": 0}))
    result = Boltz1Pipeline(model=model, processor=processor).postprocess(prepared, output)
    structure = result.predictions[1].structure
    np.testing.assert_array_equal(structure.coordinates[prepared.context.slots.atom], coordinates[0, 1].numpy())
    assert (structure.coordinate_evidence[prepared.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
    assert result.predictions[0].metrics["plddt"].axis == (ComponentRef("A", 0), ComponentRef("A", 1))
