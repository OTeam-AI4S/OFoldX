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

"""Behavior of ofoldx.models.rfdiffusion.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.conditions import ExistingSpan, GeneratedSegment
from ofoldx.models.rfdiffusion import (
    RfDiffusionConfig,
    RfDiffusionModel,
    RfDiffusionPipeline,
    RfDiffusionPlacements,
    RfDiffusionProcessor,
    RfDiffusionSamplingOutput,
)


@pytest.fixture
def designer():
    with torch.device("meta"):
        model = RfDiffusionModel(RfDiffusionConfig())
    return RfDiffusionPipeline(model=model, processor=RfDiffusionProcessor())


@pytest.fixture
def candidate(designer, peptide, library):
    r"""One motif row and one generated row, decoded from a sample whose rows are distinct integers."""
    placement = RfDiffusionPlacements(
        {"chain": {"motif": ExistingSpan("A", 0, 1), "new": GeneratedSegment("protein", 1, 1)}}
    )
    prepared = designer.processor.prepare_input(peptide("ALA", "GLY"), placement=placement, library=library)
    rows = len(prepared.features.motif_mask[0])
    output = RfDiffusionSamplingOutput(
        coordinates=torch.arange(rows * 14 * 3, dtype=torch.float32).reshape(rows, 14, 3),
        plddt_fraction=torch.full((rows,), 0.5),
        batch_size=[],
    )
    return prepared, designer.decode_candidate(prepared, output, {})


class TestPrepare:
    @pytest.mark.parametrize(
        "bind", (lambda: {"seed": 5}, lambda: {"generator": torch.Generator().manual_seed(5)}), ids=("seed", "torch")
    )
    def test_controlled_length_stream(self, designer, peptide, library, bind):
        r"""A seed and a caller's generator both make the segment-length stream reproducible."""

        def draw():
            return designer.prepare(peptide("ALA"), library=library, **bind()).lengths.integers(0, 1 << 30)

        assert draw() == draw()


class TestDecode:
    def test_generated_backbone_written(self, candidate):
        prepared, decoded = candidate
        system = prepared.context.structure.system
        nitrogen = int(system.atom_index([AtomSite("chain_0", 1, "N")])[0])
        assert decoded.structure.coordinate_mask[nitrogen]
        assert decoded.structure.coordinates[nitrogen].tolist() == [42.0, 43.0, 44.0]
        assert decoded.structure.coordinate_evidence[nitrogen] == CoordinateEvidence.GENERATED

    def test_unplaced_side_chain(self, candidate):
        prepared, decoded = candidate
        system = prepared.context.structure.system
        carbon = int(system.atom_index([AtomSite("chain_0", 1, "CG")])[0])
        assert not decoded.structure.coordinate_mask[carbon]

    def test_motif_keeps_observation(self, candidate):
        prepared, decoded = candidate
        system = prepared.context.structure.system
        alpha = int(system.atom_index([AtomSite("chain_0", 0, "CA")])[0])
        assert decoded.structure.coordinate_evidence[alpha] == CoordinateEvidence.OBSERVED
        np.testing.assert_allclose(decoded.structure.coordinates[alpha], prepared.context.structure.coordinates[alpha])

    def test_plddt_axis(self, candidate):
        _prepared, decoded = candidate
        assert decoded.metrics["plddt"].axis == (ComponentRef("chain_0", 0), ComponentRef("chain_0", 1))
        assert decoded.scores["mean_plddt"].value == pytest.approx(50.0)
