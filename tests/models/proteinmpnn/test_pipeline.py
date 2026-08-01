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

"""Behavior of ofoldx.models.proteinmpnn.pipeline."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.design import PositionSelection
from ofoldx.models.mpnn_packer import (
    MpnnPackerConfig,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
    MpnnPackerProcessorConfig,
)
from ofoldx.models.proteinmpnn import (
    ProteinMpnnConfig,
    ProteinMpnnDesignAndPackPipeline,
    ProteinMpnnModel,
    ProteinMpnnPipeline,
    ProteinMpnnProcessor,
    ProteinMpnnScorePipeline,
)

SMALL = {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1, "num_neighbors": 2, "dropout": 0.0}


@pytest.fixture
def designer():
    torch.manual_seed(79)
    return ProteinMpnnPipeline(
        model=ProteinMpnnModel(ProteinMpnnConfig(SMALL)).eval(), processor=ProteinMpnnProcessor()
    )


def only(component: int, identity: str) -> AllowedResidueIdentities:
    return AllowedResidueIdentities((ComponentRef("A", component),), frozenset({identity}))


def at(component: int) -> PositionSelection:
    return PositionSelection(components=(ComponentRef("A", component),))


def ids(structure) -> list[str]:
    return structure.system.text("component", structure.system.component_symbol).tolist()


def test_complete_call_dispatches_to_overridden_task():
    class CustomPipeline(ProteinMpnnPipeline):
        def design(self, structure, **options):
            return structure, options

    pipeline = CustomPipeline(model=ProteinMpnnModel(ProteinMpnnConfig(SMALL)), processor=ProteinMpnnProcessor())
    assert pipeline("input", seed=7) == pipeline.design("input", seed=7)
    assert inspect.signature(ProteinMpnnPipeline.__call__) == inspect.signature(ProteinMpnnPipeline.design)
    assert inspect.signature(pipeline) == inspect.signature(pipeline.design)


class TestDesign:
    def test_encoded_source_retained(self, designer, peptide, library):
        result = designer.design(
            peptide("ALA", "MSE", "GLY"), design=at(1), conditions=(only(1, "MET"),), seed=19, library=library
        )
        assert ids(result.candidates[0].target)[1] == "MSE"
        assert result.records[0].positions[0].basis == "retained_encoded"

    def test_replacement(self, designer, peptide, library):
        result = designer.design(
            peptide("ALA", "GLY"), design=at(0), conditions=(only(0, "LYS"),), seed=19, library=library
        )
        assert ids(result.candidates[0].target)[0] == "LYS"
        assert 0 in result.records[0].replaced().tolist()

    def test_score_scopes(self, designer, peptide):
        scorer = ProteinMpnnScorePipeline(model=designer.model, processor=designer.processor)
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        result = scorer.score(peptide("ALA", "GLY"), design=selection, num_decoding_orders=2, seed=29)
        assert result.scores["designed_sequence_nll"].scope == frozenset({ComponentRef("A", 1)})
        assert len(result.scores["global_sequence_nll"].evaluations) == 2


class TestDesignAndPack:
    def test_replaced_positions_packed(self, designer, peptide, library):
        packer = MpnnPackerPipeline(
            model=MpnnPackerModel(MpnnPackerConfig({**SMALL, "edge_size": 8, "num_context_atoms": 1})).eval(),
            processor=MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1)),
        )
        workflow = ProteinMpnnDesignAndPackPipeline(designer=designer, packer=packer)
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        result = workflow.design(
            peptide("ALA", "GLY", "ALA"),
            design=selection,
            conditions=(only(1, "LYS"),),
            num_packs_per_candidate=1,
            num_denoising_steps=1,
            num_torsion_samples=2,
            seed=43,
            library=library,
        )
        packed = result.packs[0].predictions[0].structure
        system = packed.system
        start, stop = system.component_atom_ptr[1], system.component_atom_ptr[2]
        names = system.text("atom", system.atom_symbol[start:stop])
        evidence = packed.coordinate_evidence[start:stop]
        assert (evidence[np.isin(names, ["CB", "NZ"])] == CoordinateEvidence.GENERATED).all()
