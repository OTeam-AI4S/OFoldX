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

"""Behavior of ofoldx.models.esmfold2.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.esmfold2 import (
    EsmFold2DesignPipeline,
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalPipeline,
    EsmFold2ExperimentalProcessor,
    EsmFold2Model,
    EsmFold2Pipeline,
    EsmFold2Processor,
    EsmFold2ProcessorConfig,
)
from ofoldx.models.esmfold2.model import (
    EsmFold2ExperimentalConfidenceOutput,
    EsmFold2ExperimentalSamplingOutput,
    EsmFold2ExperimentalSamplingWithConfidenceOutput,
    EsmFold2SamplingWithConfidenceOutput,
)


@pytest.fixture
def design_pipeline(esm_config):
    with torch.device("meta"):
        critic = EsmFold2ExperimentalPipeline(
            model=EsmFold2ExperimentalModel(esm_config(True)),
            processor=EsmFold2ExperimentalProcessor(),
        )
    return EsmFold2DesignPipeline(**{spec.role: critic for spec in EsmFold2DesignPipeline.component_layout})


class TestFoldPrepare:
    def test_sequence_input(self, esm_config, library):
        with torch.device("meta"):
            model = EsmFold2Model(esm_config(False))
        pipeline = EsmFold2Pipeline(
            model=model, processor=EsmFold2Processor(EsmFold2ProcessorConfig(msa_conditioning=False))
        )
        prepared = pipeline.prepare("AG", library=library)
        assert prepared.features.residue_type_ids[0].tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["G"]]
        assert not prepared.context.structure.coordinate_evidence.any()


@pytest.mark.parametrize("experimental", [False, True])
def test_decoded_predictions(experimental, peptide, esm_config):
    processor_type = EsmFold2ExperimentalProcessor if experimental else EsmFold2Processor
    processor = processor_type(processor_type.config_class(msa_conditioning=False), dtype=torch.float64)
    structure = peptide("ALA", "GLY")
    prepared = processor.prepare_input(structure)
    atoms, tokens, samples = len(prepared.context.slots.atom), len(prepared.context.slots.layout.token_component), 2
    coordinates = torch.arange(samples * atoms * 3, dtype=torch.float64).reshape(1, samples, atoms, 3)
    score = torch.zeros(1, samples, dtype=torch.float64)
    confidence = {
        "plddt_logits": torch.zeros(1, samples, atoms, 4),
        "plddt": torch.full((1, samples, tokens), 50.0, dtype=torch.float64),
        "plddt_per_atom": torch.full((1, samples, atoms), 50.0, dtype=torch.float64),
        "plddt_ca": torch.full((1, samples, tokens), 50.0, dtype=torch.float64),
        "mean_plddt": score,
        "complex_iplddt": score,
        "pae_logits": torch.zeros(1, samples, tokens, tokens, 4),
        "pae": torch.full((1, samples, tokens, tokens), 8.0, dtype=torch.float64),
        "ptm": score,
        "iptm": score,
        "chain_pair_iptm": torch.zeros(1, samples, 1, 1, dtype=torch.float64),
        "chain_pair_mean_tm": torch.full((1, samples, 1, 1), 0.375, dtype=torch.float64),
    }
    distogram = torch.zeros(1, tokens, tokens, 4)
    if experimental:
        output = EsmFold2ExperimentalSamplingWithConfidenceOutput(
            EsmFold2ExperimentalSamplingOutput(coordinates=coordinates, distogram_logits=distogram, batch_size=[1]),
            EsmFold2ExperimentalConfidenceOutput(**confidence, batch_size=[1]),
        )
    else:
        output = EsmFold2SamplingWithConfidenceOutput(
            coordinates=coordinates,
            distogram_logits=distogram,
            pde_logits=torch.zeros(1, samples, tokens, tokens, 4),
            pde=torch.full((1, samples, tokens, tokens), 8.0, dtype=torch.float64),
            resolved_logits=torch.zeros(1, samples, atoms, 2),
            **confidence,
        )
    model_type = EsmFold2ExperimentalModel if experimental else EsmFold2Model
    with torch.device("meta"):
        model = model_type(esm_config(experimental))
    pipeline_type = EsmFold2ExperimentalPipeline if experimental else EsmFold2Pipeline
    result = pipeline_type(model=model, processor=processor).postprocess(prepared, output)
    decoded = result.predictions[1].structure
    np.testing.assert_array_equal(decoded.coordinates[prepared.context.slots.atom], coordinates[0, 1].numpy())
    assert (decoded.coordinate_evidence[prepared.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
    assert result.predictions[0].metrics["plddt"].axis == (ComponentRef("A", 0), ComponentRef("A", 1))
    assert result.predictions[0].pair_metrics["chain_pair_iptm"].row_axis == ("A",)
    mean_tm = result.predictions[0].pair_metrics["chain_pair_mean_tm"]
    assert mean_tm.row_axis == mean_tm.column_axis == ("A",)
    np.testing.assert_array_equal(mean_tm.values, [[0.375]])


class TestDesignPrepare:
    def test_prompt_defaults(self, design_pipeline):
        prepared = design_pipeline.prepare("AG", num_binder_residues=3)
        assert prepared.prompt == "###"
        assert prepared.target_sequence == "AG"
        assert prepared.language_model_weight == pytest.approx(0.15)

    def test_antibody_weight(self, design_pipeline):
        prepared = design_pipeline.prepare("AG", num_binder_residues=2, is_antibody=True)
        assert prepared.language_model_weight == pytest.approx(0.05)

    def test_explicit_non_antibody_does_not_require_hmmer(self, design_pipeline, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))
        prepared = design_pipeline.prepare("AG", num_binder_residues=2, is_antibody=False)
        assert not prepared.is_antibody
        assert prepared.language_model_weight == pytest.approx(0.15)

    @pytest.mark.parametrize(
        "arguments,code",
        [
            ({"num_binder_residues": 0}, "value_invalid"),
            ({"num_binder_residues": 2, "num_candidates": 0}, "value_invalid"),
            ({"num_binder_residues": 2, "binder_prompt": "A"}, "value_invalid"),
            ({"num_binder_residues": 2, "binder_prompt": "AB"}, "request_invalid_sequence"),
        ],
    )
    def test_refused_requests(self, design_pipeline, arguments, code):
        with pytest.raises(InvalidRequestError, match=code):
            design_pipeline.prepare("AG", is_antibody=False, **arguments)

    def test_target_sequence_symbols(self, design_pipeline):
        with pytest.raises(InvalidRequestError, match="request_invalid_sequence"):
            design_pipeline.prepare("AB", num_binder_residues=2, is_antibody=False)
