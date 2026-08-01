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

"""Behavior of ofoldx.models.oinvfold.pipeline."""

from __future__ import annotations

import json

import pytest
import torch

from ofoldx.artifacts import save_model
from ofoldx.data.molecular import ComponentRef
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.design import PositionSelection
from ofoldx.models.oinvfold import (
    OInvFoldOutput,
    OInvFoldPolymerConfig,
    OInvFoldPolymerModel,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerProcessor,
    OInvFoldPolymerProcessorConfig,
)


@pytest.fixture
def pipeline(polymer_model):
    return OInvFoldPolymerPipeline(model=polymer_model, processor=OInvFoldPolymerProcessor(dtype=torch.float64))


def favouring(pipeline, prepared, token: str) -> OInvFoldOutput:
    rows = len(prepared.features.sequence_tokens)
    logits = torch.zeros(rows, len(TOKENS), dtype=torch.float64)
    logits[:, TOKEN_INDEX[token]] = 10.0
    return OInvFoldOutput(logits=logits.masked_fill(~pipeline.model.trained_classes, -torch.inf), batch_size=[])


def ids(structure) -> list[str]:
    return structure.system.text("component", structure.system.component_symbol).tolist()


def at(component: int) -> PositionSelection:
    return PositionSelection(components=(ComponentRef("A", component),))


def modality_pair(polymer_model, model_modality, processor_modality):
    config = OInvFoldPolymerConfig({**dict(polymer_model.config), "modality": model_modality})
    model = OInvFoldPolymerModel(config).double().eval()
    processor = OInvFoldPolymerProcessor(
        OInvFoldPolymerProcessorConfig(modality=processor_modality), dtype=torch.float64
    )
    return model, processor


@pytest.mark.parametrize(
    ("model_modality", "processor_modality"),
    [("protein", "dna"), ("protein", "rna"), ("dna", "protein"), ("dna", "rna"), ("rna", "protein"), ("rna", "dna")],
)
def test_pipeline_rejects_mismatched_modalities(polymer_model, model_modality, processor_modality):
    model, processor = modality_pair(polymer_model, model_modality, processor_modality)
    with pytest.raises(InvalidRequestError, match="value_invalid.*modality"):
        OInvFoldPolymerPipeline(model=model, processor=processor)


@pytest.mark.parametrize("existing", [False, True])
def test_save_rejects_modality_mismatch_before_writing(tmp_path, polymer_model, existing):
    model, processor = modality_pair(polymer_model, "dna", "rna")
    directory = tmp_path / "repository"
    if existing:
        directory.mkdir()
        (directory / "model.json").write_text("caller-owned metadata\n")
    before = {path.name: path.read_bytes() for path in directory.iterdir()} if existing else None

    with pytest.raises(InvalidRequestError, match="value_invalid.*modality"):
        save_model(directory, model=model, processor=processor)

    if existing:
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    else:
        assert not directory.exists()


def test_load_rejects_mismatched_modality_metadata(tmp_path, polymer_model):
    model, processor = modality_pair(polymer_model, "dna", "dna")
    save_model(tmp_path, model=model, processor=processor)
    metadata = json.loads((tmp_path / "processor.json").read_text())
    metadata["modality"] = "rna"
    (tmp_path / "processor.json").write_text(json.dumps(metadata))

    with pytest.raises(InvalidRequestError, match="value_invalid.*modality"):
        OInvFoldPolymerPipeline.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    ("modality", "polymer_type", "components", "tokens"),
    [
        ("protein", "polypeptide(L)", ("ALA", "GLY"), ("A", "G")),
        ("dna", "polyribonucleotide", ("A", "U"), ("DNA_A", "DNA_T")),
        ("rna", "polydeoxyribonucleotide", ("DA", "DT"), ("RNA_A", "RNA_U")),
    ],
)
def test_matching_modalities_round_trip_and_prepare(
    tmp_path, polymer_model, chains, modality, polymer_type, components, tokens
):
    model, processor = modality_pair(polymer_model, modality, modality)
    flow = OInvFoldPolymerPipeline(model=model, processor=processor)
    flow.save(tmp_path)
    loaded = OInvFoldPolymerPipeline.from_pretrained(tmp_path)
    prepared = loaded.prepare(chains(polymer_type, *components))
    assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX[token] for token in tokens]
    output = loaded.forward(prepared.features)
    assert torch.isfinite(output.logits[:, loaded.model.trained_classes]).all()


class TestPolymerDesign:
    @pytest.mark.parametrize("use_beam", [False, True])
    def test_replacement(self, pipeline, peptide, library, use_beam):
        prepared = pipeline.prepare(peptide("ALA", "GLY"), design=at(0))
        output = favouring(pipeline, prepared, "K")
        result = pipeline.postprocess(prepared, output, use_beam=use_beam, library=library)
        assert ids(result.best.target)[0] == "LYS"
        assert result.records[0].replaced().tolist() == [0]
        expected = output.logits[0].log_softmax(-1)[TOKEN_INDEX["K"]]
        assert result.best.scores["total_log_probability"].value == pytest.approx(float(expected))

    def test_encoded_source_retained(self, pipeline, peptide, library):
        prepared = pipeline.prepare(peptide("ALA", "MSE", "GLY"), design=at(1))
        result = pipeline.postprocess(prepared, favouring(pipeline, prepared, "M"), library=library)
        assert ids(result.best.target)[1] == "MSE"
        assert result.records[0].positions[0].basis == "retained_encoded"

    def test_beam_keeps_allowed_classes(self, pipeline, peptide, library):
        prepared = pipeline.prepare(peptide("ALA", "GLY"), design=at(0))
        logits = torch.full((2, len(TOKENS)), -torch.inf, dtype=torch.float64)
        logits[:, TOKEN_INDEX["A"]] = 0.0
        output = OInvFoldOutput(logits=logits, batch_size=[])
        result = pipeline.postprocess(prepared, output, use_beam=True, num_beams=3, library=library)
        assert len(result.candidates) == 1

    @pytest.mark.parametrize("use_beam", [False, True])
    def test_no_allowed_class(self, pipeline, peptide, use_beam):
        prepared = pipeline.prepare(peptide("ALA", "GLY"), design=at(0))
        logits = torch.zeros(2, len(TOKENS), dtype=torch.float64)
        logits[0, : TOKEN_INDEX["OTHER"]] = -torch.inf
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            pipeline.postprocess(prepared, OInvFoldOutput(logits=logits, batch_size=[]), use_beam=use_beam)
