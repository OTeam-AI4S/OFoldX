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

"""Behavior of ofoldx.models.membranempnn.pipeline."""

from __future__ import annotations

import pytest

from ofoldx.data.molecular import ComponentRef
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.design import PositionSelection
from ofoldx.models.membranempnn import MembraneMpnnScorePipeline


def test_replacement(designer, peptide, library):
    condition = AllowedResidueIdentities((ComponentRef("A", 0),), frozenset({"LYS"}))
    result = designer.design(
        peptide("ALA", "GLY"), membrane_labels=2, conditions=(condition,), seed=19, library=library
    )
    target = result.candidates[0].target
    assert target.system.text("component", target.system.component_symbol)[0] == "LYS"
    assert 0 in result.records[0].replaced().tolist()


@pytest.mark.parametrize("use_sequence", [True, False])
def test_score_scopes(designer, peptide, use_sequence):
    scorer = MembraneMpnnScorePipeline(model=designer.model, processor=designer.processor)
    result = scorer.score(
        peptide("ALA", "GLY"),
        membrane_labels={ComponentRef("A", 0): 2},
        design=PositionSelection(components=(ComponentRef("A", 1),)),
        use_sequence=use_sequence,
        num_decoding_orders=2,
        seed=29,
    )
    assert result.scores["designed_sequence_nll"].scope == frozenset({ComponentRef("A", 1)})
    assert len(result.scores["global_sequence_nll"].evaluations) == 2


def test_mode_is_saved_loaded_and_pair_mismatch_rejected(tmp_path, designer, peptide):
    import json

    import ofoldx
    from ofoldx.errors import InvalidRequestError
    from ofoldx.models.membranempnn import MembraneMpnnProcessor, MembraneMpnnProcessorConfig

    designer.save(tmp_path)
    model_path, processor_path = tmp_path / "model.json", tmp_path / "processor.json"
    model_config, processor_config = json.loads(model_path.read_text()), json.loads(processor_path.read_text())
    assert model_config["label_mode"] == processor_config["label_mode"] == "per_residue"
    loaded = ofoldx.pipeline("design", model=tmp_path, backend="eager")
    assert loaded.processor.prepare_input(peptide("ALA"), membrane_labels=2).features.membrane_labels.tolist() == [[2]]
    with pytest.raises(InvalidRequestError, match="label_mode"):
        type(designer)(
            model=designer.model, processor=MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="global"))
        )
    model_config.pop("label_mode")
    model_path.write_text(json.dumps(model_config))
    with pytest.raises(InvalidRequestError, match="label_mode"):
        ofoldx.pipeline("design", model=tmp_path, backend="eager")


def test_global_mode_survives_pipeline_roundtrip(tmp_path, peptide):
    import ofoldx
    from ofoldx.errors import InvalidRequestError
    from ofoldx.models.membranempnn import (
        MembraneMpnnConfig,
        MembraneMpnnModel,
        MembraneMpnnPipeline,
        MembraneMpnnProcessor,
        MembraneMpnnProcessorConfig,
    )

    model = MembraneMpnnModel(
        MembraneMpnnConfig(
            label_mode="global", hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2
        )
    )
    processor = MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="global"))
    MembraneMpnnPipeline(model=model, processor=processor).save(tmp_path)
    loaded = ofoldx.pipeline("score", model=tmp_path, backend="eager")
    score = loaded(peptide("ALA", "GLY"), membrane_labels=1, seed=7)
    assert score.scores["global_sequence_nll"].value == pytest.approx(score.scores["global_sequence_nll"].value)
    with pytest.raises(InvalidRequestError, match="membrane_labels"):
        loaded(peptide("ALA", "GLY"), membrane_labels=2, seed=7)
