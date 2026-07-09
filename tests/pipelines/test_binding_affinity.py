from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from chanfig import Config
from torch import nn

from ofoldx.models import (
    AutoModel,
    BaseInput,
    BindingAffinityOutput,
    Boltz2AffinityModel,
    DesignCandidate,
)
from ofoldx.pipelines import BindingAffinityPipeline, Pipeline, StructureProcessor
from ofoldx.pipelines.auto import PIPELINE_MANIFEST_NAME
from ofoldx.processors import AutoProcessor
from tests.pipelines.test_design import protein_ligand_structure, tiny_boltz2_affinity_model


@dataclass(slots=True, kw_only=True)
class CustomInput(BaseInput):
    value: torch.Tensor


class AffinityModel(nn.Module):
    def score(self, inputs: CustomInput) -> DesignCandidate:
        return DesignCandidate(scores={"affinity_pred_value": inputs.value.reshape(1, 1, 1)})


class SaveableAffinityModel(nn.Module):
    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


class SaveableProcessor:
    def __call__(self, structure: object, /) -> CustomInput:
        return CustomInput(value=torch.zeros(1))

    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


def test_binding_affinity_pipeline_returns_score_candidates() -> None:
    pipeline = BindingAffinityPipeline(AffinityModel())

    output = pipeline(CustomInput(value=torch.tensor([1.5])))

    assert isinstance(output, BindingAffinityOutput)
    torch.testing.assert_close(output.affinity_pred_value, torch.tensor([[[1.5]]]))
    torch.testing.assert_close(output.scores["affinity_pred_value"], torch.tensor([[[1.5]]]))


def test_binding_affinity_pipeline_requires_affinity_score() -> None:
    class MissingScoreModel(nn.Module):
        def score(self, inputs: CustomInput) -> DesignCandidate:
            return DesignCandidate(scores={"other": inputs.value})

    pipeline = BindingAffinityPipeline(MissingScoreModel())

    with pytest.raises(ValueError, match="affinity_pred_value"):
        pipeline(CustomInput(value=torch.tensor([1.5])))


def test_binding_affinity_pipeline_save_writes_task_manifest(tmp_path: Path) -> None:
    pipeline = BindingAffinityPipeline(SaveableAffinityModel(), processor=SaveableProcessor())

    pipeline.save_pretrained(tmp_path)

    manifest = Config.from_json(tmp_path / PIPELINE_MANIFEST_NAME)
    assert manifest["kind"] == "pipeline"
    assert manifest["task"] == "binding_affinity"
    assert manifest["entrypoint"] == "ofoldx.pipelines.binding_affinity.BindingAffinityPipeline"
    assert manifest["components"]["scorer"]["path"] == "scorer"
    assert manifest["components"]["processor"]["path"] == "processor"


def test_pipeline_from_pretrained_routes_bare_boltz2_affinity_model(tmp_path: Path) -> None:
    model = tiny_boltz2_affinity_model(tmp_path)
    model.save_pretrained(tmp_path)

    loaded = Pipeline.from_pretrained(tmp_path)
    output = loaded(protein_ligand_structure(), recycling_steps=1, num_samples=1, num_sampling_steps=1)

    assert isinstance(loaded, BindingAffinityPipeline)
    assert isinstance(output, BindingAffinityOutput)
    assert "affinity_pred_value" in output.scores


def test_binding_affinity_pipeline_e2e_runs_saved_boltz2_affinity_package(tmp_path: Path) -> None:
    model = tiny_boltz2_affinity_model(tmp_path)
    pipeline = BindingAffinityPipeline(
        model,
        processor=cast(StructureProcessor[BaseInput], AutoProcessor.from_config(model.config)),
    )

    pipeline.save_pretrained(tmp_path)
    loaded = Pipeline.from_pretrained(tmp_path)
    loaded_model = AutoModel.from_pretrained(tmp_path)
    output = loaded(protein_ligand_structure(), recycling_steps=1, num_samples=1, num_sampling_steps=1)

    assert isinstance(loaded, BindingAffinityPipeline)
    assert isinstance(loaded_model, Boltz2AffinityModel)
    assert isinstance(output, BindingAffinityOutput)
    assert output.affinity_pred_value.shape[-1] == 1
