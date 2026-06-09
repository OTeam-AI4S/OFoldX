from __future__ import annotations

from pathlib import Path

import pytest
from chanfig import Config

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
from ofoldx.pipelines.auto import (
    PIPELINE_MANIFEST_NAME,
    build_pipeline_manifest,
    component_path,
    has_pipeline_manifest,
    model_weights_name,
    pipeline_entrypoint,
    pipeline_task,
    read_pipeline_manifest,
    resolve_pipeline_class,
    save_component,
    validate_pipeline_class,
    validate_pipeline_task,
)
from ofoldx.pipelines.structure_prediction import StructurePredictionPipeline


class SaveableComponent:
    def __init__(self) -> None:
        self.saved: list[tuple[Path, dict[str, object]]] = []

    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        self.saved.append((save_directory, kwargs))
        save_directory.mkdir(parents=True)
        (save_directory / "marker.txt").write_text("saved")


def test_pipeline_manifest_round_trip(tmp_path) -> None:
    entrypoint = "ofoldx.pipelines.structure_prediction.StructurePredictionPipeline"
    manifest = build_pipeline_manifest(
        entrypoint,
        {"predictor": "predictor", "processor": "processor"},
        task="structure_prediction",
    )
    manifest.json(tmp_path / PIPELINE_MANIFEST_NAME)

    assert has_pipeline_manifest(tmp_path)
    read_manifest = read_pipeline_manifest(tmp_path)
    assert pipeline_entrypoint(read_manifest) == entrypoint
    assert pipeline_task(read_manifest) == "structure_prediction"
    assert resolve_pipeline_class(read_manifest) is StructurePredictionPipeline
    validate_pipeline_class(read_manifest, StructurePredictionPipeline)
    validate_pipeline_task(read_manifest, "structure-prediction")
    assert component_path(tmp_path, read_manifest, "predictor") == tmp_path / "predictor"


def test_pipeline_manifest_rejects_malformed_values(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_pipeline_manifest(tmp_path / "missing")
    file_path = tmp_path / PIPELINE_MANIFEST_NAME
    file_path.write_text("{}")
    with pytest.raises(NotADirectoryError, match="must be a directory"):
        read_pipeline_manifest(file_path)
    with pytest.raises(ValueError, match="fully qualified entrypoint"):
        pipeline_entrypoint(Config({"entrypoint": "Pipeline"}))
    with pytest.raises(TypeError, match="cannot be loaded"):
        validate_pipeline_class(
            build_pipeline_manifest(
                "ofoldx.pipelines.design.DesignPipeline",
                {},
                task="design_generation",
            ),
            StructurePredictionPipeline,
        )
    with pytest.raises(TypeError, match="does not resolve to a class"):
        resolve_pipeline_class(
            build_pipeline_manifest(
                "ofoldx.pipelines.auto.PIPELINE_MANIFEST_NAME",
                {},
                task="structure_prediction",
            )
        )


def test_pipeline_manifest_is_kind_checked(tmp_path) -> None:
    build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.mixin.ModelMixin",
        architecture="demo",
        files={"config": "config.json"},
        weights={"default": "model.safetensors"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)

    assert not has_pipeline_manifest(tmp_path)
    with pytest.raises(ValueError, match="kind 'model' is not 'pipeline'"):
        read_pipeline_manifest(tmp_path)


def test_pipeline_manifest_requires_task() -> None:
    manifest = build_artifact_manifest(
        kind="pipeline",
        entrypoint="ofoldx.pipelines.base.Pipeline",
        components={},
    )

    with pytest.raises(ValueError, match="task"):
        pipeline_task(manifest)


@pytest.mark.parametrize(
    "index",
    [
        Config({"components": []}),
        Config({"components": {"model": {}}}),
        Config({"components": {"model": {"path": 3}}}),
        Config({"components": {"model": {"path": "../outside"}}}),
        Config({"components": {"model": {"path": "/outside"}}}),
    ],
)
def test_component_path_rejects_malformed_components(tmp_path, index: Config) -> None:
    with pytest.raises(ValueError):
        component_path(tmp_path, index, "model")


def test_model_weights_name_handles_variants() -> None:
    assert model_weights_name(None) == "model.safetensors"
    assert model_weights_name("ema") == "model.ema.safetensors"


def test_save_component_uses_pretrained_protocol(tmp_path) -> None:
    component = SaveableComponent()

    save_component(component, tmp_path / "component", "component", variant="ema")

    assert component.saved == [(tmp_path / "component", {"variant": "ema"})]
    assert (tmp_path / "component" / "marker.txt").is_file()
    with pytest.raises(TypeError, match="save_pretrained"):
        save_component(object(), tmp_path / "bad", "bad")
