from __future__ import annotations

from chanfig import Config

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
from ofoldx.processors.mixin import ProcessorMixin


class DemoProcessor(ProcessorMixin):
    def __init__(self, *, value: int = 1) -> None:
        self.value = value

    def _processor_config(self) -> dict[str, object]:
        return {"value": self.value}


def test_processor_mixin_saves_manifest_separate_from_config(tmp_path) -> None:
    processor = DemoProcessor(value=7)

    processor.save_pretrained(tmp_path)

    manifest = Config.from_json(tmp_path / ARTIFACT_MANIFEST_NAME)
    config = Config.from_json(tmp_path / "processor_config.json")
    assert manifest["kind"] == "processor"
    assert manifest["entrypoint"] == f"{DemoProcessor.__module__}.{DemoProcessor.__qualname__}"
    assert manifest["files"]["config"] == "processor_config.json"
    assert config == {"value": 7}

    loaded = DemoProcessor.from_pretrained(tmp_path)

    assert isinstance(loaded, DemoProcessor)
    assert loaded.value == 7


def test_processor_mixin_loads_processor_component_from_model_artifact(tmp_path) -> None:
    processor_dir = tmp_path / "processor"
    DemoProcessor(value=11).save_pretrained(processor_dir)
    build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.alphafold2.model.AlphaFold2Model",
        architecture="alphafold2",
        components={"processor": "processor"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)

    loaded = DemoProcessor.from_pretrained(tmp_path)

    assert isinstance(loaded, DemoProcessor)
    assert loaded.value == 11
