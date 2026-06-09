from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
from ofoldx.hub import (
    default_widget_examples,
    model_card_metadata,
    public_architecture_name,
    publish_artifact_to_hub,
    resolve_pretrained_path,
    upload_artifact_from_args,
    upload_folder_to_hub,
    write_model_card,
)


def _read_frontmatter(path: Path) -> dict[str, object]:
    content = path.read_text()
    assert content.startswith("---\n")
    _, frontmatter, _ = content.split("---\n", 2)
    metadata = yaml.safe_load(frontmatter)
    assert isinstance(metadata, dict)
    return metadata


def test_resolve_pretrained_path_returns_local_directory(tmp_path: Path) -> None:
    assert resolve_pretrained_path(tmp_path) == tmp_path


def test_resolve_pretrained_path_downloads_hub_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    download_path = tmp_path / "downloaded"

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(download_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    resolved = resolve_pretrained_path(
        "oteam/protein-model",
        repo_type="model",
        revision="main",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        token="token",
        allow_patterns=("manifest.json", "*.safetensors"),
    )

    assert resolved == download_path
    assert calls == [
        {
            "repo_id": "oteam/protein-model",
            "repo_type": "model",
            "revision": "main",
            "cache_dir": str(tmp_path / "cache"),
            "local_files_only": True,
            "token": "token",
            "allow_patterns": ["manifest.json", "*.safetensors"],
        }
    ]


def test_upload_folder_to_hub_can_delete_upload_and_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeHfApi:
        def delete_repo(self, **kwargs: object) -> None:
            calls.append(("delete_repo", kwargs))

        def create_repo(self, **kwargs: object) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs: object) -> str:
            calls.append(("upload_folder", kwargs))
            return "uploaded"

        def create_tag(self, **kwargs: object) -> None:
            calls.append(("create_tag", kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi))

    result = upload_folder_to_hub(
        tmp_path,
        repo_id="oteam/protein-model",
        private=True,
        token="token",
        revision="main",
        commit_message="publish",
        create_pr=True,
        delete_existing=True,
        tag="v1",
        allow_patterns=("*.json",),
        ignore_patterns=("*.pt",),
    )

    assert result == "uploaded"
    assert [name for name, _ in calls] == ["delete_repo", "create_repo", "upload_folder", "create_tag"]
    assert calls[0][1] == {
        "repo_id": "oteam/protein-model",
        "repo_type": "model",
        "token": "token",
        "missing_ok": True,
    }
    assert calls[1][1] == {
        "repo_id": "oteam/protein-model",
        "repo_type": "model",
        "private": True,
        "token": "token",
        "exist_ok": True,
    }
    assert calls[2][1] == {
        "folder_path": str(tmp_path),
        "repo_id": "oteam/protein-model",
        "repo_type": "model",
        "revision": "main",
        "commit_message": "publish",
        "token": "token",
        "create_pr": True,
        "allow_patterns": ["*.json"],
        "ignore_patterns": ["*.pt"],
    }
    assert calls[3][1] == {
        "repo_id": "oteam/protein-model",
        "repo_type": "model",
        "token": "token",
        "tag": "v1",
    }


def test_write_model_card_generates_frontmatter_and_widget(tmp_path: Path) -> None:
    build_artifact_manifest(
        kind="model",
        entrypoint="proteinlib.models.DemoModel",
        architecture="demo",
        task="structure_prediction",
        files={"config": "config.json"},
        weights={"default": "model.safetensors"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)

    card_path = write_model_card(tmp_path, repo_id="oteam/protein-model")

    content = card_path.read_text()
    metadata = _read_frontmatter(card_path)
    assert metadata["library_name"] == "proteinlib"
    assert metadata["artifact_kind"] == "model"
    assert metadata["pipeline_tag"] == "other"
    assert metadata["task"] == "structure_prediction"
    assert metadata["model-index"] == [{"name": "protein-model", "results": []}]
    widget = metadata["widget"]
    assert isinstance(widget, list)
    assert isinstance(widget[0], dict)
    assert widget[0]["example_title"] == "Single-chain protein"
    assert 'example_title: "Single-chain protein"' in content
    assert "## Model Details" in content
    assert "### Model Specification" in content
    assert "| Entrypoint | `proteinlib.models.DemoModel` |" in content
    assert "## Training Details" in content
    assert "## Citation" in content
    assert "### Pipeline Usage" in content
    assert "### Direct Use" not in content
    assert "Pipeline.from_pretrained" in content
    assert "AutoModel.from_pretrained" in content
    assert content.index("Pipeline.from_pretrained") < content.index("AutoModel.from_pretrained")
    assert "manifest.json" in content


def test_write_model_card_includes_upstream_provenance_from_config(tmp_path: Path) -> None:
    build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.proteinmpnn.model.ProteinMPNNDesignGenerator",
        architecture="proteinmpnn",
        task="design_generation",
        files={"config": "config.json"},
        weights={"default": "model.safetensors"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)
    (tmp_path / "config.json").write_text('{"checkpoint_variant": "proteinmpnn_v_48_020", "k_neighbors": 48}')

    card_path = write_model_card(tmp_path, repo_id="oteam/proteinmpnn-noise020")

    content = card_path.read_text()
    assert "**Upstream Project**: ProteinMPNN / LigandMPNN release" in content
    assert "`proteinmpnn_v_48_020.pt`" in content
    assert "| k-Neighbors | `48` |" in content
    assert "OFoldX did not train these weights." in content
    assert "10.1126/science.add2187" in content
    assert "MIT for upstream ProteinMPNN and LigandMPNN code/model parameters" in content
    assert "The Hub `license` metadata, when present, reflects the source checkpoint" in content


def test_model_card_uses_noise_level_not_checkpoint_suffix_math(tmp_path: Path) -> None:
    build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.proteinmpnn.model.ProteinMPNNDesignGenerator",
        architecture="proteinmpnn",
        task="design_generation",
        files={"config": "config.json"},
        weights={"default": "model.safetensors"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)
    (tmp_path / "config.json").write_text(
        '{"checkpoint_variant": "proteinmpnn_v_48_020", "k_neighbors": 48, "noise_level": 0.2}'
    )

    card_path = write_model_card(tmp_path, repo_id="oteam/proteinmpnn-noise020")

    content = card_path.read_text()
    assert "0.20 Angstrom" in content
    assert "0.020" not in content


def test_solublempnn_model_card_uses_soluble_provenance(tmp_path: Path) -> None:
    build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.proteinmpnn.model.ProteinMPNNDesignGenerator",
        architecture="proteinmpnn",
        task="design_generation",
        files={"config": "config.json"},
        weights={"default": "model.safetensors"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)
    (tmp_path / "config.json").write_text(
        '{"checkpoint_variant": "solublempnn_v_48_020", "k_neighbors": 48, "noise_level": 0.2}'
    )

    card_path = write_model_card(tmp_path, repo_id="oteam/solublempnn-noise020")

    content = card_path.read_text()
    assert "**Upstream Project**: SolubleMPNN" in content
    assert "Computational design of soluble and functional membrane protein analogues" in content
    assert "10.1038/s41586-024-07601-y" in content


def test_model_card_metadata_merges_tags_and_can_disable_widget() -> None:
    manifest = build_artifact_manifest(
        kind="pipeline",
        entrypoint="proteinlib.pipelines.DemoPipeline",
        task="design_generation",
        components={"generator": "generator"},
    )

    metadata = model_card_metadata(
        manifest,
        metadata={"tags": ["curated"], "widget": None},
    )

    tags = metadata["tags"]
    assert isinstance(tags, list)
    assert tags[-1] == "curated"
    assert "widget" not in metadata


def test_default_widget_examples_are_static_inputs() -> None:
    examples = default_widget_examples("structure-prediction")

    assert examples
    assert examples[0]["pipeline_tag"] == "other"
    assert examples[0]["task"] == "structure_prediction"
    assert "output" not in examples[0]


def test_model_card_metadata_uses_public_architecture_aliases() -> None:
    manifest = build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.boltzgen.model.BoltzGenIFoldGenerator",
        architecture="boltzgen_ifold",
        task="design_generation",
    )

    metadata = model_card_metadata(manifest)

    assert public_architecture_name("boltzgen_ifold") == "boltzgen-inversefolding"
    tags = metadata["tags"]
    assert isinstance(tags, list)
    assert "boltzgen-inversefolding" in tags
    assert "boltzgen_ifold" not in tags
    widget = metadata["widget"]
    assert isinstance(widget, list)
    assert isinstance(widget[0], dict)
    assert widget[0]["example_title"] == "Inverse folding"


def test_model_card_metadata_includes_hub_frontmatter_for_known_release() -> None:
    manifest = build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.proteinmpnn.model.ProteinMPNNDesignGenerator",
        architecture="proteinmpnn",
        task="design_generation",
    )

    metadata = model_card_metadata(manifest, repo_id="oteam/solublempnn-noise020")

    assert metadata["license"] == "mit"
    assert metadata["model-index"] == [{"name": "solublempnn-noise020", "results": []}]


def test_model_card_metadata_records_alphafold2_license() -> None:
    manifest = build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.alphafold2.model.AlphaFold2Model",
        architecture="alphafold2",
        task="structure_prediction",
    )

    metadata = model_card_metadata(manifest, repo_id="oteam/alphafold2")

    assert metadata["license"] == "cc-by-4.0"
    assert metadata["model-index"] == [{"name": "alphafold2", "results": []}]


def test_model_card_metadata_includes_base_model_and_datasets_for_boltzgen() -> None:
    manifest = build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.boltzgen.model.BoltzGenDesignGenerator",
        architecture="boltzgen_diverse",
        task="design_generation",
    )

    metadata = model_card_metadata(manifest, repo_id="oteam/boltzgen-diverse")

    assert metadata["license"] == "mit"
    assert metadata["base_model"] == "boltz-community/boltz-2"
    assert metadata["datasets"] == ["boltzgen/boltzgen1_train", "boltzgen/inference-data"]
    assert metadata["model-index"] == [{"name": "boltzgen-diverse", "results": []}]


def test_binding_affinity_widget_uses_property_prediction_task() -> None:
    examples = default_widget_examples("binding-affinity-prediction", architecture="boltz2_affinity")

    assert examples
    assert examples[0]["task"] == "binding_affinity_prediction"
    assert examples[0]["example_title"] == "Protein-ligand affinity scoring"


def test_publish_artifact_to_hub_writes_card_then_uploads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build_artifact_manifest(
        kind="processor",
        entrypoint="proteinlib.processors.DemoProcessor",
        files={"config": "processor_config.json"},
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)
    calls: list[dict[str, object]] = []

    def upload_folder(folder_path: Path, **kwargs: object) -> str:
        assert (tmp_path / "README.md").is_file()
        calls.append({"folder_path": folder_path, **kwargs})
        return "uploaded"

    monkeypatch.setattr("ofoldx.hub.upload_folder_to_hub", upload_folder)

    result = publish_artifact_to_hub(
        tmp_path,
        repo_id="oteam/protein-processor",
        private=True,
        tag="v1",
        model_card_metadata={"tags": ["processor-card"]},
    )

    assert result == "uploaded"
    assert calls == [
        {
            "folder_path": tmp_path,
            "repo_id": "oteam/protein-processor",
            "repo_type": "model",
            "private": True,
            "token": None,
            "revision": None,
            "commit_message": None,
            "create_pr": False,
            "delete_existing": False,
            "tag": "v1",
            "allow_patterns": None,
            "ignore_patterns": None,
        }
    ]
    assert "processor-card" in (tmp_path / "README.md").read_text()


def test_upload_artifact_from_args_requires_repo_id_when_enabled(tmp_path: Path) -> None:
    args = SimpleNamespace(push_to_hub=True, repo_id=None)

    with pytest.raises(ValueError, match="--repo-id"):
        upload_artifact_from_args(tmp_path, args)
