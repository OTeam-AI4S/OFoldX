from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
from ofoldx.hub import (
    model_card_metadata,
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
        create_pr=False,
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
        "create_pr": False,
        "allow_patterns": ["*.json"],
        "ignore_patterns": ["*.pt"],
    }
    assert calls[3][1] == {
        "repo_id": "oteam/protein-model",
        "repo_type": "model",
        "token": "token",
        "tag": "v1",
        "revision": "main",
    }


def test_upload_folder_to_hub_rejects_tag_with_pull_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeHfApi:
        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_folder(self, **kwargs: object) -> str:
            return "uploaded"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi))

    # A tag cannot be created for a PR upload -- the PR revision is not yet a mergeable ref.
    with pytest.raises(ValueError, match="cannot create a Hub tag for a pull-request upload"):
        upload_folder_to_hub(
            tmp_path,
            repo_id="oteam/protein-model",
            create_pr=True,
            tag="v1",
        )


def test_write_model_card_generates_generic_card_for_bare_artifact(tmp_path: Path) -> None:
    # A bare artifact that ships no authored card gets a compact, GENERIC card built from the manifest --
    # library/kind/task/model-index front matter plus a load snippet. No per-model prose is synthesized: that
    # content lives in each family's authored README, not in the shared hub module.
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
    assert "widget" not in metadata  # no per-architecture widget synthesis
    assert "# protein-model" in content
    assert "## Usage" in content
    assert "Pipeline.from_pretrained" in content
    assert "AutoModel.from_pretrained" in content


def test_write_model_card_preserves_an_authored_card(tmp_path: Path) -> None:
    # When a self-contained authored card (with YAML front matter) is already present -- the family README that
    # save_pretrained copies in -- it IS the model card and is left untouched. The shared hub module no longer
    # regenerates per-model front matter or body.
    build_artifact_manifest(
        kind="model",
        entrypoint="proteinlib.models.DemoModel",
        architecture="demo",
        task="design",
    ).json(tmp_path / ARTIFACT_MANIFEST_NAME)
    authored = (
        "---\n"
        "library_name: ofoldx\n"
        "tags:\n"
        "  - biology\n"
        "  - protein-design\n"
        "license: mit\n"
        "---\n\n"
        "# DemoDesigner\n\nHand-authored public model card.\n\n"
        "## Citation\n\n```bibtex\n@article{demo2025, title = {Demo}}\n```\n"
    )
    (tmp_path / "README.md").write_text(authored)

    card_path = write_model_card(tmp_path, repo_id="oteam/demo", metadata={"tags": ["ignored-at-publish"]})

    assert card_path.read_text() == authored  # verbatim; the authored card is the single source of truth
    metadata = _read_frontmatter(card_path)
    assert metadata["license"] == "mit"
    assert metadata["tags"] == ["biology", "protein-design"]


def test_model_card_metadata_merges_tags_and_can_disable_widget() -> None:
    manifest = build_artifact_manifest(
        kind="pipeline",
        entrypoint="proteinlib.pipelines.DemoPipeline",
        task="design",
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


def test_model_card_metadata_model_index_name_from_repo_id() -> None:
    # The one repo-specific generic key: the model-index name is derived from the repo id, not any per-model table.
    manifest = build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner",
        architecture="proteinmpnn",
        task="design",
    )

    metadata = model_card_metadata(manifest, repo_id="oteam/solublempnn-noise020")

    assert metadata["model-index"] == [{"name": "solublempnn-noise020", "results": []}]
    assert metadata["library_name"] == "ofoldx"
    assert metadata["task"] == "design"


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
