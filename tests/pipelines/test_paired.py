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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from ofoldx.conversion import SourceCheckpoint, save_repository
from ofoldx.model import Model
from ofoldx.models.registry import MODELS
from ofoldx.pipelines import PIPELINES, PairedComponent, PairedPipeline
from ofoldx.processor import Processor

MODEL_DIR = Path(__file__).resolve().parents[2] / "ofoldx" / "models"


class RepositoryPairedPipeline(PairedPipeline[Model, Processor]):
    task: ClassVar[str] = "fold"

    model_architecture = "test_repository"

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.model.weight + self.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output

    def fold(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = fold


PIPELINES.register(
    "test_paired_pipeline",
    pipeline=RepositoryPairedPipeline,
    model_architecture="test_repository",
)


class RepositoryScorePipeline(PairedPipeline[Model, Processor]):
    task: ClassVar[str] = "score"

    model_architecture = "test_repository"

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.model.weight + self.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output.mean()

    def score(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = score


PIPELINES.register("test_repository_score", pipeline=RepositoryScorePipeline, model_architecture="test_repository")


class RepositoryCustomPipeline(RepositoryScorePipeline):
    task: ClassVar[str] = "custom"

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output.sum()


PIPELINES.register("test_repository_custom", pipeline=RepositoryCustomPipeline, model_architecture="test_repository")


@pytest.mark.parametrize("sharded", [False, True])
def test_conversion_sources_and_licenses(tmp_path, repository_model, repository_processor, sharded):
    source = tmp_path / "source"
    source.mkdir()
    if sharded:
        checkpoint = source / "model.safetensors.index.json"
        checkpoint.write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
        (source / "part1.safetensors").write_bytes(b"source first shard")
        (source / "part2.safetensors").write_bytes(b"source second shard")
    else:
        checkpoint = source / "weights.ckpt"
        checkpoint.write_bytes(b"original checkpoint bytes")
    destination = tmp_path / "converted"
    pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
    save_repository(
        pipeline,
        destination,
        sources={"checkpoint": SourceCheckpoint(checkpoint, revision="source-commit")},
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        options={"variant": "test-selected-variant"},
    )
    record = json.loads((destination / "conversion.json").read_text())
    assert record["license"] == "AGPL-3.0-or-later"
    assert record["options"]["variant"] == "test-selected-variant"
    assert record["sources"]["checkpoint"]["revision"] == "source-commit"
    recorded = {item["name"]: item["sha256"] for item in record["sources"]["checkpoint"]["files"]}
    assert recorded == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert (
        record["initial_artifacts"]["model.json"]["sha256"]
        == hashlib.sha256((destination / "model.json").read_bytes()).hexdigest()
    )
    assert (destination / "README.md").read_bytes() == (MODEL_DIR / "proteinmpnn" / "README.md").read_bytes()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (destination / "license.md").read_text()
    assert "OFoldX" in (destination / "license-faq.md").read_text()
    loaded = RepositoryPairedPipeline.from_pretrained(destination, backend="eager")
    torch.testing.assert_close(loaded.model.weight, repository_model.weight)


def test_conversion_before_upload(tmp_path, repository_model, repository_processor, monkeypatch):
    destination = tmp_path / "converted"
    destination.mkdir()
    original = {"README.md": "# Caller model card\n", "license.md": "Caller AGPL notice\n"}
    for name, content in original.items():
        (destination / name).write_text(content)
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.write_bytes(b"weights")
    uploads = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, *, folder_path, repo_id, **kwargs):
            folder = Path(folder_path)
            assert all(
                (folder / name).is_file() for name in ("README.md", "license.md", "license-faq.md", "conversion.json")
            )
            assert (folder / "README.md").read_bytes() == (MODEL_DIR / "proteinmpnn" / "README.md").read_bytes()
            assert json.loads((folder / "conversion.json").read_text())["sources"]["checkpoint"]["files"]
            uploads.append(repo_id)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    save_repository(
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
        destination,
        sources={"checkpoint": checkpoint},
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        repo_id="example/converted",
    )
    assert uploads == ["example/converted"]
    assert (destination / "license.md").read_text() == original["license.md"]


@pytest.mark.parametrize("source_kind", ["string", "path", "explicit"])
def test_conversion_refuses_missing_local_source_before_writing(
    tmp_path, repository_model, repository_processor, monkeypatch, source_kind
):
    monkeypatch.chdir(tmp_path)
    reference = "weights/missing.ckpt"
    source = {
        "string": reference,
        "path": Path(reference),
        "explicit": SourceCheckpoint(reference),
    }[source_kind]
    destination = tmp_path / "converted"
    with pytest.raises(FileNotFoundError):
        save_repository(
            RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
            destination,
            sources={"checkpoint": source},
            converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        )
    assert not destination.exists()


@pytest.mark.parametrize("compound", [False, True])
@pytest.mark.parametrize("repo_id", [None, "example/current-variant"])
def test_proteinmpnn_conversion(tmp_path, monkeypatch, compound, repo_id):
    import ofoldx
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
    )

    small = {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1, "num_neighbors": 2, "dropout": 0.0}
    designer = ProteinMpnnPipeline(model=ProteinMpnnModel(ProteinMpnnConfig(small)), processor=ProteinMpnnProcessor())
    if compound:
        packer = MpnnPackerPipeline(
            model=MpnnPackerModel(MpnnPackerConfig({**small, "edge_size": 8, "num_context_atoms": 1})),
            processor=MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1)),
        )
        pipeline = ProteinMpnnDesignAndPackPipeline(designer=designer, packer=packer)
    else:
        pipeline = designer
    source = tmp_path / "original.pt"
    source.write_bytes(b"original")
    notice = tmp_path / "UPSTREAM-LICENSE"
    notice.write_text("Original upstream notice\n")
    destination = tmp_path / "converted"
    monkeypatch.setattr("ofoldx.conversion.upload_repository", lambda *args, **kwargs: None)
    save_repository(
        pipeline,
        destination,
        sources={"checkpoint": SourceCheckpoint(source, revision="source-commit", license_files=(notice,))},
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        repo_id=repo_id,
    )
    monkeypatch.chdir(destination)
    loaded = ofoldx.pipeline("design", model=".", backend="eager")
    assert type(loaded) is type(pipeline)
    card = (destination / "README.md").read_text()
    assert "license: agpl-3.0" in card
    if not compound:
        scorer = ofoldx.pipeline("score", model=destination, backend="eager")
        assert scorer.task == "score"
    record = json.loads((destination / "conversion.json").read_text())
    assert record["sources"]["checkpoint"]["revision"] == "source-commit"
    assert (destination / "licenses" / "checkpoint" / "UPSTREAM-LICENSE").read_text() == notice.read_text()


class RepositoryPairedComponent(PairedComponent[Model, Processor]):
    model_architecture = "test_repository"


class TestPairedComponent:
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        component = RepositoryPairedComponent(model=repository_model, processor=repository_processor)
        assert component.source is None
        (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")

        component.save(tmp_path)
        loaded = RepositoryPairedComponent.from_pretrained(tmp_path, backend="eager")

        assert (tmp_path / "model.json").is_file()
        assert (tmp_path / "processor.json").is_file()
        assert not (tmp_path / "pipeline.json").exists()
        assert type(loaded.model) is type(repository_model)
        assert type(loaded.processor) is type(repository_processor)
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.source.subdirectory == ""
        assert loaded.model.backend == "eager"


class TestPairedPipeline:
    def test_construction_validates_pair_and_accepts_subclasses(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        class CustomModel(type(repository_model)):
            pass

        class CustomProcessor(type(repository_processor)):
            pass

        workflow = RepositoryPairedPipeline(model=CustomModel(), processor=CustomProcessor())
        assert isinstance(workflow.model, CustomModel)
        assert isinstance(workflow.processor, CustomProcessor)
        assert workflow(torch.ones(3)).shape == (3,)
        destination = tmp_path / "unregistered"
        with pytest.raises(TypeError, match="register the exact classes"):
            workflow.save(destination)
        assert not destination.exists()
        with pytest.raises(TypeError, match="test_repository"):
            RepositoryPairedPipeline(model=repository_model, processor=Processor())

        MODELS.register("test_custom_pair_identity", model=CustomModel, processor=CustomProcessor)

        class CustomPipeline(RepositoryPairedPipeline):
            model_architecture = "test_custom_pair_identity"

        PIPELINES.register(
            "test_custom_pair_pipeline", pipeline=CustomPipeline, model_architecture="test_custom_pair_identity"
        )
        workflow = CustomPipeline(model=CustomModel(), processor=CustomProcessor())
        workflow.save(tmp_path / "registered")
        loaded = CustomPipeline.from_pretrained(tmp_path / "registered")
        assert type(loaded.model) is CustomModel
        assert type(loaded.processor) is CustomProcessor
        torch.testing.assert_close(loaded(torch.ones(3)), workflow(torch.ones(3)))

    def test_construction_uses_eval_without_disabling_gradients(
        self, repository_model: Any, repository_processor: Any
    ) -> None:
        assert repository_model.training
        pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        assert not pipeline.model.training
        output = pipeline(torch.ones(3)).sum()
        output.backward()
        assert pipeline.model.weight.grad is not None
        pipeline.model.train()
        assert pipeline.model.training

    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        assert pipeline.source is None

        pipeline.save(tmp_path)
        loaded = RepositoryPairedPipeline.from_pretrained(tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_paired_pipeline"
        }
        assert type(loaded.model) is type(repository_model)
        assert type(loaded.processor) is type(repository_processor)
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.source.subdirectory == ""
        assert loaded.processor.config is not None
        assert repository_processor.config is not None
        assert dict(loaded.processor.config) == dict(repository_processor.config)
        assert loaded.processor.dtype is torch.float16
        assert loaded.model.backend == "eager"
        for key, expected in repository_model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], expected)

    def test_stages_and_model_repository(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        from ofoldx import AutoPipelineForFold, pipeline, save_model

        save_model(tmp_path, model=repository_model, processor=repository_processor)
        loaded = AutoPipelineForFold(tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        prepared = loaded.prepare(input)
        staged = loaded.postprocess(prepared, loaded.forward(prepared))
        expected = input * 2 * torch.arange(3) + torch.arange(3) + 3
        torch.testing.assert_close(staged, expected)
        torch.testing.assert_close(loaded(input), expected)
        torch.testing.assert_close(
            AutoPipelineForFold.from_pretrained(pretrained_model_name_or_path=tmp_path)(input), expected
        )
        torch.testing.assert_close(
            RepositoryPairedPipeline.from_pretrained(pretrained_model_name_or_path=tmp_path)(input), expected
        )
        torch.testing.assert_close(pipeline("fold", model=tmp_path)(input), expected)
        assert loaded.id == "test_paired_pipeline"
        assert loaded.task == "fold"

    def test_task_counterparts_share_repository(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        from ofoldx import AutoPipelineForFold, AutoPipelineForScore

        designer = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        designer.save(tmp_path)
        scorer = AutoPipelineForScore(tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).mean()
        torch.testing.assert_close(scorer(input), expected)
        assert scorer.task == "score"
        assert scorer.source.path == tmp_path
        scorer.save(tmp_path / "score")
        folded = AutoPipelineForFold(tmp_path / "score")
        torch.testing.assert_close(folded(input).mean(), expected)

    def test_custom_task(self, tmp_path: Path, repository_model: Any, repository_processor: Any) -> None:
        from ofoldx import pipeline, save_model

        save_model(tmp_path, model=repository_model, processor=repository_processor)
        loaded = pipeline("custom", model=tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).sum()
        torch.testing.assert_close(loaded(input), expected)
        assert loaded.task == "custom"

    def test_unregistered_subclass_inherits_task_and_executes(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        class SumPipeline(RepositoryScorePipeline):
            def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
                return output.sum()

        workflow = SumPipeline(model=repository_model, processor=repository_processor)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).sum()
        assert workflow.task == "score"
        torch.testing.assert_close(workflow(input), expected)
        with pytest.raises(KeyError):
            workflow.save(tmp_path)


def test_conversion_records_hub_snapshot_shards_by_logical_path(tmp_path, repository_model, repository_processor):
    cache = tmp_path / "models--example--model"
    blobs = cache / "blobs"
    blobs.mkdir(parents=True)
    index_bytes = json.dumps(
        {"weight_map": {"a": "first.safetensors", "b": "first.safetensors", "c": "alias.safetensors"}}
    ).encode()
    (blobs / "index-hash").write_bytes(index_bytes)
    for revision, content in (("one", b"first revision"), ("two", b"second revision")):
        snapshot = cache / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (blobs / revision).write_bytes(content)
        (snapshot / "model.safetensors.index.json").symlink_to("../../blobs/index-hash")
        (snapshot / "first.safetensors").symlink_to(f"../../blobs/{revision}")
        (snapshot / "alias.safetensors").symlink_to(f"../../blobs/{revision}")
    sources = {
        revision: SourceCheckpoint(cache / "snapshots" / revision / "model.safetensors.index.json")
        for revision in ("one", "two")
    }
    destination = tmp_path / "converted"
    save_repository(
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
        destination,
        sources=sources,
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
    )
    records = json.loads((destination / "conversion.json").read_text())["sources"]
    for revision, content in (("one", b"first revision"), ("two", b"second revision")):
        record = records[revision]
        assert record["reference"] == str(sources[revision].path)
        expected = {
            "model.safetensors.index.json": index_bytes,
            "first.safetensors": content,
            "alias.safetensors": content,
        }
        assert {entry["name"]: (entry["size"], entry["sha256"]) for entry in record["files"]} == {
            name: (len(value), hashlib.sha256(value).hexdigest()) for name, value in expected.items()
        }
        assert len(record["files"]) == len(expected)


@pytest.mark.parametrize("shard", ["../outside", "/outside", "nested/../../outside", ""])
def test_conversion_rejects_escaping_shard_names(tmp_path, repository_model, repository_processor, shard):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard}}))
    with pytest.raises(ValueError, match="repository-relative"):
        save_repository(
            RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
            tmp_path / "converted",
            sources={"checkpoint": index},
            converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        )
    assert not (tmp_path / "converted").exists()


@pytest.mark.parametrize("kind", ["missing", "broken"])
def test_conversion_rejects_missing_or_broken_shards(tmp_path, repository_model, repository_processor, kind):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights.safetensors").write_bytes(b"outside")
    shard = "weights.safetensors"
    if kind == "broken":
        (source / shard).symlink_to(outside / "missing")
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard}}))
    with pytest.raises(FileNotFoundError):
        save_repository(
            RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
            tmp_path / "converted",
            sources={"checkpoint": index},
            converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
        )
    assert not (tmp_path / "converted").exists()


def test_conversion_records_ordinary_directory(tmp_path, repository_model, repository_processor):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "config.json").write_bytes(b"config")
    (source / "nested" / "weights").write_bytes(b"weights")
    (source / ".git" / "ignore").write_bytes(b"metadata")
    destination = tmp_path / "converted"
    save_repository(
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
        destination,
        sources={"checkpoint": source},
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
    )
    record = json.loads((destination / "conversion.json").read_text())["sources"]["checkpoint"]
    assert record["kind"] == "directory" and record["reference"] == str(source)
    assert {entry["name"]: (entry["size"], entry["sha256"]) for entry in record["files"]} == {
        name: (len(content), hashlib.sha256(content).hexdigest())
        for name, content in {"config.json": b"config", "nested/weights": b"weights"}.items()
    }


@pytest.mark.parametrize("directory_link", [False, True])
def test_conversion_records_external_shard_symlink(tmp_path, repository_model, repository_processor, directory_link):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "other_disk"
    outside.mkdir()
    payload = b"external shard"
    (outside / "weights.safetensors").write_bytes(payload)
    if directory_link:
        (source / "linked").symlink_to(outside, target_is_directory=True)
        name = "linked/weights.safetensors"
    else:
        (source / "weights.safetensors").symlink_to(outside / "weights.safetensors")
        name = "weights.safetensors"
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": name}}))
    destination = tmp_path / "converted"
    save_repository(
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
        destination,
        sources={"checkpoint": index},
        converter="ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_checkpoint",
    )
    record = json.loads((destination / "conversion.json").read_text())["sources"]["checkpoint"]
    shards = {entry["name"]: entry for entry in record["files"]}
    assert set(shards) == {index.name, name}
    assert shards[name]["size"] == len(payload)
    assert shards[name]["sha256"] == hashlib.sha256(payload).hexdigest()
