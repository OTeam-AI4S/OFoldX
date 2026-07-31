from __future__ import annotations

import json

import chanfig
import pytest
import torch
from torch import nn

from ofoldx import AutoModel, AutoProcessor, Model, Processor
from ofoldx.api import pipeline
from ofoldx.artifacts import save_model
from ofoldx.models.registry import MODELS
from ofoldx.pipelines import CompositePair, CompositePipeline, PairedPipeline
from ofoldx.pipelines.registry import PIPELINES


class PairedConfig(chanfig.Config):
    size: int = 1


class PairedModel(Model):
    config_class = PairedConfig

    def __init__(self, config: PairedConfig) -> None:
        super().__init__(PairedConfig(dict(config)))
        self.weight = nn.Parameter(torch.zeros(self.config["size"]))
        self.bias = nn.Parameter(torch.zeros(self.config["size"]))


class PairedProcessor(Processor):
    pass


class AlternateProcessor(Processor):
    pass


class ConcretePipeline(PairedPipeline[PairedModel, PairedProcessor]):
    model_architecture = "test.paired"

    def fold(self, input: object) -> object:
        return input


class ConcreteCompositePipeline(CompositePipeline):
    pair_layout = (
        CompositePair("", "test.paired", "model", "processor"),
        CompositePair("alternate", "test.alternate", "alternate_model", "alternate_processor"),
    )

    def __init__(
        self,
        *,
        model: PairedModel,
        processor: PairedProcessor,
        alternate_model: PairedModel,
        alternate_processor: AlternateProcessor,
    ) -> None:
        self.model = model
        self.processor = processor
        self.alternate_model = alternate_model
        self.alternate_processor = alternate_processor
        self._validate_components()

    def fold(self, input: object) -> object:
        return input


def _register_test_artifacts() -> None:
    if "test.paired" not in MODELS:
        MODELS.register(
            "test.paired",
            model=PairedModel,
            processor=PairedProcessor,
        )
        MODELS.register(
            "test.alternate",
            model=PairedModel,
            processor=AlternateProcessor,
        )
        PIPELINES.register("test.paired.fold", pipeline=ConcretePipeline)
        PIPELINES.register("test.composite.fold", pipeline=ConcreteCompositePipeline)


class TestPairedPipeline:
    def test_save_preserves_unowned_nested_repository(self, tmp_path) -> None:
        _register_test_artifacts()
        nested = tmp_path / "independent"
        save_model(
            nested,
            model=PairedModel(PairedConfig({"size": 1})),
            processor=PairedProcessor(),
        )

        ConcretePipeline(
            model=PairedModel(PairedConfig({"size": 1})),
            processor=PairedProcessor(),
        ).save(tmp_path)

        assert (nested / "model.safetensors").is_file()

    @pytest.mark.parametrize("max_shard_size", ["5GB", 12])
    def test_repository_round_trip(self, tmp_path, max_shard_size: int | str) -> None:
        _register_test_artifacts()

        model = PairedModel(PairedConfig({"size": 3}))
        with torch.no_grad():
            model.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
            model.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))
        ConcretePipeline(model=model, processor=PairedProcessor()).save(
            tmp_path,
            max_shard_size=max_shard_size,
        )

        assert json.loads((tmp_path / "pipeline.json").read_text()) == {"pipeline": "test.paired.fold"}
        assert (tmp_path / "model.safetensors.index.json").is_file() is (max_shard_size == 12)

        loaded = pipeline("fold", model=tmp_path)
        auto_model = AutoModel(tmp_path)
        auto_processor = AutoProcessor(tmp_path)

        assert type(loaded) is ConcretePipeline
        assert type(loaded.model) is PairedModel
        assert type(loaded.processor) is PairedProcessor
        assert type(auto_model) is PairedModel
        assert type(auto_processor) is PairedProcessor
        assert torch.equal(loaded.model.weight, model.weight)
        assert torch.equal(loaded.model.bias, model.bias)
        assert torch.equal(auto_model.weight, model.weight)
        assert torch.equal(auto_model.bias, model.bias)

    def test_repository_rejects_another_registered_pair(self, tmp_path) -> None:
        _register_test_artifacts()
        save_model(
            tmp_path,
            model=PairedModel(PairedConfig({"size": 1})),
            processor=AlternateProcessor(),
        )
        (tmp_path / "pipeline.json").write_text(
            json.dumps({"pipeline": "test.paired.fold"}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="repository declares 'test.alternate'"):
            pipeline("fold", model=tmp_path)


class TestCompositePipeline:
    def test_repository_round_trip(self, tmp_path) -> None:
        _register_test_artifacts()
        alternate_model = PairedModel(PairedConfig({"size": 2}))
        with torch.no_grad():
            alternate_model.weight.copy_(torch.tensor([4.0, 5.0]))
        original = ConcreteCompositePipeline(
            model=PairedModel(PairedConfig({"size": 2})),
            processor=PairedProcessor(),
            alternate_model=alternate_model,
            alternate_processor=AlternateProcessor(),
        )
        original.save(tmp_path)

        loaded = pipeline("fold", model=tmp_path)

        assert type(loaded) is ConcreteCompositePipeline
        assert type(loaded.model) is PairedModel
        assert type(loaded.processor) is PairedProcessor
        assert type(loaded.alternate_model) is PairedModel
        assert type(loaded.alternate_processor) is AlternateProcessor
        assert torch.equal(loaded.alternate_model.weight, alternate_model.weight)
