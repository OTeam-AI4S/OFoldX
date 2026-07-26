from __future__ import annotations

import json

import chanfig
import pytest
import torch
from torch import nn

from ofoldx import Model, Processor
from ofoldx.api import pipeline
from ofoldx.artifacts import save_model
from ofoldx.models.registry import MODELS
from ofoldx.pipelines import CompositePair, CompositePipeline, PairedPipeline
from ofoldx.pipelines.registry import PIPELINES


class PairedConfig(chanfig.Config):
    pass


class PairedModel(Model):
    config_class = PairedConfig

    def __init__(self, config: PairedConfig) -> None:
        super().__init__(PairedConfig(dict(config)))
        self.weight = nn.Parameter(torch.zeros(self.config["size"]))


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
        CompositePair(
            "optional",
            "test.alternate",
            "optional_model",
            "optional_processor",
            required=False,
        ),
    )

    def __init__(
        self,
        *,
        model: PairedModel,
        processor: PairedProcessor,
        optional_model: PairedModel | None = None,
        optional_processor: AlternateProcessor | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.optional_model = optional_model
        self.optional_processor = optional_processor
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
    def test_repository_round_trip(self, tmp_path) -> None:
        _register_test_artifacts()

        model = PairedModel(PairedConfig({"size": 3}))
        with torch.no_grad():
            model.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
        ConcretePipeline(model=model, processor=PairedProcessor()).save(tmp_path)

        assert json.loads((tmp_path / "pipeline.json").read_text()) == {"pipeline": "test.paired.fold"}

        loaded = pipeline("fold", model=tmp_path)

        assert type(loaded) is ConcretePipeline
        assert type(loaded.model) is PairedModel
        assert type(loaded.processor) is PairedProcessor
        assert torch.equal(loaded.model.weight, model.weight)

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
    @pytest.mark.parametrize("with_optional", (False, True))
    def test_repository_round_trip(self, tmp_path, with_optional: bool) -> None:
        _register_test_artifacts()
        optional_model = PairedModel(PairedConfig({"size": 2})) if with_optional else None
        if optional_model is not None:
            with torch.no_grad():
                optional_model.weight.copy_(torch.tensor([4.0, 5.0]))
        original = ConcreteCompositePipeline(
            model=PairedModel(PairedConfig({"size": 2})),
            processor=PairedProcessor(),
            optional_model=optional_model,
            optional_processor=AlternateProcessor() if with_optional else None,
        )
        original.save(tmp_path)

        loaded = pipeline("fold", model=tmp_path)

        assert type(loaded) is ConcreteCompositePipeline
        assert type(loaded.model) is PairedModel
        assert type(loaded.processor) is PairedProcessor
        if optional_model is None:
            assert loaded.optional_model is None
            assert loaded.optional_processor is None
        else:
            assert type(loaded.optional_model) is PairedModel
            assert type(loaded.optional_processor) is AlternateProcessor
            assert torch.equal(loaded.optional_model.weight, optional_model.weight)
