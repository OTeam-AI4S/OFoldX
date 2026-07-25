from __future__ import annotations

import json

import chanfig
import torch
from torch import nn

from ofoldx import Model, Processor
from ofoldx.api import pipeline
from ofoldx.models.registry import MODELS
from ofoldx.pipelines import PairedPipeline
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


class ConcretePipeline(PairedPipeline[PairedModel, PairedProcessor]):
    def fold(self, input: object) -> object:
        return input


def test_marker_backed_paired_pipeline_round_trip(tmp_path) -> None:
    MODELS.register(
        "test.paired",
        model=PairedModel,
        processor=PairedProcessor,
    )
    PIPELINES.register("test.paired.fold", pipeline=ConcretePipeline)

    model = PairedModel(PairedConfig({"size": 3}))
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    ConcretePipeline(model=model, processor=PairedProcessor()).save(tmp_path)

    assert json.loads((tmp_path / "pipeline.json").read_text()) == {"pipeline": "test.paired.fold"}

    loaded = pipeline("fold", model=tmp_path)

    assert type(loaded) is ConcretePipeline
    assert isinstance(loaded.model, PairedModel)
    assert isinstance(loaded.processor, PairedProcessor)
    assert torch.equal(loaded.model.weight, model.weight)
