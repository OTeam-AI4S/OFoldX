from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ofoldx.model import Model
from ofoldx.pipelines import PIPELINES, CompositePair, CompositePipeline, PairedPipeline
from ofoldx.processor import Processor


class CompositeChildPipeline(PairedPipeline[Model, Processor]):
    model_architecture = "test_repository"


class RepositoryCompositePipeline(CompositePipeline):
    pair_layout = (
        CompositePair("", "test_repository", "model", "processor"),
        CompositePair("child", "test_repository", "child", None, CompositeChildPipeline),
    )

    def __init__(
        self,
        *,
        model: Model,
        processor: Processor,
        child: CompositeChildPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.child = child
        self._validate_components()


PIPELINES.register("test_composite_pipeline", pipeline=RepositoryCompositePipeline)


class TestCompositePipeline:
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        model_type = type(repository_model)
        model_config_type = type(repository_model.config)
        processor_type = type(repository_processor)
        processor_config_type = type(repository_processor.config)
        child_model = model_type(model_config_type({"width": 2}))
        child_processor = processor_type(processor_config_type({"scale": 7}), dtype=torch.float64)
        pipeline = RepositoryCompositePipeline(
            model=repository_model,
            processor=repository_processor,
            child=CompositeChildPipeline(model=child_model, processor=child_processor),
        )

        pipeline.save(tmp_path)
        loaded = RepositoryCompositePipeline.from_repository(tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_composite_pipeline"
        }
        assert not (tmp_path / "child" / "pipeline.json").exists()
        assert loaded.model.weight.shape == repository_model.weight.shape
        assert loaded.child.model.weight.shape == child_model.weight.shape
        assert loaded.processor.config is not None
        assert loaded.child.processor.config is not None
        assert repository_processor.config is not None
        assert child_processor.config is not None
        assert dict(loaded.processor.config) == dict(repository_processor.config)
        assert dict(loaded.child.processor.config) == dict(child_processor.config)
        assert loaded.processor.dtype is torch.float16
        assert loaded.child.processor.dtype is torch.float64
        assert loaded.model.backend == "eager"
        assert loaded.child.model.backend == "eager"
        for key, expected in repository_model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], expected)
        for key, expected in child_model.state_dict().items():
            torch.testing.assert_close(loaded.child.model.state_dict()[key], expected)
