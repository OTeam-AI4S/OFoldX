from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ofoldx.model import Model
from ofoldx.pipelines import PIPELINES, PairedPipeline
from ofoldx.processor import Processor


class RepositoryPairedPipeline(PairedPipeline[Model, Processor]):
    model_architecture = "test_repository"


PIPELINES.register("test_paired_pipeline", pipeline=RepositoryPairedPipeline)


class TestPairedPipeline:
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)

        pipeline.save(tmp_path)
        loaded = RepositoryPairedPipeline.from_repository(tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_paired_pipeline"
        }
        assert type(loaded.model) is type(repository_model)
        assert type(loaded.processor) is type(repository_processor)
        assert loaded.processor.config is not None
        assert repository_processor.config is not None
        assert dict(loaded.processor.config) == dict(repository_processor.config)
        assert loaded.processor.dtype is torch.float16
        assert loaded.model.backend == "eager"
        for key, expected in repository_model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], expected)
