from __future__ import annotations

from pathlib import Path

import torch

from ofoldx import AutoModel, AutoProcessor
from ofoldx.artifacts import save_model


class TestAutoModel:
    def test_from_config(self, repository_model) -> None:
        config = type(repository_model.config)({"width": 2})

        model = AutoModel.from_config(config, device="cpu", dtype=torch.float64, backend="eager")

        assert type(model) is type(repository_model)
        assert model.weight.shape == (2,)
        assert model.weight.dtype == torch.float64
        assert model.backend == "eager"
        assert model.training

    def test_from_pretrained(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        model = AutoModel.from_pretrained(tmp_path, device="cpu", dtype=torch.float64, backend="eager")

        assert type(model) is type(repository_model)
        assert model.weight.dtype == torch.float64
        assert model.backend == "eager"
        assert not model.training


class TestAutoProcessor:
    def test_from_pretrained(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        processor = AutoProcessor.from_pretrained(tmp_path, device="cpu", dtype=torch.float32)

        assert type(processor) is type(repository_processor)
        assert processor.config.scale == 2
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32
