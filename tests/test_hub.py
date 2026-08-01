from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ofoldx.artifacts import save_model
from ofoldx.hub import ResolvedRepository, load_model, load_processor


class TestLoadModel:
    @pytest.mark.parametrize("max_shard_size", ["5GB", 24])
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model,
        repository_processor,
        max_shard_size: int | str,
    ) -> None:
        save_model(
            tmp_path,
            model=repository_model,
            processor=repository_processor,
            max_shard_size=max_shard_size,
        )

        model = load_model(ResolvedRepository(tmp_path))
        assert type(model) is type(repository_model)
        assert not model.training
        assert torch.equal(model.weight, repository_model.weight)
        assert torch.equal(model.bias, repository_model.bias)
        assert model.weight.dtype == repository_model.weight.dtype

    def test_strict_weights(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)
        path = tmp_path / "model.safetensors"
        state_dict = load_file(path)
        weight = state_dict["weight"].clone()
        save_file({"weight": weight}, path, metadata={"format": "pt"})

        with pytest.raises(RuntimeError, match="Missing key"):
            load_model(ResolvedRepository(tmp_path))


class TestLoadProcessor:
    def test_load(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        processor = load_processor(ResolvedRepository(tmp_path))

        assert type(processor) is type(repository_processor)
        assert processor.config.scale == 2
        assert processor.dtype == torch.float16
