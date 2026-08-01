from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from ofoldx.artifacts import save_model


@pytest.mark.parametrize(
    ("max_shard_size", "sharded"),
    [("5GB", False), (16, True)],
)
def test_save_model(
    tmp_path: Path,
    repository_model,
    repository_processor,
    max_shard_size: int | str,
    sharded: bool,
) -> None:
    save_model(
        tmp_path,
        model=repository_model,
        processor=repository_processor,
        max_shard_size=max_shard_size,
    )

    assert (tmp_path / "model.safetensors").is_file() == (not sharded)
    assert (tmp_path / "model.safetensors.index.json").is_file() == sharded
    assert json.loads((tmp_path / "model.json").read_text()) == {
        "architecture": "test_repository",
        "width": 3,
    }
    assert json.loads((tmp_path / "processor.json").read_text()) == {
        "dtype": "float16",
        "scale": 2,
    }
    if not sharded:
        assert set(load_file(tmp_path / "model.safetensors")) == set(repository_model.state_dict())
