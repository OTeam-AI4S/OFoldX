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

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from ofoldx.artifacts import save_model


@pytest.mark.parametrize("existing", [False, True])
def test_processor_architecture_rejected_before_repository_mutation(
    tmp_path: Path, repository_model, repository_processor, existing: bool
) -> None:
    destination = tmp_path / "repository"
    before = {}
    if existing:
        save_model(destination, model=repository_model, processor=repository_processor)
        (destination / "pipeline.json").write_text('{"pipeline": "existing-workflow"}\n', encoding="utf-8")
        before = {path.name: path.read_bytes() for path in destination.iterdir()}
    repository_processor.config.architecture = "test_repository"

    with pytest.raises(ValueError, match="Processor config must not contain repository architecture metadata"):
        save_model(destination, model=repository_model, processor=repository_processor)

    if existing:
        assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    else:
        assert not destination.exists()


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
