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

r"""
Write standard OFoldX model repositories.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

from .model import Model
from .models.registry import MODELS
from .processor import Processor


def save_model(
    directory: str | PathLike[str],
    *,
    model: Model,
    processor: Processor,
    max_shard_size: int | str = "5GB",
) -> None:
    r"""
    Save a registered model and processor as a local OFoldX repository.

    Args:
        directory: Local model repository to create or update.
        model: Registered model whose config and owned repository parameters are saved.
        processor: Processor registered with ``model``.
        max_shard_size: Maximum safetensors shard size.

    Raises:
        ValueError: If Model or Processor configuration contains reserved repository metadata.
        RuntimeError: If safetensors serialization does not preserve the complete
            repository state dictionary.
    """

    from huggingface_hub import save_torch_state_dict

    registration = MODELS.for_pair(model, processor)

    model_values = model.repository_config()
    if "architecture" in model_values:
        raise ValueError("Model config must not contain repository architecture metadata")
    model_values["architecture"] = registration.id

    processor_values = {} if processor.config is None else dict(processor.config)
    if "architecture" in processor_values:
        raise ValueError("Processor config must not contain repository architecture metadata")
    if "dtype" in processor_values:
        raise ValueError("Processor config must not contain repository dtype metadata")
    processor_values["dtype"] = str(processor.dtype).removeprefix("torch.")

    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pipeline.json").unlink(missing_ok=True)
    (destination / "model.json").write_text(
        json.dumps(model_values, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "processor.json").write_text(
        json.dumps(processor_values, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    state_dict = model.repository_state_dict()
    expected_keys = set(state_dict)
    save_torch_state_dict(
        state_dict,
        destination,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    single_path = destination / "model.safetensors"
    index_path = destination / "model.safetensors.index.json"
    if not expected_keys and not single_path.exists() and not index_path.exists():
        from safetensors.torch import save_file

        save_file({}, single_path, metadata={"format": "pt"})

    single = single_path.exists()
    sharded = index_path.exists()
    if single == sharded:
        raise RuntimeError(
            "saving model weights must produce exactly one of model.safetensors or model.safetensors.index.json"
        )

    if single:
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as weights:
            stored_keys = set(weights.keys())
    else:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        stored_keys = set(index["weight_map"])
    if stored_keys != expected_keys:
        raise RuntimeError("safetensors elided repository state-dict keys; the repository cannot be round-tripped")
