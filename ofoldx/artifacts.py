r"""Write standard OFoldX model repositories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Protocol, cast

import torch

from .model import Model
from .models.registry import MODELS
from .processor import Processor


class SaveableModel(Protocol):
    r"""Structural model surface used only to compose task-specific Protocols."""

    @property
    def config(self) -> Mapping[str, object]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object: ...


class SaveableProcessor(Protocol):
    r"""Configured Processor state written to ``processor.json``."""

    @property
    def config(self) -> Mapping[str, object]: ...


def _config_json(config: Mapping[str, object], *, architecture: str | None = None) -> str:
    values = dict(config)
    if "architecture" in values:
        raise ValueError("Config must not contain repository architecture metadata")
    if architecture is not None:
        values["architecture"] = architecture
    return json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n"


def save_model(
    directory: str | PathLike[str],
    *,
    model: Model,
    processor: Processor,
    max_shard_size: int | str = "5GB",
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    r"""Write one registered Model/Processor repository, and optionally publish it.

    Saving is local-only by default. When ``repo_id`` is given, the fully written and
    validated repository is uploaded to that Hugging Face Hub repo. Nothing is published
    unless a caller passes an explicit ``repo_id``; ``token`` (or the ambient Hugging Face
    credential) supplies authentication. No organization or token is assumed.
    """

    from huggingface_hub import save_torch_state_dict

    registration = MODELS.for_pair(model, processor)
    model_json = _config_json(model.config, architecture=registration.id)
    processor_config = registration.processor_config
    processor_json = None
    if processor_config is not None:
        processor_json = _config_json(cast(SaveableProcessor, processor).config)

    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pipeline.json").unlink(missing_ok=True)
    (destination / "model.json").write_text(
        model_json,
        encoding="utf-8",
    )
    processor_path = destination / "processor.json"
    if processor_json is None:
        processor_path.unlink(missing_ok=True)
    else:
        processor_path.write_text(processor_json, encoding="utf-8")
    state_dict = model.state_dict()
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
            "saving model weights must produce exactly one of model.safetensors " "or model.safetensors.index.json"
        )

    if single:
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as weights:
            stored_keys = set(weights.keys())
    else:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        stored_keys = set(index["weight_map"])
    if stored_keys != expected_keys:
        raise RuntimeError("safetensors elided model state-dict keys; the repository cannot be round-tripped")

    if repo_id is not None:
        from .hub import upload_repository

        upload_repository(
            destination,
            repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
        )
