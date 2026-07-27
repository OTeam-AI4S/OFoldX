r"""
Write standard OFoldX model repositories.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from .model import Model
from .models.registry import MODELS
from .processor import Processor

MODEL_REPOSITORY_FILES = (
    "model.json",
    "processor.json",
    "pipeline.json",
    "model.safetensors",
    "model.safetensors.index.json",
)


def _config_json(config: Mapping[str, object], *, architecture: str | None = None) -> str:
    values = dict(config)
    if "architecture" in values:
        raise ValueError("Config must not contain repository architecture metadata")
    if architecture is not None:
        values["architecture"] = architecture
    return json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _processor_json(processor: Processor, config: Mapping[str, object] | None) -> str:
    values = {} if config is None else dict(config)
    if "dtype" in values:
        raise ValueError("Processor config must not contain repository dtype metadata")
    values["dtype"] = str(processor.dtype).removeprefix("torch.")
    return json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n"


def clear_model_subrepositories(
    directory: str | PathLike[str],
    *,
    keep: set[Path] | None = None,
) -> None:
    r"""
    Remove OFoldX-managed artifacts from stale descendant pair repositories.

    Args:
        directory: Root model repository being replaced.
        keep: Descendant directories owned by the new repository layout.
    """

    destination = Path(directory).expanduser()
    if not destination.is_dir():
        return
    retained = set() if keep is None else keep
    for model_path in destination.rglob("model.json"):
        child = model_path.parent
        if child == destination or child.is_symlink() or child in retained:
            continue
        if not (child / "processor.json").is_file():
            continue
        for filename in MODEL_REPOSITORY_FILES:
            (child / filename).unlink(missing_ok=True)
        for shard in child.glob("model-*-of-*.safetensors"):
            shard.unlink()


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
    r"""
    Save a registered model and processor as an OFoldX repository.

    Args:
        directory: Local model repository to create or update.
        model: Registered model whose config and state dictionary are saved.
        processor: Processor registered with ``model``.
        max_shard_size: Maximum safetensors shard size.
        repo_id: Optional Hugging Face Hub repository ID to upload after saving.
        token: Hugging Face access token used for publication.
        private: Whether a newly created Hub repository is private.
        commit_message: Optional Hub commit message.

    Raises:
        RuntimeError: If safetensors serialization does not preserve the complete state
            dictionary.
    """

    from huggingface_hub import save_torch_state_dict

    registration = MODELS.for_pair(model, processor)
    model_json = _config_json(model.config, architecture=registration.id)
    processor_json = _processor_json(processor, processor.config)

    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pipeline.json").unlink(missing_ok=True)
    (destination / "model.json").write_text(
        model_json,
        encoding="utf-8",
    )
    processor_path = destination / "processor.json"
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
