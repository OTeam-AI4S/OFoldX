r"""Write standard OFoldX model repositories."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

from torch import nn

from .models.registry import MODELS


def save_model(
    directory: str | PathLike[str],
    *,
    model: nn.Module,
    processor: object,
    max_shard_size: int | str = "5GB",
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    r"""Write one registered Model/Processor repository, and optionally publish it.

    Saving is local-only by default. When ``repo_id`` is given, the fully written and
    validated repository is uploaded to that Hugging Face Hub repo via
    [`ofoldx.hub`][ofoldx.hub]. Nothing is published unless a caller passes an explicit
    ``repo_id``; ``token`` (or the ambient Hugging Face credential) supplies authentication.
    No organization or token is assumed.
    """

    from huggingface_hub import save_torch_state_dict

    registration = MODELS.for_pair(model, processor)
    model_values = dict(model.config)  # type: ignore[attr-defined]
    if "architecture" in model_values:
        raise ValueError("Model config must not contain architecture")
    model_values["architecture"] = registration.id

    model_json = json.dumps(model_values, indent=2, sort_keys=True, allow_nan=False) + "\n"
    config_class = getattr(registration.processor, "config_class", None)
    processor_json = None
    if config_class is not None:
        processor_values = dict(processor.config)  # type: ignore[attr-defined]
        if "architecture" in processor_values:
            raise ValueError("Processor config must not contain architecture")
        processor_json = json.dumps(processor_values, indent=2, sort_keys=True, allow_nan=False) + "\n"

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
        from .hub import _upload_repository

        _upload_repository(
            destination,
            repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
        )
