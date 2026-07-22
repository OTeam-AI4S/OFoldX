r"""OFoldX model-repository resolution and loading.

The Hugging Face Hub supplies snapshots and safetensors mechanics. OFoldX owns the
repository schema, architecture dispatch, Model/Processor pairing, and construction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple, NoReturn, cast

import torch
from torch import nn

from .models.registry import (
    MODELS,
    ModelRegistration,
)
from .references import classify_reference

MODEL_METADATA = ("model.json",)
PROCESSOR_METADATA = ("model.json", "processor.json")
MODEL_WEIGHT_METADATA = (
    "model.safetensors",
    "model.safetensors.index.json",
)


class ResolvedRepository(NamedTuple):
    path: Path
    repository: str | None = None
    commit: str | None = None


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is not permitted")
    return parsed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate JSON object key {key!r}")
        values[key] = value
    return values


def load_json_object(path: Path) -> dict[str, object]:
    r"""Load a strict JSON object used to route or construct a repository."""

    values = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_json_float,
    )
    if not isinstance(values, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return values


def resolve_repository(
    reference: str | PathLike[str],
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
) -> ResolvedRepository:
    r"""Resolve local intent or download selected files from one Hub snapshot."""

    classified = classify_reference(reference)
    if isinstance(classified, Path):
        return ResolvedRepository(classified)
    return resolve_hub_repository(
        classified,
        revision=revision,
        allow_patterns=allow_patterns,
        token=token,
    )


def resolve_hub_repository(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
) -> ResolvedRepository:
    r"""Download selected files from one known Hub repository revision."""

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
            token=token,
        )
    )
    commit = snapshot.name
    return ResolvedRepository(snapshot, repo_id, commit)


# Re-publishing removes only files written by OFoldX, leaving model cards and arbitrary
# user-owned files in the same repository untouched.
MIRROR_REPOSITORY: tuple[str, ...] = (
    "model.safetensors",
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
    "*/model.safetensors",
    "*/model-*-of-*.safetensors",
    "*/model.safetensors.index.json",
    "model.json",
    "*/model.json",
    "processor.json",
    "*/processor.json",
    "pipeline.json",
    "*/pipeline.json",
)


def upload_repository(
    directory: str | PathLike[str],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_message: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    delete_patterns: Sequence[str] | None = MIRROR_REPOSITORY,
) -> object:
    r"""Upload one local model repository to a Hub repo, creating it if absent.

    The upload counterpart to :func:`resolve_hub_repository`: a thin wrapper over
    ``HfApi.create_repo``/``HfApi.upload_folder`` that pushes a fully written repository
    directory (``model.json``/``processor.json``/``pipeline.json`` and its safetensors) to
    ``repo_id``. Nothing is uploaded unless a caller passes an explicit ``repo_id``;
    authentication comes only from the caller's ``token`` or the ambient Hugging Face
    credential. No organization, repository, or token is assumed here.

    Re-publishing mirrors repository-managed files by default: ``delete_patterns`` applies to
    repository-managed files (see :data:`MIRROR_REPOSITORY`) onto ``repo_id`` so a re-publish
    deletes managed remote files that are absent locally (an orphaned ``pipeline.json``, a stale
    ``model.safetensors.index.json``, or old weight shards) instead of leaving them to misroute the
    loader. The mirror is scoped to those managed names, so a remote README / model card that the
    local ``save`` does not write is never deleted by a clean re-upload. ``upload_folder`` never
    deletes a remote file that this upload also writes (deletions are deduplicated against the
    additions) and always preserves ``.gitattributes``. ``allow_patterns`` and
    ``delete_patterns`` remain independent caller controls. Pass ``delete_patterns=None``
    to disable deletion entirely, or an explicit pattern sequence to scope it.
    """

    from huggingface_hub import HfApi

    folder = Path(directory).expanduser()
    resolved_allow = None if allow_patterns is None else list(allow_patterns)
    resolved_delete = None if delete_patterns is None else list(delete_patterns)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        revision=revision,
        commit_message=commit_message,
        allow_patterns=resolved_allow,
        delete_patterns=resolved_delete,
    )


def hydrate_weights(
    repository: ResolvedRepository,
    *,
    token: str | None = None,
) -> ResolvedRepository:
    r"""Hydrate weights from the exact commit that supplied Hub metadata."""

    hydrated = (
        repository
        if repository.repository is None
        else resolve_hub_repository(
            repository.repository,
            revision=repository.commit,
            allow_patterns=MODEL_WEIGHT_METADATA,
            token=token,
        )
    )
    single = (hydrated.path / "model.safetensors").exists()
    index_path = hydrated.path / "model.safetensors.index.json"
    sharded = index_path.exists()
    if single == sharded:
        raise ValueError("model repository must contain exactly one of model.safetensors or its index")
    if repository.repository is None or single:
        return hydrated

    from huggingface_hub import hf_hub_download

    filenames = sorted(set(_weight_map(index_path).values()))
    for filename in filenames:
        hf_hub_download(
            repo_id=repository.repository,
            filename=filename,
            revision=repository.commit,
            token=token,
        )
    return hydrated


def model_values(path: Path) -> tuple[ModelRegistration, dict[str, object]]:
    values = load_json_object(path / "model.json")
    architecture = values.pop("architecture", None)
    if not isinstance(architecture, str):
        raise TypeError("model.json architecture must be a string")
    registered = MODELS.lookup_model(architecture)
    return registered, values


def load_model(
    repository: ResolvedRepository,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    token: str | None = None,
) -> nn.Module:
    registration, values = model_values(repository.path)
    return load_registered_model(
        hydrate_weights(repository, token=token),
        registration,
        registration.model_config(values),
        device=device,
        dtype=dtype,
    )


def load_registered_model(
    repository: ResolvedRepository,
    registration: ModelRegistration,
    config: object,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    from safetensors.torch import load_file

    single_path = repository.path / "model.safetensors"
    index_path = repository.path / "model.safetensors.index.json"

    with torch.device("meta"):
        model = registration.model(config)

    if single_path.exists():
        state_dict = load_file(single_path, device="cpu")
    else:
        weight_map = _weight_map(index_path)
        state_dict = {}
        weight_paths = tuple((filename, repository.path / filename) for filename in sorted(set(weight_map.values())))
        for filename, path in weight_paths:
            for key, value in load_file(path, device="cpu").items():
                if key in state_dict:
                    raise ValueError(f"model weight {key!r} occurs in multiple shards")
                expected_filename = weight_map.get(key)
                if expected_filename is None:
                    raise ValueError(f"model weight {key!r} is absent from the index")
                if expected_filename != filename:
                    raise ValueError(f"model weight {key!r} is stored in a different shard than its index entry")
                state_dict[key] = value
        if set(state_dict) != set(weight_map):
            raise ValueError("model shard keys do not match index weight_map keys")

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_processor(
    repository: ResolvedRepository,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> object:
    registration, _ = model_values(repository.path)
    return load_registered_processor(
        repository,
        registration,
        device=device,
        dtype=dtype,
    )


def load_registered_processor(
    repository: ResolvedRepository,
    registration: ModelRegistration,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> object:
    processor_config = registration.processor_config
    config_path = repository.path / "processor.json"
    if processor_config is None:
        if config_path.exists():
            raise ValueError("processor.json must be absent for a config-less Processor")
        return registration.processor(device=device, dtype=dtype)

    values = load_json_object(config_path)
    if "architecture" in values:
        raise ValueError("processor.json must not repeat the model architecture")
    return registration.processor(processor_config(values), device=device, dtype=dtype)


def _weight_map(path: Path) -> dict[str, str]:
    r"""Read indexed weight ownership and reject paths outside the repository."""

    index = load_json_object(path)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise TypeError("model.safetensors.index.json weight_map must be a JSON object")

    weight_map = cast(dict[str, object], raw_weight_map)
    for filename in weight_map.values():
        if not isinstance(filename, str):
            raise TypeError("model.safetensors.index.json weight_map values must be strings")
        posix = PurePosixPath(filename)
        windows = PureWindowsPath(filename)
        if posix.anchor or windows.anchor or ".." in posix.parts or ".." in windows.parts:
            raise ValueError(f"model shard path must stay within the repository: {filename!r}")
    return cast(dict[str, str], weight_map)
