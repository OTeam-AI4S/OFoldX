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
from typing import NamedTuple

import torch
from torch import nn

from .models.registry import (
    _ModelRegistration,
    _registration,
)
from .references import classify_reference

_MODEL_METADATA = ("model.json",)
_PROCESSOR_METADATA = ("model.json", "processor.json")
_MODEL_WEIGHT_METADATA = (
    "model.safetensors",
    "model.safetensors.index.json",
)


class _ResolvedRepository(NamedTuple):
    path: Path
    repository: str | None = None
    commit: str | None = None


def _resolve_repository(
    reference: str | PathLike[str],
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> _ResolvedRepository:
    r"""Resolve local intent or download selected files from one Hub snapshot."""

    classified = classify_reference(reference)
    if isinstance(classified, Path):
        if revision is not None:
            raise ValueError("revision is valid only for a Hugging Face Hub model repository")
        return _ResolvedRepository(classified)
    return _resolve_hub_repository(
        classified,
        revision=revision,
        allow_patterns=allow_patterns,
    )


def _resolve_hub_repository(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> _ResolvedRepository:
    r"""Download selected files from one known Hub repository revision."""

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
        )
    )
    commit = snapshot.name
    return _ResolvedRepository(snapshot, repo_id, commit)


# Sentinel default for :func:`_upload_repository`'s ``delete_patterns``: mirror the local
# repository onto the remote (delete every remote file absent locally). A caller that passes
# ``delete_patterns=None`` explicitly opts out and keeps the append-only upload behavior.
_MIRROR_REPOSITORY: tuple[str, ...] = ("*",)


def _upload_repository(
    directory: str | PathLike[str],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_message: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    delete_patterns: Sequence[str] | None = _MIRROR_REPOSITORY,
) -> object:
    r"""Upload one local model repository to a Hub repo, creating it if absent.

    The upload counterpart to :func:`_resolve_hub_repository`: a thin wrapper over
    ``HfApi.create_repo``/``HfApi.upload_folder`` that pushes a fully written repository
    directory (``model.json``/``processor.json``/``pipeline.json`` and its safetensors) to
    ``repo_id``. Nothing is uploaded unless a caller passes an explicit ``repo_id``;
    authentication comes only from the caller's ``token`` or the ambient Hugging Face
    credential. No organization, repository, or token is assumed here.

    Re-publishing is mirror-safe by default: ``delete_patterns`` defaults to mirroring the
    local repository onto ``repo_id`` so a re-publish deletes remote files that are absent
    locally (an orphaned ``pipeline.json``, a stale ``model.safetensors.index.json``, or old
    weight shards) instead of leaving them to misroute the loader. ``upload_folder`` never
    deletes a remote file that this upload also writes (deletions are deduplicated against the
    additions) and always preserves ``.gitattributes``. When ``allow_patterns`` restricts the
    upload to a subset, the default mirror is narrowed to that same allowlist so files the
    caller deliberately withheld are never deleted. Pass ``delete_patterns=None`` to disable
    deletion entirely (append-only), or an explicit pattern sequence to scope it.
    """

    from huggingface_hub import HfApi

    folder = Path(directory).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"model repository to upload is not a directory: {folder}")
    if not repo_id:
        raise ValueError("uploading a model repository requires an explicit repo id")

    resolved_allow = None if allow_patterns is None else list(allow_patterns)
    if delete_patterns is _MIRROR_REPOSITORY and resolved_allow is not None:
        resolved_delete: list[str] | None = resolved_allow
    else:
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


def _hydrate_weights(repository: _ResolvedRepository) -> _ResolvedRepository:
    r"""Hydrate weights from the exact commit that supplied Hub metadata."""

    hydrated = (
        repository
        if repository.repository is None
        else _resolve_hub_repository(
            repository.repository,
            revision=repository.commit,
            allow_patterns=_MODEL_WEIGHT_METADATA,
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
        )
    return hydrated


def _model_values(path: Path) -> tuple[_ModelRegistration, dict[str, object]]:
    values = _read_json(path / "model.json")
    if not isinstance(values, dict):
        raise TypeError("model.json must contain a JSON object")
    architecture = values.pop("architecture", None)
    if not isinstance(architecture, str):
        raise TypeError("model.json architecture must be a string")
    registered = _registration(architecture)
    return registered, values


def _model_config(path: Path) -> tuple[_ModelRegistration, object]:
    registered, values = _model_values(path)
    return registered, registered.model_config(values)


def _load_model(
    repository: _ResolvedRepository,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    registration, config = _model_config(repository.path)
    return _load_registered_model(
        _hydrate_weights(repository),
        registration,
        config,
        device=device,
        dtype=dtype,
    )


def _load_registered_model(
    repository: _ResolvedRepository,
    registration: _ModelRegistration,
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
        owners: dict[str, str] = {}
        weight_paths = tuple((filename, repository.path / filename) for filename in sorted(set(weight_map.values())))
        for filename, path in weight_paths:
            for key, value in load_file(path, device="cpu").items():
                if key in state_dict:
                    raise ValueError(f"model weight {key!r} occurs in multiple shards")
                state_dict[key] = value
                owners[key] = filename
        if set(owners) != set(weight_map):
            raise ValueError("model shard keys do not match index weight_map keys")
        for key, filename in weight_map.items():
            if owners[key] != filename:
                raise ValueError(f"model weight {key!r} is stored in a different shard than its index entry")

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _load_processor(
    repository: _ResolvedRepository,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> object:
    registration, _ = _model_values(repository.path)
    return _load_registered_processor(
        repository,
        registration,
        device=device,
        dtype=dtype,
    )


def _load_registered_processor(
    repository: _ResolvedRepository,
    registration: _ModelRegistration,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> object:
    config_class = getattr(registration.processor, "config_class", None)
    config_path = repository.path / "processor.json"
    if config_class is None:
        if config_path.exists():
            raise ValueError("processor.json must be absent for a config-less Processor")
        return registration.processor(device=device, dtype=dtype)

    values = _read_json(config_path)
    if not isinstance(values, dict):
        raise TypeError("processor.json must contain a JSON object")
    if "architecture" in values:
        raise ValueError("processor.json must not repeat the model architecture")
    return registration.processor(config_class(values), device=device, dtype=dtype)


def _read_json(path: Path) -> object:
    def parse_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path.name} contains out-of-range JSON number {value}")
        return number

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path.name} contains non-finite JSON value {value}")

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"{path.name} contains duplicate JSON key {key!r}")
            values[key] = value
        return values

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        parse_float=parse_float,
        object_pairs_hook=object_from_pairs,
    )


def _weight_map(path: Path) -> dict[str, str]:
    r"""Read indexed weight ownership and reject paths outside the repository."""

    index = _read_json(path)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("model.safetensors.index.json weight_map must be a JSON object")

    weight_map = index["weight_map"]
    for filename in weight_map.values():
        if not isinstance(filename, str):
            raise TypeError("model.safetensors.index.json weight_map values must be strings")
        posix = PurePosixPath(filename)
        windows = PureWindowsPath(filename)
        if posix.anchor or windows.anchor or ".." in posix.parts or ".." in windows.parts:
            raise ValueError(f"model shard path must stay within the repository: {filename!r}")
    return weight_map
