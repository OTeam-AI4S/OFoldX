r"""
Resolve, publish, and load OFoldX model repositories.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, NamedTuple, NoReturn, cast

import chanfig
import torch

from .model import Model
from .models.registry import (
    MODELS,
    ModelRegistration,
)
from .processor import Processor
from .references import classify_reference

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

MODEL_METADATA = ("model.json",)
PROCESSOR_METADATA = ("model.json", "processor.json")
MODEL_WEIGHT_METADATA = (
    "model.safetensors",
    "model.safetensors.index.json",
)


class ResolvedRepository(NamedTuple):
    r"""
    A local model repository with optional Hub provenance.

    Attributes:
        path: Local repository or downloaded snapshot path.
        repository: Hugging Face Hub repository ID, when downloaded from the Hub.
        commit: Immutable Hub commit that produced ``path``.
    """

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
    r"""
    Load a finite, duplicate-free JSON object.

    Args:
        path: JSON file to read.

    Returns:
        The decoded object.

    Raises:
        TypeError: If the document root is not an object.
        ValueError: If keys are duplicated or numbers are non-finite.
    """

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
    r"""
    Resolve a local model repository or Hugging Face Hub repository ID.

    Args:
        reference: Existing or explicitly written local path, or a Hub repository ID.
        revision: Hub revision to download.
        allow_patterns: Optional repository file patterns to download.
        token: Hugging Face access token used for private repositories.

    Returns:
        The local repository path and, for Hub snapshots, its repository ID and commit.
    """

    classified = classify_reference(reference)
    if isinstance(classified, Path):
        if revision is not None:
            raise ValueError("revision is valid only for a Hub repository")
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
    r"""
    Download files from a Hugging Face Hub model repository.

    Args:
        repo_id: Hugging Face Hub repository ID.
        revision: Revision to download.
        allow_patterns: Optional repository file patterns to download.
        token: Hugging Face access token used for private repositories.

    Returns:
        The downloaded snapshot path, repository ID, and resolved commit.
    """

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
) -> CommitInfo:
    r"""
    Upload a local model repository to the Hugging Face Hub.

    Args:
        directory: Local model repository to upload.
        repo_id: Destination Hub repository ID.
        token: Hugging Face access token used for publication.
        private: Whether a newly created repository is private.
        revision: Destination branch or revision.
        commit_message: Optional commit message.
        allow_patterns: Optional local file patterns to upload.
        delete_patterns: Remote file patterns to delete before upload. By default,
            a complete upload removes repository-managed metadata and weights absent
            locally, while a selective upload leaves unselected remote files untouched.
            User-owned files such as model cards are retained in either case.

    Returns:
        Information about the created Hub commit.
    """

    from huggingface_hub import HfApi

    folder = Path(directory).expanduser()
    resolved_allow = None if allow_patterns is None else list(allow_patterns)
    resolved_delete = (
        None
        if delete_patterns is None or (allow_patterns is not None and delete_patterns is MIRROR_REPOSITORY)
        else list(delete_patterns)
    )

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
    r"""
    Download all weight files from the commit that supplied repository metadata.

    Args:
        repository: Resolved local repository or Hub metadata snapshot.
        token: Hugging Face access token used for private repositories.

    Returns:
        A repository containing one complete safetensors state dictionary.

    Raises:
        ValueError: If weight metadata is missing or ambiguous.
    """

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
    r"""
    Read model metadata and resolve its registered architecture.

    Args:
        path: Local model repository.

    Returns:
        The architecture registration and architecture-specific config values.
    """

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
) -> Model:
    r"""
    Load the model declared by a resolved repository.

    Args:
        repository: Resolved model repository containing metadata and weights.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype applied after loading.
        token: Hugging Face access token used to hydrate private Hub weights.

    Returns:
        The registered model in evaluation mode.
    """

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
    config: chanfig.Config,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Model:
    r"""
    Instantiate a registered architecture and load its state dictionary.

    Args:
        repository: Local repository containing complete safetensors weights.
        registration: Architecture registration to instantiate.
        config: Architecture configuration.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype applied after loading.

    Returns:
        The model with strictly loaded weights in evaluation mode.
    """

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
) -> Processor:
    r"""
    Load the processor declared by a resolved repository.

    Args:
        repository: Resolved model repository containing processor metadata.
        device: Device used for produced tensors.
        dtype: Floating-point dtype used for produced tensors.

    Returns:
        The registered processor.
    """

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
) -> Processor:
    r"""
    Instantiate a processor from a registration and repository metadata.

    Args:
        repository: Resolved model repository containing processor metadata.
        registration: Model registration that owns the processor type.
        device: Device used for produced tensors.
        dtype: Floating-point dtype used for produced tensors.

    Returns:
        The configured processor.
    """

    config_path = repository.path / "processor.json"
    values = load_json_object(config_path)
    if "architecture" in values:
        raise ValueError("processor.json must not repeat the model architecture")
    saved_dtype_name = values.pop("dtype")
    if not isinstance(saved_dtype_name, str):
        raise TypeError("processor.json dtype must be a torch dtype name")
    saved_dtype = getattr(torch, saved_dtype_name, None)
    if not isinstance(saved_dtype, torch.dtype):
        raise ValueError(f"unsupported processor dtype {saved_dtype_name!r}")
    resolved_dtype = saved_dtype if dtype is None else dtype

    processor_config = registration.processor.config_class
    if processor_config is None:
        if values:
            raise ValueError("a config-less Processor may persist only its dtype")
        return registration.processor(device=device, dtype=resolved_dtype)

    processor = cast(Callable[..., Processor], registration.processor)
    return processor(processor_config(values), device=device, dtype=resolved_dtype)


def _weight_map(path: Path) -> dict[str, str]:
    r"""
    Read indexed weight ownership and reject paths outside the repository.
    """

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
