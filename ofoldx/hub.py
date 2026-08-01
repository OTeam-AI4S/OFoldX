r"""
Resolve, publish, and load OFoldX model repositories.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from .model import Model
from .models.registry import MODELS, ModelRegistration
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


def load_json_object(path: Path) -> dict[str, object]:
    r"""
    Load a JSON object.

    Args:
        path: JSON file to read.

    Returns:
        The decoded object.

    Raises:
        TypeError: If the document root is not an object.
    """

    values = json.loads(path.read_text(encoding="utf-8"))
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
    return ResolvedRepository(snapshot, repo_id, snapshot.name)


# Re-publishing removes only artifacts owned by OFoldX. Model cards and arbitrary
# user files in the same repository remain untouched.
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
) -> CommitInfo:
    r"""
    Upload a complete local model repository to the Hugging Face Hub.

    Args:
        directory: Local model repository to upload.
        repo_id: Destination Hub repository ID.
        token: Hugging Face access token used for publication.
        private: Whether a newly created repository is private.
        revision: Destination branch or revision.
        commit_message: Optional commit message.

    Returns:
        Information about the created Hub commit.
    """

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(Path(directory).expanduser()),
        revision=revision,
        commit_message=commit_message,
        delete_patterns=list(MIRROR_REPOSITORY),
    )


def hydrate_weights(
    repository: ResolvedRepository,
    *,
    subdirectory: str = "",
    token: str | None = None,
) -> ResolvedRepository:
    r"""
    Download all weight files from the commit that supplied repository metadata.

    Args:
        repository: Resolved local repository or Hub metadata snapshot root.
        subdirectory: Fixed repository-relative directory containing the pair.
        token: Hugging Face access token used for private repositories.

    Returns:
        A repository containing one complete safetensors state dictionary.

    Raises:
        ValueError: If weight metadata is missing or ambiguous.
    """

    prefix = PurePosixPath(subdirectory)
    weight_patterns = tuple(str(prefix / filename) for filename in MODEL_WEIGHT_METADATA)
    hydrated_root = (
        repository
        if repository.repository is None
        else resolve_hub_repository(
            repository.repository,
            revision=repository.commit,
            allow_patterns=weight_patterns,
            token=token,
        )
    )
    hydrated = ResolvedRepository(
        hydrated_root.path / subdirectory,
        hydrated_root.repository,
        hydrated_root.commit,
    )
    single = (hydrated.path / "model.safetensors").exists()
    index_path = hydrated.path / "model.safetensors.index.json"
    sharded = index_path.exists()
    if single == sharded:
        raise ValueError("model repository must contain exactly one of model.safetensors or its index")
    if repository.repository is None or single:
        return hydrated

    from huggingface_hub import hf_hub_download

    for filename in sorted(set(_weight_map(index_path).values())):
        hf_hub_download(
            repo_id=repository.repository,
            filename=str(prefix / filename),
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
    return MODELS.lookup_model(architecture), values


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
    config = registration.model.config_class(values)
    with torch.device("meta"):
        model = registration.model(config)
    return load_registered_model(
        hydrate_weights(repository, token=token),
        model,
        device=device,
        dtype=dtype,
    )


def load_registered_model(
    repository: ResolvedRepository,
    model: Model,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Model:
    r"""
    Load a registered architecture's state dictionary into its meta instance.

    Args:
        repository: Local repository containing complete safetensors weights.
        model: Exact architecture instance constructed on the meta device.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype applied after loading.

    Returns:
        The model with strictly loaded weights in evaluation mode.
    """

    from safetensors.torch import load_file

    single_path = repository.path / "model.safetensors"
    index_path = repository.path / "model.safetensors.index.json"

    if single_path.exists():
        state_dict = load_file(single_path, device="cpu")
    else:
        weight_map = _weight_map(index_path)
        state_dict = {}
        for filename in sorted(set(weight_map.values())):
            for key, value in load_file(repository.path / filename, device="cpu").items():
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

    values = load_json_object(repository.path / "processor.json")
    if "architecture" in values:
        raise ValueError("processor.json must not repeat the model architecture")
    saved_dtype_name = values.pop("dtype", None)
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
    config = processor_config(values)
    return processor(config, device=device, dtype=resolved_dtype)


def _weight_map(path: Path) -> dict[str, str]:
    r"""
    Read indexed weight ownership and reject paths outside the repository.
    """

    raw_weight_map = load_json_object(path).get("weight_map")
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
