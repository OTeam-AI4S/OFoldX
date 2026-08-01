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

r"""Load and publish OFoldX Model–Processor repositories."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, cast

import huggingface_hub
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from ..model import Model
from ..models.registry import MODELS, ModelRegistration
from ..processor import Processor
from . import ResolvedRepository, load_json_object, resolve_hub_repository, upload_folder

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

MODEL_METADATA = ("model.json",)
PROCESSOR_METADATA = ("model.json", "processor.json")
MODEL_WEIGHT_METADATA = (
    "model.safetensors",
    "model.safetensors.index.json",
)


# Re-publishing removes only artifacts owned by OFoldX. Other files are kept,
# and a README in the upload directory is uploaded with the rest of the folder.
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

    return upload_folder(
        directory,
        repo_id,
        repo_type="model",
        token=token,
        private=private,
        revision=revision,
        commit_message=commit_message,
        delete_patterns=MIRROR_REPOSITORY,
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

    component_subdirectory = "/".join(part for part in (repository.subdirectory, subdirectory) if part)
    prefix = PurePosixPath(component_subdirectory)
    weight_patterns = tuple(str(prefix / filename) for filename in MODEL_WEIGHT_METADATA)
    if repository.repository is None:
        hydrated_path = repository.path / subdirectory
        hydrated_repository = repository.repository
        hydrated_commit = repository.commit
    else:
        hydrated_root = resolve_hub_repository(
            repository.repository,
            revision=repository.commit,
            allow_patterns=weight_patterns,
            token=token,
        )
        hydrated_path = hydrated_root.path / component_subdirectory
        hydrated_repository = hydrated_root.repository
        hydrated_commit = hydrated_root.commit
    hydrated = ResolvedRepository(
        hydrated_path,
        hydrated_repository,
        hydrated_commit,
        component_subdirectory,
    )
    single = (hydrated.path / "model.safetensors").exists()
    index_path = hydrated.path / "model.safetensors.index.json"
    sharded = index_path.exists()
    if single == sharded:
        raise ValueError("model repository must contain exactly one of model.safetensors or its index")
    if repository.repository is None or single:
        return hydrated

    for filename in sorted(set(_weight_map(index_path).values())):
        huggingface_hub.hf_hub_download(
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
    with init_empty_weights(include_buffers=False):
        model = registration.model(config)
    return load_registered_model(
        hydrate_weights(repository, token=token),
        model,
        device=device,
        dtype=dtype,
        token=token,
    )


def load_registered_model(
    repository: ResolvedRepository,
    model: Model,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    token: str | None = None,
) -> Model:
    r"""
    Load a registered architecture's state dictionary.

    Args:
        repository: Local repository containing the model's owned safetensors weights.
        model: Exact architecture instance receiving the saved state.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype applied after loading.
        token: Access token for explicitly referenced model repositories.

    Returns:
        The model with strictly loaded weights in evaluation mode.
    """

    state_dict = load_model_state_dict(repository.path)

    model.load_repository_state_dict(state_dict, token=token)
    del state_dict
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_model_state_dict(path: str | PathLike[str]) -> dict[str, torch.Tensor]:
    r"""Read one complete safetensors state from a file, directory or shard index."""
    path = Path(path).expanduser()
    directory = path if path.is_dir() else path.parent
    single_path = directory / "model.safetensors" if path.is_dir() else path
    index_path = directory / "model.safetensors.index.json" if path.is_dir() else path

    if single_path.exists() and single_path.name != "model.safetensors.index.json":
        state_dict = load_file(single_path, device="cpu")
    else:
        weight_map = _weight_map(index_path)
        state_dict = {}
        for filename in sorted(set(weight_map.values())):
            for key, value in load_file(directory / filename, device="cpu").items():
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

    return state_dict


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
