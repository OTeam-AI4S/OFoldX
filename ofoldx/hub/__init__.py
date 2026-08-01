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

r"""Resolve local and Hugging Face Hub repositories and transfer their files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import import_module
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple

import huggingface_hub

from ..references import classify_reference

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

    from .model import (
        MIRROR_REPOSITORY,
        MODEL_METADATA,
        MODEL_WEIGHT_METADATA,
        PROCESSOR_METADATA,
        hydrate_weights,
        load_model,
        load_model_state_dict,
        load_processor,
        load_registered_model,
        load_registered_processor,
        model_values,
        upload_repository,
    )


class ResolvedRepository(NamedTuple):
    r"""
    A local repository with optional Hub provenance.

    Attributes:
        path: Local repository or downloaded snapshot path.
        repository: Hugging Face Hub repository ID, when downloaded from the Hub.
        commit: Immutable Hub commit that produced ``path``.
        subdirectory: Repository-relative component directory represented by
            ``path``; the empty string selects the repository root.
        repo_type: Hub repository type retained when resolving this reference again.
    """

    path: Path
    repository: str | None = None
    commit: str | None = None
    subdirectory: str = ""
    repo_type: str = "model"


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
    reference: str | PathLike[str] | ResolvedRepository,
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
    repo_type: str = "model",
    local_files_only: bool = False,
) -> ResolvedRepository:
    r"""
    Resolve a local repository or Hugging Face Hub repository ID.

    Args:
        reference: Existing or explicitly written local path, or a Hub repository ID.
        revision: Hub revision to download.
        allow_patterns: Optional repository file patterns to download.
        token: Hugging Face access token used for private repositories.
        repo_type: Hub repository type, for example ``"model"`` or ``"dataset"``.
        local_files_only: Resolve only files already present in the Hub cache.

    Returns:
        The local repository path and, for Hub snapshots, its repository ID and commit.
    """

    if isinstance(reference, ResolvedRepository):
        if revision is not None and revision != reference.commit:
            raise ValueError("revision cannot change an already resolved repository snapshot")
        if reference.repository is None:
            return reference
        if reference.commit is None:
            raise ValueError("a resolved Hub repository requires its immutable commit")
        prefix = PurePosixPath(reference.subdirectory)
        patterns = None if allow_patterns is None else tuple(str(prefix / pattern) for pattern in allow_patterns)
        resolved = resolve_hub_repository(
            reference.repository,
            revision=reference.commit,
            allow_patterns=patterns,
            token=token,
            repo_type=reference.repo_type,
            local_files_only=local_files_only,
        )
        return ResolvedRepository(
            resolved.path / reference.subdirectory,
            resolved.repository,
            resolved.commit,
            reference.subdirectory,
            reference.repo_type,
        )
    classified = classify_reference(reference)
    if isinstance(classified, Path):
        if revision is not None:
            raise ValueError("revision is valid only for a Hub repository")
        return ResolvedRepository(classified, repo_type=repo_type)
    return resolve_hub_repository(
        classified,
        revision=revision,
        allow_patterns=allow_patterns,
        token=token,
        repo_type=repo_type,
        local_files_only=local_files_only,
    )


def resolve_hub_repository(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
    repo_type: str = "model",
    local_files_only: bool = False,
) -> ResolvedRepository:
    r"""
    Download files from one resolved Hugging Face Hub snapshot.

    Args:
        repo_id: Hugging Face Hub repository ID.
        revision: Revision to download.
        cache_dir: Optional Hugging Face cache directory.
        allow_patterns: Optional repository file patterns to download.
        token: Hugging Face access token used for private repositories.
        repo_type: Hub repository type, for example ``"model"`` or ``"dataset"``.
        local_files_only: Resolve only files already present in the Hub cache.

    Returns:
        The downloaded snapshot path, repository ID, and resolved commit.
    """

    snapshot = Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
            token=token,
            repo_type=repo_type,
            local_files_only=local_files_only,
        )
    )
    return ResolvedRepository(snapshot, repo_id, snapshot.name, repo_type=repo_type)


def upload_folder(
    directory: str | PathLike[str],
    repo_id: str,
    *,
    repo_type: str = "model",
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_message: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    delete_patterns: Sequence[str] | None = None,
) -> CommitInfo:
    r"""
    Explicitly upload files from a local directory to a Hub repository.

    ``allow_patterns`` restricts which local files are included. Remote files are
    removed only when ``delete_patterns`` explicitly names them.

    Args:
        directory: Local directory containing the files to upload.
        repo_id: Destination Hub repository ID.
        repo_type: Hub repository type, such as ``model`` or ``dataset``.
        token: Optional Hub authentication token.
        private: Whether to create a new repository as private.
        revision: Destination branch or revision.
        commit_message: Optional message for the uploaded commit.
        allow_patterns: Optional filters selecting local repository-relative file paths.
        delete_patterns: Optional patterns selecting remote files to remove.

    Returns:
        The Hub commit information for the completed upload.
    """
    api = huggingface_hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(Path(directory).expanduser()),
        revision=revision,
        commit_message=commit_message,
        allow_patterns=None if allow_patterns is None else list(allow_patterns),
        delete_patterns=None if delete_patterns is None else list(delete_patterns),
    )


# Model-specific names remain available at their original import path.
_MODEL_EXPORTS = frozenset(
    (
        "MODEL_METADATA",
        "PROCESSOR_METADATA",
        "MODEL_WEIGHT_METADATA",
        "MIRROR_REPOSITORY",
        "upload_repository",
        "hydrate_weights",
        "model_values",
        "load_model",
        "load_registered_model",
        "load_model_state_dict",
        "load_processor",
        "load_registered_processor",
    )
)


def __getattr__(name):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".model", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _MODEL_EXPORTS)
