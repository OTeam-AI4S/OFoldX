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

r"""Download and explicitly publish versioned CCD datasets on the Hugging Face Hub."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from ...chemistry.definitions import ARTIFACT_FILENAME, PROVENANCE_FILENAME
from ...chemistry.definitions import download_library as download
from ...chemistry.definitions import validate_library as validate
from ...hub import upload_folder

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

__all__ = ["download", "upload", "validate"]


def upload(
    directory: str | PathLike[str],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
) -> CommitInfo:
    r"""Validate and publish a built CCD dataset with its card and provenance.

    Args:
        directory: Dataset directory produced by `build_artifact`.
        repo_id: Destination Hub dataset repository.
        token: Optional Hub authentication token.
        private: Whether to create the repository as private.

    Returns:
        Hub commit information for the completed upload.

    Raises:
        FileNotFoundError: The dataset card or a required artifact file is missing.
        UnsupportedError: The artifact schema is unsupported.
        ContradictionError: The artifact or sidecar fails validation.

    Notes:
        Publication includes the artifact, provenance sidecar, and README only.
        Other local files and unrelated remote files are retained.
    """
    directory = Path(directory).expanduser()
    if not (directory / "README.md").is_file():
        raise FileNotFoundError(f"CCD dataset README.md is required before upload: {directory / 'README.md'}")
    validate(directory)

    return upload_folder(
        directory,
        repo_id,
        repo_type="dataset",
        token=token,
        private=private,
        allow_patterns=[ARTIFACT_FILENAME, PROVENANCE_FILENAME, "README.md"],
    )
