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

r"""Resolve CCD artifacts from one Hugging Face dataset snapshot."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from ...errors import RequestRef, fail

ARTIFACT_FILENAME = "components.safetensors"
PROVENANCE_FILENAME = f"{ARTIFACT_FILENAME}.provenance.json"
DEFAULT_REPOSITORY = "oteam/ccd"
DEFAULT_REVISION = "main"


def resolve_snapshot(
    repo_id: str = DEFAULT_REPOSITORY,
    *,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
    local_files_only: bool = False,
) -> Path:
    r"""Resolve the artifact and provenance sidecar from the same dataset snapshot."""

    if not revision.strip():
        fail("ccd_revision_invalid", RequestRef("revision", revision))
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=[ARTIFACT_FILENAME, PROVENANCE_FILENAME],
        )
    )
