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

r"""
Classify local paths and remote repository identifiers.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path, PureWindowsPath


def classify_reference(reference: str | PathLike[str]) -> Path | str:
    r"""
    Classify a reference as a local path or repository ID.

    Args:
        reference: Path-like value or string reference. Existing paths, explicit
            relative or absolute paths, home-relative paths, and Windows drive paths
            are local; other strings are repository IDs.

    Returns:
        An expanded local path or the unchanged repository ID.
    """

    if isinstance(reference, str):
        path = Path(reference)
        if reference and (
            path.exists() or reference.startswith(("/", "./", "../", "~")) or bool(PureWindowsPath(reference).anchor)
        ):
            return path.expanduser()
        return reference

    return Path(reference).expanduser()
