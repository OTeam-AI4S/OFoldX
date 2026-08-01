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
