r"""Classification at local-path-or-identifier boundaries."""

from __future__ import annotations

from os import PathLike
from pathlib import Path, PureWindowsPath


def classify_reference(reference: str | PathLike[str]) -> Path | str:
    r"""Return local intent as a Path and leave other strings unchanged."""

    if isinstance(reference, str):
        path = Path(reference)
        if path.exists() or reference.startswith(("/", "./", "../", "~")) or bool(PureWindowsPath(reference).anchor):
            return path.expanduser()
        return reference

    return Path(reference).expanduser()
