r"""Structure dataset: raw structures lowered on demand into family Model inputs.

The training-side dataset holds a family Processor (the featurizer) and applies
it lazily per item, so the exact lowering inference uses drives training too. No
model-specific logic lives here (DESIGN: no training/model provenance in
``data/``); the Processor owns featurization and this class owns iteration.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

# The frozen :mod:`ofoldx.io` readers select their parser from the path suffix
# (``.pdb`` -> PDB, ``.cif`` / ``.mmcif`` -> mmCIF), so a ``format`` hint is honored
# by admitting only that reader's suffixes rather than by a separate reader kwarg.
_FORMAT_SUFFIXES: dict[str, frozenset[str]] = {
    "pdb": frozenset({".pdb"}),
    "mmcif": frozenset({".cif", ".mmcif"}),
}
_FORMAT_ALIASES: dict[str, str] = {"pdb": "pdb", "cif": "mmcif", "mmcif": "mmcif"}
_STRUCTURE_SUFFIXES: frozenset[str] = frozenset().union(*_FORMAT_SUFFIXES.values())
_GLOB_MAGIC: tuple[str, ...] = ("*", "?", "[")


class StructureDataset(Dataset):
    r"""Iterate raw structures, lowering each to a family Input via a Processor.

    Args:
        source: The raw structures. A directory, glob, or manifest file of
            mmCIF / PDB paths (read through :mod:`ofoldx.io`). ``None`` is an
            empty dataset (used for splits that are configured but unused).
        processor: A constructed family Processor. ``processor(item, **kwargs)``
            returns the typed family Input for one structure (already batched to
            a leading singleton dimension, as the Processors emit).
        format: Optional explicit reader hint forwarded to :mod:`ofoldx.io`.
        processor_kwargs: Extra keyword arguments forwarded to every
            ``processor(...)`` call (e.g. ``msa=``, ``templates=``).

    ``__getitem__`` returns exactly what ``processor(...)`` returns — one family
    Input dataclass carrying a leading singleton batch axis. The Runner's collate
    concatenates and pads these along that axis; a batch of size ``B`` is a single
    Input whose tensor fields have leading dim ``B``.
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        format: str | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.processor = processor
        self.format = format
        self.processor_kwargs = dict(processor_kwargs or {})
        # Records stay as raw-structure paths (lazy): the Processor owns reading,
        # so ``__getitem__`` hands it the very same reference inference passes,
        # keeping the training-time and inference-time lowering identical.
        self.records: list[str] = _resolve_records(source, format)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        return self.processor(self.records[index], **self.processor_kwargs)


def _format_suffixes(format: str | None) -> frozenset[str]:
    r"""Map a reader hint to the path suffixes :mod:`ofoldx.io` reads for it."""

    if format is None:
        return _STRUCTURE_SUFFIXES
    key = _FORMAT_ALIASES.get(format.lower())
    if key is None:
        raise ValueError(f"unsupported structure format {format!r}; expected one of {sorted(_FORMAT_ALIASES)!r}")
    return _FORMAT_SUFFIXES[key]


def _suffix(path: str | os.PathLike[str]) -> str:
    return Path(path).suffix.lower()


def _resolve_records(source: Any, format: str | None) -> list[str]:
    r"""Resolve a source into a concrete, ordered list of structure-file paths.

    ``None`` is an empty dataset. A glob pattern expands (sorted) to its matches; a
    directory yields the structure files within its tree (sorted); a manifest file
    yields its listed paths in order; a lone structure file is a single record.
    ``format`` restricts a directory or glob to one reader's suffixes and validates
    an explicitly named structure file.
    """

    if source is None:
        return []
    text = os.fspath(source)
    suffixes = _format_suffixes(format)
    if any(character in text for character in _GLOB_MAGIC):
        matches = sorted(glob.glob(os.path.expanduser(text), recursive=True))
        return [match for match in matches if os.path.isfile(match) and _suffix(match) in suffixes]
    path = Path(text).expanduser()
    if path.is_dir():
        return sorted(str(entry) for entry in path.rglob("*") if entry.is_file() and _suffix(entry) in suffixes)
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if _suffix(path) in _STRUCTURE_SUFFIXES:
        if _suffix(path) not in suffixes:
            raise ValueError(f"structure file {path} does not match requested format {format!r}")
        return [str(path)]
    return _read_manifest(path)


def _read_manifest(path: Path) -> list[str]:
    r"""Read a manifest of structure paths, one per line, resolved against its dir.

    Blank lines and ``#`` comments are skipped; relative entries resolve against the
    manifest's own directory so a manifest is portable alongside its structures.
    """

    records: list[str] = []
    base = path.parent
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
            entry = base / entry
        if not entry.exists():
            raise FileNotFoundError(f"manifest {path} lists a missing structure: {entry}")
        records.append(str(entry))
    return records
