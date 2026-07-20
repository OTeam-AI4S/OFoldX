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

_STRUCTURE_SUFFIXES: frozenset[str] = frozenset({".pdb", ".cif", ".mmcif"})
_GLOB_MAGIC: tuple[str, ...] = ("*", "?", "[")


class StructureDataset(Dataset):
    r"""Iterate raw structures, lowering each to a family Input via a Processor.

    Args:
        source: The raw structures — a directory, glob, or manifest file of
            mmCIF / PDB paths. ``None`` is an empty dataset (a configured but
            unused split).
        processor: A constructed family Processor. ``processor(item, **kwargs)``
            returns the typed family Input for one structure (already batched to a
            leading singleton dimension, as the Processors emit).
        processor_kwargs: Extra keyword arguments forwarded to every
            ``processor(...)`` call (e.g. ``msa=``, ``templates=``).
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        # Records stay as raw paths (lazy): the Processor owns reading, so the same
        # reference inference passes drives training-time lowering identically.
        self.records: list[str] = _resolve_records(source)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        return self.processor(self.records[index], **self.processor_kwargs)


def _resolve_records(source: Any) -> list[str]:
    r"""Resolve a source into a concrete, ordered list of structure-file paths.

    ``None`` is an empty dataset. A glob pattern expands (sorted) to its structure
    matches; a directory yields the structure files within its tree (sorted); a
    manifest file yields its listed paths in order; a lone structure file is a
    single record.
    """

    if source is None:
        return []
    text = os.fspath(source)
    if any(character in text for character in _GLOB_MAGIC):
        matches = sorted(glob.glob(os.path.expanduser(text), recursive=True))
        return [m for m in matches if os.path.isfile(m) and Path(m).suffix.lower() in _STRUCTURE_SUFFIXES]
    path = Path(text).expanduser()
    if path.is_dir():
        return sorted(str(entry) for entry in path.rglob("*") if entry.suffix.lower() in _STRUCTURE_SUFFIXES)
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if path.suffix.lower() in _STRUCTURE_SUFFIXES:
        return [str(path)]
    return _read_manifest(path)


def _read_manifest(path: Path) -> list[str]:
    r"""Read a manifest of structure paths, one per line, resolved against its dir.

    Blank lines and ``#`` comments are skipped; relative entries resolve against the
    manifest's own directory so a manifest is portable alongside its structures.
    """

    records: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        if not entry.exists():
            raise FileNotFoundError(f"manifest {path} lists a missing structure: {entry}")
        records.append(str(entry))
    return records
