r"""
Lazy structure datasets backed by a processor.
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
    r"""
    Process structure files lazily into typed model inputs.

    Args:
        source: Directory, glob, manifest, structure file, or ``None`` for an empty
            dataset.
        processor: Callable that converts one structure reference into a model input.
        lower: How to lower each structure -- ``"process_training"`` to produce the separate
            ``(input, target)`` the diffusion families train on, or ``"call"`` for the plain Processor
            call (the sequence families, and every family's inference splits). The Runner supplies this
            explicitly from the architecture's training spec rather than sniffing the Processor's methods.
        processor_kwargs: Arguments forwarded to every processor call.
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        lower: str = "call",
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if lower not in ("call", "process_training"):
            raise ValueError(f"lower must be 'call' or 'process_training'; got {lower!r}")
        self.processor = processor
        self.lower = lower
        self.processor_kwargs = dict(processor_kwargs or {})
        # Records stay as raw paths (lazy): the Processor owns reading, one file per access.
        self.records: list[str] = _resolve_records(source)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        from ..io import read_structure

        structure = read_structure(self.records[index])
        if self.lower == "process_training":
            item = self.processor.process_training(structure, **self.processor_kwargs)
        else:
            item = self.processor(structure, **self.processor_kwargs)
        # Pair each item with its global dataset index (the sampler-assigned ``index``, independent of batch
        # size and DDP sharding) so inference can seed a per-sample generator reproducibly.
        return index, item


def _resolve_records(source: Any) -> list[str]:
    r"""
    Resolve a source into a concrete, ordered list of structure-file paths.

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
    r"""
    Read a manifest of structure paths, one per line, resolved against its dir.

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
