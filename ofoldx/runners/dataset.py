r"""
Lazy structure datasets backed by a processor.
"""

from __future__ import annotations

import glob
import json
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
        processor: Callable that converts one structure into the value consumed by
            the model strategy.
        processor_kwargs: Arguments forwarded to every processor call.
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
        # Records stay as paths and ordinary mappings; generic I/O reads their
        # canonical structure/evidence values per access.
        self.records: list[str | dict[str, Any]] = _resolve_records(source)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        from ..data import MSA, MSACollection, decode_msa_collection
        from ..io import read_msa, read_structure

        record = self.records[index]
        msa: MSA | MSACollection | None
        if isinstance(record, str):
            structure_path = record
            msa = None
        else:
            structure_path = record["structure"]
            if "msa" in record:
                msa_spec = record["msa"]
                msa = read_msa(msa_spec["path"], entity_id=msa_spec["entity_id"])
            elif "msa_collection" in record:
                payload = json.loads(Path(record["msa_collection"]).read_text(encoding="utf-8"))
                msa = decode_msa_collection(payload)
            else:
                msa = None
        structure = read_structure(structure_path)
        if msa is not None:
            return self.processor(structure, msa=msa, **self.processor_kwargs)
        return self.processor(structure, **self.processor_kwargs)


def _resolve_records(source: Any) -> list[str | dict[str, Any]]:
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
        records: list[str | dict[str, Any]] = []
        records.extend(sorted(str(entry) for entry in path.rglob("*") if entry.suffix.lower() in _STRUCTURE_SUFFIXES))
        return records
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if path.suffix.lower() in _STRUCTURE_SUFFIXES:
        return [str(path)]
    return _read_manifest(path)


def _read_manifest(path: Path) -> list[str | dict[str, Any]]:
    r"""
    Read structure paths or JSON records, one per line, resolved against its dir.

    Blank lines and ``#`` comments are skipped; relative entries resolve against the
    manifest's own directory so a manifest and its explicit MSA evidence are portable
    together.
    """

    records: list[str | dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            record = json.loads(line)
            if "msa" in record and "msa_collection" in record:
                raise ValueError("a dataset record cannot contain both `msa` and `msa_collection`")
            structure_path = Path(os.fspath(record["structure"])).expanduser()
            if not structure_path.is_absolute():
                structure_path = path.parent / structure_path
            record["structure"] = str(structure_path)
            if "msa" in record:
                msa_path = Path(os.fspath(record["msa"]["path"])).expanduser()
                if not msa_path.is_absolute():
                    msa_path = path.parent / msa_path
                record["msa"]["path"] = str(msa_path)
            if "msa_collection" in record:
                collection_path = Path(os.fspath(record["msa_collection"])).expanduser()
                if not collection_path.is_absolute():
                    collection_path = path.parent / collection_path
                record["msa_collection"] = str(collection_path)
            records.append(record)
            continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        records.append(str(entry))
    return records
