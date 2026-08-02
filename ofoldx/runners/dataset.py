r"""
Lazy structure datasets backed by a processor.
"""

from __future__ import annotations

import glob
import json
import operator
import os
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ..processor import UnresolvedTrainingCropError

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
        seq_to_pdb_index: Optional JSON sequence-to-directory mapping for indexed
            per-structure MSA loading.
        msa_path: Optional root directory for indexed MSAs.
        processor_kwargs: Arguments forwarded to every processor call.
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        seq_to_pdb_index: str | os.PathLike[str] | None = None,
        msa_path: str | os.PathLike[str] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        ccd_revision: str | None = None,
        ccd_snapshot_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if (seq_to_pdb_index is None) != (msa_path is None):
            raise ValueError("seq_to_pdb_index and msa_path must be provided together")
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        if ccd_revision is not None and ccd_snapshot_path is not None:
            raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")
        self.ccd_revision = ccd_revision
        self.ccd_snapshot_path = None if ccd_snapshot_path is None else os.fspath(ccd_snapshot_path)
        self.seq_to_pdb_index: Mapping[str, object] | None = None
        self.msa_path: Path | None = None
        if seq_to_pdb_index is not None and msa_path is not None:
            index_path = Path(seq_to_pdb_index).expanduser()
            if not index_path.is_file():
                raise FileNotFoundError(f"sequence-to-PDB index is not a file: {index_path}")
            resolved_msa_path = Path(msa_path).expanduser()
            if not resolved_msa_path.is_dir():
                raise FileNotFoundError(f"MSA path is not a directory: {resolved_msa_path}")
            self.seq_to_pdb_index = _load_seq_to_pdb_index(index_path.resolve())
            self.msa_path = resolved_msa_path.resolve()
        # Records stay as paths and ordinary mappings; generic I/O reads their
        # canonical structure/evidence values per access.
        self.records: list[str | dict[str, Any]] = _resolve_records(source)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        length = len(self)
        logical_index = operator.index(index)
        if logical_index < 0:
            logical_index += length
        if logical_index < 0 or logical_index >= length:
            raise IndexError("dataset index out of range")

        last_error: UnresolvedTrainingCropError | None = None
        for attempt in range(length):
            candidate = (logical_index + attempt) % length
            try:
                return self._process_logical_index(candidate)
            except UnresolvedTrainingCropError as error:
                last_error = error
        raise UnresolvedTrainingCropError(
            f"no resolved training crop found after {length} attempts starting at logical index {logical_index}"
        ) from last_error

    def _process_logical_index(self, index: int) -> Any:
        r"""Process the record represented by one normalized logical index."""

        return self._process_record(index)

    def _process_record(self, index: int, **extra_processor_kwargs: Any) -> Any:
        from ..data import ENTITY_PROTEIN, MSA, MSACollection, decode_msa_collection
        from ..io import read_msa, read_structure

        record = self.records[index]
        msa: MSA | MSACollection | None
        if isinstance(record, str):
            structure_path = record
        else:
            structure_path = record["structure"]
        structure = read_structure(
            structure_path,
            ccd_revision=self.ccd_revision,
            ccd_snapshot_path=self.ccd_snapshot_path,
        )
        if isinstance(record, str):
            msa = None
        else:
            if "msa" in record:
                msa_spec = record["msa"]
                msa = read_msa(msa_spec["path"], entity_id=msa_spec["entity_id"])
            elif "msa_collection" in record:
                payload = json.loads(Path(record["msa_collection"]).read_text(encoding="utf-8"))
                msa = decode_msa_collection(payload)
            else:
                msa = None
        if msa is None and self.seq_to_pdb_index is not None and self.msa_path is not None:
            has_protein = any(int(entity_type) == ENTITY_PROTEIN for entity_type in structure.system.entity_type)
            if has_protein:
                msa = read_msa(structure, self.seq_to_pdb_index, self.msa_path)
        processor_kwargs = {**self.processor_kwargs, **extra_processor_kwargs}
        if msa is not None:
            return self.processor(structure, msa=msa, **processor_kwargs)
        return self.processor(structure, **processor_kwargs)


class ClusteredStructureDataset(StructureDataset):
    r"""An eagerly indexed, asymmetric-unit CIF dataset for weighted sampling.

    A row ordinal is mapped to its complete backing CIF record. When
    ``forward_indices`` is true, the selected row is also passed to the
    processor so chain and interface rows can choose distinct spatial crop
    centres while retaining full-structure coordinate truth. Default structure
    ingestion removes hydrogens and water; configured ion handling and
    ``exclude_mol_types`` affect sampling rows only.
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        cluster_file: str,
        cluster_indices_file: str | os.PathLike[str] | None = None,
        contact_cutoff: float = 5.0,
        include_all_chain_pairs: bool = False,
        exclude_mol_types: tuple[str, ...] = ("ions",),
        beta: Mapping[str, float] | None = None,
        alpha: Mapping[str, float] | None = None,
        seq_to_pdb_index: str | os.PathLike[str] | None = None,
        msa_path: str | os.PathLike[str] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        forward_indices: bool = False,
        ccd_revision: str | None = None,
        ccd_snapshot_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(
            source,
            processor,
            seq_to_pdb_index=seq_to_pdb_index,
            msa_path=msa_path,
            processor_kwargs=processor_kwargs,
            ccd_revision=ccd_revision,
            ccd_snapshot_path=ccd_snapshot_path,
        )
        self.forward_indices = bool(forward_indices)
        invalid = [
            Path(record if isinstance(record, str) else record["structure"])
            for record in self.records
            if Path(record if isinstance(record, str) else record["structure"]).suffix.lower() not in {".cif", ".mmcif"}
        ]
        if invalid:
            raise ValueError(f"cluster sampling only supports CIF/mmCIF records: {invalid[0]}")
        from .cluster import calculate_cluster_weights, load_or_build_cluster_indices, make_cluster_indices

        self.cluster_indices_file = (
            Path(cluster_indices_file).expanduser().resolve() if cluster_indices_file is not None else None
        )
        if self.cluster_indices_file is None:
            raw_entries = make_cluster_indices(
                self.records,
                cluster_file,
                contact_cutoff=contact_cutoff,
                include_all_chain_pairs=include_all_chain_pairs,
                exclude_mol_types=exclude_mol_types,
                ccd_revision=self.ccd_revision,
                ccd_snapshot_path=self.ccd_snapshot_path,
            )
            self.entries, self.weights = calculate_cluster_weights(raw_entries, beta=beta, alpha=alpha)
            self.cluster_indices_status = None
        else:
            self.entries, self.weights, self.cluster_indices_status = load_or_build_cluster_indices(
                self.records,
                cluster_file,
                self.cluster_indices_file,
                contact_cutoff=contact_cutoff,
                include_all_chain_pairs=include_all_chain_pairs,
                exclude_mol_types=exclude_mol_types,
                beta=beta,
                alpha=alpha,
                ccd_revision=self.ccd_revision,
                ccd_snapshot_path=self.ccd_snapshot_path,
            )
        if not self.entries:
            raise ValueError("cluster sampling produced no valid chain or interface index rows")

    def __len__(self) -> int:
        return len(self.entries)

    def _process_logical_index(self, index: int) -> Any:
        r"""Process the backing record and metadata for one cluster row."""

        entry = self.entries[index]
        if self.forward_indices:
            return self._process_record(entry.record_index, indices=entry)
        return self._process_record(entry.record_index)


@cache
def _load_seq_to_pdb_index(path: Path) -> Mapping[str, object]:
    r"""Load and cache one canonical sequence-to-MSA-directory JSON mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"sequence-to-PDB index JSON must contain an object at the top level: {path}")
    return payload


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
