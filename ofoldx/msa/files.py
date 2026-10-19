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
MSA files: A3M and aligned-FASTA reading, lazy access to indexed alignment pools, and the native ``MsaCollection``
artifact.
"""

from __future__ import annotations

import gzip
import json
import lzma
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, NamedTuple, NoReturn, TextIO, cast

import numpy as np
import zstandard
from numpy.typing import NDArray
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from ..data.molecular import MonomerType
from ..data.structure import Structure
from ..data.tokenization import class_sequence, token_classes
from ..errors import RequestRef, SourceRef, StructureRef, fail
from .alignment import MSA, MsaCollection, instance_monomer, unique_rows

#: The native artifact schema; another is ``InvalidRequestError(native_schema_version)``.
_SCHEMA = "ofoldx.msa/2"
_FASTA_SUFFIXES = ("fa", "fas", "fasta", "fna", "ffn", "frn", "faa")
_ANNOTATION_PREFIXES = ("ss_", "sa_", "aa_")
_A3M_INVALID = re.compile(r"[^A-Za-z.\-]")
_A3M_QUERY_INVALID = re.compile(r"[^A-Z.]")
_A3M_INSERTION = re.compile(r"[a-z]")
_A3M_SUFFIXES = (".a3m", ".a3m.gz", ".a3m.xz", ".a3m.zst", ".a3m.zstd")


def read_msa(path: str | os.PathLike[str], *, entity_id: str, max_rows: int | None = None) -> MSA:
    r"""
    An A3M (plain, gzip, xz or zstd) or aligned-FASTA file as the pools of one entity.

    The first biological record is the ungapped query. Rows are returned as written (no deduplication); insertion
    columns are removed and counted as deletions on the query axis. The paired pool is the query row. ``max_rows``
    caps the rows read, the query included.

    Args:
        path: Existing A3M or aligned-FASTA file.
        entity_id: Target entity whose query sequence the alignment describes.
        max_rows: Optional limit on rows read, including the query.

    Returns:
        One MSA with the file's rows in its unpaired pool and the query as its paired pool. Read rows and insertion
            counts are retained in source order.

    Raises:
        ContradictionError: ``msa_syntax`` for the grammar (symbols, a gapped or lowercase query, no record),
            ``msa_length_mismatch`` for a row whose width is not the query's.
    """

    sequences, counts, headers = _read(Path(path), max_rows)
    return MSA(entity_id, sequences, counts, sequences[:1], counts[:1].copy(), headers[:1])


def _read(
    path: Path, max_rows: int | None, query: str | None = None
) -> tuple[tuple[str, ...], NDArray[np.uint32], tuple[str, ...]]:
    r"""The projected rows, their deletion counts and their headers; ``query``, when given, must be the first row.

    Every error reference is ``SourceRef(path, line)`` with the zero-based line of the offending record's defline.
    """

    name = path.name.lower()
    if name.endswith(_A3M_SUFFIXES):
        a3m = True
    elif path.suffix.lower().removeprefix(".") in _FASTA_SUFFIXES:
        a3m = False
    else:
        fail("format_unknown", RequestRef("path", str(path)))
    biological = (
        (line, record)
        for line, record in _records(path)
        if not (
            record.id is not None and (record.id.startswith(_ANNOTATION_PREFIXES) or record.id.endswith("_consensus"))
        )
    )
    chosen = list(biological if max_rows is None else islice(biological, max_rows))
    if not chosen:
        fail("msa_syntax", SourceRef(str(path), 0), reason="no biological record")
    refs = [SourceRef(str(path), line) for line, _ in chosen]
    mask = None if a3m else _normalise(chosen[0][1].sequence, refs[0])
    projected = [
        _project_a3m(record.sequence, ref, row == 0) if mask is None else _project_fasta(record.sequence, mask, ref)
        for row, ((_, record), ref) in enumerate(zip(chosen, refs))
    ]
    width = len(projected[0][0])
    for (sequence, _), ref in zip(projected, refs):
        if len(sequence) != width:
            fail("msa_length_mismatch", ref, width=len(sequence), query=width)
    if query is not None and projected[0][0] != query:
        fail("msa_query_mismatch", refs[0], query=query)
    counts = np.array([deletions for _, deletions in projected], dtype=np.uint32).reshape(len(projected), width)
    headers = tuple(" ".join(part for part in (record.id, record.comment) if part is not None) for _, record in chosen)
    return tuple(sequence for sequence, _ in projected), counts, headers


class _Record(NamedTuple):
    sequence: str
    id: str | None
    comment: str | None


def _records(path: Path) -> Iterator[tuple[int, _Record]]:
    r"""The FASTA records of a plain or compressed file with the zero-based line of their defline; a defline without
    sequence is ``msa_syntax``."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        handle = cast(TextIO, gzip.open(path, "rt"))
    elif suffix == ".xz":
        handle = cast(TextIO, lzma.open(path, "rt"))
    elif suffix in (".zst", ".zstd"):
        handle = cast(TextIO, zstandard.open(path, "rt"))
    else:
        handle = path.open()
    header: str | None = None
    start = 0
    sequence: list[str] = []
    with handle as file:
        for number, line in enumerate(file):
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _record(path, start, header, sequence)
                header, start, sequence = line[1:].strip(), number, []
            elif header is None:
                fail("msa_syntax", SourceRef(str(path), number), reason="sequence before a defline")
            else:
                sequence.append(line)
    if header is not None:
        yield _record(path, start, header, sequence)


def _record(path: Path, line: int, header: str, sequence: list[str]) -> tuple[int, _Record]:
    if not sequence:
        fail("msa_syntax", SourceRef(str(path), line), reason="defline without sequence")
    parts = header.split(None, 1)
    return line, _Record("".join(sequence), parts[0] if parts else None, parts[1] if len(parts) == 2 else None)


def _project_a3m(sequence: str, ref: SourceRef, is_query: bool) -> tuple[str, list[int]]:
    r"""One A3M row: uppercase and ``-`` are columns, lowercase letters count as deletions, ``.`` is dropped."""

    invalid = (_A3M_QUERY_INVALID if is_query else _A3M_INVALID).search(sequence)
    if invalid is not None:
        fail("msa_syntax", ref, symbol=invalid[0])
    sequence = sequence.replace(".", "")
    if _A3M_INSERTION.search(sequence) is None:
        return sequence, [0] * len(sequence)
    codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    emitted = np.flatnonzero(codes <= ord("Z"))  # uppercase or gap
    return codes[emitted].tobytes().decode("ascii"), (np.diff(emitted, prepend=-1) - 1).tolist()


def _project_fasta(sequence: str, mask: str, ref: SourceRef) -> tuple[str, list[int]]:
    r"""One aligned-FASTA row against the normalised query ``mask``; a residue under a query gap is a deletion."""

    aligned = _normalise(sequence, ref)
    if len(aligned) != len(mask):
        fail("msa_length_mismatch", ref, width=len(aligned), query=len(mask))
    letters: list[str] = []
    deletions: list[int] = []
    pending = 0
    for column, symbol in zip(mask, aligned):
        if column == "-":
            pending += symbol != "-"
        else:
            letters.append(symbol)
            deletions.append(pending)
            pending = 0
    return "".join(letters), deletions


def _normalise(sequence: str, ref: SourceRef) -> str:
    r"""Letters uppercased, ``.`` read as ``-``; any other symbol is ``msa_syntax``."""
    if invalid := _A3M_INVALID.search(sequence):
        fail("msa_syntax", ref, symbol=invalid[0])
    return sequence.upper().replace(".", "-")


class IndexedMsaLoader:
    r"""
    The one indexed MSA layout (Protenix's directory form), read lazily.

    ``index`` (a mapping or a JSON file) maps a query sequence to its directory under ``root``; ``rna_index`` and
    ``rna_root`` do so for RNA. A value is a directory name or a one-element list. Per protein directory,
    ``pairing.a3m`` is the paired pool (with its headers, not deduplicated) and the unpaired pool is
    ``unique_rows(pairing ++ non_pairing.a3m)``; per RNA directory ``d``, ``d/d_all.a3m`` is the unpaired pool and its
    query the paired pool. ``max_rows`` caps every file read.
    """

    def __init__(
        self,
        index: Mapping[str, str | Sequence[str]] | str | os.PathLike[str],
        root: str | os.PathLike[str],
        *,
        rna_index: Mapping[str, str | Sequence[str]] | str | os.PathLike[str] | None = None,
        rna_root: str | os.PathLike[str] | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.indices = {MonomerType.PROTEIN: (_load_index(index), Path(root))}
        if rna_index is not None:
            self.indices[MonomerType.RNA] = (_load_index(rna_index), Path(root if rna_root is None else rna_root))
        self.max_rows = max_rows

    def __call__(self, structure: Structure) -> Mapping[str, MSA]:
        r"""
        Instance id → pool for the protein and RNA instances whose ``class_sequence`` is indexed; an instance without
        an entry is absent. Files are read when a pool is first accessed.

        Raises:
            UnsupportedError: ``query_class_unrepresentable`` for a protein or RNA instance with a component of no
                sequence class.
            InsufficientInformationError: ``msa_index_ambiguous`` for an index value naming several directories.
            ContradictionError: ``msa_query_mismatch`` (on access) for a file whose query is not the instance's.
            OSError: (on access) for a file that cannot be opened, unchanged.
        """

        system = structure.system
        s, symbols = system.arrays(), system.symbols()
        classes = token_classes(system).token
        pointer = s["instance_component_ptr"]
        entries: dict[str, tuple[MonomerType, Path, str, str]] = {}
        for instance, monomer in enumerate(instance_monomer(system).tolist()):
            if monomer not in self.indices:
                continue
            index, root = self.indices[MonomerType(monomer)]
            query = class_sequence(classes[pointer[instance] : pointer[instance + 1]], StructureRef(instance=instance))
            value = index.get(query)
            if value is None:
                continue
            directories = [value] if isinstance(value, str) else list(value)
            if len(directories) != 1:
                fail(
                    "msa_index_ambiguous",
                    RequestRef("index", query),
                    candidates=tuple(SourceRef("msa_index", k, item=str(d)) for k, d in enumerate(directories)),
                )
            entity = symbols["entity"][int(s["entity_symbol"][s["instance_entity"][instance]])]
            name = symbols["instance"][int(s["instance_symbol"][instance])]
            entries[name] = (MonomerType(monomer), root / str(directories[0]), entity, query)
        return _Pools(entries, self.max_rows)


def _load_index(index: Mapping[str, str | Sequence[str]] | str | os.PathLike[str]) -> Mapping:
    return index if isinstance(index, Mapping) else json.loads(Path(index).read_text(encoding="utf-8"))


class _Pools(Mapping[str, MSA]):
    r"""The loader's lazy mapping: membership and iteration never read a file; a pool is read on first access and
    shared by the instances of one entity and directory."""

    def __init__(self, entries: dict[str, tuple[MonomerType, Path, str, str]], max_rows: int | None) -> None:
        self.entries, self.max_rows = entries, max_rows
        self.cache: dict[tuple[MonomerType, Path, str], MSA] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> MSA:
        monomer, directory, entity, query = self.entries[key]
        if (monomer, directory, entity) not in self.cache:
            self.cache[monomer, directory, entity] = self._read(monomer, directory, entity, query)
        return self.cache[monomer, directory, entity]

    def _read(self, monomer: MonomerType, directory: Path, entity: str, query: str) -> MSA:
        if monomer == MonomerType.RNA:
            rows, counts, headers = _read(directory / f"{directory.name}_all.a3m", self.max_rows, query)
            return MSA(entity, rows, counts, rows[:1], counts[:1].copy(), headers[:1])
        paired, paired_counts, headers = _read(directory / "pairing.a3m", self.max_rows, query)
        other, other_counts, _ = _read(directory / "non_pairing.a3m", self.max_rows, query)
        union = paired + other
        keep = unique_rows(np.array(union, dtype="S"))
        counts = np.concatenate((paired_counts, other_counts))[keep]
        return MSA(entity, tuple(union[k] for k in keep), counts, paired, paired_counts, headers)


def save_msa_collection(collection: MsaCollection, path: str | os.PathLike[str]) -> None:
    r"""
    Save a plan with its pools as one safetensors artifact: per pool its rows (ASCII bytes), deletion counts, entity id
    and paired headers (``None`` kept distinct from a tuple), then the plan and its target instances.
    """

    arrays = {"row_pairing": collection.row_pairing, "row_is_paired": collection.row_is_paired}
    for index, msa in enumerate(collection.msas):
        arrays[f"msa.{index}.unpaired_sequences"] = _bytes(msa.unpaired_sequences)
        arrays[f"msa.{index}.unpaired_deletion_counts"] = msa.unpaired_deletion_counts
        arrays[f"msa.{index}.paired_sequences"] = _bytes(msa.paired_sequences)
        arrays[f"msa.{index}.paired_deletion_counts"] = msa.paired_deletion_counts
    metadata = {
        "schema": _SCHEMA,
        "entity_ids": json.dumps([msa.entity_id for msa in collection.msas]),
        "paired_headers": json.dumps([msa.paired_headers for msa in collection.msas]),
        "target_instance_ids": json.dumps(collection.target_instance_ids),
    }
    save_file({key: np.ascontiguousarray(value) for key, value in arrays.items()}, os.fspath(path), metadata=metadata)


def load_msa_collection(path: str | os.PathLike[str]) -> MsaCollection:
    r"""
    The plan saved by ``save_msa_collection``.

    Raises:
        InvalidRequestError: ``native_schema_version`` for an artifact of another schema; ``value_invalid`` for a
            value that violates an invariant.
        ContradictionError: ``native_corrupt`` for recognised damage, the field named: a container that does not
            decode (its header, or a tensor's dtype, shape or offsets, with the decoder's reason), a header entry
            missing, no JSON or not of its type, a tensor missing, or rows that are not two-dimensional uint8 ASCII.
            An I/O error propagates as it is.
    """

    path = os.fspath(path)
    try:
        artifact = safe_open(path, framework="np")
    except SafetensorError as error:  # the container's decode; an I/O error is an OSError and propagates
        _corrupt(path, "header", reason=str(error))
    with artifact:
        metadata = artifact.metadata() or {}
        if metadata.get("schema") != _SCHEMA:
            fail("native_schema_version", SourceRef(path, 0, item="schema"), schema=metadata.get("schema"))
        arrays = {key: artifact.get_tensor(key) for key in artifact.keys()}
    entities = _header(path, metadata, "entity_ids", _strings)
    headers = _header(
        path,
        metadata,
        "paired_headers",
        lambda v: isinstance(v, list) and len(v) == len(entities) and all(h is None or _strings(h) for h in v),
    )
    targets = _header(path, metadata, "target_instance_ids", lambda v: v is None or _strings(v))
    msas = tuple(
        MSA(
            entity,
            _texts(path, arrays, f"msa.{index}.unpaired_sequences"),
            _tensor(path, arrays, f"msa.{index}.unpaired_deletion_counts"),
            _texts(path, arrays, f"msa.{index}.paired_sequences"),
            _tensor(path, arrays, f"msa.{index}.paired_deletion_counts"),
            None if header is None else tuple(header),
        )
        for index, (entity, header) in enumerate(zip(entities, headers))
    )
    pairing, paired = _tensor(path, arrays, "row_pairing"), _tensor(path, arrays, "row_is_paired")
    return MsaCollection(msas, pairing, paired, None if targets is None else tuple(targets))


def _bytes(rows: Sequence[str]) -> NDArray[np.uint8]:
    return np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8).reshape(len(rows), len(rows[0]))


def _corrupt(path: str, item: str, **detail: object) -> NoReturn:
    fail("native_corrupt", SourceRef(path, 0, item=item), **detail)


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _header(path: str, metadata: Mapping[str, str], key: str, valid: Callable[[object], bool]) -> Any:
    r"""The JSON value of header entry ``key``: ``native_corrupt`` (item ``key``) when it is missing, no JSON (the
    decoder's reason named) or not ``valid``."""
    if key not in metadata:
        _corrupt(path, key)
    try:
        value = json.loads(metadata[key])
    except json.JSONDecodeError as error:
        _corrupt(path, key, reason=str(error))
    if not valid(value):
        _corrupt(path, key)
    return value


def _tensor(path: str, arrays: Mapping[str, NDArray], name: str) -> NDArray:
    r"""The tensor ``name``: ``native_corrupt`` (item ``name``) when it is missing."""
    if name not in arrays:
        _corrupt(path, name)
    return arrays[name]


def _texts(path: str, arrays: Mapping[str, NDArray], name: str) -> tuple[str, ...]:
    r"""The rows of the tensor ``name``: ``native_corrupt`` (item ``name``) unless it is two-dimensional uint8 of
    ASCII bytes (the decoder's reason named), never repaired."""
    rows = _tensor(path, arrays, name)
    if rows.dtype != np.uint8 or rows.ndim != 2:
        _corrupt(path, name)
    try:
        return tuple(row.tobytes().decode("ascii") for row in rows)
    except UnicodeDecodeError as error:
        _corrupt(path, name, reason=str(error))
