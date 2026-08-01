r"""MSA parsing, biological-source pairing, indexed acquisition, and JSON exchange."""

from __future__ import annotations

import gzip
import json
import lzma
import operator
import os
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import zstandard
from multimolecule.io import FASTA, SequenceRecord, read_fasta_records
from numpy.typing import NDArray

from ..data.evidence import (
    MSA,
    MsaCollection,
    merge_unpaired_protein_msas,
    msa_residue_index,
    pair_msa_rows_by_key,
    unpaired_msa_row_plan,
)
from ..data.molecular import ENTITY_PROTEIN
from ..data.structure import Structure
from ..data.tokenization import TOKEN_ALPHABET, component_token_index

_ANNOTATION_PREFIXES = ("ss_", "sa_", "aa_")
_UNIPROT_ENTRY_NAME = re.compile(
    r"^(?:tr|sp)\|(?:[A-Za-z0-9]{6}|[A-Za-z0-9]{10})(?:[-_]\d+)?\|" r"(?P<entry>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)$"
)
_UNIREF100_ENTRY_NAME = re.compile(r"^UniRef100_[^_]+_(?P<species>[^_]+)$")
_MAX_PAIRED_ROWS = 16384 // 2
_MAX_ROWS_PER_KEY = 600


def read_msa(
    path: str | os.PathLike[str],
    *,
    entity_id: str,
    max_rows: int | None = None,
) -> MSA:
    r"""
    Read an A3M or aligned FASTA file as MSA evidence.

    The first biological record is the ungapped query. Insertion columns are
    removed from aligned rows and recorded as deletion counts on the query axis.

    Args:
        path: MSA path. Plain or gzip/xz/zstd-compressed ``.a3m`` selects A3M parsing;
            supported FASTA suffixes select aligned FASTA parsing.
        entity_id: Target entity ID stored with the alignment.
        max_rows: Optional maximum number of biological rows to read, including
            the query row.

    Returns:
        An MSA whose unpaired pool contains the projected rows and whose paired
        pool contains the query row.

    Raises:
        ValueError: If the suffix, symbols, row widths, or query record are invalid.
    """

    return read_msa_with_headers(path, entity_id=entity_id, max_rows=max_rows)[0]


def read_msa_with_headers(
    path: str | os.PathLike[str],
    *,
    entity_id: str,
    max_rows: int | None = None,
) -> tuple[MSA, tuple[str, ...]]:
    r"""
    Read an MSA together with its source headers.

    Header strings remain aligned with the projected biological rows and are
    intended for acquisition operations that consume them before constructing
    final MSA evidence.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if path.name.lower().endswith((".a3m", ".a3m.gz", ".a3m.xz", ".a3m.zst", ".a3m.zstd")):
        a3m = True
    elif suffix.removeprefix(".") in FASTA:
        a3m = False
    else:
        raise ValueError(f"unsupported MSA filename suffix: {suffix or '<none>'}")

    biological_records = (
        record
        for record in (
            _read_compressed_fasta_records(path)
            if suffix in {".gz", ".xz", ".zst", ".zstd"}
            else read_fasta_records(path)
        )
        if not (
            record.id is not None and (record.id.startswith(_ANNOTATION_PREFIXES) or record.id.endswith("_consensus"))
        )
    )
    if max_rows is None:
        records = list(biological_records)
    else:
        max_rows = operator.index(max_rows)
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        records = list(islice(biological_records, max_rows))
    if not records:
        raise ValueError("MSA contains no biological sequence records")

    sequences = [record.sequence for record in records]
    if a3m:
        query_row, query_deletions = _project_a3m_row(sequences[0], is_query=True)
    else:
        query_row, query_deletions = _project_aligned_fasta_row(sequences[0], sequences[0])
    width = len(query_row)

    unpaired_sequences: list[str] = [query_row]
    unpaired_deletion_counts: NDArray[np.uint32] = np.zeros((len(sequences), width), dtype=np.uint32)
    unpaired_deletion_counts[0] = query_deletions
    for row_index, sequence in enumerate(sequences[1:], start=1):
        if a3m:
            projected, deletions = _project_a3m_row(sequence, is_query=False)
        else:
            projected, deletions = _project_aligned_fasta_row(sequence, sequences[0])
        if len(projected) != width:
            raise ValueError("every MSA row must project to the query width")
        unpaired_sequences.append(projected)
        unpaired_deletion_counts[row_index] = deletions

    return (
        MSA(
            entity_id=entity_id,
            unpaired_sequences=tuple(unpaired_sequences),
            unpaired_deletion_counts=unpaired_deletion_counts,
            paired_sequences=(unpaired_sequences[0],),
            paired_deletion_counts=unpaired_deletion_counts[0:1].copy(),
        ),
        tuple(" ".join(part for part in (record.id, record.comment) if part is not None) for record in records),
    )


def pair_msas(
    unpaired_msa_paths: Mapping[
        str,
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    ],
    paired_msa_paths: Mapping[str, str | os.PathLike[str]],
    *,
    remove_paired_from_unpaired: bool = True,
) -> MsaCollection:
    r"""
    Pair independent protein alignments by their biological-source header key.

    The first mapping supplies one or more files forming each complete unpaired
    pool and fixes column order. The second supplies header-bearing paired
    candidates for the same entities. UniProt entry names and UniRef100 entry
    names are accepted. The query row is always the first paired row.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     root = Path(directory)
        ...     a = root / "a.a3m"
        ...     b = root / "b.a3m"
        ...     _ = a.write_text(
        ...         ">tr|A0A000|FOO_HUMAN\nAGV\n>tr|A0A001|FOO_MOUSE\nCGV\n"
        ...         ">duplicate\nAGV\n>insertion\nAaaGV\n>unmatched_1\nTGV\n>unmatched_2\nVGV\n"
        ...     )
        ...     _ = b.write_text(
        ...         ">tr|B0B000|BAR_HUMAN\nGAK\n>UniRef100_B0B001_MOUSE\nWAK\n"
        ...         ">unmatched_1\nAAK\n>unmatched_2\nTAK\n"
        ...     )
        ...     collection = pair_msas({"A": a, "B": b}, {"A": a, "B": b})
        ...     paired = collection.row_pairing[collection.row_is_paired.any(axis=1)]
        ...     unpaired_a = collection.row_pairing[~collection.row_is_paired[:, 0], 0]
        ...     [tuple(int(v) for v in row) for row in paired], [int(row) for row in unpaired_a if row != -1]
        ([(0, 0), (1, 1)], [2, 3])
        >>> [tuple(int(v) for v in row) for row in collection.row_pairing[~collection.row_is_paired.any(axis=1)]]
        [(2, -1), (-1, 2), (3, -1), (-1, 3)]
    """

    entity_ids = tuple(unpaired_msa_paths)
    if paired_msa_paths.keys() != unpaired_msa_paths.keys():
        raise ValueError("unpaired and paired MSA mappings must have the same entities")

    msas: list[MSA] = []
    keys_by_msa: list[tuple[str, ...]] = []
    for entity_id in entity_ids:
        unpaired_value = unpaired_msa_paths[entity_id]
        unpaired_paths = (unpaired_value,) if isinstance(unpaired_value, (str, os.PathLike)) else tuple(unpaired_value)
        paired_path = paired_msa_paths[entity_id]
        paired, headers = read_msa_with_headers(paired_path, entity_id=entity_id)
        unpaired = merge_unpaired_protein_msas(
            tuple(
                paired if Path(path) == Path(paired_path) else read_msa(path, entity_id=entity_id)
                for path in unpaired_paths
            )
        )
        if unpaired.unpaired_sequences[0] != paired.unpaired_sequences[0]:
            raise ValueError(f"unpaired and paired MSA queries differ for entity {entity_id!r}")
        msas.append(
            MSA(
                entity_id=entity_id,
                unpaired_sequences=unpaired.unpaired_sequences,
                unpaired_deletion_counts=unpaired.unpaired_deletion_counts,
                paired_sequences=paired.unpaired_sequences,
                paired_deletion_counts=paired.unpaired_deletion_counts,
            )
        )
        keys_by_msa.append(("",) + tuple(_msa_pairing_key(header) for header in headers[1:]))

    paired_blocks: list[NDArray[np.int64]] = [np.zeros((1, len(msas)), dtype=np.int64)]
    paired_count = 1
    if len({msa.unpaired_sequences[0] for msa in msas}) > 1:
        for rows in pair_msa_rows_by_key(keys_by_msa, max_rows_per_key=_MAX_ROWS_PER_KEY):
            rank = np.log(np.maximum(np.abs(rows.astype(np.float64)), 1.0)).sum(axis=1)
            paired_blocks.append(rows[np.argsort(rank, kind="stable")])
            paired_count += len(rows)
            if paired_count >= _MAX_PAIRED_ROWS:
                break
    paired_plan = np.concatenate(paired_blocks, axis=0)[:_MAX_PAIRED_ROWS]

    unpaired_sources: list[list[int]] = []
    for column, msa in enumerate(msas):
        if remove_paired_from_unpaired:
            consumed = {
                tuple(msa_residue_index(symbol, "protein") for symbol in msa.paired_sequences[int(row)])
                for row in paired_plan[:, column]
                if row != -1
            }
            unpaired_sources.append(
                [
                    row
                    for row, sequence in enumerate(msa.unpaired_sequences)
                    if tuple(msa_residue_index(symbol, "protein") for symbol in sequence) not in consumed
                ]
            )
        else:
            unpaired_sources.append(list(range(len(msa.unpaired_sequences))))

    unpaired_plan = unpaired_msa_row_plan(unpaired_sources)
    row_pairing = np.concatenate((paired_plan, unpaired_plan), axis=0)
    row_is_paired = np.concatenate(
        (paired_plan != -1, np.zeros_like(unpaired_plan, dtype=np.bool_)),
        axis=0,
    )
    return MsaCollection(tuple(msas), row_pairing, row_is_paired)


def _msa_pairing_key(header: str) -> str:
    identifier = header.strip()
    identifier = identifier.split(maxsplit=1)[0].partition("/")[0] if identifier else ""
    if match := _UNIPROT_ENTRY_NAME.fullmatch(identifier):
        entry = match.group("entry").rsplit("_", 2)
        if len(entry) < 2:
            return ""
        if entry[-1].isdigit():
            if len(entry) != 3:
                return ""
            species = entry[-2]
        else:
            species = entry[-1]
        return species.upper() if any(character.isalpha() for character in species) else ""
    if match := _UNIREF100_ENTRY_NAME.fullmatch(identifier):
        return match.group("species").upper()
    return ""


class IndexedMsaLoader:
    r"""Acquire per-entity protein MSAs from a sequence-indexed directory.

    Missing sequence-index entries use query-only evidence. An index entry, once
    present, must name a complete paired/unpaired artifact.
    """

    def __init__(
        self,
        index: Mapping[str, object] | str | os.PathLike[str],
        root: str | os.PathLike[str],
        *,
        paired_filename: str = "uniref100_hits.a3m",
        unpaired_filename: str = "mmseqs_other_hits.a3m",
    ) -> None:
        if isinstance(index, Mapping):
            self.index = index
        else:
            payload = json.loads(Path(index).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("an indexed MSA mapping must be a JSON object")
            self.index = payload
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"MSA root directory does not exist: {self.root}")
        self.paired_filename = paired_filename
        self.unpaired_filename = unpaired_filename

    def __call__(self, structure: Structure) -> MsaCollection | None:
        r"""Return protein evidence for ``structure``, or ``None`` without proteins."""

        entities = _protein_entity_sequences(structure)
        if not entities:
            return None

        paired_paths: dict[str, Path] = {}
        unpaired_paths: dict[str, Path] = {}
        for entity_id, sequence in entities:
            directory_id = self.index.get(sequence)
            if directory_id is None:
                continue
            directory = self.root / str(directory_id)
            paired_path = directory / self.paired_filename
            unpaired_path = directory / self.unpaired_filename
            missing = [path for path in (paired_path, unpaired_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"incomplete indexed MSA artifact for entity {entity_id!r}; "
                    f"required file does not exist: {missing[0]}"
                )
            paired_paths[entity_id] = paired_path
            unpaired_paths[entity_id] = unpaired_path

        if not paired_paths:
            return _query_only_collection(entities)

        unpaired_pools = {entity_id: (paired_paths[entity_id], unpaired_paths[entity_id]) for entity_id in paired_paths}
        paired = pair_msas(unpaired_pools, paired_paths)
        available = {msa.entity_id: msa for msa in paired.msas}
        msas = tuple(
            available[entity_id] if entity_id in available else _query_only_msa(entity_id, sequence)
            for entity_id, sequence in entities
        )
        row_pairing = np.full((paired.row_pairing.shape[0], len(msas)), -1, dtype=np.int64)
        row_is_paired = np.zeros_like(row_pairing, dtype=np.bool_)
        available_columns = {msa.entity_id: column for column, msa in enumerate(paired.msas)}
        for output_column, (entity_id, _sequence) in enumerate(entities):
            source_column = available_columns.get(entity_id)
            if source_column is None:
                row_pairing[0, output_column] = 0
                row_is_paired[0, output_column] = True
            else:
                row_pairing[:, output_column] = paired.row_pairing[:, source_column]
                row_is_paired[:, output_column] = paired.row_is_paired[:, source_column]
        return MsaCollection(msas, row_pairing, row_is_paired)


def _protein_entity_sequences(structure: Structure) -> tuple[tuple[str, str], ...]:
    r"""Return one canonical MSA query per distinct protein entity."""

    system = structure.system
    entities: list[tuple[str, str]] = []
    seen: set[int] = set()
    for instance, raw_entity in enumerate(system.instance_entity.tolist()):
        entity = int(raw_entity)
        if entity in seen or int(system.entity_type[entity]) != ENTITY_PROTEIN:
            continue
        seen.add(entity)
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        sequence: list[str] = []
        for component in range(start, stop):
            parent = int(structure.standard_component_id[component])
            component_id = (
                structure.source_symbols[parent]
                if parent >= 0
                else system.symbols[int(system.component_symbol[component])]
            )
            sequence.append(TOKEN_ALPHABET[component_token_index(component_id, "protein")])
        entities.append((system.entity_id(entity), "".join(sequence)))
    return tuple(entities)


def _query_only_msa(entity_id: str, sequence: str) -> MSA:
    deletions: NDArray[np.uint32] = np.zeros((1, len(sequence)), dtype=np.uint32)
    return MSA(
        entity_id,
        (sequence,),
        deletions,
        (sequence,),
        deletions.copy(),
    )


def _query_only_collection(entities: Sequence[tuple[str, str]]) -> MsaCollection:
    msas = tuple(_query_only_msa(entity_id, sequence) for entity_id, sequence in entities)
    return MsaCollection(
        msas,
        np.zeros((1, len(msas)), dtype=np.int64),
        np.ones((1, len(msas)), dtype=np.bool_),
    )


def _read_compressed_fasta_records(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    identifier: str | None = None
    comment: str | None = None
    sequence: list[str] = []
    saw_header = False

    def append_record() -> None:
        nonlocal identifier, comment, sequence, saw_header
        if not saw_header and not sequence:
            return
        if saw_header and not sequence:
            raise ValueError(f"missing sequence line after defline in {path}")
        records.append(SequenceRecord("".join(sequence), identifier, comment))
        identifier = None
        comment = None
        sequence = []
        saw_header = False

    if path.suffix.lower() == ".gz":
        handle = cast(TextIO, gzip.open(path, "rt"))
    elif path.suffix.lower() == ".xz":
        handle = cast(TextIO, lzma.open(path, "rt"))
    else:
        handle = cast(TextIO, zstandard.open(path, "rt"))
    with handle as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith(">"):
                append_record()
                saw_header = True
                parts = line[1:].strip().split(None, 1)
                identifier = parts[0] if parts else None
                comment = parts[1] if len(parts) == 2 else None
            else:
                sequence.append(line)
    append_record()
    return records


def _encode_msa(msa: MSA) -> dict[str, object]:
    r"""
    Encode an MSA as a JSON-safe object.
    """

    return {
        "entity_id": msa.entity_id,
        "unpaired_sequences": list(msa.unpaired_sequences),
        "unpaired_deletion_counts": msa.unpaired_deletion_counts.tolist(),
        "paired_sequences": list(msa.paired_sequences),
        "paired_deletion_counts": msa.paired_deletion_counts.tolist(),
    }


def _decode_msa(value: object) -> MSA:
    r"""
    Decode an MSA from a JSON-safe object.
    """

    data = cast(dict[str, object], value)
    return MSA(
        entity_id=cast(str, data["entity_id"]),
        unpaired_sequences=tuple(cast(list[str], data["unpaired_sequences"])),
        unpaired_deletion_counts=np.asarray(
            data["unpaired_deletion_counts"],
            dtype=np.uint32,
        ),
        paired_sequences=tuple(cast(list[str], data["paired_sequences"])),
        paired_deletion_counts=np.asarray(
            data["paired_deletion_counts"],
            dtype=np.uint32,
        ),
    )


def encode_msa_collection(collection: MsaCollection) -> dict[str, object]:
    r"""
    Encode an MSA collection as a JSON-safe object.
    """

    return {
        "msas": [_encode_msa(msa) for msa in collection.msas],
        "row_pairing": collection.row_pairing.tolist(),
        "row_is_paired": collection.row_is_paired.tolist(),
        "target_instance_ids": (
            None if collection.target_instance_ids is None else list(collection.target_instance_ids)
        ),
    }


def decode_msa_collection(value: object) -> MsaCollection:
    r"""
    Decode an MSA collection from a JSON-safe object.
    """

    data = cast(dict[str, object], value)
    target_value = data["target_instance_ids"]
    target_instance_ids = None if target_value is None else tuple(cast(list[str], target_value))
    return MsaCollection(
        msas=tuple(_decode_msa(item) for item in cast(list[object], data["msas"])),
        row_pairing=np.asarray(data["row_pairing"], dtype=np.int64),
        row_is_paired=np.asarray(data["row_is_paired"], dtype=np.bool_),
        target_instance_ids=target_instance_ids,
    )


def _project_a3m_row(sequence: str, *, is_query: bool) -> tuple[str, list[int]]:
    r"""
    Project one A3M row: uppercase/``-`` emit, lowercase count, ``.`` drop.
    """

    aligned: list[str] = []
    deletions: list[int] = []
    insertion_count = 0
    for symbol in sequence:
        if "a" <= symbol <= "z":
            if is_query:
                raise ValueError("the A3M query must not contain lowercase insertions")
            insertion_count += 1
        elif symbol == ".":
            continue
        elif "A" <= symbol <= "Z" or symbol == "-":
            if is_query and symbol == "-":
                raise ValueError("the A3M query must be ungapped")
            aligned.append(symbol)
            deletions.append(insertion_count)
            insertion_count = 0
        else:
            raise ValueError(f"invalid A3M symbol: {symbol!r}")
    return "".join(aligned), deletions


def _project_aligned_fasta_row(sequence: str, query: str) -> tuple[str, list[int]]:
    r"""
    Project one aligned-FASTA row against the query mask of retained columns.
    """

    row = _normalize_aligned_fasta(sequence)
    mask = _normalize_aligned_fasta(query)
    if len(row) != len(mask):
        raise ValueError("aligned FASTA rows must have equal width before projection")

    aligned: list[str] = []
    deletions: list[int] = []
    insertion_count = 0
    for query_symbol, symbol in zip(mask, row, strict=True):
        if query_symbol == "-":
            if symbol != "-":
                insertion_count += 1
        else:
            aligned.append(symbol)
            deletions.append(insertion_count)
            insertion_count = 0
    return "".join(aligned), deletions


def _normalize_aligned_fasta(sequence: str) -> str:
    r"""
    Uppercase letters and fold ``.`` into ``-``; reject other symbols.
    """

    normalized: list[str] = []
    for symbol in sequence:
        if "A" <= symbol <= "Z":
            normalized.append(symbol)
        elif "a" <= symbol <= "z":
            normalized.append(symbol.upper())
        elif symbol == "." or symbol == "-":
            normalized.append("-")
        else:
            raise ValueError(f"invalid aligned FASTA symbol: {symbol!r}")
    return "".join(normalized)
