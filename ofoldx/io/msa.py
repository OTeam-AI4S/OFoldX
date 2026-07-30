r"""
MSA file input projected to the query axis of a named entity.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path

import numpy as np
from multimolecule.io import FASTA, read_fasta_records

from ..data.evidence import MSA

_ANNOTATION_PREFIXES = ("ss_", "sa_", "aa_")


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
        path: MSA path. ``.a3m`` selects A3M parsing; supported FASTA suffixes
            select aligned FASTA parsing.
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
    if suffix == ".a3m":
        a3m = True
    elif suffix.removeprefix(".") in FASTA:
        a3m = False
    else:
        raise ValueError(f"unsupported MSA filename suffix: {suffix or '<none>'}")

    biological_records = (
        record
        for record in read_fasta_records(path)
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
    unpaired_deletion_counts = np.zeros((len(sequences), width), dtype=np.uint32)
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
            paired_deletion_counts=unpaired_deletion_counts[0:1],
        ),
        tuple(" ".join(part for part in (record.id, record.comment) if part is not None) for record in records),
    )


def pair_msa_rows_by_key(
    keys_by_msa: Sequence[Sequence[str]],
    *,
    max_rows_per_key: int,
) -> Iterator[np.ndarray]:
    r"""
    Align source-row indices for shared nonempty keys by chain coverage.
    """

    rows_by_key_by_msa: list[dict[str, list[int]]] = []
    key_coverage: dict[str, int] = {}
    min_rows_by_key: dict[str, int] = {}
    for keys in keys_by_msa:
        rows_by_key: dict[str, list[int]] = {}
        for row, key in enumerate(keys):
            if key:
                rows_by_key.setdefault(key, []).append(row)
        rows_by_key_by_msa.append(rows_by_key)
        for key in sorted(rows_by_key):
            key_coverage[key] = key_coverage.get(key, 0) + 1
            min_rows_by_key[key] = min(
                min_rows_by_key.get(key, max_rows_per_key),
                len(rows_by_key[key]),
            )

    for coverage in sorted(set(key_coverage.values()), reverse=True):
        blocks: list[np.ndarray] = []
        for key, key_count in key_coverage.items():
            if key_count != coverage or coverage <= 1:
                continue
            row_count = min_rows_by_key[key]
            block = np.full((row_count, len(keys_by_msa)), -1, dtype=np.int64)
            for column, rows_by_key in enumerate(rows_by_key_by_msa):
                if key in rows_by_key:
                    block[:, column] = rows_by_key[key][:row_count]
            blocks.append(block)
        if blocks:
            yield np.concatenate(blocks, axis=0)


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
