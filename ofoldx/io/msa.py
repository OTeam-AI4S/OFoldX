r"""
MSA file input projected to the query axis of a named entity.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from multimolecule.io import FASTA, read_fasta_records

from ..data.evidence import MSA

_ANNOTATION_PREFIXES = ("ss_", "sa_", "aa_")


def read_msa(path: str | os.PathLike[str], *, entity_id: str) -> MSA:
    r"""
    Read an A3M or aligned FASTA file as MSA evidence.

    The first biological record is the ungapped query. Insertion columns are
    removed from aligned rows and recorded as deletion counts on the query axis.

    Args:
        path: MSA path. ``.a3m`` selects A3M parsing; supported FASTA suffixes
            select aligned FASTA parsing.
        entity_id: Target entity ID stored with the alignment.

    Returns:
        An MSA whose unpaired pool contains the projected rows and whose paired
        pool contains the query row.

    Raises:
        ValueError: If the suffix, symbols, row widths, or query record are invalid.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".a3m":
        a3m = True
    elif suffix.removeprefix(".") in FASTA:
        a3m = False
    else:
        raise ValueError(f"unsupported MSA filename suffix: {suffix or '<none>'}")

    records = [
        record
        for record in read_fasta_records(path)
        if not (
            record.id is not None and (record.id.startswith(_ANNOTATION_PREFIXES) or record.id.endswith("_consensus"))
        )
    ]
    if not records:
        raise ValueError("MSA contains no biological sequence records")

    sequences = [record.sequence for record in records]
    project = _project_a3m_row if a3m else _project_aligned_fasta_row

    query_row, query_deletions = project(sequences[0], sequences[0], is_query=True)
    width = len(query_row)

    unpaired_sequences: list[str] = [query_row]
    unpaired_deletion_counts = np.zeros((len(sequences), width), dtype=np.uint32)
    unpaired_deletion_counts[0] = query_deletions
    for row_index, sequence in enumerate(sequences[1:], start=1):
        projected, deletions = project(sequence, sequences[0], is_query=False)
        if len(projected) != width:
            raise ValueError("every MSA row must project to the query width")
        unpaired_sequences.append(projected)
        unpaired_deletion_counts[row_index] = deletions

    return MSA(
        entity_id=entity_id,
        unpaired_sequences=tuple(unpaired_sequences),
        unpaired_deletion_counts=unpaired_deletion_counts,
        paired_sequences=(unpaired_sequences[0],),
        paired_deletion_counts=unpaired_deletion_counts[0:1],
    )


def _project_a3m_row(sequence: str, query: str, *, is_query: bool) -> tuple[str, list[int]]:
    r"""
    Project one A3M row: uppercase/``-`` emit, lowercase count, ``.`` drop.
    """

    del query  # A3M encodes insertions per row; no query mask is needed.
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


def _project_aligned_fasta_row(sequence: str, query: str, *, is_query: bool) -> tuple[str, list[int]]:
    r"""
    Project one aligned-FASTA row against the query mask of retained columns.
    """

    del is_query  # The query mask governs every row identically.
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
