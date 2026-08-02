r"""
MSA file input projected to the query axis of a named entity.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from itertools import islice
from pathlib import Path
from typing import overload

import numpy as np
from multimolecule.io import FASTA, read_fasta_records

from ..data.evidence import MSA, MSACollection
from ..data.molecular import ENTITY_PROTEIN
from ..data.structure import Structure

_ANNOTATION_PREFIXES = ("ss_", "sa_", "aa_")


@overload
def read_msa(
    path: str | os.PathLike[str],
    *,
    entity_id: str,
    max_rows: int | None = None,
) -> MSA:
    ...


@overload
def read_msa(
    structure: Structure,
    seq_to_pdb_index: Mapping[str, object],
    msa_path: str | os.PathLike[str],
) -> MSACollection:
    ...


def read_msa(
    path: str | os.PathLike[str] | Structure,
    seq_to_pdb_index: Mapping[str, object] | None = None,
    msa_path: str | os.PathLike[str] | None = None,
    *,
    entity_id: str | None = None,
    max_rows: int | None = None,
) -> MSA | MSACollection:
    r"""
    Read one MSA file or collect per-chain Protenix alignments for a structure.

    File mode preserves the original ``read_msa(path, *, entity_id, max_rows=None)``
    interface. Structure mode accepts an already-loaded sequence-to-directory
    mapping and an MSA root containing ``uniref100_hits.a3m`` paired candidates
    and ``mmseqs_other_hits.a3m`` unpaired pools.

    Args:
        path: MSA path in file mode, or a structure in structure mode.
        seq_to_pdb_index: Already-loaded exact protein-sequence to MSA-directory
            mapping. Only valid in structure mode.
        msa_path: Root of the indexed MSA directories. Only valid in structure mode.
        entity_id: Target entity ID stored with a file alignment.
        max_rows: Optional maximum biological rows in file mode.

    Returns:
        An [`MSA`][ofoldx.data.MSA] in file mode, or an instance-bound
        [`MSACollection`][ofoldx.data.MSACollection] in structure mode.

    Raises:
        TypeError: If arguments from the two modes are mixed.
        ValueError: If a structure has no protein chains, or an available MSA is
            malformed or has a query that differs from its chain sequence.
        FileNotFoundError: If the MSA root does not exist, or an indexed directory
            contains only one of the two required MSA files. A missing sequence
            mapping, missing indexed directory, or directory containing neither
            MSA file falls back to query-only evidence for that entity.
    """

    if isinstance(path, Structure):
        if entity_id is not None or max_rows is not None:
            raise TypeError("entity_id and max_rows are only valid when reading a single MSA file")
        if seq_to_pdb_index is None:
            raise TypeError("seq_to_pdb_index is required when reading MSAs for a structure")
        if not isinstance(seq_to_pdb_index, Mapping):
            raise TypeError("seq_to_pdb_index must be an already-loaded mapping")
        if msa_path is None:
            raise TypeError("msa_path is required when reading MSAs for a structure")
        return _read_structure_msas(path, seq_to_pdb_index, msa_path)

    if seq_to_pdb_index is not None or msa_path is not None:
        raise TypeError("seq_to_pdb_index and msa_path are only valid when reading MSAs for a structure")
    if entity_id is None:
        raise TypeError("entity_id is required when reading a single MSA file")
    return read_msa_with_headers(path, entity_id=entity_id, max_rows=max_rows)[0]


def _read_structure_msas(
    structure: Structure,
    seq_to_pdb_index: Mapping[str, object],
    msa_path: str | os.PathLike[str],
) -> MSACollection:
    r"""
    Resolve, pair, and instance-bind every protein chain in ``structure``.
    """

    root = Path(msa_path)
    if not root.is_dir():
        raise FileNotFoundError(f"MSA root directory does not exist: {root}")

    system = structure.system
    instance_entities = system.instance_entity.detach().cpu().tolist()
    component_pointers = system.instance_component_ptr.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()

    entity_bindings: list[tuple[str, str]] = []
    seen_entities: set[str] = set()
    for instance_index, entity_index in enumerate(instance_entities):
        if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
            continue

        entity_id = system.entity_id(entity_index)
        sequence = "".join(
            _protein_component_one_letter(system.symbols[component_symbols[index]])
            for index in range(component_pointers[instance_index], component_pointers[instance_index + 1])
        )
        # One MSA column per entity is the smallest correct representation: an
        # unbound column broadcasts to all repeated instances, while distinct
        # entities remain distinct even when they happen to have the same sequence.
        if entity_id in seen_entities:
            continue
        entity_bindings.append((entity_id, sequence))
        seen_entities.add(entity_id)

    if not entity_bindings:
        raise ValueError("structure contains no protein chains")

    unpaired_paths: dict[str, Path] = {}
    paired_paths: dict[str, Path] = {}
    fallback_entities: set[str] = set()
    for entity_id, sequence in entity_bindings:
        if sequence not in seq_to_pdb_index:
            fallback_entities.add(entity_id)
            continue
        directory_id = seq_to_pdb_index[sequence]

        directory = root / str(directory_id)
        if not directory.is_dir():
            fallback_entities.add(entity_id)
            continue

        paired_path = directory / "uniref100_hits.a3m"
        unpaired_path = directory / "mmseqs_other_hits.a3m"
        has_paired = paired_path.is_file()
        has_unpaired = unpaired_path.is_file()
        if not has_paired and not has_unpaired:
            fallback_entities.add(entity_id)
            continue
        if has_paired != has_unpaired:
            missing = unpaired_path if has_paired else paired_path
            raise FileNotFoundError(
                f"incomplete MSA artifact for entity {entity_id!r}; required file does not exist: {missing}"
            )

        unpaired_paths[entity_id] = unpaired_path
        paired_paths[entity_id] = paired_path

    if not paired_paths:
        query_msas = tuple(_query_only_msa(entity_id, sequence) for entity_id, sequence in entity_bindings)
        return MSACollection(
            msas=query_msas,
            row_pairing=np.zeros((1, len(query_msas)), dtype=np.int64),
            row_is_paired=np.ones((1, len(query_msas)), dtype=np.bool_),
        )

    # Lazy import avoids making the general MSA reader depend eagerly on a model
    # family and reuses Protenix's released header pairing without duplication.
    from ..models.protenix.pipeline import pair_msas

    # Protenix uses UniRef100 both for species pairing and as the first part of
    # the unpaired pool, then appends MMseqs "other" hits with stable sequence
    # deduplication. Let the family helper own species pairing, and only assemble
    # its two source pools here.
    paired = pair_msas(paired_paths, paired_paths, remove_paired_from_unpaired=False)
    paired = _merge_protenix_unpaired_pool(paired, unpaired_paths)
    available_sequences = dict(entity_bindings)
    available_msas: dict[str, MSA] = {}
    for msa in paired.msas:
        entity_id = msa.entity_id
        sequence = available_sequences[entity_id]
        query = msa.unpaired_sequences[0]
        if query != sequence:
            raise ValueError(
                f"MSA query for entity {entity_id!r} is {query!r}, but the structure chain sequence is {sequence!r}"
            )
        available_msas[entity_id] = MSA(
            entity_id=entity_id,
            unpaired_sequences=msa.unpaired_sequences,
            unpaired_deletion_counts=msa.unpaired_deletion_counts,
            paired_sequences=msa.paired_sequences,
            paired_deletion_counts=msa.paired_deletion_counts,
        )

    rebound_msas = tuple(
        _query_only_msa(entity_id, sequence)
        if entity_id in fallback_entities
        else available_msas[entity_id]
        for entity_id, sequence in entity_bindings
    )
    row_pairing = np.full((paired.row_pairing.shape[0], len(rebound_msas)), -1, dtype=np.int64)
    row_is_paired = np.zeros_like(row_pairing, dtype=np.bool_)
    available_columns = {msa.entity_id: column for column, msa in enumerate(paired.msas)}
    for output_column, (entity_id, _) in enumerate(entity_bindings):
        if entity_id in fallback_entities:
            row_pairing[0, output_column] = 0
            row_is_paired[0, output_column] = True
            continue
        source_column = available_columns[entity_id]
        row_pairing[:, output_column] = paired.row_pairing[:, source_column]
        row_is_paired[:, output_column] = paired.row_is_paired[:, source_column]

    return MSACollection(
        msas=rebound_msas,
        row_pairing=row_pairing,
        row_is_paired=row_is_paired,
    )


def _query_only_msa(entity_id: str, sequence: str) -> MSA:
    r"""Build the canonical query-only fallback for one protein entity."""

    deletions = np.zeros((1, len(sequence)), dtype=np.uint32)
    return MSA(
        entity_id=entity_id,
        unpaired_sequences=(sequence,),
        unpaired_deletion_counts=deletions,
        paired_sequences=(sequence,),
        paired_deletion_counts=deletions.copy(),
    )


def _merge_protenix_unpaired_pool(
    paired: MSACollection,
    mmseqs_paths: Mapping[str, str | os.PathLike[str]],
) -> MSACollection:
    r"""
    Append MMseqs rows to UniRef rows and rebuild only the unpaired plan.

    Species pairing and its row order are retained exactly from ``pair_msas``.
    """

    merged_msas: list[MSA] = []
    for uniref, entity_id in zip(paired.msas, mmseqs_paths, strict=True):
        mmseqs = read_msa(mmseqs_paths[entity_id], entity_id=entity_id)
        if uniref.unpaired_sequences[0] != mmseqs.unpaired_sequences[0]:
            raise ValueError(f"uniref100 and mmseqs_other MSA queries differ for entity {entity_id!r}")

        sequences: list[str] = []
        deletions: list[np.ndarray] = []
        seen: set[str] = set()
        for msa in (uniref, mmseqs):
            for sequence, deletion_counts in zip(
                msa.unpaired_sequences,
                msa.unpaired_deletion_counts,
                strict=True,
            ):
                if sequence in seen:
                    continue
                seen.add(sequence)
                sequences.append(sequence)
                deletions.append(deletion_counts)
        merged_msas.append(
            MSA(
                entity_id=entity_id,
                unpaired_sequences=tuple(sequences),
                unpaired_deletion_counts=np.stack(deletions).astype(np.uint32, copy=False),
                paired_sequences=uniref.paired_sequences,
                paired_deletion_counts=uniref.paired_deletion_counts,
            )
        )

    paired_rows = np.any(paired.row_is_paired, axis=1)
    paired_plan = paired.row_pairing[paired_rows]
    paired_is_paired = paired.row_is_paired[paired_rows]

    unpaired_sources: list[list[int]] = []
    for column, msa in enumerate(merged_msas):
        selected_paired_sequences = {
            msa.paired_sequences[int(row)] for row in paired_plan[:, column] if row != -1
        }
        unpaired_sources.append(
            [
                row
                for row, sequence in enumerate(msa.unpaired_sequences)
                if sequence not in selected_paired_sequences
            ]
        )

    unpaired_plan = np.full(
        (max((len(rows) for rows in unpaired_sources), default=0), len(merged_msas)),
        -1,
        dtype=np.int64,
    )
    for column, rows in enumerate(unpaired_sources):
        unpaired_plan[: len(rows), column] = rows

    return MSACollection(
        msas=tuple(merged_msas),
        row_pairing=np.concatenate((paired_plan, unpaired_plan), axis=0),
        row_is_paired=np.concatenate(
            (paired_is_paired, np.zeros_like(unpaired_plan, dtype=np.bool_)),
            axis=0,
        ),
    )


@cache
def _protein_component_one_letter(component_id: str) -> str:
    r"""
    Lower a protein component ID to one MSA symbol.

    Unknown CCD entries and mappings that are not exactly one character are
    represented by ``X``, matching Protenix's CCD sequence lowering.
    """

    # Biotite reads wwPDB's CCD one-letter field and therefore handles canonical
    # modified residues such as MSE and SEP. Keep it lazy so legacy file reads do
    # not pay the CCD import cost.
    from biotite.structure.info import one_letter_code

    try:
        symbol = one_letter_code(component_id)
    except KeyError:
        symbol = None
    return symbol if symbol is not None and len(symbol) == 1 else "X"


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
