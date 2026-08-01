r"""
Multiple-sequence alignments and structural templates.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .molecular import (
    _STANDARD_COMPONENTS,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MolecularSystem,
    one_letter_to_component,
)
from .structure import Structure
from .tokenization import (
    TOKEN_ALPHABET,
    TOKEN_GAP_INDEX,
    TOKEN_INDEX,
    TOKEN_OTHER_INDEX,
    TOKEN_VOCAB_SIZE,
    component_token_index,
    residue_token_index,
)

MSA_ALPHABET = TOKEN_ALPHABET
MSA_INDEX = TOKEN_INDEX
MSA_VOCAB_SIZE = TOKEN_VOCAB_SIZE
MSA_GAP_INDEX = TOKEN_GAP_INDEX
MSA_OTHER_INDEX = TOKEN_OTHER_INDEX
msa_residue_index = residue_token_index
msa_component_index = component_token_index

_MSA_ASCII_INDEX = {
    polymer_type: np.fromiter(
        (residue_token_index(chr(code), polymer_type) for code in range(128)),
        dtype=np.int64,
        count=128,
    )
    for polymer_type in ("protein", "rna", "dna")
}


# Component IDs a one-letter alphabet names unambiguously, per polymer chemistry. Used to check an MSA
# query row against its target sequence: only standard residues on both sides can genuinely conflict.
_STANDARD_COMPONENT_IDS: dict[str, frozenset[str]] = {
    polymer_type: frozenset(mapping.values()) for polymer_type, mapping in _STANDARD_COMPONENTS.items()
}
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
    ENTITY_PNA: "pna",
}

_PROTEIN_QUERY_COMPONENTS: dict[str, frozenset[str]] = {
    **{symbol: frozenset((component,)) for symbol, component in _STANDARD_COMPONENTS["protein"].items()},
    "B": frozenset(("ASP", "ASN", "ASX")),
    "Z": frozenset(("GLU", "GLN", "GLX")),
    "J": frozenset(("ILE", "LEU", "XLE")),
    "U": frozenset(("SEC",)),
    "O": frozenset(("PYL",)),
}
_KNOWN_PROTEIN_QUERY_COMPONENTS = frozenset(
    component for components in _PROTEIN_QUERY_COMPONENTS.values() for component in components
)


def encode_msa_sequences(sequences: Sequence[str], polymer_type: str) -> NDArray[np.int64]:
    r"""Encode an aligned sequence pool in the canonical token vocabulary."""

    if not sequences:
        return np.empty((0, 0), dtype=np.int64)
    width = len(sequences[0])
    if any(len(sequence) != width for sequence in sequences):
        raise ValueError("every MSA row must have the same width")
    codes = np.frombuffer("".join(sequences).encode("ascii", "replace"), dtype=np.uint8).reshape(len(sequences), width)
    table = _MSA_ASCII_INDEX.get(polymer_type)
    if table is None:
        return np.full(codes.shape, residue_token_index("?", polymer_type), dtype=np.int64)
    return table[codes]


def _instance_component_bounds(system: MolecularSystem, instance_index: int) -> tuple[int, int]:
    pointers = system.instance_component_ptr
    return int(pointers[instance_index]), int(pointers[instance_index + 1])


def _query_conflicts_with_instance(
    query: str,
    system: MolecularSystem,
    instance_index: int,
) -> int | None:
    r"""
    Locate the first query column whose residue chemistry contradicts the target sequence.

    The MSA query row is the one-letter target sequence. A column conflicts only when the query symbol
    and the bound target residue resolve to distinct known components. Protein ``B``, ``Z``, and ``J``
    accept their two IUPAC constituents; ``U`` and ``O`` name SEC and PYL exactly; only ``X`` is a wildcard.
    A modified target residue the one-letter alphabet cannot name is treated as agreeing. Returns the
    offending column index, or ``None`` when the query is consistent with the target.
    """

    entity_index = int(system.instance_entity[instance_index])
    polymer_type = _POLYMER_TYPE.get(int(system.entity_type[entity_index]))
    if polymer_type is None:
        return None
    standard_ids = (
        _STANDARD_COMPONENT_IDS["rna"] | _STANDARD_COMPONENT_IDS["dna"]
        if polymer_type == "dna_rna_hybrid"
        else _STANDARD_COMPONENT_IDS.get(polymer_type)
    )
    if standard_ids is None:
        return None
    start, stop = _instance_component_bounds(system, instance_index)
    component_symbols = system.component_symbol[start:stop].detach().cpu().tolist()
    for index, (symbol, component_symbol) in enumerate(zip(query, component_symbols)):
        target_id = system.symbols[component_symbol]
        if polymer_type == "protein":
            if target_id not in _KNOWN_PROTEIN_QUERY_COMPONENTS:
                continue
            if symbol == "X":
                continue
            accepted = _PROTEIN_QUERY_COMPONENTS.get(symbol)
            if accepted is None or target_id not in accepted:
                return index
        else:
            if target_id not in standard_ids:
                continue
            query_polymer_type = polymer_type
            if polymer_type == "dna_rna_hybrid":
                query_polymer_type = "rna" if target_id in _STANDARD_COMPONENT_IDS["rna"] else "dna"
            try:
                query_id = one_letter_to_component(symbol, query_polymer_type, policy="strict")
            except KeyError:
                continue
            if query_id != target_id:
                return index
    return None


@dataclass(eq=False)
class MSA:
    r"""
    Paired and unpaired alignments for one molecular entity.

    Sequences use uppercase one-letter MSA symbols and ``-`` for gaps. Row zero
    of both pools is the same ungapped query sequence.

    Attributes:
        entity_id: Target entity ID in the associated molecular system.
        unpaired_sequences: Query-width rows in the unpaired MSA pool.
        unpaired_deletion_counts: Unsigned deletion counts with shape
            ``(num_unpaired_rows, query_length)``.
        paired_sequences: Query-width rows in the paired MSA pool.
        paired_deletion_counts: Unsigned deletion counts with shape
            ``(num_paired_rows, query_length)``.

    Raises:
        ValueError: If query rows, row widths, deletion-count shapes, or dtypes are
            inconsistent.
    """

    entity_id: str
    unpaired_sequences: tuple[str, ...]
    unpaired_deletion_counts: NDArray[np.uint32]
    paired_sequences: tuple[str, ...]
    paired_deletion_counts: NDArray[np.uint32]

    def __post_init__(self) -> None:
        if not self.unpaired_sequences:
            raise ValueError("unpaired_sequences must contain the query row")
        if not self.paired_sequences:
            raise ValueError("paired_sequences must contain the query row")

        width = len(self.unpaired_sequences[0])
        if any(len(sequence) != width for sequence in self.unpaired_sequences):
            raise ValueError("every unpaired row must have the query width")
        if any(len(sequence) != width for sequence in self.paired_sequences):
            raise ValueError("every paired row must have the query width")

        if self.unpaired_sequences[0] != self.paired_sequences[0]:
            raise ValueError("the unpaired and paired query rows must be identical")
        if "-" in self.unpaired_sequences[0]:
            raise ValueError("the MSA query row must be ungapped")

        unpaired_shape = (len(self.unpaired_sequences), width)
        if self.unpaired_deletion_counts.shape != unpaired_shape:
            raise ValueError(f"unpaired_deletion_counts must have shape {unpaired_shape}")
        if self.unpaired_deletion_counts.dtype != np.uint32:
            raise ValueError("unpaired_deletion_counts must have dtype uint32")

        paired_shape = (len(self.paired_sequences), width)
        if self.paired_deletion_counts.shape != paired_shape:
            raise ValueError(f"paired_deletion_counts must have shape {paired_shape}")
        if self.paired_deletion_counts.dtype != np.uint32:
            raise ValueError("paired_deletion_counts must have dtype uint32")


@dataclass(eq=False)
class MsaCollection:
    r"""
    A row assembly plan spanning one or more entity MSAs.

    Column ``c`` refers to ``msas[c]``. ``row_pairing[r, c]`` selects a row from
    that MSA, or ``-1`` for a gap; ``row_is_paired[r, c]`` selects the paired or
    unpaired row pool.

    Attributes:
        msas: MSA columns participating in the assembly.
        row_pairing: Signed integer array with shape ``(num_rows, num_msas)``.
        row_is_paired: Boolean array aligned with ``row_pairing``.
        target_instance_ids: Optional exact target instance for each MSA column.
            When absent, each column applies to every instance of its entity.

    Raises:
        ValueError: If arrays, row indices, query rows, entities, or instance targets
            are inconsistent.
    """

    msas: tuple[MSA, ...]
    row_pairing: NDArray[np.int64]
    row_is_paired: NDArray[np.bool_]
    target_instance_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.msas:
            raise ValueError("msas must not be empty")
        if self.row_pairing.ndim != 2 or self.row_pairing.shape[0] == 0:
            raise ValueError("row_pairing must be a 2D array holding the query row")

        column_count = len(self.msas)
        if self.row_pairing.shape != self.row_is_paired.shape:
            raise ValueError("row_pairing and row_is_paired must have the same shape")
        if self.row_pairing.shape[1] != column_count:
            raise ValueError("row_pairing must have one column per MSA")
        if self.row_pairing.dtype != np.int64:
            raise ValueError("row_pairing must have dtype int64")
        if self.row_is_paired.dtype != np.bool_:
            raise ValueError("row_is_paired must have dtype bool")

        if not np.all(self.row_pairing[0] == 0):
            raise ValueError("the first plan row must select query row zero in every column")
        if not np.all(self.row_is_paired[0]):
            raise ValueError("the first plan row must select the paired pool in every column")

        gap = self.row_pairing == -1
        if np.any(gap & self.row_is_paired):
            raise ValueError("a gap cell cannot be marked paired")
        if np.any(np.all(gap, axis=1)):
            raise ValueError("each plan row must select at least one populated cell")

        for column_index, msa in enumerate(self.msas):
            pairing = self.row_pairing[:, column_index]
            paired = self.row_is_paired[:, column_index]
            populated = pairing != -1
            paired_cells = populated & paired
            unpaired_cells = populated & ~paired
            paired_bound = len(msa.paired_sequences)
            unpaired_bound = len(msa.unpaired_sequences)
            if np.any((pairing[paired_cells] < 0) | (pairing[paired_cells] >= paired_bound)):
                raise ValueError("row_pairing selects a paired row outside its MSA")
            if np.any((pairing[unpaired_cells] < 0) | (pairing[unpaired_cells] >= unpaired_bound)):
                raise ValueError("row_pairing selects an unpaired row outside its MSA")

        if self.target_instance_ids is None:
            entity_ids = tuple(msa.entity_id for msa in self.msas)
            if len(set(entity_ids)) != len(entity_ids):
                raise ValueError("broadcast MSAs must bind pairwise-unique entity_id columns")
        else:
            if len(self.target_instance_ids) != column_count:
                raise ValueError("target_instance_ids must have one entry per MSA")
            if len(set(self.target_instance_ids)) != len(self.target_instance_ids):
                raise ValueError("target_instance_ids must be pairwise unique")


def merge_unpaired_protein_msas(msas: Sequence[MSA]) -> MSA:
    r"""Merge and canonically deduplicate protein MSA pools for one entity."""

    if not msas:
        raise ValueError("at least one MSA is required")
    source = msas[0]
    sequences: list[str] = []
    deletion_counts: list[NDArray[np.uint32]] = []
    seen: set[tuple[int, ...]] = set()
    for msa in msas:
        if msa.entity_id != source.entity_id:
            raise ValueError("merged MSAs must target the same entity")
        if msa.unpaired_sequences[0] != source.unpaired_sequences[0]:
            raise ValueError(f"unpaired MSA queries differ for entity {source.entity_id!r}")
        for sequence, deletions in zip(msa.unpaired_sequences, msa.unpaired_deletion_counts, strict=True):
            key = tuple(msa_residue_index(symbol, "protein") for symbol in sequence)
            if key not in seen:
                seen.add(key)
                sequences.append(sequence)
                deletion_counts.append(deletions)
    counts = np.stack(deletion_counts)
    return MSA(
        source.entity_id,
        tuple(sequences),
        counts,
        (sequences[0],),
        counts[0:1].copy(),
    )


def pair_msa_rows_by_key(
    keys_by_msa: Sequence[Sequence[str]],
    *,
    max_rows_per_key: int,
) -> Iterator[NDArray[np.int64]]:
    r"""Align source-row indices for shared nonempty keys by chain coverage."""

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
        blocks: list[NDArray[np.int64]] = []
        for key, key_count in key_coverage.items():
            if key_count != coverage or coverage <= 1:
                continue
            row_count = min_rows_by_key[key]
            block: NDArray[np.int64] = np.full((row_count, len(keys_by_msa)), -1, dtype=np.int64)
            for column, rows_by_key in enumerate(rows_by_key_by_msa):
                if key in rows_by_key:
                    block[:, column] = rows_by_key[key][:row_count]
            blocks.append(block)
        if blocks:
            yield np.concatenate(blocks, axis=0)


def unpaired_msa_row_plan(rows_by_msa: Sequence[Sequence[int]]) -> NDArray[np.int64]:
    r"""Build a block-diagonal unpaired plan, interleaved by source depth."""

    plan: NDArray[np.int64] = np.full(
        (sum(len(rows) for rows in rows_by_msa), len(rows_by_msa)),
        -1,
        dtype=np.int64,
    )
    output_row = 0
    for depth in range(max((len(rows) for rows in rows_by_msa), default=0)):
        for column, rows in enumerate(rows_by_msa):
            if depth < len(rows):
                plan[output_row, column] = rows[depth]
                output_row += 1
    return plan


def msa_bindings(msa: MSA | MsaCollection) -> tuple[tuple[MSA, str | None], ...]:
    r"""
    Return each MSA column with its instance target.

    Args:
        msa: One MSA or a collection of paired MSA columns.

    Returns:
        Pairs of MSA columns and exact instance IDs; an instance ID of `None`
        denotes entity-wide broadcast.
    """

    columns = (msa,) if isinstance(msa, MSA) else msa.msas
    targets = (
        (None,) * len(columns) if isinstance(msa, MSA) or msa.target_instance_ids is None else msa.target_instance_ids
    )
    return tuple(zip(columns, targets))


def resolve_msa_bindings(
    msa: MSA | MsaCollection,
    system: MolecularSystem,
) -> tuple[tuple[MSA, int | None, int], ...]:
    r"""
    Resolve MSA entity and instance references against a molecular system.

    Args:
        msa: One MSA or a collection of paired MSA columns.
        system: Molecular system containing the referenced entities and instances.

    Returns:
        Resolved MSA columns with their optional target instance index and entity index.

    Raises:
        ValueError: If a referenced entity or instance is invalid, an instance belongs
            to another entity, an MSA width differs from its polymer length, or the MSA
            query row disagrees with the target entity sequence.
    """

    resolved: list[tuple[MSA, int | None, int]] = []
    for column, target_instance_id in msa_bindings(msa):
        try:
            entity_index = system.entity_index(column.entity_id)
        except KeyError:
            raise ValueError(f"MSA entity {column.entity_id!r} is not a polymer entity in the input system") from None
        if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
            raise ValueError(f"MSA entity {column.entity_id!r} is not a polymer entity in the input system")
        target_instance_index = None
        if target_instance_id is not None:
            try:
                target_instance_index = system.instance_index(target_instance_id)
            except KeyError:
                raise ValueError(f"MSA targets unknown input instance {target_instance_id!r}")
            target_entity_index = int(system.instance_entity[target_instance_index])
            if target_entity_index != entity_index:
                raise ValueError(
                    f"MSA target instance {target_instance_id!r} belongs to entity "
                    f"{system.entity_id(target_entity_index)!r}, not {column.entity_id!r}"
                )
        if target_instance_index is None:
            target_instances = [
                int(index)
                for index in (system.instance_entity == entity_index).nonzero(as_tuple=False).flatten().tolist()
            ]
        else:
            target_instances = [target_instance_index]
        query = column.unpaired_sequences[0]
        width = len(query)
        for instance_index in target_instances:
            start, stop = _instance_component_bounds(system, instance_index)
            component_count = stop - start
            if width != component_count:
                raise ValueError(
                    f"MSA for entity {column.entity_id!r} spans {width} columns "
                    f"but the entity has {component_count} components"
                )
            conflict = _query_conflicts_with_instance(query, system, instance_index)
            if conflict is not None:
                raise ValueError(
                    f"MSA query for entity {column.entity_id!r} disagrees with the target sequence at column {conflict}"
                )
        resolved.append((column, target_instance_index, entity_index))
    return tuple(resolved)


@dataclass(frozen=True)
class TemplateAlignment:
    r"""
    A target-to-template polymer alignment.

    Attributes:
        target_entity_id: Target entity ID in the predicted molecular system.
        target_instance_id: Optional exact target instance; ``None`` broadcasts the
            alignment to all instances of the target entity.
        template_instance_id: Polymer instance in the template structure.
        residue_pairs: Strictly increasing zero-based ``(target, template)`` component
            index pairs.

    Raises:
        ValueError: If instance identifiers or residue pairs are invalid.
    """

    target_entity_id: str
    target_instance_id: str | None
    template_instance_id: str
    residue_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.target_instance_id == "":
            raise ValueError("target_instance_id must be non-empty or None")
        if not self.template_instance_id:
            raise ValueError("template_instance_id must be non-empty")
        if not self.residue_pairs:
            raise ValueError("residue_pairs must not be empty")

        previous_target = -1
        previous_template = -1
        for target_index, template_index in self.residue_pairs:
            if target_index < 0 or template_index < 0:
                raise ValueError("residue pair indices must be non-negative")
            if target_index <= previous_target or template_index <= previous_template:
                raise ValueError("residue_pairs must be strictly increasing on both axes")
            previous_target = target_index
            previous_template = template_index


@dataclass(eq=False)
class Template:
    r"""
    Structural template evidence in a shared coordinate frame.

    Attributes:
        structure: Template structure containing every referenced template instance.
        alignments: Target-to-template polymer alignments sharing ``structure``. Two
            alignments may cover disjoint target ranges, and may place the same
            template position on the same target position, but they must not place
            *different* template positions on one target position: a consumer lowers
            alignments in order, so such a template would depend on their ordering.
            A broadcast alignment covers every instance of its target entity, so it
            overlaps an alignment naming one exact instance of that entity.

    Raises:
        ValueError: If no alignment is supplied, an alignment references an invalid
            template polymer component, or two alignments place different template
            positions on one target position.
    """

    structure: Structure
    alignments: tuple[TemplateAlignment, ...]

    def __post_init__(self) -> None:
        if not self.alignments:
            raise ValueError("alignments must not be empty")
        system = self.structure.system
        placements: dict[tuple[str, int], dict[str | None, tuple[str, int]]] = {}
        for alignment in self.alignments:
            instance_index = system.instance_index(alignment.template_instance_id)
            entity_index = int(system.instance_entity[instance_index])
            if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
                raise ValueError("template_instance_id must reference a polymer instance")
            start, stop = _instance_component_bounds(system, instance_index)
            if alignment.residue_pairs[-1][1] >= stop - start:
                raise ValueError("residue pair template index is outside the template instance")
            for target_index, template_index in alignment.residue_pairs:
                source = (alignment.template_instance_id, template_index)
                covered = placements.setdefault((alignment.target_entity_id, target_index), {})
                for instance_id, placed in covered.items():
                    if placed != source and (
                        instance_id is None or alignment.target_instance_id in (None, instance_id)
                    ):
                        raise ValueError(
                            f"template alignments place {placed[0]!r}:{placed[1]} and "
                            f"{source[0]!r}:{source[1]} on the same target position "
                            f"{alignment.target_entity_id!r}:{target_index}"
                        )
                covered[alignment.target_instance_id] = source
