r"""
Multiple-sequence alignments and structural templates.
"""

from __future__ import annotations

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


def _instance_component_bounds(system: MolecularSystem, instance_index: int) -> tuple[int, int]:
    pointers = system.instance_component_ptr
    return int(pointers[instance_index]), int(pointers[instance_index + 1])


def _entity_component_bounds(system: MolecularSystem, entity_index: int) -> tuple[int, int]:
    instance_index = int((system.instance_entity == entity_index).nonzero(as_tuple=False)[0, 0])
    return _instance_component_bounds(system, instance_index)


def _query_conflicts_with_entity(
    query: str,
    system: MolecularSystem,
    entity_index: int,
) -> int | None:
    r"""
    Locate the first query column whose residue chemistry contradicts the target sequence.

    The MSA query row is the one-letter target sequence. A column conflicts only when the query symbol
    and the bound target residue resolve to distinct *standard* components: a modified target residue the
    one-letter alphabet cannot name, or a wildcard query symbol (``X`` and the other non-standard
    conventions), is treated as agreeing. Returns the offending column index, or ``None`` when the query
    is consistent with the target.
    """

    polymer_type = _POLYMER_TYPE.get(int(system.entity_type[entity_index]))
    if polymer_type is None:
        return None
    standard_ids = _STANDARD_COMPONENT_IDS.get(polymer_type)
    if standard_ids is None:
        return None
    start, stop = _entity_component_bounds(system, entity_index)
    component_symbols = system.component_symbol[start:stop].detach().cpu().tolist()
    for index, (symbol, component_symbol) in enumerate(zip(query, component_symbols)):
        target_id = system.symbols[component_symbol]
        if target_id not in standard_ids:
            continue
        try:
            query_id = one_letter_to_component(symbol, polymer_type, policy="strict")
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

    Args:
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
class MSACollection:
    r"""
    A row assembly plan spanning one or more entity MSAs.

    Column ``c`` refers to ``msas[c]``. ``row_pairing[r, c]`` selects a row from
    that MSA, or ``-1`` for a gap; ``row_is_paired[r, c]`` selects the paired or
    unpaired row pool.

    Args:
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


def msa_bindings(msa: MSA | MSACollection) -> tuple[tuple[MSA, str | None], ...]:
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
    msa: MSA | MSACollection,
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
        query = column.unpaired_sequences[0]
        width = len(query)
        start, stop = _entity_component_bounds(system, entity_index)
        component_count = stop - start
        if width != component_count:
            raise ValueError(
                f"MSA for entity {column.entity_id!r} spans {width} columns "
                f"but the entity has {component_count} components"
            )
        conflict = _query_conflicts_with_entity(query, system, entity_index)
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

    Args:
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

    Args:
        structure: Template structure containing every referenced template instance.
        alignments: Target-to-template polymer alignments sharing ``structure``.

    Raises:
        ValueError: If no alignment is supplied or an alignment references an invalid
            template polymer component.
    """

    structure: Structure
    alignments: tuple[TemplateAlignment, ...]

    def __post_init__(self) -> None:
        if not self.alignments:
            raise ValueError("alignments must not be empty")
        for alignment in self.alignments:
            system = self.structure.system
            instance_index = system.instance_index(alignment.template_instance_id)
            entity_index = int(system.instance_entity[instance_index])
            if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
                raise ValueError("template_instance_id must reference a polymer instance")
            start, stop = _instance_component_bounds(system, instance_index)
            if alignment.residue_pairs[-1][1] >= stop - start:
                raise ValueError("residue pair template index is outside the template instance")
