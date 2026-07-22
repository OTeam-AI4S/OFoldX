r"""Canonical MSA and structural-template evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .molecular import MolecularSystem, PolymerSequence
from .structure import Structure


@dataclass(eq=False)
class MSA:
    r"""One entity-named alignment projected to the query axis.

    The alignment is packed as UPPERCASE one-letter MSA symbols (or the ``-``
    gap) in query-width rows; row 0 of each pool is the ungapped query. Cells
    are MSA symbols, not ChemicalComponent IDs, and ambiguity symbols
    (``B Z U J O X`` and nucleic ambiguity) are preserved as-is -- there is no
    alphabet field and no normalization here. The MSA holds no
    :class:`PolymerSequence`; ``entity_id`` closes over the primary
    ``MolecularSystem`` supplied beside it, and the Processor -- not this value
    -- resolves the entity and verifies query chemistry and width.
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
    r"""A raw pre-cleanup, pre-crop paired and unpaired row assembly plan.

    Columns align one-to-one with :attr:`msas` (column ``c`` binds ``msas[c]``).
    :attr:`row_pairing` is the complete dense plan: cell ``(r, c)`` with a value
    ``>= 0`` selects that source row in column ``c``'s pool while ``-1`` supplies
    a gap. :attr:`row_is_paired` chooses the PAIRED pool of ``msas[c]`` where
    ``True`` and the UNPAIRED pool where ``False`` (also recording paired feature
    semantics). ``target_instance_ids is None`` broadcasts each column to every
    instance of its entity; a tuple binds columns to exact target instances.
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
    r"""Return each MSA column with its exact instance target, or ``None`` for broadcast."""

    columns = (msa,) if isinstance(msa, MSA) else msa.msas
    targets = (
        (None,) * len(columns) if isinstance(msa, MSA) or msa.target_instance_ids is None else msa.target_instance_ids
    )
    return tuple(zip(columns, targets))


def resolve_msa_bindings(
    msa: MSA | MSACollection,
    system: MolecularSystem,
) -> tuple[tuple[MSA, str | None, PolymerSequence], ...]:
    r"""Close MSA entity/instance references over one molecular system."""

    resolved: list[tuple[MSA, str | None, PolymerSequence]] = []
    for column, target_instance_id in msa_bindings(msa):
        entity = system.entities.get(column.entity_id)
        if not isinstance(entity, PolymerSequence):
            raise ValueError(f"MSA entity {column.entity_id!r} is not a polymer entity in the input system")
        if target_instance_id is not None:
            target_entity_id = system.instances.get(target_instance_id)
            if target_entity_id is None:
                raise ValueError(f"MSA targets unknown input instance {target_instance_id!r}")
            if target_entity_id != column.entity_id:
                raise ValueError(
                    f"MSA target instance {target_instance_id!r} belongs to entity "
                    f"{target_entity_id!r}, not {column.entity_id!r}"
                )
        width = len(column.unpaired_sequences[0])
        if width != len(entity):
            raise ValueError(
                f"MSA for entity {column.entity_id!r} spans {width} columns but the entity has {len(entity)} components"
            )
        resolved.append((column, target_instance_id, entity))
    return tuple(resolved)


def msa_binding_map(msa: MSA | MSACollection) -> dict[tuple[str, str | None], MSA]:
    r"""Index MSA columns by ``(entity_id, exact_instance_or_none)``."""

    return {(column.entity_id, target_instance_id): column for column, target_instance_id in msa_bindings(msa)}


def select_msa_binding(
    bindings: dict[tuple[str, str | None], MSA],
    entity_id: str,
    instance_id: str,
) -> tuple[tuple[str, str | None], MSA] | None:
    r"""Select an exact-instance MSA column, falling back to entity broadcast."""

    exact = (entity_id, instance_id)
    if exact in bindings:
        return exact, bindings[exact]
    broadcast = (entity_id, None)
    if broadcast in bindings:
        return broadcast, bindings[broadcast]
    return None


@dataclass(eq=False)
class TemplateAlignment:
    r"""One target-to-template polymer alignment within a Template.

    ``target_entity_id`` names a target entity in the primary
    ``MolecularSystem``; this value holds no :class:`PolymerSequence`, so the
    target-side upper bound and query chemistry are the Processor's job.
    ``residue_pairs`` are increasing zero-based
    ``(target_component_index, template_component_index)`` pairs.
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
    r"""Aligned template evidence sharing one Structure coordinate frame.

    One Template is one coordinate frame; grouping multiple chain alignments
    under it preserves an interface frame without duplicating the Structure.
    The template side owns the Structure, so its component-index upper bound is
    checked here.
    """

    structure: Structure
    alignments: tuple[TemplateAlignment, ...]

    def __post_init__(self) -> None:
        if not self.alignments:
            raise ValueError("alignments must not be empty")
        for alignment in self.alignments:
            entity_id = self.structure.system.instances[alignment.template_instance_id]
            template_entity = self.structure.system.entities[entity_id]
            if not isinstance(template_entity, PolymerSequence):
                raise ValueError("template_instance_id must reference a polymer instance")
            if alignment.residue_pairs[-1][1] >= len(template_entity):
                raise ValueError("residue pair template index is outside the template instance")
