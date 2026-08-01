"""Polymer realization from explicit sequence and observed component evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ...io.polymer import POLYMER_TYPES, resolve_monomers
from ...io.records import AtomColumns
from ..components import get_builtin_component_polymer_type, get_component
from ..molecular import ChemicalComponent, PolymerAttachment, PolymerSequence, PolymerType
from .plan import ResidueKey


def _declared_polymer_type(
    declared_monomers: tuple[tuple[int, tuple[str, ...]], ...],
    entity_poly_type: str | None,
) -> PolymerType | None:
    if entity_poly_type is not None:
        return _ENTITY_POLY_TYPE_TO_POLYMER.get(entity_poly_type)
    inferred = [
        get_builtin_component_polymer_type(monomer) for _, candidates in declared_monomers for monomer in candidates
    ]
    if any(polymer_type is None for polymer_type in inferred):
        return None
    resolved = {polymer_type for polymer_type in inferred if polymer_type is not None}
    if resolved == {"dna", "rna"}:
        return "dna_rna_hybrid"
    if len(resolved) == 1:
        return next(iter(resolved))
    return None


def _resolve_declared_component(
    component_id: str,
    component_table: Mapping[str, ChemicalComponent],
) -> ChemicalComponent:
    component = component_table.get(component_id)
    if component is None:
        try:
            component = get_component(component_id)
        except KeyError:
            component = ChemicalComponent(component_id=component_id, atoms=(), bonds=None)
    return component


def _resolve_pdb_polymer_component(
    component_id: str,
    standard_id: str | None,
    component_table: Mapping[str, ChemicalComponent],
    *,
    polymer_type: PolymerType,
) -> ChemicalComponent:
    r"""Resolve one PDB polymer residue in its chain chemistry."""

    # PDB uses the same N residue name for unknown RNA and unknown DNA. OFoldX
    # keeps those as distinct canonical components once the chain chemistry is
    # known.
    if component_id == "N" and polymer_type == "dna":
        component_id = "DN"

    component = _resolve_declared_component(
        component_id,
        component_table,
    )
    if standard_id is None:
        return component
    standard = _resolve_declared_component(
        standard_id,
        component_table,
    )
    atom_ids = {atom.atom_id for atom in component.atoms}

    def inherit(
        attachment: PolymerAttachment | None,
        standard_attachment: PolymerAttachment | None,
    ) -> PolymerAttachment | None:
        if attachment is not None or standard_attachment is None:
            return attachment
        if standard_attachment.atom_id not in atom_ids:
            raise ValueError(
                f"PDB MODRES component {component_id!r} does not contain parent {standard_id!r} "
                f"attachment atom {standard_attachment.atom_id!r}"
            )
        leaving_atom_ids = standard_attachment.leaving_atom_ids & atom_ids
        leaving_atom_ids -= component.non_leaving_atom_ids
        return PolymerAttachment(
            standard_attachment.atom_id,
            leaving_atom_ids,
        )

    return replace(
        component,
        previous_attachment=inherit(component.previous_attachment, standard.previous_attachment),
        next_attachment=inherit(component.next_attachment, standard.next_attachment),
    )


def _declared_polymer(
    entity_id: str,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    entity_poly_types: Mapping[str, str],
    component_table: Mapping[str, ChemicalComponent],
    polymer_residues: Sequence[ResidueKey],
    residue_rows: Mapping[ResidueKey, NDArray[np.int64]],
    rows: AtomColumns,
) -> tuple[PolymerSequence, tuple[tuple[int, ResidueKey], ...]] | None:
    if polymer_monomers is None:
        return None
    declared_monomers = polymer_monomers.get(entity_id)
    if not declared_monomers:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has no _entity_poly_seq declaration")

    observed: dict[int, str] = {}
    indexed: list[tuple[int, ResidueKey]] = []
    label_seq_ids = rows.label_seq_id
    component_ids = rows.component_id
    for key in polymer_residues:
        row_position = residue_rows[key][0]
        label_seq_id = label_seq_ids[row_position]
        if label_seq_id is None:
            raise ValueError("observed mmCIF polymer residue lacks label_seq_id")
        index = int(label_seq_id) - 1
        if index < 0 or index >= len(declared_monomers):
            raise ValueError(f"observed mmCIF label_seq_id {label_seq_id} is outside the declared polymer sequence")
        if index in observed:
            raise ValueError(f"multiple observed mmCIF residues map to label_seq_id {label_seq_id}")
        component_id = str(component_ids[row_position])
        candidates = declared_monomers[index][1]
        if component_id not in candidates:
            raise ValueError(
                f"observed mmCIF component {component_id!r} is not declared at "
                f"entity {entity_id!r} position {label_seq_id}"
            )
        observed[index] = component_id
        indexed.append((index, key))

    declared_type = entity_poly_types.get(entity_id)
    polymer_type = _declared_polymer_type(declared_monomers, declared_type)
    if polymer_type is None:
        if declared_type is not None:
            raise ValueError(f"mmCIF polymer entity {entity_id!r} has unsupported _entity_poly.type {declared_type!r}")
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has ambiguous polymer type declarations")
    components = [
        _resolve_declared_component(identifier, component_table)
        for identifier in resolve_monomers(entity_id, declared_monomers, observed)
    ]
    links = _polymer_links(components, _observed_atom_ids(indexed, residue_rows, rows.atom_name))
    return PolymerSequence(polymer_type, tuple(components), links), tuple(indexed)


def _source_author_numbering(
    author_numbering: Mapping[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ],
    asym_id: str,
    component_index: int,
    component_id: str,
) -> tuple[str | None, str | None] | None:
    r"""Resolve candidate-specific polymer numbering before its position fallback."""

    return author_numbering.get(
        (asym_id, component_index + 1, component_id),
        author_numbering.get((asym_id, component_index + 1)),
    )


def _polymer_links(
    components: Sequence[ChemicalComponent],
    observed_atom_ids: Mapping[int, frozenset[str]],
    author_positions: Sequence[tuple[int, str]] | None = None,
) -> tuple[bool, ...]:
    r"""
    Decide which polymer junctions the source realizes as bonds.

    A realized link consumes the leaving atoms on both of its sides, so observing one of
    them -- a mid-chain ``OXT`` on the left, an ``H2`` or ``OP3`` on the right -- is the
    deposited model stating that the residue carries a free terminus and that the junction
    beside it is not bonded.

    Author numbering that skips is the second signal. Absent numbers are not a sequence
    declaration, so the reader does not invent the residues they would name; but it does
    know that the two residues either side of the skip are not the ones the backbone would
    bond, so it asserts no bond across it. An insertion code suspends that reading: ``10A``
    follows ``10`` in the author's own ordering while repeating its number, so the
    difference between two numbers no longer says what lies between them.

    Args:
        components: Component occurrences in sequence order.
        observed_atom_ids: Atom IDs actually present for each component index.
        author_positions: Author residue number and insertion code of each component, when
            the source numbers every occurrence and declares no sequence of its own.

    Returns:
        One flag per junction, or an empty tuple when every junction is bonded.
    """

    links: list[bool] = []
    for junction, left in enumerate(components[:-1]):
        right = components[junction + 1]
        linked = not (
            _leaving_atom_ids(left.next_attachment) & observed_atom_ids.get(junction, frozenset())
            or _leaving_atom_ids(right.previous_attachment) & observed_atom_ids.get(junction + 1, frozenset())
        )
        if linked and author_positions is not None:
            (number, code), (next_number, next_code) = author_positions[junction], author_positions[junction + 1]
            linked = bool(code or next_code) or next_number - number == 1
        links.append(linked)
    return () if all(links) else tuple(links)


def _leaving_atom_ids(attachment: PolymerAttachment | None) -> frozenset[str]:
    r"""Atoms a realized link at one endpoint removes, or none where the endpoint is absent."""

    return frozenset() if attachment is None else attachment.leaving_atom_ids


def _observed_atom_ids(
    indexed: Sequence[tuple[int, ResidueKey]],
    residue_rows: Mapping[ResidueKey, NDArray[np.int64]],
    atom_names: np.ndarray,
) -> dict[int, frozenset[str]]:
    r"""Collect the atom IDs each component occurrence actually contributes."""

    observed: dict[int, frozenset[str]] = {}
    for component_index, residue_key in indexed:
        observed[component_index] = frozenset(str(atom_names[position]) for position in residue_rows[residue_key])
    return observed


def _pdb_polymer_segments(
    polymer_residues: Sequence[ResidueKey],
    residue_rows: Mapping[ResidueKey, NDArray[np.int64]],
    serials: np.ndarray,
    segment_by_serial: Mapping[str, int],
) -> list[list[ResidueKey]]:
    segments: list[list[ResidueKey]] = []
    current: list[ResidueKey] = []
    previous_segment: int | None = None
    for key in polymer_residues:
        serial = serials[residue_rows[key][0]]
        segment = segment_by_serial.get(str(serial)) if serial is not None else None
        if current and previous_segment is not None and segment is not None and segment != previous_segment:
            segments.append(current)
            current = []
        current.append(key)
        previous_segment = segment
    if current:
        segments.append(current)
    return segments


def _pdb_polymer_with_declared_missing_residues(
    polymer_type: PolymerType,
    chain_id: str,
    segment_residues: Sequence[ResidueKey],
    residue_rows: Mapping[ResidueKey, NDArray[np.int64]],
    rows: AtomColumns,
    component_table: Mapping[str, ChemicalComponent],
    declarations: tuple[tuple[str, str, int, str], ...],
    modified_residues: Mapping[tuple[str, int, str, str], str],
    *,
    declared_sequence: tuple[str, ...] | None = None,
    include_terminal: bool,
) -> tuple[
    PolymerSequence,
    tuple[tuple[int, ResidueKey], ...],
    tuple[tuple[int, str, str], ...],
]:
    observed: list[tuple[tuple[int, str], ResidueKey, str]] = []
    auth_residue_ids = rows.auth_residue_id
    residue_ids = rows.residue_id
    insertion_codes = rows.insertion_code
    component_ids = rows.component_id
    for residue_key in segment_residues:
        row_position = residue_rows[residue_key][0]
        number = auth_residue_ids[row_position]
        if number is None:
            number = residue_ids[row_position]
        if number is None:
            raise ValueError("PDB polymer residue lacks author numbering")
        observed.append(
            ((int(number), str(insertion_codes[row_position] or "")), residue_key, str(component_ids[row_position]))
        )

    observed_positions = [position for position, _, _ in observed]
    declared = [
        ((number, insertion_code), component_id)
        for declared_chain, component_id, number, insertion_code in declarations
        if declared_chain == chain_id
    ]
    if declared_sequence is not None:
        observed_position_set = set(observed_positions)
        declared_entries: list[tuple[tuple[int, str], ResidueKey | None, str]] = [
            (position, residue_key, component_id) for position, residue_key, component_id in observed
        ]
        declared_entries.extend(
            (position, None, component_id)
            for position, component_id in declared
            if position not in observed_position_set
        )
        declared_entries.sort(key=lambda entry: entry[0])
        positions = [position for position, _, _ in declared_entries]
        if len(set(positions)) != len(positions):
            raise ValueError(f"PDB chain {chain_id!r} repeats an author residue key")
        if tuple(component_id for _, _, component_id in declared_entries) != declared_sequence:
            raise ValueError(f"PDB chain {chain_id!r} ATOM/HETATM and REMARK 465 occurrences do not match SEQRES")

        declared_components = tuple(
            _resolve_pdb_polymer_component(
                component_id,
                modified_residues.get((chain_id, position[0], position[1], component_id)),
                component_table,
                polymer_type=polymer_type,
            )
            for position, _, component_id in declared_entries
        )
        declared_indexed = tuple(
            (component_index, residue_key)
            for component_index, (_, residue_key, _) in enumerate(declared_entries)
            if residue_key is not None
        )
        declared_unmodeled = tuple(
            (component_index, str(position[0]), position[1])
            for component_index, (position, residue_key, _) in enumerate(declared_entries)
            if residue_key is None
        )
        links = _polymer_links(
            declared_components,
            _observed_atom_ids(declared_indexed, residue_rows, rows.atom_name),
        )
        return PolymerSequence(polymer_type, declared_components, links), declared_indexed, declared_unmodeled

    if not include_terminal:
        first, last = observed_positions[0], observed_positions[-1]
        declared = [(position, component_id) for position, component_id in declared if first < position < last]
    observed_position_set = set(observed_positions)
    declared = [
        (position, component_id) for position, component_id in declared if position not in observed_position_set
    ]
    if not declared:
        observed_components = tuple(
            _resolve_pdb_polymer_component(
                component_id,
                modified_residues.get((chain_id, position[0], position[1], component_id)),
                component_table,
                polymer_type=polymer_type,
            )
            for position, _, component_id in observed
        )
        observed_indexed = tuple((index, residue_key) for index, (_, residue_key, _) in enumerate(observed))
        return (
            PolymerSequence(
                polymer_type,
                observed_components,
                _polymer_links(
                    observed_components,
                    _observed_atom_ids(observed_indexed, residue_rows, rows.atom_name),
                    # Nothing declares this segment's sequence, so consecutive author
                    # numbering is the only evidence that two residues really adjoin.
                    [position for position, _, _ in observed],
                ),
            ),
            observed_indexed,
            (),
        )
    if any(left >= right for left, right in zip(observed_positions, observed_positions[1:])):
        raise ValueError("cannot place declared missing PDB residues in non-monotonic author numbering")

    entries: list[tuple[tuple[int, str], ResidueKey | None, str]] = [
        (position, residue_key, component_id) for position, residue_key, component_id in observed
    ]
    entries.extend((position, None, component_id) for position, component_id in declared)
    entries.sort(key=lambda entry: entry[0])
    components: list[ChemicalComponent] = []
    indexed: list[tuple[int, ResidueKey]] = []
    unmodeled: list[tuple[int, str, str]] = []
    for component_index, (position, entry_residue_key, component_id) in enumerate(entries):
        components.append(
            _resolve_pdb_polymer_component(
                component_id,
                modified_residues.get((chain_id, position[0], position[1], component_id)),
                component_table,
                polymer_type=polymer_type,
            )
        )
        if entry_residue_key is None:
            unmodeled.append((component_index, str(position[0]), position[1]))
        else:
            indexed.append((component_index, entry_residue_key))
    links = _polymer_links(components, _observed_atom_ids(indexed, residue_rows, rows.atom_name))
    return PolymerSequence(polymer_type, tuple(components), links), tuple(indexed), tuple(unmodeled)


def _pdb_component_polymer_type(component: ChemicalComponent) -> PolymerType | None:
    polymer_type = get_builtin_component_polymer_type(component.component_id)
    if polymer_type is not None:
        return polymer_type
    if (
        component.previous_attachment is not None
        and component.next_attachment is not None
        and component.previous_attachment.atom_id == "N"
        and component.next_attachment.atom_id == "C"
    ):
        return "protein"
    elements = {atom.atom_id: atom.element for atom in component.atoms}
    if all(elements.get(atom_id) == element for atom_id, element in (("N", "N"), ("CA", "C"), ("C", "C"))):
        return "protein"
    return None


_ENTITY_POLY_TYPE_TO_POLYMER = POLYMER_TYPES
