r"""PDB and mmCIF I/O for the canonical tensor molecular graph."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..data.components import get_builtin_component_polymer_type, get_component
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import (
    HYDROGEN_ELEMENTS,
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    is_water_component,
)
from ..data.structure import Structure
from .records import atom_row_residue_key, unique_identifier

_FORMATS: dict[str, Literal["pdb", "mmcif"]] = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}
_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})


def read_structure(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = True,
    include_hydrogens: bool = True,
    components: Mapping[str, ChemicalComponent] | None = None,
) -> Structure:
    r"""
    Read one model and alternate-location realization from PDB or mmCIF.

    Args:
        path: Input PDB, gzip-compressed PDB, or mmCIF path.
        model: One-based model number.
        assembly_id: Biological assembly identifier, or `None` for the stored
            asymmetric unit.
        altloc: Residue-level alternate-location choice.
        include_water: Whether to retain water entities.
        include_hydrogens: Whether to retain hydrogen atoms in the molecular graph.
        components: Optional component definitions overriding built-in chemistry.

    Returns:
        A CPU tensor Structure whose atom axis is the complete retained system graph.
    """

    source, format_name = _structure_path(path, allow_compressed=True)
    if model <= 0:
        raise ValueError("model must be a positive one-based integer")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")

    cif_file: object | None = None
    source_components: dict[str, ChemicalComponent] = {}
    declared_component_ids: set[str] = set()
    pdb_lines: tuple[str, ...] = ()
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...] = ()
    if format_name == "mmcif":
        from .mmcif import parse_mmcif

        source_rows, cif_file = parse_mmcif(source, model=model, assembly_id=assembly_id)
        from .mmcif import read_mmcif_components

        source_components, declared_component_ids = read_mmcif_components(cif_file)
    else:
        from .pdb import parse_pdb

        source_rows, pdb_lines, pdb_missing_residues = parse_pdb(
            source,
            model=model,
            assembly_id=assembly_id,
        )
    if not len(source_rows):
        raise ValueError("structure source contains no atoms for the requested model/assembly")

    component_table: dict[str, ChemicalComponent] = {}
    for component_id in {*source_rows.component_id, *declared_component_ids}:
        with contextlib.suppress(KeyError):
            component_table[str(component_id)] = get_component(str(component_id))
    for component_id, component in source_components.items():
        component_table.setdefault(component_id, component)
    if components is not None:
        for component_id, component in components.items():
            if component.component_id != component_id:
                raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
            component_table[component_id] = component

    rows = source_rows
    if not include_water:
        rows = _exclude_water_rows(format_name, rows, cif_file, component_table)
    _add_observed_components(rows, component_table)
    if not include_hydrogens:
        rows = rows[
            np.isin(
                rows.element,
                tuple(HYDROGEN_ELEMENTS),
                invert=True,
            )
        ].view(np.recarray)
        component_table = {
            component_id: component.without_hydrogens() for component_id, component in component_table.items()
        }
    rows = _select_altloc(rows, altloc)
    return _build_structure(
        format_name,
        rows,
        source_rows,
        component_table,
        include_water=include_water,
        include_hydrogens=include_hydrogens,
        include_declared_instances=assembly_id is None,
        cif_file=cif_file,
        pdb_lines=pdb_lines,
        pdb_missing_residues=pdb_missing_residues,
    )


def write_structure(structure: Structure, path: str | os.PathLike[str]) -> None:
    r"""Write a Structure through the standard mmCIF or legacy PDB projection."""

    if structure.system.system_entity_ptr.numel() != 2 or structure.system.system_instance_ptr.numel() != 2:
        raise ValueError("one structure file cannot represent multiple molecular systems")
    omitted = ~structure.coordinate_mask
    if bool((omitted & (structure.occupancy_mask | structure.b_factor_mask)).any()):
        raise ValueError(
            "a structure file cannot preserve occupancy or B-factor evidence " "for atoms without coordinates"
        )
    target, format_name = _structure_path(path)
    if format_name == "mmcif":
        from .mmcif import write_mmcif

        write_mmcif(structure, target)
    else:
        from .pdb import write_pdb

        write_pdb(structure, target)


def _structure_path(
    path: str | os.PathLike[str],
    *,
    allow_compressed: bool = False,
) -> tuple[Path, Literal["pdb", "mmcif"]]:
    resolved = Path(os.path.expanduser(os.fspath(path)))
    suffix = resolved.suffix.lower()
    if allow_compressed and suffix == ".gz":
        suffix = resolved.with_suffix("").suffix.lower()
        if suffix != ".pdb":
            raise ValueError("only PDB structure input may be gzip-compressed")
    try:
        return resolved, _FORMATS[suffix]
    except KeyError:
        expected = ".pdb, .pdb.gz, .cif, or .mmcif" if allow_compressed else ".pdb, .cif, or .mmcif"
        raise ValueError(f"structure path suffix must be {expected}") from None


def _exclude_water_rows(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    cif_file: object | None,
    component_table: Mapping[str, ChemicalComponent],
) -> np.recarray:
    entity_types: Mapping[str, str] = {}
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import _read_mmcif_entity_declarations

        entity_types, _, _, _, _, _ = _read_mmcif_entity_declarations(cif_file)

    fallback_rows = rows[np.asarray([row.entity_id is None or row.entity_id not in entity_types for row in rows])].view(
        np.recarray
    )
    fallback_components = dict(component_table)
    _add_observed_components(fallback_rows, fallback_components)
    groups: dict[tuple[object, ...], list[np.record]] = {}
    for row in fallback_rows:
        key = (
            ("entity", row.entity_id)
            if format_name == "mmcif" and row.entity_id is not None
            else ("residue", *atom_row_residue_key(row))
        )
        groups.setdefault(key, []).append(row)
    fallback_water_rows = {
        int(row.row_index)
        for group_rows in groups.values()
        if all(
            is_water_component(fallback_components[str(component_id)])
            for component_id in {row.component_id for row in group_rows}
        )
        for row in group_rows
    }
    water_entities = {entity_id for entity_id, entity_type in entity_types.items() if entity_type == "water"}
    keep = np.asarray(
        [row.entity_id not in water_entities and int(row.row_index) not in fallback_water_rows for row in rows],
        dtype=np.bool_,
    )
    return rows[keep].view(np.recarray)


def _add_observed_components(
    rows: np.recarray,
    component_table: dict[str, ChemicalComponent],
) -> None:
    atoms_by_component: dict[str, dict[str, ChemicalAtom]] = {}
    for row in rows:
        component_id = str(row.component_id)
        if component_id in component_table:
            continue
        element = str(row.element)
        if element not in ATOMIC_NUMBER_BY_ELEMENT:
            raise ValueError(
                f"source atom {row.atom_name!r} in component {component_id!r} " f"has invalid element {element!r}"
            )
        atom = ChemicalAtom(
            atom_id=str(row.atom_name),
            element=element,
            isotope=row.isotope,
            formal_charge=row.charge,
        )
        component_atoms = atoms_by_component.setdefault(component_id, {})
        previous = component_atoms.get(str(row.atom_name))
        if previous is None:
            component_atoms[str(row.atom_name)] = atom
        elif previous != atom:
            raise ValueError(f"component {component_id!r} has conflicting source chemistry for atom {row.atom_name!r}")
    component_table.update(
        {
            component_id: ChemicalComponent(
                component_id=component_id,
                atoms=tuple(atoms.values()),
                bonds=None,
            )
            for component_id, atoms in atoms_by_component.items()
        }
    )


def _altloc_position_key(row: np.record) -> tuple[object, ...]:
    # Component identity is deliberately absent: microheterogeneous components at one
    # structural position compete as alternate realizations.
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.auth_residue_id,
        row.insertion_code,
    )


def _select_altloc(
    rows: np.recarray,
    policy: Literal["highest_occupancy", "first"],
) -> np.recarray:
    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    occupancy_by_label: dict[tuple[tuple[object, ...], str], float] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        residue_key = _altloc_position_key(row)
        label = str(row.altloc_id)
        labels = labels_by_residue.setdefault(residue_key, [])
        if label not in labels:
            labels.append(label)
        key = residue_key, label
        occupancy_by_label[key] = occupancy_by_label.get(key, 0.0) + float(row.occupancy or 0.0)
    selected: dict[tuple[object, ...], str] = {}
    for residue_key, labels in labels_by_residue.items():
        if policy == "first":
            selected[residue_key] = labels[0]
        else:
            selected[residue_key] = max(
                labels,
                key=lambda label: occupancy_by_label[(residue_key, label)],
            )
    keep = np.asarray(
        [
            row.altloc_id is None or selected.get(_altloc_position_key(row), row.altloc_id) == row.altloc_id
            for row in rows
        ],
        dtype=np.bool_,
    )
    return rows[keep].view(np.recarray)


_ENTITY_POLY_TYPE_TO_POLYMER: dict[str, PolymerType] = {
    "polypeptide(l)": "protein",
    "polypeptide(d)": "protein",
    "cyclic-pseudo-peptide": "protein",
    "polyribonucleotide": "rna",
    "polydeoxyribonucleotide": "dna",
    "polydeoxyribonucleotide/polyribonucleotide hybrid": "dna_rna_hybrid",
    "peptide nucleic acid": "pna",
}


def _mmcif_residue_is_polymer(row: np.record, entity_types: Mapping[str, str]) -> bool:
    if row.entity_id is not None:
        entity_type = entity_types.get(str(row.entity_id))
        if entity_type is not None:
            return entity_type == "polymer"
    return get_builtin_component_polymer_type(str(row.component_id)) is not None


def _declared_polymer_type(
    declared_monomers: tuple[tuple[int, tuple[str, ...]], ...],
    entity_poly_type: str | None,
) -> PolymerType | None:
    resolved = {
        get_builtin_component_polymer_type(monomer) for _, candidates in declared_monomers for monomer in candidates
    } - {None}
    if resolved == {"dna", "rna"}:
        return "dna_rna_hybrid"
    if len(resolved) == 1:
        return next(iter(resolved))
    return _ENTITY_POLY_TYPE_TO_POLYMER.get(entity_poly_type) if entity_poly_type is not None else None


def _resolve_declared_component(
    component_id: str,
    component_table: Mapping[str, ChemicalComponent],
    *,
    include_hydrogens: bool,
) -> ChemicalComponent:
    component = component_table.get(component_id)
    if component is None:
        try:
            component = get_component(component_id)
        except KeyError:
            component = ChemicalComponent(component_id=component_id, atoms=(), bonds=None)
    return component if include_hydrogens else component.without_hydrogens()


def _declared_polymer(
    entity_id: str,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    entity_poly_types: Mapping[str, str],
    component_table: Mapping[str, ChemicalComponent],
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    *,
    include_hydrogens: bool,
) -> tuple[PolymerSequence, tuple[tuple[int, tuple[object, ...]], ...]] | None:
    if polymer_monomers is None:
        return None
    declared_monomers = polymer_monomers.get(entity_id)
    if not declared_monomers:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has no _entity_poly_seq declaration")

    observed: dict[int, str] = {}
    indexed: list[tuple[int, tuple[object, ...]]] = []
    for key in polymer_residues:
        row = residue_rows[key][0]
        if row.label_seq_id is None:
            raise ValueError("observed mmCIF polymer residue lacks label_seq_id")
        index = int(row.label_seq_id) - 1
        if index < 0 or index >= len(declared_monomers):
            raise ValueError(
                f"observed mmCIF label_seq_id {row.label_seq_id} " "is outside the declared polymer sequence"
            )
        if index in observed:
            raise ValueError(f"multiple observed mmCIF residues map to label_seq_id {row.label_seq_id}")
        component_id = str(row.component_id)
        candidates = declared_monomers[index][1]
        if component_id not in candidates:
            raise ValueError(
                f"observed mmCIF component {component_id!r} is not declared at "
                f"entity {entity_id!r} position {row.label_seq_id}"
            )
        observed[index] = component_id
        indexed.append((index, key))

    polymer_type = _declared_polymer_type(declared_monomers, entity_poly_types.get(entity_id))
    if polymer_type is None:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has ambiguous polymer type declarations")
    components: list[ChemicalComponent] = []
    for index, (_, candidates) in enumerate(declared_monomers):
        selected_component_id = observed.get(index)
        if selected_component_id is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"unmodeled mmCIF polymer position {index + 1} of entity {entity_id!r} "
                    f"has ambiguous monomers {candidates!r}"
                )
            selected_component_id = candidates[0]
        components.append(
            _resolve_declared_component(
                selected_component_id,
                component_table,
                include_hydrogens=include_hydrogens,
            )
        )
    return PolymerSequence(polymer_type, tuple(components)), tuple(indexed)


def _pdb_polymer_segments(
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    segment_by_serial: Mapping[str, int],
) -> list[list[tuple[object, ...]]]:
    segments: list[list[tuple[object, ...]]] = []
    current: list[tuple[object, ...]] = []
    previous_segment: int | None = None
    for key in polymer_residues:
        row = residue_rows[key][0]
        segment = segment_by_serial.get(str(row.serial)) if row.serial is not None else None
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
    segment_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    component_table: Mapping[str, ChemicalComponent],
    declarations: tuple[tuple[str, str, int, str], ...],
    *,
    include_terminal: bool,
    include_hydrogens: bool,
) -> tuple[
    PolymerSequence,
    tuple[tuple[int, tuple[object, ...]], ...],
    tuple[tuple[int, str, str], ...],
]:
    observed: list[tuple[tuple[int, str], tuple[object, ...], str]] = []
    for residue_key in segment_residues:
        row = residue_rows[residue_key][0]
        number = row.auth_residue_id if row.auth_residue_id is not None else row.residue_id
        if number is None:
            raise ValueError("PDB polymer residue lacks author numbering")
        observed.append(((int(number), str(row.insertion_code or "")), residue_key, str(row.component_id)))

    observed_positions = [position for position, _, _ in observed]
    declared = [
        ((number, insertion_code), component_id)
        for declared_chain, component_id, number, insertion_code in declarations
        if declared_chain == chain_id
    ]
    if not include_terminal:
        first, last = observed_positions[0], observed_positions[-1]
        declared = [(position, component_id) for position, component_id in declared if first < position < last]
    observed_position_set = set(observed_positions)
    declared = [
        (position, component_id) for position, component_id in declared if position not in observed_position_set
    ]
    if not declared:
        return (
            PolymerSequence(polymer_type, tuple(component_table[component_id] for _, _, component_id in observed)),
            tuple((index, residue_key) for index, (_, residue_key, _) in enumerate(observed)),
            (),
        )
    if any(left >= right for left, right in zip(observed_positions, observed_positions[1:])):
        raise ValueError("cannot place declared missing PDB residues in non-monotonic author numbering")

    entries: list[tuple[tuple[int, str], tuple[object, ...] | None, str]] = [
        (position, residue_key, component_id) for position, residue_key, component_id in observed
    ]
    entries.extend((position, None, component_id) for position, component_id in declared)
    entries.sort(key=lambda entry: entry[0])
    components: list[ChemicalComponent] = []
    indexed: list[tuple[int, tuple[object, ...]]] = []
    unmodeled: list[tuple[int, str, str]] = []
    for component_index, (position, entry_residue_key, component_id) in enumerate(entries):
        components.append(
            _resolve_declared_component(
                component_id,
                component_table,
                include_hydrogens=include_hydrogens,
            )
        )
        if entry_residue_key is None:
            unmodeled.append((component_index, str(position[0]), position[1]))
        else:
            indexed.append((component_index, entry_residue_key))
    return PolymerSequence(polymer_type, tuple(components)), tuple(indexed), tuple(unmodeled)


def _build_structure(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    source_rows: np.recarray,
    component_table: dict[str, ChemicalComponent],
    *,
    include_water: bool,
    include_hydrogens: bool,
    include_declared_instances: bool,
    cif_file: object | None,
    pdb_lines: tuple[str, ...],
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...],
) -> Structure:
    if not len(rows):
        raise ValueError("alternate-location selection contains no atoms")
    residue_rows: dict[tuple[object, ...], list[np.record]] = {}
    chain_residue_order: dict[tuple[str | None, str], list[tuple[object, ...]]] = {}
    for row in rows:
        residue_key = atom_row_residue_key(row)
        bucket = residue_rows.get(residue_key)
        if bucket is None:
            residue_rows[residue_key] = [row]
            chain_key = (row.symmetry_id, str(row.chain_id))
            chain_residue_order.setdefault(chain_key, []).append(residue_key)
        else:
            bucket.append(row)

    entity_types: dict[str, str] = {}
    polymer_monomers: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None = None
    entity_poly_types: dict[str, str] = {}
    author_numbering: dict[tuple[str, int], tuple[str | None, str | None]] = {}
    asym_entities: dict[str, str] = {}
    nonpoly_components: dict[str, str] = {}
    segment_by_serial: dict[str, int] = {}
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import (
            _read_mmcif_entity_declarations,
            read_mmcif_branch_entities,
        )

        branch_entities, branch_component_indices = read_mmcif_branch_entities(
            cif_file,
            component_table,
        )
        (
            entity_types,
            polymer_monomers,
            entity_poly_types,
            author_numbering,
            asym_entities,
            nonpoly_components,
        ) = _read_mmcif_entity_declarations(cif_file)
        for entity_id in branch_entities:
            entity_type = entity_types.get(entity_id)
            if entity_type not in {None, "branched"}:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} is declared {entity_type!r} "
                    "but also has a branched-entity definition"
                )
            entity_types[entity_id] = "branched"
        missing_branch_definition = {
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == "branched" and entity_id not in branch_entities
        }
        if missing_branch_definition:
            raise ValueError(
                "mmCIF branched entities require branch declarations: " f"{sorted(missing_branch_definition)!r}"
            )
    else:
        from .pdb import pdb_segment_by_serial

        branch_entities, branch_component_indices = {}, {}
        segment_by_serial = pdb_segment_by_serial(pdb_lines)

    groups: list[
        tuple[
            int,
            str,
            str,
            PolymerSequence | Molecule,
            tuple[tuple[int, tuple[object, ...]], ...],
        ]
    ] = []
    used_instance_ids: set[str] = set()
    entities: dict[str, PolymerSequence | Molecule] = {}
    entity: PolymerSequence | Molecule
    unmodeled_numbers: dict[ComponentReference, tuple[str, str]] = {}
    unmodeled_source_sites: dict[tuple[object, ...], AtomSite] = {}
    ambiguous_unmodeled_sites: set[tuple[object, ...]] = set()

    def add_unmodeled_site(key: tuple[object, ...], site: AtomSite) -> None:
        if key in ambiguous_unmodeled_sites:
            return
        previous = unmodeled_source_sites.get(key)
        if previous is not None and previous != site:
            unmodeled_source_sites.pop(key)
            ambiguous_unmodeled_sites.add(key)
        else:
            unmodeled_source_sites[key] = site

    for (symmetry_id, chain_id), residue_order in chain_residue_order.items():
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        if format_name == "mmcif":
            polymer_residues = [
                key for key in residue_order if _mmcif_residue_is_polymer(residue_rows[key][0], entity_types)
            ]
            polymer_segments = [polymer_residues] if polymer_residues else []
        else:
            polymer_residues = [
                key
                for key in residue_order
                if any(row.group == "ATOM" for row in residue_rows[key])
                or _pdb_component_polymer_type(component_table[str(residue_rows[key][0].component_id)]) is not None
            ]
            polymer_segments = _pdb_polymer_segments(polymer_residues, residue_rows, segment_by_serial)

        segment_declarations = [pdb_missing_residues for _ in polymer_segments]
        if format_name == "pdb" and len(polymer_segments) > 1:
            assigned: list[list[tuple[str, str, int, str]]] = [[] for _ in polymer_segments]
            segment_positions: list[list[tuple[int, str]]] = []
            for segment_residues in polymer_segments:
                positions: list[tuple[int, str]] = []
                for residue_key in segment_residues:
                    row = residue_rows[residue_key][0]
                    number = row.auth_residue_id if row.auth_residue_id is not None else row.residue_id
                    if number is None:
                        raise ValueError("PDB polymer residue lacks author numbering")
                    positions.append((int(number), str(row.insertion_code or "")))
                segment_positions.append(positions)
            for declaration in pdb_missing_residues:
                declared_chain, _, number, declared_insertion_code = declaration
                if declared_chain != chain_id:
                    continue
                position = (number, declared_insertion_code)
                candidates = [
                    index
                    for index, positions in enumerate(segment_positions)
                    if position in positions or positions[0] < position < positions[-1]
                ]
                if len(candidates) == 1:
                    assigned[candidates[0]].append(declaration)
            segment_declarations = [tuple(values) for values in assigned]

        for segment_ordinal, segment_residues in enumerate(polymer_segments):
            first = residue_rows[segment_residues[0]][0]
            source_entity_id = None if first.entity_id is None else str(first.entity_id)
            declared = None
            pdb_unmodeled: tuple[tuple[int, str, str], ...] = ()
            if format_name == "mmcif" and source_entity_id is not None:
                declared = _declared_polymer(
                    source_entity_id,
                    polymer_monomers,
                    entity_poly_types,
                    component_table,
                    segment_residues,
                    residue_rows,
                    include_hydrogens=include_hydrogens,
                )
            if declared is not None:
                declared_entity, indexed = declared
                assert source_entity_id is not None
                entity_id = source_entity_id
                entities.setdefault(entity_id, declared_entity)
                entity = declared_entity
            else:
                classified_values = {
                    (
                        get_builtin_component_polymer_type(str(residue_rows[key][0].component_id))
                        if format_name == "mmcif"
                        else _pdb_component_polymer_type(component_table[str(residue_rows[key][0].component_id)])
                    )
                    for key in segment_residues
                }
                classified = {polymer_type for polymer_type in classified_values if polymer_type is not None}
                if classified == {"dna", "rna"}:
                    chain_polymer_type: PolymerType = "dna_rna_hybrid"
                elif len(classified) == 1:
                    chain_polymer_type = next(iter(classified))
                elif not classified and format_name == "mmcif" and source_entity_id is not None:
                    mapped = _ENTITY_POLY_TYPE_TO_POLYMER.get(entity_poly_types.get(source_entity_id, ""))
                    if mapped is None:
                        raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
                    chain_polymer_type = mapped
                else:
                    raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
                if format_name == "mmcif" and source_entity_id is not None:
                    entity_id = source_entity_id
                elif segment_ordinal == 0:
                    entity_id = f"polymer_{chain_id}"
                else:
                    entity_id = f"polymer_{chain_id}#{segment_ordinal}"
                if format_name == "pdb":
                    resolved, indexed, pdb_unmodeled = _pdb_polymer_with_declared_missing_residues(
                        chain_polymer_type,
                        chain_id,
                        segment_residues,
                        residue_rows,
                        component_table,
                        segment_declarations[segment_ordinal],
                        include_terminal=len(polymer_segments) == 1,
                        include_hydrogens=include_hydrogens,
                    )
                else:
                    resolved = PolymerSequence(
                        chain_polymer_type,
                        tuple(component_table[str(residue_rows[key][0].component_id)] for key in segment_residues),
                    )
                    indexed = tuple(enumerate(segment_residues))
                entity = _reuse_entity(entities, entity_id, resolved)

            instance_id = unique_identifier(base_instance_id, used_instance_ids)
            if declared is not None:
                observed_indices = {index for index, _ in indexed}
                for component_index, component in enumerate(declared_entity.components):
                    if component_index in observed_indices:
                        continue
                    author = author_numbering.get((chain_id, component_index + 1))
                    missing_insertion_code = None if author is None else author[1]
                    for atom in component.atoms:
                        add_unmodeled_site(
                            (
                                symmetry_id,
                                chain_id,
                                component_index + 1,
                                missing_insertion_code,
                                component.component_id,
                                atom.atom_id,
                            ),
                            AtomSite(instance_id, component_index, atom.atom_id),
                        )
                    if author is not None and author[0] is not None:
                        unmodeled_numbers[(instance_id, component_index)] = (
                            str(author[0]),
                            author[1] or "",
                        )
            for component_index, residue_id, insertion_code in pdb_unmodeled:
                unmodeled_numbers[(instance_id, component_index)] = (residue_id, insertion_code)
            groups.append(
                (
                    min(int(row.row_index) for key in segment_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    indexed,
                )
            )

        polymer_residue_set = set(polymer_residues)
        nonpolymer_residues = [key for key in residue_order if key not in polymer_residue_set]
        molecule_groups: list[tuple[tuple[object, ...], ...]] = []
        if format_name == "mmcif":
            grouped_by_entity: dict[str, list[tuple[object, ...]]] = {}
            for residue_key in nonpolymer_residues:
                source_entity_id = residue_rows[residue_key][0].entity_id
                if source_entity_id in branch_entities:
                    grouped_by_entity.setdefault(str(source_entity_id), []).append(residue_key)
                else:
                    molecule_groups.append((residue_key,))
            molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
            molecule_groups.sort(
                key=lambda group: min(int(row.row_index) for key in group for row in residue_rows[key])
            )
        else:
            molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)

        for group_residues in molecule_groups:
            first = residue_rows[group_residues[0]][0]
            first_component = component_table[str(first.component_id)]
            residue_label = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
            residue_identity = f"{residue_label}{first.insertion_code or ''}"
            source_entity_id = None if first.entity_id is None else str(first.entity_id)
            if source_entity_id in branch_entities:
                entity = branch_entities[source_entity_id]
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues: list[tuple[int, tuple[object, ...]]] = []
                for residue_key in group_residues:
                    row = residue_rows[residue_key][0]
                    branch_component_index: int | None = None
                    for source_monomer in (
                        row.component_id,
                        row.auth_component_id,
                    ):
                        if source_monomer is None:
                            continue
                        for source_asym in (
                            row.auth_chain_id,
                            row.chain_id,
                        ):
                            branch_component_index = branch_component_indices.get(
                                (
                                    row.chain_id,
                                    row.entity_id,
                                    source_monomer,
                                    source_asym,
                                    row.auth_residue_id,
                                )
                            )
                            if branch_component_index is not None:
                                break
                        if branch_component_index is not None:
                            break
                    if branch_component_index is None:
                        raise ValueError(f"cannot map branched-entity residue {residue_key!r} to its declaration")
                    indexed_residues.append((branch_component_index, residue_key))
                if len({index for index, _ in indexed_residues}) != len(indexed_residues):
                    raise ValueError(f"branched entity {source_entity_id!r} maps multiple residues to one component")
                indexed_residues.sort()
            elif format_name == "mmcif" and source_entity_id is not None:
                source_entity_type = entity_types.get(source_entity_id)
                if source_entity_type == "branched":
                    raise ValueError(f"branched entity {source_entity_id!r} has no branch declaration")
                entity = Molecule(
                    (first_component,),
                    molecule_type=("water" if source_entity_type == "water" else "non-polymer"),
                )
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues = [(0, group_residues[0])]
            else:
                entity = Molecule(
                    (first_component,),
                    molecule_type=("water" if is_water_component(first_component) else "non-polymer"),
                )
                entity_id = f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            entity = _reuse_entity(entities, entity_id, entity)
            instance_id = unique_identifier(instance_seed, used_instance_ids)
            groups.append(
                (
                    min(int(row.row_index) for key in group_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(indexed_residues),
                )
            )

    observed_asym_ids = {chain_id for _, chain_id in chain_residue_order}
    declared_instances = asym_entities.items() if include_declared_instances else ()
    for declaration_index, (asym_id, source_entity_id) in enumerate(declared_instances):
        if asym_id in observed_asym_ids:
            continue
        source_entity_type = entity_types.get(source_entity_id)
        if not include_water and source_entity_type == "water":
            continue
        if source_entity_type == "polymer":
            declared = _declared_polymer(
                source_entity_id,
                polymer_monomers,
                entity_poly_types,
                component_table,
                [],
                residue_rows,
                include_hydrogens=include_hydrogens,
            )
            if declared is None:
                raise ValueError(f"unobserved mmCIF polymer entity {source_entity_id!r} " "has no sequence declaration")
            resolved, _ = declared
            entity_id = source_entity_id
            entities.setdefault(entity_id, resolved)
            entity = resolved
        elif source_entity_id in branch_entities:
            entity_id = source_entity_id
            entity = _reuse_entity(entities, entity_id, branch_entities[source_entity_id])
        else:
            component_id = nonpoly_components.get(source_entity_id)
            if component_id is None:
                continue
            component = _resolve_declared_component(
                component_id,
                component_table,
                include_hydrogens=include_hydrogens,
            )
            entity_id = source_entity_id
            entity = _reuse_entity(
                entities,
                entity_id,
                Molecule(
                    (component,),
                    molecule_type=("water" if source_entity_type == "water" else "non-polymer"),
                ),
            )

        instance_id = unique_identifier(asym_id, used_instance_ids)
        for component_index, component in enumerate(entity.components):
            author = author_numbering.get((asym_id, component_index + 1))
            source_author_number = None if author is None else author[0]
            insertion = None if author is None else author[1]
            if source_author_number is not None:
                unmodeled_numbers[(instance_id, component_index)] = (
                    source_author_number,
                    insertion or "",
                )
            if source_entity_type == "polymer":
                label_sequence: int | None = component_index + 1
            else:
                label_sequence = None
            for atom in component.atoms:
                site = AtomSite(instance_id, component_index, atom.atom_id)
                add_unmodeled_site(
                    (
                        None,
                        asym_id,
                        label_sequence,
                        insertion,
                        component.component_id,
                        atom.atom_id,
                    ),
                    site,
                )
                for source_key, branch_index in branch_component_indices.items():
                    (
                        label_asym,
                        source_branch_entity,
                        monomer,
                        auth_asym,
                        auth_sequence,
                    ) = source_key
                    if (
                        branch_index != component_index
                        or label_asym != asym_id
                        or source_branch_entity != source_entity_id
                    ):
                        continue
                    add_unmodeled_site(
                        (None, label_asym, None, None, monomer, atom.atom_id),
                        site,
                    )
                    add_unmodeled_site(
                        (
                            None,
                            auth_asym,
                            auth_sequence,
                            None,
                            monomer,
                            atom.atom_id,
                        ),
                        site,
                    )
                add_unmodeled_site(
                    (
                        None,
                        asym_id,
                        source_author_number,
                        insertion,
                        component.component_id,
                        atom.atom_id,
                    ),
                    site,
                )
        groups.append(
            (
                len(source_rows) + declaration_index,
                instance_id,
                entity_id,
                entity,
                (),
            )
        )

    groups.sort(key=lambda item: item[0])
    instances = {instance_id: entity_id for _, instance_id, entity_id, _, _ in groups}
    site_by_row_index: dict[int, AtomSite] = {}
    residue_numbers: dict[ComponentReference, tuple[str, str]] = dict(unmodeled_numbers)
    for _, instance_id, _, entity, indexed in groups:
        for component_index, residue_key in indexed:
            residue = residue_rows[residue_key]
            first = residue[0]
            source_number = first.auth_residue_id
            if format_name == "pdb" and source_number is None:
                source_number = first.residue_id
            source_insertion = str(first.insertion_code or "")
            if format_name == "mmcif":
                author = author_numbering.get((str(first.chain_id), component_index + 1))
                if author is not None:
                    if source_number is None:
                        source_number = author[0]
                    if first.insertion_code is None:
                        source_insertion = author[1] or ""
            if source_number is not None:
                residue_numbers[(instance_id, component_index)] = (
                    str(source_number),
                    source_insertion,
                )
            if format_name == "mmcif":
                component = entity.components[component_index]
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            first.chain_id,
                            first.label_seq_id,
                            first.insertion_code,
                            component.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            first.auth_chain_id,
                            first.auth_residue_id,
                            first.insertion_code,
                            first.auth_component_id or component.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
            for row in residue:
                site_by_row_index[int(row.row_index)] = AtomSite(
                    instance_id,
                    component_index,
                    str(row.atom_name),
                )
        if format_name == "mmcif" and isinstance(entity, Molecule) and indexed:
            first = residue_rows[indexed[0][1]][0]
            source_entity_id = first.entity_id
            observed_components = {component_index for component_index, _ in indexed}
            for source_key, component_index in branch_component_indices.items():
                (
                    source_label_asym_id,
                    source_branch_entity_id,
                    source_monomer_id,
                    source_auth_asym_id,
                    source_auth_number,
                ) = source_key
                if (
                    source_branch_entity_id != source_entity_id
                    or source_label_asym_id != first.chain_id
                    or component_index in observed_components
                ):
                    continue
                author = author_numbering.get((str(source_label_asym_id), component_index + 1))
                insertion = None if author is None else author[1]
                if author is not None and author[0] is not None:
                    residue_numbers[(instance_id, component_index)] = (
                        author[0],
                        insertion or "",
                    )
                component = entity.components[component_index]
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            source_label_asym_id,
                            None,
                            insertion,
                            source_monomer_id,
                            atom.atom_id,
                        ),
                        site,
                    )
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            source_auth_asym_id,
                            source_auth_number,
                            insertion,
                            source_monomer_id,
                            atom.atom_id,
                        ),
                        site,
                    )

    if format_name == "pdb":
        from .pdb import read_pdb_bonds

        source_bonds = read_pdb_bonds(pdb_lines, source_rows, rows, site_by_row_index)
    else:
        assert cif_file is not None
        from .mmcif import read_mmcif_bonds

        source_bonds = read_mmcif_bonds(
            cif_file,
            rows,
            site_by_row_index,
            unmodeled_source_sites,
            ambiguous_unmodeled_sites,
        )
    system = MolecularSystem.from_entities(
        entities,
        instances,
        source_bonds,
        instance_values={instance_id: entity for _, instance_id, _, entity, _ in groups},
    )

    atom_count = int(system.atomic_number.shape[0])
    component_count = int(system.component_symbol.shape[0])
    component_index_by_reference: dict[ComponentReference, int] = {}
    atom_index_by_site: dict[AtomSite, int] = {}
    for instance_index in range(int(system.instance_symbol.shape[0])):
        instance_id = system.instance_id(instance_index)
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        for component_index in range(component_start, component_stop):
            local_component = component_index - component_start
            component_index_by_reference[(instance_id, local_component)] = component_index
            atom_start = int(system.component_atom_ptr[component_index])
            atom_stop = int(system.component_atom_ptr[component_index + 1])
            for atom_index in range(atom_start, atom_stop):
                atom_index_by_site[
                    AtomSite(
                        instance_id,
                        local_component,
                        system.symbols[int(system.atom_symbol[atom_index])],
                    )
                ] = atom_index
    coordinates = torch.zeros((atom_count, 3), dtype=torch.float32)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    occupancy = torch.zeros(atom_count, dtype=torch.float32)
    occupancy_mask = torch.zeros(atom_count, dtype=torch.bool)
    b_factor = torch.zeros(atom_count, dtype=torch.float32)
    b_factor_mask = torch.zeros(atom_count, dtype=torch.bool)
    atom_indices = [atom_index_by_site[site_by_row_index[int(row.row_index)]] for row in rows]
    seen_atom_indices: set[int] = set()
    for atom_index in atom_indices:
        if atom_index in seen_atom_indices:
            raise ValueError(f"selected source rows map more than once to {system.atom_site(atom_index)!r}")
        seen_atom_indices.add(atom_index)
    source_atom_index = torch.tensor(atom_indices, dtype=torch.int64)
    source_atomic_number = torch.tensor(
        [ATOMIC_NUMBER_BY_ELEMENT[str(element)] for element in rows.element],
        dtype=system.atomic_number.dtype,
    )
    element_mismatch = system.atomic_number[source_atom_index].cpu() != source_atomic_number
    if bool(element_mismatch.any()):
        source_index = int(torch.nonzero(element_mismatch, as_tuple=False)[0])
        site = system.atom_site(atom_indices[source_index])
        raise ValueError(f"source element {rows.element[source_index]!r} disagrees with component atom at {site!r}")
    source_isotope_mask = torch.tensor(
        [value is not None for value in rows.isotope],
        dtype=torch.bool,
    )
    system.isotope[source_atom_index[source_isotope_mask]] = torch.tensor(
        [int(value) for value in rows.isotope if value is not None],
        dtype=system.isotope.dtype,
    )
    system.isotope_known[source_atom_index[source_isotope_mask]] = True
    source_charge_mask = torch.tensor(
        [value is not None for value in rows.charge],
        dtype=torch.bool,
    )
    system.formal_charge[source_atom_index[source_charge_mask]] = torch.tensor(
        [int(value) for value in rows.charge if value is not None],
        dtype=system.formal_charge.dtype,
    )
    system.formal_charge_known[source_atom_index[source_charge_mask]] = True

    source_coordinate_mask = torch.from_numpy(np.asarray(rows.coordinate_present, dtype=np.bool_))
    coordinate_mask[source_atom_index] = source_coordinate_mask
    coordinates[source_atom_index] = torch.from_numpy(
        np.array(rows.coordinates, dtype=np.float32, copy=True, order="C")
    )
    source_occupancy_mask = torch.tensor(
        [value is not None for value in rows.occupancy],
        dtype=torch.bool,
    )
    occupancy_mask[source_atom_index] = source_occupancy_mask
    occupancy[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows.occupancy],
        dtype=torch.float32,
    )
    source_b_factor_mask = torch.tensor(
        [value is not None for value in rows.b_factor],
        dtype=torch.bool,
    )
    b_factor_mask[source_atom_index] = source_b_factor_mask
    b_factor[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows.b_factor],
        dtype=torch.float32,
    )
    if not bool(coordinate_mask.any()):
        raise ValueError("selected structure contains no observed coordinates")

    source_symbols: list[str] = []
    source_symbol_index: dict[str, int] = {}

    def intern(value: str) -> int:
        index = source_symbol_index.get(value)
        if index is None:
            index = len(source_symbols)
            source_symbol_index[value] = index
            source_symbols.append(value)
        return index

    residue_id = torch.full((component_count,), -1, dtype=torch.int64)
    insertion_code = torch.full((component_count,), -1, dtype=torch.int64)
    for reference, (number, code) in residue_numbers.items():
        component_index = component_index_by_reference[reference]
        residue_id[component_index] = intern(number)
        if code:
            insertion_code[component_index] = intern(code)
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=tuple(source_symbols),
        residue_id=residue_id,
        insertion_code=insertion_code,
        batch_size=[],
    )


def _reuse_entity(
    entities: dict[str, PolymerSequence | Molecule],
    entity_id: str,
    entity: PolymerSequence | Molecule,
) -> PolymerSequence | Molecule:
    previous = entities.get(entity_id)
    if previous is None:
        entities[entity_id] = entity
        return entity
    if previous != entity:
        raise ValueError(f"source entity ID {entity_id!r} has conflicting molecular definitions")
    return previous


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
