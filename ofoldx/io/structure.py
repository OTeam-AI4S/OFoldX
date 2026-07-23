r"""Canonical PDB/mmCIF I/O over :mod:`ofoldx.data` domain values."""

from __future__ import annotations

import contextlib
import itertools
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..data.components import get_builtin_component_polymer_type, get_component
from ..data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    SystemBond,
)
from ..data.structure import ResidueNumber, Structure
from .records import AtomRecord, ParsedStructure, atom_record_residue_key, unique_identifier

_FORMATS: dict[str, Literal["pdb", "mmcif"]] = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}
_ATOM_LAYOUTS = frozenset({"observed", "complete_heavy", "complete_all"})
_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})


@dataclass(frozen=True)
class LossReport:
    r"""Machine-readable fields lost or rewritten by an exchange writer."""

    dropped_fields: tuple[str, ...] = ()
    rewritten_fields: tuple[str, ...] = ()


def read_structure(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    atom_layout: Literal["observed", "complete_heavy", "complete_all"] = "observed",
    components: Mapping[str, ChemicalComponent] | None = None,
) -> Structure:
    r"""Read one model/assembly with one residue-consistent altloc choice."""

    parsed, component_table = _prepare_read(path, model, assembly_id, atom_layout, components)
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if parsed.canonical_payload is not None:
        from .mmcif import decode_canonical_mmcif

        return _apply_atom_layout(decode_canonical_mmcif(parsed), atom_layout)
    selected = _select_altloc(parsed.rows, altloc)
    return _build_structure(parsed, selected, atom_layout, component_table)


def read_structures(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    atom_layout: Literal["observed", "complete_heavy", "complete_all"] = "observed",
    components: Mapping[str, ChemicalComponent] | None = None,
) -> tuple[Structure, ...]:
    r"""Expand every residue-consistent alternate-location combination."""

    parsed, component_table = _prepare_read(path, model, assembly_id, atom_layout, components)
    if parsed.canonical_payload is not None:
        from .mmcif import decode_canonical_mmcif

        return (_apply_atom_layout(decode_canonical_mmcif(parsed), atom_layout),)
    selections = _expand_altlocs(parsed.rows)
    return tuple(_build_structure(parsed, selected, atom_layout, component_table) for selected in selections)


def write_structure(
    structure: Structure,
    path: str | os.PathLike[str],
    *,
    allow_lossy: bool = False,
) -> LossReport | None:
    r"""Write one canonical Structure, refusing unacknowledged format loss."""

    target, format_name = _structure_path(path)
    if format_name == "mmcif":
        from .mmcif import write_mmcif

        write_mmcif(structure, target)
        return LossReport() if allow_lossy else None

    from . import pdb as pdb_io

    atoms, dropped, rewritten = pdb_io.prepare_pdb_atoms(structure)
    dropped_fields, rewritten_fields = pdb_io.pdb_loss_fields(structure, atoms, dropped, rewritten)
    report = LossReport(dropped_fields, rewritten_fields)
    if (report.dropped_fields or report.rewritten_fields) and not allow_lossy:
        fields = sorted({*report.dropped_fields, *report.rewritten_fields})
        raise ValueError(
            "PDB cannot represent this Structure losslessly; pass allow_lossy=True " f"to acknowledge: {fields!r}"
        )
    pdb_io.write_pdb(structure, target, atoms)
    return report if allow_lossy else None


def _prepare_read(
    path: str | os.PathLike[str],
    model: int,
    assembly_id: str | None,
    atom_layout: str,
    components: Mapping[str, ChemicalComponent] | None,
) -> tuple[ParsedStructure, dict[str, ChemicalComponent]]:
    source, format_name = _structure_path(path)
    if model <= 0:
        raise ValueError("model must be a positive one-based integer")
    if atom_layout not in _ATOM_LAYOUTS:
        raise ValueError(f"unsupported atom_layout: {atom_layout!r}")
    if format_name == "mmcif":
        from .mmcif import parse_mmcif

        parsed = parse_mmcif(source, model=model, assembly_id=assembly_id)
    else:
        from . import pdb as pdb_io

        parsed = pdb_io.parse_pdb(source, model=model, assembly_id=assembly_id)
    if not parsed.rows:
        raise ValueError("structure source contains no atoms for the requested model/assembly")

    component_table: dict[str, ChemicalComponent] = {}
    for component_id in {row.component_id for row in parsed.rows}:
        with contextlib.suppress(KeyError):
            component_table[component_id] = get_component(component_id)
    if components is not None:
        supplied = dict(components)
        for component_id, component in supplied.items():
            if component.component_id != component_id:
                raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
        component_table.update(supplied)
    if atom_layout == "complete_all":
        component_table = {
            component_id: _materialize_component_hydrogens(component)
            for component_id, component in component_table.items()
        }
    return parsed, component_table


def _materialize_component_hydrogens(component: ChemicalComponent) -> ChemicalComponent:
    counts: list[int] = []
    for atom in component.atoms:
        if atom.explicit_hydrogens is None or atom.implicit_hydrogens is None:
            raise ValueError(
                f"component {component.component_id!r} has unknown hydrogen counts and cannot be completed"
            )
        counts.append(atom.explicit_hydrogens + atom.implicit_hydrogens)
    if not any(counts):
        return component
    if component.bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph and cannot be completed")
    used_atom_ids = {atom.atom_id for atom in component.atoms}
    atoms: list[ChemicalAtom] = []
    bonds = set(component.bonds)
    reference_coordinates: list[tuple[float, float, float] | None] = []
    for atom_index, (atom, count) in enumerate(zip(component.atoms, counts)):
        atoms.append(
            ChemicalAtom(
                atom_id=atom.atom_id,
                element=atom.element,
                isotope=atom.isotope,
                formal_charge=atom.formal_charge,
                radical_electrons=atom.radical_electrons,
                explicit_hydrogens=0,
                implicit_hydrogens=0,
                implicit_hydrogens_allowed=False,
                aromatic=atom.aromatic,
                stereo=atom.stereo,
            )
        )
        if component.reference_coordinates:
            reference_coordinates.append(component.reference_coordinates[atom_index])
        for hydrogen_index in range(1, count + 1):
            seed = f"{atom.atom_id}:H{hydrogen_index}"
            hydrogen_id = seed
            suffix = 2
            while hydrogen_id in used_atom_ids:
                hydrogen_id = f"{seed}_{suffix}"
                suffix += 1
            used_atom_ids.add(hydrogen_id)
            atoms.append(
                ChemicalAtom(
                    atom_id=hydrogen_id,
                    element="H",
                    implicit_hydrogens_allowed=False,
                )
            )
            if component.reference_coordinates:
                reference_coordinates.append(None)
            bonds.add(ChemicalBond(atom.atom_id, hydrogen_id))
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atoms),
        bonds=frozenset(bonds),
        previous_attachment=component.previous_attachment,
        next_attachment=component.next_attachment,
        reference_coordinates=tuple(reference_coordinates),
    )


def _structure_path(path: str | os.PathLike[str]) -> tuple[Path, Literal["pdb", "mmcif"]]:
    resolved = Path(os.path.expanduser(os.fspath(path)))
    try:
        return resolved, _FORMATS[resolved.suffix.lower()]
    except KeyError:
        raise ValueError("structure path suffix must be .pdb, .cif, or .mmcif") from None


def _select_altloc(
    rows: tuple[AtomRecord, ...],
    policy: Literal["highest_occupancy", "first"],
) -> tuple[AtomRecord, ...]:
    choices = _altloc_choices(rows)
    selected_labels: dict[tuple[object, ...], str] = {}
    for residue_key, labels in choices:
        if policy == "first":
            selected_labels[residue_key] = labels[0]
            continue
        residue_rows = [row for row in rows if atom_record_residue_key(row) == residue_key]
        scores = {
            label: sum((row.occupancy or 0.0) for row in residue_rows if row.altloc_id == label) for label in labels
        }
        selected_labels[residue_key] = max(labels, key=lambda label: scores[label])
    return tuple(
        row
        for row in rows
        if row.altloc_id is None or selected_labels.get(atom_record_residue_key(row), row.altloc_id) == row.altloc_id
    )


def _expand_altlocs(rows: tuple[AtomRecord, ...]) -> tuple[tuple[AtomRecord, ...], ...]:
    choices = _altloc_choices(rows)
    if not choices:
        return (rows,)
    expanded: list[tuple[AtomRecord, ...]] = []
    for labels in itertools.product(*(labels for _, labels in choices)):
        selection = {residue_key: label for (residue_key, _), label in zip(choices, labels)}
        expanded.append(
            tuple(
                row
                for row in rows
                if row.altloc_id is None or selection.get(atom_record_residue_key(row), row.altloc_id) == row.altloc_id
            )
        )
    return tuple(expanded)


def _altloc_choices(
    rows: tuple[AtomRecord, ...],
) -> tuple[tuple[tuple[object, ...], tuple[str, ...]], ...]:
    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        labels = labels_by_residue.setdefault(atom_record_residue_key(row), [])
        if row.altloc_id not in labels:
            labels.append(row.altloc_id)
    return tuple((key, tuple(labels)) for key, labels in labels_by_residue.items() if labels)


def _apply_atom_layout(structure: Structure, atom_layout: str) -> Structure:
    if atom_layout == "observed":
        return structure
    system = structure.system
    if atom_layout == "complete_all":
        entities: dict[str, PolymerSequence | Molecule] = {}
        for entity_id, entity in system.entities.items():
            components = tuple(_materialize_component_hydrogens(component) for component in entity.components)
            if isinstance(entity, PolymerSequence):
                entities[entity_id] = PolymerSequence(entity.polymer_type, components)
            else:
                entities[entity_id] = Molecule(components, entity.bonds)
        system = MolecularSystem(entities, system.instances, system.bonds)
    if atom_layout == "complete_all":
        sites = tuple(site for site, _ in system.iter_atoms())
    else:
        sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H")
    source_indices = {site: index for index, site in enumerate(structure.sites)}
    coordinates = np.zeros((len(sites), 3), dtype=structure.coordinates.dtype)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    source_occupancy = structure.occupancy
    occupancy = None
    if source_occupancy is not None:
        dtype = source_occupancy.dtype if np.issubdtype(source_occupancy.dtype, np.floating) else np.float64
        occupancy = np.full(len(sites), np.nan, dtype=dtype)
    for index, site in enumerate(sites):
        source_index = source_indices.get(site)
        if source_index is None:
            continue
        coordinates[index] = structure.coordinates[source_index]
        coordinate_mask[index] = structure.coordinate_mask[source_index]
        if occupancy is not None and source_occupancy is not None:
            occupancy[index] = source_occupancy[source_index]
    return Structure(system, sites, coordinates, coordinate_mask, occupancy, structure.residue_numbers)


def _build_structure(
    parsed: ParsedStructure,
    rows: tuple[AtomRecord, ...],
    atom_layout: str,
    component_table: dict[str, ChemicalComponent],
) -> Structure:
    if not rows:
        raise ValueError("alternate-location selection contains no atoms")
    residue_rows: dict[tuple[object, ...], list[AtomRecord]] = {}
    chain_residue_order: dict[tuple[str | None, str], list[tuple[object, ...]]] = {}
    for row in rows:
        residue_key = atom_record_residue_key(row)
        residue_rows.setdefault(residue_key, []).append(row)
        chain_key = (row.symmetry_id, row.chain_id)
        order = chain_residue_order.setdefault(chain_key, [])
        if residue_key not in order:
            order.append(residue_key)

    for residue_key, residue in residue_rows.items():
        component_id = residue[0].component_id
        if component_id not in component_table:
            raise ValueError(
                f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
            )
        source_atom_names = {row.atom_name for row in residue}
        component_atom_names = {atom.atom_id for atom in component_table[component_id].atoms}
        unknown_atoms = source_atom_names - component_atom_names
        if unknown_atoms:
            raise ValueError(
                f"source residue {residue_key!r} contains atoms absent from component {component_id!r}: "
                f"{sorted(unknown_atoms)!r}"
            )

    if parsed.format == "mmcif":
        from .mmcif import read_mmcif_branch_entities

        branch_entities, branch_component_indices = read_mmcif_branch_entities(parsed, component_table, rows)
    else:
        branch_entities, branch_component_indices = {}, {}
    # mmCIF uses label_seq_id for polymer membership. PDB uses record type and the
    # built-in component class because Structure does not retain source categories.
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
    entity_by_id: dict[str, PolymerSequence | Molecule] = {}

    for (symmetry_id, chain_id), residue_order in chain_residue_order.items():
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        if parsed.format == "mmcif":
            polymer_residues = [key for key in residue_order if residue_rows[key][0].label_seq_id is not None]
        else:
            polymer_indices = [
                index
                for index, key in enumerate(residue_order)
                if any(row.group == "ATOM" for row in residue_rows[key])
                or get_builtin_component_polymer_type(residue_rows[key][0].component_id) is not None
            ]
            polymer_residues = residue_order[polymer_indices[0] : polymer_indices[-1] + 1] if polymer_indices else []
        if polymer_residues:
            classified_polymer_types = (
                get_builtin_component_polymer_type(residue_rows[key][0].component_id) for key in polymer_residues
            )
            polymer_types = {value for value in classified_polymer_types if value is not None}
            if polymer_types == {"dna", "rna"}:
                chain_polymer_type: PolymerType = "dna_rna_hybrid"
            elif len(polymer_types) == 1:
                chain_polymer_type = next(iter(polymer_types))
            else:
                raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
            first = residue_rows[polymer_residues[0]][0]
            source_entity_id = first.entity_id
            entity_id = (
                f"polymer_{source_entity_id}"
                if parsed.format == "mmcif" and source_entity_id is not None
                else f"polymer_{chain_id}"
            )
            entity = _reuse_entity(
                entity_by_id,
                entity_id,
                PolymerSequence(
                    chain_polymer_type,
                    tuple(component_table[residue_rows[key][0].component_id] for key in polymer_residues),
                ),
            )
            instance_id = unique_identifier(base_instance_id, used_instance_ids)
            groups.append(
                (
                    min(row.row_index for key in polymer_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(enumerate(polymer_residues)),
                )
            )

        polymer_residue_set = set(polymer_residues)
        nonpolymer_residues = [key for key in residue_order if key not in polymer_residue_set]
        molecule_groups: list[tuple[tuple[object, ...], ...]] = []
        if parsed.format == "mmcif":
            grouped_by_entity: dict[str, list[tuple[object, ...]]] = {}
            for residue_key in nonpolymer_residues:
                source_entity_id = residue_rows[residue_key][0].entity_id
                if source_entity_id in branch_entities:
                    grouped_by_entity.setdefault(source_entity_id, []).append(residue_key)
                else:
                    molecule_groups.append((residue_key,))
            molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
            molecule_groups.sort(key=lambda group: min(row.row_index for key in group for row in residue_rows[key]))
        else:
            molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)

        for group_residues in molecule_groups:
            first = residue_rows[group_residues[0]][0]
            first_component = component_table[first.component_id]
            residue_label = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
            residue_identity = f"{residue_label}{first.insertion_code or ''}"
            source_entity_id = first.entity_id
            if source_entity_id in branch_entities:
                entity = branch_entities[source_entity_id]
                entity_id = f"molecule_{source_entity_id}"
                instance_seed = base_instance_id
                indexed_residues: list[tuple[int, tuple[object, ...]]] = []
                for residue_key in group_residues:
                    row = residue_rows[residue_key][0]
                    scheme_key = (
                        row.chain_id,
                        row.entity_id,
                        row.component_id,
                        row.auth_chain_id,
                        row.auth_residue_id,
                    )
                    component_index = branch_component_indices.get(scheme_key)
                    if component_index is None and row.auth_component_id is not None:
                        component_index = branch_component_indices.get(
                            (
                                row.chain_id,
                                row.entity_id,
                                row.auth_component_id,
                                row.auth_chain_id,
                                row.auth_residue_id,
                            )
                        )
                    if component_index is None:
                        raise ValueError(f"cannot map branched-entity residue {residue_key!r} to its declaration")
                    indexed_residues.append((component_index, residue_key))
                if len({index for index, _ in indexed_residues}) != len(indexed_residues):
                    raise ValueError(f"branched entity {source_entity_id!r} maps multiple residues to one component")
                indexed_residues.sort()
            elif parsed.format == "mmcif" and source_entity_id is not None:
                entity = Molecule((first_component,))
                entity_id = f"molecule_{source_entity_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            else:
                entity = Molecule((first_component,))
                entity_id = f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            entity = _reuse_entity(entity_by_id, entity_id, entity)
            instance_id = unique_identifier(instance_seed, used_instance_ids)
            groups.append(
                (
                    min(row.row_index for key in group_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(indexed_residues),
                )
            )

    groups.sort(key=lambda item: item[0])
    instances = {instance_id: entity_id for _, instance_id, entity_id, _, _ in groups}
    provisional = MolecularSystem(entity_by_id, instances)

    site_by_row_index: dict[int, AtomSite] = {}
    # ResidueNumber stores one numbering axis, so author numbering takes precedence
    # over label numbering.
    residue_numbers: dict[ComponentReference, ResidueNumber] = {}
    for _, instance_id, _, group_entity, group_indexed in groups:
        is_polymer = isinstance(group_entity, PolymerSequence)
        for component_index, residue_key in group_indexed:
            residue = residue_rows[residue_key]
            if is_polymer:
                first = residue[0]
                number = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
                if number is not None:
                    residue_numbers[(instance_id, component_index)] = ResidueNumber(number, first.insertion_code or "")
            for row in residue:
                site_by_row_index[row.row_index] = AtomSite(instance_id, component_index, row.atom_name)

    for row in rows:
        site = site_by_row_index[row.row_index]
        atom = provisional.atom(site)
        if atom.element != row.element:
            raise ValueError(
                f"source element {row.element!r} disagrees with component atom {atom.element!r} at {site!r}"
            )
        if row.charge is not None and row.charge != atom.formal_charge:
            raise ValueError(
                f"source formal charge {row.charge!r} disagrees with component atom "
                f"{atom.formal_charge!r} at {site!r}"
            )

    source_bonds = _read_source_bonds(parsed, rows, site_by_row_index, provisional)
    system = MolecularSystem(entity_by_id, instances, frozenset(source_bonds))
    observed_sites = tuple(site_by_row_index[row.row_index] for row in rows)
    if len(observed_sites) != len(set(observed_sites)):
        raise ValueError("selected source rows map more than once to one canonical AtomSite")
    # Complete layouts are materialized from MolecularSystem atoms; complete_heavy
    # filters hydrogen atoms here.
    if atom_layout == "observed":
        sites = observed_sites
    elif atom_layout == "complete_all":
        sites = tuple(site for site, _ in system.iter_atoms())
    else:
        sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H")
    site_indices = {site: index for index, site in enumerate(sites)}
    coordinates = np.zeros((len(sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    occupancy = (
        np.full(len(sites), np.nan, dtype=np.float32) if any(row.occupancy is not None for row in rows) else None
    )
    for row in rows:
        site = site_by_row_index[row.row_index]
        index = site_indices[site]
        coordinates[index] = row.coordinates
        coordinate_mask[index] = row.coordinate_present
        if occupancy is not None and row.occupancy is not None:
            occupancy[index] = row.occupancy
    return Structure(
        system=system,
        sites=sites,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        residue_numbers=residue_numbers or None,
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


def _read_source_bonds(
    parsed: ParsedStructure,
    selected_rows: tuple[AtomRecord, ...],
    site_by_row_index: dict[int, AtomSite],
    system: MolecularSystem,
) -> tuple[SystemBond, ...]:
    if parsed.format == "pdb":
        from . import pdb as pdb_io

        declared = pdb_io.read_pdb_bonds(parsed, selected_rows, site_by_row_index)
    else:
        from .mmcif import read_mmcif_bonds

        declared = read_mmcif_bonds(parsed, selected_rows, site_by_row_index)
    base_bonds = {frozenset((bond.site_a, bond.site_b)): bond for bond in system.iter_bonds()}
    overlay: dict[frozenset[AtomSite], SystemBond] = {}
    for bond in declared:
        endpoints = frozenset((bond.site_a, bond.site_b))
        base = base_bonds.get(endpoints)
        if base is not None:
            if (bond.order, bond.aromatic) != (base.order, base.aromatic):
                raise ValueError(
                    f"source bond conflicts with declared intrinsic topology: " f"{bond.site_a!r}, {bond.site_b!r}"
                )
            continue
        previous = overlay.get(endpoints)
        if previous is not None:
            if (previous.order, previous.aromatic, previous.stereo) != (bond.order, bond.aromatic, bond.stereo):
                raise ValueError(f"conflicting source bond declarations: {bond.site_a!r}, {bond.site_b!r}")
            continue
        overlay[endpoints] = bond
    return tuple(overlay.values())
