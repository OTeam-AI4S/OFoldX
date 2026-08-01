r"""PDB and mmCIF I/O for the canonical tensor molecular graph."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import torch

from ..data.components import complete_component, get_builtin_component_polymer_type, get_component
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import (
    BOND_ROLE_COMPONENT,
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    MoleculeType,
    PolymerAttachment,
    PolymerSequence,
    PolymerType,
    SystemBond,
    is_water_component,
)
from ..data.structure import Structure
from .records import unique_identifier

if TYPE_CHECKING:
    from .mmcif import _MmcifEntityDeclarations

_FORMATS: dict[str, Literal["pdb", "mmcif"]] = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}
_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})
_MmcifPolymerPosition: TypeAlias = tuple[str | None, str, str, int]
_MMCIF_MOLECULE_TYPES: dict[str, MoleculeType] = {
    "macrolide": "macrolide",
    "non-polymer": "non-polymer",
    "water": "water",
}


def _graph_component_signatures(graph: MolecularSystem) -> tuple[tuple[object, ...], ...]:
    r"""Exact value signatures of every realized component subgraph."""

    component_count = int(graph.component_atom_ptr.numel()) - 1
    component_by_atom = [0] * int(graph.atom_symbol.numel())
    atoms_by_component: list[tuple[tuple[object, ...], ...]] = []
    for component_index in range(component_count):
        start = int(graph.component_atom_ptr[component_index])
        stop = int(graph.component_atom_ptr[component_index + 1])
        component_by_atom[start:stop] = [component_index] * (stop - start)
        atoms_by_component.append(
            tuple(
                (
                    graph.symbols[int(graph.atom_symbol[atom])],
                    int(graph.atomic_number[atom]),
                    int(graph.isotope[atom]) if bool(graph.isotope_known[atom]) else None,
                    int(graph.formal_charge[atom]) if bool(graph.formal_charge_known[atom]) else None,
                    int(graph.radical_electrons[atom]) if bool(graph.radical_electrons_known[atom]) else None,
                    int(graph.explicit_hydrogens[atom]) if bool(graph.explicit_hydrogens_known[atom]) else None,
                    int(graph.implicit_hydrogens[atom]) if bool(graph.implicit_hydrogens_known[atom]) else None,
                    (
                        bool(graph.implicit_hydrogens_allowed[atom])
                        if bool(graph.implicit_hydrogens_allowed_known[atom])
                        else None
                    ),
                    bool(graph.atom_aromatic[atom]) if bool(graph.atom_aromatic_known[atom]) else None,
                    None if int(graph.atom_stereo[atom]) < 0 else graph.symbols[int(graph.atom_stereo[atom])],
                    (
                        tuple(float(value) for value in graph.reference_coordinates[atom])
                        if bool(graph.reference_coordinate_mask[atom])
                        else None
                    ),
                )
                for atom in range(start, stop)
            )
        )

    edges_by_component: list[list[tuple[object, ...]]] = [[] for _ in range(component_count)]
    for edge in range(graph.edge_index.shape[1]):
        if int(graph.bond_role[edge]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(graph.edge_index[0, edge])
        atom_b = int(graph.edge_index[1, edge])
        component_index = component_by_atom[atom_a]
        if component_by_atom[atom_b] != component_index:
            continue
        start = int(graph.component_atom_ptr[component_index])
        stereo = int(graph.bond_stereo[edge])
        stereo_atoms = graph.bond_stereo_index[:, edge]
        edges_by_component[component_index].append(
            (
                atom_a - start,
                atom_b - start,
                float(graph.bond_order[edge]) if bool(graph.bond_order_known[edge]) else None,
                bool(graph.bond_aromatic[edge]) if bool(graph.bond_aromatic_known[edge]) else None,
                int(graph.bond_kind[edge]),
                int(graph.bond_direction[edge]),
                None if stereo < 0 else graph.symbols[stereo],
                tuple(-1 if int(atom) < 0 else int(atom) - start for atom in stereo_atoms),
            )
        )
    return tuple(
        (
            bool(graph.component_topology_complete[index]),
            atoms_by_component[index],
            tuple(edges_by_component[index]),
        )
        for index in range(component_count)
    )


def _realized_component_signature(
    component: ChemicalComponent,
    polymer_type: PolymerType | None,
    *,
    has_previous: bool = False,
    has_next: bool = False,
) -> tuple[object, ...]:
    r"""Signature after realizing a component in its stated linear-polymer context."""

    if polymer_type is None:
        entity: Molecule | PolymerSequence = Molecule((component,))
        component_index = 0
    else:
        left = ChemicalComponent(
            "__LEFT__",
            (ChemicalAtom("X", "C"),),
            bonds=frozenset(),
            next_attachment=PolymerAttachment("X"),
        )
        right = ChemicalComponent(
            "__RIGHT__",
            (ChemicalAtom("X", "C"),),
            bonds=frozenset(),
            previous_attachment=PolymerAttachment("X"),
        )
        components = (
            *((left,) if has_previous else ()),
            component,
            *((right,) if has_next else ()),
        )
        entity = PolymerSequence(polymer_type, components)
        component_index = int(has_previous)
    expected = MolecularSystem.from_entities({"expected": entity}, {"expected": "expected"})
    return _graph_component_signatures(expected)[component_index]


def read_structure(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = True,
    include_hydrogens: bool = True,
    components: Mapping[str, ChemicalComponent] | None = None,
    component_resolver: Callable[[str], ChemicalComponent] | None = None,
) -> Structure:
    r"""
    Read one model and alternate-location realization from PDB or mmCIF.

    Args:
        path: Input PDB or mmCIF path, optionally gzip-compressed.
        model: One-based model number.
        assembly_id: Biological assembly identifier, or `None` for the stored
            asymmetric unit.
        altloc: Residue-level alternate-location choice.
        include_water: Whether to retain water entities.
        include_hydrogens: Whether to retain hydrogen atoms in the molecular graph.
        components: Optional component definitions overriding built-in chemistry.
        component_resolver: Optional lazy component lookup used before the built-in
            component library. A missing component must raise ``KeyError``.

    Returns:
        A CPU tensor Structure whose atom axis is the complete retained system graph.
    """

    source, format_name = _structure_path(path, allow_compressed=True)
    if model <= 0:
        raise ValueError("model must be a positive one-based integer")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")

    cif_file: object | None = None
    mmcif_placements: dict[tuple[str | None, str], np.ndarray] = {}
    mmcif_topology_placements: set[tuple[str, str, str | None]] = set()
    source_components: dict[str, ChemicalComponent] = {}
    source_component_parents: dict[str, str] = {}
    source_component_stereo_known: dict[str, tuple[frozenset[str], frozenset[frozenset[str]]]] = {}
    declared_component_ids: set[str] = set()
    pdb_lines: tuple[str, ...] = ()
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...] = ()
    pdb_declared_sequences: dict[str, tuple[str, ...]] = {}
    pdb_assembly_placements: dict[str, tuple[np.ndarray, frozenset[str]]] | None = None
    mmcif_polymer_monomers: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None = None
    mmcif_entity_declarations: _MmcifEntityDeclarations | None = None
    mmcif_entity_types: Mapping[str, str] = {}
    mmcif_atom_provenance: frozenset[AtomSite] | None = None
    if format_name == "mmcif":
        from .mmcif import (
            _read_mmcif_atom_provenance,
            _read_mmcif_entity_declarations,
            parse_mmcif,
            read_mmcif_resolutions,
        )

        source_rows, cif_file, mmcif_placements = parse_mmcif(source, model=model, assembly_id=assembly_id)
        source_resolutions = read_mmcif_resolutions(cif_file)
        mmcif_atom_provenance = _read_mmcif_atom_provenance(cif_file)
        from .mmcif import read_mmcif_components

        (
            source_components,
            declared_component_ids,
            source_component_parents,
            source_component_stereo_known,
        ) = read_mmcif_components(cif_file, owned=() if components is None else tuple(components))
        mmcif_entity_declarations = _read_mmcif_entity_declarations(cif_file)
        mmcif_entity_types, mmcif_polymer_monomers, *_ = mmcif_entity_declarations
    else:
        from .pdb import parse_pdb, read_pdb_resolutions

        source_rows, pdb_lines, pdb_missing_residues, pdb_assembly_placements = parse_pdb(
            source,
            model=model,
            assembly_id=assembly_id,
        )
        source_resolutions = read_pdb_resolutions(pdb_lines)
        from .pdb import pdb_sequences

        pdb_declared_sequences = pdb_sequences(pdb_lines)
    source_non_leaving_atom_ids = {
        component_id: component.non_leaving_atom_ids for component_id, component in source_components.items()
    }
    source_formal_charge_atoms = {
        component_id: frozenset(atom.atom_id for atom in component.atoms if atom.formal_charge is not None)
        for component_id, component in source_components.items()
    }
    if not len(source_rows) and format_name == "pdb":
        raise ValueError("structure source contains no atoms for the requested model/assembly")

    component_table: dict[str, ChemicalComponent] = {}
    pdb_declared_component_ids = {
        component_id for sequence in pdb_declared_sequences.values() for component_id in sequence
    }
    component_ids = {*source_rows.component_id, *declared_component_ids, *pdb_declared_component_ids}
    explicit_component_ids = set() if components is None else set(components)
    observed_components = _observed_components(source_rows)
    component_topology_complete: dict[str, bool] = {}
    for raw_component_id in component_ids:
        component_id = str(raw_component_id)
        if component_id in explicit_component_ids:
            continue
        source_component = source_components.get(component_id)
        source_topology_complete = source_component is not None and source_component.bonds is not None
        observed_component = observed_components.get(component_id)
        if observed_component is not None:
            if source_component is None:
                source_component = observed_component
                source_topology_complete = observed_component.bonds is not None
            else:
                source_component, source_topology_complete = complete_component(
                    observed_component,
                    source_component,
                    occurrence_topology_complete=False,
                )
        reference_component: ChemicalComponent | None = None
        if component_resolver is not None:
            with contextlib.suppress(KeyError):
                reference_component = component_resolver(component_id)
        with contextlib.suppress(KeyError):
            if reference_component is None:
                reference_component = get_component(component_id)
        if reference_component is not None and reference_component.component_id != component_id:
            raise ValueError("component resolver keys must equal ChemicalComponent.component_id")
        if source_component is not None and reference_component is not None:
            atom_stereo_known, bond_stereo_known = source_component_stereo_known.get(
                component_id,
                (frozenset(), frozenset()),
            )
            component_table[component_id], component_topology_complete[component_id] = complete_component(
                source_component,
                reference_component,
                occurrence_topology_complete=source_topology_complete,
                atom_stereo_known=atom_stereo_known,
                bond_stereo_known=bond_stereo_known,
            )
        elif source_component is not None:
            component_table[component_id] = source_component
            component_topology_complete[component_id] = source_topology_complete
        elif reference_component is not None:
            component_table[component_id] = reference_component
            component_topology_complete[component_id] = reference_component.bonds is not None
    if components is not None:
        for component_id, component in components.items():
            if component.component_id != component_id:
                raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
            component_table[component_id] = component
            component_topology_complete[component_id] = component.bonds is not None
            source_non_leaving_atom_ids[component_id] = component.non_leaving_atom_ids
    for component_id, component in observed_components.items():
        component_table.setdefault(component_id, component)
    for component_id, component in component_table.items():
        component_topology_complete.setdefault(component_id, component.bonds is not None)
    if mmcif_atom_provenance is not None:
        assert cif_file is not None
        from .mmcif import _validate_mmcif_atom_provenance

        _validate_mmcif_atom_provenance(cif_file, mmcif_atom_provenance, component_table)

    selected_for_copy = source_rows
    if format_name == "mmcif":
        selected_for_copy = _select_mmcif_polymer_microheterogeneity(
            selected_for_copy,
            mmcif_polymer_monomers,
            altloc,
        )
    selected_for_copy = _select_altloc(selected_for_copy, altloc)
    retained_for_copy = selected_for_copy
    if not include_water:
        retained_for_copy = _exclude_water_rows(
            format_name,
            retained_for_copy,
            component_table,
            mmcif_entity_types,
        )
    crystallographic_rows_added = False
    if format_name == "pdb" and pdb_assembly_placements is None:
        from .pdb import _materialize_pdb_crystallographic_copies, _pdb_connection_symmetry_endpoints

        crystallographic_endpoints = _pdb_connection_symmetry_endpoints(
            pdb_lines,
            selected_for_copy,
            include_water=include_water,
            components=component_table,
        )
        if crystallographic_endpoints:
            source_row_count = len(source_rows)
            source_rows = _materialize_pdb_crystallographic_copies(
                source_rows,
                retained_for_copy,
                pdb_lines,
                crystallographic_endpoints,
                include_water=include_water,
                components=component_table,
            )
            crystallographic_rows_added = len(source_rows) > source_row_count
    elif format_name == "mmcif" and assembly_id is None:
        assert cif_file is not None
        from .mmcif import _materialize_mmcif_crystallographic_copies

        source_row_count = len(source_rows)
        source_rows, mmcif_placements, mmcif_topology_placements = _materialize_mmcif_crystallographic_copies(
            cif_file,
            source_rows,
            retained_for_copy,
            mmcif_placements,
            include_water=include_water,
        )
        crystallographic_rows_added = len(source_rows) > source_row_count
    if crystallographic_rows_added:
        rows = source_rows
        if format_name == "mmcif":
            rows = _select_mmcif_polymer_microheterogeneity(rows, mmcif_polymer_monomers, altloc)
        rows = _select_altloc(rows, altloc)
        if not include_water:
            rows = _exclude_water_rows(
                format_name,
                rows,
                component_table,
                mmcif_entity_types,
            )
    else:
        rows = retained_for_copy
    if format_name == "pdb":
        from .pdb import read_pdb_component_bonds

        read_pdb_component_bonds(pdb_lines, rows, component_table)
    structure = _build_structure(
        format_name,
        rows,
        source_rows,
        component_table,
        include_water=include_water,
        include_declared_instances=assembly_id is None,
        cif_file=cif_file,
        pdb_lines=pdb_lines,
        pdb_missing_residues=pdb_missing_residues,
        pdb_declared_sequences=pdb_declared_sequences,
        source_component_parents=source_component_parents,
        mmcif_placements=mmcif_placements,
        mmcif_topology_placements=mmcif_topology_placements,
        pdb_assembly_placements=pdb_assembly_placements,
        source_resolutions=source_resolutions,
        source_formal_charge_atoms=source_formal_charge_atoms,
        source_non_leaving_atom_ids=source_non_leaving_atom_ids,
        source_formal_charge_provenance=mmcif_atom_provenance,
        component_topology_complete=component_topology_complete,
        mmcif_entity_declarations=mmcif_entity_declarations,
    )
    topology_complete = structure.system.component_topology_complete.clone()
    for component_index in range(int(topology_complete.numel())):
        component_id = structure.system.symbols[int(structure.system.component_symbol[component_index])]
        topology_complete[component_index] = component_topology_complete.get(
            component_id,
            bool(topology_complete[component_index]),
        )
    structure = structure.replace(  # type: ignore[attr-defined]
        system=structure.system.replace(  # type: ignore[attr-defined]
            component_topology_complete=topology_complete,
        ),
    )
    return structure if include_hydrogens else structure.suppress_hydrogens()


def write_structure(structure: Structure, path: str | os.PathLike[str]) -> None:
    r"""
    Write a Structure through the standard mmCIF or legacy PDB projection.

    mmCIF is the lossless projection: it either writes a structure whose readback compares
    equal or rejects chemistry that conflicts with the authoritative definition of its
    component ID.

    PDB preserves blank occupancy and B-factor fields, and refuses coordinates or numeric
    metadata that its fixed-width decimal fields would round. It likewise refuses a
    structure without author residue identifiers because manufacturing identifiers would
    change the public value; write mmCIF when those source identities are absent.

    What PDB cannot state at all it refuses rather than drops. ``SEQRES`` owns each
    polymer sequence, ``REMARK 465`` marks its unresolved positions, and observed
    non-polymers use ``HETATM`` plus ``CONECT``. A structure that observes no atom -- a
    topology-only [`ofoldx.data.Structure`][] built from a bare sequence, say -- still has
    no PDB projection. PDB identifies a residue by chain, number, and insertion code alone,
    so two components in one instance cannot share one of those keys.

    Args:
        structure: Realization to write.
        path: Destination; the suffix selects mmCIF or PDB.

    Raises:
        ValueError: If the structure holds more than one molecular system, or carries
            occupancy or B-factor evidence for atoms without coordinates; or, writing PDB,
            if the structure observes no atom or two components in one instance share one
            author residue key.
    """

    if structure.system.system_entity_ptr.numel() != 2 or structure.system.system_instance_ptr.numel() != 2:
        raise ValueError("one structure file cannot represent multiple molecular systems")
    omitted = ~structure.coordinate_mask
    if bool((omitted & (structure.occupancy_mask | structure.b_factor_mask)).any()):
        raise ValueError(
            "a structure file cannot preserve occupancy or B-factor evidence for atoms without coordinates"
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
    try:
        return resolved, _FORMATS[suffix]
    except KeyError:
        expected = ".pdb, .pdb.gz, .cif, .cif.gz, .mmcif, or .mmcif.gz" if allow_compressed else ".pdb, .cif, or .mmcif"
        raise ValueError(f"structure path suffix must be {expected}") from None


def _exclude_water_rows(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    component_table: Mapping[str, ChemicalComponent],
    entity_types: Mapping[str, str],
) -> np.recarray:
    water_entities = {entity_id for entity_id, entity_type in entity_types.items() if entity_type == "water"}
    keep = np.ones(len(rows), dtype=np.bool_)
    fallback_groups: dict[tuple[object, ...], tuple[list[int], bool]] = {}
    for index, (
        entity_id,
        symmetry_id,
        chain_id,
        residue_id,
        auth_residue_id,
        insertion_code,
        component_id,
    ) in enumerate(
        zip(
            rows.entity_id,
            rows.symmetry_id,
            rows.chain_id,
            rows.residue_id,
            rows.auth_residue_id,
            rows.insertion_code,
            rows.component_id,
            strict=True,
        )
    ):
        if entity_id in water_entities:
            keep[index] = False
            continue
        if entity_id is not None and entity_id in entity_types:
            continue
        key = (
            ("entity", entity_id)
            if format_name == "mmcif" and entity_id is not None
            else (
                "residue",
                symmetry_id,
                chain_id,
                residue_id,
                auth_residue_id,
                insertion_code,
                component_id,
            )
        )
        indices, all_water = fallback_groups.setdefault(key, ([], True))
        indices.append(index)
        if all_water and not is_water_component(component_table[str(component_id)]):
            fallback_groups[key] = (indices, False)
    for indices, all_water in fallback_groups.values():
        if all_water:
            keep[indices] = False
    return rows[keep].view(np.recarray)


def _observed_components(rows: np.recarray) -> dict[str, ChemicalComponent]:
    r"""Return the positive atom chemistry stated by coordinate records."""

    atoms_by_component: dict[str, dict[str, ChemicalAtom]] = {}
    for raw_component_id, raw_atom_name, raw_element in zip(
        rows.component_id,
        rows.atom_name,
        rows.element,
        strict=True,
    ):
        component_id = str(raw_component_id)
        atom_name = str(raw_atom_name)
        element = str(raw_element)
        if element not in ATOMIC_NUMBER_BY_ELEMENT:
            raise ValueError(f"source atom {atom_name!r} in component {component_id!r} has invalid element {element!r}")
        atom = ChemicalAtom(
            atom_id=atom_name,
            element=element,
        )
        component_atoms = atoms_by_component.setdefault(component_id, {})
        previous = component_atoms.get(atom_name)
        if previous is None:
            component_atoms[atom_name] = atom
            continue
        if previous.element != atom.element:
            raise ValueError(f"component {component_id!r} atom {atom_name!r} has conflicting source elements")
    return {
        component_id: ChemicalComponent(
            component_id=component_id,
            atoms=tuple(atoms.values()),
            bonds=None,
        )
        for component_id, atoms in atoms_by_component.items()
    }


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


def _mean_present_occupancy(values: Sequence[float | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / max(len(present), 1)


def _select_mmcif_polymer_microheterogeneity(
    rows: np.recarray,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    policy: Literal["highest_occupancy", "first"],
) -> np.recarray:
    r"""Select one deterministic deposited identity at each polymer position."""

    if polymer_monomers is None:
        return rows
    declared = {
        (entity_id, number): frozenset(candidates)
        for entity_id, positions in polymer_monomers.items()
        for number, candidates in positions
    }
    microheterogeneous = {key for key, candidates in declared.items() if len(candidates) > 1}
    if not microheterogeneous:
        return rows

    position_rows: dict[_MmcifPolymerPosition, list[np.record]] = {}
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            continue
        entity_id = str(row.entity_id)
        number = int(row.label_seq_id)
        if (entity_id, number) not in microheterogeneous:
            continue
        key = (
            None if row.symmetry_id is None else str(row.symmetry_id),
            str(row.chain_id),
            entity_id,
            number,
        )
        position_rows.setdefault(key, []).append(row)

    selected: dict[_MmcifPolymerPosition, tuple[str | None, str]] = {}
    for position, source_rows in position_rows.items():
        _, _, entity_id, number = position
        observed_components = {str(row.component_id) for row in source_rows}
        if not observed_components <= declared[(entity_id, number)]:
            unexpected = sorted(observed_components - declared[(entity_id, number)])
            raise ValueError(
                f"observed mmCIF components {unexpected!r} are not declared at "
                f"entity {entity_id!r} position {number}"
            )

        rows_by_label: dict[str, list[np.record]] = {}
        for row in source_rows:
            if row.altloc_id is not None:
                rows_by_label.setdefault(str(row.altloc_id), []).append(row)
        labels: dict[str, str] = {}
        for label, label_rows in rows_by_label.items():
            components = {str(row.component_id) for row in label_rows}
            if len(components) != 1:
                raise ValueError(f"mmCIF polymer alternate label {label!r} maps to multiple monomers at {position!r}")
            labels[label] = next(iter(components))

        if labels:
            if observed_components - set(labels.values()):
                raise ValueError(
                    f"mmCIF polymer microheterogeneity at {position!r} mixes labelled and unlabelled monomers"
                )
            choices = list(labels)
            if policy == "first":
                label = choices[0]
            else:
                label = max(
                    choices,
                    key=lambda candidate: _mean_present_occupancy([row.occupancy for row in rows_by_label[candidate]]),
                )
            selected[position] = (label, labels[label])
            continue

        rows_by_component: dict[str, list[np.record]] = {}
        for row in source_rows:
            rows_by_component.setdefault(str(row.component_id), []).append(row)
        choices = list(rows_by_component)
        if policy == "first":
            component_id = choices[0]
        else:
            component_id = max(
                choices,
                key=lambda candidate: _mean_present_occupancy([row.occupancy for row in rows_by_component[candidate]]),
            )
        selected[position] = (None, component_id)

    if not selected:
        return rows
    keep = []
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            keep.append(True)
            continue
        position = (
            None if row.symmetry_id is None else str(row.symmetry_id),
            str(row.chain_id),
            str(row.entity_id),
            int(row.label_seq_id),
        )
        realization = selected.get(position)
        if realization is None:
            keep.append(True)
            continue
        selected_label, component_id = realization
        row_label = None if row.altloc_id is None else str(row.altloc_id)
        keep.append(
            str(row.component_id) == component_id
            and (selected_label is None or row_label is None or row_label == selected_label)
        )
    return rows[np.asarray(keep, dtype=np.bool_)].view(np.recarray)


def _select_altloc(
    rows: np.recarray,
    policy: Literal["highest_occupancy", "first"],
) -> np.recarray:
    if not any(label is not None for label in rows.altloc_id):
        return rows

    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    occupancies_by_label: dict[tuple[tuple[object, ...], str], list[float]] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        residue_key = _altloc_position_key(row)
        label = str(row.altloc_id)
        labels = labels_by_residue.setdefault(residue_key, [])
        if label not in labels:
            labels.append(label)
        key = residue_key, label
        if row.occupancy is not None:
            occupancies_by_label.setdefault(key, []).append(float(row.occupancy))
    selected: dict[tuple[object, ...], str] = {}
    for residue_key, labels in labels_by_residue.items():
        if policy == "first":
            selected[residue_key] = labels[0]
        else:
            selected[residue_key] = max(
                labels,
                key=lambda label: _mean_present_occupancy(occupancies_by_label.get((residue_key, label), ())),
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
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[int]],
    rows: np.recarray,
) -> tuple[PolymerSequence, tuple[tuple[int, tuple[object, ...]], ...]] | None:
    if polymer_monomers is None:
        return None
    declared_monomers = polymer_monomers.get(entity_id)
    if not declared_monomers:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has no _entity_poly_seq declaration")

    observed: dict[int, str] = {}
    indexed: list[tuple[int, tuple[object, ...]]] = []
    label_seq_ids = rows["label_seq_id"]
    component_ids = rows["component_id"]
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
            )
        )
    links = _polymer_links(components, _observed_atom_ids(indexed, residue_rows, rows["atom_name"]))
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
    indexed: Sequence[tuple[int, tuple[object, ...]]],
    residue_rows: Mapping[tuple[object, ...], list[int]],
    atom_names: np.ndarray,
) -> dict[int, frozenset[str]]:
    r"""Collect the atom IDs each component occurrence actually contributes."""

    observed: dict[int, frozenset[str]] = {}
    for component_index, residue_key in indexed:
        observed[component_index] = frozenset(str(atom_names[position]) for position in residue_rows[residue_key])
    return observed


def _pdb_polymer_segments(
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[int]],
    serials: np.ndarray,
    segment_by_serial: Mapping[str, int],
) -> list[list[tuple[object, ...]]]:
    segments: list[list[tuple[object, ...]]] = []
    current: list[tuple[object, ...]] = []
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
    segment_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[int]],
    rows: np.recarray,
    component_table: Mapping[str, ChemicalComponent],
    declarations: tuple[tuple[str, str, int, str], ...],
    modified_residues: Mapping[tuple[str, int, str, str], str],
    *,
    declared_sequence: tuple[str, ...] | None = None,
    include_terminal: bool,
) -> tuple[
    PolymerSequence,
    tuple[tuple[int, tuple[object, ...]], ...],
    tuple[tuple[int, str, str], ...],
]:
    observed: list[tuple[tuple[int, str], tuple[object, ...], str]] = []
    auth_residue_ids = rows["auth_residue_id"]
    residue_ids = rows["residue_id"]
    insertion_codes = rows["insertion_code"]
    component_ids = rows["component_id"]
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
        declared_entries: list[tuple[tuple[int, str], tuple[object, ...] | None, str]] = [
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
            _observed_atom_ids(declared_indexed, residue_rows, rows["atom_name"]),
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
                    _observed_atom_ids(observed_indexed, residue_rows, rows["atom_name"]),
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
    links = _polymer_links(components, _observed_atom_ids(indexed, residue_rows, rows["atom_name"]))
    return PolymerSequence(polymer_type, tuple(components), links), tuple(indexed), tuple(unmodeled)


def _build_structure(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    source_rows: np.recarray,
    component_table: dict[str, ChemicalComponent],
    *,
    include_water: bool,
    include_declared_instances: bool,
    cif_file: object | None,
    pdb_lines: tuple[str, ...],
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...],
    pdb_declared_sequences: Mapping[str, tuple[str, ...]],
    source_component_parents: Mapping[str, str],
    mmcif_placements: Mapping[tuple[str | None, str], np.ndarray],
    mmcif_topology_placements: set[tuple[str, str, str | None]],
    pdb_assembly_placements: dict[str, tuple[np.ndarray, frozenset[str]]] | None,
    source_resolutions: tuple[tuple[float, int], ...],
    source_formal_charge_atoms: Mapping[str, frozenset[str]],
    source_non_leaving_atom_ids: Mapping[str, frozenset[str]],
    source_formal_charge_provenance: frozenset[AtomSite] | None,
    component_topology_complete: Mapping[str, bool],
    mmcif_entity_declarations: _MmcifEntityDeclarations | None,
) -> Structure:
    if not len(rows) and format_name == "pdb":
        raise ValueError("alternate-location selection contains no atoms")
    residue_rows: dict[tuple[object, ...], list[int]] = {}
    chain_residue_order: dict[tuple[str | None, str], list[tuple[object, ...]]] = {}
    symmetry_ids = rows["symmetry_id"]
    chain_ids = rows["chain_id"]
    residue_ids = rows["residue_id"]
    auth_residue_ids = rows["auth_residue_id"]
    insertion_codes = rows["insertion_code"]
    component_ids = rows["component_id"]
    entity_ids = rows["entity_id"]
    record_groups = rows["group"]
    serials = rows["serial"]
    atom_names = rows["atom_name"]
    label_seq_ids = rows["label_seq_id"]
    auth_chain_ids = rows["auth_chain_id"]
    auth_component_ids = rows["auth_component_id"]
    row_indices = rows["row_index"]
    for position in range(len(rows)):
        source_residue_key = (
            symmetry_ids[position],
            chain_ids[position],
            residue_ids[position],
            auth_residue_ids[position],
            insertion_codes[position],
            component_ids[position],
        )
        bucket = residue_rows.get(source_residue_key)
        if bucket is None:
            residue_rows[source_residue_key] = [position]
            chain_key = (symmetry_ids[position], str(chain_ids[position]))
            chain_residue_order.setdefault(chain_key, []).append(source_residue_key)
        else:
            bucket.append(position)

    entity_types: dict[str, str] = {}
    polymer_monomers: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None = None
    entity_poly_types: dict[str, str] = {}
    author_numbering: dict[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ] = {}
    asym_entities: dict[str, str] = {}
    nonpoly_components: dict[str, str] = {}
    nonpoly_occurrences: dict[
        str,
        tuple[tuple[str, str, str, str | None, str | None, str | None], ...],
    ] = {}
    segment_by_serial: dict[str, int] = {}
    pdb_entities: dict[str, str] = {}
    pdb_modified: dict[tuple[str, int, str, str], str] = {}
    branch_entities: dict[str, dict[str | None, Molecule]]
    if format_name == "mmcif":
        assert cif_file is not None
        assert mmcif_entity_declarations is not None
        from .mmcif import read_mmcif_branch_entities

        branch_entities, branch_component_indices, branch_source_bonds = read_mmcif_branch_entities(
            cif_file,
            component_table,
            rows,
        )
        branch_source_bonds_by_realization = {
            id(branch_entities[entity_id][label_asym_id]): bonds
            for (entity_id, label_asym_id), bonds in branch_source_bonds.items()
        }
        (
            declared_entity_types,
            polymer_monomers,
            entity_poly_types,
            author_numbering,
            asym_entities,
            nonpoly_components,
            nonpoly_occurrences,
        ) = mmcif_entity_declarations
        entity_types = dict(declared_entity_types)
        for entity_id in branch_entities:
            entity_type = entity_types.get(entity_id)
            if entity_type not in {None, "branched"}:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} is declared {entity_type!r} but also has a branched-entity definition"
                )
            entity_types[entity_id] = "branched"
        missing_branch_definition = {
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == "branched" and entity_id not in branch_entities
        }
        if missing_branch_definition:
            raise ValueError(
                f"mmCIF branched entities require branch declarations: {sorted(missing_branch_definition)!r}"
            )
        for raw_asym_id, raw_entity_id, raw_component_id in zip(
            rows["chain_id"],
            rows["entity_id"],
            rows["component_id"],
            strict=True,
        ):
            asym_id = str(raw_asym_id)
            source_entity_id = None if raw_entity_id is None else str(raw_entity_id)
            declared_entity_id = asym_entities.get(asym_id)
            if declared_entity_id is not None and declared_entity_id != source_entity_id:
                raise ValueError(f"mmCIF atom_site for asym {asym_id!r} disagrees with _struct_asym")
            declared_component_id = nonpoly_components.get(source_entity_id or "")
            if declared_component_id is not None and declared_component_id != str(raw_component_id):
                raise ValueError(
                    f"mmCIF atom_site for non-polymer entity {source_entity_id!r} disagrees with its component"
                )
    else:
        from .pdb import pdb_entity_by_chain, pdb_modified_residues, pdb_segment_by_serial

        branch_entities, branch_component_indices = {}, {}
        branch_source_bonds = {}
        branch_source_bonds_by_realization = {}
        segment_by_serial = pdb_segment_by_serial(pdb_lines)
        pdb_entities = pdb_entity_by_chain(pdb_lines)
        pdb_modified = pdb_modified_residues(pdb_lines)

    def branch_entity(entity_id: str, asym_id: str) -> Molecule:
        variants = branch_entities[entity_id]
        entity = variants.get(asym_id)
        if entity is None:
            entity = variants.get(None)
        if entity is None:
            raise ValueError(f"cannot select a branched-entity realization for entity {entity_id!r}, asym {asym_id!r}")
        return entity

    groups: list[
        tuple[
            int,
            str,
            str,
            PolymerSequence | Molecule,
            tuple[tuple[int, tuple[object, ...]], ...],
        ]
    ] = []
    source_asym_by_instance: dict[str, str] = {}
    used_instance_ids: set[str] = set()
    entities: dict[str, PolymerSequence | Molecule] = {}
    entity: PolymerSequence | Molecule
    matched_nonpoly_occurrences: set[tuple[str, str]] = set()
    unmodeled_numbers: dict[ComponentReference, tuple[str, str]] = {}
    standard_components: dict[ComponentReference, str] = {}
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

    chain_items = list(chain_residue_order.items())
    if format_name == "pdb":
        declared_pdb_chains: tuple[tuple[str | None, str], ...]
        if include_declared_instances:
            declared_pdb_chains = tuple((None, chain_id) for chain_id in pdb_declared_sequences)
        else:
            assert pdb_assembly_placements is not None
            declared_pdb_chains = tuple(
                (symmetry_id, chain_id)
                for symmetry_id, (_, chains) in pdb_assembly_placements.items()
                for chain_id in pdb_declared_sequences
                if chain_id in chains
            )
        chain_items.extend((placement, []) for placement in declared_pdb_chains if placement not in chain_residue_order)
    for (symmetry_id, chain_id), residue_order in chain_items:
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        declared_pdb_sequence = pdb_declared_sequences.get(chain_id) if format_name == "pdb" else None
        if format_name == "mmcif":
            polymer_residues = []
            for key in residue_order:
                position = residue_rows[key][0]
                source_entity_id = entity_ids[position]
                entity_type = None if source_entity_id is None else entity_types.get(str(source_entity_id))
                if entity_type == "polymer" or (
                    entity_type is None and get_builtin_component_polymer_type(str(component_ids[position])) is not None
                ):
                    polymer_residues.append(key)
            polymer_segments = [polymer_residues] if polymer_residues else []
        else:
            polymer_residues = []
            for key in residue_order:
                residue_positions = residue_rows[key]
                first_position = residue_positions[0]
                source_number = auth_residue_ids[first_position]
                if source_number is None:
                    source_number = residue_ids[first_position]
                if any(record_groups[position] == "ATOM" for position in residue_positions) or (
                    (
                        chain_id,
                        int(source_number),
                        str(insertion_codes[first_position] or ""),
                        str(component_ids[first_position]),
                    )
                    in pdb_modified
                ):
                    polymer_residues.append(key)
            polymer_segments = (
                [polymer_residues]
                if declared_pdb_sequence is not None
                else _pdb_polymer_segments(polymer_residues, residue_rows, serials, segment_by_serial)
            )

        if (
            format_name == "pdb"
            and declared_pdb_sequence is None
            and any(declared_chain == chain_id for declared_chain, *_ in pdb_missing_residues)
        ):
            raise ValueError(f"PDB REMARK 465 for chain {chain_id!r} requires a SEQRES declaration")

        segment_declarations = [pdb_missing_residues for _ in polymer_segments]
        if format_name == "pdb" and len(polymer_segments) > 1:
            assigned: list[list[tuple[str, str, int, str]]] = [[] for _ in polymer_segments]
            segment_positions: list[list[tuple[int, str]]] = []
            for segment_residues in polymer_segments:
                segment_position_values: list[tuple[int, str]] = []
                for residue_key in segment_residues:
                    first_position = residue_rows[residue_key][0]
                    number = auth_residue_ids[first_position]
                    if number is None:
                        number = residue_ids[first_position]
                    if number is None:
                        raise ValueError("PDB polymer residue lacks author numbering")
                    segment_position_values.append((int(number), str(insertion_codes[first_position] or "")))
                segment_positions.append(segment_position_values)
            for declaration in pdb_missing_residues:
                declared_chain, _, number, declared_insertion_code = declaration
                if declared_chain != chain_id:
                    continue
                declared_position = (number, declared_insertion_code)
                candidates = [
                    index
                    for index, segment_position_values in enumerate(segment_positions)
                    if declared_position in segment_position_values
                    or segment_position_values[0] < declared_position < segment_position_values[-1]
                ]
                if len(candidates) == 1:
                    assigned[candidates[0]].append(declaration)
            segment_declarations = [tuple(values) for values in assigned]

        for segment_ordinal, segment_residues in enumerate(polymer_segments):
            segment_first_position = None if not segment_residues else residue_rows[segment_residues[0]][0]
            source_entity_id = (
                None
                if segment_first_position is None or entity_ids[segment_first_position] is None
                else str(entity_ids[segment_first_position])
            )
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
                    rows,
                )
            if declared is not None:
                declared_entity, indexed = declared
                assert source_entity_id is not None
                entity_id = source_entity_id
                entities.setdefault(entity_id, declared_entity)
                entity = declared_entity
            else:
                classified_component_ids = (
                    declared_pdb_sequence
                    if declared_pdb_sequence is not None
                    else tuple(str(component_ids[residue_rows[key][0]]) for key in segment_residues)
                )
                classified_values: set[PolymerType | None] = set()
                for component_id in classified_component_ids:
                    if format_name == "mmcif":
                        classification = get_builtin_component_polymer_type(component_id)
                    else:
                        # PDB N is deliberately unresolved until the rest of the
                        # chain (or an observed sugar atom) establishes RNA vs DNA.
                        if component_id == "N":
                            continue
                        component = component_table.get(component_id)
                        if component is None and component_id not in {"I", "DI", "DU"}:
                            raise ValueError(
                                f"PDB SEQRES component {component_id!r} has no chemical definition; "
                                "supply it through components="
                            )
                        if component_id == "I":
                            classification = "rna"
                        elif component_id in {"DI", "DU"}:
                            classification = "dna"
                        else:
                            assert component is not None
                            classification = _pdb_component_polymer_type(component)
                        if classification is None:
                            parent_types: set[PolymerType] = set()
                            for (modified_chain, _, _, modified_id), parent_id in pdb_modified.items():
                                if modified_chain != chain_id or modified_id != component_id:
                                    continue
                                parent_type = get_builtin_component_polymer_type(parent_id)
                                if parent_type is not None:
                                    parent_types.add(parent_type)
                            if len(parent_types) > 1:
                                raise ValueError(
                                    f"PDB MODRES declarations give component {component_id!r} conflicting polymer types"
                                )
                            if parent_types:
                                classification = next(iter(parent_types))
                    classified_values.add(classification)
                classified = {polymer_type for polymer_type in classified_values if polymer_type is not None}
                if format_name == "pdb" and "N" in classified_component_ids:
                    if classified - {"dna", "rna"}:
                        raise ValueError(
                            f"PDB chain {base_instance_id!r} mixes nucleotide N with a non-nucleic polymer"
                        )
                    if classified == {"dna", "rna"}:
                        raise ValueError(
                            f"PDB chain {base_instance_id!r} uses N in a DNA/RNA hybrid without identifying its sugar"
                        )
                    if not classified:
                        n_atom_ids = {
                            str(atom_names[position])
                            for key in segment_residues
                            for position in residue_rows[key]
                            if str(component_ids[position]) == "N"
                        }
                        rna_identified = bool({"O2'", "HO2'"} & n_atom_ids)
                        dna_identified = "H2''" in n_atom_ids
                        if rna_identified == dna_identified:
                            raise ValueError(f"PDB chain {base_instance_id!r} contains only ambiguous N nucleotides")
                        classified.add("rna" if rna_identified else "dna")
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
                elif format_name == "pdb" and chain_id in pdb_entities:
                    entity_id = pdb_entities[chain_id]
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
                        rows,
                        component_table,
                        segment_declarations[segment_ordinal],
                        pdb_modified,
                        declared_sequence=declared_pdb_sequence,
                        include_terminal=len(polymer_segments) == 1,
                    )
                else:
                    resolved = PolymerSequence(
                        chain_polymer_type,
                        tuple(component_table[str(component_ids[residue_rows[key][0]])] for key in segment_residues),
                    )
                    indexed = tuple(enumerate(segment_residues))
                if format_name == "pdb" and chain_id in pdb_entities:
                    entities.setdefault(entity_id, resolved)
                    entity = resolved
                else:
                    entity = _reuse_entity(entities, entity_id, resolved)

            instance_id = unique_identifier(base_instance_id, used_instance_ids)
            if format_name == "mmcif":
                source_asym_by_instance[instance_id] = chain_id
            if declared is not None:
                observed_indices = {index for index, _ in indexed}
                for component_index, component in enumerate(declared_entity.components):
                    if component_index in observed_indices:
                        continue
                    author = _source_author_numbering(
                        author_numbering,
                        chain_id,
                        component_index,
                        component.component_id,
                    )
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
            if format_name == "pdb":
                source_positions: dict[int, tuple[int, str]] = {
                    component_index: (int(residue_id), insertion_code)
                    for component_index, residue_id, insertion_code in pdb_unmodeled
                }
                source_component_ids = (
                    dict(enumerate(declared_pdb_sequence)) if declared_pdb_sequence is not None else {}
                )
                for component_index, residue_key in indexed:
                    first_position = residue_rows[residue_key][0]
                    residue_number = auth_residue_ids[first_position]
                    if residue_number is None:
                        residue_number = residue_ids[first_position]
                    if residue_number is None:
                        raise ValueError("PDB polymer residue lacks author numbering")
                    source_positions[component_index] = (
                        int(residue_number),
                        str(insertion_codes[first_position] or ""),
                    )
                    source_component_ids[component_index] = str(component_ids[first_position])
                for component_index, (residue_number, source_insertion) in source_positions.items():
                    component = entity.components[component_index]
                    source_component_id = source_component_ids.get(component_index, component.component_id)
                    standard_component = pdb_modified.get(
                        (chain_id, residue_number, source_insertion, source_component_id)
                    )
                    if standard_component is not None:
                        standard_components[(instance_id, component_index)] = standard_component
                    for atom in component.atoms:
                        site = AtomSite(instance_id, component_index, atom.atom_id)
                        source_key = (
                            symmetry_id,
                            chain_id or None,
                            residue_number,
                            source_insertion or None,
                            source_component_id,
                            atom.atom_id,
                        )
                        add_unmodeled_site(source_key, site)
                        canonical_key = source_key[:4] + (component.component_id, atom.atom_id)
                        if canonical_key != source_key:
                            add_unmodeled_site(canonical_key, site)
            groups.append(
                (
                    (
                        min(int(row_indices[position]) for key in segment_residues for position in residue_rows[key])
                        if segment_residues
                        else len(source_rows) + segment_ordinal
                    ),
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
                source_entity_id = entity_ids[residue_rows[residue_key][0]]
                if source_entity_id in branch_entities:
                    grouped_by_entity.setdefault(str(source_entity_id), []).append(residue_key)
                else:
                    molecule_groups.append((residue_key,))
            molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
            molecule_groups.sort(
                key=lambda group: min(int(row_indices[position]) for key in group for position in residue_rows[key])
            )
        else:
            molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)

        for group_residues in molecule_groups:
            first_position = residue_rows[group_residues[0]][0]
            first_component = component_table[str(component_ids[first_position])]
            residue_label = auth_residue_ids[first_position]
            if residue_label is None:
                residue_label = residue_ids[first_position]
            residue_identity = f"{residue_label}{insertion_codes[first_position] or ''}"
            source_entity_id = None if entity_ids[first_position] is None else str(entity_ids[first_position])
            if source_entity_id in branch_entities:
                entity = branch_entity(source_entity_id, chain_id)
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues: list[tuple[int, tuple[object, ...]]] = []
                for residue_key in group_residues:
                    position = residue_rows[residue_key][0]
                    branch_component_index: int | None = None
                    source_keys = (
                        (
                            chain_ids[position],
                            entity_ids[position],
                            component_ids[position],
                            chain_ids[position],
                            None,
                            insertion_codes[position],
                        ),
                        (
                            chain_ids[position],
                            entity_ids[position],
                            auth_component_ids[position],
                            auth_chain_ids[position],
                            auth_residue_ids[position],
                            insertion_codes[position],
                        ),
                    )
                    for branch_lookup_key in source_keys:
                        branch_component_index = branch_component_indices.get(branch_lookup_key)
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
                if source_entity_type is not None and source_entity_type not in _MMCIF_MOLECULE_TYPES:
                    raise ValueError(
                        f"mmCIF entity {source_entity_id!r} is declared {source_entity_type!r} "
                        "but observed as a molecule"
                    )
                source_number = (
                    None if auth_residue_ids[first_position] is None else str(auth_residue_ids[first_position])
                )
                source_insertion = str(insertion_codes[first_position] or "")
                source_chain = str(auth_chain_ids[first_position] or chain_ids[first_position])
                source_monomers = {
                    str(component_ids[first_position]),
                    str(auth_component_ids[first_position] or component_ids[first_position]),
                }
                declared_occurrences = nonpoly_occurrences.get(str(chain_ids[first_position]), ())
                occurrence_matches = [
                    ndb_code
                    for ndb_code, occurrence_entity, monomer, author_number, insertion, pdb_chain in (
                        declared_occurrences
                    )
                    if occurrence_entity == source_entity_id
                    and monomer in source_monomers
                    and (author_number is None or source_number == author_number)
                    and source_insertion == (insertion or "")
                    and (pdb_chain is None or pdb_chain == source_chain)
                ]
                if len(occurrence_matches) > 1:
                    raise ValueError(f"cannot uniquely match non-polymer occurrence {residue_identity!r}")
                if declared_occurrences and not occurrence_matches:
                    raise ValueError(f"cannot match non-polymer occurrence {residue_identity!r} to its declaration")
                if occurrence_matches:
                    matched_nonpoly_occurrences.add((str(chain_ids[first_position]), occurrence_matches[0]))
                molecule_type = (
                    _MMCIF_MOLECULE_TYPES[source_entity_type]
                    if source_entity_type is not None
                    else ("water" if is_water_component(first_component) else "non-polymer")
                )
                entity = Molecule(
                    (first_component,),
                    molecule_type=molecule_type,
                )
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues = [(0, group_residues[0])]
            else:
                entity = Molecule(
                    (first_component,),
                    molecule_type=("water" if is_water_component(first_component) else "non-polymer"),
                )
                entity_id = (
                    f"molecule_{first_component.component_id}"
                    if format_name == "pdb"
                    else f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
                )
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            if source_entity_id in branch_entities:
                entities.setdefault(entity_id, entity)
            else:
                entity = _reuse_entity(entities, entity_id, entity)
            instance_id = unique_identifier(instance_seed, used_instance_ids)
            if format_name == "mmcif":
                source_asym_by_instance[instance_id] = chain_id
            groups.append(
                (
                    min(int(row_indices[position]) for key in group_residues for position in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(indexed_residues),
                )
            )

    nonpoly_declaration_placements: tuple[tuple[str | None, str, str | None], ...]
    if format_name == "mmcif":
        nonpoly_declaration_placements = (
            (
                *tuple((None, asym_id, None) for asym_id in nonpoly_occurrences),
                *tuple(sorted(placement for placement in mmcif_topology_placements if placement[2] is not None)),
            )
            if include_declared_instances
            else tuple(
                (symmetry_id, asym_id, None)
                for symmetry_id, asym_id in mmcif_placements
                if asym_id in nonpoly_occurrences
            )
        )
    else:
        nonpoly_declaration_placements = ()
    if nonpoly_declaration_placements:
        declaration_index = 0
        for symmetry_id, asym_id, target_ndb_code in nonpoly_declaration_placements:
            occurrences = nonpoly_occurrences[asym_id]
            for ndb_code, source_entity_id, monomer, author_number, insertion, pdb_chain in occurrences:
                if target_ndb_code is not None and ndb_code != target_ndb_code:
                    continue
                if (
                    target_ndb_code is None
                    and (asym_id, ndb_code) in matched_nonpoly_occurrences
                    and (symmetry_id, asym_id) in chain_residue_order
                ):
                    continue
                source_entity_type = entity_types.get(source_entity_id)
                if not include_water and source_entity_type == "water":
                    continue
                if source_entity_type is not None and source_entity_type not in _MMCIF_MOLECULE_TYPES:
                    raise ValueError(
                        f"mmCIF entity {source_entity_id!r} is declared {source_entity_type!r} but used as a molecule"
                    )
                component = _resolve_declared_component(
                    monomer,
                    component_table,
                )
                if not include_water and source_entity_type is None and is_water_component(component):
                    continue
                molecule_type = (
                    _MMCIF_MOLECULE_TYPES[source_entity_type]
                    if source_entity_type is not None
                    else ("water" if is_water_component(component) else "non-polymer")
                )
                entity = _reuse_entity(
                    entities,
                    source_entity_id,
                    Molecule(
                        (component,),
                        molecule_type=molecule_type,
                    ),
                )
                residue_identity = author_number or ndb_code
                instance_seed = asym_id if symmetry_id is None else f"{asym_id}#{symmetry_id}"
                occurrence_id = (
                    f"{instance_seed}:{residue_identity}:{monomer}" if len(occurrences) > 1 else instance_seed
                )
                instance_id = unique_identifier(occurrence_id, used_instance_ids)
                source_asym_by_instance[instance_id] = asym_id
                if author_number is not None:
                    unmodeled_numbers[(instance_id, 0)] = (author_number, insertion or "")
                for atom in component.atoms:
                    site = AtomSite(instance_id, 0, atom.atom_id)
                    add_unmodeled_site(
                        (symmetry_id, asym_id, None, insertion, monomer, atom.atom_id),
                        site,
                    )
                    add_unmodeled_site(
                        (symmetry_id, pdb_chain or asym_id, author_number, insertion, monomer, atom.atom_id),
                        site,
                    )
                groups.append(
                    (
                        len(source_rows) + declaration_index,
                        instance_id,
                        source_entity_id,
                        entity,
                        (),
                    )
                )
                declaration_index += 1

    observed_asym_placements = set(chain_residue_order)
    declared_instances: tuple[tuple[str | None, str, str], ...]
    if format_name == "mmcif":
        if include_declared_instances:
            values: list[tuple[str | None, str, str]] = [
                (None, asym_id, entity_id) for asym_id, entity_id in asym_entities.items()
            ]
            values.extend(
                (symmetry_id, asym_id, asym_entities[asym_id])
                for symmetry_id, asym_id, ndb_code in sorted(mmcif_topology_placements)
                if ndb_code is None
            )
            declared_instances = tuple(values)
        else:
            declared_instances_list: list[tuple[str | None, str, str]] = []
            for symmetry_id, asym_id in mmcif_placements:
                source_entity_id = asym_entities.get(asym_id)
                if source_entity_id is None:
                    raise ValueError(f"mmCIF assembly references undeclared asym {asym_id!r}")
                declared_instances_list.append((symmetry_id, asym_id, source_entity_id))
            declared_instances = tuple(declared_instances_list)
    else:
        declared_instances = ()
    for declaration_index, (symmetry_id, asym_id, source_entity_id) in enumerate(declared_instances):
        if (symmetry_id, asym_id) in observed_asym_placements or asym_id in nonpoly_occurrences:
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
                rows,
            )
            if declared is None:
                raise ValueError(f"unobserved mmCIF polymer entity {source_entity_id!r} has no sequence declaration")
            resolved, _ = declared
            entity_id = source_entity_id
            entities.setdefault(entity_id, resolved)
            entity = resolved
        elif source_entity_id in branch_entities:
            entity_id = source_entity_id
            entity = branch_entity(source_entity_id, asym_id)
            entities.setdefault(entity_id, entity)
        else:
            declared_component_id = nonpoly_components.get(source_entity_id)
            if declared_component_id is None:
                if source_entity_type in _MMCIF_MOLECULE_TYPES:
                    raise ValueError(
                        f"unobserved mmCIF {source_entity_type} entity {source_entity_id!r} "
                        "has no component declaration"
                    )
                continue
            component = _resolve_declared_component(
                declared_component_id,
                component_table,
            )
            inferred_entity_type = (
                "water"
                if source_entity_type is None and is_water_component(component)
                else source_entity_type or "non-polymer"
            )
            if not include_water and inferred_entity_type == "water":
                continue
            entity_id = source_entity_id
            entity = _reuse_entity(
                entities,
                entity_id,
                Molecule(
                    (component,),
                    molecule_type=_MMCIF_MOLECULE_TYPES[inferred_entity_type],
                ),
            )

        instance_seed = asym_id if symmetry_id is None else f"{asym_id}#{symmetry_id}"
        instance_id = unique_identifier(instance_seed, used_instance_ids)
        source_asym_by_instance[instance_id] = asym_id
        for component_index, component in enumerate(entity.components):
            author = _source_author_numbering(
                author_numbering,
                asym_id,
                component_index,
                component.component_id,
            )
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
                        symmetry_id,
                        asym_id,
                        label_sequence,
                        insertion,
                        component.component_id,
                        atom.atom_id,
                    ),
                    site,
                )
                for branch_source_key, branch_index in branch_component_indices.items():
                    (
                        branch_label_asym,
                        branch_entity_id,
                        branch_monomer,
                        branch_auth_asym,
                        branch_auth_sequence,
                        branch_auth_insertion,
                    ) = branch_source_key
                    if (
                        branch_index != component_index
                        or branch_label_asym != asym_id
                        or branch_entity_id != source_entity_id
                    ):
                        continue
                    add_unmodeled_site(
                        (
                            symmetry_id,
                            branch_label_asym,
                            None,
                            branch_auth_insertion,
                            branch_monomer,
                            atom.atom_id,
                        ),
                        site,
                    )
                    add_unmodeled_site(
                        (
                            symmetry_id,
                            branch_auth_asym,
                            branch_auth_sequence,
                            branch_auth_insertion,
                            branch_monomer,
                            atom.atom_id,
                        ),
                        site,
                    )
                add_unmodeled_site(
                    (
                        symmetry_id,
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
    if not groups:
        raise ValueError("selected structure contains no molecular instances")
    instances = {instance_id: entity_id for _, instance_id, entity_id, _, _ in groups}
    instance_values = {instance_id: entity for _, instance_id, _, entity, _ in groups}
    site_by_row_index: dict[int, AtomSite] = {}
    residue_numbers: dict[ComponentReference, tuple[str, str]] = dict(unmodeled_numbers)
    for _, instance_id, _, entity, indexed in groups:
        for component_index, residue_key in indexed:
            residue_positions = residue_rows[residue_key]
            first_position = residue_positions[0]
            source_number = auth_residue_ids[first_position]
            if format_name == "pdb" and source_number is None:
                source_number = residue_ids[first_position]
            source_insertion = str(insertion_codes[first_position] or "")
            if format_name == "mmcif":
                author = _source_author_numbering(
                    author_numbering,
                    str(chain_ids[first_position]),
                    component_index,
                    entity.components[component_index].component_id,
                )
                if author is not None:
                    if source_number is None:
                        source_number = author[0]
                    if insertion_codes[first_position] is None:
                        source_insertion = author[1] or ""
            if source_number is not None:
                residue_numbers[(instance_id, component_index)] = (
                    str(source_number),
                    source_insertion,
                )
            if format_name == "mmcif":
                component = entity.components[component_index]
                label_prefix = (
                    symmetry_ids[first_position],
                    chain_ids[first_position],
                    label_seq_ids[first_position],
                    insertion_codes[first_position],
                    component.component_id,
                )
                auth_prefix = (
                    symmetry_ids[first_position],
                    auth_chain_ids[first_position],
                    auth_residue_ids[first_position],
                    insertion_codes[first_position],
                    auth_component_ids[first_position] or component.component_id,
                )
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    add_unmodeled_site((*label_prefix, atom.atom_id), site)
                    add_unmodeled_site((*auth_prefix, atom.atom_id), site)
            for position in residue_positions:
                site_by_row_index[int(row_indices[position])] = AtomSite(
                    instance_id,
                    component_index,
                    str(atom_names[position]),
                )
        if format_name == "mmcif" and isinstance(entity, Molecule) and indexed:
            first_position = residue_rows[indexed[0][1]][0]
            source_entity_id = entity_ids[first_position]
            observed_components = {component_index for component_index, _ in indexed}
            for branch_source_key, component_index in branch_component_indices.items():
                (
                    source_label_asym_id,
                    source_branch_entity_id,
                    source_monomer_id,
                    source_auth_asym_id,
                    source_auth_number,
                    source_auth_insertion,
                ) = branch_source_key
                if (
                    source_branch_entity_id != source_entity_id
                    or source_label_asym_id != chain_ids[first_position]
                    or component_index in observed_components
                ):
                    continue
                author = author_numbering.get((str(source_label_asym_id), component_index + 1))
                branch_insertion_code = None if source_auth_insertion is None else str(source_auth_insertion)
                insertion = branch_insertion_code if author is None else author[1] or branch_insertion_code
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
                            symmetry_ids[first_position],
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
                            symmetry_ids[first_position],
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

        source_bonds = read_pdb_bonds(
            pdb_lines,
            source_rows,
            rows,
            site_by_row_index,
            unmodeled_source_sites,
            ambiguous_unmodeled_sites,
            component_table,
            include_water=include_water,
            assembly_placements=pdb_assembly_placements,
        )
    else:
        assert cif_file is not None
        from .mmcif import read_mmcif_bonds

        source_bonds = read_mmcif_bonds(
            cif_file,
            source_rows,
            rows,
            site_by_row_index,
            unmodeled_source_sites,
            ambiguous_unmodeled_sites,
            component_table,
            placements=mmcif_placements,
            component_topology_complete=component_topology_complete,
            source_non_leaving_atom_ids=source_non_leaving_atom_ids,
            declared_branch_bonds={
                frozenset((site_a, site_b)): (
                    SystemBond(
                        site_a,
                        site_b,
                        order=bond.order,
                        aromatic=bond.aromatic,
                        stereo=bond.stereo,
                        stereo_sites=(
                            None
                            if bond.stereo_atoms is None
                            else (
                                AtomSite(instance_id, *bond.stereo_atoms[0]),
                                AtomSite(instance_id, *bond.stereo_atoms[1]),
                            )
                        ),
                        role="entity",
                        leaving_a=bond.leaving_a,
                        leaving_b=bond.leaving_b,
                        kind=bond.kind,
                        direction=bond.direction,
                    ),
                    leaving_a_known,
                    leaving_b_known,
                )
                for instance_id, entity in instance_values.items()
                if isinstance(entity, Molecule) and entity.molecule_type == "branched"
                for bond, leaving_a_known, leaving_b_known in branch_source_bonds_by_realization[id(entity)]
                for site_a, site_b in (
                    (
                        AtomSite(instance_id, bond.component_a, bond.atom_a),
                        AtomSite(instance_id, bond.component_b, bond.atom_b),
                    ),
                )
            },
        )
        lowered_bonds: list[SystemBond] = []
        for bond in source_bonds:
            bond_entity = instance_values.get(bond.site_a.instance_id)
            if (
                bond.kind == "covalent"
                and bond.site_a.instance_id == bond.site_b.instance_id
                and isinstance(bond_entity, PolymerSequence)
                and abs(bond.site_a.component_index - bond.site_b.component_index) == 1
                and bond.order in {None, 1.0}
                and bond.aromatic in {None, False}
            ):
                if bond.site_a.component_index < bond.site_b.component_index:
                    left_site, right_site = bond.site_a, bond.site_b
                    left_leaving, right_leaving = bond.leaving_a, bond.leaving_b
                else:
                    left_site, right_site = bond.site_b, bond.site_a
                    left_leaving, right_leaving = bond.leaving_b, bond.leaving_a
                left = bond_entity.components[left_site.component_index]
                right = bond_entity.components[right_site.component_index]
                left_attachment = left.next_attachment
                right_attachment = right.previous_attachment
                if (
                    not bond_entity.is_linked(left_site.component_index)
                    and left_attachment is not None
                    and right_attachment is not None
                    and left_attachment.atom_id == left_site.atom_id
                    and right_attachment.atom_id == right_site.atom_id
                    and len(left_leaving) == len(left_attachment.leaving_atom_ids)
                    and len(right_leaving) == len(right_attachment.leaving_atom_ids)
                ):
                    bond = replace(bond, order=1.0, aromatic=False, role="polymer")
            lowered_bonds.append(bond)
        source_bonds = tuple(lowered_bonds)
    system = MolecularSystem.from_entities(
        entities,
        instances,
        source_bonds,
        instance_values=instance_values,
    )

    def source_atom_site(atom_index: int) -> AtomSite:
        site = system.atom_site(atom_index)
        return AtomSite(
            source_asym_by_instance[site.instance_id],
            site.component_index,
            site.atom_id,
        )

    if source_formal_charge_provenance is not None:
        realized_source_sites = {source_atom_site(atom_index) for atom_index in range(int(system.atom_symbol.numel()))}
        missing_provenance = source_formal_charge_provenance - realized_source_sites
        if not include_water:
            filtered_water_asym_ids = {
                asym_id for asym_id, entity_id in asym_entities.items() if entity_types.get(entity_id) == "water"
            }
            missing_provenance = frozenset(
                site for site in missing_provenance if site.instance_id not in filtered_water_asym_ids
            )
        if missing_provenance:
            site = min(missing_provenance, key=repr)
            raise ValueError(f"mmCIF _ofoldx_source_formal_charge references unrealized atom {site!r}")

    symbols = system.symbols
    instance_symbol = system.instance_symbol
    instance_component_ptr = system.instance_component_ptr
    component_symbol = system.component_symbol
    component_atom_ptr = system.component_atom_ptr
    atom_symbol = system.atom_symbol
    atomic_number = system.atomic_number
    isotope = system.isotope
    isotope_known = system.isotope_known
    formal_charge = system.formal_charge
    formal_charge_known = system.formal_charge_known

    atom_count = int(atomic_number.shape[0])
    component_count = int(component_symbol.shape[0])
    component_index_by_reference: dict[ComponentReference, int] = {}
    component_references: list[ComponentReference] = [("", 0)] * component_count
    atom_index_by_site: dict[AtomSite, int] = {}
    for instance_index in range(int(instance_symbol.shape[0])):
        instance_id = symbols[int(instance_symbol[instance_index])]
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        for component_index in range(component_start, component_stop):
            local_component = component_index - component_start
            reference = instance_id, local_component
            component_index_by_reference[reference] = component_index
            component_references[component_index] = reference
            atom_start = int(component_atom_ptr[component_index])
            atom_stop = int(component_atom_ptr[component_index + 1])
            for atom_index in range(atom_start, atom_stop):
                atom_index_by_site[
                    AtomSite(
                        instance_id,
                        local_component,
                        symbols[int(atom_symbol[atom_index])],
                    )
                ] = atom_index
    coordinates = torch.zeros((atom_count, 3), dtype=torch.float32)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    occupancy = torch.zeros(atom_count, dtype=torch.float32)
    occupancy_mask = torch.zeros(atom_count, dtype=torch.bool)
    b_factor = torch.zeros(atom_count, dtype=torch.float32)
    b_factor_mask = torch.zeros(atom_count, dtype=torch.bool)
    atom_indices = [atom_index_by_site[site_by_row_index[int(row_index)]] for row_index in rows["row_index"]]
    seen_atom_indices: set[int] = set()
    for atom_index in atom_indices:
        if atom_index in seen_atom_indices:
            raise ValueError(f"selected source rows map more than once to {system.atom_site(atom_index)!r}")
        seen_atom_indices.add(atom_index)
    source_atom_index = torch.tensor(atom_indices, dtype=torch.int64)
    source_atomic_number = torch.tensor(
        [ATOMIC_NUMBER_BY_ELEMENT[str(element)] for element in rows["element"]],
        dtype=atomic_number.dtype,
    )
    element_mismatch = atomic_number[source_atom_index].cpu() != source_atomic_number
    if bool(element_mismatch.any()):
        source_index = int(torch.nonzero(element_mismatch, as_tuple=False)[0])
        site = system.atom_site(atom_indices[source_index])
        raise ValueError(f"source element {rows['element'][source_index]!r} disagrees with component atom at {site!r}")
    source_isotope_mask = torch.tensor(
        [value is not None for value in rows["isotope"]],
        dtype=torch.bool,
    )
    isotope[source_atom_index[source_isotope_mask]] = torch.tensor(
        [int(value) for value in rows["isotope"] if value is not None],
        dtype=isotope.dtype,
    )
    isotope_known[source_atom_index[source_isotope_mask]] = True
    source_charge_mask = torch.tensor(
        [value is not None for value in rows["charge"]],
        dtype=torch.bool,
    )
    source_formal_charge_mask = torch.zeros(atom_count, dtype=torch.bool)
    for component_index, reference in enumerate(component_references):
        component_id = symbols[int(component_symbol[component_index])]
        for atom_id in source_formal_charge_atoms.get(component_id, ()):
            source_atom = atom_index_by_site.get(AtomSite(*reference, atom_id))
            if source_atom is not None:
                source_formal_charge_mask[source_atom] = True
    source_formal_charge_mask[source_atom_index[source_charge_mask]] = True
    if source_formal_charge_provenance is not None:
        source_formal_charge_mask = torch.tensor(
            [source_atom_site(index) in source_formal_charge_provenance for index in range(atom_count)],
            dtype=torch.bool,
        )
    formal_charge[source_atom_index[source_charge_mask]] = torch.tensor(
        [int(value) for value in rows["charge"] if value is not None],
        dtype=formal_charge.dtype,
    )
    formal_charge_known[source_atom_index[source_charge_mask]] = True

    source_coordinate_mask = torch.from_numpy(np.asarray(rows["coordinate_present"], dtype=np.bool_))
    coordinate_mask[source_atom_index] = source_coordinate_mask
    coordinates[source_atom_index] = torch.from_numpy(
        np.array(rows["coordinates"], dtype=np.float32, copy=True, order="C")
    )
    source_occupancy_mask = torch.tensor(
        [value is not None for value in rows["occupancy"]],
        dtype=torch.bool,
    )
    occupancy_mask[source_atom_index] = source_occupancy_mask
    occupancy[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows["occupancy"]],
        dtype=torch.float32,
    )
    source_b_factor_mask = torch.tensor(
        [value is not None for value in rows["b_factor"]],
        dtype=torch.bool,
    )
    b_factor_mask[source_atom_index] = source_b_factor_mask
    b_factor[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows["b_factor"]],
        dtype=torch.float32,
    )
    if not bool(coordinate_mask.any()) and format_name == "pdb":
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

    residue_id_values = torch.full((component_count,), -1, dtype=torch.int64)
    insertion_code_values = torch.full((component_count,), -1, dtype=torch.int64)
    standard_component_ids = torch.full((component_count,), -1, dtype=torch.int64)
    for reference, (number, code) in residue_numbers.items():
        component_index = component_index_by_reference[reference]
        residue_id_values[component_index] = intern(number)
        if code:
            insertion_code_values[component_index] = intern(code)
    for component_index in range(component_count):
        reference = component_references[component_index]
        standard_component = standard_components.get(reference)
        if standard_component is None:
            component_id = symbols[int(component_symbol[component_index])]
            standard_component = source_component_parents.get(component_id)
        if standard_component is not None:
            standard_component_ids[component_index] = intern(standard_component)
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=tuple(source_symbols),
        residue_id=residue_id_values,
        insertion_code=insertion_code_values,
        standard_component_id=standard_component_ids,
        source_formal_charge_mask=source_formal_charge_mask,
        system_resolution_ptr=torch.tensor(
            [0, len(source_resolutions)],
            dtype=torch.int64,
            device=coordinates.device,
        ),
        resolution=torch.tensor(
            [value for value, _ in source_resolutions],
            dtype=torch.float64,
            device=coordinates.device,
        ),
        resolution_kind=torch.tensor(
            [kind for _, kind in source_resolutions],
            dtype=torch.uint8,
            device=coordinates.device,
        ),
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
