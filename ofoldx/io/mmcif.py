r"""Generic mmCIF codec for canonical [`ofoldx.data.Structure`][] values."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from ..data.codec import decode_structure, encode_structure
from ..data.molecular import AtomSite, ChemicalComponent, Molecule, MoleculeBond, PolymerSequence, SystemBond
from ..data.structure import Structure
from .records import (
    AtomRecord,
    ParsedStructure,
    atom_records_from_array,
    optional_integer,
    ordered_output_bonds,
    present_string,
    reject_duplicate_atom_records,
    selected_source_sites,
    unique_identifier,
)


class _CifColumn(Protocol):
    def as_array(self) -> Iterable[object]: ...


class _CifCategory(Protocol):
    row_count: int

    def __contains__(self, name: object) -> bool: ...

    def __getitem__(self, name: str) -> _CifColumn: ...


class _CifBlock(Protocol):
    def __contains__(self, name: object) -> bool: ...

    def __getitem__(self, name: str) -> _CifCategory: ...


class _CifFile(Protocol):
    block: _CifBlock


_CIF_BOND_ORDERS = {
    "sing": (1.0, False),
    "doub": (2.0, False),
    "trip": (3.0, False),
    "quad": (4.0, False),
    "arom": (1.5, True),
}


def _expect_object(value: object, owner: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{owner} keys must be strings")
    return cast(dict[str, object], value)


def parse_mmcif(path: Path, *, model: int, assembly_id: str | None) -> ParsedStructure:
    r"""Parse one mmCIF model or assembly into format-neutral atom records."""

    from biotite.structure.io import pdbx

    cif_file = pdbx.CIFFile.read(str(path))
    block = cif_file.block
    canonical_payload: dict[str, object] | None = None
    if "ofoldx_structure" in block:
        category = block["ofoldx_structure"]
        if "json" not in category or category.row_count != 1:
            raise ValueError("_ofoldx_structure must contain exactly one json value")
        raw_payload = str(category["json"].as_array()[0])
        decoded_payload = json.loads(raw_payload)
        if not isinstance(decoded_payload, dict):
            raise ValueError("_ofoldx_structure.json must decode to an object")
        canonical_payload = decoded_payload
        if model != 1:
            raise ValueError("an OFoldX canonical mmCIF exchange view contains model 1 only")
        if assembly_id is not None:
            raise ValueError("an OFoldX canonical mmCIF exchange view contains one materialized assembly")

    atom_site = block.get("atom_site")
    if atom_site is None:
        raise ValueError("mmCIF input has no atom_site category")
    available = set(atom_site.keys())
    requested = [
        field
        for field in (
            "auth_asym_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_atom_id",
            "label_entity_id",
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
            "ofoldx_instance_id",
            "ofoldx_component_index",
            "ofoldx_atom_id",
            "ofoldx_coordinate_mask",
        )
        if field in available
    ]
    for special, source_column in (
        ("atom_id", "id"),
        ("b_factor", "B_iso_or_equiv"),
        ("occupancy", "occupancy"),
        ("charge", "pdbx_formal_charge"),
    ):
        if source_column in available:
            requested.append(special)
    try:
        if assembly_id is None:
            array = pdbx.get_structure(
                cif_file,
                model=model,
                altloc="all",
                extra_fields=requested,
                use_author_fields=False,
                include_bonds=False,
            )
        else:
            assemblies = pdbx.list_assemblies(cif_file)
            if assembly_id not in assemblies:
                raise ValueError(f"unknown mmCIF assembly_id {assembly_id!r}; available={sorted(assemblies)!r}")
            array = pdbx.get_assembly(
                cif_file,
                assembly_id=assembly_id,
                model=model,
                altloc="all",
                extra_fields=requested,
                use_author_fields=False,
                include_bonds=False,
            )
    except IndexError as error:
        raise ValueError(f"mmCIF does not contain requested model {model}") from error
    rows = atom_records_from_array(array, format_name="mmcif")
    reject_duplicate_atom_records(rows)
    return ParsedStructure(
        format="mmcif",
        rows=rows,
        canonical_payload=canonical_payload,
        mmcif_file=cif_file,
    )


def decode_canonical_mmcif(parsed: ParsedStructure) -> Structure:
    r"""Decode and verify an OFoldX canonical mmCIF exchange payload."""

    container = parsed.canonical_payload
    if container is None:
        raise ValueError("canonical mmCIF payload is missing")
    if set(container) != {"metadata", "arrays"}:
        raise ValueError("canonical mmCIF payload must contain metadata and arrays")
    payload = _expect_object(container["metadata"], "canonical mmCIF metadata")
    array_values = _expect_object(container["arrays"], "canonical mmCIF arrays")
    rows = tuple(sorted(parsed.rows, key=lambda row: row.row_index))
    coordinates_key = payload.get("coordinates")
    coordinate_mask_key = payload.get("coordinate_mask")
    occupancy_key = payload.get("occupancy")
    if not isinstance(coordinates_key, str) or not coordinates_key:
        raise ValueError("canonical mmCIF payload has invalid coordinates array reference")
    if not isinstance(coordinate_mask_key, str) or not coordinate_mask_key:
        raise ValueError("canonical mmCIF payload has invalid coordinate_mask array reference")
    coordinate_dtype_value = payload.get("coordinates_dtype")
    if not isinstance(coordinate_dtype_value, str):
        raise ValueError("canonical mmCIF coordinate dtype is missing")
    coordinate_dtype = np.dtype(coordinate_dtype_value)
    occupancy_dtype_value = payload.get("occupancy_dtype")
    occupancy_dtype = None if occupancy_dtype_value is None else np.dtype(occupancy_dtype_value)
    arrays: dict[str, np.ndarray] = {
        coordinates_key: np.asarray(array_values[coordinates_key], dtype=coordinate_dtype),
        coordinate_mask_key: np.asarray(array_values[coordinate_mask_key], dtype=np.bool_),
    }
    if occupancy_key is not None:
        if not isinstance(occupancy_key, str) or not occupancy_key:
            raise ValueError("canonical mmCIF payload has invalid occupancy array reference")
        if occupancy_dtype is None:
            raise ValueError("canonical mmCIF occupancy dtype is missing")
        occupancy_values = array_values[occupancy_key]
        if not isinstance(occupancy_values, list):
            raise ValueError("canonical mmCIF occupancy array must be a list")
        arrays[occupancy_key] = np.asarray(
            [np.nan if value is None else value for value in occupancy_values],
            dtype=occupancy_dtype,
        )
    # Structure has no per-site B-factor field, so canonical payloads do not retain it.
    structure = decode_structure(payload, arrays)
    row_sites = tuple(row.canonical_site for row in rows)
    if any(site is None for site in row_sites) or row_sites != structure.sites:
        raise ValueError("canonical mmCIF atom_site identity does not match _ofoldx_structure payload")
    cif_file = cast(_CifFile, parsed.mmcif_file)
    formal_charges = _cif_column(cif_file.block["atom_site"], "pdbx_formal_charge")
    for row, site in zip(rows, structure.sites):
        atom = structure.system.atom(site)
        if row.element != atom.element:
            raise ValueError(f"canonical mmCIF element disagrees with System atom at {site!r}")
        if optional_integer(formal_charges[row.row_index]) != atom.formal_charge:
            raise ValueError(f"canonical mmCIF formal charge disagrees with System atom at {site!r}")
    return structure


def _cif_column(category: _CifCategory, name: str) -> tuple[str | None, ...]:
    if name not in category:
        return (None,) * category.row_count
    return tuple(present_string(value) for value in category[name].as_array())


def _cif_bond_order(value: str | None) -> tuple[float, bool]:
    if value is None:
        raise ValueError("mmCIF covalent bond order is missing")
    key = value.lower()
    try:
        return _CIF_BOND_ORDERS[key]
    except KeyError:
        raise ValueError(f"unsupported mmCIF covalent bond order: {value!r}") from None


def _remove_component_atoms(component: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
    atom_by_id = {atom.atom_id: atom for atom in component.atoms}
    removed = set(atom_by_id).intersection(atom_ids)
    if not removed:
        return component
    bonds = component.bonds
    if bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph")
    for bond in bonds:
        if bond.atom_a in removed and atom_by_id[bond.atom_b].element == "H":
            removed.add(bond.atom_b)
        if bond.atom_b in removed and atom_by_id[bond.atom_a].element == "H":
            removed.add(bond.atom_a)
    for bond in bonds:
        if (
            bond.stereo_atoms is not None
            and removed.intersection(bond.stereo_atoms)
            and not removed.intersection((bond.atom_a, bond.atom_b))
        ):
            raise ValueError("a declared leaving atom is required to define retained bond stereochemistry")
    keep = tuple(atom.atom_id not in removed for atom in component.atoms)
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atom for atom, present in zip(component.atoms, keep) if present),
        bonds=frozenset(bond for bond in bonds if bond.atom_a not in removed and bond.atom_b not in removed),
        previous_attachment=component.previous_attachment,
        next_attachment=component.next_attachment,
        reference_coordinates=(
            tuple(coordinate for coordinate, present in zip(component.reference_coordinates, keep) if present)
            if component.reference_coordinates
            else ()
        ),
    )


def read_mmcif_branch_entities(
    parsed: ParsedStructure,
    component_table: Mapping[str, ChemicalComponent],
    selected_rows: tuple[AtomRecord, ...],
) -> tuple[dict[str, Molecule], dict[tuple[object, ...], int]]:
    r"""Decode selected mmCIF branched entities and source component indices."""

    cif_file = parsed.mmcif_file
    if cif_file is None:
        return {}, {}
    block = cast(_CifFile, cif_file).block
    if "pdbx_entity_branch_list" not in block:
        return {}, {}

    selected_entity_ids = {row.entity_id for row in selected_rows if row.entity_id is not None}
    category = block["pdbx_entity_branch_list"]
    entity_ids = _cif_column(category, "entity_id")
    component_ids = _cif_column(category, "comp_id")
    numbers = _cif_column(category, "num")
    declarations: dict[str, dict[int, str]] = {}
    for entity_id, component_id, number in zip(entity_ids, component_ids, numbers):
        if entity_id is None or entity_id not in selected_entity_ids:
            continue
        component_index = optional_integer(number)
        if component_id is None or component_index is None:
            raise ValueError("mmCIF branched-entity components require entity_id, comp_id, and num")
        entity_components = declarations.setdefault(entity_id, {})
        if component_index in entity_components:
            raise ValueError(f"duplicate branched-entity component number {component_index} for entity {entity_id!r}")
        entity_components[component_index] = component_id

    number_to_index = {
        entity_id: {number: index for index, number in enumerate(sorted(components))}
        for entity_id, components in declarations.items()
    }
    removed_atoms: dict[tuple[str, int], set[str]] = {}
    bonds: dict[str, set[MoleculeBond]] = {entity_id: set() for entity_id in declarations}
    if "pdbx_entity_branch_link" in block:
        links = block["pdbx_entity_branch_link"]
        link_entities = _cif_column(links, "entity_id")
        number_1 = _cif_column(links, "entity_branch_list_num_1")
        number_2 = _cif_column(links, "entity_branch_list_num_2")
        component_1 = _cif_column(links, "comp_id_1")
        component_2 = _cif_column(links, "comp_id_2")
        atom_1 = _cif_column(links, "atom_id_1")
        atom_2 = _cif_column(links, "atom_id_2")
        leaving_1 = _cif_column(links, "leaving_atom_id_1")
        leaving_2 = _cif_column(links, "leaving_atom_id_2")
        orders = _cif_column(links, "value_order")
        for values in zip(
            link_entities,
            number_1,
            component_1,
            atom_1,
            leaving_1,
            number_2,
            component_2,
            atom_2,
            leaving_2,
            orders,
        ):
            (
                entity_id,
                first_number,
                first_component,
                first_atom,
                first_leaving,
                second_number,
                second_component,
                second_atom,
                second_leaving,
                order_name,
            ) = values
            if entity_id not in declarations:
                continue
            first_number_int = optional_integer(first_number)
            second_number_int = optional_integer(second_number)
            if first_number_int is None or second_number_int is None or first_atom is None or second_atom is None:
                raise ValueError("mmCIF branched-entity links require two component numbers and atom IDs")
            declared = declarations[entity_id]
            if declared.get(first_number_int) != first_component or declared.get(second_number_int) != second_component:
                raise ValueError(f"branched-entity link disagrees with component declaration for entity {entity_id!r}")
            first_index = number_to_index[entity_id][first_number_int]
            second_index = number_to_index[entity_id][second_number_int]
            order, aromatic = _cif_bond_order(order_name)
            bonds[entity_id].add(
                MoleculeBond(first_index, first_atom, second_index, second_atom, order=order, aromatic=aromatic)
            )
            if first_leaving is not None:
                removed_atoms.setdefault((entity_id, first_index), set()).add(first_leaving)
            if second_leaving is not None:
                removed_atoms.setdefault((entity_id, second_index), set()).add(second_leaving)

    entities: dict[str, Molecule] = {}
    for entity_id, numbered_components in declarations.items():
        components: list[ChemicalComponent] = []
        for component_index, component_number in enumerate(sorted(numbered_components)):
            component_id = numbered_components[component_number]
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            components.append(
                _remove_component_atoms(component, removed_atoms.get((entity_id, component_index), set()))
            )
        entities[entity_id] = Molecule(
            components=tuple(components),
            bonds=frozenset(bonds[entity_id]),
        )

    component_indices: dict[tuple[object, ...], int] = {}
    if "pdbx_branch_scheme" in block:
        scheme = block["pdbx_branch_scheme"]
        label_asym_ids = _cif_column(scheme, "asym_id")
        scheme_entities = _cif_column(scheme, "entity_id")
        monomer_ids = _cif_column(scheme, "mon_id")
        scheme_numbers = _cif_column(scheme, "num")
        pdb_asym_ids = _cif_column(scheme, "pdb_asym_id")
        pdb_monomer_ids = _cif_column(scheme, "pdb_mon_id")
        pdb_sequence_numbers = _cif_column(scheme, "pdb_seq_num")
        auth_asym_ids = _cif_column(scheme, "auth_asym_id")
        auth_monomer_ids = _cif_column(scheme, "auth_mon_id")
        auth_sequence_numbers = _cif_column(scheme, "auth_seq_num")
        for values in zip(
            label_asym_ids,
            scheme_entities,
            monomer_ids,
            scheme_numbers,
            pdb_asym_ids,
            pdb_monomer_ids,
            pdb_sequence_numbers,
            auth_asym_ids,
            auth_monomer_ids,
            auth_sequence_numbers,
        ):
            (
                label_asym_id,
                entity_id,
                monomer_id,
                number,
                pdb_asym_id,
                pdb_monomer_id,
                pdb_sequence_number,
                auth_asym_id,
                auth_monomer_id,
                auth_sequence_number,
            ) = values
            if entity_id not in entities:
                continue
            number_int = optional_integer(number)
            if number_int is None:
                raise ValueError("mmCIF branch scheme requires an integer component number")
            component_index = number_to_index[entity_id][number_int]
            source_monomer_ids = {monomer_id, pdb_monomer_id, auth_monomer_id} - {None}
            source_positions = {
                (source_asym_id, source_sequence_number)
                for source_asym_id, source_sequence_number in (
                    (pdb_asym_id, optional_integer(pdb_sequence_number)),
                    (auth_asym_id, optional_integer(auth_sequence_number)),
                )
                if source_asym_id is not None and source_sequence_number is not None
            }
            for source_monomer_id in source_monomer_ids:
                for source_asym_id, source_sequence_number in source_positions:
                    key = (
                        label_asym_id,
                        entity_id,
                        source_monomer_id,
                        source_asym_id,
                        source_sequence_number,
                    )
                    previous = component_indices.get(key)
                    if previous is not None and previous != component_index:
                        raise ValueError(f"ambiguous mmCIF branch-scheme identity: {key!r}")
                    component_indices[key] = component_index
    return entities, component_indices


def read_mmcif_bonds(
    parsed: ParsedStructure,
    selected_rows: tuple[AtomRecord, ...],
    site_by_row_index: dict[int, AtomSite],
) -> tuple[SystemBond, ...]:
    r"""Decode selected mmCIF covalent connections into canonical bonds."""

    cif_file = parsed.mmcif_file
    if cif_file is None:
        return ()
    block = cast(_CifFile, cif_file).block
    if "struct_conn" not in block:
        return ()
    category = block["struct_conn"]
    conn_types = _cif_column(category, "conn_type_id")
    label_asym_1 = _cif_column(category, "ptnr1_label_asym_id")
    label_asym_2 = _cif_column(category, "ptnr2_label_asym_id")
    label_seq_1 = _cif_column(category, "ptnr1_label_seq_id")
    label_seq_2 = _cif_column(category, "ptnr2_label_seq_id")
    label_comp_1 = _cif_column(category, "ptnr1_label_comp_id")
    label_comp_2 = _cif_column(category, "ptnr2_label_comp_id")
    label_atom_1 = _cif_column(category, "ptnr1_label_atom_id")
    label_atom_2 = _cif_column(category, "ptnr2_label_atom_id")
    auth_asym_1 = _cif_column(category, "ptnr1_auth_asym_id")
    auth_asym_2 = _cif_column(category, "ptnr2_auth_asym_id")
    auth_seq_1 = _cif_column(category, "ptnr1_auth_seq_id")
    auth_seq_2 = _cif_column(category, "ptnr2_auth_seq_id")
    auth_comp_1 = _cif_column(category, "ptnr1_auth_comp_id")
    auth_comp_2 = _cif_column(category, "ptnr2_auth_comp_id")
    auth_atom_1 = _cif_column(category, "ptnr1_auth_atom_id")
    auth_atom_2 = _cif_column(category, "ptnr2_auth_atom_id")
    ins_1 = _cif_column(category, "pdbx_ptnr1_PDB_ins_code")
    ins_2 = _cif_column(category, "pdbx_ptnr2_PDB_ins_code")
    symmetry_1 = _cif_column(category, "ptnr1_symmetry")
    symmetry_2 = _cif_column(category, "ptnr2_symmetry")
    value_orders = _cif_column(category, "pdbx_value_order")
    selected_sites = selected_source_sites(selected_rows, site_by_row_index)
    symmetry_ids = {row.symmetry_id for row in selected_rows}
    bonds: list[SystemBond] = []
    for index, conn_type in enumerate(conn_types):
        if conn_type is None or not (conn_type.startswith("covale") or conn_type.startswith("disulf")):
            continue
        if symmetry_1[index] not in {None, "1_555"} or symmetry_2[index] not in {None, "1_555"}:
            continue
        order, aromatic = _cif_bond_order(value_orders[index])
        endpoint_1 = (
            label_asym_1[index],
            optional_integer(label_seq_1[index]),
            ins_1[index],
            label_comp_1[index],
            label_atom_1[index],
        )
        endpoint_2 = (
            label_asym_2[index],
            optional_integer(label_seq_2[index]),
            ins_2[index],
            label_comp_2[index],
            label_atom_2[index],
        )
        auth_endpoint_1 = (
            auth_asym_1[index] or label_asym_1[index],
            optional_integer(auth_seq_1[index] or label_seq_1[index]),
            ins_1[index],
            auth_comp_1[index] or label_comp_1[index],
            auth_atom_1[index] or label_atom_1[index],
        )
        auth_endpoint_2 = (
            auth_asym_2[index] or label_asym_2[index],
            optional_integer(auth_seq_2[index] or label_seq_2[index]),
            ins_2[index],
            auth_comp_2[index] or label_comp_2[index],
            auth_atom_2[index] or label_atom_2[index],
        )
        for symmetry_id in symmetry_ids:
            site_a = selected_sites.get((symmetry_id, *endpoint_1)) or selected_sites.get(
                (symmetry_id, *auth_endpoint_1)
            )
            site_b = selected_sites.get((symmetry_id, *endpoint_2)) or selected_sites.get(
                (symmetry_id, *auth_endpoint_2)
            )
            if site_a is None or site_b is None:
                raise ValueError(
                    "mmCIF struct_conn references an atom absent from the selected assembly/model: "
                    f"label={endpoint_1!r}, {endpoint_2!r}; auth={auth_endpoint_1!r}, {auth_endpoint_2!r}"
                )
            bonds.append(
                SystemBond(
                    site_a,
                    site_b,
                    order=order,
                    aromatic=aromatic,
                    origin="source",
                )
            )
    return tuple(bonds)


def write_mmcif(structure: Structure, path: Path) -> None:
    r"""Write a lossless canonical mmCIF exchange representation."""

    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    payload, _ = encode_structure(structure, array_prefix="atom_site")
    # Structure has no source-identity, alternate-location, or B-factor fields. The
    # writer derives identity columns, emits one conformer, and writes B factor 0.
    residue_numbers = structure.residue_numbers or {}
    label_asym_by_instance: dict[str, str] = {}
    used_label_asym_ids: set[str] = set()
    for instance_id in structure.system.instances:
        label_asym_by_instance[instance_id] = unique_identifier(instance_id, used_label_asym_ids)
    columns: dict[str, list[object]] = {
        "group_PDB": [],
        "id": [],
        "type_symbol": [],
        "label_atom_id": [],
        "label_alt_id": [],
        "label_comp_id": [],
        "label_asym_id": [],
        "label_entity_id": [],
        "label_seq_id": [],
        "pdbx_PDB_ins_code": [],
        "Cartn_x": [],
        "Cartn_y": [],
        "Cartn_z": [],
        "occupancy": [],
        "B_iso_or_equiv": [],
        "pdbx_formal_charge": [],
        "auth_seq_id": [],
        "auth_comp_id": [],
        "auth_asym_id": [],
        "auth_atom_id": [],
        "pdbx_PDB_model_num": [],
        "ofoldx_instance_id": [],
        "ofoldx_component_index": [],
        "ofoldx_atom_id": [],
        "ofoldx_coordinate_mask": [],
    }
    endpoint_identity: dict[AtomSite, tuple[str, str, str, str, str, str, str, str]] = {}
    for index, site in enumerate(structure.sites):
        atom = structure.system.atom(site)
        entity_id = structure.system.instances[site.instance_id]
        entity = structure.system.entities[entity_id]
        component = structure.system.component((site.instance_id, site.component_index))
        component_id = component.component_id
        residue_number = residue_numbers.get((site.instance_id, site.component_index))
        if isinstance(entity, PolymerSequence):
            group = "ATOM"
            label_seq_id: object = site.component_index + 1
        else:
            group = "HETATM"
            label_seq_id = "."
        label_entity_id = entity_id
        label_asym_id = label_asym_by_instance[site.instance_id]
        label_component_id = component_id
        label_atom_id = site.atom_id
        auth_asym_id = label_asym_id
        auth_seq_id: object = residue_number.number if residue_number is not None else site.component_index + 1
        auth_component_id = label_component_id
        auth_atom_id = label_atom_id
        insertion_code = (
            residue_number.insertion_code if residue_number is not None and residue_number.insertion_code else "?"
        )
        coordinate = structure.coordinates[index]
        if not np.all(np.isfinite(coordinate)):
            raise ValueError("mmCIF cannot represent non-finite coordinates")
        occupancy = (
            "."
            if structure.occupancy is None or np.isnan(structure.occupancy[index])
            else float(structure.occupancy[index])
        )
        if occupancy != "." and not np.isfinite(occupancy):
            raise ValueError("mmCIF cannot represent non-finite occupancy")
        b_factor = "."
        altloc = "."
        values = {
            "group_PDB": group,
            "id": index + 1,
            "type_symbol": atom.element,
            "label_atom_id": label_atom_id,
            "label_alt_id": altloc,
            "label_comp_id": label_component_id,
            "label_asym_id": label_asym_id,
            "label_entity_id": label_entity_id,
            "label_seq_id": label_seq_id,
            "pdbx_PDB_ins_code": insertion_code,
            "Cartn_x": float(coordinate[0]),
            "Cartn_y": float(coordinate[1]),
            "Cartn_z": float(coordinate[2]),
            "occupancy": occupancy,
            "B_iso_or_equiv": b_factor,
            "pdbx_formal_charge": "?" if atom.formal_charge is None else atom.formal_charge,
            "auth_seq_id": auth_seq_id,
            "auth_comp_id": auth_component_id,
            "auth_asym_id": auth_asym_id,
            "auth_atom_id": auth_atom_id,
            "pdbx_PDB_model_num": 1,
            "ofoldx_instance_id": site.instance_id,
            "ofoldx_component_index": site.component_index,
            "ofoldx_atom_id": site.atom_id,
            "ofoldx_coordinate_mask": 1 if bool(structure.coordinate_mask[index]) else 0,
        }
        for name, value in values.items():
            columns[name].append(value)
        endpoint_identity[site] = (
            str(label_asym_id),
            str(label_seq_id),
            str(label_component_id),
            str(label_atom_id),
            str(auth_asym_id),
            str(auth_seq_id),
            str(auth_component_id),
            str(auth_atom_id),
        )

    categories: dict[str, object] = {
        "atom_site": CIFCategory(columns),
        "ofoldx_structure": CIFCategory(
            {
                "json": [
                    json.dumps(
                        {
                            "metadata": payload,
                            "arrays": {
                                payload["coordinates"]: structure.coordinates.tolist(),
                                payload["coordinate_mask"]: structure.coordinate_mask.tolist(),
                                **(
                                    {}
                                    if structure.occupancy is None
                                    else {
                                        payload["occupancy"]: [
                                            None if np.isnan(value) else value for value in structure.occupancy.tolist()
                                        ]
                                    }
                                ),
                            },
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ]
            }
        ),
    }
    all_bonds = ordered_output_bonds(structure.system)
    if all_bonds:
        conn_columns: dict[str, list[object]] = {
            "id": [],
            "conn_type_id": [],
            "ptnr1_label_asym_id": [],
            "ptnr1_label_seq_id": [],
            "ptnr1_label_comp_id": [],
            "ptnr1_label_atom_id": [],
            "ptnr2_label_asym_id": [],
            "ptnr2_label_seq_id": [],
            "ptnr2_label_comp_id": [],
            "ptnr2_label_atom_id": [],
            "ptnr1_auth_asym_id": [],
            "ptnr1_auth_seq_id": [],
            "ptnr1_auth_comp_id": [],
            "ptnr1_auth_atom_id": [],
            "ptnr2_auth_asym_id": [],
            "ptnr2_auth_seq_id": [],
            "ptnr2_auth_comp_id": [],
            "ptnr2_auth_atom_id": [],
            "pdbx_value_order": [],
        }
        order_names = {1.0: "sing", 2.0: "doub", 3.0: "trip", 4.0: "quad"}
        for index, bond in enumerate(all_bonds, 1):
            if bond.site_a not in endpoint_identity or bond.site_b not in endpoint_identity:
                continue
            first = endpoint_identity[bond.site_a]
            second = endpoint_identity[bond.site_b]
            conn_values = (
                index,
                "covale",
                *first[:4],
                *second[:4],
                *first[4:],
                *second[4:],
                "arom" if bond.aromatic else order_names[bond.order],
            )
            for name, value in zip(conn_columns, conn_values):
                conn_columns[name].append(value)
        if conn_columns["id"]:
            categories["struct_conn"] = CIFCategory(conn_columns)
    cif_file = CIFFile({"ofoldx": CIFBlock(categories)})
    cif_file.write(str(path))
