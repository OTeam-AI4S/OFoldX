r"""Standard mmCIF projection of the canonical tensor molecular graph."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from ..data.components import get_component
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    HYDROGEN_ELEMENTS,
    MOLECULE_BRANCHED,
    MOLECULE_NON_POLYMER,
    MOLECULE_WATER,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    PolymerType,
    SystemBond,
)
from ..data.structure import Structure
from .records import (
    atom_rows_from_array,
    optional_integer,
    present_string,
    reject_duplicate_atom_rows,
    selected_source_sites,
    unique_identifier,
)

if TYPE_CHECKING:
    from biotite.structure.io.pdbx import CIFCategory, CIFFile


_CIF_BOND_ORDERS = {
    "sing": (1.0, False),
    "doub": (2.0, False),
    "trip": (3.0, False),
    "quad": (4.0, False),
    "arom": (1.5, True),
}


def parse_mmcif(path: Path, *, model: int, assembly_id: str | None) -> tuple[np.recarray, object]:
    r"""
    Parse one mmCIF model or assembly into transient source columns.

    Args:
        path: Source mmCIF path.
        model: One-based model number to select.
        assembly_id: Biological assembly identifier, or `None` to read the asymmetric unit.

    Returns:
        Selected source atom rows and the mmCIF document.

    Raises:
        ValueError: If the requested model or assembly is unavailable, atom rows are
            duplicated, or no atom-site category exists.
    """

    from biotite.structure.io import pdbx

    cif_file = pdbx.CIFFile.read(str(path))
    block = cif_file.block
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
        )
        if field in available
    ]
    for special, source_column in (
        ("atom_id", "id"),
        ("b_factor", "B_iso_or_equiv"),
        ("occupancy", "occupancy"),
        ("pdbx_formal_charge", "pdbx_formal_charge"),
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
    rows = atom_rows_from_array(array, format_name="mmcif")
    source_ids = _cif_column(atom_site, "id")
    if not all(source_id is not None for source_id in source_ids):
        raise ValueError("mmCIF _atom_site.id is required")
    source_values = {
        source_id: (occupancy, b_factor, charge, author_number)
        for source_id, occupancy, b_factor, charge, author_number in zip(
            source_ids,
            _cif_column(atom_site, "occupancy"),
            _cif_column(atom_site, "B_iso_or_equiv"),
            _cif_column(atom_site, "pdbx_formal_charge"),
            _cif_column(atom_site, "auth_seq_id"),
        )
    }
    if len(source_values) != len(source_ids):
        raise ValueError("mmCIF _atom_site.id values must be unique")
    for row in rows:
        source_id = str(row.serial)
        try:
            occupancy, b_factor, charge, author_number = source_values[source_id]
        except KeyError:
            raise ValueError(f"mmCIF assembly atom {source_id!r} has no source _atom_site row") from None
        row.occupancy = None if occupancy is None else float(occupancy)
        row.b_factor = None if b_factor is None else float(b_factor)
        row.charge = optional_integer(charge)
        row.auth_residue_id = author_number
    reject_duplicate_atom_rows(rows)
    return rows, cif_file


def _cif_column(category: CIFCategory, name: str) -> tuple[str | None, ...]:
    if name not in category:
        return (None,) * category.row_count
    return tuple(present_string(value) for value in category[name].as_array())


def read_mmcif_components(
    cif_file: object,
) -> tuple[dict[str, ChemicalComponent], set[str]]:
    r"""Read embedded component definitions and every declared component ID."""

    block = cast("CIFFile", cif_file).block
    declared_components: set[str] = set()
    for category_name, column_name in (
        ("entity_poly_seq", "mon_id"),
        ("pdbx_entity_branch_list", "comp_id"),
        ("pdbx_entity_nonpoly", "comp_id"),
    ):
        if category_name in block:
            declared_components.update(
                value for value in _cif_column(block[category_name], column_name) if value is not None
            )
    if "chem_comp_atom" not in block:
        return {}, declared_components
    atom_category = block["chem_comp_atom"]
    component_ids = _cif_column(atom_category, "comp_id")
    declared_components.update(component_id for component_id in component_ids if component_id is not None)
    atom_ids = _cif_column(atom_category, "atom_id")
    elements = _cif_column(atom_category, "type_symbol")
    charges = _cif_column(atom_category, "charge")
    aromatic_flags = _cif_column(atom_category, "pdbx_aromatic_flag")
    stereo_values = _cif_column(atom_category, "pdbx_stereo_config")
    reference_x = _cif_column(atom_category, "model_Cartn_x")
    reference_y = _cif_column(atom_category, "model_Cartn_y")
    reference_z = _cif_column(atom_category, "model_Cartn_z")

    atoms: dict[str, list[ChemicalAtom]] = {}
    coordinates: dict[str, list[tuple[float, float, float] | None]] = {}
    for values in zip(
        component_ids,
        atom_ids,
        elements,
        charges,
        aromatic_flags,
        stereo_values,
        reference_x,
        reference_y,
        reference_z,
    ):
        component_id, atom_id, element, charge, aromatic, stereo, x, y, z = values
        if component_id is None or atom_id is None or element is None:
            raise ValueError("mmCIF chem_comp_atom row is missing comp_id, atom_id, or type_symbol")
        atoms.setdefault(component_id, []).append(
            ChemicalAtom(
                atom_id=atom_id,
                element=element.capitalize(),
                formal_charge=optional_integer(charge),
                aromatic=(
                    None
                    if aromatic is None or aromatic.upper() not in {"Y", "N"}
                    else aromatic.upper() == "Y"
                ),
                stereo=None if stereo in {None, "N"} else stereo,
            )
        )
        coordinates.setdefault(component_id, []).append(
            None if x is None or y is None or z is None else (float(x), float(y), float(z))
        )

    bonds_by_component: dict[str, set[ChemicalBond]] = {component_id: set() for component_id in atoms}
    components_with_bonds: set[str] = set()
    if "chem_comp_bond" in block:
        bond_category = block["chem_comp_bond"]
        order_names = (
            _cif_column(bond_category, "value_order")
            if "value_order" in bond_category
            else ("sing",) * bond_category.row_count
        )
        for component_id, atom_a, atom_b, order_name, aromatic, stereo in zip(
            _cif_column(bond_category, "comp_id"),
            _cif_column(bond_category, "atom_id_1"),
            _cif_column(bond_category, "atom_id_2"),
            order_names,
            _cif_column(bond_category, "pdbx_aromatic_flag"),
            _cif_column(bond_category, "pdbx_stereo_config"),
        ):
            if component_id is None or atom_a is None or atom_b is None:
                raise ValueError("mmCIF chem_comp_bond row is missing comp_id or atom IDs")
            components_with_bonds.add(component_id)
            order, order_aromatic = _cif_component_bond_order(order_name)
            if aromatic is None:
                aromatic_value = order_aromatic
            else:
                normalized_aromatic = aromatic.upper()
                aromatic_value = (
                    normalized_aromatic == "Y"
                    if normalized_aromatic in {"Y", "N"}
                    else None
                )
            bonds_by_component.setdefault(component_id, set()).add(
                ChemicalBond(
                    atom_a,
                    atom_b,
                    order=order,
                    aromatic=aromatic_value,
                    stereo=None if stereo in {None, "N"} else stereo,
                )
            )
    return (
        {
            component_id: ChemicalComponent(
                component_id=component_id,
                atoms=tuple(component_atoms),
                bonds=(
                    frozenset(bonds_by_component[component_id])
                    if len(component_atoms) == 1 or component_id in components_with_bonds
                    else None
                ),
                reference_coordinates=tuple(coordinates[component_id]),
            )
            for component_id, component_atoms in atoms.items()
        },
        declared_components,
    )


def _normalized_component_id(component_id: str) -> str:
    return {"ASX": "ASP", "GLX": "GLU"}.get(component_id, component_id)


def _normalized_atom_id(component_id: str, atom_id: str) -> str:
    if component_id == "ASX" and atom_id in {"XD1", "XD2"}:
        return "O" + atom_id[1:]
    if component_id == "GLX" and atom_id in {"XE1", "XE2"}:
        return "O" + atom_id[1:]
    return atom_id


def read_mmcif_declared_component_ids(cif_file: object) -> set[str]:
    """Read structural component declarations without consulting embedded chemistry categories."""

    block = cast("CIFFile", cif_file).block
    declared: set[str] = set()
    for category_name, column_name in (
        ("entity_poly_seq", "mon_id"),
        ("pdbx_entity_branch_list", "comp_id"),
        ("pdbx_entity_nonpoly", "comp_id"),
    ):
        if category_name not in block:
            continue
        declared.update(
            _normalized_component_id(value)
            for value in _cif_column(block[category_name], column_name)
            if value is not None
        )
    return declared


def _read_mmcif_entity_declarations(
    cif_file: object,
) -> tuple[
    dict[str, str],
    dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    dict[str, str],
    dict[tuple[str, int], tuple[str | None, str | None]],
    dict[str, str],
    dict[str, str],
]:
    r"""
    Decode mmCIF entity-type, polymer-sequence, and author-numbering declarations.

    The ``_entity.type`` declaration is the authoritative polymer/non-polymer/water
    classification, and ``_entity_poly_seq`` is the ordered canonical monomer sequence
    per polymer entity including residues unmodeled in the coordinates.

    Args:
        cif_file: Parsed mmCIF document.

    Returns:
        Entity declarations gathered from the source. Absent categories yield empty
        mappings; malformed rows in a present category raise.
    """

    block = cast("CIFFile", cif_file).block

    entity_types: dict[str, str] = {}
    if "entity" in block:
        category = block["entity"]
        for entity_id, entity_type in zip(_cif_column(category, "id"), _cif_column(category, "type")):
            if entity_id is None or entity_type is None:
                raise ValueError("mmCIF _entity row is missing id or type")
            if entity_id in entity_types:
                raise ValueError(f"mmCIF _entity contains duplicate id {entity_id!r}")
            entity_types[entity_id] = entity_type.lower()

    polymer_monomers: dict[str, dict[int, list[str]]] = {}
    has_polymer_sequences = "entity_poly_seq" in block
    if has_polymer_sequences:
        category = block["entity_poly_seq"]
        for entity_id, number, monomer in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "num"),
            _cif_column(category, "mon_id"),
        ):
            index = optional_integer(number)
            if entity_id is None or index is None or monomer is None:
                raise ValueError("mmCIF _entity_poly_seq row is missing entity_id, num, or mon_id")
            if index < 1:
                raise ValueError(f"mmCIF _entity_poly_seq.num must be positive, got {index}")
            monomer = _normalized_component_id(monomer)
            candidates = polymer_monomers.setdefault(entity_id, {}).setdefault(index, [])
            if monomer not in candidates:
                candidates.append(monomer)
        for entity_id, positions in polymer_monomers.items():
            numbers = sorted(positions)
            if numbers != list(range(1, len(numbers) + 1)):
                raise ValueError(f"mmCIF polymer entity {entity_id!r} has a non-contiguous declared sequence")

    entity_poly_types: dict[str, str] = {}
    if "entity_poly" in block:
        category = block["entity_poly"]
        for entity_id, poly_type in zip(_cif_column(category, "entity_id"), _cif_column(category, "type")):
            if entity_id is None or poly_type is None:
                raise ValueError("mmCIF _entity_poly row is missing entity_id or type")
            if entity_id in entity_poly_types:
                raise ValueError(f"mmCIF _entity_poly contains duplicate entity_id {entity_id!r}")
            entity_poly_types[entity_id] = poly_type.lower()

    author_numbering: dict[tuple[str, int], tuple[str | None, str | None]] = {}
    if "pdbx_poly_seq_scheme" in block:
        category = block["pdbx_poly_seq_scheme"]
        for asym_id, seq_id, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "seq_id"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            seq = optional_integer(seq_id)
            if asym_id is None or seq is None:
                raise ValueError("mmCIF _pdbx_poly_seq_scheme row is missing asym_id or seq_id")
            author_residue_number = present_string(auth_number)
            if author_residue_number is None:
                author_residue_number = present_string(pdb_number)
            key = (asym_id, seq)
            numbering = (author_residue_number, ins_code)
            if key in author_numbering and author_numbering[key] != numbering:
                raise ValueError(f"mmCIF _pdbx_poly_seq_scheme has conflicting numbering for {key!r}")
            author_numbering[key] = numbering

    if "pdbx_branch_scheme" in block:
        category = block["pdbx_branch_scheme"]
        for asym_id, number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            component_number = optional_integer(number)
            if asym_id is None or component_number is None:
                raise ValueError("mmCIF _pdbx_branch_scheme row is missing asym_id or num")
            author_residue_number = present_string(pdb_number)
            key = (asym_id, component_number)
            numbering = (author_residue_number, ins_code)
            if key in author_numbering and author_numbering[key] != numbering:
                raise ValueError("mmCIF _pdbx_branch_scheme has conflicting numbering " f"for {key!r}")
            author_numbering[key] = numbering

    if "pdbx_nonpoly_scheme" in block:
        category = block["pdbx_nonpoly_scheme"]
        for asym_id, ndb_number, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "ndb_seq_num"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            component_number = optional_integer(ndb_number)
            if asym_id is None or component_number is None:
                raise ValueError("mmCIF _pdbx_nonpoly_scheme row is missing asym_id or ndb_seq_num")
            author_residue_number = auth_number or pdb_number
            key = (asym_id, component_number)
            numbering = (author_residue_number, ins_code)
            if key in author_numbering and author_numbering[key] != numbering:
                raise ValueError("mmCIF _pdbx_nonpoly_scheme has conflicting numbering " f"for {key!r}")
            author_numbering[key] = numbering

    asym_entities: dict[str, str] = {}
    if "struct_asym" in block:
        category = block["struct_asym"]
        for asym_id, entity_id in zip(
            _cif_column(category, "id"),
            _cif_column(category, "entity_id"),
        ):
            if asym_id is None or entity_id is None:
                raise ValueError("mmCIF _struct_asym row is missing id or entity_id")
            previous = asym_entities.get(asym_id)
            if previous is not None and previous != entity_id:
                raise ValueError(f"mmCIF _struct_asym has conflicting entity for {asym_id!r}")
            asym_entities[asym_id] = entity_id

    nonpoly_components: dict[str, str] = {}
    if "pdbx_entity_nonpoly" in block:
        category = block["pdbx_entity_nonpoly"]
        for entity_id, component_id in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "comp_id"),
        ):
            if entity_id is None or component_id is None:
                raise ValueError("mmCIF _pdbx_entity_nonpoly row is missing entity_id or comp_id")
            component_id = _normalized_component_id(component_id)
            previous = nonpoly_components.get(entity_id)
            if previous is not None and previous != component_id:
                raise ValueError(f"mmCIF non-polymer entity {entity_id!r} has conflicting components")
            nonpoly_components[entity_id] = component_id

    return (
        entity_types,
        (
            {
                entity_id: tuple((number, tuple(positions[number])) for number in sorted(positions))
                for entity_id, positions in polymer_monomers.items()
            }
            if has_polymer_sequences
            else None
        ),
        entity_poly_types,
        author_numbering,
        asym_entities,
        nonpoly_components,
    )


def _cif_bond_order(value: str | None) -> tuple[float | None, bool | None]:
    if value is None:
        return None, None
    try:
        return _CIF_BOND_ORDERS[value.lower()]
    except KeyError:
        raise ValueError(f"unsupported mmCIF covalent bond order: {value!r}") from None


def _cif_component_bond_order(value: str | None) -> tuple[float | None, bool | None]:
    order, aromatic = _cif_bond_order(value)
    if value is not None and value.lower() != "arom":
        aromatic = None
    return order, aromatic


def _remove_component_atoms(component: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
    atom_by_id = {atom.atom_id: atom for atom in component.atoms}
    removed = set(atom_by_id).intersection(atom_ids)
    if not removed:
        return component
    bonds = component.bonds
    if bonds is not None:
        for bond in bonds:
            if bond.atom_a in removed and atom_by_id[bond.atom_b].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_b)
            if bond.atom_b in removed and atom_by_id[bond.atom_a].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_a)
        for bond in bonds:
            if (
                bond.stereo_atoms is not None
                and removed.intersection(bond.stereo_atoms)
                and not removed.intersection((bond.atom_a, bond.atom_b))
            ):
                raise ValueError("a declared leaving atom is required to define retained bond stereochemistry")
    keep = tuple(atom.atom_id not in removed for atom in component.atoms)
    retained = set(atom_by_id) - removed
    previous = component.previous_attachment
    if previous is not None:
        previous = (
            PolymerAttachment(
                previous.atom_id,
                previous.leaving_atom_ids & retained,
            )
            if previous.atom_id in retained
            else None
        )
    following = component.next_attachment
    if following is not None:
        following = (
            PolymerAttachment(
                following.atom_id,
                following.leaving_atom_ids & retained,
            )
            if following.atom_id in retained
            else None
        )
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atom for atom, present in zip(component.atoms, keep) if present),
        bonds=(
            None
            if bonds is None
            else frozenset(bond for bond in bonds if bond.atom_a not in removed and bond.atom_b not in removed)
        ),
        previous_attachment=previous,
        next_attachment=following,
        reference_coordinates=(
            tuple(coordinate for coordinate, present in zip(component.reference_coordinates, keep) if present)
            if component.reference_coordinates
            else ()
        ),
    )


def _mmcif_branch_entities_with_components(
    cif_file: object,
    component_ids: set[str],
) -> set[str]:
    """Return branch entity IDs declaring at least one selected component ID."""

    block = cast("CIFFile", cif_file).block
    if "pdbx_entity_branch_list" not in block:
        return set()
    category = block["pdbx_entity_branch_list"]
    return {
        entity_id
        for entity_id, component_id in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "comp_id"),
        )
        if entity_id is not None
        and component_id is not None
        and _normalized_component_id(component_id) in component_ids
    }


def read_mmcif_branch_entities(
    cif_file: object,
    component_table: Mapping[str, ChemicalComponent],
    *,
    excluded_entity_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Molecule], dict[tuple[object, ...], frozenset[int]]]:
    r"""
    Decode selected mmCIF branched entities and source component indices.

    Args:
        cif_file: Parsed mmCIF document.
        component_table: Chemical components available by component ID.
        excluded_entity_ids: Raw branch entities intentionally removed by the
            ingestion policy.

    Returns:
        Branched entities by entity ID and source residue keys mapped to component indices.

    Raises:
        ValueError: If branch declarations, links, bond orders, component references,
            or source numbering are invalid.
    """

    block = cast("CIFFile", cif_file).block
    if "pdbx_entity_branch_list" not in block:
        return {}, {}

    category = block["pdbx_entity_branch_list"]
    entity_ids = _cif_column(category, "entity_id")
    component_ids = _cif_column(category, "comp_id")
    numbers = _cif_column(category, "num")
    declarations: dict[str, dict[int, str]] = {}
    for entity_id, component_id, number in zip(entity_ids, component_ids, numbers):
        if entity_id is None or entity_id in excluded_entity_ids:
            continue
        component_index = optional_integer(number)
        if component_id is None or component_index is None:
            raise ValueError("mmCIF branched-entity components require entity_id, comp_id, and num")
        component_id = _normalized_component_id(component_id)
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
        orders = _cif_column(links, "value_order") if "value_order" in links else ("sing",) * links.row_count
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
            if first_component is not None:
                first_atom = None if first_atom is None else _normalized_atom_id(first_component, first_atom)
                first_leaving = (
                    None if first_leaving is None else _normalized_atom_id(first_component, first_leaving)
                )
                first_component = _normalized_component_id(first_component)
            if second_component is not None:
                second_atom = None if second_atom is None else _normalized_atom_id(second_component, second_atom)
                second_leaving = (
                    None if second_leaving is None else _normalized_atom_id(second_component, second_leaving)
                )
                second_component = _normalized_component_id(second_component)
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
            molecule_type="branched",
        )

    component_index_candidates: dict[tuple[object, ...], set[int]] = {}
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
            monomer_id = None if monomer_id is None else _normalized_component_id(monomer_id)
            pdb_monomer_id = None if pdb_monomer_id is None else _normalized_component_id(pdb_monomer_id)
            auth_monomer_id = None if auth_monomer_id is None else _normalized_component_id(auth_monomer_id)
            number_int = optional_integer(number)
            if number_int is None:
                raise ValueError("mmCIF branch scheme requires an integer component number")
            try:
                component_index = number_to_index[entity_id][number_int]
            except KeyError:
                raise ValueError(
                    f"mmCIF branch scheme component number {number_int} is not declared for entity {entity_id!r}"
                ) from None
            if monomer_id != declarations[entity_id][number_int]:
                raise ValueError(
                    f"mmCIF branch scheme component {number_int} mon_id {monomer_id!r} "
                    f"disagrees with declaration {declarations[entity_id][number_int]!r}"
                )
            keys = []
            if label_asym_id is not None and monomer_id is not None:
                keys.append(("label", label_asym_id, entity_id, monomer_id, label_asym_id, None))
            if pdb_asym_id is not None and pdb_monomer_id is not None:
                keys.append(
                    (
                        "pdb",
                        label_asym_id,
                        entity_id,
                        pdb_monomer_id,
                        pdb_asym_id,
                        present_string(pdb_sequence_number),
                    )
                )
            if auth_asym_id is not None and auth_monomer_id is not None:
                keys.append(
                    (
                        "author",
                        label_asym_id,
                        entity_id,
                        auth_monomer_id,
                        auth_asym_id,
                        present_string(auth_sequence_number),
                    )
                )
            for key in keys:
                component_index_candidates.setdefault(key, set()).add(component_index)
    return entities, {key: frozenset(indices) for key, indices in component_index_candidates.items()}


def read_mmcif_bonds(
    cif_file: object,
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
    unmodeled_source_sites: dict[tuple[object, ...], AtomSite],
    ambiguous_unmodeled_sites: set[tuple[object, ...]],
) -> tuple[SystemBond, ...]:
    r"""
    Decode selected mmCIF covalent connections into system bonds.

    Args:
        cif_file: Parsed mmCIF document.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.
        unmodeled_source_sites: Declared but unobserved polymer atoms indexed by
            their label identities.
        ambiguous_unmodeled_sites: Source identities that name multiple unobserved
            canonical atoms.

    Returns:
        Resolved covalent system bonds. Connections whose endpoints are absent from the
        selected model (an unmodeled cap or modified residue, or a partner in another
        assembly) are skipped.

    Raises:
        ValueError: If a selected covalent connection declares an unsupported bond order
            or its label/author identities identify conflicting or ambiguous selected atoms.
    """

    block = cast("CIFFile", cif_file).block
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
    selected_sites, ambiguous_source_sites = selected_source_sites(selected_rows, site_by_row_index)
    for key in ambiguous_unmodeled_sites:
        selected_sites.pop(key, None)
    ambiguous_source_sites.update(ambiguous_unmodeled_sites)
    for key, site in unmodeled_source_sites.items():
        if key in ambiguous_source_sites:
            continue
        previous = selected_sites.get(key)
        if previous is not None and previous != site:
            selected_sites.pop(key)
            ambiguous_source_sites.add(key)
        else:
            selected_sites[key] = site
    symmetry_ids = {row.symmetry_id for row in selected_rows}
    bonds: list[SystemBond] = []
    allowed_connection_types = {
        "covale",
        "covale_base",
        "covale_phosphate",
        "covale_sugar",
        "disulf",
        "modres",
        "modres_link",
    }
    for index, conn_type in enumerate(conn_types):
        if conn_type is None or conn_type.casefold() not in allowed_connection_types:
            continue
        if symmetry_1[index] not in {None, "1_555"} or symmetry_2[index] not in {None, "1_555"}:
            continue
        original_label_component_1 = label_comp_1[index]
        original_label_component_2 = label_comp_2[index]
        normalized_label_component_1 = (
            None if original_label_component_1 is None else _normalized_component_id(original_label_component_1)
        )
        normalized_label_component_2 = (
            None if original_label_component_2 is None else _normalized_component_id(original_label_component_2)
        )
        endpoint_1 = (
            label_asym_1[index],
            optional_integer(label_seq_1[index]),
            ins_1[index],
            normalized_label_component_1,
            (
                label_atom_1[index]
                if original_label_component_1 is None or label_atom_1[index] is None
                else _normalized_atom_id(original_label_component_1, label_atom_1[index])
            ),
        )
        endpoint_2 = (
            label_asym_2[index],
            optional_integer(label_seq_2[index]),
            ins_2[index],
            normalized_label_component_2,
            (
                label_atom_2[index]
                if original_label_component_2 is None or label_atom_2[index] is None
                else _normalized_atom_id(original_label_component_2, label_atom_2[index])
            ),
        )
        original_auth_component_1 = auth_comp_1[index] or original_label_component_1
        original_auth_component_2 = auth_comp_2[index] or original_label_component_2
        auth_endpoint_1 = (
            auth_asym_1[index] or label_asym_1[index],
            present_string(auth_seq_1[index]) or label_seq_1[index],
            ins_1[index],
            None if original_auth_component_1 is None else _normalized_component_id(original_auth_component_1),
            (
                auth_atom_1[index] or label_atom_1[index]
                if original_auth_component_1 is None
                else _normalized_atom_id(
                    original_auth_component_1,
                    auth_atom_1[index] or label_atom_1[index] or "",
                )
            ),
        )
        auth_endpoint_2 = (
            auth_asym_2[index] or label_asym_2[index],
            present_string(auth_seq_2[index]) or label_seq_2[index],
            ins_2[index],
            None if original_auth_component_2 is None else _normalized_component_id(original_auth_component_2),
            (
                auth_atom_2[index] or label_atom_2[index]
                if original_auth_component_2 is None
                else _normalized_atom_id(
                    original_auth_component_2,
                    auth_atom_2[index] or label_atom_2[index] or "",
                )
            ),
        )
        for symmetry_id in symmetry_ids:
            resolved_sites: list[AtomSite | None] = []
            for endpoint_number, (label_endpoint, auth_endpoint) in enumerate(
                ((endpoint_1, auth_endpoint_1), (endpoint_2, auth_endpoint_2)),
                start=1,
            ):
                keys = ((symmetry_id, *label_endpoint), (symmetry_id, *auth_endpoint))
                candidates = {selected_sites[key] for key in keys if key in selected_sites}
                if len(candidates) > 1 or (not candidates and any(key in ambiguous_source_sites for key in keys)):
                    raise ValueError(f"mmCIF struct_conn row {index} endpoint {endpoint_number} is ambiguous")
                resolved_sites.append(next(iter(candidates)) if candidates else None)
            site_a, site_b = resolved_sites
            if site_a is None or site_b is None:
                # A struct_conn endpoint may reference an atom absent from the selected model -- an
                # unmodeled cap or modified residue (ACE, NH2, TPO, ...), or a partner in another
                # assembly. The bond is then unrepresentable, so skip it rather than dropping the
                # whole structure (excluded water lands here too, its sites having been removed).
                continue
            if site_a.instance_id == site_b.instance_id and site_a.component_index == site_b.component_index:
                continue
            bonds.append(
                SystemBond(
                    site_a,
                    site_b,
                    order=1.0,
                    aromatic=False,
                    role="assembly",
                )
            )
    return tuple(bonds)


def write_mmcif(structure: Structure, path: Path) -> None:
    r"""Write the observed projection and molecular declarations as standard mmCIF."""

    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    system = structure.system.cpu()  # type: ignore[attr-defined]
    coordinates = structure.coordinates.detach().cpu()
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    occupancy = structure.occupancy.detach().cpu()
    occupancy_mask = structure.occupancy_mask.detach().cpu()
    b_factor = structure.b_factor.detach().cpu()
    b_factor_mask = structure.b_factor_mask.detach().cpu()
    residue_id = structure.residue_id.detach().cpu()
    insertion_code = structure.insertion_code.detach().cpu()
    instance_entity = system.instance_entity.detach()
    entity_type = system.entity_type.detach()
    entity_molecule_type = system.entity_molecule_type.detach()
    instance_component_ptr = system.instance_component_ptr.detach()
    component_symbol = system.component_symbol.detach()
    component_topology_complete = system.component_topology_complete.detach()
    component_atom_ptr = system.component_atom_ptr.detach()
    atom_symbol = system.atom_symbol.detach()
    atom_stereo = system.atom_stereo.detach()
    atomic_number = system.atomic_number.detach()
    isotope = system.isotope.detach()
    isotope_known = system.isotope_known.detach()
    formal_charge = system.formal_charge.detach()
    formal_charge_known = system.formal_charge_known.detach()
    atom_aromatic = system.atom_aromatic.detach()
    atom_aromatic_known = system.atom_aromatic_known.detach()
    reference_coordinates = system.reference_coordinates.detach()
    reference_coordinate_mask = system.reference_coordinate_mask.detach()
    edge_index = system.edge_index.detach()
    bond_order = system.bond_order.detach()
    bond_order_known = system.bond_order_known.detach()
    bond_aromatic = system.bond_aromatic.detach()
    bond_aromatic_known = system.bond_aromatic_known.detach()
    bond_kind = system.bond_kind.detach()
    bond_direction = system.bond_direction.detach()
    bond_role = system.bond_role.detach()
    bond_stereo = system.bond_stereo.detach()
    bond_stereo_index = system.bond_stereo_index.detach()
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}

    instance_count = int(system.instance_symbol.shape[0])
    component_count = int(component_symbol.shape[0])
    atom_count = int(atom_symbol.shape[0])
    component_instance = [0] * component_count
    atom_component = [0] * atom_count
    for instance_index in range(instance_count):
        start = int(instance_component_ptr[instance_index])
        stop = int(instance_component_ptr[instance_index + 1])
        for component_index in range(start, stop):
            component_instance[component_index] = instance_index
            atom_start = int(component_atom_ptr[component_index])
            atom_stop = int(component_atom_ptr[component_index + 1])
            atom_component[atom_start:atom_stop] = [component_index] * (atom_stop - atom_start)
    label_asym_by_instance: dict[int, str] = {}
    used_asym_ids: set[str] = set()
    for instance_index in range(instance_count):
        label_asym_by_instance[instance_index] = unique_identifier(
            system.instance_id(instance_index),
            used_asym_ids,
        )
    entity_id_by_index: dict[int, str] = {}
    used_entity_ids: set[str] = set()
    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id_by_index[entity_index] = unique_identifier(
            system.entity_id(entity_index),
            used_entity_ids,
        )

    def component_id(component_index: int) -> str:
        return system.symbols[int(component_symbol[component_index])]

    def atom_id(atom_index: int) -> str:
        return system.symbols[int(atom_symbol[atom_index])]

    def element(atom_index: int) -> str:
        number = int(atomic_number[atom_index])
        if number == 1 and bool(isotope_known[atom_index]):
            if int(isotope[atom_index]) == 2:
                return "D"
            if int(isotope[atom_index]) == 3:
                return "T"
        return element_by_number[number]

    def source_value(values: torch.Tensor, index: int) -> str | None:
        symbol_index = int(values[index])
        return None if symbol_index < 0 else structure.source_symbols[symbol_index]

    def authoritative_occurrence_component(
        component_index: int,
    ) -> ChemicalComponent:
        component = get_component(component_id(component_index))
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        return component if bool((atomic_number[start:stop] == 1).any()) else component.without_hydrogens()

    def order_name(index: int) -> str:
        if not bool(bond_order_known[index]):
            return "?"
        if bool(bond_aromatic_known[index]) and bool(bond_aromatic[index]) and float(bond_order[index]) == 1.5:
            return "arom"
        try:
            return {
                1.0: "sing",
                2.0: "doub",
                3.0: "trip",
                4.0: "quad",
            }[float(bond_order[index])]
        except KeyError:
            raise ValueError("mmCIF cannot encode known bond order " f"{float(bond_order[index])!r}") from None

    def connection_order_name(index: int) -> str:
        value = order_name(index)
        decoded_order, decoded_aromatic = _cif_bond_order(None if value == "?" else value)
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError(
                "mmCIF struct_conn cannot encode partially known or " "inconsistent bond order and aromaticity"
            )
        return value

    def component_bond_values(index: int) -> tuple[str, str]:
        order_value = order_name(index)
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        decoded_order, order_aromatic = _cif_component_bond_order(None if order_value == "?" else order_value)
        if expected_aromatic is None:
            decoded_aromatic = order_aromatic
            aromatic_value = "?"
        else:
            if order_aromatic is not None and order_aromatic != expected_aromatic:
                raise ValueError("mmCIF chem_comp_bond cannot encode inconsistent bond " "order and aromaticity")
            decoded_aromatic = expected_aromatic
            aromatic_value = "Y" if expected_aromatic else "N"
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError("mmCIF chem_comp_bond cannot encode partially known bond " "order and aromaticity")
        return order_value, aromatic_value

    entity_columns: dict[str, list[object]] = {"id": [], "type": []}
    struct_asym_columns: dict[str, list[object]] = {"id": [], "entity_id": []}
    entity_poly_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    entity_poly_seq_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "mon_id": [],
        "hetero": [],
    }
    poly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "seq_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "auth_seq_num": [],
        "pdb_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
        "hetero": [],
    }
    branch_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    nonpoly_columns: dict[str, list[object]] = {
        "entity_id": [],
        "name": [],
        "comp_id": [],
    }
    nonpoly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "pdb_seq_num": [],
        "auth_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
    }
    branch_list_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "comp_id": [],
        "hetero": [],
    }
    branch_link_columns: dict[str, list[object]] = {
        "entity_id": [],
        "link_id": [],
        "entity_branch_list_num_1": [],
        "comp_id_1": [],
        "atom_id_1": [],
        "leaving_atom_id_1": [],
        "entity_branch_list_num_2": [],
        "comp_id_2": [],
        "atom_id_2": [],
        "leaving_atom_id_2": [],
        "value_order": [],
    }
    branch_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "num": [],
        "pdb_asym_id": [],
        "pdb_mon_id": [],
        "pdb_seq_num": [],
        "pdb_ins_code": [],
        "auth_asym_id": [],
        "auth_mon_id": [],
        "auth_seq_num": [],
    }

    entity_type_name = {
        ENTITY_PROTEIN: "polypeptide(L)",
        ENTITY_RNA: "polyribonucleotide",
        ENTITY_DNA: "polydeoxyribonucleotide",
        ENTITY_DNA_RNA_HYBRID: "polydeoxyribonucleotide/polyribonucleotide hybrid",
        ENTITY_PNA: "peptide nucleic acid",
    }
    first_instance_by_entity: dict[int, int] = {}
    instances_by_entity: dict[int, list[int]] = {}
    branch_leaving_by_endpoint: dict[tuple[int, int], str] = {}
    branch_leaving_by_component: dict[int, frozenset[str]] = {}
    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        first_instance_by_entity.setdefault(entity_index, instance_index)
        instances_by_entity.setdefault(entity_index, []).append(instance_index)
        struct_asym_columns["id"].append(label_asym_by_instance[instance_index])
        struct_asym_columns["entity_id"].append(entity_id_by_index[entity_index])

    def instance_components(instance_index: int) -> tuple[int, ...]:
        return tuple(
            range(
                int(instance_component_ptr[instance_index]),
                int(instance_component_ptr[instance_index + 1]),
            )
        )

    def local_atom(instance_index: int, atom_index: int) -> tuple[int, str]:
        component_index = atom_component[atom_index]
        if component_instance[component_index] != instance_index:
            raise ValueError("an entity-owned bond crosses molecular instances")
        return (
            component_index - int(instance_component_ptr[instance_index]),
            atom_id(atom_index),
        )

    def owned_edge_signature(
        instance_index: int,
        role: int,
    ) -> tuple[tuple[object, ...], ...]:
        rows = []
        for edge in range(edge_index.shape[1]):
            if int(bond_role[edge]) != role:
                continue
            atom_a = int(edge_index[0, edge])
            atom_b = int(edge_index[1, edge])
            if component_instance[atom_component[atom_a]] != instance_index:
                continue
            endpoint_a = local_atom(instance_index, atom_a)
            endpoint_b = local_atom(instance_index, atom_b)
            stereo_atoms = tuple(
                (None if int(atom) < 0 else local_atom(instance_index, int(atom)))
                for atom in bond_stereo_index[:, edge]
            )
            direction = int(bond_direction[edge])
            leaving_atoms: tuple[str | None, str | None] | None = None
            if role == BOND_ROLE_ENTITY:
                leaving_atoms = (
                    branch_leaving_by_endpoint.get((edge, 0)),
                    branch_leaving_by_endpoint.get((edge, 1)),
                )
            if endpoint_b < endpoint_a:
                endpoint_a, endpoint_b = endpoint_b, endpoint_a
                stereo_atoms = stereo_atoms[::-1]
                direction = -direction
                if leaving_atoms is not None:
                    leaving_atoms = leaving_atoms[::-1]
            stereo = int(bond_stereo[edge])
            row = (
                endpoint_a,
                endpoint_b,
                (
                    bool(bond_order_known[edge]),
                    float(bond_order[edge]),
                ),
                (
                    bool(bond_aromatic_known[edge]),
                    bool(bond_aromatic[edge]),
                ),
                int(bond_kind[edge]),
                direction,
                None if stereo < 0 else system.symbols[stereo],
                stereo_atoms,
            )
            rows.append(row if leaving_atoms is None else (*row, leaving_atoms))
        return tuple(sorted(rows))

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_a = atom_component[atom_a]
        component_b = atom_component[atom_b]
        instance_a = component_instance[component_a]
        instance_b = component_instance[component_b]
        if role == BOND_ROLE_COMPONENT and component_a != component_b:
            raise ValueError("a component-owned bond crosses component occurrences")
        if role in {BOND_ROLE_POLYMER, BOND_ROLE_ENTITY} and instance_a != instance_b:
            raise ValueError("an entity-owned bond crosses molecular instances")
        if role == BOND_ROLE_POLYMER and component_a == component_b:
            raise ValueError("a polymer-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY and component_a == component_b:
            raise ValueError("an entity-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY:
            entity_index = int(instance_entity[instance_a])
            if (
                int(entity_type[entity_index]) != ENTITY_MOLECULE
                or int(entity_molecule_type[entity_index]) != MOLECULE_BRANCHED
            ):
                raise ValueError("only a branched molecule can carry entity-owned bonds")

    branch_endpoints_by_component: dict[int, list[tuple[int, int, str]]] = {}
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_ENTITY:
            continue
        for side in range(2):
            atom = int(edge_index[side, edge])
            branch_endpoints_by_component.setdefault(
                atom_component[atom],
                [],
            ).append((edge, side, atom_id(atom)))

    for component_index, endpoints in branch_endpoints_by_component.items():
        try:
            component = authoritative_occurrence_component(component_index)
        except KeyError:
            continue
        if component.bonds is None:
            raise ValueError(f"component {component.component_id!r} has no known bond graph")
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        present = {atom_id(atom) for atom in range(start, stop)}
        missing = {atom.atom_id for atom in component.atoms} - present
        selected: set[str] = set()
        for edge, side, endpoint in endpoints:
            candidates = set()
            for bond in component.bonds:
                if bond.atom_a == endpoint and bond.atom_b in missing:
                    candidates.add(bond.atom_b)
                elif bond.atom_b == endpoint and bond.atom_a in missing:
                    candidates.add(bond.atom_a)
            if len(candidates) > 1:
                raise ValueError(
                    "mmCIF cannot infer a unique leaving atom for " f"{component.component_id!r}:{endpoint}"
                )
            if candidates:
                leaving_atom = next(iter(candidates))
                if leaving_atom in selected:
                    raise ValueError("mmCIF cannot assign one leaving atom to multiple " "branched-entity links")
                selected.add(leaving_atom)
                branch_leaving_by_endpoint[(edge, side)] = leaving_atom
        branch_leaving_by_component[component_index] = frozenset(selected)

    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id = entity_id_by_index[entity_index]
        type_code = int(entity_type[entity_index])
        first_instance = first_instance_by_entity[entity_index]
        components = list(instance_components(first_instance))
        component_start = components[0] if components else int(instance_component_ptr[first_instance])
        component_ids_by_position: list[tuple[str, ...]] = []
        for instance_index in instances_by_entity[entity_index]:
            if len(instance_components(instance_index)) != len(components):
                raise ValueError(f"mmCIF entity {entity_id!r} has inconsistent component-axis lengths")
        for local_index in range(len(components)):
            component_ids = tuple(
                dict.fromkeys(
                    component_id(instance_components(instance_index)[local_index])
                    for instance_index in instances_by_entity[entity_index]
                )
            )
            component_ids_by_position.append(component_ids)
        if type_code == ENTITY_MOLECULE:
            if any(len(component_ids) != 1 for component_ids in component_ids_by_position):
                raise ValueError(f"mmCIF cannot encode divergent molecule instances for entity {entity_id!r}")
            first_entity_edges = owned_edge_signature(
                first_instance,
                BOND_ROLE_ENTITY,
            )
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY) != first_entity_edges
                for instance_index in instances_by_entity[entity_index][1:]
            ):
                raise ValueError(f"mmCIF cannot encode divergent entity bonds for entity {entity_id!r}")
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_POLYMER)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a molecule cannot carry polymer-owned bonds")
            molecule_type = int(entity_molecule_type[entity_index])
            if molecule_type == MOLECULE_BRANCHED:
                entity_category = "branched"
            elif molecule_type in {
                MOLECULE_NON_POLYMER,
                MOLECULE_WATER,
            }:
                if len(components) != 1 or first_entity_edges:
                    category = "water" if molecule_type == MOLECULE_WATER else "non-polymer"
                    raise ValueError(f"mmCIF {category} entities require one component " "and no entity-owned bonds")
                entity_category = "water" if molecule_type == MOLECULE_WATER else "non-polymer"
            else:
                raise ValueError(f"unsupported molecule type code {molecule_type}")
        else:
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a polymer cannot carry entity-owned bonds")
            for instance_index in instances_by_entity[entity_index]:
                expected_polymer_edges = []
                occurrence_components = instance_components(instance_index)
                for local_index, (left_index, right_index) in enumerate(
                    zip(occurrence_components, occurrence_components[1:])
                ):
                    try:
                        left = get_component(component_id(left_index))
                        right = get_component(component_id(right_index))
                    except KeyError:
                        continue
                    left_start = int(component_atom_ptr[left_index])
                    left_stop = int(component_atom_ptr[left_index + 1])
                    right_start = int(component_atom_ptr[right_index])
                    right_stop = int(component_atom_ptr[right_index + 1])
                    left_attachment = left.next_attachment
                    right_attachment = right.previous_attachment
                    if (
                        left_attachment is not None
                        and not bool((atomic_number[left_start:left_stop] == 1).any())
                        and next(atom.element for atom in left.atoms if atom.atom_id == left_attachment.atom_id)
                        in {"H", "D", "T"}
                    ):
                        left_attachment = None
                    if (
                        right_attachment is not None
                        and not bool((atomic_number[right_start:right_stop] == 1).any())
                        and next(atom.element for atom in right.atoms if atom.atom_id == right_attachment.atom_id)
                        in {"H", "D", "T"}
                    ):
                        right_attachment = None
                    if left_attachment is None or right_attachment is None:
                        continue
                    endpoint_a = (local_index, left_attachment.atom_id)
                    endpoint_b = (
                        local_index + 1,
                        right_attachment.atom_id,
                    )
                    expected_polymer_edges.append(
                        (
                            endpoint_a,
                            endpoint_b,
                            (True, 1.0),
                            (True, False),
                            BOND_KIND_COVALENT,
                            BOND_DIRECTION_UNDIRECTED,
                            None,
                            (None, None),
                        )
                    )
                if owned_edge_signature(
                    instance_index,
                    BOND_ROLE_POLYMER,
                ) != tuple(expected_polymer_edges):
                    raise ValueError(f"mmCIF cannot reconstruct polymer bonds for entity {entity_id!r}")
            entity_category = "polymer"
        entity_columns["id"].append(entity_id)
        entity_columns["type"].append(entity_category)

        if type_code != ENTITY_MOLECULE:
            entity_poly_columns["entity_id"].append(entity_id)
            entity_poly_columns["type"].append(entity_type_name[type_code])
            for local_index, candidate_ids in enumerate(
                component_ids_by_position,
                1,
            ):
                if len(candidate_ids) > 1:
                    for instance_index in instances_by_entity[entity_index]:
                        occurrence = instance_components(instance_index)[local_index - 1]
                        atom_start = int(component_atom_ptr[occurrence])
                        atom_stop = int(component_atom_ptr[occurrence + 1])
                        if not bool(coordinate_mask[atom_start:atom_stop].any()):
                            raise ValueError(
                                "mmCIF cannot select an unobserved " "microheterogeneous polymer component"
                            )
                for candidate in candidate_ids:
                    entity_poly_seq_columns["entity_id"].append(entity_id)
                    entity_poly_seq_columns["num"].append(local_index)
                    entity_poly_seq_columns["mon_id"].append(candidate)
                    entity_poly_seq_columns["hetero"].append("y" if len(candidate_ids) > 1 else "n")
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    author_number = source_value(residue_id, component_index)
                    monomer = component_id(component_index)
                    asym_id = label_asym_by_instance[instance_index]
                    poly_scheme_values = (
                        asym_id,
                        entity_id,
                        local_index + 1,
                        monomer,
                        local_index + 1,
                        author_number or "?",
                        author_number or "?",
                        monomer,
                        monomer,
                        asym_id,
                        source_value(insertion_code, component_index) or "?",
                        ("y" if len(component_ids_by_position[local_index]) > 1 else "n"),
                    )
                    for name, value in zip(poly_scheme_columns, poly_scheme_values):
                        poly_scheme_columns[name].append(value)
        elif int(entity_molecule_type[entity_index]) == MOLECULE_BRANCHED:
            branch_columns["entity_id"].append(entity_id)
            branch_columns["type"].append("oligosaccharide")
            for local_index, component_index in enumerate(components, 1):
                branch_list_columns["entity_id"].append(entity_id)
                branch_list_columns["num"].append(local_index)
                branch_list_columns["comp_id"].append(component_id(component_index))
                branch_list_columns["hetero"].append("n")

            link_id = 1
            for bond_index in range(edge_index.shape[1]):
                if int(bond_role[bond_index]) != BOND_ROLE_ENTITY:
                    continue
                atom_a = int(edge_index[0, bond_index])
                atom_b = int(edge_index[1, bond_index])
                component_a = atom_component[atom_a]
                component_b = atom_component[atom_b]
                if (
                    component_instance[component_a] != first_instance
                    or component_instance[component_b] != first_instance
                ):
                    continue
                local_a = component_a - component_start
                local_b = component_b - component_start
                if local_a == local_b:
                    continue
                branch_link_values = (
                    entity_id,
                    link_id,
                    local_a + 1,
                    component_id(component_a),
                    atom_id(atom_a),
                    branch_leaving_by_endpoint.get((bond_index, 0), "?"),
                    local_b + 1,
                    component_id(component_b),
                    atom_id(atom_b),
                    branch_leaving_by_endpoint.get((bond_index, 1), "?"),
                    connection_order_name(bond_index),
                )
                for name, value in zip(branch_link_columns, branch_link_values):
                    branch_link_columns[name].append(value)
                link_id += 1

            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                monomer_counts = {
                    component_id(index): sum(component_id(other) == component_id(index) for other in components)
                    for index in components
                }
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    source_author_number = source_value(
                        residue_id,
                        component_index,
                    )
                    monomer = component_id(component_index)
                    atom_start = int(component_atom_ptr[component_index])
                    atom_stop = int(component_atom_ptr[component_index + 1])
                    if (
                        source_author_number is None
                        and monomer_counts[monomer] > 1
                        and bool(coordinate_mask[atom_start:atom_stop].any())
                    ):
                        raise ValueError(
                            "mmCIF cannot identify an observed repeated "
                            "branched-entity monomer without an author "
                            "sequence number"
                        )
                    auth_number = source_author_number or "?"
                    branch_scheme_values = (
                        asym_id,
                        entity_id,
                        monomer,
                        local_index + 1,
                        asym_id,
                        monomer,
                        auth_number,
                        source_value(insertion_code, component_index) or "?",
                        asym_id,
                        monomer,
                        auth_number,
                    )
                    for name, value in zip(branch_scheme_columns, branch_scheme_values):
                        branch_scheme_columns[name].append(value)
        else:
            monomer = component_id(components[0])
            nonpoly_columns["entity_id"].append(entity_id)
            nonpoly_columns["name"].append(monomer)
            nonpoly_columns["comp_id"].append(monomer)
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                component_index = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                author_number = source_value(residue_id, component_index) or "?"
                nonpoly_scheme_values = (
                    asym_id,
                    entity_id,
                    monomer,
                    1,
                    author_number,
                    author_number,
                    monomer,
                    monomer,
                    asym_id,
                    source_value(insertion_code, component_index) or "?",
                )
                for name, value in zip(nonpoly_scheme_columns, nonpoly_scheme_values):
                    nonpoly_scheme_columns[name].append(value)

    chem_comp_columns: dict[str, list[object]] = {"id": [], "type": []}
    chem_atom_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id": [],
        "type_symbol": [],
        "charge": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
        "model_Cartn_x": [],
        "model_Cartn_y": [],
        "model_Cartn_z": [],
    }
    chem_bond_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id_1": [],
        "atom_id_2": [],
        "value_order": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
    }
    component_bond_indices: dict[int, list[int]] = {}
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] == component_index:
            component_bond_indices.setdefault(component_index, []).append(bond_index)

    def component_definition(index: int) -> tuple[object, ...]:
        start = int(component_atom_ptr[index])
        stop = int(component_atom_ptr[index + 1])
        atoms = tuple(
            (
                atom_id(atom_index),
                element(atom_index),
                (int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else None),
                (bool(atom_aromatic[atom_index]) if bool(atom_aromatic_known[atom_index]) else None),
                (system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else None),
                (
                    tuple(float(value) for value in reference_coordinates[atom_index])
                    if bool(reference_coordinate_mask[atom_index])
                    else None
                ),
            )
            for atom_index in range(start, stop)
        )
        bonds = tuple(
            (
                atom_id(int(edge_index[0, bond_index])),
                atom_id(int(edge_index[1, bond_index])),
                *component_bond_values(bond_index),
                (system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else None),
            )
            for bond_index in component_bond_indices.get(index, ())
        )
        return bool(component_topology_complete[index]), atoms, bonds

    def graph_component_signature(
        graph: MolecularSystem,
        index: int,
    ) -> tuple[object, ...]:
        start = int(graph.component_atom_ptr[index])
        stop = int(graph.component_atom_ptr[index + 1])
        atoms = tuple(
            (
                graph.symbols[int(graph.atom_symbol[atom])],
                int(graph.atomic_number[atom]),
                int(graph.isotope[atom]) if bool(graph.isotope_known[atom]) else None,
                (int(graph.formal_charge[atom]) if bool(graph.formal_charge_known[atom]) else None),
                (int(graph.radical_electrons[atom]) if bool(graph.radical_electrons_known[atom]) else None),
                (int(graph.explicit_hydrogens[atom]) if bool(graph.explicit_hydrogens_known[atom]) else None),
                (int(graph.implicit_hydrogens[atom]) if bool(graph.implicit_hydrogens_known[atom]) else None),
                (
                    bool(graph.implicit_hydrogens_allowed[atom])
                    if bool(graph.implicit_hydrogens_allowed_known[atom])
                    else None
                ),
                (bool(graph.atom_aromatic[atom]) if bool(graph.atom_aromatic_known[atom]) else None),
                (None if int(graph.atom_stereo[atom]) < 0 else graph.symbols[int(graph.atom_stereo[atom])]),
                (
                    tuple(float(value) for value in graph.reference_coordinates[atom])
                    if bool(graph.reference_coordinate_mask[atom])
                    else None
                ),
            )
            for atom in range(start, stop)
        )
        edges = []
        for edge in range(graph.edge_index.shape[1]):
            if int(graph.bond_role[edge]) != BOND_ROLE_COMPONENT:
                continue
            atom_a = int(graph.edge_index[0, edge])
            atom_b = int(graph.edge_index[1, edge])
            if not (start <= atom_a and atom_b < stop):
                continue
            stereo = int(graph.bond_stereo[edge])
            stereo_atoms = graph.bond_stereo_index[:, edge]
            edges.append(
                (
                    atom_a - start,
                    atom_b - start,
                    (float(graph.bond_order[edge]) if bool(graph.bond_order_known[edge]) else None),
                    (bool(graph.bond_aromatic[edge]) if bool(graph.bond_aromatic_known[edge]) else None),
                    int(graph.bond_kind[edge]),
                    int(graph.bond_direction[edge]),
                    None if stereo < 0 else graph.symbols[stereo],
                    tuple(-1 if int(atom) < 0 else int(atom) - start for atom in stereo_atoms),
                )
            )
        return (
            bool(graph.component_topology_complete[index]),
            atoms,
            tuple(edges),
        )

    def known_component_signatures(
        component_index: int,
        expected: tuple[object, ...],
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        actual = graph_component_signature(cpu_system, component_index)
        actual_atoms = cast(tuple[tuple[object, ...], ...], actual[1])
        expected_atoms = cast(tuple[tuple[object, ...], ...], expected[1])
        if len(actual_atoms) != len(expected_atoms):
            return actual, expected

        normalized_atoms = []
        start = int(component_atom_ptr[component_index])
        for offset, (actual_atom, expected_atom) in enumerate(zip(actual_atoms, expected_atoms)):
            normalized = list(expected_atom)
            if actual_atom[:2] == expected_atom[:2]:
                observed = bool(coordinate_mask[start + offset])
                if actual_atom[2] != expected_atom[2]:
                    if not observed or actual_atom[1] != 1 or actual_atom[2] not in {2, 3}:
                        raise ValueError("mmCIF cannot encode this isotope difference " "for a known component")
                    normalized[2] = actual_atom[2]
                if actual_atom[3] != expected_atom[3]:
                    if not observed or actual_atom[3] is None:
                        raise ValueError("mmCIF cannot encode this formal-charge " "difference for a known component")
                    normalized[3] = actual_atom[3]
            normalized_atoms.append(tuple(normalized))
        return actual, (expected[0], tuple(normalized_atoms), expected[2])

    components_by_id: dict[str, list[int]] = {}
    for component_index in range(component_count):
        components_by_id.setdefault(component_id(component_index), []).append(component_index)
    embedded_component_by_id: dict[str, int] = {}
    cpu_system = system
    polymer_types: dict[int, PolymerType] = {
        ENTITY_PROTEIN: "protein",
        ENTITY_RNA: "rna",
        ENTITY_DNA: "dna",
        ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
        ENTITY_PNA: "pna",
    }
    polymer_pairs = {
        tuple(
            sorted(
                (
                    atom_component[int(edge_index[0, edge])],
                    atom_component[int(edge_index[1, edge])],
                )
            )
        )
        for edge in range(edge_index.shape[1])
        if int(bond_role[edge]) == BOND_ROLE_POLYMER
    }

    expected_definitions: dict[
        tuple[str, int, bool, bool, bool, tuple[str, ...]],
        tuple[object, ...],
    ] = {}
    for name, component_indices in components_by_id.items():
        try:
            get_component(name)
        except KeyError:
            first = component_indices[0]
            definition = component_definition(first)
            if any(component_definition(component_index) != definition for component_index in component_indices[1:]):
                raise ValueError("mmCIF cannot encode conflicting realized definitions for " f"component ID {name!r}")
            embedded_component_by_id[name] = first
            continue

        for component_index in component_indices:
            instance_index = component_instance[component_index]
            component_start = int(instance_component_ptr[instance_index])
            component_stop = int(instance_component_ptr[instance_index + 1])
            has_previous = component_index > component_start and (component_index - 1, component_index) in polymer_pairs
            has_next = component_index + 1 < component_stop and (component_index, component_index + 1) in polymer_pairs
            entity_index = int(instance_entity[instance_index])
            type_code = int(entity_type[entity_index])
            start = int(component_atom_ptr[component_index])
            stop = int(component_atom_ptr[component_index + 1])
            has_hydrogens = bool((atomic_number[start:stop] == 1).any())
            leaving_atoms = tuple(sorted(branch_leaving_by_component.get(component_index, ())))
            key = (
                name,
                type_code,
                has_previous,
                has_next,
                has_hydrogens,
                leaving_atoms,
            )
            expected = expected_definitions.get(key)
            if expected is None:
                expected_component = authoritative_occurrence_component(component_index)
                if leaving_atoms:
                    expected_component = _remove_component_atoms(
                        expected_component,
                        set(leaving_atoms),
                    )
                if type_code == ENTITY_MOLECULE:
                    expected_entity: Molecule | PolymerSequence = Molecule((expected_component,))
                    expected_index = 0
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
                    expected_components = (
                        *((left,) if has_previous else ()),
                        expected_component,
                        *((right,) if has_next else ()),
                    )
                    expected_entity = PolymerSequence(
                        polymer_types[type_code],
                        expected_components,
                    )
                    expected_index = int(has_previous)
                expected_system = MolecularSystem.from_entities(
                    {"expected": expected_entity},
                    {"expected": "expected"},
                )
                expected = graph_component_signature(
                    expected_system,
                    expected_index,
                )
                expected_definitions[key] = expected
            actual, normalized_expected = known_component_signatures(
                component_index,
                expected,
            )
            if actual != normalized_expected:
                raise ValueError(f"component ID {name!r} does not match its authoritative definition")

    for name in embedded_component_by_id:
        for component_index in components_by_id[name]:
            start = int(component_atom_ptr[component_index])
            stop = int(component_atom_ptr[component_index + 1])
            decoded_topology_complete = stop - start == 1 or bool(component_bond_indices.get(component_index))
            if bool(component_topology_complete[component_index]) != decoded_topology_complete:
                raise ValueError("mmCIF cannot encode this unknown component's topology completeness")
            isotope_representable = (atomic_number[start:stop] == 1) & torch.isin(
                isotope[start:stop],
                isotope.new_tensor((2, 3)),
            )
            if bool((isotope_known[start:stop] & ~isotope_representable).any()):
                raise ValueError(
                    "mmCIF cannot encode explicit isotopes other than deuterium " "and tritium for an unknown component"
                )
            for field in (
                "radical_electrons_known",
                "explicit_hydrogens_known",
                "implicit_hydrogens_known",
                "implicit_hydrogens_allowed_known",
            ):
                if bool(getattr(cpu_system, field)[start:stop].any()):
                    raise ValueError(
                        f"mmCIF cannot encode known {field.removesuffix('_known')} for an unknown component"
                    )
            for edge in component_bond_indices.get(component_index, ()):
                if (
                    int(bond_kind[edge]) != BOND_KIND_COVALENT
                    or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
                    or bool((cpu_system.bond_stereo_index[:, edge] >= 0).any())
                ):
                    raise ValueError(
                        "mmCIF chem_comp_bond cannot encode directed, dative, " "or stereo-defining component bonds"
                    )

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        if role == BOND_ROLE_ENTITY and (
            int(bond_kind[edge]) != BOND_KIND_COVALENT
            or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
            or int(bond_stereo[edge]) >= 0
            or bool((cpu_system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF branched-entity links cannot encode directed, dative, " "or stereochemical bonds")
        if role >= BOND_ROLE_POLYMER and (
            int(bond_stereo[edge]) >= 0 or bool((cpu_system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF struct_conn cannot encode bond stereochemistry")
    component_type_by_code = {
        ENTITY_PROTEIN: "L-PEPTIDE LINKING",
        ENTITY_RNA: "RNA LINKING",
        ENTITY_DNA: "DNA LINKING",
        ENTITY_DNA_RNA_HYBRID: "DNA/RNA LINKING",
        ENTITY_PNA: "PEPTIDE NUCLEIC ACID LINKING",
        ENTITY_MOLECULE: "NON-POLYMER",
    }
    for name, component_index in embedded_component_by_id.items():
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        chem_comp_columns["id"].append(name)
        chem_comp_columns["type"].append(component_type_by_code[int(entity_type[entity_index])])
        atom_start = int(component_atom_ptr[component_index])
        atom_stop = int(component_atom_ptr[component_index + 1])
        for atom_index in range(atom_start, atom_stop):
            reference = reference_coordinates[atom_index]
            chem_atom_values = (
                name,
                atom_id(atom_index),
                element(atom_index),
                int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
                ("?" if not bool(atom_aromatic_known[atom_index]) else "Y" if bool(atom_aromatic[atom_index]) else "N"),
                (system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else "N"),
                float(reference[0]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[1]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[2]) if bool(reference_coordinate_mask[atom_index]) else "?",
            )
            for column, value in zip(chem_atom_columns, chem_atom_values):
                chem_atom_columns[column].append(value)

    emitted_component_bonds: set[tuple[str, str, str]] = set()
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] != component_index:
            continue
        name = component_id(component_index)
        if embedded_component_by_id.get(name) != component_index:
            continue
        component_bond_key = (name, atom_id(atom_a), atom_id(atom_b))
        if component_bond_key in emitted_component_bonds:
            continue
        emitted_component_bonds.add(component_bond_key)
        order_value, aromatic_value = component_bond_values(bond_index)
        chem_bond_values = (
            name,
            atom_id(atom_a),
            atom_id(atom_b),
            order_value,
            aromatic_value,
            (system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else "N"),
        )
        for column, value in zip(chem_bond_columns, chem_bond_values):
            chem_bond_columns[column].append(value)

    atom_site_columns: dict[str, list[object]] = {
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
    }
    endpoint_identity: dict[int, tuple[object, ...]] = {}
    serial = 1
    for atom_index in range(atom_count):
        component_index = atom_component[atom_index]
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        local_component = component_index - int(instance_component_ptr[instance_index])
        polymer = int(entity_type[entity_index]) != ENTITY_MOLECULE
        asym_id = label_asym_by_instance[instance_index]
        monomer = component_id(component_index)
        source_author_number = source_value(residue_id, component_index)
        author_number = source_author_number or "?"
        code = source_value(insertion_code, component_index)
        label_sequence: object = local_component + 1 if polymer else "."
        endpoint_identity[atom_index] = (
            asym_id,
            label_sequence,
            monomer,
            atom_id(atom_index),
            asym_id,
            author_number,
            monomer,
            atom_id(atom_index),
            code or "?",
        )
        if not bool(coordinate_mask[atom_index]):
            continue
        xyz = coordinates[atom_index]
        if not bool(torch.isfinite(xyz).all()):
            raise ValueError("mmCIF cannot represent non-finite observed coordinates")
        atom_occupancy: object = float(occupancy[atom_index]) if bool(occupancy_mask[atom_index]) else "."
        atom_b_factor: object = float(b_factor[atom_index]) if bool(b_factor_mask[atom_index]) else "."
        if atom_occupancy != "." and not np.isfinite(atom_occupancy):
            raise ValueError("mmCIF cannot represent non-finite occupancy")
        if atom_b_factor != "." and not np.isfinite(atom_b_factor):
            raise ValueError("mmCIF cannot represent non-finite B factor")
        atom_site_values = (
            "ATOM" if polymer else "HETATM",
            serial,
            element(atom_index),
            atom_id(atom_index),
            ".",
            monomer,
            asym_id,
            entity_id_by_index[entity_index],
            label_sequence,
            code or "?",
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            atom_occupancy,
            atom_b_factor,
            int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
            author_number,
            monomer,
            asym_id,
            atom_id(atom_index),
            1,
        )
        for column, value in zip(atom_site_columns, atom_site_values):
            atom_site_columns[column].append(value)
        serial += 1

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
        "pdbx_ptnr1_PDB_ins_code": [],
        "pdbx_ptnr2_PDB_ins_code": [],
        "ptnr1_role": [],
        "ptnr2_role": [],
        "ptnr1_symmetry": [],
        "ptnr2_symmetry": [],
        "pdbx_value_order": [],
    }
    connection_id = 1
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) < BOND_ROLE_POLYMER:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        kind = int(bond_kind[bond_index])
        direction = int(bond_direction[bond_index])
        if kind == BOND_KIND_DATIVE:
            if direction == BOND_DIRECTION_ENDPOINT_0_DONOR:
                roles = ("donor", "acceptor")
            elif direction == BOND_DIRECTION_ENDPOINT_1_DONOR:
                roles = ("acceptor", "donor")
            else:
                raise ValueError("mmCIF struct_conn requires a donor direction for a dative bond")
            conn_type = "metalc"
        elif kind == BOND_KIND_COVALENT:
            if direction != BOND_DIRECTION_UNDIRECTED:
                raise ValueError("mmCIF struct_conn cannot encode a directed covalent bond")
            roles = ("?", "?")
            conn_type = "covale"
        else:
            raise ValueError(f"unknown canonical bond kind {kind}")
        value_order = connection_order_name(bond_index)
        if value_order == "arom":
            raise ValueError("mmCIF struct_conn pdbx_value_order does not support aromatic bonds")
        first_endpoint = endpoint_identity[atom_a]
        second_endpoint = endpoint_identity[atom_b]
        connection_values = (
            connection_id,
            conn_type,
            *first_endpoint[:4],
            *second_endpoint[:4],
            *first_endpoint[4:8],
            *second_endpoint[4:8],
            first_endpoint[8],
            second_endpoint[8],
            *roles,
            "1_555",
            "1_555",
            value_order,
        )
        for column, value in zip(conn_columns, connection_values):
            conn_columns[column].append(value)
        connection_id += 1

    categories: dict[str, object] = {
        "entity": CIFCategory(entity_columns),
        "struct_asym": CIFCategory(struct_asym_columns),
        "atom_site": CIFCategory(atom_site_columns),
    }
    for name, columns in (
        ("entity_poly", entity_poly_columns),
        ("entity_poly_seq", entity_poly_seq_columns),
        ("pdbx_poly_seq_scheme", poly_scheme_columns),
        ("pdbx_entity_branch", branch_columns),
        ("pdbx_entity_branch_list", branch_list_columns),
        ("pdbx_entity_branch_link", branch_link_columns),
        ("pdbx_branch_scheme", branch_scheme_columns),
        ("pdbx_entity_nonpoly", nonpoly_columns),
        ("pdbx_nonpoly_scheme", nonpoly_scheme_columns),
        ("chem_comp", chem_comp_columns),
        ("chem_comp_atom", chem_atom_columns),
        ("chem_comp_bond", chem_bond_columns),
        ("struct_conn", conn_columns),
    ):
        if columns and tuple(columns.values())[0]:
            categories[name] = CIFCategory(columns)
    CIFFile({"ofoldx": CIFBlock(categories)}).write(str(path))
