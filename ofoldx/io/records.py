r"""
Format-neutral atom records shared by structure codecs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import AtomSite, MolecularSystem, SystemBond

if TYPE_CHECKING:
    from biotite.structure.io.pdbx import CIFFile


@dataclass(frozen=True, slots=True)
class AtomRecord:
    r"""
    One format-neutral atom row parsed from PDB or mmCIF.

    Args:
        row_index: Zero-based row position in the selected model.
        group: PDB-style ``ATOM`` or ``HETATM`` record category.
        serial: Source atom serial when present.
        element: Chemical element symbol.
        isotope: Isotope mass number when the source element identifies one.
        charge: Formal charge when present.
        atom_name: Label atom identifier.
        component_id: Label component identifier.
        chain_id: Label chain identifier.
        entity_id: Label entity identifier when present.
        label_seq_id: Label sequence number when present.
        residue_id: Parsed residue number when present.
        auth_chain_id: Author chain identifier when present.
        auth_residue_id: Author residue number when present.
        insertion_code: Author insertion code when present.
        auth_component_id: Author component identifier when present.
        auth_atom_name: Author atom identifier when present.
        altloc_id: Alternate-location identifier when present.
        occupancy: Occupancy when present.
        b_factor: Isotropic B factor when present.
        coordinates: Cartesian coordinates in angstroms.
        coordinate_present: Whether the source contains usable coordinates.
        symmetry_id: Symmetry operation identifier when present.
        canonical_site: Exact atom occurrence supplied by embedded OFoldX metadata.
    """

    row_index: int
    group: Literal["ATOM", "HETATM"]
    serial: str | None
    element: str
    isotope: int | None
    charge: int | None
    atom_name: str
    component_id: str
    chain_id: str
    entity_id: str | None
    label_seq_id: int | None
    residue_id: int | None
    auth_chain_id: str | None
    auth_residue_id: int | None
    insertion_code: str | None
    auth_component_id: str | None
    auth_atom_name: str | None
    altloc_id: str | None
    occupancy: float | None
    b_factor: float | None
    coordinates: tuple[float, float, float]
    coordinate_present: bool
    symmetry_id: str | None = None
    canonical_site: AtomSite | None = None


@dataclass(frozen=True)
class ParsedStructure:
    r"""
    Format-neutral records and metadata parsed from one structure file.

    Args:
        format: Parsed file format.
        rows: Atom records from the selected model and assembly.
        canonical_payload: Embedded OFoldX metadata when present.
        mmcif_file: Parsed mmCIF document when the source is mmCIF.
        pdb_lines: Original PDB lines when the source is PDB.
    """

    format: Literal["pdb", "mmcif"]
    rows: tuple[AtomRecord, ...]
    canonical_payload: dict[str, object] | None
    mmcif_file: CIFFile | None = None
    pdb_lines: tuple[str, ...] = ()


class AtomArray(Protocol):
    r"""
    Structural Biotite atom-array contract required by record normalization.
    """

    chain_id: NDArray[np.str_]
    res_id: NDArray[np.integer]
    coord: NDArray[np.floating]
    hetero: NDArray[np.bool_]
    element: NDArray[np.str_]
    atom_name: NDArray[np.str_]
    res_name: NDArray[np.str_]
    ins_code: NDArray[np.str_]

    def get_annotation_categories(self) -> list[str]:
        r"""
        Return the available annotation names.

        Returns:
            Annotation category names exposed by the atom array.
        """

        ...

    def array_length(self) -> int:
        r"""
        Return the number of atom rows.

        Returns:
            The atom-array length.
        """

        ...

    def get_annotation(self, name: str) -> NDArray:
        r"""
        Return one annotation array.

        Args:
            name: Annotation category name.

        Returns:
            Values for the requested annotation.
        """

        ...


def atom_records_from_array(
    array: AtomArray,
    *,
    format_name: Literal["pdb", "mmcif"],
) -> tuple[AtomRecord, ...]:
    r"""
    Normalize one Biotite atom array into format-neutral atom records.

    Args:
        array: Atom array providing coordinates, identities, and optional annotations.
        format_name: Source format controlling author-field interpretation.

    Returns:
        Normalized atom records in source row order.

    Raises:
        ValueError: If canonical identity annotations or coordinate-mask values are invalid.
    """

    categories = set(array.get_annotation_categories())
    length = int(array.array_length())

    def annotation(name: str) -> NDArray | None:
        if name not in categories:
            return None
        return array.get_annotation(name)

    atom_serials = annotation("atom_id")
    charges = annotation("charge" if format_name == "pdb" else "pdbx_formal_charge")
    occupancies = annotation("occupancy")
    b_factors = annotation("b_factor")
    altloc_ids = annotation("altloc_id")
    symmetry_ids = annotation("sym_id")
    auth_asym_ids = annotation("auth_asym_id")
    auth_seq_ids = annotation("auth_seq_id")
    auth_comp_ids = annotation("auth_comp_id")
    auth_atom_ids = annotation("auth_atom_id")
    label_entity_ids = annotation("label_entity_id")
    label_asym_ids = annotation("label_asym_id")
    label_seq_ids = annotation("label_seq_id")
    canonical_instances = annotation("ofoldx_instance_id")
    canonical_components = annotation("ofoldx_component_index")
    canonical_atoms = annotation("ofoldx_atom_id")
    coordinate_masks = annotation("ofoldx_coordinate_mask")

    rows: list[AtomRecord] = []
    for index in range(length):
        if format_name == "pdb":
            chain_id = str(array.chain_id[index]).strip()
        else:
            chain_id = present_string(_at(label_asym_ids, index)) or present_string(array.chain_id[index]) or "_"
        label_seq_id = optional_integer(_at(label_seq_ids, index))
        residue_id = label_seq_id
        if residue_id is None:
            residue_id = int(array.res_id[index])
        canonical_site = None
        canonical_instance = present_string(_at(canonical_instances, index))
        canonical_component = present_string(_at(canonical_components, index))
        canonical_atom = present_string(_at(canonical_atoms, index))
        if canonical_instance is not None or canonical_component is not None or canonical_atom is not None:
            if canonical_instance is None or canonical_component is None or canonical_atom is None:
                raise ValueError("OFoldX canonical atom identity fields must be supplied together")
            canonical_component_index = optional_integer(canonical_component)
            if canonical_component_index is None:
                raise ValueError("ofoldx_component_index must be an integer")
            canonical_site = AtomSite(canonical_instance, canonical_component_index, canonical_atom)
        coordinate_present = True
        mask_text = present_string(_at(coordinate_masks, index))
        if mask_text is not None:
            if mask_text not in {"0", "1"}:
                raise ValueError("ofoldx_coordinate_mask values must be 0 or 1")
            coordinate_present = mask_text == "1"
        x, y, z = array.coord[index]
        if coordinate_present and not np.all(np.isfinite((x, y, z))):
            raise ValueError(
                f"atom record {index} has a non-finite present coordinate {(float(x), float(y), float(z))}; "
                "a present coordinate must be finite (mask it with ofoldx_coordinate_mask=0 or drop the atom)"
            )
        coordinate = (float(x), float(y), float(z))
        charge = optional_integer(_at(charges, index))
        if format_name == "pdb" and charge == 0:
            charge = None
        element = str(array.element[index]).strip().capitalize()
        isotope = None
        if element == "D":
            element = "H"
            isotope = 2
        elif element == "T":
            element = "H"
            isotope = 3
        rows.append(
            AtomRecord(
                row_index=index,
                group="HETATM" if bool(array.hetero[index]) else "ATOM",
                serial=present_string(_at(atom_serials, index)),
                element=element,
                isotope=isotope,
                charge=charge,
                atom_name=str(array.atom_name[index]),
                component_id=str(array.res_name[index]),
                chain_id=chain_id,
                entity_id=present_string(_at(label_entity_ids, index)),
                label_seq_id=label_seq_id,
                residue_id=residue_id,
                auth_chain_id=(
                    present_string(_at(auth_asym_ids, index))
                    if format_name == "mmcif"
                    else present_string(array.chain_id[index])
                ),
                auth_residue_id=(
                    optional_integer(_at(auth_seq_ids, index)) if format_name == "mmcif" else int(array.res_id[index])
                ),
                insertion_code=present_string(array.ins_code[index]),
                auth_component_id=(
                    present_string(_at(auth_comp_ids, index)) if format_name == "mmcif" else str(array.res_name[index])
                ),
                auth_atom_name=(
                    present_string(_at(auth_atom_ids, index)) if format_name == "mmcif" else str(array.atom_name[index])
                ),
                altloc_id=present_string(_at(altloc_ids, index)),
                occupancy=_optional_float(_at(occupancies, index)),
                b_factor=_optional_float(_at(b_factors, index)),
                coordinates=coordinate,
                coordinate_present=coordinate_present,
                symmetry_id=present_string(_at(symmetry_ids, index)),
                canonical_site=canonical_site,
            )
        )
    return tuple(rows)


def _at(values: NDArray | None, index: int) -> object | None:
    return None if values is None else values[index]


def present_string(value: object | None) -> str | None:
    r"""
    Normalize optional source text.

    Args:
        value: Source value to normalize.

    Returns:
        Stripped text, or `None` for absent PDBx markers and empty values.
    """

    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def optional_integer(value: object | None) -> int | None:
    r"""
    Parse an optional source identity integer.

    Args:
        value: Source value to parse.

    Returns:
        The parsed integer, or `None` when the value is absent.

    Raises:
        ValueError: If a present value is not an integer.
    """

    text = present_string(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"expected an integer identity, got {text!r}") from None


def _optional_float(value: object | None) -> float | None:
    text = present_string(value)
    return None if text is None else float(text)


def reject_duplicate_atom_records(records: tuple[AtomRecord, ...]) -> None:
    r"""
    Reject duplicate declarations of one source atom identity.

    Args:
        records: Atom records to validate.

    Raises:
        ValueError: If two rows declare the same source atom identity.
    """

    seen: set[tuple[object, ...]] = set()
    for row in records:
        key = (
            row.symmetry_id,
            row.chain_id,
            row.residue_id,
            row.auth_residue_id,
            row.insertion_code,
            row.component_id,
            row.atom_name,
            row.altloc_id,
        )
        if key in seen:
            raise ValueError(f"duplicate source atom declaration: {key!r}")
        seen.add(key)


def atom_record_residue_key(record: AtomRecord) -> tuple[object, ...]:
    r"""
    Return the source residue identity key for an atom record.

    Args:
        record: Atom record to identify.

    Returns:
        The record's symmetry, label, author, insertion, and component identity tuple.
    """

    return (
        record.symmetry_id,
        record.chain_id,
        record.residue_id,
        record.auth_residue_id,
        record.insertion_code,
        record.component_id,
    )


def selected_source_sites(
    selected_records: tuple[AtomRecord, ...], site_by_row_index: dict[int, AtomSite]
) -> dict[tuple[object, ...], AtomSite]:
    r"""
    Map unambiguous label and author atom identities to canonical sites.

    Args:
        selected_records: Atom records retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.

    Returns:
        Unambiguous source atom identity keys mapped to canonical sites.
    """

    mapping: dict[tuple[object, ...], AtomSite] = {}
    ambiguous: set[tuple[object, ...]] = set()
    for row in selected_records:
        site = site_by_row_index[row.row_index]
        label_key = (
            row.symmetry_id,
            row.chain_id,
            row.label_seq_id,
            row.insertion_code,
            row.component_id,
            row.atom_name,
        )
        auth_key = (
            row.symmetry_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.auth_component_id,
            row.auth_atom_name,
        )
        for key in (label_key, auth_key):
            if key in ambiguous:
                continue
            previous = mapping.get(key)
            if previous is not None and previous != site:
                mapping.pop(key)
                ambiguous.add(key)
                continue
            mapping[key] = site
    return mapping


def source_record_key(record: AtomRecord) -> tuple[object, ...]:
    r"""
    Return the source atom key used for PDB bond resolution.

    Args:
        record: Atom record to identify.

    Returns:
        The record's symmetry, chain, residue, insertion, component, and atom tuple.
    """

    return (
        record.symmetry_id,
        record.chain_id,
        record.residue_id,
        record.insertion_code,
        record.component_id,
        record.atom_name,
    )


def unique_identifier(seed: str, used: set[str]) -> str:
    r"""
    Reserve a unique identifier derived from a seed.

    Args:
        seed: Preferred identifier.
        used: Mutable set of identifiers already reserved.

    Returns:
        The reserved identifier, with a numeric suffix when needed.
    """

    candidate = seed
    suffix = 2
    while candidate in used:
        candidate = f"{seed}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def ordered_output_bonds(system: MolecularSystem) -> tuple[SystemBond, ...]:
    r"""
    Return canonical system bonds in stable output order.

    Args:
        system: Molecular system whose bonds are ordered.

    Returns:
        Bonds sorted by canonical endpoint identity.
    """

    return tuple(
        sorted(
            system.iter_bonds(),
            key=lambda bond: (
                bond.site_a.instance_id,
                bond.site_a.component_index,
                bond.site_a.atom_id,
                bond.site_b.instance_id,
                bond.site_b.component_index,
                bond.site_b.atom_id,
            ),
        )
    )
