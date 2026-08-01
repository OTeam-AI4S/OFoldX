"""Interpret CCD source categories as records shared by runtime and offline tools.

The record is the CCD resolver's storage schema. ChemicalComponent construction
remains with the resolver; this module does not write files or prepare structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

from ...io.cif import CifCategory, column_integers, column_points
from ...io.declarations import read_component_parents
from .._cif import parse_chemical_stereo, parse_optional_flag
from ..components import polymer_attachment_anchors, terminal_leaving_groups

_VALUE_ORDER = {"SING": 1.0, "DOUB": 2.0, "TRIP": 3.0, "QUAD": 4.0, "AROM": 1.5}


class CcdRecord(TypedDict):
    component_id: str
    chem_type: str
    parent_component_ids: list[str]
    atom_ids: list[str]
    elements: list[str]
    formal_charges: list[int | None]
    aromatic: list[bool | None]
    stereo: list[str | None]
    bond_atom_a: list[str]
    bond_atom_b: list[str]
    bond_orders: list[float | None]
    bond_aromatic: list[bool | None]
    bond_stereo: list[str | None]
    reference_coordinates: list[list[float] | None]
    leaving_atom_ids: list[str]
    non_leaving_atom_ids: list[str]
    previous_attachment_atom: str | None
    previous_leaving_atoms: list[str]
    next_attachment_atom: str | None
    next_leaving_atoms: list[str]


def decode_component(block: Mapping[str, CifCategory]) -> CcdRecord:
    """Decode one component using a single category/column representation."""
    component = block.get("chem_comp", CifCategory({}))
    atoms = block.get("chem_comp_atom", CifCategory({}))
    bonds = block.get("chem_comp_bond", CifCategory({}))
    component_id = _item(component, "id")
    raw_type = _item(component, "type", default="")
    parents = read_component_parents(component).get(component_id, ())
    atom_ids = _required_strings(atoms, "atom_id")
    elements = _required_strings(atoms, "type_symbol")
    leaving = [_optional_flag(value) for value in atoms.strings("pdbx_leaving_atom_flag")]
    leaving_atom_ids = [name for name, flag in zip(atom_ids, leaving, strict=True) if flag is True]
    non_leaving_atom_ids = [name for name, flag in zip(atom_ids, leaving, strict=True) if flag is False]
    atom_stereo = [parse_chemical_stereo(value, category="atom") for value in atoms.strings("pdbx_stereo_config")]
    points = column_points(atoms, "pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal")

    bond_atom_a = _required_strings(bonds, "atom_id_1")
    bond_atom_b = _required_strings(bonds, "atom_id_2")
    bond_aromatic = [_optional_flag(value) for value in bonds.strings("pdbx_aromatic_flag")]
    if "pdbx_aromatic_flag" in atoms:
        aromatic = [_optional_flag(value) for value in atoms.strings("pdbx_aromatic_flag")]
    else:
        # Preserve the established interpretation for CCD snapshots that omit
        # atom aromaticity: explicit aromatic bond evidence can establish it.
        aromatic_by_atom: dict[str, bool | None] = dict.fromkeys(atom_ids, False)
        for atom_a, atom_b, is_aromatic in zip(bond_atom_a, bond_atom_b, bond_aromatic, strict=True):
            for name in (atom_a, atom_b):
                if is_aromatic is True:
                    aromatic_by_atom[name] = True
                elif is_aromatic is None and aromatic_by_atom.get(name) is not True:
                    aromatic_by_atom[name] = None
        aromatic = [aromatic_by_atom[name] for name in atom_ids]
    complete_leaving_evidence = len(leaving_atom_ids) + len(non_leaving_atom_ids) == len(atom_ids)
    previous_atom, previous_leaving, next_atom, next_leaving = _attachments(
        raw_type,
        atom_ids,
        leaving_atom_ids if complete_leaving_evidence else None,
        bond_atom_a,
        bond_atom_b,
    )
    return {
        "component_id": component_id,
        "chem_type": _classify_chem_type(raw_type),
        "parent_component_ids": list(parents),
        "atom_ids": atom_ids,
        "elements": elements,
        "formal_charges": list(column_integers(atoms, "charge")),
        "aromatic": aromatic,
        "stereo": atom_stereo,
        "bond_atom_a": bond_atom_a,
        "bond_atom_b": bond_atom_b,
        "bond_orders": [_bond_order(value) for value in bonds.strings("value_order")],
        "bond_aromatic": bond_aromatic,
        "bond_stereo": [parse_chemical_stereo(value, category="bond") for value in bonds.strings("pdbx_stereo_config")],
        "reference_coordinates": [None if point is None else list(point) for point in points],
        "leaving_atom_ids": leaving_atom_ids,
        "non_leaving_atom_ids": non_leaving_atom_ids,
        "previous_attachment_atom": previous_atom,
        "previous_leaving_atoms": previous_leaving,
        "next_attachment_atom": next_atom,
        "next_leaving_atoms": next_leaving,
    }


def _item(category: CifCategory, name: str, *, default: str | None = None) -> str:
    column = category.get(name)
    if column is not None and len(column) > 1:
        raise ValueError(f"CCD item '_chem_comp.{name}' must describe exactly one component")
    value = None if column is None or not len(column) else column.values[0]
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"CCD block is missing required item '_chem_comp.{name}'")
    return value


def _required_strings(category: CifCategory, name: str) -> list[str]:
    if category.row_count and name not in category:
        raise ValueError(f"CCD block is missing required item {name!r}")
    values = category.strings(name)
    if any(value is None or not value for value in values):
        raise ValueError(f"CCD required item {name!r} contains a missing identifier")
    return cast(list[str], list(values))


def _optional_flag(value: str | None) -> bool | None:
    return parse_optional_flag(value, item="CCD flag")


def _classify_chem_type(raw_type: str) -> str:
    upper = raw_type.upper()
    if ("PEPTIDE" in upper and upper != "PEPTIDE-LIKE") or upper in {
        "C-GAMMA LINKING",
        "C-DELTA LINKING",
    }:
        return "protein"
    if "DNA" in upper and "RNA" in upper:
        return "dna_rna_hybrid"
    if "DNA" in upper:
        return "dna"
    if "RNA" in upper:
        return "rna"
    return "non-polymer"


def _attachments(
    chem_comp_type: str,
    atom_ids: Sequence[str],
    leaving_atom_ids: Sequence[str] | None,
    bond_atom_a: Sequence[str] = (),
    bond_atom_b: Sequence[str] = (),
) -> tuple[str | None, list[str], str | None, list[str]]:
    r"""
    Derive default linear-polymer attachments for a polymer component.

    Non-polymer components carry no linear attachments but retain their generic
    leaving-atom facts in the row. For polymers each connected leaving group is
    assigned to its unique non-leaving graph boundary; this recovers nonstandard
    peptide backbones without hard-coding every CCD component ID.
    """

    previous_anchor, next_anchor = polymer_attachment_anchors(chem_comp_type, atom_ids)
    if (previous_anchor is None and next_anchor is None) or leaving_atom_ids is None:
        return None, [], None, []

    order = {atom_id: index for index, atom_id in enumerate(atom_ids)}

    def attachment(anchor: str | None) -> tuple[str | None, list[str]]:
        if anchor is None:
            return None, []
        matching = terminal_leaving_groups(
            atom_ids,
            zip(bond_atom_a, bond_atom_b),
            leaving_atom_ids,
            anchor,
        )
        if len(matching) > 1:
            return None, []
        if not matching:
            return anchor, []
        return anchor, sorted(matching[0], key=order.__getitem__)

    previous_anchor, previous_leaving = attachment(previous_anchor)
    next_anchor, next_leaving = attachment(next_anchor)
    return (
        previous_anchor,
        previous_leaving,
        next_anchor,
        next_leaving,
    )


def _bond_order(value: str | None) -> float | None:
    if value is None:
        return None
    token = value.strip().upper()
    try:
        return _VALUE_ORDER[token]
    except KeyError as error:
        raise ValueError(f"unknown CCD bond order {value!r}") from error
