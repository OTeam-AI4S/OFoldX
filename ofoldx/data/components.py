r"""
Built-in chemical components.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import _nucleic_components, _protein_components
from .molecular import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    PolymerAttachment,
    PolymerType,
)

AtomRow = tuple[str, str, int | None, bool | None, str | None]
BondRow = tuple[str, str, float, bool] | tuple[str, str, float, bool, str | None]


def component_from_rows(
    component_id: str,
    atom_rows: Iterable[AtomRow],
    bond_rows: Iterable[BondRow],
    *,
    reference_coordinates: tuple[tuple[float, float, float] | None, ...],
    previous_attachment: PolymerAttachment | None,
    next_attachment: PolymerAttachment | None,
) -> ChemicalComponent:
    r"""
    Build a [`ChemicalComponent`][] from atom and bond rows.

    Atom rows use `(atom_id, element, formal_charge, aromatic, stereo)` and bond
    rows use `(atom_a, atom_b, order, aromatic[, stereo])`.

    Args:
        component_id: Chemical component identifier.
        atom_rows: Atom identity and chemistry rows.
        bond_rows: Bond endpoint and chemistry rows.
        reference_coordinates: Reference coordinate for each atom, or `None` where
            no coordinate is available.
        previous_attachment: Attachment used to connect the preceding polymer component.
        next_attachment: Attachment used to connect the following polymer component.

    Returns:
        The assembled chemical component.
    """

    atoms = tuple(
        ChemicalAtom(
            atom_id=atom_id,
            element=element,
            isotope=None,
            formal_charge=formal_charge,
            radical_electrons=0,
            explicit_hydrogens=0,
            implicit_hydrogens=0,
            implicit_hydrogens_allowed=False,
            aromatic=aromatic,
            stereo=stereo,
        )
        for atom_id, element, formal_charge, aromatic, stereo in atom_rows
    )
    bonds = frozenset(
        ChemicalBond(
            atom_a,
            atom_b,
            order=order,
            aromatic=aromatic,
            stereo=(stereo[0] if stereo else None),
        )
        for atom_a, atom_b, order, aromatic, *stereo in bond_rows
    )
    return ChemicalComponent(
        component_id=component_id,
        atoms=atoms,
        bonds=bonds,
        previous_attachment=previous_attachment,
        next_attachment=next_attachment,
        reference_coordinates=reference_coordinates,
    )


def _build_protein_component(
    component_id: str,
    data: _protein_components.ComponentRows,
) -> ChemicalComponent:
    atom_rows, bond_rows = data
    reference_rows = _protein_components.REFERENCE_COORDINATE_ROWS.get(component_id)
    reference_coordinates: tuple[tuple[float, float, float] | None, ...] = ()
    if reference_rows is not None:
        reference_by_atom = dict(reference_rows)
        reference_coordinates = tuple(reference_by_atom.get(atom_id) for atom_id, *_ in atom_rows)
    previous_leaving_atoms = frozenset({"H"}) if component_id == "PRO" else frozenset({"H2"})
    return component_from_rows(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=reference_coordinates,
        previous_attachment=PolymerAttachment("N", previous_leaving_atoms),
        next_attachment=PolymerAttachment("C", frozenset({"OXT", "HXT"})),
    )


def _build_nucleic_component(
    component_id: str,
    data: _nucleic_components.ComponentRows,
) -> ChemicalComponent:
    atom_rows, bond_rows = data
    reference_by_atom = dict(_nucleic_components.REFERENCE_COORDINATE_ROWS[component_id])
    generated = component_id in {"N", "DN"}
    return component_from_rows(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=tuple(reference_by_atom.get(atom_id) for atom_id, *_ in atom_rows),
        previous_attachment=PolymerAttachment(
            "P",
            frozenset() if generated else frozenset({"OP3", "HOP3"}),
        ),
        next_attachment=PolymerAttachment(
            "O3'",
            frozenset() if generated else frozenset({"HO3'"}),
        ),
    )


_BUILTIN_COMPONENTS = {
    **{
        component_id: _build_protein_component(component_id, data)
        for component_id, data in _protein_components.COMPONENT_ROWS.items()
    },
    **{
        component_id: _build_nucleic_component(component_id, data)
        for component_id, data in _nucleic_components.COMPONENT_ROWS.items()
    },
}
_PROTEIN_COMPONENT_IDS = frozenset(_protein_components.COMPONENT_ROWS)
_NUCLEIC_COMPONENT_IDS = frozenset(_nucleic_components.COMPONENT_ROWS)


def get_builtin_component_polymer_type(component_id: str) -> PolymerType | None:
    r"""
    Return the polymer type of a built-in component ID.

    Args:
        component_id: Exact component identifier.

    Returns:
        The component's polymer type, or `None` when it is not a built-in polymer
        component.
    """

    if component_id in _PROTEIN_COMPONENT_IDS:
        return "protein"
    if component_id in _NUCLEIC_COMPONENT_IDS:
        return "dna" if component_id.startswith("D") else "rna"
    return None


def get_component(component_id: str) -> ChemicalComponent:
    r"""
    Return a built-in chemical component by ID.

    Args:
        component_id: Exact component identifier.

    Returns:
        The matching built-in component.

    Raises:
        KeyError: If no built-in component has ``component_id``.
    """

    return _BUILTIN_COMPONENTS[component_id]
