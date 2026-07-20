r"""Package-owned audited chemical components."""

from __future__ import annotations

from collections.abc import Iterable

from ._nucleic_components import (
    _AUDITED_NUCLEIC_COMPONENT_DATA,
    _AUDITED_NUCLEIC_REFERENCE_COORDINATES,
)
from ._protein_components import (
    _AUDITED_PROTEIN_COMPONENT_DATA,
    _AUDITED_PROTEIN_REFERENCE_COORDINATES,
    _RawComponent,
)
from .molecular import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    PolymerAttachment,
)


def _component_from_raw(
    component_id: str,
    atom_rows: Iterable[tuple[str, str, int, bool, str | None]],
    bond_rows: Iterable[tuple],
    *,
    reference_coordinates: tuple[tuple[float, float, float] | None, ...],
    previous_attachment: PolymerAttachment | None,
    next_attachment: PolymerAttachment | None,
) -> ChemicalComponent:
    r"""Build a `ChemicalComponent` from raw atom and bond rows.

    Shared boundary-time constructor for every audited/lowered source: it fixes the
    package-owned atom node facts (no isotope, zero radical electrons, zero explicit
    and implicit hydrogens, implicit hydrogens disallowed) and the bond frozenset
    build in one place. Each ``atom_row`` is
    ``(atom_id, element, formal_charge, aromatic, stereo)``; each ``bond_row`` is
    ``(atom_a, atom_b, order, aromatic)`` with an optional trailing ``stereo``.
    Per-source facts — reference coordinates and polymer attachments — are supplied
    by the caller.
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


def _build_protein_component(component_id: str, data: _RawComponent) -> ChemicalComponent:
    atom_rows, bond_rows = data
    reference_rows = _AUDITED_PROTEIN_REFERENCE_COORDINATES.get(component_id)
    reference_coordinates: tuple[tuple[float, float, float] | None, ...] = ()
    if reference_rows is not None:
        reference_by_atom = dict(reference_rows)
        reference_coordinates = tuple(reference_by_atom.get(atom_id) for atom_id, *_ in atom_rows)
    previous_leaving_atoms = frozenset({"H"}) if component_id == "PRO" else frozenset({"H2"})
    return _component_from_raw(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=reference_coordinates,
        previous_attachment=PolymerAttachment("N", previous_leaving_atoms),
        next_attachment=PolymerAttachment("C", frozenset({"OXT", "HXT"})),
    )


def _build_nucleic_component(component_id: str, data: _RawComponent) -> ChemicalComponent:
    atom_rows, bond_rows = data
    reference_by_atom = dict(_AUDITED_NUCLEIC_REFERENCE_COORDINATES[component_id])
    generated = component_id in {"N", "DN"}
    return _component_from_raw(
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


_AUDITED_COMPONENTS = {
    **{
        component_id: _build_protein_component(component_id, data)
        for component_id, data in _AUDITED_PROTEIN_COMPONENT_DATA.items()
    },
    **{
        component_id: _build_nucleic_component(component_id, data)
        for component_id, data in _AUDITED_NUCLEIC_COMPONENT_DATA.items()
    },
}


def get_component(component_id: str) -> ChemicalComponent:
    r"""Return the package-audited component with this exact ID.

    Unknown or non-audited component IDs raise ``KeyError``. OFoldX never guesses a
    chemical graph from a residue name.
    """

    return _AUDITED_COMPONENTS[component_id]
