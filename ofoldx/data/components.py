r"""
Built-in chemical components.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import fields, replace

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


def terminal_leaving_groups(
    atom_ids: Sequence[str],
    bonds: Iterable[tuple[str, str]],
    leaving_atom_ids: Collection[str],
    endpoint: str,
) -> tuple[frozenset[str], ...]:
    r"""Return terminal fragments selected by leaving atoms adjacent to an endpoint."""

    present = set(atom_ids)
    selected = set(leaving_atom_ids)
    if endpoint not in present or not selected:
        return ()
    if not selected <= present:
        raise ValueError("leaving group references an unknown atom")

    adjacency: dict[str, set[str]] = {atom_id: set() for atom_id in atom_ids}
    for atom_a, atom_b in bonds:
        if atom_a not in present or atom_b not in present:
            raise ValueError("bond references an unknown atom")
        adjacency[atom_a].add(atom_b)
        adjacency[atom_b].add(atom_a)

    groups: set[frozenset[str]] = set()
    unseen = set(selected)
    while unseen:
        start = unseen.pop()
        seeds = {start}
        pending = [start]
        while pending:
            for neighbor in adjacency[pending.pop()] & unseen:
                unseen.remove(neighbor)
                seeds.add(neighbor)
                pending.append(neighbor)
        if not adjacency[endpoint].intersection(seeds):
            continue

        fragment = set(seeds)
        pending = list(seeds)
        while pending:
            for neighbor in adjacency[pending.pop()] - fragment - {endpoint}:
                fragment.add(neighbor)
                pending.append(neighbor)
        boundary = [(atom_id, neighbor) for atom_id in fragment for neighbor in adjacency[atom_id] - fragment]
        if len(boundary) == 1 and boundary[0][1] == endpoint:
            groups.add(frozenset(fragment))

    order = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    return tuple(sorted(groups, key=lambda group: tuple(sorted(order[atom_id] for atom_id in group))))


def polymer_attachment_anchors(
    chem_comp_type: str | None,
    atom_ids: Collection[str],
) -> tuple[str | None, str | None]:
    r"""Return the previous and next anchors declared by a CCD polymer type."""

    declared_type = "" if chem_comp_type is None else chem_comp_type.upper()
    present = set(atom_ids)
    protein = ("PEPTIDE" in declared_type and declared_type != "PEPTIDE-LIKE") or declared_type in {
        "C-GAMMA LINKING",
        "C-DELTA LINKING",
    }
    if protein:
        previous = None if "NH3 AMINO TERMINUS" in declared_type else "N"
        if "COOH CARBOXY TERMINUS" in declared_type:
            following = None
        elif "C-GAMMA LINKING" in declared_type:
            following = "CG"
        elif "C-DELTA LINKING" in declared_type:
            following = "CD"
        else:
            following = "C"
        return (
            previous if previous in present else None,
            following if following in present else None,
        )
    if "DNA" in declared_type or "RNA" in declared_type:
        previous = None if "OH 5 PRIME TERMINUS" in declared_type else "P"
        following = None if "OH 3 PRIME TERMINUS" in declared_type else "O3'"
        return (
            previous if previous in present else None,
            following if following in present else None,
        )
    return None, None


def complete_component(
    occurrence: ChemicalComponent,
    reference: ChemicalComponent,
    *,
    occurrence_topology_complete: bool,
    atom_stereo_known: Collection[str] = (),
    bond_stereo_known: Collection[frozenset[str]] = (),
) -> tuple[ChemicalComponent, bool]:
    r"""
    Fill chemistry an occurrence leaves unknown from a reference component.

    The occurrence remains authoritative for every atom and bond fact it states.
    The reference contributes missing atoms, bonds, attachments, and reference
    coordinates without deleting source-only chemistry.

    Args:
        occurrence: Chemistry stated by one source component.
        reference: Component definition used only to fill missing facts.
        occurrence_topology_complete: Whether absent occurrence bonds are known absent.
        atom_stereo_known: Occurrence atoms whose stereo field is explicitly stated,
            including atoms explicitly stated to have no stereochemistry.
        bond_stereo_known: Occurrence edges whose stereo field is explicitly stated,
            including edges explicitly stated to have no stereochemistry.

    Returns:
        The completed component and whether its resulting topology is complete.

    Examples:
        >>> occurrence = ChemicalComponent(
        ...     "LIG",
        ...     (ChemicalAtom("N1", "N", isotope=15, formal_charge=-3),),
        ...     frozenset(),
        ... )
        >>> reference = ChemicalComponent(
        ...     "LIG",
        ...     (ChemicalAtom("N1", "N", formal_charge=1, aromatic=True), ChemicalAtom("C1", "C")),
        ...     frozenset({ChemicalBond("N1", "C1", order=1.5, aromatic=True)}),
        ...     reference_coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
        ... )
        >>> completed, topology_complete = complete_component(
        ...     occurrence, reference, occurrence_topology_complete=True
        ... )
        >>> completed.atoms[0].isotope, completed.atoms[0].formal_charge, completed.atoms[0].aromatic
        (15, -3, True)
        >>> [atom.atom_id for atom in completed.atoms]
        ['N1', 'C1']
        >>> topology_complete
        True
    """

    if occurrence.component_id != reference.component_id:
        raise ValueError("occurrence and reference component IDs must match")
    atom_stereo_known = frozenset(atom_stereo_known)
    bond_stereo_known = frozenset(bond_stereo_known)
    stated_atoms = {atom.atom_id: atom for atom in occurrence.atoms}
    stated_atom_ids = frozenset(stated_atoms)
    reference_atom_ids = frozenset(atom.atom_id for atom in reference.atoms)
    stated_coordinates = {
        atom.atom_id: coordinate
        for atom, coordinate in zip(occurrence.atoms, occurrence.reference_coordinates)
        if coordinate is not None
    }
    reference_coordinates = dict(zip((atom.atom_id for atom in reference.atoms), reference.reference_coordinates))
    atoms: list[ChemicalAtom] = []
    coordinates: list[tuple[float, float, float] | None] = []
    for atom in reference.atoms:
        stated = stated_atoms.pop(atom.atom_id, None)
        if stated is not None and stated.element != atom.element and stated.element != "X" and atom.element != "X":
            raise ValueError(
                f"component {occurrence.component_id!r} atom {atom.atom_id!r} "
                "has conflicting source and reference elements"
            )
        stated_values = (
            {}
            if stated is None
            else {
                field.name: value
                for field in fields(stated)
                if ((value := getattr(stated, field.name)) is not None)
                or (field.name == "stereo" and stated.atom_id in atom_stereo_known)
            }
        )
        if stated is not None and stated.element == "X" and atom.element != "X":
            stated_values.pop("element")
        atoms.append(atom if stated is None else replace(atom, **stated_values))
        coordinates.append(stated_coordinates.get(atom.atom_id, reference_coordinates.get(atom.atom_id)))
    for atom in stated_atoms.values():
        atoms.append(atom)
        coordinates.append(stated_coordinates.get(atom.atom_id))

    reference_bonds = (
        {} if reference.bonds is None else {frozenset((bond.atom_a, bond.atom_b)): bond for bond in reference.bonds}
    )

    def completed_bond(bond: ChemicalBond) -> ChemicalBond:
        pair = frozenset((bond.atom_a, bond.atom_b))
        filler = reference_bonds.get(pair)
        if filler is None:
            return bond
        stereo_atoms = filler.stereo_atoms
        if stereo_atoms is not None and filler.atom_a != bond.atom_a:
            stereo_atoms = (stereo_atoms[1], stereo_atoms[0])
        if pair in bond_stereo_known:
            stereo = bond.stereo
            if stereo is None:
                stereo_atoms = None
        else:
            stereo = filler.stereo if bond.stereo is None else bond.stereo
        return ChemicalBond(
            bond.atom_a,
            bond.atom_b,
            order=filler.order if bond.order is None else bond.order,
            aromatic=filler.aromatic if bond.aromatic is None else bond.aromatic,
            stereo=stereo,
            stereo_atoms=stereo_atoms if bond.stereo_atoms is None else bond.stereo_atoms,
            kind=bond.kind,
            direction=bond.direction,
        )

    stated_bonds = {
        frozenset((bond.atom_a, bond.atom_b)): bond for bond in (() if occurrence.bonds is None else occurrence.bonds)
    }
    bonds = frozenset(
        [completed_bond(bond) for bond in stated_bonds.values()]
        + [
            bond
            for pair, bond in reference_bonds.items()
            if pair not in stated_bonds and (not occurrence_topology_complete or not pair <= stated_atom_ids)
        ]
    )
    topology_complete = (occurrence_topology_complete and reference_atom_ids <= stated_atom_ids) or (
        reference.bonds is not None and stated_atom_ids <= reference_atom_ids
    )
    source_attachment_leaving_atom_ids = frozenset(
        atom_id
        for attachment in (occurrence.previous_attachment, occurrence.next_attachment)
        if attachment is not None
        for atom_id in attachment.leaving_atom_ids
    )
    occurrence_leaving_flags_known = occurrence.leaving_atom_ids | occurrence.non_leaving_atom_ids
    leaving_atom_ids = occurrence.leaving_atom_ids | (reference.leaving_atom_ids - occurrence_leaving_flags_known)
    non_leaving_atom_ids = occurrence.non_leaving_atom_ids | (
        reference.non_leaving_atom_ids - occurrence_leaving_flags_known - source_attachment_leaving_atom_ids
    )

    previous_attachment = occurrence.previous_attachment or reference.previous_attachment
    next_attachment = occurrence.next_attachment or reference.next_attachment

    return (
        ChemicalComponent(
            component_id=occurrence.component_id,
            atoms=tuple(atoms),
            bonds=bonds,
            previous_attachment=previous_attachment,
            next_attachment=next_attachment,
            reference_coordinates=tuple(coordinates),
            leaving_atom_ids=leaving_atom_ids,
            non_leaving_atom_ids=non_leaving_atom_ids,
        ),
        topology_complete,
    )


def component_from_rows(
    component_id: str,
    atom_rows: Iterable[AtomRow],
    bond_rows: Iterable[BondRow],
    *,
    reference_coordinates: tuple[tuple[float, float, float] | None, ...],
    previous_attachment: PolymerAttachment | None,
    next_attachment: PolymerAttachment | None,
    leaving_atom_ids: frozenset[str] = frozenset(),
    non_leaving_atom_ids: frozenset[str] = frozenset(),
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
        leaving_atom_ids: Source-declared leaving atoms.
        non_leaving_atom_ids: Source-declared non-leaving atoms.

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
        leaving_atom_ids=leaving_atom_ids,
        non_leaving_atom_ids=non_leaving_atom_ids,
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
    next_leaving_atoms = frozenset({"OXT", "HXT"})
    return component_from_rows(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=reference_coordinates,
        previous_attachment=PolymerAttachment("N", previous_leaving_atoms),
        next_attachment=PolymerAttachment("C", next_leaving_atoms),
        leaving_atom_ids=previous_leaving_atoms | next_leaving_atoms,
        non_leaving_atom_ids=frozenset(atom_id for atom_id, *_ in atom_rows)
        - previous_leaving_atoms
        - next_leaving_atoms,
    )


def _build_nucleic_component(
    component_id: str,
    data: _nucleic_components.ComponentRows,
) -> ChemicalComponent:
    atom_rows, bond_rows = data
    reference_by_atom = dict(_nucleic_components.REFERENCE_COORDINATE_ROWS[component_id])
    generated = component_id in {"N", "DN"}
    previous_leaving_atoms = frozenset() if generated else frozenset({"OP3", "HOP3"})
    next_leaving_atoms = frozenset() if generated else frozenset({"HO3'"})
    return component_from_rows(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=tuple(reference_by_atom.get(atom_id) for atom_id, *_ in atom_rows),
        previous_attachment=PolymerAttachment(
            "P",
            previous_leaving_atoms,
        ),
        next_attachment=PolymerAttachment(
            "O3'",
            next_leaving_atoms,
        ),
        leaving_atom_ids=previous_leaving_atoms | next_leaving_atoms,
        non_leaving_atom_ids=frozenset(atom_id for atom_id, *_ in atom_rows)
        - previous_leaving_atoms
        - next_leaving_atoms,
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
