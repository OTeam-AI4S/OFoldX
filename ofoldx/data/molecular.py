r"""
Coordinate-free molecular values and exact occurrence references.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

import torch
from torch import Tensor

from ._tensorclass import frozen_tensorclass
from .elements import ATOMIC_NUMBER_BY_ELEMENT

#: Supported polymer chemistries.
PolymerType: TypeAlias = Literal["protein", "rna", "dna", "dna_rna_hybrid", "pna"]
#: PDBx entity category retained by a non-polymer construction value.
MoleculeType: TypeAlias = Literal["non-polymer", "branched", "water"]
#: Molecular-system component occurrence as ``(instance_id, component_index)``.
ComponentReference: TypeAlias = tuple[str, int]
BondKind: TypeAlias = Literal["covalent", "dative"]
BondDirection: TypeAlias = Literal["undirected", "a_to_b", "b_to_a"]
BondRole: TypeAlias = Literal["component", "polymer", "entity", "assembly"]

ENTITY_MOLECULE = 0
ENTITY_PROTEIN = 1
ENTITY_RNA = 2
ENTITY_DNA = 3
ENTITY_DNA_RNA_HYBRID = 4
ENTITY_PNA = 5

MOLECULE_NON_POLYMER = 0
MOLECULE_BRANCHED = 1
MOLECULE_WATER = 2

BOND_KIND_COVALENT = 0
BOND_KIND_DATIVE = 1

BOND_DIRECTION_UNDIRECTED = 0
BOND_DIRECTION_ENDPOINT_0_DONOR = 1
BOND_DIRECTION_ENDPOINT_1_DONOR = -1

BOND_ROLE_COMPONENT = 0
BOND_ROLE_POLYMER = 1
BOND_ROLE_ENTITY = 2
BOND_ROLE_ASSEMBLY = 3

_ENTITY_TYPE = {
    "protein": ENTITY_PROTEIN,
    "rna": ENTITY_RNA,
    "dna": ENTITY_DNA,
    "dna_rna_hybrid": ENTITY_DNA_RNA_HYBRID,
    "pna": ENTITY_PNA,
}
_MOLECULE_TYPE = {
    "non-polymer": MOLECULE_NON_POLYMER,
    "branched": MOLECULE_BRANCHED,
    "water": MOLECULE_WATER,
}
_BOND_KIND = {"covalent": BOND_KIND_COVALENT, "dative": BOND_KIND_DATIVE}
_BOND_DIRECTION = {
    "undirected": BOND_DIRECTION_UNDIRECTED,
    "a_to_b": BOND_DIRECTION_ENDPOINT_0_DONOR,
    "b_to_a": BOND_DIRECTION_ENDPOINT_1_DONOR,
}
_BOND_ROLE = {
    "component": BOND_ROLE_COMPONENT,
    "polymer": BOND_ROLE_POLYMER,
    "entity": BOND_ROLE_ENTITY,
    "assembly": BOND_ROLE_ASSEMBLY,
}

# One-letter symbols resolve against a shared standard core. Strict mode includes
# unambiguous extensions; permissive mode uses polymer-specific unknown components.
SymbolPolicy: TypeAlias = Literal["strict", "permissive"]

# Standard residues shared by both symbol policies.
_STANDARD_COMPONENTS: dict[str, dict[str, str]] = {
    "protein": {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    },
    "rna": {"A": "A", "C": "C", "G": "G", "U": "U"},
    "dna": {"A": "DA", "C": "DC", "G": "DG", "T": "DT"},
}
# Rare-but-unambiguous residues resolved only under the strict policy (never a
# permissive fallback target: a permissive caller routes these to the unknown token).
_EXTENDED_COMPONENTS: dict[str, dict[str, str]] = {
    "protein": {"U": "SEC", "O": "PYL"},
    "rna": {"I": "I"},
    "dna": {"U": "DU", "I": "DI"},
}
# The unknown-residue placeholder component per polymer type (permissive fallback).
_UNKNOWN_COMPONENT: dict[str, str] = {"protein": "UNK", "rna": "N", "dna": "DN"}


def one_letter_to_component(
    symbol: str,
    polymer_type: str = "protein",
    *,
    policy: SymbolPolicy = "strict",
) -> str:
    r"""
    Resolve a polymer one-letter symbol to a component ID.

    Args:
        symbol: One-letter residue symbol.
        polymer_type: Polymer alphabet in which to interpret ``symbol``.
        policy: ``"strict"`` accepts standard and unambiguous extended residues;
            ``"permissive"`` maps non-standard symbols to the polymer's unknown component.

    Returns:
        The corresponding component ID.

    Raises:
        KeyError: If ``polymer_type`` is unknown or strict resolution fails.
    """

    standard = _STANDARD_COMPONENTS[polymer_type]
    if policy == "permissive":
        return standard.get(symbol, _UNKNOWN_COMPONENT[polymer_type])
    if symbol in standard:
        return standard[symbol]
    extended = _EXTENDED_COMPONENTS.get(polymer_type, {})
    if symbol in extended:
        return extended[symbol]
    raise KeyError(symbol)


# Coordinate-free one-letter/component mappings derived from the canonical tables.
AMINO_ACID_ONE_TO_THREE: dict[str, str] = dict(_STANDARD_COMPONENTS["protein"])
AMINO_ACID_THREE_TO_ONE: dict[str, str] = {component: letter for letter, component in AMINO_ACID_ONE_TO_THREE.items()}
#: The 20 standard amino-acid component IDs as an order-independent set.
STANDARD_AMINO_ACID_IDS: frozenset[str] = frozenset(AMINO_ACID_THREE_TO_ONE)


# Element symbols treated as hydrogen, dropped from heavy-atom sets.
HYDROGEN_ELEMENTS: frozenset[str] = frozenset({"H", "D", "T"})


@dataclass(frozen=True)
class ChemicalAtom:
    r"""
    An atom in a chemical component.

    Args:
        atom_id: Component-local atom identifier.
        element: Chemical element symbol.
        isotope: Optional isotope mass number.
        formal_charge: Optional formal charge in elementary-charge units.
        radical_electrons: Optional number of radical electrons.
        explicit_hydrogens: Optional explicit hydrogen count.
        implicit_hydrogens: Optional implicit hydrogen count.
        implicit_hydrogens_allowed: Whether implicit hydrogens may be inferred.
        aromatic: Optional aromaticity assignment.
        stereo: Optional atom stereochemistry descriptor.
    """

    atom_id: str
    element: str
    isotope: int | None = None
    formal_charge: int | None = None
    radical_electrons: int | None = None
    explicit_hydrogens: int | None = None
    implicit_hydrogens: int | None = None
    implicit_hydrogens_allowed: bool | None = None
    aromatic: bool | None = None
    stereo: str | None = None


@dataclass(frozen=True)
class ChemicalBond:
    r"""
    A chemical bond within a component.

    Args:
        atom_a: Component-local identifier of the first endpoint.
        atom_b: Component-local identifier of the second endpoint.
        order: Bond order.
        aromatic: Whether the bond is aromatic.
        stereo: Optional bond stereochemistry descriptor.
        stereo_atoms: Optional component-local atoms defining bond stereochemistry.
        kind: Covalent or dative bond kind.
        direction: Donor direction for a dative bond.
    """

    atom_a: str
    atom_b: str
    order: float | None = 1.0
    aromatic: bool | None = False
    stereo: str | None = None
    stereo_atoms: tuple[str, str] | None = None
    kind: BondKind = "covalent"
    direction: BondDirection = "undirected"


@dataclass(frozen=True)
class PolymerAttachment:
    r"""
    A component-local endpoint for a linear polymer link.

    Args:
        atom_id: Atom forming the inter-component bond.
        leaving_atom_ids: Atoms removed when the link is realized.
    """

    atom_id: str
    leaving_atom_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ChemicalComponent:
    r"""
    A reusable chemical component definition.

    Args:
        component_id: Chemical component identifier.
        atoms: Component atoms in their defined order.
        bonds: Complete bond graph, or ``None`` when connectivity is unknown.
        previous_attachment: Endpoint connecting to the preceding polymer component.
        next_attachment: Endpoint connecting to the following polymer component.
        reference_coordinates: Optional coordinates aligned with ``atoms``; ``None``
            entries denote atoms without reference coordinates.

    Raises:
        ValueError: If atom identifiers, bonds, attachments, or reference coordinates
            are internally inconsistent.
    """

    component_id: str
    atoms: tuple[ChemicalAtom, ...]
    bonds: frozenset[ChemicalBond] | None
    previous_attachment: PolymerAttachment | None = None
    next_attachment: PolymerAttachment | None = None
    reference_coordinates: tuple[tuple[float, float, float] | None, ...] = ()

    def __post_init__(self) -> None:
        atom_ids = {atom.atom_id for atom in self.atoms}
        if len(atom_ids) != len(self.atoms):
            raise ValueError("component atom IDs must be unique")

        if self.bonds is not None:
            edges: set[frozenset[str]] = set()
            for bond in self.bonds:
                endpoints = frozenset((bond.atom_a, bond.atom_b))
                if len(endpoints) != 2:
                    raise ValueError("a component bond cannot connect an atom to itself")
                if not endpoints <= atom_ids:
                    raise ValueError("component bond references an unknown atom")
                if endpoints in edges:
                    raise ValueError("component bonds contain a duplicate physical edge")
                edges.add(endpoints)
                if bond.stereo_atoms is not None and not set(bond.stereo_atoms) <= atom_ids:
                    raise ValueError("component bond stereo references an unknown atom")

        for attachment in (self.previous_attachment, self.next_attachment):
            if attachment is not None and not {attachment.atom_id, *attachment.leaving_atom_ids} <= atom_ids:
                raise ValueError("polymer attachment references an unknown atom")
            if attachment is not None and attachment.atom_id in attachment.leaving_atom_ids:
                raise ValueError("a polymer attachment endpoint cannot also be a leaving atom")

        if self.reference_coordinates and len(self.reference_coordinates) != len(self.atoms):
            raise ValueError("component reference coordinates must align with its atoms")

    def without_hydrogens(self) -> ChemicalComponent:
        r"""
        Return the induced component graph without hydrogen isotopes.

        Bond stereochemistry is retained unless one of its defining atoms is
        removed.
        """

        keep = tuple(atom.element not in HYDROGEN_ELEMENTS for atom in self.atoms)
        if all(keep):
            return self
        retained = {atom.atom_id for atom, present in zip(self.atoms, keep) if present}

        bonds = self.bonds
        if bonds is not None:
            retained_bonds: set[ChemicalBond] = set()
            for bond in bonds:
                if bond.atom_a not in retained or bond.atom_b not in retained:
                    continue
                stereo_retained = bond.stereo_atoms is None or set(bond.stereo_atoms) <= retained
                retained_bonds.add(
                    ChemicalBond(
                        bond.atom_a,
                        bond.atom_b,
                        order=bond.order,
                        aromatic=bond.aromatic,
                        stereo=bond.stereo if stereo_retained else None,
                        stereo_atoms=(bond.stereo_atoms if stereo_retained else None),
                        kind=bond.kind,
                        direction=bond.direction,
                    )
                )
            bonds = frozenset(retained_bonds)

        previous = self.previous_attachment
        if previous is not None:
            previous = (
                PolymerAttachment(
                    previous.atom_id,
                    previous.leaving_atom_ids & retained,
                )
                if previous.atom_id in retained
                else None
            )
        following = self.next_attachment
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
            component_id=self.component_id,
            atoms=tuple(atom for atom, present in zip(self.atoms, keep) if present),
            bonds=bonds,
            previous_attachment=previous,
            next_attachment=following,
            reference_coordinates=(
                tuple(
                    coordinate
                    for coordinate, present in zip(
                        self.reference_coordinates,
                        keep,
                    )
                    if present
                )
                if self.reference_coordinates
                else ()
            ),
        )


_WATER_COMPONENT_ALIASES = frozenset(
    {
        "HOH",
        "DOD",
        "WAT",
        "TIP3",
        "H2O",
        "OH2",
        "TIP",
        "TIP2",
        "TIP4",
        "SPC",
        "SOL",
        "T3P",
    }
)


def is_water_component(component: ChemicalComponent) -> bool:
    r"""
    Return whether a component represents water.

    Args:
        component: Chemical component to classify.

    Returns:
        Whether the component has a known water ID and contains only hydrogen and
        oxygen atoms.
    """

    if component.component_id not in _WATER_COMPONENT_ALIASES:
        return False
    elements = {atom.element for atom in component.atoms}
    return "O" in elements and elements <= {*HYDROGEN_ELEMENTS, "O"}


@dataclass(frozen=True)
class PolymerSequence:
    r"""
    An ordered sequence of chemical component occurrences.

    Args:
        polymer_type: Polymer chemistry represented by the sequence.
        components: Component occurrences in sequence order.
    """

    polymer_type: PolymerType
    components: tuple[ChemicalComponent, ...]

    @classmethod
    def from_one_letter(
        cls,
        sequence: str,
        *,
        polymer_type: PolymerType = "protein",
    ) -> PolymerSequence:
        r"""
        Construct a polymer sequence from one-letter residue symbols.

        Args:
            sequence: One-letter polymer sequence.
            polymer_type: Polymer alphabet in which to interpret the sequence.

        Returns:
            A sequence of built-in chemical components.

        Raises:
            ValueError: If the polymer type requires explicit components or a symbol is
                unsupported or ambiguous.
        """

        if polymer_type == "dna_rna_hybrid":
            raise ValueError("a DNA/RNA hybrid requires explicit component occurrences")

        from .components import get_component

        components: list[ChemicalComponent] = []
        for index, symbol in enumerate(sequence):
            try:
                components.append(get_component(one_letter_to_component(symbol, polymer_type, policy="strict")))
            except KeyError:
                raise ValueError(
                    f"unsupported or ambiguous {polymer_type} one-letter symbol {symbol!r} at index {index}"
                ) from None
        return cls(polymer_type, tuple(components))

    def to_one_letter(self) -> str:
        r"""
        Return the exact one-letter sequence for built-in polymer components.

        Returns:
            One-letter symbols in component order.

        Raises:
            ValueError: If a component has no unambiguous built-in symbol.

        Examples:
            >>> PolymerSequence.from_one_letter("ACU").to_one_letter()
            'ACU'
        """

        polymer_types = ("dna", "rna") if self.polymer_type == "dna_rna_hybrid" else (self.polymer_type,)
        symbols = {
            component_id: symbol
            for polymer_type in polymer_types
            for symbol, component_id in {
                **_STANDARD_COMPONENTS[polymer_type],
                **_EXTENDED_COMPONENTS.get(polymer_type, {}),
            }.items()
        }
        sequence: list[str] = []
        for index, component in enumerate(self.components):
            try:
                sequence.append(symbols[component.component_id])
            except KeyError:
                raise ValueError(
                    f"{self.polymer_type} component {component.component_id!r} at index {index} "
                    "has no unambiguous one-letter symbol"
                ) from None
        return "".join(sequence)

    def __len__(self) -> int:
        return len(self.components)


@dataclass(frozen=True)
class MoleculeBond:
    r"""
    A chemical bond between component occurrences in a molecule.

    Args:
        component_a: Index of the first component occurrence.
        atom_a: Component-local atom identifier for the first endpoint.
        component_b: Index of the second component occurrence.
        atom_b: Component-local atom identifier for the second endpoint.
        order: Bond order.
        aromatic: Whether the bond is aromatic.
        stereo: Optional bond stereochemistry descriptor.
        stereo_atoms: Optional occurrence-local atoms defining bond stereochemistry.
        leaving_a: Atoms removed from the first component when the bond forms.
        leaving_b: Atoms removed from the second component when the bond forms.
        kind: Covalent or dative bond kind.
        direction: Donor direction for a dative bond.
    """

    component_a: int
    atom_a: str
    component_b: int
    atom_b: str
    order: float | None = 1.0
    aromatic: bool | None = False
    stereo: str | None = None
    stereo_atoms: tuple[tuple[int, str], tuple[int, str]] | None = None
    leaving_a: frozenset[str] = frozenset()
    leaving_b: frozenset[str] = frozenset()
    kind: BondKind = "covalent"
    direction: BondDirection = "undirected"


@dataclass(frozen=True)
class Molecule:
    r"""
    An ordered molecular graph.

    Args:
        components: Chemical component occurrences in molecule order.
        bonds: Bonds between different component occurrences.
        molecule_type: Retained PDBx entity category.

    Raises:
        ValueError: If a bond is duplicated, self-referential, or references an unknown
            component atom.
    """

    components: tuple[ChemicalComponent, ...]
    bonds: frozenset[MoleculeBond] = frozenset()
    molecule_type: MoleculeType = "non-polymer"

    def __post_init__(self) -> None:
        sites = {
            (component_index, atom.atom_id)
            for component_index, component in enumerate(self.components)
            for atom in component.atoms
        }
        edges: set[frozenset[tuple[int, str]]] = set()
        for bond in self.bonds:
            endpoints = frozenset(((bond.component_a, bond.atom_a), (bond.component_b, bond.atom_b)))
            if len(endpoints) != 2:
                raise ValueError("a molecule bond cannot connect an atom to itself")
            if not endpoints <= sites:
                raise ValueError("molecule bond references an unknown component atom")
            if bond.component_a == bond.component_b:
                raise ValueError("MoleculeBond must connect different component occurrences")
            if endpoints in edges:
                raise ValueError("molecule bonds contain a duplicate physical edge")
            edges.add(endpoints)
            if bond.stereo_atoms is not None and not set(bond.stereo_atoms) <= sites:
                raise ValueError("molecule bond stereo references an unknown component atom")
            leaving = {
                *((bond.component_a, atom_id) for atom_id in bond.leaving_a),
                *((bond.component_b, atom_id) for atom_id in bond.leaving_b),
            }
            if not leaving <= sites:
                raise ValueError("molecule bond leaving set references an unknown component atom")
            if endpoints & leaving:
                raise ValueError("a molecule bond endpoint cannot also be a leaving atom")

    @staticmethod
    def from_smiles(
        smiles: str,
        *,
        atom_ids: tuple[str, ...] | None = None,
    ) -> Molecule:
        r"""
        Construct a molecule from SMILES with RDKit.

        Args:
            smiles: SMILES representation to parse.
            atom_ids: Optional component-local identifiers in RDKit atom order.

        Returns:
            The parsed molecular graph.
        """

        from ._smiles_import import molecule_from_smiles

        return molecule_from_smiles(smiles, atom_ids=atom_ids)


@dataclass(frozen=True)
class AtomSite:
    r"""
    An atom occurrence in a molecular system.

    Args:
        instance_id: Molecular instance containing the atom.
        component_index: Zero-based component occurrence within the instance.
        atom_id: Component-local atom identifier.
    """

    instance_id: str
    component_index: int
    atom_id: str


@dataclass(frozen=True)
class SystemBond:
    r"""
    A bond between atom occurrences with an explicit mathematical owner.

    Args:
        site_a: First atom occurrence.
        site_b: Second atom occurrence.
        order: Bond order.
        aromatic: Whether the bond is aromatic.
        stereo: Optional bond stereochemistry descriptor.
        stereo_sites: Optional atom occurrences defining bond stereochemistry.
        role: Mathematical owner of the bond.
        leaving_a: Atoms removed beside the first endpoint when the bond forms.
        leaving_b: Atoms removed beside the second endpoint when the bond forms.
        kind: Covalent or dative bond kind.
        direction: Donor direction for a dative bond.
    """

    site_a: AtomSite
    site_b: AtomSite
    order: float | None = 1.0
    aromatic: bool | None = False
    stereo: str | None = None
    stereo_sites: tuple[AtomSite, AtomSite] | None = None
    role: BondRole = "assembly"
    leaving_a: frozenset[str] = frozenset()
    leaving_b: frozenset[str] = frozenset()
    kind: BondKind = "covalent"
    direction: BondDirection = "undirected"


_BondRecord: TypeAlias = tuple[
    int,
    int,
    float | None,
    bool | None,
    int,
    int,
    int,
    str | None,
    int,
    int,
]
_Facet = TypeVar("_Facet")


def _merge_optional(
    name: str,
    first: _Facet | None,
    second: _Facet | None,
    edge: tuple[int, int],
) -> _Facet | None:
    if first is not None and second is not None and first != second:
        raise ValueError(f"conflicting {name} for physical edge {edge}: {first!r} vs {second!r}")
    return first if second is None else second


def _canonical_bond(
    atom_a: int,
    atom_b: int,
    order: float | None,
    aromatic: bool | None,
    kind: int,
    direction: int,
    role: int,
    stereo: str | None,
    stereo_a: int,
    stereo_b: int,
) -> _BondRecord:
    if atom_a == atom_b:
        raise ValueError("a bond cannot connect an atom to itself")
    if (kind == BOND_KIND_DATIVE) != (direction != BOND_DIRECTION_UNDIRECTED):
        raise ValueError("only a dative bond has a directed donor endpoint")
    if atom_b < atom_a:
        atom_a, atom_b = atom_b, atom_a
        stereo_a, stereo_b = stereo_b, stereo_a
        direction = -direction
    return atom_a, atom_b, order, aromatic, kind, direction, role, stereo, stereo_a, stereo_b


def _merge_bond(first: _BondRecord, second: _BondRecord) -> _BondRecord:
    edge = first[0], first[1]
    if edge != (second[0], second[1]):
        raise ValueError("only bonds on the same physical edge can be merged")
    if first[4] != second[4]:
        raise ValueError(f"conflicting bond kind for physical edge {edge}")
    if first[5] != second[5]:
        raise ValueError(f"conflicting bond direction for physical edge {edge}")
    order = _merge_optional("bond order", first[2], second[2], edge)
    aromatic = _merge_optional("aromaticity", first[3], second[3], edge)
    stereo = _merge_optional("stereochemistry", first[7], second[7], edge)
    first_sites = None if first[8] < 0 else (first[8], first[9])
    second_sites = None if second[8] < 0 else (second[8], second[9])
    stereo_sites = _merge_optional("stereochemistry sites", first_sites, second_sites, edge)
    stereo_a, stereo_b = (-1, -1) if stereo_sites is None else stereo_sites
    return (
        first[0],
        first[1],
        order,
        aromatic,
        first[4],
        first[5],
        min(first[6], second[6]),
        stereo,
        stereo_a,
        stereo_b,
    )


@frozen_tensorclass
class MolecularSystem:
    r"""
    Tensor-native molecular topology over entity, instance, component, atom, and bond axes.
    """

    symbols: tuple[str, ...]

    system_entity_ptr: Tensor
    system_instance_ptr: Tensor
    entity_symbol: Tensor
    entity_type: Tensor
    entity_molecule_type: Tensor
    instance_symbol: Tensor
    instance_entity: Tensor
    instance_component_ptr: Tensor
    component_symbol: Tensor
    component_topology_complete: Tensor
    component_atom_ptr: Tensor

    atom_symbol: Tensor
    atomic_number: Tensor
    isotope: Tensor
    isotope_known: Tensor
    formal_charge: Tensor
    formal_charge_known: Tensor
    radical_electrons: Tensor
    radical_electrons_known: Tensor
    explicit_hydrogens: Tensor
    explicit_hydrogens_known: Tensor
    implicit_hydrogens: Tensor
    implicit_hydrogens_known: Tensor
    implicit_hydrogens_allowed: Tensor
    implicit_hydrogens_allowed_known: Tensor
    atom_aromatic: Tensor
    atom_aromatic_known: Tensor
    atom_stereo: Tensor
    reference_coordinates: Tensor
    reference_coordinate_mask: Tensor

    edge_index: Tensor
    bond_order: Tensor
    bond_order_known: Tensor
    bond_aromatic: Tensor
    bond_aromatic_known: Tensor
    bond_kind: Tensor
    bond_direction: Tensor
    bond_role: Tensor
    bond_stereo: Tensor
    bond_stereo_index: Tensor

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> MolecularSystem:
        r"""
        Move the graph while casting only floating-point values.

        This follows ``nn.Module.to`` rather than casting pointer, index, chemistry,
        and mask tensors to the requested floating dtype.
        """

        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype was specified twice")
            dtype = device
            device = None
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("MolecularSystem.to only accepts floating or complex dtypes")

        def move(value: Tensor) -> Tensor:
            return value.to(
                device=device,
                dtype=(dtype if value.is_floating_point() or value.is_complex() else None),
                non_blocking=non_blocking,
            )

        return MolecularSystem(  # type: ignore[call-arg]
            symbols=self.symbols,
            system_entity_ptr=move(self.system_entity_ptr),
            system_instance_ptr=move(self.system_instance_ptr),
            entity_symbol=move(self.entity_symbol),
            entity_type=move(self.entity_type),
            entity_molecule_type=move(self.entity_molecule_type),
            instance_symbol=move(self.instance_symbol),
            instance_entity=move(self.instance_entity),
            instance_component_ptr=move(self.instance_component_ptr),
            component_symbol=move(self.component_symbol),
            component_topology_complete=move(self.component_topology_complete),
            component_atom_ptr=move(self.component_atom_ptr),
            atom_symbol=move(self.atom_symbol),
            atomic_number=move(self.atomic_number),
            isotope=move(self.isotope),
            isotope_known=move(self.isotope_known),
            formal_charge=move(self.formal_charge),
            formal_charge_known=move(self.formal_charge_known),
            radical_electrons=move(self.radical_electrons),
            radical_electrons_known=move(self.radical_electrons_known),
            explicit_hydrogens=move(self.explicit_hydrogens),
            explicit_hydrogens_known=move(self.explicit_hydrogens_known),
            implicit_hydrogens=move(self.implicit_hydrogens),
            implicit_hydrogens_known=move(self.implicit_hydrogens_known),
            implicit_hydrogens_allowed=move(self.implicit_hydrogens_allowed),
            implicit_hydrogens_allowed_known=move(self.implicit_hydrogens_allowed_known),
            atom_aromatic=move(self.atom_aromatic),
            atom_aromatic_known=move(self.atom_aromatic_known),
            atom_stereo=move(self.atom_stereo),
            reference_coordinates=move(self.reference_coordinates),
            reference_coordinate_mask=move(self.reference_coordinate_mask),
            edge_index=move(self.edge_index),
            bond_order=move(self.bond_order),
            bond_order_known=move(self.bond_order_known),
            bond_aromatic=move(self.bond_aromatic),
            bond_aromatic_known=move(self.bond_aromatic_known),
            bond_kind=move(self.bond_kind),
            bond_direction=move(self.bond_direction),
            bond_role=move(self.bond_role),
            bond_stereo=move(self.bond_stereo),
            bond_stereo_index=move(self.bond_stereo_index),
            batch_size=[],
        )

    def float(self) -> MolecularSystem:
        return self.to(dtype=torch.float32)

    def double(self) -> MolecularSystem:
        return self.to(dtype=torch.float64)

    def half(self) -> MolecularSystem:
        return self.to(dtype=torch.float16)

    def bfloat16(self) -> MolecularSystem:
        return self.to(dtype=torch.bfloat16)

    @classmethod
    def from_entities(
        cls,
        entities: Mapping[str, PolymerSequence | Molecule],
        instances: Mapping[str, str],
        bonds: Iterable[SystemBond] = (),
        *,
        instance_values: Mapping[str, PolymerSequence | Molecule] | None = None,
        device: torch.device | str | None = None,
    ) -> MolecularSystem:
        r"""
        Materialize construction values into one realized tensor graph.
        """

        if set(entities) != set(instances.values()):
            raise ValueError("entities must be exactly those named by instances")
        if instance_values is not None and not set(instance_values) <= set(instances):
            raise ValueError("instance_values keys must name declared instances")

        symbols: list[str] = []
        symbol_index: dict[str, int] = {}

        def symbol(value: str | None) -> int:
            if value is None:
                return -1
            index = symbol_index.get(value)
            if index is None:
                index = len(symbols)
                symbols.append(value)
                symbol_index[value] = index
            return index

        entity_rows = {entity_id: index for index, entity_id in enumerate(entities)}
        entity_symbol = [symbol(entity_id) for entity_id in entities]
        entity_type = [
            _ENTITY_TYPE[entity.polymer_type] if isinstance(entity, PolymerSequence) else ENTITY_MOLECULE
            for entity in entities.values()
        ]
        entity_molecule_type = [
            (MOLECULE_NON_POLYMER if isinstance(entity, PolymerSequence) else _MOLECULE_TYPE[entity.molecule_type])
            for entity in entities.values()
        ]

        instance_symbol: list[int] = []
        instance_entity: list[int] = []
        instance_component_ptr = [0]
        component_symbol: list[int] = []
        component_topology_complete: list[bool] = []
        component_atom_ptr = [0]

        atom_symbol: list[int] = []
        atomic_number: list[int] = []
        isotope: list[int] = []
        isotope_known: list[bool] = []
        formal_charge: list[int] = []
        formal_charge_known: list[bool] = []
        radical_electrons: list[int] = []
        radical_electrons_known: list[bool] = []
        explicit_hydrogens: list[int] = []
        explicit_hydrogens_known: list[bool] = []
        implicit_hydrogens: list[int] = []
        implicit_hydrogens_known: list[bool] = []
        implicit_hydrogens_allowed: list[bool] = []
        implicit_hydrogens_allowed_known: list[bool] = []
        atom_aromatic: list[bool] = []
        atom_aromatic_known: list[bool] = []
        atom_stereo: list[int] = []
        reference_coordinates: list[tuple[float, float, float]] = []
        reference_coordinate_mask: list[bool] = []

        atom_by_site: dict[AtomSite, int] = {}
        edge_rows: dict[tuple[int, int], _BondRecord] = {}
        removed_atoms: set[int] = set()
        formed_endpoints: set[int] = set()

        def optional_integer(value: int | None, values: list[int], known: list[bool]) -> None:
            values.append(0 if value is None else value)
            known.append(value is not None)

        def add_bond(record: _BondRecord) -> None:
            key = record[0], record[1]
            existing = edge_rows.get(key)
            edge_rows[key] = record if existing is None else _merge_bond(existing, record)

        def resolve_stereo(
            instance_id: str,
            first: tuple[int, str] | None,
            second: tuple[int, str] | None,
        ) -> tuple[int, int]:
            if first is None or second is None:
                return -1, -1
            return (
                atom_by_site[AtomSite(instance_id, first[0], first[1])],
                atom_by_site[AtomSite(instance_id, second[0], second[1])],
            )

        def mark_leaving(instance_id: str, component_index: int, atom_ids: Iterable[str]) -> None:
            for atom_id in atom_ids:
                removed_atoms.add(atom_by_site[AtomSite(instance_id, component_index, atom_id)])

        for instance_id, entity_id in instances.items():
            declared_entity = entities[entity_id]
            entity = declared_entity if instance_values is None else instance_values.get(instance_id, declared_entity)
            if isinstance(entity, PolymerSequence) != isinstance(declared_entity, PolymerSequence):
                raise ValueError(f"instance {instance_id!r} must have the same entity kind as " f"entity {entity_id!r}")
            if (
                isinstance(entity, PolymerSequence)
                and isinstance(declared_entity, PolymerSequence)
                and entity.polymer_type != declared_entity.polymer_type
            ):
                raise ValueError(
                    f"instance {instance_id!r} must have the same polymer type as " f"entity {entity_id!r}"
                )
            if (
                isinstance(entity, Molecule)
                and isinstance(declared_entity, Molecule)
                and entity.molecule_type != declared_entity.molecule_type
            ):
                raise ValueError(
                    f"instance {instance_id!r} must have the same molecule type as " f"entity {entity_id!r}"
                )
            if len(entity.components) != len(declared_entity.components):
                raise ValueError(
                    f"instance {instance_id!r} must have the same component-axis " f"length as entity {entity_id!r}"
                )
            instance_symbol.append(symbol(instance_id))
            instance_entity.append(entity_rows[entity_id])
            instance_component_ptr.append(instance_component_ptr[-1] + len(entity.components))
            for component_index, component in enumerate(entity.components):
                component_symbol.append(symbol(component.component_id))
                component_topology_complete.append(component.bonds is not None)
                for atom_index, atom in enumerate(component.atoms):
                    global_atom = len(atom_symbol)
                    atom_by_site[AtomSite(instance_id, component_index, atom.atom_id)] = global_atom
                    atom_symbol.append(symbol(atom.atom_id))
                    hydrogen_isotope = {"D": 2, "T": 3}.get(atom.element)
                    atomic_number.append(
                        ATOMIC_NUMBER_BY_ELEMENT["H" if hydrogen_isotope is not None else atom.element]
                    )
                    atom_isotope = (
                        hydrogen_isotope if hydrogen_isotope is not None and atom.isotope is None else atom.isotope
                    )
                    optional_integer(atom_isotope, isotope, isotope_known)
                    optional_integer(atom.formal_charge, formal_charge, formal_charge_known)
                    optional_integer(atom.radical_electrons, radical_electrons, radical_electrons_known)
                    optional_integer(atom.explicit_hydrogens, explicit_hydrogens, explicit_hydrogens_known)
                    optional_integer(atom.implicit_hydrogens, implicit_hydrogens, implicit_hydrogens_known)
                    implicit_hydrogens_allowed.append(
                        False if atom.implicit_hydrogens_allowed is None else atom.implicit_hydrogens_allowed
                    )
                    implicit_hydrogens_allowed_known.append(atom.implicit_hydrogens_allowed is not None)
                    atom_aromatic.append(False if atom.aromatic is None else atom.aromatic)
                    atom_aromatic_known.append(atom.aromatic is not None)
                    atom_stereo.append(symbol(atom.stereo))
                    coordinate = (
                        component.reference_coordinates[atom_index] if component.reference_coordinates else None
                    )
                    reference_coordinates.append((0.0, 0.0, 0.0) if coordinate is None else coordinate)
                    reference_coordinate_mask.append(coordinate is not None)
                component_atom_ptr.append(len(atom_symbol))

            for component_index, component in enumerate(entity.components):
                if component.bonds is None:
                    continue
                for component_bond in component.bonds:
                    atom_a = atom_by_site[AtomSite(instance_id, component_index, component_bond.atom_a)]
                    atom_b = atom_by_site[AtomSite(instance_id, component_index, component_bond.atom_b)]
                    stereo_a, stereo_b = resolve_stereo(
                        instance_id,
                        (
                            None
                            if component_bond.stereo_atoms is None
                            else (component_index, component_bond.stereo_atoms[0])
                        ),
                        (
                            None
                            if component_bond.stereo_atoms is None
                            else (component_index, component_bond.stereo_atoms[1])
                        ),
                    )
                    add_bond(
                        _canonical_bond(
                            atom_a,
                            atom_b,
                            component_bond.order,
                            component_bond.aromatic,
                            _BOND_KIND[component_bond.kind],
                            _BOND_DIRECTION[component_bond.direction],
                            BOND_ROLE_COMPONENT,
                            component_bond.stereo,
                            stereo_a,
                            stereo_b,
                        )
                    )

            if isinstance(entity, PolymerSequence):
                for component_index, (left, right) in enumerate(zip(entity.components, entity.components[1:])):
                    if left.next_attachment is None or right.previous_attachment is None:
                        continue
                    mark_leaving(instance_id, component_index, left.next_attachment.leaving_atom_ids)
                    mark_leaving(
                        instance_id,
                        component_index + 1,
                        right.previous_attachment.leaving_atom_ids,
                    )
                    add_bond(
                        _canonical_bond(
                            atom_by_site[AtomSite(instance_id, component_index, left.next_attachment.atom_id)],
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    component_index + 1,
                                    right.previous_attachment.atom_id,
                                )
                            ],
                            1.0,
                            False,
                            BOND_KIND_COVALENT,
                            BOND_DIRECTION_UNDIRECTED,
                            BOND_ROLE_POLYMER,
                            None,
                            -1,
                            -1,
                        )
                    )
                    formed_endpoints.update(
                        (
                            atom_by_site[AtomSite(instance_id, component_index, left.next_attachment.atom_id)],
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    component_index + 1,
                                    right.previous_attachment.atom_id,
                                )
                            ],
                        )
                    )
            else:
                for molecule_bond in entity.bonds:
                    mark_leaving(instance_id, molecule_bond.component_a, molecule_bond.leaving_a)
                    mark_leaving(instance_id, molecule_bond.component_b, molecule_bond.leaving_b)
                    stereo_a, stereo_b = resolve_stereo(
                        instance_id,
                        None if molecule_bond.stereo_atoms is None else molecule_bond.stereo_atoms[0],
                        None if molecule_bond.stereo_atoms is None else molecule_bond.stereo_atoms[1],
                    )
                    add_bond(
                        _canonical_bond(
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    molecule_bond.component_a,
                                    molecule_bond.atom_a,
                                )
                            ],
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    molecule_bond.component_b,
                                    molecule_bond.atom_b,
                                )
                            ],
                            molecule_bond.order,
                            molecule_bond.aromatic,
                            _BOND_KIND[molecule_bond.kind],
                            _BOND_DIRECTION[molecule_bond.direction],
                            BOND_ROLE_ENTITY,
                            molecule_bond.stereo,
                            stereo_a,
                            stereo_b,
                        )
                    )
                    formed_endpoints.update(
                        (
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    molecule_bond.component_a,
                                    molecule_bond.atom_a,
                                )
                            ],
                            atom_by_site[
                                AtomSite(
                                    instance_id,
                                    molecule_bond.component_b,
                                    molecule_bond.atom_b,
                                )
                            ],
                        )
                    )

        for system_bond in bonds:
            atom_a = atom_by_site[system_bond.site_a]
            atom_b = atom_by_site[system_bond.site_b]
            mark_leaving(
                system_bond.site_a.instance_id,
                system_bond.site_a.component_index,
                system_bond.leaving_a,
            )
            mark_leaving(
                system_bond.site_b.instance_id,
                system_bond.site_b.component_index,
                system_bond.leaving_b,
            )
            stereo_a, stereo_b = (
                (-1, -1)
                if system_bond.stereo_sites is None
                else (
                    atom_by_site[system_bond.stereo_sites[0]],
                    atom_by_site[system_bond.stereo_sites[1]],
                )
            )
            add_bond(
                _canonical_bond(
                    atom_a,
                    atom_b,
                    system_bond.order,
                    system_bond.aromatic,
                    _BOND_KIND[system_bond.kind],
                    _BOND_DIRECTION[system_bond.direction],
                    _BOND_ROLE[system_bond.role],
                    system_bond.stereo,
                    stereo_a,
                    stereo_b,
                )
            )
            formed_endpoints.update((atom_a, atom_b))

        bond_rows = [edge_rows[key] for key in sorted(edge_rows)]
        bond_stereo = [symbol(row[7]) for row in bond_rows]

        def tensor(values: Any, dtype: torch.dtype) -> Tensor:
            return torch.tensor(values, dtype=dtype, device=device)

        graph = cls(  # type: ignore[call-arg]
            symbols=tuple(symbols),
            system_entity_ptr=tensor([0, len(entities)], torch.int64),
            system_instance_ptr=tensor([0, len(instances)], torch.int64),
            entity_symbol=tensor(entity_symbol, torch.int64),
            entity_type=tensor(entity_type, torch.uint8),
            entity_molecule_type=tensor(entity_molecule_type, torch.uint8),
            instance_symbol=tensor(instance_symbol, torch.int64),
            instance_entity=tensor(instance_entity, torch.int64),
            instance_component_ptr=tensor(instance_component_ptr, torch.int64),
            component_symbol=tensor(component_symbol, torch.int64),
            component_topology_complete=tensor(component_topology_complete, torch.bool),
            component_atom_ptr=tensor(component_atom_ptr, torch.int64),
            atom_symbol=tensor(atom_symbol, torch.int64),
            atomic_number=tensor(atomic_number, torch.int16),
            isotope=tensor(isotope, torch.int16),
            isotope_known=tensor(isotope_known, torch.bool),
            formal_charge=tensor(formal_charge, torch.int16),
            formal_charge_known=tensor(formal_charge_known, torch.bool),
            radical_electrons=tensor(radical_electrons, torch.int16),
            radical_electrons_known=tensor(radical_electrons_known, torch.bool),
            explicit_hydrogens=tensor(explicit_hydrogens, torch.int16),
            explicit_hydrogens_known=tensor(explicit_hydrogens_known, torch.bool),
            implicit_hydrogens=tensor(implicit_hydrogens, torch.int16),
            implicit_hydrogens_known=tensor(implicit_hydrogens_known, torch.bool),
            implicit_hydrogens_allowed=tensor(implicit_hydrogens_allowed, torch.bool),
            implicit_hydrogens_allowed_known=tensor(implicit_hydrogens_allowed_known, torch.bool),
            atom_aromatic=tensor(atom_aromatic, torch.bool),
            atom_aromatic_known=tensor(atom_aromatic_known, torch.bool),
            atom_stereo=tensor(atom_stereo, torch.int64),
            reference_coordinates=tensor(reference_coordinates, torch.float32).reshape(-1, 3),
            reference_coordinate_mask=tensor(reference_coordinate_mask, torch.bool),
            edge_index=tensor(
                ([row[0] for row in bond_rows], [row[1] for row in bond_rows]),
                torch.int64,
            ).reshape(2, -1),
            bond_order=tensor([0.0 if row[2] is None else row[2] for row in bond_rows], torch.float32),
            bond_order_known=tensor([row[2] is not None for row in bond_rows], torch.bool),
            bond_aromatic=tensor([False if row[3] is None else row[3] for row in bond_rows], torch.bool),
            bond_aromatic_known=tensor([row[3] is not None for row in bond_rows], torch.bool),
            bond_kind=tensor([row[4] for row in bond_rows], torch.uint8),
            bond_direction=tensor([row[5] for row in bond_rows], torch.int8),
            bond_role=tensor([row[6] for row in bond_rows], torch.uint8),
            bond_stereo=tensor(bond_stereo, torch.int64),
            bond_stereo_index=tensor(
                ([row[8] for row in bond_rows], [row[9] for row in bond_rows]),
                torch.int64,
            ).reshape(2, -1),
            batch_size=[],
        )
        if not removed_atoms:
            return graph
        keep = torch.ones(len(atom_symbol), dtype=torch.bool, device=device)
        keep[tensor(sorted(removed_atoms), torch.int64)] = False
        if formed_endpoints & removed_atoms:
            raise ValueError("a bond endpoint cannot also be a leaving atom")
        return _atom_subgraph(graph, keep)[0]

    def _unique_index(self, values: Tensor, value: str) -> int:
        rows = [index for index, symbol in enumerate(values.tolist()) if self.symbols[symbol] == value]
        if not rows:
            raise KeyError(value)
        if len(rows) != 1:
            raise ValueError(f"{value!r} is ambiguous in a batched molecular system")
        return rows[0]

    def entity_index(self, entity_id: str) -> int:
        return self._unique_index(self.entity_symbol, entity_id)

    def instance_index(self, instance_id: str) -> int:
        return self._unique_index(self.instance_symbol, instance_id)

    def component_index(self, reference: ComponentReference) -> int:
        instance = self.instance_index(reference[0])
        start = int(self.instance_component_ptr[instance].item())
        stop = int(self.instance_component_ptr[instance + 1].item())
        index = start + reference[1]
        if index < start or index >= stop:
            raise IndexError(reference[1])
        return index

    def atom_index(self, site: AtomSite) -> int:
        component = self.component_index((site.instance_id, site.component_index))
        start = int(self.component_atom_ptr[component].item())
        stop = int(self.component_atom_ptr[component + 1].item())
        rows = [
            index
            for index, symbol in enumerate(self.atom_symbol[start:stop].tolist())
            if self.symbols[symbol] == site.atom_id
        ]
        if len(rows) != 1:
            raise KeyError(site)
        return start + rows[0]

    def entity_id(self, index: int) -> str:
        if index < 0:
            raise IndexError(index)
        return str(self.symbols[int(self.entity_symbol[index].item())])

    def instance_id(self, index: int) -> str:
        if index < 0:
            raise IndexError(index)
        return str(self.symbols[int(self.instance_symbol[index].item())])

    def component_reference(self, index: int) -> ComponentReference:
        if index < 0:
            raise IndexError(index)
        instance = int(
            torch.searchsorted(
                self.instance_component_ptr[1:],
                torch.tensor(index, device=self.instance_component_ptr.device),
                right=True,
            ).item()
        )
        return self.instance_id(instance), index - int(self.instance_component_ptr[instance].item())

    def atom_site(self, index: int) -> AtomSite:
        if index < 0:
            raise IndexError(index)
        component = int(
            torch.searchsorted(
                self.component_atom_ptr[1:],
                torch.tensor(index, device=self.component_atom_ptr.device),
                right=True,
            ).item()
        )
        instance_id, component_index = self.component_reference(component)
        return AtomSite(
            instance_id,
            component_index,
            str(self.symbols[int(self.atom_symbol[index].item())]),
        )

    @classmethod
    def batch(cls, systems: Sequence[MolecularSystem]) -> MolecularSystem:
        r"""
        Concatenate molecular systems as a ragged disjoint union.
        """

        device = systems[0].edge_index.device
        symbols: list[str] = []
        entity_offset = atom_offset = symbol_offset = 0

        def pointers(name: str) -> Tensor:
            counts = torch.cat([getattr(system, name)[1:] - getattr(system, name)[:-1] for system in systems])
            return torch.cat([torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)])

        def concatenate(name: str) -> Tensor:
            return torch.cat([getattr(system, name) for system in systems])

        entity_symbols = []
        instance_symbols = []
        component_symbols = []
        atom_symbols = []
        atom_stereo = []
        bond_stereo = []
        instance_entities = []
        edge_indices = []
        stereo_indices = []
        for system in systems:
            symbols.extend(system.symbols)
            entity_symbols.append(system.entity_symbol + symbol_offset)
            instance_symbols.append(system.instance_symbol + symbol_offset)
            component_symbols.append(system.component_symbol + symbol_offset)
            atom_symbols.append(system.atom_symbol + symbol_offset)
            atom_stereo.append(
                torch.where(
                    system.atom_stereo < 0,
                    system.atom_stereo,
                    system.atom_stereo + symbol_offset,
                )
            )
            bond_stereo.append(
                torch.where(
                    system.bond_stereo < 0,
                    system.bond_stereo,
                    system.bond_stereo + symbol_offset,
                )
            )
            instance_entities.append(system.instance_entity + entity_offset)
            edge_indices.append(system.edge_index + atom_offset)
            stereo_indices.append(
                torch.where(
                    system.bond_stereo_index < 0,
                    system.bond_stereo_index,
                    system.bond_stereo_index + atom_offset,
                )
            )
            entity_offset += system.entity_symbol.numel()
            atom_offset += system.atom_symbol.numel()
            symbol_offset += len(system.symbols)

        return cls(  # type: ignore[call-arg]
            symbols=tuple(symbols),
            system_entity_ptr=pointers("system_entity_ptr"),
            system_instance_ptr=pointers("system_instance_ptr"),
            entity_symbol=torch.cat(entity_symbols),
            entity_type=concatenate("entity_type"),
            entity_molecule_type=concatenate("entity_molecule_type"),
            instance_symbol=torch.cat(instance_symbols),
            instance_entity=torch.cat(instance_entities),
            instance_component_ptr=pointers("instance_component_ptr"),
            component_symbol=torch.cat(component_symbols),
            component_topology_complete=concatenate("component_topology_complete"),
            component_atom_ptr=pointers("component_atom_ptr"),
            atom_symbol=torch.cat(atom_symbols),
            atomic_number=concatenate("atomic_number"),
            isotope=concatenate("isotope"),
            isotope_known=concatenate("isotope_known"),
            formal_charge=concatenate("formal_charge"),
            formal_charge_known=concatenate("formal_charge_known"),
            radical_electrons=concatenate("radical_electrons"),
            radical_electrons_known=concatenate("radical_electrons_known"),
            explicit_hydrogens=concatenate("explicit_hydrogens"),
            explicit_hydrogens_known=concatenate("explicit_hydrogens_known"),
            implicit_hydrogens=concatenate("implicit_hydrogens"),
            implicit_hydrogens_known=concatenate("implicit_hydrogens_known"),
            implicit_hydrogens_allowed=concatenate("implicit_hydrogens_allowed"),
            implicit_hydrogens_allowed_known=concatenate("implicit_hydrogens_allowed_known"),
            atom_aromatic=concatenate("atom_aromatic"),
            atom_aromatic_known=concatenate("atom_aromatic_known"),
            atom_stereo=torch.cat(atom_stereo),
            reference_coordinates=concatenate("reference_coordinates"),
            reference_coordinate_mask=concatenate("reference_coordinate_mask"),
            edge_index=torch.cat(edge_indices, dim=1),
            bond_order=concatenate("bond_order"),
            bond_order_known=concatenate("bond_order_known"),
            bond_aromatic=concatenate("bond_aromatic"),
            bond_aromatic_known=concatenate("bond_aromatic_known"),
            bond_kind=concatenate("bond_kind"),
            bond_direction=concatenate("bond_direction"),
            bond_role=concatenate("bond_role"),
            bond_stereo=torch.cat(bond_stereo),
            bond_stereo_index=torch.cat(stereo_indices, dim=1),
            batch_size=[],
        )

    def unbatch(self) -> tuple[MolecularSystem, ...]:
        r"""
        Split a ragged disjoint union into single-system values.
        """

        outputs = []
        for system_index in range(self.system_entity_ptr.numel() - 1):
            entity_start = int(self.system_entity_ptr[system_index].item())
            entity_stop = int(self.system_entity_ptr[system_index + 1].item())
            instance_start = int(self.system_instance_ptr[system_index].item())
            instance_stop = int(self.system_instance_ptr[system_index + 1].item())
            component_start = int(self.instance_component_ptr[instance_start].item())
            component_stop = int(self.instance_component_ptr[instance_stop].item())
            atom_start = int(self.component_atom_ptr[component_start].item())
            atom_stop = int(self.component_atom_ptr[component_stop].item())
            edge_mask = (self.edge_index[0] >= atom_start) & (self.edge_index[1] < atom_stop)
            entity_symbol = self.entity_symbol[entity_start:entity_stop]
            instance_symbol = self.instance_symbol[instance_start:instance_stop]
            component_symbol = self.component_symbol[component_start:component_stop]
            atom_symbol = self.atom_symbol[atom_start:atom_stop]
            atom_stereo = self.atom_stereo[atom_start:atom_stop]
            bond_stereo = self.bond_stereo[edge_mask]
            used_symbols = torch.unique(
                torch.cat(
                    (
                        entity_symbol,
                        instance_symbol,
                        component_symbol,
                        atom_symbol,
                        atom_stereo[atom_stereo >= 0],
                        bond_stereo[bond_stereo >= 0],
                    )
                ),
                sorted=True,
            )
            symbol_map = torch.full(
                (len(self.symbols),),
                -1,
                dtype=torch.int64,
                device=entity_symbol.device,
            )
            symbol_map[used_symbols] = torch.arange(
                used_symbols.numel(),
                dtype=torch.int64,
                device=entity_symbol.device,
            )
            symbols = tuple(self.symbols[index] for index in used_symbols.detach().cpu().tolist())
            outputs.append(
                MolecularSystem(  # type: ignore[call-arg]
                    symbols=symbols,
                    system_entity_ptr=self.system_entity_ptr.new_tensor([0, entity_stop - entity_start]),
                    system_instance_ptr=self.system_instance_ptr.new_tensor([0, instance_stop - instance_start]),
                    entity_symbol=symbol_map[entity_symbol],
                    entity_type=self.entity_type[entity_start:entity_stop],
                    entity_molecule_type=self.entity_molecule_type[entity_start:entity_stop],
                    instance_symbol=symbol_map[instance_symbol],
                    instance_entity=(self.instance_entity[instance_start:instance_stop] - entity_start),
                    instance_component_ptr=(
                        self.instance_component_ptr[instance_start : instance_stop + 1] - component_start
                    ),
                    component_symbol=symbol_map[component_symbol],
                    component_topology_complete=self.component_topology_complete[component_start:component_stop],
                    component_atom_ptr=(self.component_atom_ptr[component_start : component_stop + 1] - atom_start),
                    atom_symbol=symbol_map[atom_symbol],
                    atomic_number=self.atomic_number[atom_start:atom_stop],
                    isotope=self.isotope[atom_start:atom_stop],
                    isotope_known=self.isotope_known[atom_start:atom_stop],
                    formal_charge=self.formal_charge[atom_start:atom_stop],
                    formal_charge_known=self.formal_charge_known[atom_start:atom_stop],
                    radical_electrons=self.radical_electrons[atom_start:atom_stop],
                    radical_electrons_known=self.radical_electrons_known[atom_start:atom_stop],
                    explicit_hydrogens=self.explicit_hydrogens[atom_start:atom_stop],
                    explicit_hydrogens_known=self.explicit_hydrogens_known[atom_start:atom_stop],
                    implicit_hydrogens=self.implicit_hydrogens[atom_start:atom_stop],
                    implicit_hydrogens_known=self.implicit_hydrogens_known[atom_start:atom_stop],
                    implicit_hydrogens_allowed=self.implicit_hydrogens_allowed[atom_start:atom_stop],
                    implicit_hydrogens_allowed_known=self.implicit_hydrogens_allowed_known[atom_start:atom_stop],
                    atom_aromatic=self.atom_aromatic[atom_start:atom_stop],
                    atom_aromatic_known=self.atom_aromatic_known[atom_start:atom_stop],
                    atom_stereo=torch.where(
                        atom_stereo < 0,
                        atom_stereo,
                        symbol_map[atom_stereo.clamp_min(0)],
                    ),
                    reference_coordinates=self.reference_coordinates[atom_start:atom_stop],
                    reference_coordinate_mask=self.reference_coordinate_mask[atom_start:atom_stop],
                    edge_index=self.edge_index[:, edge_mask] - atom_start,
                    bond_order=self.bond_order[edge_mask],
                    bond_order_known=self.bond_order_known[edge_mask],
                    bond_aromatic=self.bond_aromatic[edge_mask],
                    bond_aromatic_known=self.bond_aromatic_known[edge_mask],
                    bond_kind=self.bond_kind[edge_mask],
                    bond_direction=self.bond_direction[edge_mask],
                    bond_role=self.bond_role[edge_mask],
                    bond_stereo=torch.where(
                        bond_stereo < 0,
                        bond_stereo,
                        symbol_map[bond_stereo.clamp_min(0)],
                    ),
                    bond_stereo_index=torch.where(
                        self.bond_stereo_index[:, edge_mask] < 0,
                        self.bond_stereo_index[:, edge_mask],
                        self.bond_stereo_index[:, edge_mask] - atom_start,
                    ),
                    batch_size=[],
                )
            )
        return tuple(outputs)


def disjoint_union(
    systems: Sequence[MolecularSystem],
) -> tuple[MolecularSystem, tuple[Tensor, ...], tuple[Tensor, ...]]:
    r"""
    Join single-system molecular graphs without identifying any of their nodes.

    Entity, instance, component, atom, and edge rows concatenate in input order.
    Entity and instance symbols must be unique across inputs because the union
    preserves, rather than merges, their identities.

    Args:
        systems: Single-system molecular graphs to join.

    Returns:
        The joined single-system graph, followed by one component-row mapping and
        one atom-row mapping for each input graph.
    """

    entity_symbols: set[str] = set()
    instance_symbols: set[str] = set()
    component_maps: list[Tensor] = []
    atom_maps: list[Tensor] = []
    component_offset = 0
    atom_offset = 0

    for index, system in enumerate(systems):
        if system.system_entity_ptr.numel() != 2 or system.system_instance_ptr.numel() != 2:
            raise ValueError(f"disjoint_union input {index} must contain exactly one system")

        input_entity_symbols = {system.symbols[row] for row in system.entity_symbol.tolist()}
        duplicate_entities = entity_symbols & input_entity_symbols
        if duplicate_entities:
            raise ValueError(
                "disjoint_union entity symbols must be unique across inputs: " f"{sorted(duplicate_entities)!r}"
            )
        entity_symbols.update(input_entity_symbols)

        input_instance_symbols = {system.symbols[row] for row in system.instance_symbol.tolist()}
        duplicate_instances = instance_symbols & input_instance_symbols
        if duplicate_instances:
            raise ValueError(
                "disjoint_union instance symbols must be unique across inputs: " f"{sorted(duplicate_instances)!r}"
            )
        instance_symbols.update(input_instance_symbols)

        component_count = system.component_symbol.numel()
        atom_count = system.atom_symbol.numel()
        component_maps.append(
            torch.arange(
                component_count,
                dtype=torch.int64,
                device=system.component_symbol.device,
            )
            + component_offset
        )
        atom_maps.append(
            torch.arange(
                atom_count,
                dtype=torch.int64,
                device=system.atom_symbol.device,
            )
            + atom_offset
        )
        component_offset += component_count
        atom_offset += atom_count

    union = MolecularSystem.batch(systems)
    union = union.replace(  # type: ignore[attr-defined]
        system_entity_ptr=union.system_entity_ptr.new_tensor([0, union.entity_symbol.numel()]),
        system_instance_ptr=union.system_instance_ptr.new_tensor([0, union.instance_symbol.numel()]),
    )
    return union, tuple(component_maps), tuple(atom_maps)


def water_component_mask(system: MolecularSystem) -> Tensor:
    r"""
    Return the canonical component-axis mask for declared water components.
    """

    instance_is_water = system.entity_molecule_type[system.instance_entity] == MOLECULE_WATER
    return torch.repeat_interleave(
        instance_is_water,
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )


def _atom_subgraph(
    system: MolecularSystem,
    atom_mask: Tensor,
) -> tuple[MolecularSystem, Tensor]:
    r"""
    Keep selected atoms, compact atom indices, and return the old-to-new mapping.
    """

    old_to_new = torch.full(
        (system.atom_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    old_to_new[atom_mask] = torch.arange(
        int(atom_mask.sum().item()),
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    atom_component = torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    atom_counts = torch.zeros_like(system.component_symbol)
    atom_counts.scatter_add_(0, atom_component, atom_mask.to(torch.int64))
    component_atom_ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=system.atom_symbol.device),
            atom_counts.cumsum(0),
        ]
    )
    edge_mask = atom_mask[system.edge_index].all(0)
    affected_atoms = torch.unique(system.edge_index[:, ~edge_mask])
    affected_atoms = old_to_new[affected_atoms]
    affected_atoms = affected_atoms[affected_atoms >= 0]
    atom_stereo = system.atom_stereo[atom_mask].clone()
    atom_stereo[affected_atoms] = -1
    stereo = system.bond_stereo_index[:, edge_mask]
    stereo_has_any_site = (stereo >= 0).any(0)
    stereo_has_all_sites = (stereo >= 0).all(0)
    stereo_sites_retained = stereo_has_all_sites & atom_mask[stereo.clamp_min(0)].all(0)
    stereo_index = torch.where(
        stereo_sites_retained.unsqueeze(0),
        old_to_new[stereo.clamp_min(0)],
        torch.full_like(stereo, -1),
    )
    bond_stereo = torch.where(
        ~stereo_has_any_site | stereo_sites_retained,
        system.bond_stereo[edge_mask],
        torch.full_like(system.bond_stereo[edge_mask], -1),
    )
    return (
        MolecularSystem(  # type: ignore[call-arg]
            symbols=system.symbols,
            system_entity_ptr=system.system_entity_ptr,
            system_instance_ptr=system.system_instance_ptr,
            entity_symbol=system.entity_symbol,
            entity_type=system.entity_type,
            entity_molecule_type=system.entity_molecule_type,
            instance_symbol=system.instance_symbol,
            instance_entity=system.instance_entity,
            instance_component_ptr=system.instance_component_ptr,
            component_symbol=system.component_symbol,
            component_topology_complete=system.component_topology_complete,
            component_atom_ptr=component_atom_ptr,
            atom_symbol=system.atom_symbol[atom_mask],
            atomic_number=system.atomic_number[atom_mask],
            isotope=system.isotope[atom_mask],
            isotope_known=system.isotope_known[atom_mask],
            formal_charge=system.formal_charge[atom_mask],
            formal_charge_known=system.formal_charge_known[atom_mask],
            radical_electrons=system.radical_electrons[atom_mask],
            radical_electrons_known=system.radical_electrons_known[atom_mask],
            explicit_hydrogens=system.explicit_hydrogens[atom_mask],
            explicit_hydrogens_known=system.explicit_hydrogens_known[atom_mask],
            implicit_hydrogens=system.implicit_hydrogens[atom_mask],
            implicit_hydrogens_known=system.implicit_hydrogens_known[atom_mask],
            implicit_hydrogens_allowed=system.implicit_hydrogens_allowed[atom_mask],
            implicit_hydrogens_allowed_known=system.implicit_hydrogens_allowed_known[atom_mask],
            atom_aromatic=system.atom_aromatic[atom_mask],
            atom_aromatic_known=system.atom_aromatic_known[atom_mask],
            atom_stereo=atom_stereo,
            reference_coordinates=system.reference_coordinates[atom_mask],
            reference_coordinate_mask=system.reference_coordinate_mask[atom_mask],
            edge_index=old_to_new[system.edge_index[:, edge_mask]],
            bond_order=system.bond_order[edge_mask],
            bond_order_known=system.bond_order_known[edge_mask],
            bond_aromatic=system.bond_aromatic[edge_mask],
            bond_aromatic_known=system.bond_aromatic_known[edge_mask],
            bond_kind=system.bond_kind[edge_mask],
            bond_direction=system.bond_direction[edge_mask],
            bond_role=system.bond_role[edge_mask],
            bond_stereo=bond_stereo,
            bond_stereo_index=stereo_index,
            batch_size=[],
        ),
        old_to_new,
    )


def instance_subgraph(
    system: MolecularSystem,
    instance_mask: Tensor,
) -> tuple[MolecularSystem, Tensor, Tensor]:
    r"""
    Keep selected instances and return the old component and atom mappings.
    """

    if instance_mask.ndim != 1:
        raise ValueError("instance_mask must be one-dimensional")
    if instance_mask.dtype != torch.bool:
        raise TypeError("instance_mask must have dtype torch.bool")
    if instance_mask.numel() != system.instance_symbol.numel():
        raise ValueError("instance_mask must have the canonical instance-axis length")

    device = system.atom_symbol.device
    entity_count = system.entity_symbol.numel()
    instance_count = system.instance_symbol.numel()
    component_count = system.component_symbol.numel()
    system_count = system.system_entity_ptr.numel() - 1

    retained_entity_instances = torch.zeros(
        entity_count,
        dtype=torch.int64,
        device=device,
    )
    retained_entity_instances.scatter_add_(
        0,
        system.instance_entity,
        instance_mask.to(torch.int64),
    )
    entity_mask = retained_entity_instances > 0
    entity_to_new = torch.where(
        entity_mask,
        entity_mask.to(torch.int64).cumsum(0) - 1,
        torch.full(
            (entity_count,),
            -1,
            dtype=torch.int64,
            device=device,
        ),
    )

    system_index = torch.arange(system_count, dtype=torch.int64, device=device)
    entity_system = torch.repeat_interleave(
        system_index,
        system.system_entity_ptr[1:] - system.system_entity_ptr[:-1],
    )
    instance_system = torch.repeat_interleave(
        system_index,
        system.system_instance_ptr[1:] - system.system_instance_ptr[:-1],
    )
    retained_entities_per_system = torch.zeros(
        system_count,
        dtype=torch.int64,
        device=device,
    )
    retained_entities_per_system.scatter_add_(
        0,
        entity_system,
        entity_mask.to(torch.int64),
    )
    retained_instances_per_system = torch.zeros_like(retained_entities_per_system)
    retained_instances_per_system.scatter_add_(
        0,
        instance_system,
        instance_mask.to(torch.int64),
    )
    system_entity_ptr = torch.cat(
        [
            retained_entities_per_system.new_zeros(1),
            retained_entities_per_system.cumsum(0),
        ]
    )
    system_instance_ptr = torch.cat(
        [
            retained_instances_per_system.new_zeros(1),
            retained_instances_per_system.cumsum(0),
        ]
    )

    components_per_instance = system.instance_component_ptr[1:] - system.instance_component_ptr[:-1]
    component_instance = torch.repeat_interleave(
        torch.arange(instance_count, dtype=torch.int64, device=device),
        components_per_instance,
    )
    component_mask = instance_mask[component_instance]
    component_to_new = torch.where(
        component_mask,
        component_mask.to(torch.int64).cumsum(0) - 1,
        torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=device,
        ),
    )
    instance_component_ptr = torch.cat(
        [
            system.instance_component_ptr.new_zeros(1),
            components_per_instance[instance_mask].cumsum(0),
        ]
    )

    atoms_per_component = system.component_atom_ptr[1:] - system.component_atom_ptr[:-1]
    atom_component = torch.repeat_interleave(
        torch.arange(component_count, dtype=torch.int64, device=device),
        atoms_per_component,
    )
    atom_mask = component_mask[atom_component]
    atom_to_new = torch.where(
        atom_mask,
        atom_mask.to(torch.int64).cumsum(0) - 1,
        torch.full(
            (system.atom_symbol.numel(),),
            -1,
            dtype=torch.int64,
            device=device,
        ),
    )
    component_atom_ptr = torch.cat(
        [
            system.component_atom_ptr.new_zeros(1),
            atoms_per_component[component_mask].cumsum(0),
        ]
    )

    edge_mask = atom_mask[system.edge_index].all(0)
    affected_atoms = atom_to_new[system.edge_index[:, ~edge_mask]].reshape(-1)
    affected_atoms = affected_atoms[affected_atoms >= 0]
    atom_stereo = system.atom_stereo[atom_mask].clone()
    atom_stereo[affected_atoms] = -1

    stereo = system.bond_stereo_index[:, edge_mask]
    stereo_has_any_site = (stereo >= 0).any(0)
    stereo_has_all_sites = (stereo >= 0).all(0)
    stereo_sites_retained = stereo_has_all_sites & atom_mask[stereo.clamp_min(0)].all(0)
    stereo_index = torch.where(
        stereo_sites_retained.unsqueeze(0),
        atom_to_new[stereo.clamp_min(0)],
        torch.full_like(stereo, -1),
    )
    bond_stereo = torch.where(
        ~stereo_has_any_site | stereo_sites_retained,
        system.bond_stereo[edge_mask],
        torch.full_like(system.bond_stereo[edge_mask], -1),
    )

    return (
        MolecularSystem(  # type: ignore[call-arg]
            symbols=system.symbols,
            system_entity_ptr=system_entity_ptr,
            system_instance_ptr=system_instance_ptr,
            entity_symbol=system.entity_symbol[entity_mask],
            entity_type=system.entity_type[entity_mask],
            entity_molecule_type=system.entity_molecule_type[entity_mask],
            instance_symbol=system.instance_symbol[instance_mask],
            instance_entity=entity_to_new[system.instance_entity[instance_mask]],
            instance_component_ptr=instance_component_ptr,
            component_symbol=system.component_symbol[component_mask],
            component_topology_complete=system.component_topology_complete[component_mask],
            component_atom_ptr=component_atom_ptr,
            atom_symbol=system.atom_symbol[atom_mask],
            atomic_number=system.atomic_number[atom_mask],
            isotope=system.isotope[atom_mask],
            isotope_known=system.isotope_known[atom_mask],
            formal_charge=system.formal_charge[atom_mask],
            formal_charge_known=system.formal_charge_known[atom_mask],
            radical_electrons=system.radical_electrons[atom_mask],
            radical_electrons_known=system.radical_electrons_known[atom_mask],
            explicit_hydrogens=system.explicit_hydrogens[atom_mask],
            explicit_hydrogens_known=system.explicit_hydrogens_known[atom_mask],
            implicit_hydrogens=system.implicit_hydrogens[atom_mask],
            implicit_hydrogens_known=system.implicit_hydrogens_known[atom_mask],
            implicit_hydrogens_allowed=system.implicit_hydrogens_allowed[atom_mask],
            implicit_hydrogens_allowed_known=system.implicit_hydrogens_allowed_known[atom_mask],
            atom_aromatic=system.atom_aromatic[atom_mask],
            atom_aromatic_known=system.atom_aromatic_known[atom_mask],
            atom_stereo=atom_stereo,
            reference_coordinates=system.reference_coordinates[atom_mask],
            reference_coordinate_mask=system.reference_coordinate_mask[atom_mask],
            edge_index=atom_to_new[system.edge_index[:, edge_mask]],
            bond_order=system.bond_order[edge_mask],
            bond_order_known=system.bond_order_known[edge_mask],
            bond_aromatic=system.bond_aromatic[edge_mask],
            bond_aromatic_known=system.bond_aromatic_known[edge_mask],
            bond_kind=system.bond_kind[edge_mask],
            bond_direction=system.bond_direction[edge_mask],
            bond_role=system.bond_role[edge_mask],
            bond_stereo=bond_stereo,
            bond_stereo_index=stereo_index,
            batch_size=[],
        ),
        component_to_new,
        atom_to_new,
    )


def _component_rows(
    component_index: Tensor,
    component_count: int,
    name: str,
) -> list[int]:
    if component_index.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if component_index.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    rows = component_index.detach().cpu().tolist()
    if rows and (min(rows) < 0 or max(rows) >= component_count):
        raise IndexError(f"{name} contains an out-of-range component index")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{name} must contain pairwise-distinct component indices")
    return rows


def _splice_components(
    target: MolecularSystem,
    target_components: list[int],
    source: MolecularSystem,
    source_components: list[int],
    target_edge_source_atom: Tensor,
    source_polymer_edge_index: Tensor,
) -> tuple[MolecularSystem, Tensor, Tensor]:
    device = target.atom_symbol.device
    symbols = list(target.symbols)
    symbol_index: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        symbol_index.setdefault(symbol, index)

    def intern(symbol: str | None) -> int:
        if symbol is None:
            return -1
        index = symbol_index.get(symbol)
        if index is None:
            index = len(symbols)
            symbols.append(symbol)
            symbol_index[symbol] = index
        return index

    def source_symbols(values: Tensor) -> Tensor:
        return target.atom_symbol.new_tensor(
            [-1 if int(index) < 0 else intern(source.symbols[int(index)]) for index in values.detach().cpu().tolist()]
        )

    target_component_index = torch.tensor(
        target_components,
        dtype=torch.int64,
        device=device,
    )
    source_component_index = torch.tensor(
        source_components,
        dtype=torch.int64,
        device=device,
    )
    target_component_count = target.component_symbol.numel()
    source_component_count = source.component_symbol.numel()
    target_atom_count = target.atom_symbol.numel()
    source_atom_count = source.atom_symbol.numel()

    source_by_target = torch.full(
        (target_component_count,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    source_by_target[target_component_index] = source_component_index
    selected_target_component = source_by_target >= 0

    target_component_atom_count = target.component_atom_ptr[1:] - target.component_atom_ptr[:-1]
    source_component_atom_count = source.component_atom_ptr[1:] - source.component_atom_ptr[:-1]
    output_component_atom_count = target_component_atom_count.clone()
    output_component_atom_count[target_component_index] = source_component_atom_count[source_component_index]
    component_atom_ptr = torch.cat(
        [
            target.component_atom_ptr.new_zeros(1),
            output_component_atom_count.cumsum(0),
        ]
    )
    output_atom_component = torch.repeat_interleave(
        torch.arange(
            target_component_count,
            dtype=torch.int64,
            device=device,
        ),
        output_component_atom_count,
    )
    output_atom_index = torch.arange(
        output_atom_component.numel(),
        dtype=torch.int64,
        device=device,
    )
    output_atom_local = output_atom_index - component_atom_ptr[output_atom_component]
    output_from_source = selected_target_component[output_atom_component]
    output_source_component = source_by_target[output_atom_component].clamp_min(0)
    new_to_target = target.component_atom_ptr[output_atom_component] + output_atom_local
    new_to_target = torch.where(
        output_from_source,
        torch.full_like(new_to_target, -1),
        new_to_target,
    )
    new_to_source = source.component_atom_ptr[output_source_component] + output_atom_local
    new_to_source = torch.where(
        output_from_source,
        new_to_source,
        torch.full_like(new_to_source, -1),
    )

    target_to_new = torch.full(
        (target_atom_count,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    target_output = new_to_target >= 0
    target_to_new[new_to_target[target_output]] = output_atom_index[target_output]
    source_to_new = torch.full(
        (source_atom_count,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    source_output = new_to_source >= 0
    source_to_new[new_to_source[source_output]] = output_atom_index[source_output]

    target_ptr = target.component_atom_ptr.detach().cpu().tolist()
    source_ptr = source.component_atom_ptr.detach().cpu().tolist()
    target_atom_symbol = target.atom_symbol.detach().cpu().tolist()
    source_atom_symbol = source.atom_symbol.detach().cpu().tolist()
    source_to_new_host = source_to_new.detach().cpu().tolist()
    matched_target_atoms: list[int] = []
    matched_output_atoms: list[int] = []
    for target_component, source_component in zip(
        target_components,
        source_components,
        strict=True,
    ):
        source_atom_by_name = {
            source.symbols[source_atom_symbol[atom]]: source_to_new_host[atom]
            for atom in range(source_ptr[source_component], source_ptr[source_component + 1])
        }
        for atom in range(target_ptr[target_component], target_ptr[target_component + 1]):
            output_atom = source_atom_by_name.get(target.symbols[target_atom_symbol[atom]])
            if output_atom is not None:
                matched_target_atoms.append(atom)
                matched_output_atoms.append(output_atom)
    target_to_new[target.atom_symbol.new_tensor(matched_target_atoms)] = target.atom_symbol.new_tensor(
        matched_output_atoms
    )

    selected_target_index = torch.nonzero(
        selected_target_component,
        as_tuple=False,
    ).flatten()
    selected_source_index = source_by_target[selected_target_index]
    component_symbol = target.component_symbol.clone()
    component_symbol[selected_target_index] = source_symbols(source.component_symbol[selected_source_index])
    component_topology_complete = target.component_topology_complete.clone()
    component_topology_complete[selected_target_index] = source.component_topology_complete[selected_source_index]

    atom_symbol = target.atom_symbol.new_empty(output_atom_index.shape)
    atom_symbol[target_output] = target.atom_symbol[new_to_target[target_output]]
    atom_symbol[source_output] = source_symbols(source.atom_symbol[new_to_source[source_output]])
    output_atom_stereo = target.atom_stereo.new_empty(output_atom_index.shape)
    output_atom_stereo[target_output] = target.atom_stereo[new_to_target[target_output]]
    output_atom_stereo[source_output] = source_symbols(source.atom_stereo[new_to_source[source_output]])

    atom_values: dict[str, Tensor] = {}
    for name in (
        "atomic_number",
        "isotope",
        "isotope_known",
        "formal_charge",
        "formal_charge_known",
        "radical_electrons",
        "radical_electrons_known",
        "explicit_hydrogens",
        "explicit_hydrogens_known",
        "implicit_hydrogens",
        "implicit_hydrogens_known",
        "implicit_hydrogens_allowed",
        "implicit_hydrogens_allowed_known",
        "atom_aromatic",
        "atom_aromatic_known",
        "reference_coordinates",
        "reference_coordinate_mask",
    ):
        target_value = getattr(target, name)
        source_value = getattr(source, name)
        output_value = target_value.new_empty((output_atom_index.numel(), *target_value.shape[1:]))
        output_value[target_output] = target_value[new_to_target[target_output]]
        output_value[source_output] = source_value[new_to_source[source_output]]
        atom_values[name] = output_value

    target_atom_component = torch.repeat_interleave(
        torch.arange(
            target_component_count,
            dtype=torch.int64,
            device=device,
        ),
        target_component_atom_count,
    )
    source_atom_component = torch.repeat_interleave(
        torch.arange(
            source_component_count,
            dtype=torch.int64,
            device=device,
        ),
        source_component_atom_count,
    )

    def canonical_edges(
        edge_index: Tensor,
        direction: Tensor,
        stereo_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        swap = edge_index[1] < edge_index[0]
        return (
            torch.stack(
                [
                    torch.minimum(edge_index[0], edge_index[1]),
                    torch.maximum(edge_index[0], edge_index[1]),
                ]
            ),
            torch.where(swap, -direction, direction),
            torch.where(
                swap.unsqueeze(0),
                stereo_index.flip(0),
                stereo_index,
            ),
        )

    def mapped_stereo(
        stereo_index: Tensor,
        atom_map: Tensor,
        stereo_symbol: Tensor,
    ) -> tuple[Tensor, Tensor]:
        has_any_site = (stereo_index >= 0).any(0)
        has_all_sites = (stereo_index >= 0).all(0)
        remapped = atom_map[stereo_index.clamp_min(0)]
        retained = has_all_sites & (remapped >= 0).all(0)
        remapped = torch.where(
            retained.unsqueeze(0),
            remapped,
            torch.full_like(remapped, -1),
        )
        return (
            remapped,
            torch.where(
                has_any_site & ~retained,
                torch.full_like(stereo_symbol, -1),
                stereo_symbol,
            ),
        )

    target_edge_component = target_atom_component[target.edge_index]
    target_edge_touches_replacement = selected_target_component[target_edge_component].any(0)
    target_edge_dropped_component = (target.bond_role == BOND_ROLE_COMPONENT) & target_edge_touches_replacement
    mapped_target_edge = target_to_new[target.edge_index]
    source_endpoint = target_edge_source_atom >= 0
    mapped_source_endpoint = torch.full_like(target_edge_source_atom, -1)
    mapped_source_endpoint[source_endpoint] = source_to_new[target_edge_source_atom[source_endpoint]]
    mapped_target_edge[source_endpoint] = mapped_source_endpoint[source_endpoint]
    mapped_target_edge[target_edge_source_atom == -1] = -1
    target_edge_survives = (mapped_target_edge >= 0).all(0)
    target_edge_keep = ~target_edge_dropped_component & target_edge_survives
    target_stereo_index, target_bond_stereo = mapped_stereo(
        target.bond_stereo_index,
        target_to_new,
        target.bond_stereo,
    )
    (
        mapped_target_edge,
        target_bond_direction,
        target_stereo_index,
    ) = canonical_edges(
        mapped_target_edge,
        target.bond_direction,
        target_stereo_index,
    )

    mapped_source_edge = source_to_new[source.edge_index]
    source_edge_component = source_atom_component[source.edge_index]
    selected_source_component = torch.zeros(
        source_component_count,
        dtype=torch.bool,
        device=device,
    )
    selected_source_component[source_component_index] = True
    source_edge_keep = (
        (source.bond_role == BOND_ROLE_COMPONENT)
        & selected_source_component[source_edge_component].all(0)
        & (mapped_source_edge >= 0).all(0)
    )
    source_stereo_index, source_bond_stereo = mapped_stereo(
        source.bond_stereo_index,
        source_to_new,
        source.bond_stereo,
    )
    (
        mapped_source_edge,
        source_bond_direction,
        source_stereo_index,
    ) = canonical_edges(
        mapped_source_edge,
        source.bond_direction,
        source_stereo_index,
    )

    mapped_polymer_edge = source_to_new[source_polymer_edge_index]
    polymer_stereo_index = torch.full_like(mapped_polymer_edge, -1)
    (
        mapped_polymer_edge,
        polymer_bond_direction,
        polymer_stereo_index,
    ) = canonical_edges(
        mapped_polymer_edge,
        target.bond_direction.new_full(
            (mapped_polymer_edge.shape[1],),
            BOND_DIRECTION_UNDIRECTED,
        ),
        polymer_stereo_index,
    )

    target_changed_edge = ~target_edge_dropped_component & (target_edge_touches_replacement | ~target_edge_survives)
    output_edge_key = torch.cat(
        [
            mapped_target_edge[:, target_edge_keep],
            mapped_source_edge[:, source_edge_keep],
            mapped_polymer_edge,
        ],
        dim=1,
    )
    key_base = max(output_atom_index.numel(), 1)
    output_edge_key = output_edge_key[0] * key_base + output_edge_key[1]
    source_external = ~source_edge_keep & (mapped_source_edge >= 0).any(0)
    source_external_key = mapped_source_edge[0].clamp_min(0) * key_base + mapped_source_edge[1].clamp_min(0)
    source_external_recreated = (mapped_source_edge >= 0).all(0) & torch.isin(source_external_key, output_edge_key)
    source_changed_edge = source_external & ~source_external_recreated
    changed_stereo_atoms = torch.cat(
        [
            mapped_target_edge[:, target_changed_edge].flatten(),
            mapped_source_edge[:, source_changed_edge].flatten(),
            mapped_polymer_edge.flatten(),
        ]
    )
    changed_stereo_atoms = changed_stereo_atoms[changed_stereo_atoms >= 0]
    output_atom_stereo[changed_stereo_atoms] = -1

    edge_index = torch.cat(
        [
            mapped_target_edge[:, target_edge_keep],
            mapped_source_edge[:, source_edge_keep],
            mapped_polymer_edge,
        ],
        dim=1,
    )
    bond_order = torch.cat(
        [
            target.bond_order[target_edge_keep],
            source.bond_order[source_edge_keep],
            target.bond_order.new_ones(mapped_polymer_edge.shape[1]),
        ]
    )
    bond_order_known = torch.cat(
        [
            target.bond_order_known[target_edge_keep],
            source.bond_order_known[source_edge_keep],
            target.bond_order_known.new_ones(mapped_polymer_edge.shape[1]),
        ]
    )
    bond_aromatic = torch.cat(
        [
            target.bond_aromatic[target_edge_keep],
            source.bond_aromatic[source_edge_keep],
            target.bond_aromatic.new_zeros(mapped_polymer_edge.shape[1]),
        ]
    )
    bond_aromatic_known = torch.cat(
        [
            target.bond_aromatic_known[target_edge_keep],
            source.bond_aromatic_known[source_edge_keep],
            target.bond_aromatic_known.new_ones(mapped_polymer_edge.shape[1]),
        ]
    )
    bond_kind = torch.cat(
        [
            target.bond_kind[target_edge_keep],
            source.bond_kind[source_edge_keep],
            target.bond_kind.new_full(
                (mapped_polymer_edge.shape[1],),
                BOND_KIND_COVALENT,
            ),
        ]
    )
    bond_direction = torch.cat(
        [
            target_bond_direction[target_edge_keep],
            source_bond_direction[source_edge_keep],
            polymer_bond_direction,
        ]
    )
    bond_role = torch.cat(
        [
            target.bond_role[target_edge_keep],
            source.bond_role[source_edge_keep],
            target.bond_role.new_full(
                (mapped_polymer_edge.shape[1],),
                BOND_ROLE_POLYMER,
            ),
        ]
    )
    bond_stereo = torch.cat(
        [
            target_bond_stereo[target_edge_keep],
            source_symbols(source_bond_stereo[source_edge_keep]),
            target.bond_stereo.new_full(
                (mapped_polymer_edge.shape[1],),
                -1,
            ),
        ]
    )
    bond_stereo_index = torch.cat(
        [
            target_stereo_index[:, target_edge_keep],
            source_stereo_index[:, source_edge_keep],
            polymer_stereo_index,
        ],
        dim=1,
    )

    edge_order = torch.argsort(
        edge_index[0] * key_base + edge_index[1],
        stable=True,
    )
    edge_index = edge_index[:, edge_order]
    bond_order = bond_order[edge_order]
    bond_order_known = bond_order_known[edge_order]
    bond_aromatic = bond_aromatic[edge_order]
    bond_aromatic_known = bond_aromatic_known[edge_order]
    bond_kind = bond_kind[edge_order]
    bond_direction = bond_direction[edge_order]
    bond_role = bond_role[edge_order]
    bond_stereo = bond_stereo[edge_order]
    bond_stereo_index = bond_stereo_index[:, edge_order]

    duplicate = (edge_index[:, 1:] == edge_index[:, :-1]).all(0)
    if bool(duplicate.any().item()):
        duplicate_rows = torch.nonzero(duplicate, as_tuple=False).flatten()
        groups: list[tuple[int, int]] = []
        for row in duplicate_rows.detach().cpu().tolist():
            if groups and row == groups[-1][1] - 1:
                groups[-1] = groups[-1][0], row + 2
            else:
                groups.append((row, row + 2))
        edge_keep = torch.ones(
            edge_index.shape[1],
            dtype=torch.bool,
            device=device,
        )
        for start, stop in groups:
            endpoints = edge_index[:, start:stop].T.detach().cpu().tolist()
            orders = bond_order[start:stop].detach().cpu().tolist()
            order_known = bond_order_known[start:stop].detach().cpu().tolist()
            aromatics = bond_aromatic[start:stop].detach().cpu().tolist()
            aromatic_known = bond_aromatic_known[start:stop].detach().cpu().tolist()
            kinds = bond_kind[start:stop].detach().cpu().tolist()
            directions = bond_direction[start:stop].detach().cpu().tolist()
            roles = bond_role[start:stop].detach().cpu().tolist()
            stereos = bond_stereo[start:stop].detach().cpu().tolist()
            stereo_indices = bond_stereo_index[:, start:stop].T.detach().cpu().tolist()
            records = [
                (
                    int(endpoint[0]),
                    int(endpoint[1]),
                    float(order) if order_is_known else None,
                    bool(aromatic) if aromatic_is_known else None,
                    int(kind),
                    int(direction),
                    int(role),
                    None if int(stereo) < 0 else symbols[int(stereo)],
                    int(stereo_index[0]),
                    int(stereo_index[1]),
                )
                for (
                    endpoint,
                    order,
                    order_is_known,
                    aromatic,
                    aromatic_is_known,
                    kind,
                    direction,
                    role,
                    stereo,
                    stereo_index,
                ) in zip(
                    endpoints,
                    orders,
                    order_known,
                    aromatics,
                    aromatic_known,
                    kinds,
                    directions,
                    roles,
                    stereos,
                    stereo_indices,
                    strict=True,
                )
            ]
            merged = records[0]
            for record in records[1:]:
                merged = _merge_bond(merged, record)
            bond_order[start] = 0.0 if merged[2] is None else merged[2]
            bond_order_known[start] = merged[2] is not None
            bond_aromatic[start] = False if merged[3] is None else merged[3]
            bond_aromatic_known[start] = merged[3] is not None
            bond_kind[start] = merged[4]
            bond_direction[start] = merged[5]
            bond_role[start] = merged[6]
            bond_stereo[start] = intern(merged[7])
            bond_stereo_index[:, start] = bond_stereo_index.new_tensor([merged[8], merged[9]])
            edge_keep[start + 1 : stop] = False
        edge_index = edge_index[:, edge_keep]
        bond_order = bond_order[edge_keep]
        bond_order_known = bond_order_known[edge_keep]
        bond_aromatic = bond_aromatic[edge_keep]
        bond_aromatic_known = bond_aromatic_known[edge_keep]
        bond_kind = bond_kind[edge_keep]
        bond_direction = bond_direction[edge_keep]
        bond_role = bond_role[edge_keep]
        bond_stereo = bond_stereo[edge_keep]
        bond_stereo_index = bond_stereo_index[:, edge_keep]

    return (
        MolecularSystem(  # type: ignore[call-arg]
            symbols=tuple(symbols),
            system_entity_ptr=target.system_entity_ptr,
            system_instance_ptr=target.system_instance_ptr,
            entity_symbol=target.entity_symbol,
            entity_type=target.entity_type,
            entity_molecule_type=target.entity_molecule_type,
            instance_symbol=target.instance_symbol,
            instance_entity=target.instance_entity,
            instance_component_ptr=target.instance_component_ptr,
            component_symbol=component_symbol,
            component_topology_complete=component_topology_complete,
            component_atom_ptr=component_atom_ptr,
            atom_symbol=atom_symbol,
            atomic_number=atom_values["atomic_number"],
            isotope=atom_values["isotope"],
            isotope_known=atom_values["isotope_known"],
            formal_charge=atom_values["formal_charge"],
            formal_charge_known=atom_values["formal_charge_known"],
            radical_electrons=atom_values["radical_electrons"],
            radical_electrons_known=atom_values["radical_electrons_known"],
            explicit_hydrogens=atom_values["explicit_hydrogens"],
            explicit_hydrogens_known=atom_values["explicit_hydrogens_known"],
            implicit_hydrogens=atom_values["implicit_hydrogens"],
            implicit_hydrogens_known=atom_values["implicit_hydrogens_known"],
            implicit_hydrogens_allowed=atom_values["implicit_hydrogens_allowed"],
            implicit_hydrogens_allowed_known=atom_values["implicit_hydrogens_allowed_known"],
            atom_aromatic=atom_values["atom_aromatic"],
            atom_aromatic_known=atom_values["atom_aromatic_known"],
            atom_stereo=output_atom_stereo,
            reference_coordinates=atom_values["reference_coordinates"],
            reference_coordinate_mask=atom_values["reference_coordinate_mask"],
            edge_index=edge_index,
            bond_order=bond_order,
            bond_order_known=bond_order_known,
            bond_aromatic=bond_aromatic,
            bond_aromatic_known=bond_aromatic_known,
            bond_kind=bond_kind,
            bond_direction=bond_direction,
            bond_role=bond_role,
            bond_stereo=bond_stereo,
            bond_stereo_index=bond_stereo_index,
            batch_size=[],
        ),
        target_to_new,
        source_to_new,
    )


def replace_components(
    system: MolecularSystem,
    component_index: Tensor,
    replacements: Sequence[ChemicalComponent],
) -> tuple[MolecularSystem, Tensor, Tensor]:
    r"""
    Replace selected component occurrences with construction values.

    The returned atom maps align with the original system and the concatenated raw
    replacement atom order, respectively.
    """

    target_components = _component_rows(
        component_index,
        int(system.component_symbol.numel()),
        "component_index",
    )
    if len(target_components) != len(replacements):
        raise ValueError("component_index and replacements must have the same length")
    if not target_components:
        return (
            system,
            torch.arange(
                system.atom_symbol.numel(),
                dtype=torch.int64,
                device=system.atom_symbol.device,
            ),
            system.atom_symbol.new_empty(0, dtype=torch.int64),
        )

    replacement_system = MolecularSystem.from_entities(
        {"replacement": Molecule(tuple(replacements))},
        {"replacement": "replacement"},
        device=system.atom_symbol.device,
    )
    replacement_by_target = {
        target_component: (replacement_component, replacement)
        for replacement_component, (target_component, replacement) in enumerate(
            zip(target_components, replacements, strict=True)
        )
    }
    replacement_atom_by_name: list[dict[str, int]] = []
    replacement_atom_offset = 0
    for replacement in replacements:
        replacement_atom_by_name.append(
            {atom.atom_id: replacement_atom_offset + atom_index for atom_index, atom in enumerate(replacement.atoms)}
        )
        replacement_atom_offset += len(replacement.atoms)
    target_atom_component = torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    keep = torch.ones_like(replacement_system.atom_symbol, dtype=torch.bool)
    target_edge_source_atom = torch.full_like(system.edge_index, -2)
    target_edge_component = target_atom_component[system.edge_index]
    polymer_edge_index = torch.nonzero(
        system.bond_role == BOND_ROLE_POLYMER,
        as_tuple=False,
    ).flatten()
    polymer_edges = polymer_edge_index.detach().cpu().tolist()
    polymer_components = target_edge_component[:, polymer_edge_index].T.detach().cpu().tolist()
    target_edge_updates: list[tuple[int, int, int]] = []
    leaving_atoms: set[int] = set()
    existing_polymer_pairs: set[tuple[int, int]] = set()
    for edge, edge_components in zip(
        polymer_edges,
        polymer_components,
        strict=True,
    ):
        existing_polymer_pairs.add(tuple(sorted(edge_components)))
        for endpoint, target_component in enumerate(edge_components):
            selected = replacement_by_target.get(target_component)
            if selected is None:
                continue
            replacement_component, replacement = selected
            other_component = edge_components[1 - endpoint]
            attachment = (
                replacement.previous_attachment if other_component < target_component else replacement.next_attachment
            )
            if attachment is None:
                target_edge_updates.append((endpoint, edge, -1))
                continue
            atom_by_name = replacement_atom_by_name[replacement_component]
            target_edge_updates.append((endpoint, edge, atom_by_name[attachment.atom_id]))
            leaving_atoms.update(atom_by_name[atom_id] for atom_id in attachment.leaving_atom_ids)
    if target_edge_updates:
        endpoints, edges, source_atoms = zip(*target_edge_updates, strict=True)
        target_edge_source_atom[
            system.edge_index.new_tensor(endpoints),
            system.edge_index.new_tensor(edges),
        ] = system.edge_index.new_tensor(source_atoms)

    instance_component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    instance_entity = system.instance_entity.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    selected_target = set(target_components)
    raw_polymer_edges: list[tuple[int, int]] = []
    for instance, entity in enumerate(instance_entity):
        if entity_type[entity] == ENTITY_MOLECULE:
            continue
        start = instance_component_ptr[instance]
        stop = instance_component_ptr[instance + 1]
        for left in range(start, stop - 1):
            right = left + 1
            if left not in selected_target or right not in selected_target or (left, right) in existing_polymer_pairs:
                continue
            left_component, left_replacement = replacement_by_target[left]
            right_component, right_replacement = replacement_by_target[right]
            next_attachment = left_replacement.next_attachment
            previous_attachment = right_replacement.previous_attachment
            if next_attachment is None or previous_attachment is None:
                continue
            left_atoms = replacement_atom_by_name[left_component]
            right_atoms = replacement_atom_by_name[right_component]
            raw_polymer_edges.append(
                (
                    left_atoms[next_attachment.atom_id],
                    right_atoms[previous_attachment.atom_id],
                )
            )
            leaving_atoms.update(left_atoms[atom_id] for atom_id in next_attachment.leaving_atom_ids)
            leaving_atoms.update(right_atoms[atom_id] for atom_id in previous_attachment.leaving_atom_ids)

    if not leaving_atoms:
        lowered_replacements = replacement_system
        raw_to_lowered = torch.arange(
            replacement_system.atom_symbol.numel(),
            dtype=torch.int64,
            device=replacement_system.atom_symbol.device,
        )
    else:
        keep[replacement_system.atom_symbol.new_tensor(sorted(leaving_atoms))] = False
        lowered_replacements, raw_to_lowered = _atom_subgraph(
            replacement_system,
            keep,
        )
        attached = target_edge_source_atom >= 0
        target_edge_source_atom[attached] = raw_to_lowered[target_edge_source_atom[attached]]
    if raw_polymer_edges:
        source_polymer_edge_index = raw_to_lowered[replacement_system.edge_index.new_tensor(raw_polymer_edges).T]
    else:
        source_polymer_edge_index = replacement_system.edge_index.new_empty((2, 0))

    replaced, old_to_new, lowered_to_new = _splice_components(
        system,
        target_components,
        lowered_replacements,
        list(range(len(replacements))),
        target_edge_source_atom,
        source_polymer_edge_index,
    )
    replacement_to_new = torch.full_like(raw_to_lowered, -1)
    retained = raw_to_lowered >= 0
    replacement_to_new[retained] = lowered_to_new[raw_to_lowered[retained]]
    return replaced, old_to_new, replacement_to_new


def copy_components(
    target: MolecularSystem,
    target_component_index: Tensor,
    source: MolecularSystem,
    source_component_index: Tensor,
) -> tuple[MolecularSystem, Tensor, Tensor]:
    r"""
    Copy exact realized source component subgraphs into target occurrences.
    """

    target_components = _component_rows(
        target_component_index,
        int(target.component_symbol.numel()),
        "target_component_index",
    )
    source_components = _component_rows(
        source_component_index,
        int(source.component_symbol.numel()),
        "source_component_index",
    )
    if len(target_components) != len(source_components):
        raise ValueError("target_component_index and source_component_index must have the same length")
    if not target_components:
        return (
            target,
            torch.arange(
                target.atom_symbol.numel(),
                dtype=torch.int64,
                device=target.atom_symbol.device,
            ),
            torch.full(
                (source.atom_symbol.numel(),),
                -1,
                dtype=torch.int64,
                device=source.atom_symbol.device,
            ),
        )
    target_component_tensor = torch.tensor(
        target_components,
        dtype=torch.int64,
        device=target.atom_symbol.device,
    )
    source_component_tensor = torch.tensor(
        source_components,
        dtype=torch.int64,
        device=source.atom_symbol.device,
    )
    source_by_target = torch.full(
        (target.component_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=target.atom_symbol.device,
    )
    source_by_target[target_component_tensor] = source_component_tensor
    target_atom_component = torch.repeat_interleave(
        torch.arange(
            target.component_symbol.numel(),
            dtype=torch.int64,
            device=target.atom_symbol.device,
        ),
        target.component_atom_ptr[1:] - target.component_atom_ptr[:-1],
    )
    source_atom_component = torch.repeat_interleave(
        torch.arange(
            source.component_symbol.numel(),
            dtype=torch.int64,
            device=source.atom_symbol.device,
        ),
        source.component_atom_ptr[1:] - source.component_atom_ptr[:-1],
    )
    source_component_instance = torch.repeat_interleave(
        torch.arange(
            source.instance_symbol.numel(),
            dtype=torch.int64,
            device=source.atom_symbol.device,
        ),
        (source.instance_component_ptr[1:] - source.instance_component_ptr[:-1]),
    )
    source_edge_component = source_atom_component[source.edge_index]
    source_edge_instance = source_component_instance[source_edge_component]
    source_polymer = (
        (source.bond_role == BOND_ROLE_POLYMER)
        & (source_edge_instance[0] == source_edge_instance[1])
        & ((source_edge_component[0] - source_edge_component[1]).abs() == 1)
    )
    source_component_a = source_edge_component[0, source_polymer]
    source_component_b = source_edge_component[1, source_polymer]
    source_atom_a = source.edge_index[0, source_polymer]
    source_atom_b = source.edge_index[1, source_polymer]
    a_is_earlier = source_component_a < source_component_b
    earlier_component = torch.where(
        a_is_earlier,
        source_component_a,
        source_component_b,
    )
    later_component = torch.where(
        a_is_earlier,
        source_component_b,
        source_component_a,
    )
    earlier_atom = torch.where(a_is_earlier, source_atom_a, source_atom_b)
    later_atom = torch.where(a_is_earlier, source_atom_b, source_atom_a)
    source_previous_atom = torch.full(
        (source.component_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=source.atom_symbol.device,
    )
    source_next_atom = torch.full_like(source_previous_atom, -1)
    source_next_atom[earlier_component] = earlier_atom
    source_previous_atom[later_component] = later_atom

    target_edge_component = target_atom_component[target.edge_index]
    endpoint_source_component = source_by_target[target_edge_component.clamp_min(0)]
    selected_endpoint = (target.bond_role == BOND_ROLE_POLYMER).unsqueeze(0) & (endpoint_source_component >= 0)
    other_component = target_edge_component.flip(0)
    source_attachment_atom = torch.where(
        other_component < target_edge_component,
        source_previous_atom[endpoint_source_component.clamp_min(0)],
        source_next_atom[endpoint_source_component.clamp_min(0)],
    )
    target_edge_source_atom = torch.full_like(target.edge_index, -2)
    target_edge_source_atom[selected_endpoint] = source_attachment_atom[selected_endpoint]
    return _splice_components(
        target,
        target_components,
        source,
        source_components,
        target_edge_source_atom,
        source.edge_index.new_empty((2, 0)),
    )


def form_bond(
    system: MolecularSystem,
    atom_a: int,
    atom_b: int,
    leaving_a: Tensor,
    leaving_b: Tensor,
    *,
    order: float | None = 1.0,
    aromatic: bool | None = False,
    kind: BondKind = "covalent",
    direction: BondDirection = "undirected",
    stereo: str | None = None,
    stereo_atoms: tuple[int, int] | None = None,
    role: BondRole = "assembly",
) -> tuple[MolecularSystem, Tensor]:
    r"""
    Apply one complete bond-formation graph edit.
    """

    if atom_a < 0 or atom_b < 0:
        raise IndexError("bond atom indices must be non-negative")
    if stereo_atoms is not None and min(stereo_atoms) < 0:
        raise IndexError("bond stereo atom indices must be non-negative")
    if leaving_a.ndim != 1:
        raise ValueError("leaving_a must be one-dimensional")
    if leaving_a.dtype != torch.int64:
        raise TypeError("leaving_a must have dtype torch.int64")
    if leaving_b.ndim != 1:
        raise ValueError("leaving_b must be one-dimensional")
    if leaving_b.dtype != torch.int64:
        raise TypeError("leaving_b must have dtype torch.int64")
    if bool((leaving_a < 0).any().item()) or bool((leaving_b < 0).any().item()):
        raise IndexError("leaving atom indices must be non-negative")
    keep = torch.ones(
        system.atom_symbol.numel(),
        dtype=torch.bool,
        device=system.atom_symbol.device,
    )
    keep[leaving_a] = False
    keep[leaving_b] = False
    if not bool(keep[atom_a].item()) or not bool(keep[atom_b].item()):
        raise ValueError("a bond endpoint cannot also be a leaving atom")
    edited, old_to_new = _atom_subgraph(system, keep)
    atom_a = int(old_to_new[atom_a].item())
    atom_b = int(old_to_new[atom_b].item())
    stereo_a, stereo_b = (
        (-1, -1)
        if stereo_atoms is None
        else (
            int(old_to_new[stereo_atoms[0]].item()),
            int(old_to_new[stereo_atoms[1]].item()),
        )
    )
    if stereo_atoms is not None and (stereo_a < 0 or stereo_b < 0):
        stereo = None
        stereo_a = stereo_b = -1
    record = _canonical_bond(
        atom_a,
        atom_b,
        order,
        aromatic,
        _BOND_KIND[kind],
        _BOND_DIRECTION[direction],
        _BOND_ROLE[role],
        stereo,
        stereo_a,
        stereo_b,
    )
    existing = torch.nonzero(
        (edited.edge_index[0] == record[0]) & (edited.edge_index[1] == record[1]),
        as_tuple=False,
    ).flatten()
    if existing.numel():
        raise ValueError("bond formation would create a duplicate physical edge")

    symbols = edited.symbols
    if stereo is None:
        stereo_symbol = -1
    else:
        try:
            stereo_symbol = symbols.index(stereo)
        except ValueError:
            stereo_symbol = len(symbols)
            symbols = (*symbols, stereo)

    def append(value: Tensor, item: int | float | bool) -> Tensor:
        return torch.cat([value, value.new_tensor([item])])

    edge_index = torch.cat(
        [
            edited.edge_index,
            edited.edge_index.new_tensor([[record[0]], [record[1]]]),
        ],
        dim=1,
    )
    bond_order = append(edited.bond_order, 0.0 if order is None else order)
    bond_order_known = append(edited.bond_order_known, order is not None)
    bond_aromatic = append(
        edited.bond_aromatic,
        False if aromatic is None else aromatic,
    )
    bond_aromatic_known = append(edited.bond_aromatic_known, aromatic is not None)
    bond_kind = append(edited.bond_kind, record[4])
    bond_direction = append(edited.bond_direction, record[5])
    bond_role = append(edited.bond_role, record[6])
    bond_stereo = append(edited.bond_stereo, stereo_symbol)
    bond_stereo_index = torch.cat(
        [
            edited.bond_stereo_index,
            edited.bond_stereo_index.new_tensor([[record[8]], [record[9]]]),
        ],
        dim=1,
    )
    edge_order = torch.argsort(edge_index[0] * edited.atom_symbol.numel() + edge_index[1])
    atom_stereo = edited.atom_stereo.clone()
    atom_stereo[edited.edge_index.new_tensor([record[0], record[1]])] = -1
    return (
        edited.replace(  # type: ignore[attr-defined]
            symbols=symbols,
            atom_stereo=atom_stereo,
            edge_index=edge_index[:, edge_order],
            bond_order=bond_order[edge_order],
            bond_order_known=bond_order_known[edge_order],
            bond_aromatic=bond_aromatic[edge_order],
            bond_aromatic_known=bond_aromatic_known[edge_order],
            bond_kind=bond_kind[edge_order],
            bond_direction=bond_direction[edge_order],
            bond_role=bond_role[edge_order],
            bond_stereo=bond_stereo[edge_order],
            bond_stereo_index=bond_stereo_index[:, edge_order],
        ),
        old_to_new,
    )
