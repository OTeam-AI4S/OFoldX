r"""Canonical coordinate-free molecular values and exact occurrence references."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

PolymerType: TypeAlias = Literal["protein", "rna", "dna", "dna_rna_hybrid"]
ComponentReference: TypeAlias = tuple[str, int]
BondOrigin: TypeAlias = Literal[
    "component",
    "molecule",
    "polymer_rule",
    "source",
    "user",
]

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
    r"""Resolve one polymer one-letter symbol to a component id under an explicit policy.

    ``policy="strict"`` resolves standard residues and the rare-but-unambiguous
    extensions (protein ``U``->``SEC``/``O``->``PYL``, nucleic inosine, DNA ``U``) and
    raises ``KeyError`` on any other symbol. ``policy="permissive"`` resolves only the
    standard core and returns the polymer's unknown-residue placeholder for everything
    else.
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
# The 20 standard amino-acid component ids as an order-independent membership set, for
# callers that only need "is this a standard amino acid" (e.g. the PDB reader). This is a
# general chemistry fact, distinct from any model's token ordering (e.g. the MPNN vocab).
STANDARD_AMINO_ACID_IDS: frozenset[str] = frozenset(AMINO_ACID_THREE_TO_ONE)
# ``AMINO_ACID_ONE_TO_THREE`` extended with the unknown-residue letter ``X`` -> ``UNK``. Kept
# distinct so callers opt in to ``X``.
AMINO_ACID_ONE_TO_THREE_WITH_UNKNOWN: dict[str, str] = {
    **AMINO_ACID_ONE_TO_THREE,
    "X": "UNK",
}
# One-letter -> CCD-code maps for nucleic polymers: standard core plus the strict
# extensions and permissive unknowns.
DNA_ONE_TO_THREE: dict[str, str] = {
    **_STANDARD_COMPONENTS["dna"],
    **_EXTENDED_COMPONENTS["dna"],
    "X": _UNKNOWN_COMPONENT["dna"],
    "N": _UNKNOWN_COMPONENT["dna"],
}
RNA_ONE_TO_THREE: dict[str, str] = {
    **_STANDARD_COMPONENTS["rna"],
    **_EXTENDED_COMPONENTS["rna"],
    "X": _UNKNOWN_COMPONENT["rna"],
    "N": _UNKNOWN_COMPONENT["rna"],
}

# Heavy leaving atoms trimmed at internal (non-terminal) residues, kept at chain termini
# (verified from the CCD: protein C-terminal ``OXT``, nucleic 5'-terminal ``OP3``). Keyed
# by molecule-type name so structure processing and heavy-atom selection share one table.
LEAVING_ATOMS: dict[str, frozenset[str]] = {
    "protein": frozenset({"OXT"}),
    "rna": frozenset({"OP3"}),
    "dna": frozenset({"OP3"}),
}
# Element symbols treated as hydrogen (protium and deuterium), dropped from heavy-atom sets.
HYDROGEN_ELEMENTS: frozenset[str] = frozenset({"H", "D"})


@dataclass(frozen=True)
class ChemicalAtom:
    r"""One atom in an exact chemical graph."""

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
    r"""One undirected covalent bond in a chemical component."""

    atom_a: str
    atom_b: str
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    stereo_atoms: tuple[str, str] | None = None


@dataclass(frozen=True)
class PolymerAttachment:
    r"""One component-local endpoint for the default linear polymer link."""

    atom_id: str
    leaving_atom_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ChemicalComponent:
    r"""One complete chemical graph or open partial atom observation."""

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
    r"""Return whether one candidate alias declares only water elements."""

    if component.component_id not in _WATER_COMPONENT_ALIASES:
        return False
    elements = {atom.element for atom in component.atoms}
    return "O" in elements and elements <= {"H", "O"}


@dataclass(frozen=True)
class PolymerSequence:
    r"""One fixed-length polymer as ordered chemical component occurrences."""

    polymer_type: PolymerType
    components: tuple[ChemicalComponent, ...]

    @classmethod
    def from_one_letter(
        cls,
        sequence: str,
        *,
        polymer_type: PolymerType = "protein",
    ) -> PolymerSequence:
        r"""Resolve an unambiguous one-letter sequence to built-in components."""

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

    def __len__(self) -> int:
        return len(self.components)


@dataclass(frozen=True)
class MoleculeBond:
    r"""One bond between two component occurrences in a Molecule."""

    component_a: int
    atom_a: str
    component_b: int
    atom_b: str
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    stereo_atoms: tuple[tuple[int, str], tuple[int, str]] | None = None


@dataclass(frozen=True)
class Molecule:
    r"""One ordered non-polymer molecular graph."""

    components: tuple[ChemicalComponent, ...]
    bonds: frozenset[MoleculeBond] = frozenset()

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

    @staticmethod
    def from_smiles(
        smiles: str,
        *,
        atom_ids: tuple[str, ...] | None = None,
    ) -> Molecule:
        r"""Parse the exactly representable SMILES chemistry subset with RDKit."""

        from ._smiles_import import molecule_from_smiles

        return molecule_from_smiles(smiles, atom_ids=atom_ids)


@dataclass(frozen=True)
class AtomSite:
    r"""An exact atom occurrence in a MolecularSystem."""

    instance_id: str
    component_index: int
    atom_id: str


@dataclass(frozen=True)
class SystemBond:
    r"""One assembly-level bond between exact atom occurrences."""

    site_a: AtomSite
    site_b: AtomSite
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    stereo_sites: tuple[AtomSite, AtomSite] | None = None
    origin: BondOrigin = "user"


class MolecularSystem:
    r"""A coordinate-free assembly of molecular entities and their instances."""

    entities: dict[str, PolymerSequence | Molecule]
    instances: dict[str, str]
    bonds: frozenset[SystemBond]

    def __init__(
        self,
        entities: dict[str, PolymerSequence | Molecule],
        instances: dict[str, str],
        bonds: frozenset[SystemBond] = frozenset(),
    ) -> None:
        self.entities = dict(entities)
        self.instances = dict(instances)
        self.bonds = bonds

        if set(self.entities) != set(self.instances.values()):
            raise ValueError("entities must be exactly those named by instances")

        for bond in self.bonds:
            self.atom(bond.site_a)
            self.atom(bond.site_b)
            if bond.stereo_sites is not None:
                stereo_a, stereo_b = bond.stereo_sites
                self.atom(stereo_a)
                self.atom(stereo_b)

    def component(self, reference: ComponentReference) -> ChemicalComponent:
        r"""Resolve an exact component occurrence."""

        instance_id, component_index = reference
        if component_index < 0:
            raise IndexError(f"component index {component_index} is outside instance {instance_id!r}")
        entity_id = self.instances[instance_id]
        return self.entities[entity_id].components[component_index]

    def atom(self, site: AtomSite) -> ChemicalAtom:
        r"""Resolve an exact realized atom occurrence."""

        entity = self.entities[self.instances[site.instance_id]]
        component = self.component((site.instance_id, site.component_index))
        if isinstance(entity, PolymerSequence):
            previous_attachment = component.previous_attachment
            if (
                site.component_index > 0
                and previous_attachment is not None
                and entity.components[site.component_index - 1].next_attachment is not None
                and site.atom_id in previous_attachment.leaving_atom_ids
            ):
                raise KeyError(site)
            next_attachment = component.next_attachment
            if (
                site.component_index + 1 < len(entity)
                and next_attachment is not None
                and entity.components[site.component_index + 1].previous_attachment is not None
                and site.atom_id in next_attachment.leaving_atom_ids
            ):
                raise KeyError(site)
        for atom in component.atoms:
            if atom.atom_id == site.atom_id:
                return atom
        raise KeyError(site)

    def iter_atoms(self) -> Iterator[tuple[AtomSite, ChemicalAtom]]:
        r"""Iterate realized atoms in instance, component, and atom order."""

        for instance_id, entity_id in self.instances.items():
            entity = self.entities[entity_id]
            for component_index, component in enumerate(entity.components):
                previous_leaving: frozenset[str] | None = None
                next_leaving: frozenset[str] | None = None
                if isinstance(entity, PolymerSequence):
                    if (
                        component_index > 0
                        and component.previous_attachment is not None
                        and entity.components[component_index - 1].next_attachment is not None
                    ):
                        previous_leaving = component.previous_attachment.leaving_atom_ids
                    if (
                        component_index + 1 < len(entity)
                        and component.next_attachment is not None
                        and entity.components[component_index + 1].previous_attachment is not None
                    ):
                        next_leaving = component.next_attachment.leaving_atom_ids

                for atom in component.atoms:
                    if previous_leaving is not None and atom.atom_id in previous_leaving:
                        continue
                    if next_leaving is not None and atom.atom_id in next_leaving:
                        continue
                    yield AtomSite(instance_id, component_index, atom.atom_id), atom

    def iter_bonds(self) -> Iterator[SystemBond]:
        r"""Materialize the known covalent graph as one SystemBond per stored covalent edge."""

        realized = {site for site, _ in self.iter_atoms()}
        for instance_id, entity_id in self.instances.items():
            entity = self.entities[entity_id]
            for component_index, component in enumerate(entity.components):
                if component.bonds is None:
                    continue
                for bond in component.bonds:
                    site_a = AtomSite(instance_id, component_index, bond.atom_a)
                    site_b = AtomSite(instance_id, component_index, bond.atom_b)
                    if site_a in realized and site_b in realized:
                        yield SystemBond(
                            site_a,
                            site_b,
                            order=bond.order,
                            aromatic=bond.aromatic,
                            stereo=None,
                            stereo_sites=None,
                            origin="component",
                        )
            if isinstance(entity, Molecule):
                for molecule_bond in entity.bonds:
                    site_a = AtomSite(instance_id, molecule_bond.component_a, molecule_bond.atom_a)
                    site_b = AtomSite(instance_id, molecule_bond.component_b, molecule_bond.atom_b)
                    if site_a in realized and site_b in realized:
                        yield SystemBond(
                            site_a,
                            site_b,
                            order=molecule_bond.order,
                            aromatic=molecule_bond.aromatic,
                            stereo=None,
                            stereo_sites=None,
                            origin="molecule",
                        )
            elif isinstance(entity, PolymerSequence):
                for component_index in range(len(entity.components) - 1):
                    left = entity.components[component_index]
                    right = entity.components[component_index + 1]
                    if left.next_attachment is not None and right.previous_attachment is not None:
                        site_a = AtomSite(instance_id, component_index, left.next_attachment.atom_id)
                        site_b = AtomSite(instance_id, component_index + 1, right.previous_attachment.atom_id)
                        if site_a in realized and site_b in realized:
                            yield SystemBond(
                                site_a,
                                site_b,
                                order=1.0,
                                aromatic=False,
                                stereo=None,
                                stereo_sites=None,
                                origin="polymer_rule",
                            )

        yield from self.bonds

    def replace_components(
        self,
        replacements: Mapping[ComponentReference, ChemicalComponent],
    ) -> MolecularSystem:
        r"""Return a system with selected component occurrences replaced.

        Copies of one entity remain shared when they realize the same component
        sequence and split into distinct entities when they diverge. Stored system
        bonds are retained whenever all of their sites still exist.
        """

        if not replacements:
            return self

        changed_references = frozenset(
            reference
            for reference, replacement in replacements.items()
            if self.component(reference).component_id != replacement.component_id
        )
        instances_by_entity: dict[str, list[str]] = {entity_id: [] for entity_id in self.entities}
        for instance_id, entity_id in self.instances.items():
            instances_by_entity[entity_id].append(instance_id)

        used_ids = set(self.entities)
        entities: dict[str, PolymerSequence | Molecule] = {}
        instance_entities: dict[str, str] = {}
        for entity_id, original in self.entities.items():
            original_signature = tuple(component.component_id for component in original.components)
            groups: dict[tuple[str, ...], tuple[PolymerSequence | Molecule, list[str]]] = {}
            for instance_id in instances_by_entity[entity_id]:
                components = tuple(
                    replacements.get((instance_id, index), component)
                    for index, component in enumerate(original.components)
                )
                if components == original.components:
                    realized = original
                elif isinstance(original, PolymerSequence):
                    realized = PolymerSequence(original.polymer_type, components)
                else:
                    realized = Molecule(components, original.bonds)
                signature = tuple(component.component_id for component in components)
                if signature in groups:
                    groups[signature][1].append(instance_id)
                else:
                    groups[signature] = (realized, [instance_id])

            primary_signature = original_signature if original_signature in groups else next(iter(groups))
            ordered_signatures = (
                primary_signature,
                *(signature for signature in groups if signature != primary_signature),
            )
            for signature in ordered_signatures:
                entity, instance_ids = groups[signature]
                if signature == primary_signature:
                    output_id = entity_id
                else:
                    suffix = 2
                    output_id = f"{entity_id}_{suffix}"
                    while output_id in used_ids:
                        suffix += 1
                        output_id = f"{entity_id}_{suffix}"
                    used_ids.add(output_id)
                entities[output_id] = original if signature == original_signature else entity
                for instance_id in instance_ids:
                    instance_entities[instance_id] = output_id

        instances = {instance_id: instance_entities[instance_id] for instance_id in self.instances}
        candidate = MolecularSystem(entities, instances)
        bonds = self._bonds_after_component_replacement(candidate, changed_references)
        return MolecularSystem(entities, instances, bonds)

    def _bonds_after_component_replacement(
        self,
        replaced: MolecularSystem,
        changed_references: frozenset[ComponentReference],
    ) -> frozenset[SystemBond]:
        r"""Drop a stored bond only when component replacement removed one of its sites."""

        retained = []
        for bond in self.bonds:
            sites = (
                (bond.site_a, bond.site_b)
                if bond.stereo_sites is None
                else (bond.site_a, bond.site_b, *bond.stereo_sites)
            )
            removed_by_mutation = False
            for site in sites:
                try:
                    replaced.atom(site)
                except KeyError:
                    if (site.instance_id, site.component_index) not in changed_references:
                        raise
                    removed_by_mutation = True
            if not removed_by_mutation:
                retained.append(bond)
        return frozenset(retained)
