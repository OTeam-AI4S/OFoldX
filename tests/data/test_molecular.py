from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from ofoldx.data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    SystemBond,
)

# These tests target the frozen §3.4.1 identity surface and the frozen §3.4.2
# retained-atom predicate.  The §3.3 value shape and the §3.4.3 topology
# accessor (``iter_bonds``/``SystemBond``) are REOPENED; the bond tests below
# only assert what ``molecular.py`` actually implements and are flagged
# not-yet-frozen so they can move when §3.4.3 is settled.


def _residue(component_id: str, *, side_chain: str | None = None) -> ChemicalComponent:
    r"""
    A minimal protein-like residue with an N-terminal and C-terminal leaving atom.
    """

    atoms = [
        ChemicalAtom("N", "N"),
        ChemicalAtom("H2", "H"),
        ChemicalAtom("CA", "C"),
        ChemicalAtom("C", "C"),
        ChemicalAtom("OXT", "O"),
    ]
    bonds = {ChemicalBond("N", "CA"), ChemicalBond("CA", "C"), ChemicalBond("C", "OXT")}
    if side_chain is not None:
        atoms.append(ChemicalAtom(side_chain, "C"))
        bonds.add(ChemicalBond("CA", side_chain))
    return ChemicalComponent(
        component_id,
        tuple(atoms),
        frozenset(bonds),
        previous_attachment=PolymerAttachment("N", frozenset({"H2"})),
        next_attachment=PolymerAttachment("C", frozenset({"OXT"})),
    )


def _ligand() -> Molecule:
    return Molecule(
        (
            ChemicalComponent(
                "LIG",
                (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O"), ChemicalAtom("N1", "N")),
                frozenset({ChemicalBond("C1", "O1")}),
            ),
        )
    )


# --------------------------------------------------------------------------- #
# §3.3 value shape: entity ID is not a field                                   #
# --------------------------------------------------------------------------- #


def test_entity_values_carry_no_entity_id_field() -> None:
    sequence = PolymerSequence("protein", (_residue("R0"), _residue("R1")))
    molecule = _ligand()

    assert {field.name for field in fields(PolymerSequence)} == {"polymer_type", "components"}
    assert {field.name for field in fields(Molecule)} == {"components", "bonds"}
    assert not hasattr(sequence, "entity_id")
    assert not hasattr(molecule, "entity_id")

    assert sequence.polymer_type == "protein"
    assert len(sequence) == len(sequence.components) == 2


def test_from_smiles_signature_drops_entity_id() -> None:
    # ``from_smiles`` is a staticmethod taking keyword-only ``atom_ids`` and no
    # ``entity_id`` (§3.3).  The RDKit body is not exercised (no rdkit here).
    assert isinstance(inspect.getattr_static(Molecule, "from_smiles"), staticmethod)
    parameters = inspect.signature(Molecule.from_smiles).parameters
    assert list(parameters) == ["smiles", "atom_ids"]
    assert parameters["atom_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "entity_id" not in parameters


# --------------------------------------------------------------------------- #
# from_one_letter: audited ambiguity resolution and ValueError on non-audited  #
# --------------------------------------------------------------------------- #


def test_from_one_letter_resolves_audited_symbols() -> None:
    protein = PolymerSequence.from_one_letter("AGM")
    assert tuple(component.component_id for component in protein.components) == ("ALA", "GLY", "MET")
    assert protein.polymer_type == "protein"
    assert len(protein) == 3

    rna = PolymerSequence.from_one_letter("ACGU", polymer_type="rna")
    assert tuple(component.component_id for component in rna.components) == ("A", "C", "G", "U")

    dna = PolymerSequence.from_one_letter("ACGT", polymer_type="dna")
    assert tuple(component.component_id for component in dna.components) == ("DA", "DC", "DG", "DT")


def test_from_one_letter_rejects_hybrid_and_non_audited_targets() -> None:
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("A", polymer_type="dna_rna_hybrid")

    # Present in the one-letter map but resolving to a non-audited component ID.
    for symbol in ("U", "O"):  # SEC, PYL are not audited
        with pytest.raises(ValueError):
            PolymerSequence.from_one_letter(symbol)
    with pytest.raises(ValueError):  # rna "I" -> non-audited "I"
        PolymerSequence.from_one_letter("I", polymer_type="rna")
    for symbol in ("U", "I"):  # dna "U"/"I" -> non-audited "DU"/"DI"
        with pytest.raises(ValueError):
            PolymerSequence.from_one_letter(symbol, polymer_type="dna")

    # Symbols absent from the one-letter map.
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("a")  # lowercase
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("B")  # unknown letter


def test_from_one_letter_strict_policy_rejects_ambiguity_symbols() -> None:
    # from_one_letter uses the strict component policy; ambiguity symbols require
    # the separate permissive parser path rather than becoming canonical components.
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("X")  # protein
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("N", polymer_type="rna")
    with pytest.raises(ValueError):
        PolymerSequence.from_one_letter("N", polymer_type="dna")


# --------------------------------------------------------------------------- #
# ChemicalComponent / ChemicalBond referential closure                         #
# --------------------------------------------------------------------------- #


def test_chemical_component_validates_referential_closure() -> None:
    atoms = (ChemicalAtom("C", "C"), ChemicalAtom("O", "O"))

    # A partial component carries bonds=None and skips edge validation (§3.2.1).
    partial = ChemicalComponent("PART", atoms, None)
    assert partial.bonds is None

    with pytest.raises(ValueError):  # duplicate atom IDs
        ChemicalComponent("DUP", (ChemicalAtom("C", "C"), ChemicalAtom("C", "N")), frozenset())
    with pytest.raises(ValueError):  # self-loop edge
        ChemicalComponent("LOOP", (ChemicalAtom("C", "C"),), frozenset({ChemicalBond("C", "C")}))
    with pytest.raises(ValueError):  # bond references an unknown atom
        ChemicalComponent("MISS", atoms, frozenset({ChemicalBond("C", "missing")}))
    with pytest.raises(ValueError):  # both directions are one duplicate physical edge
        ChemicalComponent("DUPE", atoms, frozenset({ChemicalBond("C", "O"), ChemicalBond("O", "C")}))
    with pytest.raises(ValueError):  # attachment references an unknown atom
        ChemicalComponent("ATT", atoms, frozenset(), previous_attachment=PolymerAttachment("Q"))
    with pytest.raises(ValueError):  # retained endpoint cannot also be a leaving atom
        ChemicalComponent("ATT", atoms, frozenset(), previous_attachment=PolymerAttachment("C", frozenset({"C"})))
    with pytest.raises(ValueError):  # reference coordinates must align with atoms
        ChemicalComponent("REF", atoms, frozenset(), reference_coordinates=((0.0, 0.0, 0.0),))


def test_chemical_bond_is_a_directional_value() -> None:
    # ChemicalBond is a plain frozen value; there is no canonical undirected
    # equality (the codec reorients on serialize, not the value itself).
    assert ChemicalBond("C", "O") != ChemicalBond("O", "C")
    stereo = ChemicalBond("C1", "C2", order=2.0, stereo="E", stereo_atoms=("F1", "F2"))
    assert stereo.stereo == "E"
    assert stereo.stereo_atoms == ("F1", "F2")
    assert ChemicalBond("a", "b").order == 1.0
    assert ChemicalBond("a", "b").aromatic is False


# --------------------------------------------------------------------------- #
# Molecule / MoleculeBond cross-component topology                             #
# --------------------------------------------------------------------------- #


def test_molecule_validates_cross_component_bonds() -> None:
    c0 = ChemicalComponent("A0", (ChemicalAtom("X", "C"),), frozenset())
    c1 = ChemicalComponent("A1", (ChemicalAtom("Y", "C"),), frozenset())

    bond = MoleculeBond(0, "X", 1, "Y")
    assert (bond.component_a, bond.atom_a, bond.component_b, bond.atom_b) == (0, "X", 1, "Y")
    assert bond.order == 1.0 and bond.aromatic is False
    assert bond.stereo is None and bond.stereo_atoms is None

    molecule = Molecule((c0, c1), frozenset({bond}))
    assert bond in molecule.bonds
    assert Molecule((c0, c1)).bonds == frozenset()

    with pytest.raises(ValueError):  # duplicate physical edge across occurrences
        Molecule((c0, c1), frozenset({MoleculeBond(0, "X", 1, "Y"), MoleculeBond(0, "X", 1, "Y", order=2.0)}))
    with pytest.raises(ValueError):  # endpoint atom does not resolve
        Molecule((c0, c1), frozenset({MoleculeBond(0, "X", 1, "missing")}))

    multi = ChemicalComponent("M", (ChemicalAtom("X", "C"), ChemicalAtom("Z", "C")), frozenset())
    with pytest.raises(ValueError):  # must connect two distinct component occurrences
        Molecule((multi,), frozenset({MoleculeBond(0, "X", 0, "Z")}))


def test_molecule_bond_is_a_directional_value() -> None:
    forward = MoleculeBond(0, "X", 1, "Y")
    reverse = MoleculeBond(1, "Y", 0, "X")
    assert forward == MoleculeBond(0, "X", 1, "Y")
    assert forward != reverse  # no canonical reorientation in the value


# --------------------------------------------------------------------------- #
# §3.4.1 MolecularSystem construction, invariant, and identity equality        #
# --------------------------------------------------------------------------- #


def test_molecular_system_enforces_entity_instance_invariant() -> None:
    protein = PolymerSequence("protein", (_residue("R0"), _residue("R1")))

    system = MolecularSystem({"prot": protein}, {"chain_a": "prot", "chain_b": "prot"})
    assert set(system.entities) == {"prot"}
    assert system.instances == {"chain_a": "prot", "chain_b": "prot"}

    assert MolecularSystem({}, {}).entities == {}  # an empty system is valid

    with pytest.raises(ValueError):  # instance names an absent entity
        MolecularSystem({"prot": protein}, {"chain": "other"})
    with pytest.raises(ValueError):  # entity has no instance
        MolecularSystem({"prot": protein, "extra": protein}, {"chain": "prot"})


def test_molecular_system_has_object_identity_equality_and_shallow_copies() -> None:
    protein = PolymerSequence("protein", (_residue("R0"),))
    entities = {"prot": protein}
    instances = {"chain": "prot"}

    system = MolecularSystem(entities, instances)
    twin = MolecularSystem(entities, instances)

    assert system == system  # object identity, not value equality
    assert system != twin
    assert len({system, twin}) == 2

    # The constructor shallow-copies the two outer dictionaries.
    assert system.entities is not entities
    assert system.instances is not instances
    entities["late"] = protein
    instances["late"] = "late"
    assert "late" not in system.entities
    assert "late" not in system.instances


def test_component_resolution_rejects_negative_and_out_of_range() -> None:
    protein = PolymerSequence("protein", (_residue("R0"), _residue("R1")))
    ligand = _ligand()
    system = MolecularSystem({"prot": protein, "lig": ligand}, {"chain": "prot", "l": "lig"})

    assert system.component(("chain", 0)) is protein.components[0]
    assert system.component(("chain", 1)) is protein.components[1]
    assert system.component(("l", 0)) is ligand.components[0]

    with pytest.raises(IndexError):  # negative indexing is not canonical identity
        system.component(("chain", -1))
    with pytest.raises(IndexError):  # positive out-of-range
        system.component(("chain", 5))
    with pytest.raises(KeyError):  # unknown instance
        system.component(("missing", 0))


def test_replace_components_splits_divergent_entities_and_retains_resolvable_bonds() -> None:
    protein = PolymerSequence("protein", (_residue("R0", side_chain="SG"), _residue("R1")))
    ligand = _ligand()
    removed = SystemBond(AtomSite("chain_a", 0, "SG"), AtomSite("ligand", 0, "C1"))
    retained = SystemBond(AtomSite("chain_b", 0, "SG"), AtomSite("ligand", 0, "O1"))
    system = MolecularSystem(
        {"protein": protein, "ligand": ligand},
        {"chain_a": "protein", "chain_b": "protein", "ligand": "ligand"},
        frozenset({removed, retained}),
    )

    replaced = system.replace_components({("chain_a", 0): _residue("NEW")})

    assert replaced.instances == {
        "chain_a": "protein_2",
        "chain_b": "protein",
        "ligand": "ligand",
    }
    assert replaced.entities["protein"] is protein
    assert replaced.component(("chain_a", 0)).component_id == "NEW"
    assert replaced.component(("chain_b", 0)).component_id == "R0"
    assert replaced.bonds == frozenset({retained})

    shared = system.replace_components(
        {
            ("chain_a", 0): _residue("NEW"),
            ("chain_b", 0): _residue("NEW"),
        }
    )
    assert shared.instances["chain_a"] == shared.instances["chain_b"] == "protein"
    assert "protein_2" not in shared.entities
    assert system.replace_components({}) is system


# --------------------------------------------------------------------------- #
# §3.4.2 (FROZEN) paired-adjacency retained-atom predicate                     #
# --------------------------------------------------------------------------- #


def test_retained_atom_predicate_hides_leaving_atoms_only_when_paired() -> None:
    protein = PolymerSequence("protein", (_residue("R0", side_chain="SG"), _residue("R1")))
    system = MolecularSystem({"prot": protein}, {"chain": "prot"})

    realized = [(site.component_index, site.atom_id) for site, _ in system.iter_atoms()]
    # Active pair (left has next_attachment, right has previous_attachment):
    # left OXT and right H2 are hidden; both endpoints (C, N) and the terminal
    # leaving atoms (left H2, right OXT) remain.
    assert realized == [
        (0, "N"),
        (0, "H2"),
        (0, "CA"),
        (0, "C"),
        (0, "SG"),
        (1, "N"),
        (1, "CA"),
        (1, "C"),
        (1, "OXT"),
    ]
    for site, atom in system.iter_atoms():
        assert atom.atom_id == site.atom_id

    # atom() applies the same predicate.
    assert system.atom(AtomSite("chain", 0, "N")).element == "N"
    assert system.atom(AtomSite("chain", 1, "OXT")).element == "O"  # terminal, retained
    assert system.atom(AtomSite("chain", 0, "H2")).element == "H"  # terminal, retained
    with pytest.raises(KeyError):  # left OXT hidden by the active pair
        system.atom(AtomSite("chain", 0, "OXT"))
    with pytest.raises(KeyError):  # right H2 hidden by the active pair
        system.atom(AtomSite("chain", 1, "H2"))
    with pytest.raises(KeyError):  # unknown atom id
        system.atom(AtomSite("chain", 0, "ZZ"))
    with pytest.raises(KeyError):  # unknown instance
        system.atom(AtomSite("missing", 0, "N"))
    with pytest.raises(IndexError):  # component index out of range
        system.atom(AtomSite("chain", 9, "N"))


def test_retained_atom_predicate_keeps_terminal_and_molecule_atoms() -> None:
    # A single-component polymer has no active adjacency, so every leaving atom
    # (H2, OXT) survives.
    monomer = PolymerSequence("protein", (_residue("R0"),))
    system = MolecularSystem({"prot": monomer}, {"chain": "prot"})
    monomer_atoms = {site.atom_id for site, _ in system.iter_atoms()}
    assert monomer_atoms == {"N", "H2", "CA", "C", "OXT"}

    # The predicate applies only to PolymerSequence adjacency, never to Molecule.
    molecule_system = MolecularSystem({"lig": _ligand()}, {"l": "lig"})
    molecule_atoms = {site.atom_id for site, _ in molecule_system.iter_atoms()}
    assert molecule_atoms == {"C1", "O1", "N1"}


# --------------------------------------------------------------------------- #
# §3.4.3 (REOPENED, not-yet-frozen) topology materialization via iter_bonds()  #
# --------------------------------------------------------------------------- #


def test_iter_bonds_emits_component_and_polymer_rule_edges() -> None:
    # NOTE: §3.4.3 is REOPENED (no cross-layer dedup, no intrinsic-stereo
    # lifting).  This asserts only the committed single-edge-per-owner-layer
    # accessor behavior.
    protein = PolymerSequence("protein", (_residue("R0", side_chain="SG"), _residue("R1")))
    system = MolecularSystem({"prot": protein}, {"chain": "prot"})

    bonds = tuple(system.iter_bonds())
    origins = {bond.origin for bond in bonds}
    assert origins == {"component", "polymer_rule"}
    assert sum(bond.origin == "polymer_rule" for bond in bonds) == 1

    # The default linear-adjacency edge joins left C to right N with order 1.0.
    assert SystemBond(AtomSite("chain", 0, "C"), AtomSite("chain", 1, "N"), origin="polymer_rule") in set(bonds)
    # An intrinsic component edge that survives realization is emitted.
    assert SystemBond(AtomSite("chain", 0, "CA"), AtomSite("chain", 0, "SG"), origin="component") in set(bonds)
    # An intrinsic edge incident to a hidden leaving atom is not materialized.
    hidden = AtomSite("chain", 0, "OXT")
    assert all(bond.site_a != hidden and bond.site_b != hidden for bond in bonds)
    # Intrinsic edges lift no stereo (§3.4.3: consumer reads the owner declaration).
    for bond in bonds:
        assert bond.stereo is None and bond.stereo_sites is None


def test_iter_bonds_materializes_molecule_edges_once_per_instance() -> None:
    nag = ChemicalComponent(
        "NAG",
        (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        frozenset({ChemicalBond("C1", "O1")}),
    )
    molecule = Molecule((nag, nag), frozenset({MoleculeBond(0, "C1", 1, "O1")}))
    system = MolecularSystem({"glycan": molecule}, {"glycan_1": "glycan", "glycan_2": "glycan"})

    molecule_edges = {bond for bond in system.iter_bonds() if bond.origin == "molecule"}
    assert molecule_edges == {
        SystemBond(AtomSite("glycan_1", 0, "C1"), AtomSite("glycan_1", 1, "O1"), origin="molecule"),
        SystemBond(AtomSite("glycan_2", 0, "C1"), AtomSite("glycan_2", 1, "O1"), origin="molecule"),
    }


def test_iter_bonds_passes_stored_overlays_through_unchanged() -> None:
    nag = ChemicalComponent(
        "NAG",
        (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        frozenset({ChemicalBond("C1", "O1")}),
    )
    molecule = Molecule((nag, nag))
    overlay = SystemBond(AtomSite("g1", 0, "C1"), AtomSite("g2", 0, "C1"))
    assert overlay.origin == "user"  # default overlay origin

    system = MolecularSystem(
        {"glycan": molecule},
        {"g1": "glycan", "g2": "glycan"},
        frozenset({overlay}),
    )
    assert system.bonds == frozenset({overlay})
    assert overlay in set(system.iter_bonds())


def test_system_bond_is_a_directional_value() -> None:
    # REOPENED (§3.4.3): SystemBond is a plain frozen value with no custom
    # undirected equality; the codec reorients on serialize, not the value.
    a = AtomSite("chain", 0, "SG")
    b = AtomSite("lig", 0, "C1")
    assert SystemBond(a, b) == SystemBond(a, b)
    assert SystemBond(a, b) != SystemBond(b, a)
    default = SystemBond(a, b)
    assert default.order == 1.0
    assert default.aromatic is False
    assert default.stereo is None
    assert default.stereo_sites is None
    assert default.origin == "user"
