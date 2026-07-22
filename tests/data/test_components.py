from __future__ import annotations

import pytest

from ofoldx.data import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    PolymerAttachment,
    PolymerSequence,
    get_component,
)
from ofoldx.data.components import _BUILTIN_COMPONENTS

# DESIGN.md 3.2.1: the package-audited set is exactly 31 IDs -- the 21 protein components
# (the 20 standard residues plus the wwPDB CCD ``UNK`` placeholder) and 10 nucleic
# components (8 standard CCD graphs plus the ``N`` / ``DN`` generated placeholders).
_BUILTIN_PROTEIN_IDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
    }
)
_BUILTIN_NUCLEIC_IDS = frozenset({"A", "C", "G", "U", "DA", "DC", "DG", "DT", "N", "DN"})
_BUILTIN_IDS = _BUILTIN_PROTEIN_IDS | _BUILTIN_NUCLEIC_IDS


def _undirected_bonds(bonds: frozenset[ChemicalBond]) -> set[tuple[frozenset[str], float, bool]]:
    # Frozen chemistry is undirected; ChemicalBond declaration direction is an open facet,
    # so compare edges without depending on stored atom_a/atom_b order.
    return {(frozenset((bond.atom_a, bond.atom_b)), bond.order, bond.aromatic) for bond in bonds}


# --------------------------------------------------------------------------------------
# 3.2 / 3.2.1 frozen value-shape invariants
# --------------------------------------------------------------------------------------


def test_chemical_atom_requires_id_and_element_and_defaults_the_rest_to_none() -> None:
    # 3.2.1: atom_id and element are always known; every optional chemical fact defaults
    # to None so absence never silently means neutral, zero, non-aromatic, or permitted.
    atom = ChemicalAtom("N", "N")

    assert atom.atom_id == "N"
    assert atom.element == "N"
    assert atom.isotope is None
    assert atom.formal_charge is None
    assert atom.radical_electrons is None
    assert atom.explicit_hydrogens is None
    assert atom.implicit_hydrogens is None
    assert atom.implicit_hydrogens_allowed is None
    assert atom.aromatic is None
    assert atom.stereo is None

    with pytest.raises(TypeError):
        ChemicalAtom("N")  # type: ignore[call-arg]  # element is required


def test_polymer_attachment_carries_endpoint_and_leaving_atoms() -> None:
    default = PolymerAttachment("N")
    assert default.atom_id == "N"
    assert default.leaving_atom_ids == frozenset()

    attachment = PolymerAttachment("N", frozenset({"H2"}))
    assert attachment.atom_id == "N"
    assert attachment.leaving_atom_ids == frozenset({"H2"})


def test_bonds_none_is_open_partial_and_empty_frozenset_is_bondless_complete() -> None:
    # 3.2.1: bonds=None asserts neither complete atom inventory nor topology; frozenset()
    # is the different, positive statement that a complete component has no bonds.
    atoms = (ChemicalAtom("NA", "Na"),)
    partial = ChemicalComponent("X", atoms, None)
    complete = ChemicalComponent("X", atoms, frozenset())

    assert partial.bonds is None
    assert complete.bonds == frozenset()
    assert complete.bonds is not None
    # The distinction is value-carrying: the two components are not equal.
    assert partial != complete


def test_atom_ids_must_be_unique_independent_of_bond_completeness() -> None:
    # 3.2.1: atom-ID closure is independent of the bond field -- checked even when partial.
    with pytest.raises(ValueError):
        ChemicalComponent("X", (ChemicalAtom("A", "C"), ChemicalAtom("A", "C")), None)
    with pytest.raises(ValueError):
        ChemicalComponent("X", (ChemicalAtom("A", "C"), ChemicalAtom("A", "C")), frozenset())


def test_bond_endpoint_and_edge_closure_checked_only_when_bonds_present() -> None:
    atoms = (ChemicalAtom("A", "C"), ChemicalAtom("B", "C"))

    # A partial observation asserts no topology, so no endpoint check runs.
    assert ChemicalComponent("X", atoms, None).bonds is None

    # A declared bond set is checked for endpoint and physical-edge closure.
    with pytest.raises(ValueError):
        ChemicalComponent("X", atoms, frozenset({ChemicalBond("A", "GHOST")}))
    with pytest.raises(ValueError):
        ChemicalComponent("X", atoms, frozenset({ChemicalBond("A", "A")}))
    with pytest.raises(ValueError):
        ChemicalComponent("X", atoms, frozenset({ChemicalBond("A", "B"), ChemicalBond("B", "A")}))


def test_attachment_closure_is_independent_of_bonds() -> None:
    atoms = (ChemicalAtom("A", "C"), ChemicalAtom("B", "C"))

    component = ChemicalComponent(
        "X",
        atoms,
        None,
        previous_attachment=PolymerAttachment("A", frozenset({"B"})),
    )
    assert component.previous_attachment is not None
    assert component.previous_attachment.atom_id == "A"

    # Attachment closure is enforced even when the component is an open partial (bonds=None).
    with pytest.raises(ValueError):
        ChemicalComponent("X", atoms, None, previous_attachment=PolymerAttachment("Z"))
    with pytest.raises(ValueError):
        ChemicalComponent("X", atoms, None, previous_attachment=PolymerAttachment("A", frozenset({"A"})))


def test_reference_coordinates_default_empty_and_align_with_atoms_when_present() -> None:
    # 3.2: reference coordinates align with atom declaration order, one xyz triple or None
    # per atom. (The committed value uses () as the ordinary no-conformer state.)
    atoms = (ChemicalAtom("C", "C"), ChemicalAtom("O", "O"))

    assert ChemicalComponent("LIG", atoms, frozenset()).reference_coordinates == ()

    aligned = ChemicalComponent(
        "LIG",
        atoms,
        frozenset(),
        reference_coordinates=((0.0, 0.0, 0.0), None),
    )
    assert len(aligned.reference_coordinates) == len(aligned.atoms)

    with pytest.raises(ValueError):
        ChemicalComponent("LIG", atoms, frozenset(), reference_coordinates=((0.0, 0.0, 0.0),))


def test_partial_observation_needs_no_ccd_graph_and_filters_by_element() -> None:
    # 3.2.1: released LigandMPNN / NA-MPNN parsers accept arbitrary non-water HETATM
    # observations and consume their atom IDs, elements, and coordinates with no CCD graph.
    observation = ChemicalComponent(
        component_id="LIG",
        atoms=(
            ChemicalAtom("C1", "C"),
            ChemicalAtom("O1", "O"),
            ChemicalAtom("H1", "H"),
        ),
        bonds=None,
        reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), None),
    )

    assert observation.bonds is None
    heavy = {atom.atom_id for atom in observation.atoms if atom.element != "H"}
    assert heavy == {"C1", "O1"}


# --------------------------------------------------------------------------------------
# get_component: the audited-31 self-contained set
# --------------------------------------------------------------------------------------


def test_audited_component_set_is_exactly_thirty_one_ids() -> None:
    assert len(_BUILTIN_PROTEIN_IDS) == 21
    assert len(_BUILTIN_NUCLEIC_IDS) == 10
    assert len(_BUILTIN_IDS) == 31
    assert set(_BUILTIN_COMPONENTS) == _BUILTIN_IDS

    for component_id in _BUILTIN_IDS:
        assert get_component(component_id).component_id == component_id


def test_audited_components_are_self_contained_complete_graphs() -> None:
    for component_id in _BUILTIN_IDS:
        component = get_component(component_id)
        atom_ids = {atom.atom_id for atom in component.atoms}

        # Unique atom IDs and a complete (non-partial) declared graph.
        assert len(atom_ids) == len(component.atoms)
        assert component.bonds is not None

        # Every declared bond endpoint resolves locally -- no external CCD lookup needed.
        for bond in component.bonds:
            assert {bond.atom_a, bond.atom_b} <= atom_ids

        # Both linear-polymer attachments resolve locally.
        for attachment in (component.previous_attachment, component.next_attachment):
            assert attachment is not None
            assert attachment.atom_id in atom_ids
            assert set(attachment.leaving_atom_ids) <= atom_ids

        # Reference coordinates, when present, align with the atom tuple.
        if component.reference_coordinates:
            assert len(component.reference_coordinates) == len(component.atoms)

        # 3.2.1: audited components pass every known zero and boolean explicitly.
        for atom in component.atoms:
            assert atom.element
            assert atom.isotope is None
            assert atom.formal_charge is not None
            assert atom.radical_electrons == 0
            assert atom.explicit_hydrogens == 0
            assert atom.implicit_hydrogens == 0
            assert atom.implicit_hydrogens_allowed is False
            assert atom.aromatic in (True, False)


def test_get_component_returns_the_same_audited_snapshot() -> None:
    alanine = get_component("ALA")
    assert alanine.component_id == "ALA"
    assert get_component("ALA") == alanine


def test_get_component_rejects_non_audited_ids() -> None:
    # OFoldX never guesses a chemical graph from a residue name.
    for unknown in ("MSE", "SEP", "LIG", "HOH", ""):
        with pytest.raises(KeyError):
            get_component(unknown)


# --------------------------------------------------------------------------------------
# audited component chemistry (protein + nucleic)
# --------------------------------------------------------------------------------------


def test_standard_protein_reference_coordinates_are_atom_aligned() -> None:
    sequence = PolymerSequence.from_one_letter("ARNDCQEGHILKMFPSTWYV")

    for component in sequence.components:
        assert len(component.reference_coordinates) == len(component.atoms)
        for atom, coordinate in zip(
            component.atoms,
            component.reference_coordinates,
            strict=True,
        ):
            assert (coordinate is None) == (atom.element == "H" or atom.atom_id == "OXT")

    alanine = get_component("ALA")
    coordinates = dict(
        zip(
            (atom.atom_id for atom in alanine.atoms),
            alanine.reference_coordinates,
            strict=True,
        )
    )
    assert coordinates["N"] == (-0.01003183238208294, -1.2073018550872803, -1.0555061101913452)
    assert get_component("UNK").reference_coordinates == ()


def test_unknown_protein_component_is_the_audited_wwpdb_placeholder() -> None:
    unknown = get_component("UNK")

    assert tuple(atom.atom_id for atom in unknown.atoms) == (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OXT",
        "H",
        "H2",
        "HA",
        "HB1",
        "HB2",
        "HG1",
        "HG2",
        "HG3",
        "HXT",
    )
    assert unknown.previous_attachment is not None
    assert unknown.previous_attachment.atom_id == "N"
    assert unknown.next_attachment is not None
    assert unknown.next_attachment.atom_id == "C"


def test_generated_nucleic_components_are_exact_odesign_placeholders() -> None:
    rna = get_component("N")
    dna = get_component("DN")

    assert tuple(
        (atom.atom_id, atom.element, atom.formal_charge, atom.aromatic, atom.stereo) for atom in rna.atoms
    ) == (
        ("P", "P", 0, False, None),
        ("OP1", "O", 0, False, None),
        ("OP2", "O", 0, False, None),
        ("O5'", "O", 0, False, None),
        ("C5'", "C", 0, False, None),
        ("C4'", "C", 0, False, "R"),
        ("O4'", "O", 0, False, None),
        ("C3'", "C", 0, False, "S"),
        ("O3'", "O", 0, False, None),
        ("C2'", "C", 0, False, "R"),
        ("O2'", "O", 0, False, None),
        ("C1'", "C", 0, False, "R"),
        ("N", "N", 0, False, None),
    )
    assert tuple(atom.atom_id for atom in dna.atoms) == (
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N",
    )
    assert _undirected_bonds(rna.bonds) == _undirected_bonds(
        frozenset(
            {
                ChemicalBond("OP1", "P", 2.0),
                ChemicalBond("OP2", "P"),
                ChemicalBond("O5'", "P"),
                ChemicalBond("C5'", "O5'"),
                ChemicalBond("C4'", "C5'"),
                ChemicalBond("C4'", "O4'"),
                ChemicalBond("C3'", "C4'"),
                ChemicalBond("C1'", "O4'"),
                ChemicalBond("C3'", "O3'"),
                ChemicalBond("C2'", "C3'"),
                ChemicalBond("C2'", "O2'"),
                ChemicalBond("C1'", "C2'"),
                ChemicalBond("C1'", "N"),
            }
        )
    )
    assert _undirected_bonds(dna.bonds) == _undirected_bonds(
        frozenset(bond for bond in rna.bonds if "O2'" not in (bond.atom_a, bond.atom_b))
    )
    for component in (rna, dna):
        assert component.previous_attachment is not None
        assert component.previous_attachment.atom_id == "P"
        assert component.previous_attachment.leaving_atom_ids == frozenset()
        assert component.next_attachment is not None
        assert component.next_attachment.atom_id == "O3'"
        assert component.next_attachment.leaving_atom_ids == frozenset()
        assert len(component.reference_coordinates) == len(component.atoms)
        assert all(coordinate is not None for coordinate in component.reference_coordinates)


def test_standard_nucleic_components_are_complete_released_ccd_graphs() -> None:
    expected_counts = {
        "A": (37, 39),
        "C": (35, 36),
        "G": (38, 40),
        "U": (34, 35),
        "DA": (36, 38),
        "DC": (34, 35),
        "DG": (37, 39),
        "DT": (36, 37),
    }

    for component_id, (atom_count, bond_count) in expected_counts.items():
        component = get_component(component_id)
        assert component.bonds is not None
        assert len(component.atoms) == atom_count
        assert len(component.bonds) == bond_count
        assert component.previous_attachment is not None
        assert component.previous_attachment.atom_id == "P"
        assert component.previous_attachment.leaving_atom_ids == frozenset({"OP3", "HOP3"})
        assert component.next_attachment is not None
        assert component.next_attachment.atom_id == "O3'"
        assert component.next_attachment.leaving_atom_ids == frozenset({"HO3'"})
        for atom, coordinate in zip(component.atoms, component.reference_coordinates, strict=True):
            is_model_atom = atom.element != "H" and atom.atom_id not in {"OP3", "HO3'"}
            assert (coordinate is not None) == is_model_atom

    adenine = get_component("A")
    assert adenine.bonds is not None
    assert all(bond.order == 1.5 for bond in adenine.bonds if bond.aromatic)
    assert dict(
        zip(
            (atom.atom_id for atom in adenine.atoms),
            adenine.reference_coordinates,
            strict=True,
        )
    )[
        "P"
    ] == (1.024, -0.137, -4.723)


# --------------------------------------------------------------------------------------
# SMILES boundary (RDKit only; skipped when RDKit is absent)
# --------------------------------------------------------------------------------------


def test_from_smiles_builds_component_chemistry_with_explicit_zero_facts() -> None:
    pytest.importorskip("rdkit")

    from ofoldx.data import Molecule

    molecule = Molecule.from_smiles("CCO")
    (component,) = molecule.components

    assert isinstance(component, ChemicalComponent)
    assert component.bonds is not None
    for atom in component.atoms:
        # 3.2.1: SMILES importers pass every known zero and boolean explicitly.
        assert atom.formal_charge is not None
        assert atom.radical_electrons is not None
        assert atom.explicit_hydrogens is not None
        assert atom.implicit_hydrogens is not None
        assert isinstance(atom.implicit_hydrogens_allowed, bool)
        assert isinstance(atom.aromatic, bool)
