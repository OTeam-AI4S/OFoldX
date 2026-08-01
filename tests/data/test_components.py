from __future__ import annotations

import pytest

from ofoldx.data import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    PolymerAttachment,
    get_component,
)
from ofoldx.data.components import complete_component, polymer_attachment_anchors

_COMPONENT_IDS = (
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
    "A",
    "C",
    "G",
    "U",
    "I",
    "DA",
    "DC",
    "DG",
    "DT",
    "DI",
    "DU",
    "N",
    "DN",
)


def _undirected_bonds(bonds: frozenset[ChemicalBond]) -> set[tuple[frozenset[str], float, bool]]:
    # Frozen chemistry is undirected; ChemicalBond declaration direction is an open facet,
    # so compare edges without depending on stored atom_a/atom_b order.
    return {(frozenset((bond.atom_a, bond.atom_b)), bond.order, bond.aromatic) for bond in bonds}


@pytest.mark.parametrize(
    ("chem_comp_type", "expected"),
    [
        ("L-PEPTIDE LINKING", ("N", "C")),
        ("D-BETA-PEPTIDE, C-GAMMA LINKING", ("N", "CG")),
        ("L-GAMMA-PEPTIDE, C-DELTA LINKING", ("N", "CD")),
        ("D-PEPTIDE NH3 AMINO TERMINUS", (None, "C")),
        ("L-PEPTIDE COOH CARBOXY TERMINUS", ("N", None)),
        ("RNA OH 5 PRIME TERMINUS", (None, "O3'")),
        ("DNA OH 3 PRIME TERMINUS", ("P", None)),
        ("PEPTIDE-LIKE", (None, None)),
    ],
)
def test_polymer_attachment_anchors(
    chem_comp_type: str,
    expected: tuple[str | None, str | None],
) -> None:
    atom_ids = {"N", "C", "CG", "CD", "P", "O3'"}

    assert polymer_attachment_anchors(chem_comp_type, atom_ids) == expected


def test_complete_component_unstated_chemistry() -> None:
    occurrence = ChemicalComponent(
        "LIG",
        (ChemicalAtom("N1", "X", formal_charge=-1), ChemicalAtom("C1", "C")),
        frozenset({ChemicalBond("N1", "C1", order=2.0, aromatic=False)}),
        non_leaving_atom_ids=frozenset({"N1", "C1"}),
    )
    reference = ChemicalComponent(
        "LIG",
        (ChemicalAtom("N1", "N", formal_charge=1), ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C")),
        frozenset(
            {
                ChemicalBond("N1", "C1", order=1.0, aromatic=False),
                ChemicalBond("C1", "C2", order=1.0, aromatic=False),
            }
        ),
        reference_coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        leaving_atom_ids=frozenset({"C2"}),
        non_leaving_atom_ids=frozenset({"N1", "C1"}),
    )

    completed, topology_complete = complete_component(
        occurrence,
        reference,
        occurrence_topology_complete=False,
    )

    assert (completed.atoms[0].element, completed.atoms[0].formal_charge) == ("N", -1)
    assert {frozenset((bond.atom_a, bond.atom_b)): bond.order for bond in completed.bonds or ()} == {
        frozenset(("N1", "C1")): 2.0,
        frozenset(("C1", "C2")): 1.0,
    }
    assert completed.reference_coordinates == reference.reference_coordinates
    assert completed.leaving_atom_ids == frozenset({"C2"})
    assert topology_complete


def test_complete_component_contextual_attachment() -> None:
    atoms = (ChemicalAtom("N", "N"), ChemicalAtom("H", "H"))
    occurrence = ChemicalComponent(
        "LIG",
        atoms,
        None,
        non_leaving_atom_ids=frozenset({"N", "H"}),
    )
    reference = ChemicalComponent(
        "LIG",
        atoms,
        frozenset({ChemicalBond("N", "H")}),
        previous_attachment=PolymerAttachment("N", frozenset({"H"})),
        leaving_atom_ids=frozenset({"H"}),
        non_leaving_atom_ids=frozenset({"N"}),
    )

    completed, _ = complete_component(occurrence, reference, occurrence_topology_complete=False)

    assert completed.leaving_atom_ids == frozenset()
    assert completed.previous_attachment == reference.previous_attachment


def test_complete_component_source_attachment() -> None:
    atoms = (ChemicalAtom("N", "N"), ChemicalAtom("H", "H"))
    occurrence = ChemicalComponent(
        "LIG",
        atoms,
        None,
        previous_attachment=PolymerAttachment("N", frozenset({"H"})),
    )
    reference = ChemicalComponent(
        "LIG",
        atoms,
        frozenset({ChemicalBond("N", "H")}),
        non_leaving_atom_ids=frozenset({"N", "H"}),
    )

    completed, _ = complete_component(occurrence, reference, occurrence_topology_complete=False)

    assert completed.previous_attachment == occurrence.previous_attachment
    assert completed.leaving_atom_ids == frozenset()
    assert completed.non_leaving_atom_ids == frozenset({"N"})


@pytest.mark.parametrize("component_id", _COMPONENT_IDS)
def test_get_component_graph(component_id: str) -> None:
    component = get_component(component_id)
    atom_ids = {atom.atom_id for atom in component.atoms}

    assert component.component_id == component_id
    assert len(atom_ids) == len(component.atoms)
    assert component.bonds is not None
    assert all({bond.atom_a, bond.atom_b} <= atom_ids for bond in component.bonds)
    for attachment in (component.previous_attachment, component.next_attachment):
        assert attachment is not None
        assert attachment.atom_id in atom_ids
        assert set(attachment.leaving_atom_ids) <= atom_ids
    if component.reference_coordinates:
        assert len(component.reference_coordinates) == len(component.atoms)
    for atom in component.atoms:
        assert atom.element
        assert atom.isotope is None
        assert atom.formal_charge is not None
        assert atom.radical_electrons == 0
        assert atom.explicit_hydrogens == 0
        assert atom.implicit_hydrogens == 0
        assert atom.implicit_hydrogens_allowed is False
        assert atom.aromatic in (True, False)


@pytest.mark.parametrize("component_id", ["MSE", "SEP", "LIG", "HOH", ""])
def test_get_component_unknown(component_id: str) -> None:
    with pytest.raises(KeyError):
        get_component(component_id)


# --------------------------------------------------------------------------------------
# audited component chemistry (protein + nucleic)
# --------------------------------------------------------------------------------------


def test_protein_reference_coordinates() -> None:
    for component_id in _COMPONENT_IDS[:20]:
        component = get_component(component_id)
        assert len(component.reference_coordinates) == len(component.atoms)
        # The reference is the complete wwPDB CCD ideal conformer: every atom -- heavy and
        # hydrogen, including the terminal OXT -- carries an idealized position.
        assert all(coordinate is not None for coordinate in component.reference_coordinates)

    alanine = get_component("ALA")
    coordinates = dict(
        zip(
            (atom.atom_id for atom in alanine.atoms),
            alanine.reference_coordinates,
            strict=True,
        )
    )
    assert coordinates["N"] == (-0.966, 0.493, 1.5)
    # UNK is the unknown-residue placeholder: the checkpoint cannot generate its side chain, so it
    # deliberately carries no reference conformer.
    assert get_component("UNK").reference_coordinates == ()


def test_unknown_protein_placeholder() -> None:
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


def test_generated_nucleic_placeholders() -> None:
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


def test_nucleic_component_graphs() -> None:
    expected_counts = {
        "A": (37, 39),
        "C": (35, 36),
        "G": (38, 40),
        "U": (34, 35),
        "I": (36, 38),
        "DA": (36, 38),
        "DC": (34, 35),
        "DG": (37, 39),
        "DT": (36, 37),
        "DI": (35, 37),
        "DU": (33, 34),
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
        # Complete CCD ideal conformer: every atom -- heavy and hydrogen, including the
        # 5'-terminal OP3 -- has an idealized position.
        assert len(component.reference_coordinates) == len(component.atoms)
        assert all(coordinate is not None for coordinate in component.reference_coordinates)

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
