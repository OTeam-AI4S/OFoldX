from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalBond, ChemicalComponent, Molecule
from ofoldx.io import LossReport, read_structure, read_structures, write_structure
from tests.data.test_codec import _heterogeneous_structure

_ALTLOC_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 10.00           O
ATOM      5  CB AALA A   1       1.000   1.000   0.000  0.30 10.00           C
ATOM      6  CB BALA A   1       1.000  -1.000   0.000  0.70 10.00           C
END
"""


@pytest.mark.xfail(
    reason="reopened §3.5: the _ofoldx_structure mmCIF exchange payload and source-exact "
    "round-trip are the reopened §3.5 exchange-writing contract; §3.5.1 explicitly does NOT "
    "freeze encode_structure/decode_structure or any exchange writer (the codec depends on an "
    "unfrozen system ordering contract), so this runs ahead of the frozen carrier boundary",
    strict=False,
)
def test_mmcif_semantic_round_trip_preserves_complete_canonical_value(tmp_path: Path) -> None:
    # NOTE(§3.5.1): encode_structure/decode_structure and this mmCIF exchange writer are NOT
    # part of the frozen six-field carrier -- §3.5.1 marks the codec as not-yet-frozen (ahead
    # of the frozen boundary), so this round-trip exercises reopened §3.5 exchange machinery.
    structure = _heterogeneous_structure()
    path = tmp_path / "result.cif"

    assert write_structure(structure, path) is None
    restored = read_structure(path)

    # §3.5.1 gave Structure object-identity equality, so the round-trip is checked on the
    # decoded system/site content rather than the removed value-equality/private _source bag.
    ligand = restored.system.entities["ligand"]
    assert type(ligand) is Molecule
    assert tuple(component.component_id for component in ligand.components) == ("LFT", "RGT")
    assert restored.sites == structure.sites
    text = path.read_text(encoding="utf-8")
    assert "_atom_site.ofoldx_component_index" in text
    assert "ofoldx_monomer_index" not in text
    assert next(iter(restored.system.bonds)).origin == "source"
    report = write_structure(structure, tmp_path / "acknowledged.mmcif", allow_lossy=True)
    assert report == LossReport()


@pytest.mark.xfail(
    reason="reopened §3.5: altloc selection policy and multi-altloc plural expansion; "
    "§3.5.1 removed the altloc_ids carrier field (one altloc realization is one Structure)",
    strict=False,
)
def test_pdb_altloc_policy_and_plural_expansion_are_residue_consistent(tmp_path: Path) -> None:
    path = tmp_path / "altloc.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    # §3.5.1 removed the altloc_ids carrier field: one altloc realization is one Structure and
    # altloc identity stays reopened §3.5. Only the residue-consistent plural expansion count and
    # a consistent per-selection site axis remain observable on the frozen carrier.
    highest = read_structure(path)
    first = read_structure(path, altloc="first")
    expanded = read_structures(path)

    assert len(expanded) == 2
    assert len(highest.sites) == len(first.sites)


@pytest.mark.xfail(
    reason="reopened §3.5: the atom_layout site-materialization knob (complete_heavy); "
    "§3.5.1 removed the atom_layout carrier field and the system.sites() accessor",
    strict=False,
)
def test_complete_heavy_materializes_declared_sites_with_missing_masks(tmp_path: Path) -> None:
    path = tmp_path / "alanine.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    structure = read_structure(path, atom_layout="complete_heavy")

    # §3.5.1/§3.5.2 removed the atom_layout carrier field and the system.sites() accessor;
    # heavy-atom realization is a reopened §3.5 read-time element filter. Only the materialized
    # site axis and its partially-false mask remain observable on the frozen carrier.
    assert int(structure.coordinate_mask.sum()) == 5
    assert len(structure.sites) > 5


def test_unknown_component_requires_an_explicit_complete_graph(tmp_path: Path) -> None:
    path = tmp_path / "ligand.pdb"
    path.write_text(
        "HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "HETATM    2  O1  LIG L   1       1.200   0.000   0.000  1.00 10.00           O  \n"
        "CONECT    1    2\nEND\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="unknown component 'LIG'"):
        read_structure(path)

    ligand = ChemicalComponent(
        "LIG",
        atoms=(ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
    )
    structure = read_structure(path, components={"LIG": ligand})
    assert len(structure.system.entities) == 1
    assert structure.system.atom(structure.sites[0]).element == "C"


def test_pdb_element_symbols_are_normalized_before_graph_comparison(tmp_path: Path) -> None:
    path = tmp_path / "elements.pdb"
    path.write_text(
        "HETATM    1 ZN   ZN  Z   1       0.000   0.000   0.000  1.00 10.00          ZN  \n"
        "HETATM    2 CL   CL  C   2       2.000   0.000   0.000  1.00 10.00          CL  \n"
        "END\n",
        encoding="ascii",
    )
    components = {
        "ZN": ChemicalComponent("ZN", atoms=(ChemicalAtom("ZN", "Zn"),), bonds=frozenset()),
        "CL": ChemicalComponent("CL", atoms=(ChemicalAtom("CL", "Cl"),), bonds=frozenset()),
    }

    structure = read_structure(path, components=components)

    assert tuple(structure.system.atom(site).element for site in structure.sites) == ("Zn", "Cl")


def test_pdb_keeps_separate_het_residues_as_separate_single_component_molecules(tmp_path: Path) -> None:
    path = tmp_path / "separate-ligands.pdb"
    path.write_text(
        "HETATM    1  C1  LFT L   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "HETATM    2  C1  RGT L   2       2.000   0.000   0.000  1.00 10.00           C  \n"
        "END\n",
        encoding="ascii",
    )
    components = {
        "LFT": ChemicalComponent("LFT", atoms=(ChemicalAtom("C1", "C"),), bonds=frozenset()),
        "RGT": ChemicalComponent("RGT", atoms=(ChemicalAtom("C1", "C"),), bonds=frozenset()),
    }

    structure = read_structure(path, components=components)

    assert len(structure.system.entities) == 2
    assert len(structure.system.instances) == 2
    assert all(
        type(entity) is Molecule and len(entity.components) == 1 for entity in structure.system.entities.values()
    )
    assert {site.component_index for site in structure.sites} == {0}


@pytest.mark.xfail(
    reason="reopened §3.5: the LossReport contract (dropped/rewritten field vocabulary and the "
    "removed format field) is the reopened §3.5 lossy-write reporting decision",
    strict=False,
)
def test_pdb_fails_by_default_and_returns_exact_loss_report_when_acknowledged(tmp_path: Path) -> None:
    structure = _heterogeneous_structure()
    path = tmp_path / "lossy.pdb"

    with pytest.raises(ValueError, match="allow_lossy=True"):
        write_structure(structure, path)
    assert not path.exists()

    report = write_structure(structure, path, allow_lossy=True)
    assert report is not None
    # §3.5.1: LossReport.format was removed; the report exposes only dropped/rewritten field
    # vocabularies, whose exact contents stay the reopened §3.5 lossy-write reporting decision.
    assert "system.nonpolymer_graph" in report.dropped_fields
    assert "system.bond_origin" in report.rewritten_fields
    assert path.read_text(encoding="ascii").endswith("END\n")


def test_source_duplicate_atoms_fail_before_canonical_frozensets(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.pdb"
    line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  \n"
    path.write_text(line + line + "END\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate source atom declaration"):
        read_structure(path)


def test_external_mmcif_struct_conn_becomes_source_origin_overlay(tmp_path: Path) -> None:
    path = tmp_path / "external.cif"
    path.write_text(
        """data_external
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N N  . ALA A 1 1 ? 0 0 0 1 10 0 1 ALA X N  1
ATOM   2 C CA . ALA A 1 1 ? 1 0 0 1 10 0 1 ALA X CA 1
HETATM 3 C C1 . LIG L 2 . ? 2 0 0 1 10 0 7 LIG Z C1 1
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.pdbx_value_order
1 covale A 1 ALA CA L . LIG C1 X 1 ALA CA Z 7 LIG C1 sing
""",
        encoding="utf-8",
    )
    ligand = ChemicalComponent("LIG", atoms=(ChemicalAtom("C1", "C"),), bonds=frozenset())

    structure = read_structure(path, components={"LIG": ligand})

    assert len(structure.system.bonds) == 1
    bond = next(iter(structure.system.bonds))
    assert bond.origin == "source"
    assert {bond.site_a.atom_id, bond.site_b.atom_id} == {"CA", "C1"}


@pytest.mark.xfail(
    reason="reopened §3.5: source-evidence nonpolymer grouping (which HETATM occurrences share "
    "one Molecule entity and their component order) is the reopened §3.5 classification decision",
    strict=False,
)
def test_external_mmcif_groups_ordered_nonpolymer_positions_by_entity_and_asym(tmp_path: Path) -> None:
    path = tmp_path / "multi-component-ligand.cif"
    path.write_text(
        """data_ligand
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
HETATM 1 N N1 . RGT L 7 1 ? 1 0 0 1 10 0 11 RGT Z N1 1
HETATM 2 O O1 . RGT L 7 1 ? 2 0 0 1 10 0 11 RGT Z O1 1
HETATM 3 C C1 . LFT L 7 2 ? 3 0 0 1 10 0 12 LFT Z C1 1
HETATM 4 N N1 . LFT L 7 2 ? 4 0 0 1 10 0 12 LFT Z N1 1
""",
        encoding="utf-8",
    )
    components = {
        "LFT": ChemicalComponent(
            "LFT",
            atoms=(ChemicalAtom("C1", "C"), ChemicalAtom("N1", "N")),
            bonds=frozenset({ChemicalBond("C1", "N1")}),
        ),
        "RGT": ChemicalComponent(
            "RGT",
            atoms=(ChemicalAtom("N1", "N"), ChemicalAtom("O1", "O")),
            bonds=frozenset({ChemicalBond("N1", "O1")}),
        ),
    }

    structure = read_structure(path, components=components)

    assert len(structure.system.entities) == 1
    assert len(structure.system.instances) == 1
    molecule = next(iter(structure.system.entities.values()))
    assert type(molecule) is Molecule
    assert tuple(component.component_id for component in molecule.components) == ("RGT", "LFT")
    assert structure.sites == (
        AtomSite("L", 0, "N1"),
        AtomSite("L", 0, "O1"),
        AtomSite("L", 1, "C1"),
        AtomSite("L", 1, "N1"),
    )


def test_pdb_assembly_materializes_distinct_instances_of_one_entity(tmp_path: Path) -> None:
    path = tmp_path / "assembly.pdb"
    path.write_text(
        """REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000
REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
END
""",
        encoding="ascii",
    )

    structure = read_structure(path, assembly_id="1")

    assert tuple(structure.system.instances) == ("A#0", "A#1")
    assert len(structure.system.entities) == 1
    assert structure.coordinates[:, 0].tolist() == [0.0, 10.0]


@pytest.mark.xfail(
    reason="reopened §3.5: complete_all hydrogen-count materialization is a reopened §3.5 I/O "
    "graph transform; §3.5.1/§3.3 also removed component provenance",
    strict=False,
)
def test_complete_all_materializes_hydrogen_counts_as_an_explicit_graph_transform(tmp_path: Path) -> None:
    path = tmp_path / "methane.pdb"
    path.write_text(
        "HETATM    1  C1  MET L   1       0.000   0.000   0.000  1.00 10.00           C  \nEND\n",
        encoding="ascii",
    )
    component = ChemicalComponent(
        "MET",
        atoms=(
            ChemicalAtom("C1", "C", implicit_hydrogens=3),
            ChemicalAtom("O1", "O", implicit_hydrogens=1),
        ),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
    )

    structure = read_structure(path, atom_layout="complete_all", components={"MET": component})

    assert len(structure.sites) == 6
    assert int(structure.coordinate_mask.sum()) == 1
    molecule = next(iter(structure.system.entities.values()))
    assert type(molecule) is Molecule
    materialized = molecule.components[0]
    assert all(atom.explicit_hydrogens == atom.implicit_hydrogens == 0 for atom in materialized.atoms)
    # §3.3/§3.5.1 removed ChemicalComponent.provenance; the materialization transform record is
    # reopened, so only the resulting hydrogen graph and reference coordinates remain observable.
    reference_by_atom = dict(zip((atom.atom_id for atom in materialized.atoms), materialized.reference_coordinates))
    assert reference_by_atom["C1"] == (0.0, 0.0, 0.0)
    assert reference_by_atom["O1"] == (1.4, 0.0, 0.0)
    assert all(reference_by_atom[atom.atom_id] is None for atom in materialized.atoms if atom.element == "H")
