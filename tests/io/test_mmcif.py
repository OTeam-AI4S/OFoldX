from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalBond, ChemicalComponent, Molecule
from ofoldx.io import LossReport, read_structure, write_structure
from tests.data.test_codec import _heterogeneous_structure


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
