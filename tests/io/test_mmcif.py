from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ofoldx.data import ChemicalAtom, ChemicalBond, ChemicalComponent, Molecule
from ofoldx.io import LossReport, read_structure, write_structure
from tests.data.test_codec import _heterogeneous_structure


def test_mmcif_semantic_round_trip_preserves_complete_canonical_value(tmp_path: Path) -> None:
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
    ligand = ChemicalComponent("LIG", atoms=(ChemicalAtom("C1", "C", formal_charge=0),), bonds=frozenset())

    structure = read_structure(path, components={"LIG": ligand})

    assert len(structure.system.bonds) == 1
    bond = next(iter(structure.system.bonds))
    assert bond.origin == "source"
    assert {bond.site_a.atom_id, bond.site_b.atom_id} == {"CA", "C1"}


def test_external_mmcif_rejects_conflicting_nonpolymer_definitions_for_one_entity(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="source entity ID .* has conflicting molecular definitions"):
        read_structure(path, components=components)


def test_canonical_mmcif_reads_coordinates_from_atom_site(tmp_path: Path) -> None:
    from biotite.structure.io import pdbx

    path = tmp_path / "coordinates.cif"
    write_structure(_heterogeneous_structure(), path)
    cif_file = pdbx.CIFFile.read(str(path))
    coordinates = cif_file.block["atom_site"]["Cartn_x"].as_array()
    coordinates[0] = 123.0
    cif_file.block["atom_site"]["Cartn_x"] = pdbx.CIFColumn(coordinates)
    cif_file.write(str(path))

    restored = read_structure(path)

    assert restored.coordinates[0, 0] == 123.0


def test_mmcif_writes_finite_placeholder_for_masked_coordinates(tmp_path: Path) -> None:
    structure = _heterogeneous_structure()
    coordinates = structure.coordinates.copy()
    coordinate_mask = structure.coordinate_mask.copy()
    coordinates[0] = np.nan
    coordinate_mask[0] = False
    masked = type(structure)(
        structure.system,
        structure.sites,
        coordinates,
        coordinate_mask,
        structure.occupancy,
        structure.b_factor,
        structure.residue_numbers,
    )
    path = tmp_path / "masked.cif"

    write_structure(masked, path)
    restored = read_structure(path)

    assert not restored.coordinate_mask[0]
    assert np.array_equal(restored.coordinates[0], np.zeros(3, dtype=restored.coordinates.dtype))
