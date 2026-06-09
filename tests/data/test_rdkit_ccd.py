from __future__ import annotations

import pytest

from ofoldx.data.rdkit_ccd import (
    MolFromCcdError,
    ensure_conformer,
    mol_from_ccd_block,
    parse_atom_data,
    parse_bond_data,
    reference_component_from_mol,
)

# A minimal alanine-like CCD block: N-CA(-CB)-C(=O), with deposited ideal coordinates.
_ALA_BLOCK = {
    "_chem_comp.id": ["ALA"],
    "_chem_comp_atom.atom_id": ["N", "CA", "C", "O", "CB"],
    "_chem_comp_atom.type_symbol": ["N", "C", "C", "O", "C"],
    "_chem_comp_atom.charge": ["0", "0", "0", "0", "0"],
    "_chem_comp_atom.pdbx_leaving_atom_flag": ["N", "N", "N", "N", "N"],
    "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.5", "2.0", "1.5", "2.0"],
    "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0", "1.3", "2.4", "-1.2"],
    "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0", "0.0", "0.0", "0.0", "0.0"],
    "_chem_comp_bond.atom_id_1": ["N", "CA", "C", "CA"],
    "_chem_comp_bond.atom_id_2": ["CA", "C", "O", "CB"],
    "_chem_comp_bond.value_order": ["SING", "SING", "DOUB", "SING"],
    "_chem_comp_bond.pdbx_aromatic_flag": ["N", "N", "N", "N"],
}


def test_parse_atom_data_reads_names_types_charges() -> None:
    names, types, charges, leaving = parse_atom_data(_ALA_BLOCK, force_parse=False)
    assert names == ["N", "CA", "C", "O", "CB"]
    assert types == ["N", "C", "C", "O", "C"]
    assert charges == [0, 0, 0, 0, 0]
    assert leaving == ["N", "N", "N", "N", "N"]


def test_parse_atom_data_force_parse_fixes_missing_charge() -> None:
    block = {**_ALA_BLOCK, "_chem_comp_atom.charge": ["0", "?", "0", "0", "0"]}
    _, _, charges, _ = parse_atom_data(block, force_parse=True)
    assert charges == [0, 0, 0, 0, 0]


def test_parse_bond_data_maps_orders() -> None:
    begins, ends, orders, aromatics = parse_bond_data(_ALA_BLOCK, force_parse=False)
    assert begins == ["N", "CA", "C", "CA"]
    assert ends == ["CA", "C", "O", "CB"]
    assert [str(order) for order in orders] == ["SINGLE", "SINGLE", "DOUBLE", "SINGLE"]
    assert aromatics == [False, False, False, False]


def test_mol_from_block_builds_expected_component() -> None:
    mol = mol_from_ccd_block(_ALA_BLOCK, force_parse=True)
    component = reference_component_from_mol(mol)
    assert component is not None
    assert set(component.atom_map) == {"N", "CA", "C", "O", "CB"}
    assert len(component.bonds) == 4
    assert component.coords.shape == (5, 3)
    assert component.mask.all()  # deposited ideal coords are present


def test_missing_aromatic_flag_raises_without_force_parse() -> None:
    block = {key: value for key, value in _ALA_BLOCK.items() if key != "_chem_comp_bond.pdbx_aromatic_flag"}
    with pytest.raises(MolFromCcdError):
        mol_from_ccd_block(block, force_parse=False)


def test_etkdg_conformer_generated_when_ideal_missing() -> None:
    block = {key: value for key, value in _ALA_BLOCK.items() if "ideal" not in key}
    mol = mol_from_ccd_block(block, force_parse=True)
    assert mol.GetNumConformers() == 0
    mol = ensure_conformer(mol, random_seed=0)
    assert mol.GetNumConformers() >= 1
    component = reference_component_from_mol(mol)
    assert component is not None
    assert component.mask.all()
