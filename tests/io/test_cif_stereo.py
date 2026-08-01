from pathlib import Path

import pytest
import torch
from biotite.structure.io.pdbx import CIFColumn, CIFFile

from ofoldx.data import (
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    Structure,
)
from ofoldx.data._cif_chemistry import read_mmcif_components
from ofoldx.data.ccd.reader import decode_component
from ofoldx.data.chemistry import to_rdkit_component
from ofoldx.io import write_structure
from ofoldx.io.cif import parse_cif

_SOURCE = """data_TEST
_chem_comp.id TEST
_chem_comp.type non-polymer
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_stereo_config
TEST C1 C {atom}
TEST C2 C ?
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_stereo_config
TEST C1 C2 doub {bond}
"""


@pytest.mark.parametrize("atom,bond", [("R", "E"), ("S", "Z"), ("N", "N"), ("?", ".")])
def test_ccd_and_embedded_cif_stereo_have_the_same_case_semantics(atom, bond):
    upper = parse_cif(_SOURCE.format(atom=atom, bond=bond))
    lower = parse_cif(_SOURCE.format(atom=atom.lower(), bond=bond.lower()))
    upper_components, _, _, upper_known = read_mmcif_components(upper)
    lower_components, _, _, lower_known = read_mmcif_components(lower)
    assert upper_components == lower_components
    assert upper_known == lower_known
    assert decode_component(upper.block) == decode_component(lower.block)
    expected_atom = None if atom in {"N", "?"} else atom
    expected_bond = None if bond in {"N", "."} else bond
    component = lower_components["TEST"]
    assert component.atoms[0].stereo == expected_atom
    assert next(iter(component.bonds)).stereo == expected_bond
    assert ("C1" in lower_known["TEST"][0]) == (atom != "?")


@pytest.mark.parametrize("atom,bond", [("E", "N"), ("N", "R"), ("CIS", "N"), ("N", "TRANS")])
def test_cif_rejects_stereo_values_outside_the_item_vocabulary(atom, bond):
    cif = parse_cif(_SOURCE.format(atom=atom, bond=bond))
    with pytest.raises(ValueError, match="pdbx_stereo_config"):
        read_mmcif_components(cif)
    with pytest.raises(ValueError, match="pdbx_stereo_config"):
        decode_component(cif.block)


def test_real_cys_lowercase_retains_chirality_and_explicit_absence():
    path = Path(__file__).parent / "fixtures" / "cys_stereo.cif"
    results = []
    for lower in (False, True):
        cif = CIFFile.read(path)
        if lower:
            for category in ("chem_comp_atom", "chem_comp_bond"):
                column = cif.block[category]["pdbx_stereo_config"].as_array(str)
                cif.block[category]["pdbx_stereo_config"] = CIFColumn([x.lower() for x in column])
        components, _, _, known = read_mmcif_components(parse_cif(cif.serialize()))
        component = components["CYS"]
        system = MolecularSystem.from_entities({"L": Molecule((component,))}, {"A": "L"})
        mol = to_rdkit_component(system, 0)
        ca = next(i for i, atom in enumerate(component.atoms) if atom.atom_id == "CA")
        assert mol.GetAtomWithIdx(ca).GetProp("_CIPCode") == "R"
        assert all(atom.stereo is None for atom in component.atoms if atom.atom_id != "CA")
        results.append((component, known))
    assert results[0] == results[1]


@pytest.mark.parametrize("stereo", ["r", "s"])
def test_writer_does_not_encode_canonical_pseudoasymmetry_as_cif_ucode(tmp_path, stereo):
    # This checks the format boundary, not whether a one-atom component could
    # physically have this chirality: the canonical descriptor must not be lost.
    component = ChemicalComponent("TEST", (ChemicalAtom("C", "C", stereo=stereo),), frozenset())
    system = MolecularSystem.from_entities({"L": Molecule((component,))}, {"A": "L"})
    structure = Structure(
        system=system,
        coordinates=torch.zeros(1, 3),
        coordinate_mask=torch.ones(1, dtype=torch.bool),
        occupancy=torch.ones(1),
        occupancy_mask=torch.ones(1, dtype=torch.bool),
        b_factor=torch.zeros(1),
        b_factor_mask=torch.ones(1, dtype=torch.bool),
        source_symbols=("1",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
    )
    path = tmp_path / "output.cif"
    with pytest.raises(ValueError, match="pdbx_stereo_config cannot encode"):
        write_structure(structure, path)
    assert not path.exists()


@pytest.mark.parametrize("literal", ["'.'", "'?'", "''"])
def test_quoted_missing_markers_are_invalid_stereo_codes(literal):
    cif = parse_cif(_SOURCE.format(atom=literal, bond="N"))
    assert cif.block["chem_comp_atom"]["pdbx_stereo_config"].present[0]
    with pytest.raises(ValueError, match="pdbx_stereo_config"):
        read_mmcif_components(cif)
    with pytest.raises(ValueError, match="pdbx_stereo_config"):
        decode_component(cif.block)


@pytest.mark.parametrize("invalid", ["'?'", "'.'", "maybe"])
def test_aromatic_atom_flag_cannot_silently_become_false(invalid):
    text = _SOURCE.format(atom="N", bond="N").replace(
        "_chem_comp_atom.pdbx_stereo_config\nTEST C1 C N\nTEST C2 C ?",
        f"_chem_comp_atom.pdbx_stereo_config\n_chem_comp_atom.pdbx_aromatic_flag\nTEST C1 C N {invalid}\nTEST C2 C ? ?",
    )
    cif = parse_cif(text)
    with pytest.raises(ValueError, match="aromatic_flag"):
        read_mmcif_components(cif)
    with pytest.raises(ValueError, match="CCD flag"):
        decode_component(cif.block)
