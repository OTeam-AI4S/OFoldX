import pytest

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalBond, ChemicalComponent, MolecularSystem, Molecule, SystemBond
from ofoldx.io.cif import parse_cif


class TestMaterialization:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_leaving_conflict(self, reverse):
        component = ChemicalComponent(
            "LIG",
            (ChemicalAtom("C", "C"), ChemicalAtom("O", "O"), ChemicalAtom("H", "H")),
            frozenset({ChemicalBond("C", "O"), ChemicalBond("C", "H")}),
        )
        a, b = AtomSite("A", 0, "C"), AtomSite("A", 0, "O")
        connection = (
            SystemBond(b, a, leaving_b=frozenset({"H"})) if reverse else SystemBond(a, b, leaving_a=frozenset({"H"}))
        )
        with pytest.raises(ValueError, match="conflicting leaving atoms for physical edge"):
            MolecularSystem.from_entities({"L": Molecule((component,))}, {"A": "L"}, (connection,))

    def test_order_conflict(self):
        component = ChemicalComponent(
            "LIG",
            (ChemicalAtom("C", "C"), ChemicalAtom("O", "O")),
            frozenset({ChemicalBond("C", "O", order=2.0)}),
        )
        connection = SystemBond(AtomSite("A", 0, "C"), AtomSite("A", 0, "O"), order=1.0)
        with pytest.raises(ValueError, match="conflicting"):
            MolecularSystem.from_entities({"L": Molecule((component,))}, {"A": "L"}, (connection,))

    def test_ccd_numeric_contract(self):
        from ofoldx.data.ccd.reader import decode_component

        text = """data_X
    _chem_comp.id X
    _chem_comp.type non-polymer
    _chem_comp_atom.atom_id C1
    _chem_comp_atom.type_symbol C
    _chem_comp_atom.charge 0
    _chem_comp_atom.pdbx_model_Cartn_x_ideal 12.34(5)
    _chem_comp_atom.pdbx_model_Cartn_y_ideal 1.23(4)e2
    _chem_comp_atom.pdbx_model_Cartn_z_ideal 0
    """
        row = decode_component(parse_cif(text).block)
        assert row["reference_coordinates"] == [[12.34, 123.0, 0.0]]
        with pytest.raises(ValueError, match="row 1"):
            decode_component(parse_cif(text.replace("12.34(5)", "'12.34(5)'")).block)
