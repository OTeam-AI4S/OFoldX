import pytest

from ofoldx.io.cif import CifCategory, parse_cif
from ofoldx.io.declarations import (
    EntityDeclarations,
    read_component_parents,
    read_entity_declarations,
    read_mmcif_resolutions,
)


class TestDeclarations:
    def test_owned_selection(self):
        types = {"p": "polymer", "l": "non-polymer"}
        asym = {"A": "p", "B": "p", "C": "l"}
        declarations = EntityDeclarations(
            entity_types=types,
            polymer_monomers={"p": ((1, ("ALA",)),)},
            polymer_types={"p": "polypeptide(L)"},
            author_numbering={("A", 1): ("007", None), ("B", 1): ("9", None)},
            asym_entities=asym,
            nonpolymer_components={"l": "ATP"},
            nonpolymer_occurrences={"C": (("01", "l", "ATP", "1", None, "Q"),)},
        )
        types["p"] = "water"
        asym.clear()
        assert declarations.entity_types["p"] == "polymer"
        selected = declarations.select_asym_ids({"B"})
        assert dict(selected.asym_entities) == {"B": "p"}
        assert dict(selected.entity_types) == {"p": "polymer"}
        assert dict(selected.author_numbering) == {("B", 1): ("9", None)}
        assert not selected.nonpolymer_components
        assert not selected.nonpolymer_occurrences
        assert len(declarations.asym_entities) == 3
        with pytest.raises(TypeError):
            declarations.entity_types["p"] = "water"

    def test_absent_sequence(self):
        declared = read_entity_declarations(parse_cif("data_x\n_entity.id 1\n_entity.type polymer\n").block)
        assert declared.polymer_monomers is None
        assert declared.select_asym_ids(set()).polymer_monomers is None

    @pytest.mark.parametrize("token", ["1_0", "'10'", "10.0", "10(1)"])
    def test_sequence_integers(self, token):
        from ofoldx.io.declarations import read_entity_declarations

        doc = parse_cif(
            f"data_x\n_entity_poly_seq.entity_id A\n_entity_poly_seq.num {token}\n_entity_poly_seq.mon_id ALA\n"
        )
        with pytest.raises(ValueError, match="row 1"):
            read_entity_declarations(doc.block)

    @pytest.mark.parametrize("token,expected", [("12.34(5)", 12.34), ("1.23e2(4)", 123.0), ("1.23(4)e2", 123.0)])
    def test_resolution_numbers(self, token, expected):
        block = parse_cif(f"data_x\n_refine.ls_d_res_high {token}\n").block
        assert read_mmcif_resolutions(block)[0][0] == expected

    @pytest.mark.parametrize(
        "value,expected",
        [(None, {}), (False, {}), ("MET", {"MSE": ("MET",)}), ("SER, TYR,GLY", {"MSE": ("SER", "TYR", "GLY")})],
    )
    def test_parent_lists(self, value, expected):
        category = CifCategory({"id": ["MSE"], "mon_nstd_parent_comp_id": [value]})
        assert read_component_parents(category) == expected

    def test_absent_parents(self):
        assert read_component_parents(None) == {}
        assert read_component_parents(CifCategory({"id": ["ALA"]})) == {}

    @pytest.mark.parametrize("parents", [["ALA,", "ALA,"], ["ALA", "GLY"]], ids=["empty-parent", "conflicting-parents"])
    def test_invalid_parents(self, parents):
        category = CifCategory({"id": ["LIG", "LIG"], "mon_nstd_parent_comp_id": parents})
        with pytest.raises(ValueError, match="parent"):
            read_component_parents(category)
