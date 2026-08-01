import numpy as np
import pytest

from ofoldx.io import CifSource, parse_cif
from ofoldx.io.symmetry import SymmetryContext


class TestSymmetry:
    @pytest.mark.parametrize(
        "prefix, scale, offset", [("fract", "0.125(1)", "0.25(1)"), ("Cartn", "8.0(1)", "-2.0(1)")]
    )
    def test_crystal_numeric_transform(self, prefix, scale, offset):
        text = """data_crystal
_space_group_symop.id 2
_space_group_symop.operation_xyz '-x,-y,-z'
"""
        text += "".join(
            f"_atom_sites.{prefix}_transf_matrix[{row}][{column}] {scale if row == column else '0'}\n"
            for row in (1, 2, 3)
            for column in (1, 2, 3)
        )
        text += "".join(
            f"_atom_sites.{prefix}_transf_vector[{row}] {offset if row == 1 else '0'}\n" for row in (1, 2, 3)
        )
        operation, transform = SymmetryContext(parse_cif(text).block).transform("2_555")
        assert operation == "2_555"
        np.testing.assert_array_equal(transform, [[-1, 0, 0, -4], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        quoted = text.replace(scale, f"'{scale.split('(')[0]}'", 1)
        with pytest.raises(ValueError, match="CIF number"):
            SymmetryContext(parse_cif(quoted).block).transform("2_555")

    def test_assembly_coordinates(self, cif_text):
        from ofoldx.data._placed_atoms import PlacedAtoms

        source = CifSource(parse_cif(cif_text))
        rows = source.atoms.by_index(1)
        before = rows.copy()
        shift = np.eye(4)
        shift[:3, 3] = [10, 20, 30]
        placed = PlacedAtoms.assembly(rows, {("1", "A"): np.eye(4), ("2", "A"): shift})
        selected = placed.select_source_rows([rows.row_index[1]])
        assert selected.source_positions.tolist() == [1, 1]
        assert selected.symmetry_id.tolist() == ["1", "2"]
        np.testing.assert_array_equal(selected.coordinates, [rows.coordinates[1], rows.coordinates[1] + [10, 20, 30]])
        np.testing.assert_array_equal(rows, before)

    def test_assembly_connections(self):
        from ofoldx.data import ChemicalAtom, ChemicalComponent
        from ofoldx.data.materialize import materialize_structure
        from ofoldx.io import StructureView

        text = """data_copies
    loop_
    _entity.id
    _entity.type
    1 non-polymer
    2 non-polymer
    loop_
    _struct_asym.id
    _struct_asym.entity_id
    A 1
    B 2
    loop_
    _pdbx_entity_nonpoly.entity_id
    _pdbx_entity_nonpoly.comp_id
    1 XXX
    2 YYY
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
    _atom_site.auth_atom_id
    _atom_site.auth_comp_id
    _atom_site.auth_asym_id
    _atom_site.auth_seq_id
    _atom_site.Cartn_x
    _atom_site.Cartn_y
    _atom_site.Cartn_z
    _atom_site.pdbx_PDB_model_num
    HETATM 1 C C1 . XXX A 1 . C1 XXX Q 7 0 0 0 1
    HETATM 2 O O1 . YYY B 2 . O1 YYY R 8 1 0 0 1
    loop_
    _struct_conn.id
    _struct_conn.conn_type_id
    _struct_conn.ptnr1_label_asym_id
    _struct_conn.ptnr1_label_comp_id
    _struct_conn.ptnr1_label_atom_id
    _struct_conn.ptnr1_symmetry
    _struct_conn.ptnr2_label_asym_id
    _struct_conn.ptnr2_label_comp_id
    _struct_conn.ptnr2_label_atom_id
    _struct_conn.ptnr2_symmetry
    _struct_conn.pdbx_leaving_atom_flag
    1 covale A XXX C1 1_555 B YYY O1 1_555 none
    _pdbx_struct_assembly_gen.assembly_id 1
    _pdbx_struct_assembly_gen.oper_expression '(1-3)'
    _pdbx_struct_assembly_gen.asym_id_list A,B
    loop_
    _pdbx_struct_oper_list.id
    """
        text += "".join(f"_pdbx_struct_oper_list.matrix[{i}][{j}]\n" for i in (1, 2, 3) for j in (1, 2, 3))
        text += "".join(f"_pdbx_struct_oper_list.vector[{i}]\n" for i in (1, 2, 3))
        text += "".join(f"{copy} 1 0 0 0 1 0 0 0 1 {copy * 10} 0 0\n" for copy in range(1, 4))
        source = CifSource(parse_cif(text))
        components = {
            "XXX": ChemicalComponent("XXX", (ChemicalAtom("C1", "C"),), bonds=frozenset()),
            "YYY": ChemicalComponent("YYY", (ChemicalAtom("O1", "O"),), bonds=frozenset()),
        }
        structure = materialize_structure(StructureView(source, assembly_id="1"), components=components)
        assert structure.coordinate_mask.sum() == 6
        assert structure.system.edge_index.shape[1] == 3
        assert len(source.atoms.by_index(1)) == 2
        # Author-only references use the actual source crosswalk, including its
        # label placement. Missing label fields cannot silently erase a valid edge.
        author = text.replace("_struct_conn.ptnr1_label_asym_id", "_struct_conn.ptnr1_auth_asym_id")
        author = author.replace("_struct_conn.ptnr2_label_asym_id", "_struct_conn.ptnr2_auth_asym_id")
        author = author.replace("1 covale A XXX C1 1_555 B YYY O1", "1 covale Q XXX C1 1_555 R YYY O1")
        projected = materialize_structure(
            StructureView(CifSource(parse_cif(author)), assembly_id="1"), components=components
        )
        assert projected.system.edge_index.shape[1] == 3
