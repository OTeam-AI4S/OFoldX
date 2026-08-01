import numpy as np
import pytest
from biotite.structure import AtomArray

from ofoldx.io.records import atom_rows_from_array


class TestAtomRecords:
    @pytest.mark.parametrize("annotation_dtype", [np.float32, np.float64])
    def test_atom_columns(self, annotation_dtype):
        array = AtomArray(3)
        array.chain_id = np.array(["fallback"] * 3)
        array.res_id = np.array([7, 8, 9])
        array.atom_name = np.array(["C1", "D1", "T1"])
        array.res_name = np.array(["LIG"] * 3)
        array.element = np.array(["C", "D", "T"])
        array.coord[:] = [[1.234567890123456, 2, 3], [4, 5, 6], [np.nan, 0, 0]]
        for name, values in {
            "atom_id": ["001", "2", "3"],
            "label_asym_id": [" A ", ".", "?"],
            "label_seq_id": ["1", ".", "?"],
            "label_entity_id": ["1", ".", "?"],
            "auth_seq_id": ["001", "1A", "?"],
            "pdbx_formal_charge": ["-1", ".", "0"],
        }.items():
            array.set_annotation(name, np.array(values))
        array.set_annotation("occupancy", np.array([0.33333333333333, 0.7, 1], dtype=annotation_dtype))
        array.set_annotation("b_factor", np.array([13.1234567890123, 20, 0], dtype=annotation_dtype))
        rows = atom_rows_from_array(array, format_name="mmcif")
        assert rows["serial"].tolist() == ["001", "2", "3"]
        assert rows["chain_id"].tolist() == ["A", "fallback", "fallback"]
        assert rows["residue_id"].tolist() == [1, 8, 9]
        assert rows["auth_residue_id"].tolist() == ["001", "1A", None]
        assert rows["entity_id"].tolist() == ["1", None, None]
        assert rows["charge"].tolist() == [-1, None, 0]
        assert rows["element"].tolist() == ["C", "H", "H"]
        assert rows["isotope"].tolist() == [None, 2, 3]
        np.testing.assert_array_equal(rows["occupancy"].astype(np.float64), array.occupancy)
        np.testing.assert_array_equal(rows["b_factor"].astype(np.float64), array.b_factor)
        assert rows["coordinates"].dtype == np.float64
        np.testing.assert_array_equal(rows["coordinates"][:2], array.coord[:2])
        assert rows["coordinate_present"].tolist() == [True, True, False]
        assert rows["coordinates"][2].tolist() == [0, 0, 0]
