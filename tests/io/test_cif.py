import numpy as np
import pytest

from ofoldx.io.cif import INAPPLICABLE, PRESENT, UNKNOWN, CifCategory, CifColumn, column_points, parse_cif


class TestCifColumn:
    def test_null_states(self):
        doc = parse_cif("data_syntax\nloop_\n_EXAMPLE.ID\n_example.value\n1 ?\n2 .\n3 '?'\n4 '.'\n5\n;two\nlines\n;\n")
        table = doc.block["Example"]
        assert table["VALUE"].mask.tolist() == [UNKNOWN, INAPPLICABLE, PRESENT, PRESENT, PRESENT]
        assert table["value"].values == (None, None, "?", ".", "two\nlines")
        assert table.value("value", 2) == "?"
        assert table.value("absent", 0) is None
        assert "absent" not in table

    def test_numeric_fill(self):
        col = CifColumn(["0", None, False, "1.5"])
        with pytest.raises(ValueError, match="explicit missing"):
            col.as_array(np.float64)
        values = col.as_array(np.float64, np.nan)
        np.testing.assert_array_equal(np.isnan(values), [False, True, True, False])
        assert values[0] == 0
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 123

    def test_owned_arrays(self):
        values = ["a", None]
        owned = CifColumn(values)
        values[0] = "changed"
        exported = owned.as_array(object)
        exported.setflags(write=True)
        exported[0] = "changed"
        assert owned.values == ("a", None)
        assert owned.as_array(object).tolist() == ["a", "?"]
        col = parse_cif("data_x\nloop_\n_a.v\n?\n.\n'?'\n\".\"\n' 001 '\n").block["a"]["v"]
        assert col.raw_values == ("?", ".", "'?'", '"."', "' 001 '")
        assert col.values == (None, None, "?", ".", " 001 ")
        assert col.mask.tolist() == [2, 1, 0, 0, 0]
        for array in (col.mask, col.present):
            with pytest.raises(ValueError):
                array.setflags(write=True)
        numeric = CifColumn(["1", None]).as_array(float, np.nan)
        with pytest.raises(ValueError):
            numeric.setflags(write=True)

    @pytest.mark.parametrize(
        "token,expected,su",
        [
            ("12.34(5)", 12.34, 0.05),
            ("1.23e2(4)", 123.0, 4.0),
            ("1.23(4)e2", 123.0, 4.0),
            ("-1.23(4)e-2", -0.0123, 0.0004),
            ("+2.5", 2.5, None),
            ("0.0(0)", 0.0, 0.0),
        ],
    )
    def test_numeric_forms(self, token, expected, su):
        doc = parse_cif(f"data_x\n_a.x {token}\n_a.y 2\n_a.z 3\n")
        column = doc.block["a"]["x"]
        assert column.as_array(float)[0] == expected
        assert column_points(doc.block["a"], "x", "y", "z") == ((expected, 2.0, 3.0),)
        if su is None:
            assert not column.uncertainty_present[0]
            assert np.isnan(column.uncertainty[0])
        else:
            assert column.uncertainty[0] == pytest.approx(su)

    @pytest.mark.parametrize(
        "token", ["'1.2'", '"1.2"', "1()", "1(2)(3)", "1_000", "NaN", "inf", "Infinity", "12bad(3)", "1e999", "1e-999"]
    )
    def test_invalid_numbers(self, token):
        column = parse_cif(f"data_x\n_a.x {token}\n").block["a"]["x"]
        with pytest.raises(ValueError, match="row 1"):
            column.as_array(float, np.nan)

    def test_assignment(self):
        column = CifColumn(["a", None])
        with pytest.raises(AttributeError):
            column.values = ("changed",)
        with pytest.raises(AttributeError):
            column.mask = np.zeros(2)

    def test_quoted_strings(self):
        doc = parse_cif("data_x\nloop_\n_a.v\n''\n\"a ' b\"\n; leading\ninner  \n;\n ;bare\n")
        assert doc.block["a"]["v"].values == ("", "a ' b", " leading\ninner  ", ";bare")

    @pytest.mark.parametrize("values", [[None], [""], [False]])
    def test_invalid_raw_tokens(self, values):
        with pytest.raises(ValueError, match="raw CIF tokens"):
            CifColumn(values, raw=True)
        with pytest.raises(ValueError, match="raw CIF tokens"):
            CifCategory({"v": values}, raw=True)

    def test_optional_strings(self):
        category = CifCategory({"value": [" 001 ", "1A", None, False, "N", " n "]})
        assert category.strings("value") == (" 001 ", "1A", None, None, "N", " n ")
        assert category.strings("absent") == (None,) * 6
