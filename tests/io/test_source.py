import gzip

import numpy as np
import pytest

from ofoldx.io import (
    CifSource,
    UnsupportedFormatError,
    parse_cif,
    parse_source,
    read_source,
    read_structure,
    read_template_coordinates,
)
from ofoldx.io.cif import UNKNOWN
from ofoldx.io.identity import Endpoint


class TestSource:
    def test_unresolved_connections(self, cif_text, tmp_path):
        text = (
            cif_text
            + "\nloop_\n_struct_conn.id\n_struct_conn.conn_type_id\n_struct_conn.ptnr1_symmetry\nmissing covale 3_555\n"
        )
        path = tmp_path / "source.cif.gz"
        with gzip.open(path, "wt") as f:
            f.write(text)
        source = read_source(path)
        assert source.block["struct_conn"].value("ptnr1_symmetry", 0) == "3_555"
        assert source.atoms.model_ids == (1, 2)
        assert len(source.atoms.by_index(1)) == 11
        assert len(source.atoms.by_index(2)) == 1
        assert source.atoms.by_index(2).row_index.tolist() == [11]
        assert source.block["atom_site"].row_count == 12

    def test_opaque_identifiers(self, cif_text):
        text = cif_text.replace("ATOM 1 N", "ATOM 001 N").replace("7 ALA Q", "007A ALA Q")
        source = CifSource(parse_cif(text))
        rows = source.atoms.by_index(1)
        assert rows.serial[0] == "001"
        assert rows.auth_residue_id[0] == "007A"
        assert rows.label_seq_id[0] == 1

    def test_interleaved_models(self, cif_text):
        lines = cif_text.splitlines()
        model2 = lines.pop()
        first = next(i for i, line in enumerate(lines) if line.startswith("ATOM 1 "))
        lines.insert(first + 1, model2)
        source = CifSource(parse_cif("\n".join(lines) + "\n"))
        assert len(source.atoms.by_index(1)) == 11
        assert source.atoms.by_index(2).coordinates[0].tolist() == [101, 0, 0]

    def test_missing_coordinate(self, cif_text):
        source = CifSource(parse_cif(cif_text.replace("0 1 0 1 10 7", "? 1 0 1 10 7", 1)))
        assert not source.atoms.by_index(1).coordinate_present[0]
        assert source.atoms.by_index(1).coordinates[0].tolist() == [0, 0, 0]
        assert source.block["atom_site"]["Cartn_x"].mask[0] == UNKNOWN
        assert source.block["atom_site"]["Cartn_y"].values[0] == "1"

    def test_observation_precision(self, cif_text):
        value = 1.234567890123456
        occupancy = 0.33333333333333
        b_factor = 13.1234567890123
        text = cif_text.replace("0 1 0 1 10 7", f"{value} {-value} {value} {occupancy} {b_factor} 7", 1)
        source = CifSource(parse_cif(text))
        category = source.block["atom_site"]
        xyz = [category[name].as_array(np.float64)[0] for name in ("Cartn_x", "Cartn_y", "Cartn_z")]
        assert xyz == [value, -value, value]
        assert category["occupancy"].as_array(np.float64)[0] == occupancy
        assert category["B_iso_or_equiv"].as_array(np.float64)[0] == b_factor
        for rows in (source.atoms.by_index(1), source.atoms.by_id(1), source.view(asym_ids={"A"}).rows):
            assert rows.coordinates.dtype == np.float64
            assert rows.coordinates[0].tolist() == xyz
            assert rows.coordinate_present[0]
            assert rows.occupancy[0] == occupancy
            assert rows.b_factor[0] == b_factor

    def test_native_pdb_models(self, tmp_path):
        from ofoldx.io import PdbSource, parse_source, read_source
        from ofoldx.io.pdb import prepare_pdb_view

        atom = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  "
        # Preserve irregular producer numbering as source facts; do not reinterpret it
        # as a claim that the producer follows PDB's sequential MODEL convention.
        text = f"MODEL        7\n{atom}\nENDMDL\nMODEL       12\n{atom.replace('1.000', '9.000')}\nENDMDL\n"
        source = parse_source(text, format="pdb")
        assert isinstance(source, PdbSource)
        assert source.atoms.model_ids == (7, 12)
        view = source.view(model_id=12)
        assert view.model_index == 2 and view.model_id == 12
        assert view.rows.coordinates.dtype == np.float64
        assert view.rows.coordinates.tolist() == [[9, 2, 3]]
        np.testing.assert_array_equal(source.atoms.by_id(12), source.atoms.by_index(2))
        # Model-specific missing residues use MODEL serials, including range headers.
        row = list(" " * 80)
        row[:10] = "REMARK 465"
        row[11:14], row[15:18], row[19], row[21:26] = " 12", "GLY", "A", "    2"
        missing = parse_source("".join(row) + "\n" + text, format="pdb")
        assert prepare_pdb_view(missing.view(model_id=12))[2] == (("A", "GLY", 2, ""),)
        assert prepare_pdb_view(missing.view(model_id=7))[2] == ()
        row[11:14] = "   "
        missing = parse_source("REMARK 465   MODELS 10-15\n" + "".join(row) + "\n" + text, format="pdb")
        assert prepare_pdb_view(missing.view(model_id=12))[2] == (("A", "GLY", 2, ""),)
        assert prepare_pdb_view(missing.view(model_id=7))[2] == ()
        with pytest.raises(ValueError, match="mutually exclusive"):
            source.view(model_index=1, model_id=12)
        with pytest.raises(ValueError, match="model_id"):
            source.view(model_id=2)
        path = tmp_path / "source.pdb"
        path.write_text(text)
        stored = read_source(path)
        path.unlink()
        assert prepare_pdb_view(stored.view(model_id=12))[0].coordinates.tolist() == [[9, 2, 3]]
        assert stored.text == text
        assert parse_source(atom + "\n", format="pdb").view().model_id is None
        from ofoldx.io import read_structure

        source = parse_source("SEQRES   1 A    1  ALA\nSEQRES   1 B    1  GLY\n" + atom + "\n", format="pdb")
        selected = read_structure(source.view(asym_ids=frozenset({"A"})))
        assert [selected.system.instance_id(i) for i in range(selected.system.instance_symbol.numel())] == ["A"]
        assert "SEQRES   1 B" in source.text


class TestStructureView:
    def test_source_reuse(self, cif_path):
        source = read_source(cif_path)
        snapshot = source.block["atom_site"]["id"].values
        cif_path.unlink()
        projected = read_template_coordinates(source, ["A"])["A"].chains["A"]
        structure = read_structure(source.view(asym_ids={"A"}), include_hydrogens=False)
        assert projected.gather([0], ["CA"])[0].tolist() == [[[1, 0, 0]]]
        names = [structure.system.symbols[index] for index in structure.system.atom_symbol.tolist()]
        ca = names.index("CA")
        assert structure.coordinate_mask[ca]
        assert structure.coordinates[ca].tolist() == [1, 0, 0]
        assert source.block["atom_site"]["id"].values == snapshot
        assert source.atoms.model_ids == (1, 2)

    @pytest.mark.parametrize(
        "options", [{"model_index": 2}, {"model_id": 2}, {"assembly_id": "1"}], ids=["index", "identifier", "assembly"]
    )
    def test_view_options(self, cif_text, options):
        source = CifSource(parse_cif(cif_text))
        with pytest.raises(ValueError, match="selection belongs"):
            read_structure(source.view(), **options)

    def test_owned_selection(self):
        from ofoldx.io import CifSource, StructureView

        chains = {"A"}
        view = StructureView(CifSource(parse_cif("data_x\n_a.v 1\n")), asym_ids=chains)
        chains.add("B")
        assert view.asym_ids == frozenset({"A"})

    @pytest.mark.parametrize("format", ["mmcif", "pdb"])
    def test_common_capabilities(self, cif_text, format):
        source = parse_source(cif_text if format == "mmcif" else "END\n", format=format)
        view = source.view()
        assert view.model_index == 1
        assert not any(
            hasattr(view, name) for name in ("cif", "block", "declarations", "connections", "deposited_index")
        )
        if format == "mmcif":
            assert source.block["atom_site"].row_count == 12
            assert source.declarations.asym_entities == {"A": "1", "B": "2"}
            assert source.atoms.index(1).resolve(Endpoint(("A", 1, None, "ALA", "CA"), (None,) * 5)) == (1,)
        else:
            assert not hasattr(source.atoms, "index")

    def test_unsupported_formats(self, tmp_path):
        with pytest.raises(UnsupportedFormatError):
            parse_source("", format="xyz")
        with pytest.raises(UnsupportedFormatError):
            read_source(tmp_path / "input.xyz")
