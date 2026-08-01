from __future__ import annotations

import gzip

import numpy as np
import pytest
import torch

from ofoldx.data import Structure
from ofoldx.io import (
    UnsupportedFormatError,
    parse_source,
    read_template_coordinates,
    read_template_structure,
    read_template_structures,
)
from ofoldx.io.identity import InsufficientIdentityEvidenceError

_AUTHOR_SCHEME = """
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 ALA Q
"""


@pytest.fixture(params=[read_template_structures, read_template_coordinates], ids=["structure", "coordinates"])
def template_reader(request):
    return request.param


def _atom(structure, residue, name):
    system = structure.system
    start, stop = system.component_atom_ptr[residue : residue + 2].tolist()
    return next(i for i in range(start, stop) if system.symbols[int(system.atom_symbol[i])] == name)


class TestTemplateIO:
    def test_multiple_chains(self, cif_text, cif_path, template_reader):
        text = cif_text.replace("A 1\nB 2", "A 1\nC 1\nB 2")
        for line in cif_text.splitlines():
            if line.startswith(("ATOM ", "HETATM ")) and " A 1 " in line:
                fields = line.split()
                fields[1] = str(int(fields[1]) + 100)
                fields[6] = "C"
                fields[17] = "R"
                text += " ".join(fields) + "\n"
        cif_path.write_text(text)
        expected = {chain: template_reader(cif_path, [chain], chain_id_kind="auth")[chain] for chain in ("Q", "R")}
        actual = template_reader(cif_path, ["Q", "R", "Q"], chain_id_kind="auth")
        assert list(actual) == ["Q", "R"]
        for chain, result in actual.items():
            if isinstance(result, Structure):
                torch.testing.assert_close(
                    result.state_dict(), expected[chain].state_dict(), rtol=0, atol=0, equal_nan=True
                )
            else:
                assert tuple(result.chains) == tuple(expected[chain].chains)
                for label, value in result.chains.items():
                    reference = expected[chain].chains[label]
                    assert value.component_ids == reference.component_ids
                    assert value.parent_component_ids == reference.parent_component_ids
                    np.testing.assert_array_equal(value.coordinates, reference.coordinates)
                    np.testing.assert_array_equal(value.coordinate_mask, reference.coordinate_mask)
        # Revisiting a path reads the current file contents.
        cif_path.write_text(text.replace("? 1 0 0 1 10 7", "? 9 0 0 1 10 7"))
        updated = template_reader(cif_path, ["Q"], chain_id_kind="auth")["Q"]
        if isinstance(updated, Structure):
            assert updated.coordinates[_atom(updated, 0, "CA")].tolist() == [9, 0, 0]
        else:
            assert updated.chains["A"].gather([0], ["CA"])[0].tolist() == [[[9, 0, 0]]]

    def test_chain_errors(self, cif_path, template_reader):
        with pytest.raises(KeyError):
            template_reader(cif_path, ["missing", "A"])
        errors = []
        actual = template_reader(
            cif_path, ["missing", "A"], on_error=lambda chain, error: errors.append((chain, error))
        )
        assert list(actual) == ["A"]
        assert len(errors) == 1 and errors[0][0] == "missing" and isinstance(errors[0][1], KeyError)
        with pytest.raises(TypeError, match="sequence"):
            template_reader(cif_path, "A")

    def test_observed_sequence(self, cif_path):
        result = read_template_structure(cif_path, "Q", chain_id_kind="auth")
        projected = read_template_coordinates(cif_path, ["Q"], chain_id_kind="auth")["Q"].chains["A"]
        assert projected.component_ids == ("ALA", "GLY", "MSE")
        assert projected.parent_component_ids == ((), (), ("MET",))
        assert projected.gather([0], ["CB"])[0].tolist() == [[[1, -1, 1]]]
        assert not projected.gather([1], ["CA"])[1].any()
        assert result.system.instance_id(0) == "A"
        assert result.system.instance_component_ptr.tolist() == [0, 3]
        assert result.system.component_topology_complete.tolist() == [False] * 3
        assert result.system.edge_index.numel() == 0
        assert not result.coordinate_mask[
            result.system.component_atom_ptr[1] : result.system.component_atom_ptr[2]
        ].any()
        assert result.component_parents(2) == ("MET",)
        assert result.source_symbols[int(result.residue_id[0])] == "7"
        assert result.source_symbols[int(result.insertion_code[2])] == "A"
        assert result.coordinates[_atom(result, 0, "CB")].tolist() == [1, -1, 1]
        assert result.system.symbols[int(result.system.component_symbol[2])] == "MSE"
        assert result.coordinates.shape[0] == 9

    def test_model_namespace(self, cif_text, cif_path, tmp_path):
        compressed = tmp_path / "template.cif.gz"
        with gzip.open(compressed, "wt") as stream:
            stream.write(cif_text)
        result = read_template_structure(compressed, "A", model_index=2)
        assert result.coordinates[_atom(result, 0, "CA")].tolist() == [101, 0, 0]
        with pytest.raises(KeyError):
            read_template_structure(cif_path, "A", chain_id_kind="auth")
        with pytest.raises(ValueError, match="model"):
            read_template_structure(cif_path, "A", model_index=3)

    def test_invalid_atoms(self, cif_text, cif_path):
        cif_path.write_text(
            cif_text.replace("ATOM 12 C CA", "ATOM 13 C CA . ALA A 1 1 ? 99 0 0 1 10 7 ALA Q CA 1\nATOM 12 C CA")
        )
        with pytest.raises(ValueError, match="duplicate"):
            read_template_structure(cif_path, "A")
        cif_path.write_text(cif_text.replace("CB A ALA A 1 1", "CB A SER A 1 1"))
        with pytest.raises(ValueError):
            read_template_structure(cif_path, "A", altloc="first")

    @pytest.mark.parametrize("altloc,component", [("first", "ALA"), ("highest_occupancy", "SER")])
    def test_microheterogeneity(self, cif_text, cif_path, altloc, component):
        text = cif_text.replace("1 1 ALA n", "1 1 ALA y\n1 1 SER y")
        # All atoms in each realization carry its label; SER has greater occupancy.
        rows = text.splitlines()
        updated = []
        for row in rows:
            fields = row.split()
            if len(fields) == 20 and fields[0] == "ATOM" and fields[19] == "1":
                if fields[4] == "B":
                    continue
                fields[4], fields[13] = "A", ".3"
                updated.append(" ".join(fields))
                fields[1], fields[4], fields[5], fields[13], fields[16] = (
                    str(100 + int(fields[1])),
                    "B",
                    "SER",
                    ".7",
                    "SER",
                )
                updated.append(" ".join(fields))
            else:
                updated.append(row)
        cif_path.write_text("\n".join(updated) + "\n")
        result = read_template_structure(cif_path, "A", altloc=altloc)
        assert result.system.symbols[int(result.system.component_symbol[0])] == component
        assert read_template_coordinates(cif_path, ["A"], altloc=altloc)["A"].chains["A"].component_ids[0] == component

    def test_absent_model_chain(self, cif_text, cif_path):
        lines = []
        for row in cif_text.splitlines():
            fields = row.split()
            if len(fields) == 20 and fields[6] == "A" and fields[19] == "1":
                continue
            lines.append(row)
        cif_path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="no atoms in requested model 1"):
            read_template_structure(cif_path, "A", model_index=1)
        result = read_template_structure(cif_path, "A", model_index=2)
        assert result.coordinates[_atom(result, 0, "CA")].tolist() == [101, 0, 0]

    @pytest.mark.parametrize(
        "author,error,match",
        [("Q", ValueError, "ambiguous author"), ("?", InsufficientIdentityEvidenceError, "unknown author")],
        ids=["ambiguous", "unknown"],
    )
    def test_author_alternatives(self, cif_text, template_reader, author, error, match):
        text = cif_text.replace("A 1\nB 2", "A 1\nC 1\nB 2")
        text += f"ATOM 13 C CA . ALA C 1 1 ? 99 0 0 1 10 7 ALA {author} CA 1\n"
        with pytest.raises(error, match=match):
            template_reader(parse_source(text, format="mmcif"), ["Q"], chain_id_kind="auth")

    @pytest.mark.parametrize("kind", ["atom", "scheme"])
    def test_author_conflict(self, cif_text, template_reader, kind):
        if kind == "atom":
            text = cif_text.replace("7 ALA Q N 1", "7 ALA R N 1", 1)
        else:
            text = cif_text + _AUTHOR_SCHEME.replace("ALA Q", "ALA R")
        errors = []
        with pytest.raises(ValueError, match="author"):
            template_reader(parse_source(text, format="mmcif"), ["A"], on_error=lambda *event: errors.append(event))
        assert errors == []

    def test_scheme_author(self, cif_text, template_reader):
        text = cif_text.replace(" ALA Q ", " ALA ? ").replace(" MSE Q ", " MSE ? ")
        result = template_reader(parse_source(text + _AUTHOR_SCHEME, format="mmcif"), ["Q"], chain_id_kind="auth")
        assert list(result) == ["Q"]

    def test_unobserved_author(self, cif_text, template_reader):
        text = cif_text.replace("A 1\nB 2", "A 1\nC 1\nB 2")
        text += _AUTHOR_SCHEME.replace("A 1 ALA Q", "C 1 ALA R")
        with pytest.raises(ValueError, match="no atoms"):
            template_reader(parse_source(text, format="mmcif"), ["R"], chain_id_kind="auth")

    def test_missing_declarations(self, incomplete_cif, template_reader):
        category, text = incomplete_cif
        errors = []
        with pytest.raises(ValueError, match="_" + category):
            template_reader(parse_source(text, format="mmcif"), ["A"], on_error=lambda *event: errors.append(event))
        assert errors == []

    @pytest.mark.parametrize("kind", ["numeric", "duplicate"])
    def test_decode_before_callback(self, cif_text, template_reader, kind):
        if kind == "numeric":
            text = cif_text.replace("? 0 1 0 1 10 7", "? bad 1 0 1 10 7", 1)
        else:
            text = cif_text + "ATOM 13 C CA . ALA A 1 1 ? 99 0 0 1 10 7 ALA Q CA 1\n"
        errors = []
        with pytest.raises(ValueError):
            template_reader(
                parse_source(text, format="mmcif"), ["missing", "A"], on_error=lambda *event: errors.append(event)
            )
        assert errors == []

    def test_pdb_unsupported(self, template_reader):
        source = parse_source("END\n", format="pdb")
        errors = []
        with pytest.raises(UnsupportedFormatError):
            template_reader(source, ["A"], on_error=lambda *event: errors.append(event))
        assert errors == []

    def test_multiple_parents(self, cif_text, template_reader):
        text = cif_text.replace("MSE", "MOD").replace("MOD MET", "MOD 'SER,TYR,GLY'")
        source = parse_source(text, format="mmcif")
        result = template_reader(source, ["A"])["A"]
        if isinstance(result, Structure):
            assert result.component_parents(2) == ("SER", "TYR", "GLY")
        else:
            assert result.chains["A"].parent_component_ids == ((), (), ("SER", "TYR", "GLY"))

    @pytest.mark.parametrize("declaration", ["polymer-type", "parents"])
    def test_invalid_declarations(self, cif_text, template_reader, declaration):
        if declaration == "polymer-type":
            text = cif_text.replace("polypeptide(L)", "unsupported-polymer")
        else:
            text = cif_text.replace("MSE MET", "MSE 'MET,'")
        errors = []
        with pytest.raises(ValueError):
            template_reader(
                parse_source(text, format="mmcif"), ["missing", "A"], on_error=lambda *event: errors.append(event)
            )
        assert errors == []
