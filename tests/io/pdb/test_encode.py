# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

"""Behaviour of ofoldx.io.pdb.encode: records at the decoder's columns, and what PDB cannot carry."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.projection import select
from ofoldx.data.structure import OperationNumbering, Structure
from ofoldx.errors import UnsupportedError
from ofoldx.io import parse_source, read_structure
from ofoldx.io.pdb.decode import decode_pdb, split_lines
from ofoldx.io.pdb.encode import encode_pdb
from ofoldx.provenance import Basis, Resolution, SubjectKind, known, state

CHAINS = (("A", "1", ("ALA", "GLY")), ("B", "2", ("ZN",)))
PEPTIDE = CHAINS[:1]  # PDB carries each of its facts; a heterogen's entity it does not (COMPND names polymers)
# a polymer (entity 1) and a second entity in one author chain (mmCIF)
BESIDE = """data_HB
loop_
_entity.id
_entity.type
1 polymer
2 {kind}
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
ATOM 1 C CA GLY A 1 1 0.000 0.000 0.000 1 A
{row}
"""
ZINC = {"kind": "non-polymer", "row": "HETATM 2 ZN ZN ZN B 2 . 5.000 5.000 5.000 101 A"}
SERINE = {"kind": "polymer", "row": "ATOM 2 C CA SER B 2 1 5.000 5.000 5.000 11 A"}
WATERS = {"kind": "water", "row": "HETATM 2 O O HOH B 2 . 5.000 5.000 5.000 101 A"}
WATERS["row"] += "\nHETATM 3 O O HOH C 2 . 9.000 9.000 9.000 201 A"  # a second water instance of chain A
PEPTIDE_LINES = ("COMPND    MOL_ID: 1; CHAIN: A;", "SEQRES   1 A    1  GLY")
CRYST1 = "CRYST1   10.000   20.000   40.000  90.00  90.00  90.00 P 1           1"
SCALE = (  # [A | u], an origin shift u = (-1/4, 0, 0) included
    "SCALE1      0.100000  0.000000  0.000000       -0.25000",
    "SCALE2      0.000000  0.050000  0.000000        0.00000",
    "SCALE3      0.000000  0.000000  0.025000        0.00000",
)
PLACEHOLDER = (
    "EXPDTA    ELECTRON MICROSCOPY",
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 2 3         1",
)


def encode(structure: Structure, **options):
    return encode_pdb(structure, **{"identity": "source", "chains": None, **options})


def entities(structure: Structure) -> list[str]:
    return [structure.system.symbols()["entity"][e] for e in structure.system.entity_symbol]


def chainless(structure: Structure) -> Structure:
    arrays = structure.build()
    arrays.columns["instance_author_chain"][:] = -1
    arrays.columns["instance_author_chain_state"][:] = state(Resolution.UNKNOWN, Basis.NONE)
    return Structure.from_arrays(structure.system, arrays)


class TestEncodePdb:
    @pytest.mark.parametrize("basis", [Basis.DERIVED, Basis.DECLARED])  # a declared type re-reads as the same one
    def test_atoms_read_back(self, library, placed, basis):
        structure = placed(*PEPTIDE, pdb=True)
        system = structure.system.build()
        system.columns["entity_polymer_type_state"][:] = state(Resolution.KNOWN, basis)
        structure = Structure.from_arrays(MolecularSystem.from_arrays(system), structure.build())
        data, records = encode(structure, library=library)
        atoms = decode_pdb(split_lines(data), 1).atoms
        assert np.array_equal(atoms.coordinates, structure.coordinates)
        assert not len(records)

    def test_mmcif_identity_reported(self, placed):  # label asym, linkage and type states, 0.01 Å unit: per instance
        _, records = encode(placed(*PEPTIDE))
        chain = (SubjectKind.INSTANCE, 0)
        assert sorted((r.detail["family"], r.detail["item"], r.subject) for r in records) == [
            ("asym", "chainID", chain),
            ("coordinate_quantum", "x y z", chain),  # exact values: only their unit is lost, never `width`
            ("nstd_linkage", "SEQRES", chain),
            ("polymer_type", "SEQRES", chain),  # unstated, a re-read derives it from SEQRES
        ]

    def test_occupancy_su_reported(self, placed):  # PDB states no su: one record per instance, counting its atoms
        structure = placed(*PEPTIDE, ("B", "2", ("ALA",)), pdb=True)  # a lone GLY derives no hand
        arrays = structure.build()
        arrays.columns["occupancy_uncertainty"][:] = 0.01
        arrays.columns["occupancy_uncertainty_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        _, records = encode(Structure.from_arrays(structure.system, arrays))
        assert [(r.detail["family"], r.detail["item"], r.detail["count"], r.subject) for r in records] == [
            ("uncertainty", "occupancy", 4, (SubjectKind.INSTANCE, 0)),
            ("uncertainty", "occupancy", 2, (SubjectKind.INSTANCE, 1)),
        ]

    def test_declared_parent_written(self, parented):
        data, records = encode(parented(*PEPTIDE, pdb=True))
        assert decode_pdb(split_lines(data), 1).modified.seq.tolist() == [1]
        assert not len(records)

    @pytest.mark.parametrize("cell", [(Resolution.UNKNOWN, Basis.DECLARED), (Resolution.CONFLICT, Basis.NONE)])
    def test_unknown_cell_written_blank(self, placed, cell):
        structure = placed(*CHAINS)
        arrays = structure.build()
        arrays.columns["cell_state"][:] = state(*cell)
        data, _ = encode(Structure.from_arrays(structure.system, arrays))
        assert not known(decode_pdb(split_lines(data), 1).crystal.cell_state).any()

    @pytest.mark.parametrize(
        ("lines", "frame"),
        [
            pytest.param((CRYST1, *SCALE), (Resolution.KNOWN, Basis.DECLARED), id="declared"),
            pytest.param((CRYST1,), (Resolution.KNOWN, Basis.DERIVED), id="derived"),
            pytest.param(PLACEHOLDER, (Resolution.INAPPLICABLE, Basis.DEFINED), id="inapplicable"),
        ],
    )
    def test_frame_reread(self, library, atom_line, lines, frame):  # R(W(R(f))) ≃ R(f)
        def read(text):
            return read_structure(parse_source(text, format="pdb"), library=library)[0]

        structure = read("\n".join([*lines, *PEPTIDE_LINES, atom_line(1, "CA", "GLY", "A", 1), "END"]) + "\n")
        data, records = encode(structure)
        again = read(data.decode())
        assert again.frame_state.tolist() == structure.frame_state.tolist() == [state(*frame)] and not len(records)
        assert again.frame.tolist() == structure.frame.tolist() and again.cell.tolist() == structure.cell.tolist()

    @pytest.mark.parametrize(("chains", "lost"), [(None, []), (["A"], [0])])
    def test_polymer_type_reread(self, library, atom_line, chains, lost):  # one molecule, SEQRES for chain A only
        lines = ["COMPND    MOL_ID: 1; CHAIN: A, B;", "SEQRES   1 A    2  GLY ALA"]
        sites = (("GLY", "A", 1), ("ALA", "A", 2), ("GLY", "B", 1))
        lines += [atom_line(k + 1, "CA", *site) for k, site in enumerate(sites)] + ["END"]
        structure, _ = read_structure(parse_source("\n".join(lines) + "\n", format="pdb"), library=library)
        _, records = encode(structure if chains is None else select(structure, chains=chains), library=library)
        assert [r.subject[1] for r in records if r.detail["family"] == "polymer_type"] == lost

    @pytest.mark.parametrize(("unit", "kept"), [(1e-3, True), (0.1, False)])
    def test_cell_unit_reported(self, placed, unit, kept):  # 1hy0: 100.6 Å prints as 100.600, a finer unit
        structure = placed(*PEPTIDE)
        arrays = structure.build()
        arrays.columns["cell"][:] = (100.6, 100.6, 168.0, 90.0, 90.0, 120.0)
        arrays.columns["cell_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        arrays.columns["cell_quantum"][:] = [unit] * 3 + [0.01] * 3
        _, records = encode(Structure.from_arrays(structure.system, arrays))
        assert (("width", "CRYST1") in {(record.detail["family"], record.detail["item"]) for record in records}) != kept

    def test_group_without_cell_written(self, placed):
        structure = placed(*CHAINS)
        arrays = structure.build()
        arrays.columns["space_group"][:], arrays.symbols["space_group"] = 0, ("P 1",)
        arrays.columns["space_group_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        data, _ = encode(Structure.from_arrays(structure.system, arrays))
        assert known(decode_pdb(split_lines(data), 1).crystal.space_group_state)

    @pytest.mark.parametrize(("cell", "reported"), [(Basis.NONE, True), (Basis.DECLARED, False)])
    def test_blank_cell_basis_reported(self, placed, cell, reported):  # 2z33: a blank cell re-reads UNKNOWN×DECLARED
        structure = placed(*PEPTIDE, pdb=True)
        arrays = structure.build()
        arrays.columns["space_group"][:], arrays.symbols["space_group"] = 0, ("P 1",)
        arrays.columns["space_group_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        arrays.columns["operation_numbering"][:] = OperationNumbering.ITA
        arrays.columns["cell_state"][:] = state(Resolution.UNKNOWN, cell)
        _, records = encode(Structure.from_arrays(structure.system, arrays))
        assert [(r.detail["family"], r.detail["item"]) for r in records] == [("basis", "CRYST1")] * reported

    def test_blank_chain_written(self, library, atom_line):  # the reader's blank chain is the stated none
        lines = ["COMPND    MOL_ID: 1;", "COMPND   2 CHAIN: NULL;", "SEQRES   1      1  GLY"]
        text = "\n".join([*lines, atom_line(1, "CA", "GLY", " ", 1), "END"]) + "\n"
        structure = read_structure(parse_source(text, format="pdb"), library=library)[0]
        again = read_structure(parse_source(encode(structure)[0].decode(), format="pdb"), library=library)[0]
        blank = state(Resolution.INAPPLICABLE, Basis.DECLARED)
        assert structure.instance_author_chain_state.tolist() == again.instance_author_chain_state.tolist() == [blank]

    def test_missing_residue_number_five_columns(self, library, atom_line):  # REMARK 465 seqNum: columns 22-26
        lines = ["REMARK 465   M RES C SSSEQI", "REMARK 465     GLY A 10000", "COMPND    MOL_ID: 1;"]
        lines += ["COMPND   2 CHAIN: A;", "SEQRES   1 A    2  GLY GLY"]
        text = "\n".join([*lines, atom_line(1, "CA", "GLY", "A", 1), "END"]) + "\n"
        data, _ = encode(read_structure(parse_source(text, format="pdb"), library=library)[0])
        assert decode_pdb(split_lines(data), 1).unobserved.seq.tolist() == [10000]

    def test_missing_residue_in_model_1000(self, library, atom_line):  # REMARK 465's model field: three columns
        lines = ["REMARK 465   M RES C SSSEQI", "REMARK 465     GLY A    2", "COMPND    MOL_ID: 1;"]
        lines += ["COMPND   2 CHAIN: A;", "SEQRES   1 A    2  GLY GLY", "MODEL     1000"]
        text = "\n".join([*lines, atom_line(1, "CA", "GLY", "A", 1), "ENDMDL", "END"]) + "\n"
        structure = read_structure(parse_source(text, format="pdb"), library=library)[0]
        again = read_structure(parse_source(encode(structure)[0], format="pdb"), library=library)[0]
        assert again.component_author_number.tolist() == structure.component_author_number.tolist() == [1, 2]

    @pytest.mark.parametrize(
        ("column", "value", "item"),
        [("system_model_number", 10000, "MODEL serial"), ("formal_charge", 10, "charge")],
    )
    def test_wide_number_refused(self, placed, column, value, item):  # never cut to its columns
        structure = placed(*PEPTIDE)
        system, arrays = structure.system.build(), structure.build()
        columns = arrays.columns if column in arrays.columns else system.columns
        columns[column][:], columns[f"{column}_state"][:] = value, state(Resolution.KNOWN, Basis.DECLARED)
        with pytest.raises(UnsupportedError) as caught:
            encode(Structure.from_arrays(MolecularSystem.from_arrays(system), arrays))
        assert (caught.value.code, caught.value.detail["item"]) == ("write_identifier_unfit", item)

    def test_derived_model_number_reported(self, placed):  # PDB states a MODEL serial only as declared
        structure = placed(*PEPTIDE, pdb=True)
        arrays = structure.build()
        arrays.columns["system_model_number"][:] = 2
        arrays.columns["system_model_number_state"][:] = state(Resolution.KNOWN, Basis.DERIVED)
        _, records = encode(Structure.from_arrays(structure.system, arrays))
        assert [(record.detail["family"], record.detail["item"]) for record in records] == [("basis", "MODEL")]

    def test_unparsed_table_not_written(self, library, atom_line):
        link = f"LINK{'':8}{'SG':<4} CYS A   1{'':16}{'ZN':<4}  ZN A 101{'':3}{'1555':>6} {'2555':>6}"
        lines = ["CRYST1   10.000   20.000   30.000  90.00  90.00  90.00 P 1 21 1      2", "SEQRES   1 A    1  CYS"]
        lines += [
            "REMARK 290     NNNMMM   OPERATOR",
            "REMARK 290       1555   X,Y,Z",
            "REMARK 290       2555   -X,Y,-Z,",
        ]
        atoms = [atom_line(1, "SG", "CYS", "A", 1, element="S")]
        atoms += [atom_line(2, "ZN", "ZN", "A", 101, element="ZN", record="HETATM")]
        text = "\n".join([*lines, link, *atoms, "END"]) + "\n"
        data, records = encode(read_structure(parse_source(text, format="pdb"), library=library)[0])
        assert b"REMARK 290" not in data and b"P 1 21 1" in data and b"2555" not in data
        assert ("symmetry", "REMARK 290") in {(r.detail["family"], r.detail["item"]) for r in records}

    def test_inverted_table_link_kept(self, library, atom_line):  # its x,y,z is 2555: an end's own frame
        link = f"LINK{'':8}{'SG':<4} CYS A   1{'':16}{'ZN':<4}  ZN A 101{'':3}{'2555':>6} {'1555':>6}"
        lines = ["CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P -1", "SEQRES   1 A    1  CYS"]
        lines += ["REMARK 290     NNNMMM   OPERATOR", "REMARK 290       1555   -X,-Y,-Z"]
        lines += ["REMARK 290       2555   X,Y,Z"]
        atoms = [atom_line(1, "SG", "CYS", "A", 1, (1.0, 1.0, 1.0), element="S")]
        atoms += [atom_line(2, "ZN", "ZN", "A", 101, (-2.33, -2.33, -2.33), element="ZN", record="HETATM")]
        text = "\n".join([*lines, link, *atoms, "END"]) + "\n"
        structure = read_structure(parse_source(text, format="pdb"), library=library)[0]
        data, records = encode(structure)
        again = read_structure(parse_source(data, format="pdb"), library=library)[0]
        moves = [s.system.transform_rotation[s.system.external_transform].tolist() for s in (structure, again)]
        assert "external_connection" not in {r.detail["family"] for r in records}
        assert moves[0] == moves[1] == [(-np.eye(3)).tolist()] * 2  # SG and ZN each see the other inverted

    def test_hybrid36_number_written(self, library, atom_line):  # its text re-reads as its integer
        lines = ["COMPND    MOL_ID: 1;", "COMPND   2 CHAIN: A;", "SEQRES   1 A    1  GLY"]
        text = "\n".join([*lines, atom_line(1, "CA", "GLY", "A", "A000"), "END"]) + "\n"
        data, _ = encode(read_structure(parse_source(text, format="pdb"), library=library)[0])
        assert decode_pdb(split_lines(data), 1).atoms.auth_seq.tolist() == [10000]

    def test_derived_entities_written(self, library, atom_line):  # no COMPND: the reader derives them again
        def read(text):
            return read_structure(parse_source(text, format="pdb"), library=library)[0]

        zinc = atom_line(2, "ZN", "ZN", "A", 101, element="ZN", record="HETATM")
        structure = read("\n".join(["SEQRES   1 A    1  GLY", atom_line(1, "CA", "GLY", "A", 1), zinc, "END"]) + "\n")
        data, records = encode(structure)
        again = read(data)
        assert entities(again) == entities(structure)
        assert [record.detail["item"] for record in records] == ["COMPND MOL_ID"] * 2

    def test_heterogen_entity_reported(self, library):  # COMPND states a chain's polymer: the reader derives the ZN's
        def read(text, format):
            return read_structure(parse_source(text, format=format), library=library)[0]

        data, records = encode(read(BESIDE.format(**ZINC), "mmcif"))
        assert [record.detail["item"] for record in records].count("COMPND MOL_ID") == 1
        assert entities(read(data, "pdb")) == ["1", "A_ZN_101"]

    @pytest.mark.parametrize(
        ("other", "limit"), [(SERINE, "one polymer per chain"), (WATERS, "one water instance per chain")]
    )
    def test_shared_chain_refused(self, library, other, limit):  # re-read, a chain has one polymer, one water instance
        structure = read_structure(parse_source(BESIDE.format(**other), format="mmcif"), library=library)[0]
        with pytest.raises(UnsupportedError) as caught:
            encode(structure)
        assert (caught.value.code, caught.value.detail["limit"]) == ("write_identifier_unfit", limit)

    def test_shared_site_refused(self, placed):  # re-read, the second residue would be a candidate of the first's site
        with pytest.raises(UnsupportedError) as caught:
            encode(placed(*CHAINS), chains={"B": "A"})
        assert (caught.value.code, caught.value.detail["item"]) == ("write_identifier_unfit", "resSeq")

    def test_long_chain_refused(self, placed):  # cut, it would name another chain
        with pytest.raises(UnsupportedError) as caught:
            encode(placed(*CHAINS), chains={"A": "AB"})
        detail = caught.value.detail
        assert (caught.value.code, detail["item"], detail["value"], detail["limit"]) == (
            "write_identifier_unfit",
            "chainID",
            "AB",
            "1 character",
        )

    def test_unknown_chain_refused(self, placed):
        with pytest.raises(UnsupportedError) as caught:
            encode(chainless(placed(*CHAINS)))
        assert caught.value.code == "write_identity_unknown"

    def test_unknown_chain_assigned(self, placed):
        _, records = encode(chainless(placed(*CHAINS)), identity="assign")
        assert [record.detail["value"] for record in records if record.code == "identity_assigned"] == ["A", "B"]

    @pytest.mark.parametrize(("identity", "assigned"), [("source", []), ("assign", ["1", "2"])])
    def test_unknown_sequence_unstated(self, library, atom_line, identity, assigned):  # ATOM order states no sequence
        text = "\n".join([atom_line(1, "CA", "GLY", "A", 1), atom_line(2, "CA", "GLY", "A", 2), "END"]) + "\n"
        data, records = encode(read_structure(parse_source(text, format="pdb"), library=library)[0], identity=identity)
        assert [r.detail["value"] for r in records if r.code == "identity_assigned"] == assigned
        assert (b"SEQRES" in data) == bool(assigned)

    def test_blank_ssbond_written(self, library, atom_line):  # its unstated transform re-reads UNKNOWN×NONE
        lines = ["SEQRES   1 A    1  CYS", "SEQRES   1 B    1  CYS", "SSBOND   1 CYS A    1    CYS B    1"]
        atoms = [
            atom_line(1, "SG", "CYS", "A", 1, element="S"),
            atom_line(2, "SG", "CYS", "B", 1, (3.0, 2.0, 3.0), element="S"),
        ]

        def read(text):
            return read_structure(parse_source(text, format="pdb"), library=library)[0]

        structure = read("\n".join([*lines, *atoms, "END"]) + "\n")
        data, records = encode(structure)
        again = read(data)
        assert data.count(b"SSBOND") == 1 and "external_connection" not in {r.detail["family"] for r in records}
        assert (
            again.system.transform_state[again.system.external_transform].tolist()
            == [state(Resolution.UNKNOWN, Basis.NONE)] * 2
        )
