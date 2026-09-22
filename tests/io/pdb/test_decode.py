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

"""Behaviour of ofoldx.io.pdb.decode: PDB records into the shared tables, author namespace only."""

import pytest

from ofoldx.errors import ContradictionError, SourceRef
from ofoldx.io.pdb.decode import decode_pdb, pdb_metadata, split_lines
from ofoldx.provenance import Basis, Resolution, state

KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
UNKNOWN = state(Resolution.UNKNOWN, Basis.DECLARED)
ABSENT = state(Resolution.UNKNOWN, Basis.NONE)


def decode(text: str, model: int = 1):
    return decode_pdb(split_lines(text.encode()), model)


class TestLines:
    @pytest.mark.parametrize("line", ["REMARK 999 " + "FREE TEXT " * 9, "label fa_atr fa_rep " + "0.5 " * 20])
    def test_unread_line_any_width(self, pdb_text, line):  # e.g. a Rosetta score table
        assert len(decode(f"{line}\n{pdb_text}").atoms.row) == 3

    def test_read_line_width_checked(self, pdb_text):
        line = "REMARK   2 RESOLUTION.    2.10 ANGSTROMS."
        with pytest.raises(ContradictionError, match="pdb_column"):
            decode(pdb_text.replace(line, line.ljust(80) + "X"))

    def test_unrecognised_lines_counted(self, pdb_text):  # a blank line states nothing
        text = "FOOBAR\n   \n" + pdb_text.replace("END\n", " ATOM shifted\nHE\nEND\n")
        records = decode(text).records.where(code="pdb_line_unrecognised")
        assert len(records) == 1 and records[0].detail == {"count": 3}
        assert records[0].sources == tuple(SourceRef("line", row) for row in (0, 24, 25))


class TestAtoms:
    def test_author_namespace_only(self, pdb_text):
        atoms = decode(pdb_text).atoms
        assert (atoms.label_asym_state == ABSENT).all()
        assert [atoms.symbols["chain"][c] for c in atoms.auth_asym] == ["A", "B", "A"]

    def test_blank_charge_unknown(self, pdb_text):
        atoms = decode(pdb_text).atoms
        assert atoms.charge.tolist()[:2] == [-1, 0]
        assert atoms.charge_state.tolist()[:2] == [KNOWN, UNKNOWN]

    def test_hybrid36_serial(self, pdb_text, atom_line):
        text = pdb_text.replace(
            atom_line(3, "ZN", "ZN", "A", 101, element="ZN", record="HETATM"),
            atom_line("A0000", "ZN", "ZN", "A", 101, element="ZN", record="HETATM"),
        )
        assert decode(text).atoms.serial.tolist() == [1, 2, 100000]

    def test_models(self, atom_line):
        body = atom_line(1, "CA", "ALA", "A", 1, element="C")
        text = f"MODEL        3\n{body}\nENDMDL\nMODEL        5\n{body}\nENDMDL\n"
        assert decode(text, 5).atoms.model.tolist() == [3, 5]

    def test_atom_outside_model_refused(self, atom_line):
        body = atom_line(1, "CA", "ALA", "A", 1, element="C")
        with pytest.raises(ContradictionError, match="pdb_column"):
            decode(f"MODEL        1\n{body}\nENDMDL\n{body}\n")


class TestConnections:
    def test_blank_symmetry_by_record(self, pdb_text):
        connections = decode(pdb_text).connections
        assert connections.symmetry_state.tolist() == [[ABSENT, ABSENT], [DEFINED, KNOWN]]

    def test_link_code_as_symop(self, pdb_text):
        tables = decode(pdb_text)
        assert tables.atoms.symbols["symmetry"][tables.connections.symmetry[1, 1]] == "1_555"

    def test_ssbond_names_sulfur(self, pdb_text):
        tables = decode(pdb_text)
        assert tables.atoms.symbols["atom"][tables.connections.auth_atom[0, 0]] == "SG"

    def test_conect_pairs(self, pdb_text):
        assert decode(pdb_text).conect.serial.tolist() == [[1, 2], [3, 1]]


class TestRecords:
    @pytest.mark.parametrize("operator", ["-X,Y+1/2,-Z", "-X, Y+1/2, -Z"])
    def test_remark_290_operations(self, pdb_text, operator):  # the operator as stated, never a dropped row
        crystal = decode(pdb_text.replace("-X,Y+1/2,-Z", operator)).crystal
        assert [crystal.symbols["operation_id"][c] for c in crystal.operation_id] == ["1", "2"]
        assert crystal.symbols["operation_text"][crystal.operation_text[1]] == operator

    def test_biomolecule_without_biomt_refused(self, pdb_text):
        remark = "REMARK 350 BIOMOLECULE: 1\nREMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
        with pytest.raises(ContradictionError, match="pdb_column"):
            decode(remark + pdb_text)

    def test_biomt_out_of_columns_refused(self, pdb_text):  # serial 2 is never silently dropped
        rows = ("1.000000  0.000000  0.000000", "0.000000  1.000000  0.000000", "0.000000  0.000000  1.000000")
        remark = "REMARK 350 BIOMOLECULE: 1\nREMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n" + "".join(
            f"REMARK 350   BIOMT{i}   1  {row}        0.00000\n" for i, row in enumerate(rows, 1)
        )
        with pytest.raises(ContradictionError, match="pdb_column"):
            decode(remark + "REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000 X\n" + pdb_text)

    def test_compound_chains(self, pdb_text):
        tables = decode(pdb_text)
        assert [tables.atoms.symbols["chain"][c] for c in tables.asyms.id] == ["A", "B"]
        assert len(set(tables.asyms.entity.tolist())) == 1

    def test_cryst1_restated(self, pdb_text):
        line = next(line for line in pdb_text.splitlines() if line.startswith("CRYST1"))
        assert decode(pdb_text.replace(line, f"{line}\n{line}")).crystal.cell.tolist() == [10, 20, 30, 90, 90, 90]
        with pytest.raises(ContradictionError, match="key_invalid"):
            decode(pdb_text.replace(line, f"{line}\n{line.replace('10.000', '99.000')}"))

    def test_expdta_restated(self, pdb_text):  # a continuation line continues; another record restates
        first = "EXPDTA    X-RAY DIFFRACTION; NEUTRON\nEXPDTA   2 DIFFRACTION\n"
        tables = decode(first + "EXPDTA    NEUTRON DIFFRACTION; X-RAY DIFFRACTION\n" + pdb_text)
        methods = [tables.atoms.symbols["text"][m] for m in tables.experiment.method if m >= 0]
        assert methods == ["X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"]
        with pytest.raises(ContradictionError, match="key_invalid"):
            decode(first + "EXPDTA    SOLUTION NMR\n" + pdb_text)

    def test_seqres_count_refused(self, pdb_text):
        with pytest.raises(ContradictionError, match="pdb_column"):
            decode(pdb_text.replace("SEQRES   1 A    2  CYS GLY", "SEQRES   1 A    3  CYS GLY"))

    def test_metadata(self, pdb_text):
        metadata = pdb_metadata(split_lines(pdb_text.encode()))
        assert (metadata.entry_id, metadata.release_date) == ("1ABC", "15-MAR-99")
        assert metadata.entities == (("1", "", "TINY PEPTIDE", None, None),)
