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

"""Behaviour of ofoldx.io.write: every fact the format carries written, every other one reported, atomic publish."""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest

from ofoldx.chemistry.definitions import load_library, write_artifact
from ofoldx.data.molecular import AtomFate, MolecularSystem
from ofoldx.data.projection import project_atoms
from ofoldx.data.structure import CoordinateEvidence, OperationNumbering, Structure, replace_coordinates
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.io import parse_source, read_source, read_structure
from ofoldx.io.cif import read_cif
from ofoldx.io.write import write_structure
from ofoldx.provenance import Basis, Resolution, SubjectKind, state

from ..chemistry.builders import provenance

CHAINS = (("A", "1", ("ALA", "GLY")), ("B", "2", ("SER",)))  # polymers: PDB carries each of their facts (pdb=True)
# one ALA residue whose CB is observed
ALANINE = """data_AB
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
_entity_poly_seq.entity_id 1
_entity_poly_seq.num 1
_entity_poly_seq.mon_id ALA
_struct_asym.id A
_struct_asym.entity_id 1
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
ATOM 1 C CA ALA A 1 1 0.000 0.000 0.000 1 A
ATOM 2 C CB ALA A 1 1 1.530 0.000 0.000 1 A
"""
# file rows stating ALA's CA-CB bond AROM
AROMATIC = """loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
ALA CA C
ALA CB C
_chem_comp_bond.comp_id ALA
_chem_comp_bond.atom_id_1 CA
_chem_comp_bond.atom_id_2 CB
_chem_comp_bond.value_order AROM
"""
CELL = """_cell.length_a {0}
_cell.length_b {1}
_cell.length_c {2}
_cell.angle_alpha 90.00
_cell.angle_beta 90.00
_cell.angle_gamma 90.00
"""
# diag(0.1, 0.05, 0.0334): c is 29.94 Å beside a cell of 30.004 Å, beyond both printings' relaxed ranges
FRACT = "".join(
    f"_atom_sites.fract_transf_matrix[{i}][{j}] {value:.6f}\n"
    for i, row in enumerate(((0.1, 0, 0), (0, 0.05, 0), (0, 0, 0.0334)), 1)
    for j, value in enumerate(row, 1)
) + "".join(f"_atom_sites.fract_transf_vector[{i}] 0.00000\n" for i in (1, 2, 3))
# fract_transf = I, no cell: the placeholder stated through the matrix (1ho7)
IDENTITY = "".join(
    f"_atom_sites.fract_transf_matrix[{i}][{j}] {float(i == j):.6f}\n" for i in (1, 2, 3) for j in (1, 2, 3)
) + "".join(f"_atom_sites.fract_transf_vector[{i}] 0.00000\n" for i in (1, 2, 3))
NMR = "_exptl.entry_id AB\n_exptl.method 'SOLUTION NMR'\n"
# diag(10, 20, 30) printed to 0.01 Å, the frame stated as its inverse: c may be 30.004 Å beside the cell
CARTN = "".join(
    f"_atom_sites.Cartn_transf_matrix[{i}][{j}] {10 * i if i == j else 0:.2f}\n" for i in (1, 2, 3) for j in (1, 2, 3)
) + "".join(f"_atom_sites.Cartn_transf_vector[{i}] 0.00\n" for i in (1, 2, 3))


@pytest.fixture
def predicted(placed):
    structure = placed(*CHAINS, pdb=True)
    n = len(structure.coordinates)
    every = np.ones(n, bool)
    return replace_coordinates(
        structure, np.arange(n), structure.coordinates, scope=every, evidence=CoordinateEvidence.PREDICTED
    )[0]


class TestWriteStructure:
    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    def test_carried_value_reports_nothing(self, placed, tmp_path, name):
        report = write_structure(placed(*CHAINS, pdb=name == "x.pdb"), tmp_path / name)
        assert not len(report.records) and (tmp_path / name).exists()

    def test_uncarried_facts_are_reported(self, predicted, tmp_path):  # exact values printed without a source unit
        report = write_structure(predicted, tmp_path / "x.pdb")
        families = {record.detail["family"] for record in report.records}
        assert families == {"coordinate_evidence", "basis", "coordinate_quantum"}
        assert (tmp_path / "x.pdb").exists()

    @pytest.mark.parametrize(
        ("column", "row", "value", "names", "reported"),
        [
            ("instance_connections_complete", 0, False, ["x.pdb"], ("connections", ())),  # PDB has no record for it
            # an undetermined leaving state: no item of either format states one per atom
            ("atom_fate", 1, AtomFate.UNDETERMINED, ["x.cif", "x.pdb"], ("atom_fate", (1,))),
        ],
        ids=["open-connections", "undetermined-fate"],
    )
    def test_instance_state_reported(self, placed, tmp_path, column, row, value, names, reported):  # per instance
        for name in names:
            structure = placed(*CHAINS, pdb=name == "x.pdb")
            arrays = structure.system.build()
            arrays.columns[column][row] = value
            edited = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())
            report = write_structure(edited, tmp_path / name)
            assert [(r.detail["family"], r.detail["atoms"], r.subject) for r in report.records] == [
                (*reported, (SubjectKind.INSTANCE, 0))
            ]

    def test_refusal_writes_nothing(self, placed, tmp_path):  # a cut chain would name another chain
        with pytest.raises(UnsupportedError) as caught:
            write_structure(placed(*CHAINS), tmp_path / "x.pdb", chains={"A": "AB"})
        assert caught.value.code == "write_identifier_unfit" and not list(tmp_path.iterdir())

    def test_block_code_from_file_name(self, placed, tmp_path):
        write_structure(placed(*CHAINS), tmp_path / "my protein.cif")
        assert read_cif(tmp_path / "my protein.cif").blocks[0].name == "my_protein"

    def test_unknown_suffix(self, placed, tmp_path):
        with pytest.raises(InvalidRequestError) as caught:
            write_structure(placed(*CHAINS), tmp_path / "x.txt")
        assert caught.value.code == "format_unknown"

    def test_file_mode(self, placed, tmp_path):  # a new file as the umask leaves it, a replaced one keeps its mode
        (tmp_path / "kept.cif").write_text("")
        (tmp_path / "kept.cif").chmod(0o640)
        umask = os.umask(0o022)
        try:
            for name in ("new.cif", "kept.cif"):
                write_structure(placed(*CHAINS), tmp_path / name)
        finally:
            os.umask(umask)
        assert [stat.S_IMODE((tmp_path / name).stat().st_mode) for name in ("new.cif", "kept.cif")] == [0o644, 0o640]

    def test_unknown_format_refused(self, placed, tmp_path):  # never written as another format
        with pytest.raises(InvalidRequestError) as caught:
            write_structure(placed(*CHAINS), tmp_path / "x.cif", format="cif")
        assert caught.value.code == "format_unknown" and not list(tmp_path.iterdir())


class TestChargeCell:
    """The row's cell, never the effective charge; an edit's declared charge is the cell."""

    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    def test_declared_charge_written(self, placed, tmp_path, name):  # the source's cell `?`
        structure = placed(*CHAINS, pdb=name == "x.pdb")
        arrays = structure.system.build()
        arrays.columns["formal_charge"][0] = -1
        arrays.columns["formal_charge_state"][0] = state(Resolution.KNOWN, Basis.DECLARED)
        report = write_structure(
            Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build()), tmp_path / name
        )
        atoms = read_source(tmp_path / name).tables(1).atoms
        assert (atoms.charge[0], atoms.charge_state[0]) == (-1, state(Resolution.KNOWN, Basis.DECLARED))
        assert not len(report.records)

    @pytest.mark.parametrize(("name", "item"), [("x.cif", "_atom_site.pdbx_formal_charge"), ("x.pdb", "charge")])
    def test_stale_cell_reported(self, placed, tmp_path, name, item):  # it states a charge the atom no longer declares
        structure = placed(*CHAINS, pdb=name == "x.pdb")
        arrays = structure.build()
        arrays.columns["atom_site_charge"][0] = 1
        arrays.columns["atom_site_charge_state"][0] = state(Resolution.KNOWN, Basis.DECLARED)
        report = write_structure(Structure.from_arrays(structure.system, arrays), tmp_path / name)
        atoms = read_source(tmp_path / name).tables(1).atoms
        assert atoms.charge_state[0] == state(Resolution.UNKNOWN, Basis.DECLARED)
        assert [(record.detail["family"], record.detail["item"]) for record in report.records] == [("charge", item)]


class TestProjectedStructure:
    """A projection is written best effort; a removed atom is neither written back nor left."""

    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    def test_removed_atom_reads_back_unobserved(self, library, tmp_path, name):
        structure, _ = read_structure(parse_source(ALANINE, format="mmcif"), library=library)
        system = structure.system
        projected, _ = project_atoms(structure, system.text("atom", system.atom_symbol) != "CB")
        write_structure(projected, tmp_path / name)
        again, _ = read_structure(tmp_path / name, library=library)
        cb = again.system.text("atom", again.system.atom_symbol) == "CB"  # rebuilt from the CCD: not leaving
        assert cb.sum() == 1 and again.coordinate_evidence[cb][0] == CoordinateEvidence.NONE


class TestDefinitions:
    """A definition equal to the write library's entry for its id is the archive's, whatever its scope: neither
    written nor reported lost."""

    @pytest.mark.parametrize(
        ("row", "restated"),
        [
            ("", False),
            ("_chem_comp.one_letter_code X\n", True),
            (AROMATIC, True),  # over the archive's SING, written as AROM again
        ],
    )
    def test_restated_by_content(self, library, tmp_path, row, restated):  # the read gives each file scope
        text = ALANINE.replace("_struct_asym.id", f"_chem_comp.id ALA\n{row}_struct_asym.id")
        structure, report = read_structure(parse_source(text, format="mmcif"), library=library)
        cif = write_structure(structure, tmp_path / "x.cif", definitions=report.definitions, library=library)
        pdb = write_structure(structure, tmp_path / "x.pdb", library=library)
        again, _ = read_structure(tmp_path / "x.cif", library=library)
        lost = ["component_definition" in {r.detail["family"] for r in w.records} for w in (cif, pdb)]
        assert lost == [False, restated]
        assert ("chem_comp_atom" in read_cif(tmp_path / "x.cif").block(0).categories) == restated
        digests = [value.system.component_definition_digest.tolist() for value in (structure, again)]
        assert digests[0] == digests[1]

    @pytest.mark.parametrize("known_provenance", [True, False], ids=["different", "unknown"])
    def test_unnamed_library_writes_rows(self, library, archive, tmp_path, known_provenance):
        text = ALANINE.replace("_struct_asym.id", "_chem_comp.id ALA\n_struct_asym.id")
        structure, report = read_structure(parse_source(text, format="mmcif"), library=library)
        write_artifact([archive("ALA")], tmp_path / "ala.safetensors", provenance("ala"))
        other = load_library(tmp_path / "ala.safetensors")
        if not known_provenance:
            other.provenance = None
        write_structure(structure, tmp_path / "x.cif", definitions=report.definitions, library=other)
        assert "chem_comp_atom" in read_cif(tmp_path / "x.cif").block(0).categories

    @pytest.mark.parametrize(("entries", "lost"), [(("ALA",), False), (("GLY",), True)])
    def test_archive_scope_compared(self, library, archive, tmp_path, entries, lost):  # the write's library decides
        write_artifact([archive(cid) for cid in entries], tmp_path / "ccd.safetensors", provenance("other"))
        structure, _ = read_structure(parse_source(ALANINE, format="mmcif"), library=library)  # no rows: archive scope
        report = write_structure(structure, tmp_path / "x.pdb", library=load_library(tmp_path / "ccd.safetensors"))
        assert ("component_definition" in {r.detail["family"] for r in report.records}) == lost


class TestOperatorTable:
    """A table not in force is written as none and reported; no code is rebound."""

    def test_unparsed_table_not_written(self, library, zinc_site, tmp_path):  # one of its rows states no operation
        text = zinc_site().replace(" -x,y,-z\n", " -x,y,-z,\n")
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        report = write_structure(structure, tmp_path / "x.cif")
        written = read_cif(tmp_path / "x.cif").block(0).categories
        assert structure.operation_numbering.tolist() == [OperationNumbering.NONE]
        assert "space_group_symop" not in written and "symmetry" in written
        assert ("symmetry", "_space_group_symop") in {(r.detail["family"], r.detail["item"]) for r in report.records}
        assert "_555" not in (tmp_path / "x.cif").read_text()  # no symmetry code names an operation

    def test_complete_table_round_trips(self, library, zinc_site, tmp_path):
        structure, _ = read_structure(parse_source(zinc_site(), format="mmcif"), library=library)
        write_structure(structure, tmp_path / "x.cif")
        again, _ = read_structure(tmp_path / "x.cif", library=library)
        assert again.operation_numbering.tolist() == [OperationNumbering.FILE]
        assert np.array_equal(again.operation_rotation, structure.operation_rotation)


class TestFrame:
    """A frame a re-read of the written file would state otherwise is reported, never written so."""

    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    @pytest.mark.parametrize(
        "crystal",
        [
            pytest.param(CELL.format("10.000", "20.000", "30.004") + FRACT, id="re-derived-from-the-cell"),
            pytest.param(FRACT.replace("0.033400", "0.000000"), id="singular-without-cell"),  # reads back UNKNOWN
        ],
    )
    def test_conflicting_frame_reported(self, library, tmp_path, name, crystal):
        text = ALANINE + crystal
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        report = write_structure(structure, tmp_path / name)
        assert ("frame", "_atom_sites.fract_transf_matrix" if name == "x.cif" else "SCALEn") in {
            (record.detail["family"], record.detail["item"]) for record in report.records
        }

    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    @pytest.mark.parametrize("crystal", [CELL.format("1.000", "1.000", "1.000"), IDENTITY], ids=["cell", "transform"])
    def test_withdrawn_method_reported(self, library, tmp_path, name, crystal):  # the placeholder alone is a crystal
        text = ALANINE + crystal + NMR
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library)
        n = len(structure.coordinates)
        every = np.ones(n, bool)
        predicted, _ = replace_coordinates(
            structure, np.arange(n), structure.coordinates, scope=every, evidence=CoordinateEvidence.PREDICTED
        )
        report = write_structure(predicted, tmp_path / name)
        assert "frame" in {record.detail["family"] for record in report.records}

    @pytest.mark.parametrize("name", ["x.cif", "x.pdb"])
    @pytest.mark.parametrize("cell", ["", CELL.format("0.000", "0.000", "0.000")], ids=["1ho7", "2z33"])
    def test_placeholder_transform_round_trips(self, library, tmp_path, name, cell):  # the transform written back
        structure, _ = read_structure(parse_source(ALANINE + cell + IDENTITY + NMR, format="mmcif"), library=library)
        report = write_structure(structure, tmp_path / name)
        again, _ = read_structure(tmp_path / name, library=library)
        assert structure.frame_state.tolist() == [state(Resolution.INAPPLICABLE, Basis.DEFINED)]
        assert "frame" not in {record.detail["family"] for record in report.records}
        for column in ("cell", "cell_state", "frame", "frame_state", "frame_quantum"):
            assert np.array_equal(again.arrays()[column], structure.arrays()[column]), column

    @pytest.mark.parametrize(("name", "item"), [("x.cif", "_atom_sites.fract_transf_matrix"), ("x.pdb", "SCALEn")])
    def test_cartesian_resolution_reported(self, library, tmp_path, name, item):  # no format prints its units
        text = ALANINE + CELL.format("10.000", "20.000", "30.004") + CARTN
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library)
        report = write_structure(structure, tmp_path / name)
        families = {(record.detail["family"], record.detail["item"]) for record in report.records}
        assert ("frame", item) in families and structure.frame_quantum.all()
