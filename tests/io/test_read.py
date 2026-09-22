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

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ofoldx.chemistry.definitions import StereoBasis, define_component, load_library, write_artifact
from ofoldx.chemistry.rdkit import StereoArbitration
from ofoldx.data.molecular import BondKind, DefinitionScope, MolecularSystem
from ofoldx.data.structure import CoordinateEvidence, CopyKind, OperationNumbering, Structure, replace_coordinates
from ofoldx.errors import ContradictionError, InsufficientInformationError, InvalidRequestError, SourceRef
from ofoldx.io import parse_source, read_structure, write_structure
from ofoldx.provenance import Basis, Resolution, known, state

from ..chemistry.builders import provenance

SSBOND = "SSBOND   1 CYS A    1    CYS B    1"  # blank symmetry is not stated
# a CRO chromophore (stereocentres CA1, CB1; a Z double bond) observed at one atom
CHROMOPHORE = """data_CRO
_entity.id 1
_entity.type non-polymer
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
HETATM 1 C CA2 CRO A 1 . 1.0 2.0 3.0 1 A
"""
ALANINE = CHROMOPHORE.replace("data_CRO", "data_ALA\n_chem_comp.id ALA").replace("CA2 CRO", "CA ALA")
# QQQ is defined in no scope (neither the subset nor the file): no component of the read has a definition
UNDEFINED = CHROMOPHORE.replace("CA2 CRO", "C1 QQQ") + "HETATM 2 O O1 QQQ A 1 . 2.0 2.0 3.0 1 A\n"
COVALE = """_struct_conn.id covale1
_struct_conn.conn_type_id covale
_struct_conn.ptnr1_label_asym_id A
_struct_conn.ptnr1_label_comp_id QQQ
_struct_conn.ptnr1_label_seq_id .
_struct_conn.ptnr1_label_atom_id C1
_struct_conn.ptnr2_label_asym_id A
_struct_conn.ptnr2_label_comp_id QQQ
_struct_conn.ptnr2_label_seq_id .
_struct_conn.ptnr2_label_atom_id O1
_struct_conn.ptnr1_symmetry 1_555
_struct_conn.ptnr2_symmetry 1_555
"""
CELL = """_cell.length_a 74.800
_cell.length_b 74.800
_cell.length_c 30.00
_cell.angle_alpha 90.00
_cell.angle_beta 90.00
_cell.angle_gamma 120.0
"""
FRACT = "".join(  # the cell's fractionalisation, as wwPDB prints it
    f"_atom_sites.fract_transf_matrix[{i + 1}][{j + 1}] {value}\n"
    for i, row in enumerate(("0.013369 0.007719 0.000000", "0.000000 0.015437 0.000000", "0.000000 0.000000 0.033333"))
    for j, value in enumerate(row.split())
) + "".join(f"_atom_sites.fract_transf_vector[{i}] 0.00000\n" for i in (1, 2, 3))
PLACEHOLDER = CELL.replace("74.800", "1.000").replace("30.00", "1.00").replace("120.0", "90.00")
EM = "_exptl.entry_id ALA\n_exptl.method 'ELECTRON MICROSCOPY'\n"
ASSEMBLY = """loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 'identity operation' 1 0 0 0 0 1 0 0 0 0 1 0
2 'point symmetry operation' 1 0 0 40 0 1 0 0 0 0 1 0
_pdbx_struct_assembly.id 1
_pdbx_struct_assembly_gen.assembly_id 1
_pdbx_struct_assembly_gen.oper_expression 1,2
_pdbx_struct_assembly_gen.asym_id_list A
"""
# 2zex: a triclinic cell, its fractionalisation printed to six decimals (off the cell's by 6 of its half-units), and
# crystal operator rows wwPDB computed from the cell: 1_556 as deposited, 1_655 one angstrom off
TRICLINIC = (
    """_cell.length_a 32.466
_cell.length_b 48.829
_cell.length_c 49.356
_cell.angle_alpha 62.03
_cell.angle_beta 84.88
_cell.angle_gamma 85.95
_symmetry.space_group_name_H-M 'P 1'
"""
    + "".join(
        f"_atom_sites.fract_transf_matrix[{i + 1}][{j + 1}] {value}\n"
        for i, row in enumerate(
            ("0.030801 -0.002180 -0.001974", "0.000000 0.020531 -0.010803", "0.000000 0.000000 0.022987")
        )
        for j, value in enumerate(row.split())
    )
    + "".join(f"_atom_sites.fract_transf_vector[{i}] 0.00000\n" for i in (1, 2, 3))
)
CRYSTAL_OPERATORS = """loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.name
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 'identity operation' 1_555 1 0 0 0 0 1 0 0 0 0 1 0
2 'crystal symmetry operation' 1_556 1.0000000000 0.0000000000 0.0000000000 4.4046269361
0.0000000000 1.0000000000 0.0000000000 22.8945042048 0.0000000000 0.0000000000 1.0000000000 43.5023640136
3 'crystal symmetry operation' 1_655 1.0000000000 0.0000000000 0.0000000000 33.4660000000
0.0000000000 1.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 1.0000000000 0.0000000000
loop_
_pdbx_struct_assembly.id
1
2
3
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 1 A
2 2 A
3 3 A
"""
# a ZN coordinated by its own 2_554 image in P 21 21 21, 3 A away under ITA's -x+1/2,-y,z+1/2 and 13 A under the
# table's x+1/2,-y+1/2,-z: the order PHENIX writes
SCREWED = """data_ZN
_entity.id 1
_entity.type non-polymer
_struct_asym.id B
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
HETATM 1 ZN ZN ZN B 1 . 2.500 0.000 0.600
_struct_conn.id metalc1
_struct_conn.conn_type_id metalc
_struct_conn.ptnr1_label_asym_id B
_struct_conn.ptnr1_label_comp_id ZN
_struct_conn.ptnr1_label_seq_id .
_struct_conn.ptnr1_label_atom_id ZN
_struct_conn.ptnr2_label_asym_id B
_struct_conn.ptnr2_label_comp_id ZN
_struct_conn.ptnr2_label_seq_id .
_struct_conn.ptnr2_label_atom_id ZN
_struct_conn.ptnr1_symmetry 1_555
_struct_conn.ptnr2_symmetry 2_554
_cell.length_a 10.0
_cell.length_b 20.0
_cell.length_c 6.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_space_group_symop.id
_space_group_symop.operation_xyz
1 x,y,z
2 x+1/2,-y+1/2,-z
3 -x,y+1/2,-z+1/2
4 -x+1/2,-y,z+1/2
"""
DECLARED, DERIVED = state(Resolution.KNOWN, Basis.DECLARED), state(Resolution.KNOWN, Basis.DERIVED)
IDENTITY = ("1.000000  0.000000  0.000000", "0.000000  1.000000  0.000000", "0.000000  0.000000  1.000000")


def disulfides(structure) -> NDArray[np.bool_]:
    r"""The edges between two observed sulfur atoms."""
    system = structure.system
    sulfur = np.flatnonzero((system.atomic_number == 16) & structure.coordinate_mask)
    first, second = system.edge_index
    return np.isin(first, sulfur) & np.isin(second, sulfur)


def names(structure, component: int) -> set[str]:
    system = structure.system
    start, stop = system.component_atom_ptr[component : component + 2]
    return set(system.text("atom", system.atom_symbol[start:stop]).tolist())


@pytest.fixture
def read(library):
    def read(text: str, *, format: str = "mmcif", **options):
        return read_structure(parse_source(text, format=format), library=library, **options)

    return read


class TestReadStructure:
    def test_junction_removes_leaving_atoms(self, read, dipeptide):
        structure, _ = read(dipeptide())
        assert not {"OXT", "HXT"} & names(structure, 0)
        assert "H2" not in names(structure, 1)
        assert {"OXT", "HXT"} <= names(structure, 1)

    def test_materialised_hydrogens_unobserved(self, read, dipeptide):
        structure, _ = read(dipeptide())
        hydrogen = structure.system.atomic_number == 1
        assert hydrogen.any()
        assert (structure.coordinate_evidence[hydrogen] == CoordinateEvidence.NONE).all()
        assert (structure.coordinate_evidence[~hydrogen] == CoordinateEvidence.OBSERVED).all()

    def test_report_names_value(self, read, dipeptide):
        structure, report = read(dipeptide())
        assert report.structure_digest == structure.digest()
        assert report.bulk["standard_junction_endpoint"] == 2
        assert report.source.format == "mmcif"

    def test_model_selected(self, read, dipeptide):
        first, _ = read(dipeptide(models=2))
        second, report = read(dipeptide(models=2), model=2)
        assert report.selection.model_number == 2
        assert not np.array_equal(first.coordinates, second.coordinates)

    def test_unknown_model_refused(self, read, dipeptide):
        with pytest.raises(InvalidRequestError) as raised:
            read(dipeptide(), model=9)
        assert raised.value.code == "model_unknown"

    @pytest.mark.parametrize("policy", ["Raise", "ignore", True])
    def test_unknown_policy_refused(self, read, disulfide, policy):  # never a silent "record"
        with pytest.raises(InvalidRequestError) as raised:
            read(disulfide, on_conflict=policy)
        assert raised.value.code == "value_invalid"

    def test_conflict_raises(self, read, disulfide):
        with pytest.raises(ContradictionError) as raised:
            read(disulfide)
        assert raised.value.code == "read_conflicts"
        assert len(raised.value.detail["report"].conflicts())

    def test_conflict_recorded(self, read, disulfide):
        structure, report = read(disulfide, on_conflict="record")
        assert "placed_leaving_atom" in report.summary()
        assert report.structure_digest == structure.digest()

    def test_restated_disulfide_counted_once(self, read, pdb_text):  # SSBOND and CONECT, one edge
        text = pdb_text.replace(SSBOND, f"{SSBOND:<59}{'1555':>6} {'1555':>6}")
        structure, _ = read(text, format="pdb", on_conflict="record")
        bonded = disulfides(structure)
        assert bonded.sum() == 1
        assert structure.system.bond_kind[bonded][0] == BondKind.COVALENT

    def test_doubled_conect_serial_refused(self, read, pdb_text):  # CONECT 1 names SG A and ZN
        with pytest.raises(ContradictionError) as raised:
            read(pdb_text.replace("HETATM    3 ZN", "HETATM    1 ZN"), format="pdb")
        assert raised.value.code == "key_invalid"

    def test_blank_symmetry_restatement_ambiguous(self, read, pdb_text):  # identity or image partner
        with pytest.raises(InsufficientInformationError) as raised:
            read(pdb_text, format="pdb")
        assert raised.value.code == "read_ambiguous_connections" and len(raised.value.candidates) == 2
        assert {ref.category for ref in raised.value.refs if isinstance(ref, SourceRef)} == {"SSBOND", "CONECT"}
        assert [record.code for record in raised.value.detail["connections"]] == ["ambiguous_connection"]
        _, report = read(pdb_text, format="pdb", on_conflict="record")
        assert report.summary()["ambiguous_connection"] == 1

    @pytest.mark.parametrize(
        ("symmetry", "code"), [("", "ambiguous_connection"), (f"{'1555':>6} {'2555':>6}", "conflicting_connection")]
    )
    def test_stated_disulfide_kept_once(self, read, pdb_text, symmetry, code):  # it is an edge in every reading
        stated, other = f"{SSBOND:<59}{'1555':>6} {'1555':>6}", f"{SSBOND.replace('SSBOND   1', 'SSBOND   2'):<59}"
        text = pdb_text.replace(SSBOND, f"{stated}\n{other}{symmetry}")
        structure, report = read(text, format="pdb", on_conflict="record")
        assert disulfides(structure).sum() == 1 and report.summary()[code] == 1

    def test_implausible_restatement_conflict(self, read, pdb_text):  # the 2555 image lies beyond any contact
        text = pdb_text.replace(SSBOND, f"{SSBOND:<59}{'1555':>6} {'2555':>6}")
        with pytest.raises(ContradictionError) as raised:
            read(text, format="pdb")
        assert raised.value.code == "read_conflicts"
        assert "contact_implausible" in raised.value.detail["report"].summary()

    @pytest.mark.parametrize("restated", [{"reverse": True}, {"ids": ("2", "1")}], ids=["reversed", "renumbered"])
    def test_restated_source_reads_same(self, read, zinc_site, restated):  # its identity row represents the edge
        def outcome(text):  # every column but the file's own operation names, and every finding
            structure, report = read(text)
            columns = columns_of(structure)
            del columns["operation_id"]
            return columns, dict(report.summary())

        assert outcome(zinc_site(**restated)) == outcome(zinc_site())

    @pytest.mark.parametrize(
        ("site", "numbering"),
        [
            pytest.param("RCSB", OperationNumbering.ITA, id="rcsb"),
            pytest.param("PDBE", OperationNumbering.ITA, id="pdbe"),
            pytest.param("?", OperationNumbering.FILE, id="unknown"),  # only a recognised archive site marks it
            pytest.param("XYZ", OperationNumbering.FILE, id="other"),
            pytest.param(None, OperationNumbering.FILE, id="user"),
        ],
    )
    def test_wwpdb_codes_read_by_ita(self, read, site, numbering):  # a user file keeps its table's ids
        status = f"_pdbx_database_status.process_site {site}\n" if site else ""
        structure, report = read(SCREWED + status, on_conflict="record")
        system, conflicts = structure.system, int(numbering == OperationNumbering.FILE)
        assert structure.operation_numbering.tolist() == [numbering]
        assert report.records.counts().get("contact_implausible", 0) == conflicts
        assert known(system.transform_state[system.external_transform]).all() == (not conflicts)

    @pytest.mark.parametrize(("assembly", "deviates"), [("1", 0), ("2", 0), ("3", 1)])
    def test_applied_operators_checked(self, read, assembly, deviates):  # 2zex: assembly 1 applies only 1_555
        _, report = read(ALANINE + TRICLINIC + CRYSTAL_OPERATORS, assembly=assembly, on_conflict="record")
        assert report.records.counts().get("operator_matrix_deviates", 0) == deviates

    def test_unparsed_operators_read_unknown(self, read, zinc_site):  # 1l6j: a trailing comma after every operator
        text = zinc_site().replace(" x,y,z\n", " x,y,z,\n").replace(" -x,y,-z\n", " -x,y,-z,\n")
        structure, report = read(text)  # 2_555 names an Unknown operation, but no reading bonds covalently
        assert structure.operation_numbering.tolist() == [OperationNumbering.NONE]
        assert report.summary()["operation_table_unparsable"] == 2 and report.summary()["ambiguous_connection"] == 1

    def test_setting_operations_not_stored(self, read, zinc_site):  # the space group's setting regenerates them
        structure, _ = read(zinc_site().split("loop_\n_space_group_symop")[0])
        assert structure.operation_numbering.tolist() == [OperationNumbering.ITA]
        assert len(structure.operation_id) == 0 and structure.symbols()["operation"] == ()
        assert known(structure.system.transform_state).all()  # 2_555 still names -x,y,-z

    @pytest.mark.parametrize(("connection", "edges"), [("", []), (COVALE, [[0, 1]])], ids=["none", "struct_conn"])
    def test_undefined_components_read_observed(self, read, connection, edges):  # the observed graph, facts Unknown
        structure, report = read(UNDEFINED + connection)
        system = structure.system
        assert system.text("atom", system.atom_symbol).tolist() == ["C1", "O1"]
        assert system.edge_index.T.tolist() == edges and not system.component_topology_complete.any()
        assert report.summary()["reference_incomplete_not_in_library"] == 1

    @pytest.mark.parametrize(("component", "findings"), [("ALA", 2), ("UNL", 0)])
    def test_extra_atom_finding_needs_atoms(self, read, component, findings):  # UNL: its reference_incomplete_absent
        _, report = read(UNDEFINED.replace("QQQ", component))
        assert report.summary().get("observed_atom_not_in_reference", 0) == findings

    def test_sequence_only_source_reads(self, read, dipeptide):  # no atom_site: every atom unobserved
        structure, _ = read(dipeptide().split("loop_\n_atom_site")[0])
        assert structure.system.text("component", structure.system.component_symbol).tolist() == ["GLY", "GLY"]
        assert (structure.coordinate_evidence == CoordinateEvidence.NONE).all()

    def test_sequence_only_pdb_reads(self, read):  # no ATOM/HETATM: as an mmCIF without atom_site
        structure, _ = read("SEQRES   1 A    2  GLY GLY\nEND\n", format="pdb")
        assert structure.system.text("component", structure.system.component_symbol).tolist() == ["GLY", "GLY"]
        assert (structure.coordinate_evidence == CoordinateEvidence.NONE).all()

    def test_residueless_asym_reads(self, read):  # an asym nothing gives residues: an instance without components
        structure, _ = read(CHROMOPHORE.split("loop_")[0])
        assert structure.system.instance_component_ptr.tolist() == [0, 0]

    def test_chain_from_observed_residue(self, read, dipeptide):  # residue 1 unobserved, no scheme names its chain
        head, tail = dipeptide().split("loop_\n_pdbx_poly_seq_scheme")
        text = head + tail[tail.index("loop_\n_atom_site") :]
        structure, _ = read("\n".join(line for line in text.splitlines() if " GLY A 1 1 ? " not in line))
        chain, chain_state = structure.instance_author_chain[0], structure.instance_author_chain_state[0]
        assert (structure.symbols()["chain"][chain], chain_state) == ("A", state(Resolution.KNOWN, Basis.DECLARED))

    @pytest.mark.parametrize(("stated", "unknown"), [("GLY GLY A", "GLY GLY ?"), ("GLY 1 1", "GLY ? 1")])
    def test_scheme_unknown_part_from_rows(self, read, dipeptide, stated, unknown):  # the rows state what `?` leaves
        structure, _ = read(dipeptide().replace(stated, unknown))
        assert structure.instance_author_chain_state.tolist() == [DECLARED]
        assert structure.component_author_number.tolist() == [1, 2]

    def test_author_number_as_source_reads(self, read, atom_line):  # "+1" (mmCIF) and hybrid-36 "A000" (PDB)
        signed, _ = read(ALANINE.replace(" 1 A\n", " +1 A\n"))
        hybrid, _ = read(atom_line(1, "ZN", "ZN", "A", "A000", element="ZN", record="HETATM") + "\nEND\n", format="pdb")
        numbers = [(s.component_author_number[0], s.component_author_number_state[0]) for s in (signed, hybrid)]
        assert numbers == [(1, DECLARED), (10000, DECLARED)]

    def test_blank_chain_assembly(self, read, atom_line):  # a REMARK 350 list naming no chain names the blank chain
        lines = ["REMARK 350 BIOMOLECULE: 1", "REMARK 350 APPLY THE FOLLOWING TO CHAINS:  "]
        lines += [f"REMARK 350   BIOMT{k}   1  {row}        0.00000" for k, row in enumerate(IDENTITY, 1)]
        structure, _ = read(
            "\n".join([*lines, atom_line(1, "CA", "GLY", " ", 1), "END"]) + "\n", format="pdb", assembly="1"
        )
        assert structure.instance_copy_kind.tolist() == [CopyKind.ASSEMBLY]

    def test_remark_465_number_kept(self, read, atom_line):
        lines = ["SEQRES   1 A    2  GLY GLY", "REMARK 465   M RES C SSSEQI", "REMARK 465     GLY A     7"]
        structure, _ = read("\n".join([*lines, atom_line(1, "CA", "GLY", "A", 8), "END"]) + "\n", format="pdb")
        texts = [structure.symbols()["author_number"][k] for k in structure.component_author_text]
        assert texts == ["7", "8"] and structure.component_author_number.tolist() == [7, 8]


def columns_of(structure) -> dict:
    r"""Every column, code columns as text, but the source document (a written file is a new source) and the definition
    scope (a file-scope definition equal to the archive's reads back at archive scope)."""
    values = {}
    for record in (structure.system, structure):
        for field in record.SCHEMA.fields:
            if field.name not in ("source_sha256", "component_definition_scope"):
                column = record.arrays()[field.name]
                symbols = np.asarray((*record.symbols()[field.symbols], ""), dtype=object) if field.symbols else None
                values[field.name] = (symbols[column] if field.symbols else column).tolist()
    return values


def with_system(structure, column: str, row: int, value) -> Structure:
    r"""``structure`` with one cell of its system changed."""
    arrays = structure.system.build()
    arrays.columns[column][row] = value
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())


@pytest.fixture
def reread(library, tmp_path):
    def reread(structure, report):
        write_structure(structure, tmp_path / "x.cif", definitions=report.definitions, library=library)
        return read_structure(tmp_path / "x.cif", library=library, on_conflict="record")

    return reread


class TestRoundTrip:
    """A native write → read keeps the graph, the knowledge states and their bases."""

    @pytest.fixture
    def library(self, ccd, tmp_path):  # ALA's CA parity inverted by its descriptor
        first = define_component(ccd["ALA"], scope=DefinitionScope.ARCHIVE)
        basis = np.array([StereoBasis.DESCRIPTOR_VERIFIED], np.uint8)
        parity = (3 - first.stereo_parity).astype(np.int8)
        arbitration = StereoArbitration(parity, first.stereo_neighbors, basis, ("CA",), (), True)
        verified = define_component(ccd["ALA"], scope=DefinitionScope.ARCHIVE, stereo=arbitration)
        write_artifact([verified], tmp_path / "ccd.safetensors", provenance("verified"))
        return load_library(tmp_path / "ccd.safetensors")

    @pytest.mark.parametrize("length_c", ["30.00", "3E2"])  # 3E2: a unit above the units digit
    def test_printed_resolution_restored(self, read, reread, length_c):  # 1sbs: 74.800 written 74.8 widened it
        structure, report = read(ALANINE.replace("1.0 2.0 3.0", "1.500 2.000 3.000") + CELL.replace("30.00", length_c))
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    @pytest.mark.parametrize(
        ("crystal", "frame"),
        [
            pytest.param(CELL, (Resolution.KNOWN, Basis.DERIVED), id="derived"),
            pytest.param(CELL + FRACT, (Resolution.KNOWN, Basis.DECLARED), id="declared"),
            pytest.param(PLACEHOLDER + EM, (Resolution.INAPPLICABLE, Basis.DEFINED), id="inapplicable"),
        ],
    )
    def test_frame_restored(self, read, reread, crystal, frame):  # the frame, its state and unit
        structure, report = read(ALANINE + crystal)
        assert structure.frame_state.tolist() == [state(*frame)]
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    def test_occupancy_su_restored(self, read, reread):  # 1.02 is kept by its su
        text = ALANINE.replace("_atom_site.Cartn_z\n", "_atom_site.Cartn_z\n_atom_site.occupancy\n")
        structure, report = read(text.replace("1.0 2.0 3.0", "1.0 2.0 3.0 1.02(1)"))
        row = structure.coordinate_mask  # CA, the one atom_site row
        assert (structure.occupancy[row].tolist(), structure.occupancy_uncertainty[row].tolist()) == ([1.02], [0.01])
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    def test_verified_parity_restored(self, read, reread):  # 1dct: a differing file scope over an arbitrated entry
        text = ALANINE.replace("_chem_comp.id ALA", "_chem_comp.id ALA\n_chem_comp.one_letter_code X")
        structure, report = read(text)
        again, again_report = reread(structure, report)
        assert again_report.definitions.by_id["ALA"].stereo_basis.tolist() == [StereoBasis.DESCRIPTOR_VERIFIED]
        assert columns_of(again) == columns_of(structure)

    def test_same_library_records_nothing(self, read, reread):
        structure, report = read(ALANINE)
        assert not len(reread(structure, report)[1].records.where(code="written_library_differs"))

    def test_other_library_names_both(self, read, library, ccd_artifact, tmp_path):  # 1dct reread without arbitration
        structure, report = read(ALANINE)
        write_structure(structure, tmp_path / "x.cif", definitions=report.definitions, library=library)
        other = load_library(ccd_artifact)
        _, again = read_structure(tmp_path / "x.cif", library=other, on_conflict="record")
        (found,) = again.records.where(code="written_library_differs")
        assert found.detail == {
            "written_version": "verified",
            "written_sha256": report.definitions.library.sha256,
            "content_version": "subset",
            "sha256": other.provenance.sha256,
        }


class TestPrivateCategories:
    """Each private category of a written file restores the facts it carries."""

    def test_fact_basis_restores_conflict(self, read, reread):
        structure, report = read(CHROMOPHORE + "_reflns.pdbx_ordinal 1\n_reflns.d_resolution_high 0\n")
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    def test_coordinate_evidence_restores_prediction(self, read, reread):
        structure, report = read(CHROMOPHORE)
        placed = structure.coordinate_evidence == CoordinateEvidence.OBSERVED
        atoms = np.flatnonzero(placed)
        predicted = replace_coordinates(
            structure, atoms, structure.coordinates[atoms], scope=placed, evidence=CoordinateEvidence.PREDICTED
        )[0]
        assert columns_of(reread(predicted, report)[0]) == columns_of(predicted)

    def test_instance_restores_assembly_copies(self, read, reread):
        structure, report = read(CHROMOPHORE + ASSEMBLY, assembly="1")
        assert (structure.instance_copy_kind == CopyKind.ASSEMBLY).all()
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    def test_instance_keeps_connections_open(self, read, reread, disulfide):  # 1gkt: the partner is not modelled
        structure, report = read("\n".join(line for line in disulfide.splitlines() if " B 1 1 " not in line) + "\n")
        assert not structure.system.instance_connections_complete.any()
        assert columns_of(reread(structure, report)[0]) == columns_of(structure)

    @pytest.mark.parametrize("column", ["stereocentre_state", "stereobond_state"])
    def test_configuration_restores_derived_state(self, read, reread, column):
        structure, report = read(CHROMOPHORE)
        derived = with_system(structure, column, 0, DERIVED)
        assert columns_of(reread(derived, report)[0]) == columns_of(derived)

    @pytest.mark.parametrize(
        "rows",
        [
            "_ofoldx_instance.asym_id\n_ofoldx_instance.copy_kind\n_ofoldx_instance.connections\n"
            "A deposited complete\nA crystal complete",
            "_ofoldx_atom_parity.asym_id\n_ofoldx_atom_parity.component\n_ofoldx_atom_parity.atom_id\n"
            f"_ofoldx_atom_parity.parity\n_ofoldx_atom_parity.state\nA 0 CA1 1 {DERIVED}\nA 0 CA1 2 {DERIVED}",
            "_ofoldx_instance.asym_id\n_ofoldx_instance.copy_kind\nA deposited\n",
            "_ofoldx_library.content_version\n_ofoldx_library.sha256\nsubset ?",
        ],
        ids=["instance-repeated", "atom_parity-repeated", "instance-without-connections", "library-without-sha256"],
    )
    def test_malformed_private_row_refused(self, read, rows):  # two statements of one fact, or an absent one
        with pytest.raises(ContradictionError) as raised:
            read(f"{CHROMOPHORE}loop_\n{rows}\n")
        assert raised.value.code == "private_row_invalid"
