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

from ofoldx.chemistry.realize import valence_excess
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.stereo import StereoGroupKind
from ofoldx.io import parse_source, read_structure
from ofoldx.provenance import Basis, Resolution, SubjectKind, known, state

KNOWN, UNKNOWN = state(Resolution.KNOWN, Basis.DERIVED), state(Resolution.UNKNOWN, Basis.DERIVED)
# 1b9c: CRO (subset CCD parents THR, TYR, GLY) annotated GLY, TYR, GLY by the file
ANNOTATED_CRO = """data_CRO
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
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
HETATM 1 C CA2 CRO CRO A 1 . 1.0 2.0 3.0 1 A
loop_
_pdbx_struct_mod_residue.id
_pdbx_struct_mod_residue.auth_asym_id
_pdbx_struct_mod_residue.auth_seq_id
_pdbx_struct_mod_residue.auth_comp_id
_pdbx_struct_mod_residue.parent_comp_id
1 A 1 CRO GLY
2 A 1 CRO TYR
3 A 1 CRO GLY
"""
# IAS: the CCD lists no parent (?), the file annotates ASP
ANNOTATED_IAS = ANNOTATED_CRO.replace("CA2 CRO CRO", "CA IAS IAS").replace(
    "1 A 1 CRO GLY\n2 A 1 CRO TYR\n3 A 1 CRO GLY\n", "1 A 1 IAS ASP\n"
)


# a CYS SG linked to the N1 of an imine LIG (C/C=N/[H]): the stated N1=C2 configuration's reference H leaves
LINKED_IMINE = """data_IM
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
HETATM 1 C CB CYS A 1 . -1.810 0.000 0.000
HETATM 2 S SG CYS A 1 . 0.000 0.000 0.000
HETATM 3 N N1 LIG B 2 . 1.700 0.000 0.000
HETATM 4 C C2 LIG B 2 . 2.400 1.100 0.000
HETATM 5 C C1 LIG B 2 . 3.900 1.100 0.000
_struct_conn.id covale1
_struct_conn.conn_type_id covale
_struct_conn.ptnr1_label_asym_id A
_struct_conn.ptnr1_label_comp_id CYS
_struct_conn.ptnr1_label_seq_id .
_struct_conn.ptnr1_label_atom_id SG
_struct_conn.ptnr2_label_asym_id B
_struct_conn.ptnr2_label_comp_id LIG
_struct_conn.ptnr2_label_seq_id .
_struct_conn.ptnr2_label_atom_id N1
"""


def carbon_bond(system) -> int:
    first, second = system.edge_index
    return int(np.flatnonzero((system.atomic_number[first] == 6) & (system.atomic_number[second] == 6))[0])


class TestValenceExcess:
    def test_balanced_value_zero(self, ligand):
        result = valence_excess(ligand("OCC"))
        assert set(result.state.tolist()) == {KNOWN}
        assert not result.excess.any()

    def test_raised_order_positive(self, ligand, edit):
        system = ligand("OCC")
        raised = valence_excess(edit(system, bond_order=(carbon_bond(system), 2)))
        carbons = np.flatnonzero(system.atomic_number == 6)
        assert raised.excess[np.isin(raised.atom, carbons)].tolist() == [1, 1]
        assert set(raised.state.tolist()) == {KNOWN}

    def test_unknown_order_unknown(self, ligand, edit):
        system = ligand("OCC")
        result = valence_excess(
            edit(
                system,
                bond_order=(carbon_bond(system), 0),
                bond_order_state=(carbon_bond(system), state(Resolution.UNKNOWN, Basis.NONE)),
            )
        )
        carbons = np.isin(result.atom, np.flatnonzero(system.atomic_number == 6))
        assert set(result.state[carbons].tolist()) == {UNKNOWN}
        assert not result.excess[carbons].any()


class TestRealizeGraph:
    def test_hydrogens_are_atoms(self, ligand):
        system = ligand("OCC")
        assert known(system.implicit_hydrogens_state).all() and not system.implicit_hydrogens.any()
        assert system.component_hydrogens_complete.all() and system.component_topology_complete.all()

    def test_stated_parity_declared(self, ligand):
        system = ligand("N[C@@H](C)C(=O)O")
        assert system.stereocentre_state.tolist() == [state(Resolution.KNOWN, Basis.DECLARED)]

    def test_stated_cis_trans_declared(self, ligand):
        system = ligand("C/C=C/C")
        assert system.stereobond_state.tolist() == [state(Resolution.KNOWN, Basis.DECLARED)]

    def test_stereo_group_lowered(self, ligand):
        system = ligand("C[C@H](N)[C@@H](C)O |&1:1,3|")
        assert system.stereogroup_kind.tolist() == [StereoGroupKind.AND]
        assert sorted(system.stereogroup_member.tolist()) == [0, 1]

    def test_cleared_stereobond_recorded(self, library, archive):  # the link takes N1's reference H
        imine = read_smiles("C/C=N/[H]", component_id="LIG").component
        source = parse_source(LINKED_IMINE, format="mmcif")
        structure, report = read_structure(source, library=library([archive("CYS")]), components={"LIG": imine})
        cleared = report.records.where(code="parity_cleared_heavy_neighbour_removed")
        assert cleared.subject_kind.tolist() == [SubjectKind.EDGE]
        assert structure.system.stereobond_state.tolist() == [UNKNOWN]


class TestParents:
    @pytest.fixture
    def annotated(self, library, archive):
        return read_structure(parse_source(ANNOTATED_CRO, format="mmcif"), library=library([archive("CRO")]))

    def test_mismatch_is_a_finding(self, annotated):
        _, report = annotated
        assert report.summary()["parent_annotation_mismatch"] == 1
        assert not len(report.conflicts())

    @pytest.mark.parametrize(
        ("chem_comp", "definition"),
        [
            ("", ["artifact"]),
            ("_chem_comp.id CRO\n_chem_comp.mon_nstd_parent_comp_id GLY,THR,TYR\n", ["chem_comp", "artifact"]),
        ],
        ids=["archive", "file"],
    )
    def test_mismatch_names_both_statements(self, library, archive, chem_comp, definition):
        source = parse_source(ANNOTATED_CRO + chem_comp, format="mmcif")
        _, report = read_structure(source, library=library([archive("CRO")]))
        record = report.records.where(code="parent_annotation_mismatch")[0]
        assert record.detail == {"annotation": "GLY,TYR", "definition": "GLY,THR,TYR"}
        assert [ref.category for ref in record.sources] == ["pdbx_struct_mod_residue"] * 3 + definition

    def test_both_scopes_kept(self, annotated):
        system = annotated[0].system
        assert system.text("parent", system.component_parent).tolist() == ["GLY", "TYR"]
        assert system.text("parent", system.component_definition_parent).tolist() == ["GLY", "THR", "TYR"]

    def test_unlisted_parent_no_finding(self, library, archive):
        structure, report = read_structure(
            parse_source(ANNOTATED_IAS, format="mmcif"), library=library([archive("IAS")])
        )
        assert "parent_annotation_mismatch" not in report.summary()
        assert structure.system.text("parent", structure.system.component_parent).tolist() == ["ASP"]
