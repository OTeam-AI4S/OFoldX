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

"""Behaviour of ofoldx.io.mmcif.decode: mmCIF items into the shared tables through the codec rows."""

import re

import numpy as np
import pytest

from ofoldx.chemistry.declarations import OrderToken
from ofoldx.errors import ContradictionError
from ofoldx.io.cif import parse_cif
from ofoldx.io.mmcif.decode import decode_components, decode_mmcif, mmcif_metadata
from ofoldx.provenance import Basis, Resolution, state

KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
DEFAULT = state(Resolution.KNOWN, Basis.DEFAULT)
DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
UNKNOWN = state(Resolution.UNKNOWN, Basis.DECLARED)
ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
NO_VALUE = state(Resolution.INAPPLICABLE, Basis.DEFINED)
DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
UNREAD = state(Resolution.UNKNOWN, Basis.DERIVED)
# one _ofoldx_fact_basis row for the first partner's symmetry code of the struct_conn row
BASIS = """loop_
_ofoldx_fact_basis.category
_ofoldx_fact_basis.row
_ofoldx_fact_basis.item
_ofoldx_fact_basis.state
struct_conn 0 ptnr1_symmetry {state}
"""
# one NAG 2 C1 – NAG 1 O4 link of entity 1, its value_order {order}
BRANCH_LINK = """loop_
_pdbx_entity_branch_link.link_id
_pdbx_entity_branch_link.entity_id
_pdbx_entity_branch_link.entity_branch_list_num_1
_pdbx_entity_branch_link.comp_id_1
_pdbx_entity_branch_link.atom_id_1
_pdbx_entity_branch_link.leaving_atom_id_1
_pdbx_entity_branch_link.entity_branch_list_num_2
_pdbx_entity_branch_link.comp_id_2
_pdbx_entity_branch_link.atom_id_2
_pdbx_entity_branch_link.leaving_atom_id_2
_pdbx_entity_branch_link.value_order
1 1 2 NAG C1 O1 1 NAG O4 HO4 {order}
"""

GLY = """data_GLY
_chem_comp.id GLY
_chem_comp.type 'PEPTIDE LINKING'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_model_Cartn_x_ideal
GLY N N 1.931
GLY CA C 0.761
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
GLY N CA
"""


def decode(text: str, model: int = 1):
    block = parse_cif(text).blocks[0]
    return decode_mmcif(block.categories, model, block=block.name)


def symbol(tables, family: str, code: int) -> str:
    return tables.atoms.symbols[family][code]


def first_occupancy(text: str, occupancy: str, esd: str | None = None) -> str:
    r"""The first atom row's occupancy replaced; with ``esd``, its ``occupancy_esd`` stated (the other rows' ``?``)."""
    text = text.replace("2.000 3.000 1.00", f"2.000 3.000 {occupancy}")
    if esd is None:
        return text
    text = text.replace("_atom_site.pdbx_PDB_model_num\n", "_atom_site.pdbx_PDB_model_num\n_atom_site.occupancy_esd\n")
    rows = iter([esd])
    return re.sub(r"^((?:ATOM|HETATM) .*)$", lambda row: f"{row[1]} {next(rows, '?')}", text, flags=re.M)


class TestAtoms:
    def test_namespaces_kept_apart(self, mmcif_text):
        atoms = decode(mmcif_text).atoms
        assert symbol(decode(mmcif_text), "chain", atoms.label_asym[0]) == "A"
        assert symbol(decode(mmcif_text), "chain", atoms.auth_asym[0]) == "P"
        assert atoms.auth_seq[0] == 10 and atoms.label_seq[0] == 1

    def test_absent_charge_defaults(self, mmcif_text):
        atoms = decode(mmcif_text).atoms
        assert (atoms.charge_state == DEFAULT).all() and (atoms.charge == 0).all()

    def test_stated_unknown_charge(self, mmcif_text):
        text = mmcif_text.replace(
            "_atom_site.pdbx_PDB_model_num", "_atom_site.pdbx_PDB_model_num\n_atom_site.pdbx_formal_charge"
        )
        text = "\n".join(line + " ?" if line.startswith(("ATOM ", "HETATM ")) else line for line in text.splitlines())
        assert (decode(text).atoms.charge_state == UNKNOWN).all()

    def test_deuterium_is_hydrogen(self, mmcif_text):
        atoms = decode(mmcif_text).atoms
        assert (atoms.element[1], atoms.isotope[1]) == (1, 2)

    @pytest.mark.parametrize(
        ("token", "reading"),
        [("0000000000000000101", (101, KNOWN)), ("9223372036854775807", (2**63 - 1, KNOWN)), ("9" * 19, (0, UNREAD))],
    )
    def test_author_number_within_int64(self, mmcif_text, token, reading):  # cif.parse_integers' grammar and range
        atoms = decode(mmcif_text.replace("20.0 101 ZN P ZN 1", f"20.0 {token} ZN P ZN 1")).atoms
        assert (atoms.auth_seq[2], atoms.auth_seq_state[2]) == reading

    def test_insertion_question_mark(self, mmcif_text):
        assert (decode(mmcif_text).atoms.insertion_state == NO_VALUE).all()

    def test_null_model_refused(self, mmcif_text):
        with pytest.raises(ContradictionError, match="model_number_null"):
            decode(mmcif_text.replace(" P ZN 2\n", " P ZN ?\n"))


class TestConnections:
    def test_symmetry_states(self, mmcif_text):
        connections = decode(mmcif_text).connections
        assert connections.symmetry_state[0].tolist() == [KNOWN, ABSENT]
        stated = decode(mmcif_text.replace("ZN 1_555\n", "ZN .\n")).connections
        assert stated.symmetry_state[0, 0] == DEFINED

    def test_absent_order_defaults(self, mmcif_text):
        connections = decode(mmcif_text).connections
        assert (connections.order[0], connections.order_state[0]) == (1, DEFAULT)

    def test_unlisted_type_recorded(self, mmcif_text):
        tables = decode(mmcif_text.replace("metalc1 metalc", "x1 bond"))
        assert tables.records.counts() == {"conn_type_enumeration": 1}


class TestExperiment:
    def test_nonpositive_resolution_unknown(self, mmcif_text):  # a finding, the read never refused
        tables = decode(mmcif_text + "_refine.pdbx_refine_id 'X-RAY DIFFRACTION'\n_refine.ls_d_res_high 0\n")
        assert tables.experiment.resolution_state.tolist() == [UNKNOWN]
        assert tables.records.counts() == {"resolution_invalid": 1}


class TestBranchLinks:
    @pytest.mark.parametrize(
        ("token", "order", "state"),
        [("sing", OrderToken.SING, KNOWN), ("arom", OrderToken.AROM, KNOWN), ("?", 0, UNKNOWN)],
    )
    def test_value_order(self, mmcif_text, token, order, state):  # arom, delo, pi, poly: stated, no order
        links = decode(mmcif_text + BRANCH_LINK.format(order=token)).branch_links
        assert (links.order[0], links.order_state[0]) == (order, state)


class TestFactBasis:
    @pytest.mark.parametrize(("token", "restored"), [("1_555", DERIVED), ("?", CONFLICT)])
    def test_token_state_restored(self, mmcif_text, token, restored):
        text = mmcif_text.replace("ZN 1_555\n", f"ZN {token}\n") + BASIS.format(state=restored)
        assert decode(text).connections.symmetry_state[0, 0] == restored

    @pytest.mark.parametrize(("token", "restored"), [("1_555", CONFLICT), ("?", DERIVED)])
    def test_state_unfit_token_refused(self, mmcif_text, token, restored):  # KNOWN exactly on a value
        with pytest.raises(ContradictionError, match="private_row_invalid"):
            decode(mmcif_text.replace("ZN 1_555\n", f"ZN {token}\n") + BASIS.format(state=restored))

    def test_model_number_row_refused(self, mmcif_text):  # one fact of the system, no row state
        row = BASIS.replace("struct_conn 0 ptnr1_symmetry", "atom_site 5 pdbx_PDB_model_num")
        with pytest.raises(ContradictionError, match="private_row_invalid"):
            decode(mmcif_text + row.format(state=DERIVED), model=2)

    def test_restored_conflict_recorded(self, mmcif_text):
        tables = decode(mmcif_text.replace("ZN 1_555\n", "ZN ?\n") + BASIS.format(state=CONFLICT))
        assert tables.records.counts() == {"conflict_restored": 1}


class TestRefusals:
    def test_entity_type_enumeration(self, mmcif_text):
        with pytest.raises(ContradictionError, match="entity_type_enumeration"):
            decode(mmcif_text.replace("2 non-polymer", "2 ligand"))

    def test_conn_aromatic_order_refused(self, mmcif_text):
        text = mmcif_text.replace(
            "_struct_conn.ptnr1_symmetry\n", "_struct_conn.ptnr1_symmetry\n_struct_conn.pdbx_value_order\n"
        )
        with pytest.raises(ContradictionError, match="item_enumeration"):
            decode(text.replace("ZN 1_555\n", "ZN 1_555 arom\n"))

    @pytest.mark.parametrize(
        "rows",
        [
            "_pdbx_struct_oper_list.id\n1\n1",
            "_pdbx_struct_oper_list.id\n?",
            "_pdbx_struct_assembly_gen.assembly_id\n_pdbx_struct_assembly_gen.oper_expression\n"
            "_pdbx_struct_assembly_gen.asym_id_list\n? 1 A",
        ],
    )
    def test_assembly_key_invalid(self, mmcif_text, rows):
        with pytest.raises(ContradictionError, match="key_invalid"):
            decode(f"{mmcif_text}loop_\n{rows}\n")

    @pytest.mark.parametrize(
        ("occupancy", "esd", "uncertainty"),
        [("1.05(1)", None, 0.01), ("1.05", "0.01", 0.01), ("-0.04(1)", None, 0.01), ("1.41", None, None)],  # 1btw: 1.41
    )
    def test_occupancy_out_of_range(self, mmcif_text, occupancy, esd, uncertainty):
        with pytest.raises(ContradictionError) as caught:
            decode(first_occupancy(mmcif_text, occupancy, esd))
        assert caught.value.code == "occupancy_out_of_range"
        assert caught.value.detail == {"value": float(occupancy.split("(")[0]), "uncertainty": uncertainty}

    @pytest.mark.parametrize(
        ("occupancy", "esd", "uncertainty"),
        [
            ("1.02(1)", None, 0.01),
            ("1.02", "0.01", 0.01),
            ("1.02(1)", "0.01", 0.01),
            ("1.03(1)", None, 0.01),  # at the bound
            ("1.36(12)", None, 0.12),  # at the bound, which 1 + 3 * 0.12 misses in floats
            ("-0.02(1)", None, 0.01),
        ],
    )
    def test_occupancy_within_three_uncertainties(self, mmcif_text, occupancy, esd, uncertainty):
        atoms = decode(first_occupancy(mmcif_text, occupancy, esd)).atoms
        assert (atoms.occupancy[0], atoms.occupancy_uncertainty[0]) == (float(occupancy.split("(")[0]), uncertainty)

    @pytest.mark.parametrize(("esd", "expected"), [(None, ABSENT), ("?", UNKNOWN), ("0", KNOWN)])
    def test_unstated_su_stays_unknown(self, mmcif_text, esd, expected):
        assert decode(first_occupancy(mmcif_text, "1.00", esd)).atoms.occupancy_uncertainty_state[0] == expected

    def test_two_different_su_refused(self, mmcif_text):
        with pytest.raises(ContradictionError, match="occupancy_uncertainty_conflict"):
            decode(first_occupancy(mmcif_text, "1.02(1)", "0.02"))

    @pytest.mark.parametrize("occupancy", ["?", ".", "0.50"])  # an unknown occupancy has no range; its su is still one
    def test_negative_su_refused(self, mmcif_text, occupancy):
        with pytest.raises(ContradictionError) as caught:
            decode(first_occupancy(mmcif_text, occupancy, "-0.01"))
        assert (caught.value.code, caught.value.detail) == ("occupancy_uncertainty_negative", {"value": -0.01})


class TestCrystal:
    def test_cell_and_operations(self, mmcif_text):
        tables = decode(mmcif_text)
        crystal = tables.crystal
        np.testing.assert_allclose(crystal.cell_quantum, [0.05, 0.05, 0.005, 0.05, 0.05, 0.05])
        assert crystal.symbols["space_group"][crystal.space_group[0]] == "P 1 21 1"
        assert [crystal.symbols["operation_text"][c] for c in crystal.operation_text] == ["x,y,z", "-x,y+1/2,-z"]

    def test_every_statement_kept(self, mmcif_text):  # neither category wins by being present
        name = "_symmetry.space_group_name_H-M 'P 1 21 1'"
        text = mmcif_text.replace(name, "_symmetry.space_group_name_H-M ?\n_space_group.name_H-M_alt 'P 1 21 1'")
        crystal = decode(text + "loop_\n_symmetry_equiv.id\n_symmetry_equiv.pos_as_xyz\n2 -x,y+1/2,-z\n").crystal
        assert crystal.space_group_state.tolist() == [UNKNOWN, KNOWN]
        assert [crystal.symbols["operation_id"][c] for c in crystal.operation_id] == ["1", "2", "2"]


class TestComponents:
    @pytest.fixture
    def glycine(self):
        block = parse_cif(GLY).blocks[0]
        return decode_components(block.categories, block=block.name).components["GLY"]

    def test_items_by_atom(self, glycine):
        assert glycine.component["id"] == ("GLY", KNOWN)
        assert glycine.atoms["atom_id"][0].tolist() == [b"N", b"CA"]
        assert glycine.element[0].tolist() == [7, 6]

    def test_verified_defaults_filled(self, glycine):
        charge, charge_state, _ = glycine.numbers["charge"]
        assert charge.tolist() == [0, 0] and (charge_state == DEFAULT).all()
        order, order_state = glycine.bonds["value_order"]
        assert order.tolist() == [b"sing"] and order_state.tolist() == [DEFAULT]


class TestMetadata:
    def test_document_facts(self, mmcif_text):
        metadata = mmcif_metadata(parse_cif(mmcif_text).blocks[0].categories)
        assert (metadata.entry_id, metadata.release_date) == ("TEST", "2001-02-03")
        assert metadata.entities[0] == ("1", "polymer", "tiny peptide", "polypeptide(L)", None)
