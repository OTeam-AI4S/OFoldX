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

"""Behaviour of ofoldx.io.mmcif.encode: the tokens written read back as the facts and states of the value."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.io import parse_source, read_structure
from ofoldx.io.cif import TokenState, parse_cif
from ofoldx.io.mmcif.encode import encode_mmcif, mmcif_capability
from ofoldx.io.pdb.encode import encode_pdb
from ofoldx.provenance import Basis, Resolution, state

CHAINS = (("A", "1", ("ALA", "GLY")), ("B", "2", ("ZN",)))
# NAG 2 C1 – NAG 1 O4 of one branched asym, the link stating aromatic (no integer order)
AROMATIC_LINK = """data_BR
_entity.id 1
_entity.type branched
_struct_asym.id B
_struct_asym.entity_id 1
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
B 1 NAG 1 B NAG 1
B 1 NAG 2 B NAG 2
loop_
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
1 1 2 NAG C1 O1 1 NAG O4 HO4 arom
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
HETATM 1 O O4 NAG B 1 . 7.718 25.767 23.375 1 B
HETATM 2 C C1 NAG B 1 . 7.460 25.550 24.824 2 B
"""
LINK = "LINK         C   GLY A   1                 N   GLY A   2   1555   1555  1.33\n"  # restating a junction
KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
ONE_SIDED = {"atom": 0, "partner_instance": 1, "partner_component": 0, "partner_atom": 0, "transform": 1}
ONE_SIDED |= {"order": 0, "aromatic": False, "kind": 2, "direction": 0, "origin": 3, "connection_type": -1}


def categories(structure: Structure):
    data, records = encode_mmcif(structure, block="test", identity="source")
    return parse_cif(data).block(0).categories, records


def texts(category, *items: str) -> list[tuple[str, ...]]:
    columns = [category.text(item)[0].astype(str) for item in items]
    return list(zip(*columns))


def one_sided(structure: Structure) -> Structure:
    r"""``structure`` whose first atom is coordinated to instance 1's first atom under a placement no table row holds,
    and instance 1 names no connection back."""
    arrays = structure.system.build()
    c, unknown = arrays.columns, state(Resolution.UNKNOWN, Basis.NONE)
    c["transform_rotation"] = np.stack((c["transform_rotation"][0], np.zeros((3, 3), np.int8)))
    c["transform_translation"] = np.zeros((2, 3), np.int16)
    c["transform_code"], c["transform_state"] = np.array([-1, -1], np.int32), np.append(c["transform_state"], unknown)
    for name, value in ONE_SIDED.items():
        c[f"external_{name}"] = np.array([value], c[f"external_{name}"].dtype)
    for name in ("order", "aromatic", "kind", "direction"):
        c[f"external_{name}_state"] = np.array([KNOWN if name == "kind" else unknown], np.uint8)
    c["external_declared_leaving_ptr"] = np.zeros(2, np.int64)
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())


class TestEncodeMmcif:
    @pytest.mark.parametrize("block", ["my protein", "", "x" * 76, "蛋白"])
    def test_invalid_block_refused(self, placed, invalid, block):
        invalid(encode_mmcif, placed(*CHAINS), block=block, identity="source")

    def test_coordinates_exact(self, placed):
        structure = placed(*CHAINS)
        written, records = categories(structure)
        assert np.array_equal(written["atom_site"].floats("Cartn_x")[0], structure.coordinates[:, 0])
        assert not len(records)

    def test_hybrid36_number_recorded(self, placed):  # mmCIF has no hybrid-36: A000 re-reads with no integer
        structure = placed(*CHAINS)
        arrays = structure.build()
        arrays.symbols["author_number"] = (*arrays.symbols["author_number"], "A000")
        arrays.columns["component_author_text"][2] = len(arrays.symbols["author_number"]) - 1
        arrays.columns["component_author_number"][2] = 10000
        _, records = categories(Structure.from_arrays(structure.system, arrays))
        assert [(record.detail["family"], record.detail["item"]) for record in records] == [
            ("author_number", "_atom_site.auth_seq_id")
        ]

    def test_derived_model_number_reported(self, placed):  # mmCIF restores a model number only as declared
        structure = placed(*CHAINS)
        arrays = structure.build()
        arrays.columns["system_model_number"][:] = 2
        arrays.columns["system_model_number_state"][:] = state(Resolution.KNOWN, Basis.DERIVED)
        _, records = categories(Structure.from_arrays(structure.system, arrays))
        assert [(record.detail["family"], record.detail["item"]) for record in records] == [
            ("basis", "_atom_site.pdbx_PDB_model_num")
        ]

    def test_unplaced_row_written(self, placed):  # a `?`-coordinate row keeps its row, its coordinates `?`
        structure = placed(*CHAINS)
        arrays = structure.build()
        for name in ("coordinates", "coordinate_evidence", "coordinate_quantum"):
            arrays.columns[name][0] = 0  # CoordinateEvidence.NONE; the occupancy and B of its row stay
        written, records = categories(Structure.from_arrays(structure.system, arrays))
        assert written["atom_site"].nrows == len(structure.coordinates) and not len(records)
        assert written["atom_site"].text("Cartn_x")[1][0] == TokenState.UNKNOWN

    def test_default_item_omitted(self, placed):
        structure = placed(*CHAINS)
        arrays = structure.build()
        arrays.columns["occupancy_state"][:] = state(Resolution.KNOWN, Basis.DEFAULT)
        written, _ = categories(Structure.from_arrays(structure.system, arrays))
        assert "occupancy" not in written["atom_site"].items

    def test_derived_carries_basis(self, placed):
        written, _ = categories(placed(*CHAINS))  # OFoldX numbered the positions: KNOWN×DERIVED
        rows = texts(written["ofoldx_fact_basis"], "category", "item", "state")
        assert ("pdbx_poly_seq_scheme", "seq_id", str(state(Resolution.KNOWN, Basis.DERIVED))) in rows

    def test_prediction_kept_private(self, placed):
        structure = placed(*CHAINS)
        n = len(structure.coordinates)
        predicted, _ = replace_coordinates(
            structure,
            np.arange(n),
            structure.coordinates,
            scope=np.ones(n, bool),
            evidence=CoordinateEvidence.PREDICTED,
        )
        written, _ = categories(predicted)
        assert set(written["ofoldx_coordinate_evidence"].text("evidence")[0].astype(str)) == {"predicted"}
        assert written["ofoldx_coordinate_evidence"].nrows == n

    def test_declared_parent_written(self, parented):  # at its label address too, which tells copies apart
        written, records = categories(parented(*CHAINS))
        items = ("label_asym_id", "label_seq_id", "auth_comp_id", "parent_comp_id")
        assert texts(written["pdbx_struct_mod_residue"], *items) == [("A", "1", "ALA", "MET")]
        assert not len(records)

    def test_withdrawn_atom_not_written(self, placed):  # 4z84: its altloc and charge cell state no row
        structure = placed(*CHAINS)
        n = len(structure.coordinates)
        kept, every = np.arange(n - 1), np.ones(n, bool)
        decoded, _ = replace_coordinates(
            structure, kept, structure.coordinates[kept], scope=every, evidence=CoordinateEvidence.PREDICTED
        )
        written, _ = categories(decoded)
        assert written["atom_site"].nrows == n - 1

    def test_one_sided_external_reported(self, placed):  # 1dat's assembly: its row would read back from both sides
        written, records = categories(one_sided(placed(*CHAINS)))
        assert "struct_conn" not in written
        assert [(record.detail["family"], record.detail["item"]) for record in records] == [
            ("external_connection", "_struct_conn")
        ]

    @pytest.mark.parametrize("link", ["", LINK], ids=["unlinked", "linked"])
    def test_untyped_junction_written_as_row(self, dipeptide, library, link):  # 5vi6: a PDB chain deriving no type
        source, _ = read_structure(parse_source(dipeptide(), format="mmcif"), library=library)
        pdb = encode_pdb(source, identity="source", chains=None)[0].decode().replace("ATOM", link + "ATOM", 1)
        structure, _ = read_structure(parse_source(pdb, format="pdb"), library=library)
        data, records = encode_mmcif(structure, block="test", identity="source")
        again, _ = read_structure(parse_source(data.decode(), format="mmcif"), library=library)
        assert again.system.edge_index.shape == structure.system.edge_index.shape
        assert [(record.detail["family"], record.detail["item"]) for record in records] == [
            ("junction", "_entity_poly.type")
        ]

    def test_assumption_reported_not_stated(self, placed):
        structure = placed(*CHAINS)
        arrays = structure.system.build()
        arrays.columns["entity_type_state"][0] = state(Resolution.KNOWN, Basis.ASSUMED)
        assumed = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())
        written, records = categories(assumed)
        assert written["entity"].text("type")[1][0] == TokenState.UNKNOWN
        assert {record.detail["family"] for record in records} == {"assumed"}
        assert mmcif_capability(assumed, identity="source")[0].detail["family"] == "assumed"

    @pytest.mark.parametrize(
        ("conn_type", "kind_state"),
        [("?", (Resolution.UNKNOWN, Basis.DECLARED)), ("", (Resolution.UNKNOWN, Basis.NONE))],
        ids=["unknown", "absent"],
    )
    def test_undetermined_kind_kept(self, library, mmcif_text, conn_type, kind_state):  # never guessed
        text = mmcif_text.replace("metalc1 metalc", f"metalc1 {conn_type}")
        text = text if conn_type else text.replace("_struct_conn.conn_type_id\n", "")
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        data, records = encode_mmcif(structure, block="test", identity="source")
        source = parse_source(data.decode(), format="mmcif")
        system = read_structure(source, library=library, on_conflict="record")[0].system
        assert system.external_kind_state.tolist() == [state(*kind_state)] * 2  # its partner's symmetry unstated
        assert "connection_kind" not in {record.detail["family"] for record in records}

    def test_branch_token_reported(self, library):  # the value holds arom as `?`: written `?`, never silently
        structure, _ = read_structure(parse_source(AROMATIC_LINK, format="mmcif"), library=library)
        written, records = categories(structure)
        assert written["pdbx_entity_branch_link"].text("value_order")[1].tolist() == [TokenState.UNKNOWN]
        assert ("order", "_pdbx_entity_branch_link.value_order") in [
            (record.detail["family"], record.detail["item"]) for record in records
        ]
