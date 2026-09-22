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

"""Behaviour of ofoldx.data.schema: the schema digest, publishing and the two validators."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import AtomFate, MolecularSystem
from ofoldx.data.schema import Field, Schema, publish
from ofoldx.data.structure import CoordinateEvidence, OperationNumbering, Structure
from ofoldx.errors import UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

UN, UD = state(Resolution.UNKNOWN, Basis.NONE), state(Resolution.UNKNOWN, Basis.DECLARED)
KD, UDER = state(Resolution.KNOWN, Basis.DECLARED), state(Resolution.UNKNOWN, Basis.DERIVED)
IDEF, CN = state(Resolution.INAPPLICABLE, Basis.DEFINED), state(Resolution.CONFLICT, Basis.NONE)


class TestDigest:
    def test_field_order_is_irrelevant(self):
        schema = MolecularSystem.SCHEMA
        assert Schema(schema.fields[::-1], schema.axes).digest() == schema.digest()

    @pytest.mark.parametrize(
        "changed",
        [
            Field("w", "int32", "row"),
            Field("v", "int64", "row"),
            Field("v", "int32", "atom"),
            Field("v", "int32", "row", (2,)),
            Field("v", "int32", "row", leading=(2,)),
            Field("v", "int32", "row", state=True),
            Field("v", "int32", "row", symbols="atom"),
        ],
    )
    def test_declaration_changes_digest(self, changed):
        base = Field("v", "int32", "row")
        assert Schema((base,), ("row", "atom")).digest() != Schema((changed,), ("row", "atom")).digest()


class TestPublish:
    def test_overflow_is_precision_loss(self, system_arrays):
        arrays = system_arrays()
        arrays.columns["instance_entity"] = np.full(3, 2**40, dtype=np.int64)
        with pytest.raises(UnsupportedError) as caught:
            publish(arrays.columns, arrays.symbols, MolecularSystem.SCHEMA)
        assert caught.value.code == "precision_loss"

    @pytest.mark.parametrize("record", [MolecularSystem, Structure])
    def test_fixed_dimensions_are_checked(self, record, system_arrays, structure_arrays, invalid):
        arrays = system_arrays() if record is MolecularSystem else structure_arrays()
        for field in record.SCHEMA.fields:
            if field.leading or field.trailing:
                columns = dict(arrays.columns)
                columns[field.name] = columns[field.name][..., :-1] if field.trailing else columns[field.name][:-1]
                invalid(publish, columns, arrays.symbols, record.SCHEMA)

    @pytest.mark.parametrize("record", [MolecularSystem, Structure])
    @pytest.mark.parametrize("table", [None, ["X"], (0,)], ids=["missing", "list", "not-texts"])
    def test_declared_families_are_checked(self, record, table, system, system_arrays, structure_arrays, invalid):
        for family in {field.symbols for field in record.SCHEMA.fields} - {None}:
            arrays = system_arrays() if record is MolecularSystem else structure_arrays()
            del arrays.symbols[family]
            if table is not None:
                arrays.symbols[family] = table
            if record is MolecularSystem:
                invalid(MolecularSystem.from_arrays, arrays)
            else:
                invalid(Structure.from_arrays, system, arrays)


def set_items(**changes):
    """A change writing ``column[index] = value`` for each ``column=(index, value)``."""

    def change(columns):
        for name, (index, value) in changes.items():
            columns[name][index] = value

    return change


def edge_across_systems(columns):
    """Split W into a second system, then bond LIG O1 to the water oxygen."""
    columns["system_entity_ptr"] = np.array([0, 2, 3])
    columns["system_instance_ptr"] = np.array([0, 2, 3])
    columns["edge_index"] = np.concatenate((columns["edge_index"], [[16], [17]]), axis=1)
    for name in [name for name in columns if name.startswith("bond_") and "declared" not in name]:
        columns[name] = np.append(columns[name], columns[name][-1])
    columns["bond_declared_leaving_ptr"] = np.append(columns["bond_declared_leaving_ptr"], 1)


SYSTEM_VIOLATIONS = {
    "state-byte-zero": set_items(atomic_number_state=(0, 0)),
    "value-under-unknown": set_items(implicit_hydrogens=(10, 1)),
    "unknown-parity-with-slots": set_items(stereocentre_state=(0, UN), stereocentre_parity=(0, 0)),
    "duplicate-atom-name": set_items(atom_symbol=(1, 0)),
    "isotope-below-element": set_items(isotope=(0, 3)),
    "left-atom": set_items(atom_fate=(0, AtomFate.LEFT)),
    "duplicate-edge": set_items(edge_index=((slice(None), 1), [0, 1])),
    "edge-across-systems": edge_across_systems,
    "identity-external-to-present-partner": set_items(external_transform=(0, 0)),
    "rotation-of-infinite-order": set_items(transform_rotation=(1, [[2, 1, 0], [1, 1, 0], [0, 0, 1]])),
    "claim-over-undetermined-atom": set_items(atom_fate=(0, AtomFate.UNDETERMINED)),
    "mixed-conformer-sources": set_items(reference_pool_source=(0, 2)),
    "slot-not-bonded": set_items(stereocentre_neighbors=(0, [0, 2, 4, 3])),
    "two-implicit-slots": set_items(stereocentre_neighbors=(1, [12, 15, -1, -1])),
    "stereobond-reference-not-bonded": set_items(stereobond_reference=(0, [15, 14])),
}


ENUMERATED = {
    MolecularSystem: (
        *("entity_type", "entity_nstd_linkage", "component_definition_scope", "component_monomer_type"),
        *("atom_fate", "atom_endpoint_state", "atom_designation", "reference_pool_source"),
        *("bond_kind", "bond_direction", "bond_origin", "bond_declared_leaving_side"),
        *("external_kind", "external_direction", "external_origin", "external_declared_leaving_side"),
        "stereogroup_kind",
    ),
    Structure: (*("coordinate_evidence", "instance_copy_kind", "resolution_kind", "operation_numbering"),),
}


class TestValidateSystem:
    @pytest.mark.parametrize("change", SYSTEM_VIOLATIONS.values(), ids=SYSTEM_VIOLATIONS)
    def test_violation_is_invalid(self, change, system_arrays, invalid):
        arrays = system_arrays()
        change(arrays.columns)
        invalid(MolecularSystem.from_arrays, arrays)

    def test_duplicate_symbol_is_invalid(self, system_arrays, invalid):
        arrays = system_arrays()
        arrays.symbols["atom"] = (*arrays.symbols["atom"], "N")
        invalid(MolecularSystem.from_arrays, arrays)

    @pytest.mark.parametrize("record", [MolecularSystem, Structure])
    def test_codes_index_their_family(self, record, system, system_arrays, structure_arrays, invalid):
        for field in record.SCHEMA.fields:
            if field.symbols is None:
                continue
            arrays = system_arrays() if record is MolecularSystem else structure_arrays()
            arrays.columns[field.name].reshape(-1)[0] = len(arrays.symbols[field.symbols])
            if record is MolecularSystem:
                invalid(MolecularSystem.from_arrays, arrays)
            else:
                invalid(Structure.from_arrays, system, arrays)

    @pytest.mark.parametrize("record", [MolecularSystem, Structure])
    def test_enum_values_are_checked(self, record, system, system_arrays, structure_arrays, invalid):
        for name in ENUMERATED[record]:
            arrays = system_arrays() if record is MolecularSystem else structure_arrays()
            arrays.columns[name][0] = 100  # a member of no enumeration
            if f"{name}_state" in arrays.columns:
                arrays.columns[f"{name}_state"][0] = KD
            if record is MolecularSystem:
                invalid(MolecularSystem.from_arrays, arrays)
            else:
                invalid(Structure.from_arrays, system, arrays)


def untabled(numbering, **changes):
    """No operation rows, ``numbering`` in force, then ``set_items(**changes)``."""

    def change(columns):
        for name in ("operation_id", "operation_rotation", "operation_translation"):
            columns[name] = columns[name][:0]
        columns["operation_ptr"][:] = 0
        columns["operation_numbering"][0] = numbering
        set_items(**changes)(columns)

    return change


NO_CELL = {"cell_state": (0, UN), "cell": (0, 0), "cell_quantum": (0, 0)}


def atom_dropped(columns):
    for field in Structure.SCHEMA.fields:
        if field.axis == "atom":
            columns[field.name] = columns[field.name][:-1]


STRUCTURE_VIOLATIONS = {
    "coordinates-without-evidence": set_items(coordinates=(5, [1, 0, 0])),
    "coordinate-not-finite": set_items(coordinates=((0, 0), np.nan)),
    "quantum-without-coordinates": set_items(coordinate_quantum=(5, 0.001)),
    "deposited-observation-without-unit": set_items(coordinate_quantum=(0, 0.0)),
    "occupancy-above-one-unknown-su": set_items(occupancy=(0, 1.02)),
    "predicted-keeps-occupancy": set_items(coordinate_evidence=(0, CoordinateEvidence.PREDICTED)),
    "charge-without-document": set_items(source_sha256=(0, 0)),
    "unplaced-occupancy-without-document": set_items(
        source_sha256=(0, 0), atom_site_charge_state=(slice(None), UN), occupancy_state=(5, KD)
    ),
    "author-number-differs-from-text": set_items(component_author_number=(0, 5)),
    "author-number-without-text": set_items(component_author_text=(0, -1)),
    "unread-text-not-derived": set_items(component_author_number=(0, 0), component_author_number_state=(0, UD)),
    "assembly-without-operators": set_items(instance_copy_kind=(0, 1)),
    "duplicate-copy-id": set_items(instance_source_asym=(1, 0)),
    "zero-cell-length": set_items(cell=((0, 0), 0)),
    "left-handed-frame": set_items(frame=((0, 0, 0), -0.02)),
    "inapplicable-left-handed-frame": set_items(frame=((0, 0, 0), -0.02), frame_state=(0, IDEF), **NO_CELL),
    "inapplicable-transform-beside-cell": set_items(frame_state=(0, IDEF)),  # an inapplicable frame rests on the cell
    "transform-beside-conflicting-cell": set_items(frame_state=(0, IDEF), **{**NO_CELL, "cell_state": (0, CN)}),
    "singular-operation": set_items(operation_rotation=(1, 0)),
    "numbering-none-with-table": set_items(operation_numbering=(0, OperationNumbering.NONE)),
    "setting-rows-stored": set_items(operation_numbering=(0, OperationNumbering.ITA)),  # regenerated, never stored
    "file-numbering-without-table": untabled(OperationNumbering.FILE),
    "setting-without-space-group": untabled(OperationNumbering.ITA, space_group=(0, -1), space_group_state=(0, UN)),
    "atom-count": atom_dropped,
}


class TestValidateStructure:
    @pytest.mark.parametrize("change", STRUCTURE_VIOLATIONS.values(), ids=STRUCTURE_VIOLATIONS)
    def test_violation_is_invalid(self, change, system, structure_arrays, invalid):
        arrays = structure_arrays()
        change(arrays.columns)
        invalid(Structure.from_arrays, system, arrays)

    @pytest.mark.parametrize(("text", "number", "number_state"), [("+5", 5, KD), ("A000", 10000, KD), ("12A", 0, UDER)])
    def test_source_reading_constructs(self, text, number, number_state, system, structure_arrays):  # io reads it
        arrays = structure_arrays()
        arrays.symbols["author_number"] = (text, "2", "101", "201")
        arrays.columns["component_author_number"][0] = number
        arrays.columns["component_author_number_state"][0] = number_state
        Structure.from_arrays(system, arrays)

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(set_items(component_author_number=(slice(0, 3, 2), [7, -1])), id="leading-zero-and-minus"),
            pytest.param(
                set_items(
                    component_author_text=(0, -1), component_author_number=(0, 0), component_author_number_state=(0, UD)
                ),
                id="stated-unknown-number",
            ),
            pytest.param(set_items(operation_rotation=(1, [[1, 2, 0], [0, -1, 0], [0, 0, -1]])), id="rotation-entry-2"),
            pytest.param(untabled(OperationNumbering.ITA), id="setting-without-rows"),
            pytest.param(set_items(frame_state=(0, IDEF), **NO_CELL), id="inapplicable-frame-keeps-transform"),
            pytest.param(set_items(occupancy_state=(5, KD)), id="unplaced-occupancy-beside-document"),
            pytest.param(
                set_items(occupancy=(0, 1.02), occupancy_uncertainty=(0, 0.01), occupancy_uncertainty_state=(0, KD)),
                id="occupancy-within-three-su",
            ),
        ],
    )
    def test_valid_edge_cases_construct(self, change, system, structure_arrays):
        arrays = structure_arrays()
        arrays.symbols["author_number"] = ("007", "2", "-1", "201")
        arrays.columns["component_author_number"][:] = [7, 2, -1, 201]
        change(arrays.columns)
        Structure.from_arrays(system, arrays)
