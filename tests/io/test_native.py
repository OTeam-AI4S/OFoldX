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

"""Behaviour of ofoldx.io.native: exact round trips (with definitions), refused foreign schemas and corrupt files."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest
from safetensors.numpy import save_file

from ofoldx.chemistry.definitions import Definitions
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.errors import ContradictionError, InvalidRequestError, SourceRef
from ofoldx.io.native import NATIVE_SCHEMA, _digest, _read, _write, load_definitions, load_structure, save_structure
from ofoldx.provenance import Records

CHAINS = (("A", "1", ("ALA", "GLY")), ("B", "2", ("ZN",)))


def trailing_byte(header) -> None:
    header["structure.b_factor"]["data_offsets"][1] += 1  # into the next tensor's bytes


HEADER_EDITS = {
    "dtype": lambda header: header["structure.b_factor"].update(dtype="I64"),
    "negative-dimension": lambda header: header["structure.coordinates"].update(shape=[-1, 3]),
    "trailing-byte": trailing_byte,
}
TENSOR_EDITS = {
    "missing-column": lambda columns: columns.pop("structure.b_factor"),
    "extra-column": lambda columns: columns.update({"structure.extra": np.zeros(1)}),
    "symbols-without-offset": lambda columns: columns.pop("structure.symbols.chain.offset"),
    "offset-without-symbols": lambda columns: columns.pop("structure.symbols.chain"),
    "decreasing-offsets": lambda columns: columns.update({"structure.symbols.chain.offset": np.array([0, 2, 1])}),
}


def replaced(prefix: str, tensors):
    r"""An edit setting each ``<prefix>.<name>`` tensor of ``tensors``."""
    return lambda columns: columns.update({f"{prefix}.{name}": value for name, value in tensors.items()})


DEFINITION_EDITS = {
    "charge": lambda columns: np.put(columns["definitions.artifact.charge"], 0, 1),
    "column-missing": lambda columns: columns.pop("definitions.artifact.charge"),
    "scopes": lambda columns: columns.update({"definitions.scope": columns["definitions.scope"][:0]}),
    "texts-truncated": replaced("definitions", {"library.offset": np.zeros(1, np.int64)}),
    "offset-missing": lambda columns: columns.pop("definitions.library.offset"),
    "library-missing": lambda columns: [columns.pop(f"definitions.library{s}") for s in ("", ".offset")],
    "no-text": replaced("definitions", {"library": np.zeros(0, np.uint8), "library.offset": np.zeros(1, np.int64)}),
    "not-json": replaced(
        "definitions", {"library": np.frombuffer(b"{]", np.uint8), "library.offset": np.array([0, 2])}
    ),
    "scope-shape": replaced("definitions", {"scope": np.full((1, 1), 4, np.uint8)}),
    "scope-dtype": replaced("definitions", {"scope": np.full(1, 4.0, np.float32)}),
    "id-not-utf8": lambda columns: np.put(columns["definitions.artifact.component_id"], 0, 0xFF),
}


@pytest.fixture
def defined(placed, archive, library, tmp_path):
    r"""``placed(*CHAINS)`` saved with definitions of one SMILES ligand and CCD ALA: (path, ligand)."""
    path = tmp_path / "x.safetensors"
    ligand = read_smiles("CCO", component_id="LIG").component
    refs = {"LIG": (SourceRef("caller", 0, item="LIG"),)}
    definitions = Definitions({"LIG": ligand, "ALA": archive("ALA")}, Records.empty(), library.provenance, refs)
    save_structure(placed(*CHAINS), path, definitions=definitions)
    return path, ligand


def same(a, b) -> bool:
    r"""Every field equal: arrays elementwise, everything else by value."""
    pairs = ((getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in pairs)


class TestStructureArtifact:
    def test_round_trip(self, placed, tmp_path):
        structure = placed(*CHAINS)
        save_structure(structure, tmp_path / "x.safetensors")
        assert load_structure(tmp_path / "x.safetensors").digest() == structure.digest()
        assert load_definitions(tmp_path / "x.safetensors") is None

    def test_foreign_schema_refused(self, tmp_path):
        save_file({"x": np.zeros(1)}, tmp_path / "x.safetensors", metadata={"schema": "ofoldx.other/1"})
        with pytest.raises(InvalidRequestError) as caught:
            load_structure(tmp_path / "x.safetensors")
        assert caught.value.code == "native_schema_version"

    def test_corrupt_refused(self, placed, tmp_path):
        path = tmp_path / "x.safetensors"
        save_structure(placed(*CHAINS), path)
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))
        with pytest.raises(ContradictionError) as caught:
            load_structure(path)
        assert caught.value.code == "native_corrupt"

    @pytest.mark.parametrize("edit", HEADER_EDITS.values(), ids=HEADER_EDITS)
    def test_edited_header_refused(self, edit, placed, tmp_path):
        path = tmp_path / "x.safetensors"
        save_structure(placed(*CHAINS), path)
        data = path.read_bytes()
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8 : 8 + size])
        edit(header)
        del header["__metadata__"]["sha256"]
        header["__metadata__"]["sha256"] = _digest(header, data[8 + size :])  # a layout the digest vouches for
        text = json.dumps(header).encode()
        path.write_bytes(len(text).to_bytes(8, "little") + text + data[8 + size :])  # the payload as written
        with pytest.raises(ContradictionError) as caught:
            load_structure(path)
        assert caught.value.code == "native_corrupt"

    def test_invalid_value_refused(self, placed, tmp_path, invalid):
        path = tmp_path / "x.safetensors"
        save_structure(placed(*CHAINS), path)
        columns, metadata = _read(path, NATIVE_SCHEMA)
        columns["structure.occupancy"] = columns["structure.b_factor"]  # 20: outside [0, 1]
        _write(path, columns, metadata)
        invalid(load_structure, path)

    @pytest.mark.parametrize("edit", TENSOR_EDITS.values(), ids=TENSOR_EDITS)
    def test_malformed_tensors_refused(self, edit, placed, tmp_path):
        path = tmp_path / "x.safetensors"
        save_structure(placed(*CHAINS), path)
        columns, metadata = _read(path, NATIVE_SCHEMA)
        edit(columns)
        _write(path, columns, metadata)
        with pytest.raises(ContradictionError) as caught:
            load_structure(path)
        assert caught.value.code == "native_corrupt"

    def test_definitions_round_trip(self, defined, library):  # the archive's ALA comes back from the library
        path, ligand = defined
        loaded = load_definitions(path)
        assert (list(loaded.by_id), loaded.by_id["LIG"].digest, loaded.by_id["LIG"].scope) == (
            ["LIG"],
            ligand.digest,
            ligand.scope,
        )
        assert loaded.library == library.provenance

    @pytest.mark.parametrize("edit", DEFINITION_EDITS.values(), ids=DEFINITION_EDITS)
    def test_edited_definitions_refused(self, edit, defined):
        path = defined[0]
        columns, metadata = _read(path, NATIVE_SCHEMA)
        edit(columns)
        _write(path, columns, metadata)
        with pytest.raises(ContradictionError) as caught:
            load_definitions(path)
        assert caught.value.code == "native_corrupt"
