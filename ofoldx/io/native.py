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

r"""
Native artifacts: a ``Structure`` with its ``MolecularSystem`` (and optionally the definitions it
restates, those not of archive scope, with the CCD artifact they were resolved against), one safetensors file written
deterministically and atomically. No diagnostic report is kept: every state column already holds
what the read decided.

Every column is stored as it is and every symbol table as packed text columns, so every state column and basis
category survives. The header names the schema and the sha256 of the header (every tensor's dtype, shape and
offsets, and the metadata) and the payload; a structure artifact's schema is ``NATIVE_SCHEMA``, which changes with
either value schema, so an artifact of another schema is refused (no compatibility loading) and one whose header or
payload does not match its digest is corrupt. The digest detects corruption, it proves no provenance: a load constructs
the value through ``from_arrays``, validated as any other value.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..chemistry.definitions import (
    ArtifactProvenance,
    ChemicalComponent,
    ComponentLibrary,
    Definitions,
    artifact_columns,
)
from ..data.molecular import DefinitionScope, MolecularSystem, SystemArrays
from ..data.structure import Structure, StructureArrays
from ..errors import RequestRef, SourceRef, fail
from ..provenance import Records
from .writing import publish

NATIVE_SCHEMA = f"ofoldx.structure/11+{MolecularSystem.SCHEMA.digest()}+{Structure.SCHEMA.digest()}"
_DTYPES = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype(np.uint16),
    "I16": np.dtype(np.int16),
    "U32": np.dtype(np.uint32),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}
_NAMES = {dtype: name for name, dtype in _DTYPES.items()}


def _digest(header: Mapping[str, object], payload: bytes) -> str:
    r"""sha256 over the header (without the digest) and the payload."""
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(len(text).to_bytes(8, "little") + text + payload).hexdigest()


def _write(path: str | PathLike, columns: Mapping[str, NDArray], metadata: Mapping[str, str]) -> None:
    r"""One safetensors file: header keys sorted (byte-identical for equal input), the sha256 of header and payload."""
    header, chunks, offset = {}, [], 0
    for name in sorted(columns):
        array = np.ascontiguousarray(columns[name])
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {
            "dtype": _NAMES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    payload = b"".join(chunks)
    header["__metadata__"] = dict(metadata)
    header["__metadata__"]["sha256"] = _digest(header, payload)
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    publish(path, struct.pack("<Q", len(text)) + text + payload)


def _read(path: str | PathLike, schema: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    r"""
    The columns (owned copies) and metadata of an artifact of ``schema``.

    Raises:
        InvalidRequestError: ``native_schema_version`` for an artifact of another schema.
        ContradictionError: ``native_corrupt`` when the header is unreadable, header and payload do not match their
            digest, or a tensor (named) does not state a shape and the byte range of exactly its values inside the
            payload.
    """
    data = Path(path).read_bytes()
    where = SourceRef(str(path), 0)
    try:
        (size,) = struct.unpack("<Q", data[:8])
        header = json.loads(_utf8(data[8 : 8 + size], where, "header"))
        metadata = header["__metadata__"]
        claimed = metadata.pop("sha256", None)
    except (struct.error, ValueError, KeyError, TypeError, AttributeError):
        fail("native_corrupt", where, item="header")
    if metadata.get("schema") != schema:
        fail("native_schema_version", RequestRef("schema", str(metadata.get("schema"))), expected=schema)
    payload = data[8 + size :]
    if _digest(header, payload) != claimed:
        fail("native_corrupt", where, item="sha256")
    del header["__metadata__"]
    columns = {}
    for name, entry in header.items():
        try:
            dtype, shape, (start, stop) = _DTYPES[entry["dtype"]], list(entry["shape"]), entry["data_offsets"]
        except (KeyError, ValueError, TypeError):
            fail("native_corrupt", where, item="header")
        counts = all(type(n) is int and n >= 0 for n in (*shape, start, stop))
        if not counts or not start <= stop <= len(payload) or stop - start != dtype.itemsize * math.prod(shape):
            fail("native_corrupt", where, item=name)
        array = np.frombuffer(payload[start:stop], dtype=dtype.newbyteorder("<"))
        columns[name] = array.astype(dtype).reshape(shape)  # an owned copy, never a view of the file bytes
    return columns, metadata


def _pack(columns: dict[str, NDArray], name: str, texts: Sequence[str]) -> None:
    encoded = [text.encode() for text in texts]
    columns[name] = np.frombuffer(b"".join(encoded), np.uint8).copy()
    columns[f"{name}.offset"] = np.cumsum([0, *map(len, encoded)], dtype=np.int64)


def _utf8(data: bytes, where: SourceRef, item: str) -> str:
    r"""The text of ``data``, a field the format stores as UTF-8: ``native_corrupt`` (``item``, the decoder's reason)
    for bytes that are no UTF-8, never repaired."""
    try:
        return data.decode()
    except UnicodeDecodeError as error:
        fail("native_corrupt", where, item=item, reason=str(error))


def _unpack(columns: Mapping[str, NDArray], name: str, where: SourceRef) -> tuple[str, ...]:
    r"""The texts of the packed table ``name``; ``native_corrupt`` unless its two tensors are exactly their packing."""
    try:
        blob, offsets = columns[name].tobytes(), columns[f"{name}.offset"].tolist()
        texts = tuple(_utf8(blob[a:b], where, name) for a, b in zip(offsets[:-1], offsets[1:]))
    except (KeyError, TypeError):  # a tensor of the pair missing; offsets that are no integer list
        fail("native_corrupt", where, item=name)
    packed: dict[str, NDArray] = {}
    _pack(packed, name, texts)
    if any(columns[k].dtype != v.dtype or not np.array_equal(columns[k], v) for k, v in packed.items()):
        fail("native_corrupt", where, item=name)  # not exactly the packing of its texts
    return texts


def _unpack_json(columns: Mapping[str, NDArray], name: str, where: SourceRef) -> object:
    r"""The value of the one JSON text of the packed table ``name``."""
    try:
        (text,) = _unpack(columns, name, where)
        return json.loads(text)
    except ValueError:  # another count of texts, or a text that is not JSON
        fail("native_corrupt", where, item=name)


# Structure ----------------------------------------------------------------------------------------------------------


def save_structure(structure: Structure, path: str | PathLike, *, definitions: Definitions | None = None) -> None:
    r"""
    Save ``structure`` as one native artifact; with ``definitions`` (a read's ``report.definitions``), also the ones
    it restates (those not of archive scope) and the CCD artifact they were resolved against, as columns under
    ``definitions.``, so the value can be written as mmCIF again (``write_structure(..., definitions=)``).
    """
    columns: dict[str, NDArray] = {}
    for prefix, record in (("system", structure.system), ("structure", structure)):
        for name, array in record.arrays().items():
            columns[f"{prefix}.{name}"] = array
        for family, table in record.symbols().items():
            _pack(columns, f"{prefix}.symbols.{family}", table)
    if definitions is not None:
        _put_definitions(columns, "definitions", definitions)
    _write(path, columns, {"schema": NATIVE_SCHEMA})


def load_structure(path: str | PathLike) -> Structure:
    r"""
    The structure of a native artifact, constructed through ``from_arrays`` (validated as any value).

    Raises:
        InvalidRequestError: ``native_schema_version``; ``value_invalid`` for a value that violates an invariant.
        ContradictionError: ``native_corrupt``, also for tensors that are not exactly the declared columns in their
            dtypes, or a symbol family that is not the packing of its texts.
    """
    columns, _ = _read(path, NATIVE_SCHEMA)
    where = SourceRef(str(path), 0)

    def part(prefix: str, record) -> tuple[dict[str, tuple[str, ...]], dict[str, NDArray]]:
        start = f"{prefix}.symbols."
        families = {k[len(start) :].removesuffix(".offset") for k in columns if k.startswith(start)}
        symbols = {family: _unpack(columns, start + family, where) for family in families}
        stored = {k[len(prefix) + 1 :] for k in columns if k.startswith(f"{prefix}.") and not k.startswith(start)}
        declared = {field.name for field in record.SCHEMA.fields}
        if stored != declared:
            missing, extra = sorted(declared - stored), sorted(stored - declared)
            fail("native_corrupt", where, item=prefix, missing=missing, extra=extra)
        for field in record.SCHEMA.fields:  # another dtype's bytes are other values, never cast
            if columns[f"{prefix}.{field.name}"].dtype != np.dtype(field.dtype):
                fail("native_corrupt", where, item=f"{prefix}.{field.name}")
        return symbols, {name: columns[f"{prefix}.{name}"] for name in declared}

    system = MolecularSystem.from_arrays(SystemArrays(*part("system", MolecularSystem)))
    return Structure.from_arrays(system, StructureArrays(*part("structure", Structure)))


def load_definitions(path: str | PathLike) -> Definitions | None:
    r"""
    The definitions the saved structure restates and the CCD artifact they were resolved against, ``None`` when none
    were saved.

    Raises:
        ContradictionError: ``native_corrupt``, also for a definition whose digest is not that of its fields, scopes
            that are not one uint8 definition scope per stored definition, or a text table that is not exactly the
            packing of its texts.
    """
    columns, _ = _read(path, NATIVE_SCHEMA)
    if not any(name.startswith("definitions.") for name in columns):
        return None
    return _get_definitions(columns, "definitions", SourceRef(str(path), 0))


def _put_definitions(columns: dict[str, NDArray], prefix: str, definitions: Definitions) -> None:
    r"""The definitions a writer restates (an archive one comes back from the library by digest and content
    version) and the CCD artifact's provenance."""
    local = [c for c in definitions.by_id.values() if c.scope != DefinitionScope.ARCHIVE]
    components = sorted(local, key=lambda c: c.component_id)
    for name, array in artifact_columns(components).items():
        columns[f"{prefix}.artifact.{name}"] = array
    columns[f"{prefix}.scope"] = np.array([c.scope for c in components], dtype=np.uint8)
    library = None if definitions.library is None else asdict(definitions.library)
    _pack(columns, f"{prefix}.library", [json.dumps(library, sort_keys=True)])


def _get_definitions(columns: Mapping[str, NDArray], prefix: str, where: SourceRef) -> Definitions:
    start = f"{prefix}.artifact."
    library = _unpack_json(columns, f"{prefix}.library", where)
    try:
        provenance = None if library is None else ArtifactProvenance(**cast(Mapping[str, str], library))
    except TypeError:  # JSON that is not a provenance
        fail("native_corrupt", where, item=prefix)
    artifact = ComponentLibrary({k[len(start) :]: v for k, v in columns.items() if k.startswith(start)}, provenance)
    scope, ids = columns.get(f"{prefix}.scope"), _unpack(columns, f"{start}component_id", where)
    if (
        scope is None
        or scope.dtype != np.uint8
        or scope.shape != (len(ids),)
        or not set(scope.tolist()) <= set(DefinitionScope)
        or not all(cid in artifact for cid in ids)
    ):
        fail("native_corrupt", where, item=f"{prefix}.scope")
    by_id = {
        cid: cast(ChemicalComponent, artifact.definition(cid)).at_scope(DefinitionScope(s))
        for cid, s in zip(ids, scope.tolist())
    }
    return Definitions(MappingProxyType(by_id), Records.empty(), provenance)
