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
The frozen array-record substrate: declared schemas, read-only publishing, the schema digest and the two
validators called by the two constructors.

A canonical value is an ``ArrayRecord``: a frozen record of read-only numpy columns, one declared ``Schema`` per class,
one validating constructor (``from_arrays``). Every fact that can be unknown has a sibling ``<column>_state`` column;
where its resolution is not KNOWN the paired value holds canonical zero (0, or ``-1`` for codes and indices), but for
the transform an INAPPLICABLE frame rests on while no cell value is stated.

The validators check the invariants whose violation leaves a value scientifically wrong or internally inconsistent.
The first violation is ``InvalidRequestError(value_invalid)`` naming the column, the invariant and the first bad row.
``check_rows`` and ``check_mask`` are the one check of a caller's indices and masks, run at an entry before any of them
indexes or selects.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, StructureRef, fail
from ..provenance import ALLOWED_STATES, Basis, Resolution, known, resolution, state


@dataclass(frozen=True, slots=True)
class Field:
    r"""
    One declared column.

    ``axis`` is a row axis, or ``"<axis>+1"`` for a CSR pointer over ``<axis>``. ``leading`` and ``trailing`` are the
    fixed dimensions before and after the row axis (``edge_index`` is ``[2, m]``). ``state`` marks a column holding fact
    state bytes; ``symbols`` names the symbol family the column's codes index (``-1`` = none).
    """

    name: str
    dtype: str
    axis: str
    trailing: tuple[int, ...] = ()
    state: bool = False
    symbols: str | None = None
    leading: tuple[int, ...] = ()

    def stated(self) -> tuple[Field, Field]:
        r"""This column and its sibling fact-state column ``<name>_state``."""
        return self, Field(f"{self.name}_state", "uint8", self.axis, state=True)


@dataclass(frozen=True)
class Schema:
    r"""The declared columns of one value class and the row axes they live on."""

    fields: tuple[Field, ...]
    axes: tuple[str, ...]

    def digest(self) -> str:
        r"""sha256 over every field's declaration, independent of field order (columns are addressed by name)."""
        text = ""
        for field in sorted(self.fields, key=lambda field: field.name):
            parts = (field.name, field.dtype, field.axis, field.trailing, field.leading, field.state, field.symbols)
            text += "\x1f".join(map(str, parts)) + "\x1e"
        return hashlib.sha256(("|".join(self.axes) + "\x1d" + text).encode()).hexdigest()


class ArrayRecord:
    r"""A frozen record of read-only columns and symbol tables; ``from_arrays`` of each subclass constructs it."""

    SCHEMA: ClassVar[Schema]
    __slots__ = ("_columns", "_symbols", "_cache")

    @classmethod
    def _from_published(cls, columns: Mapping[str, NDArray], symbols: Mapping[str, tuple[str, ...]]):
        value = object.__new__(cls)
        for array in columns.values():
            array.flags.writeable = False  # unpickled arrays come back writable
        object.__setattr__(value, "_columns", MappingProxyType(dict(columns)))
        object.__setattr__(value, "_symbols", MappingProxyType({k: tuple(v) for k, v in symbols.items()}))
        object.__setattr__(value, "_cache", {})
        return value

    def __getattr__(self, name: str) -> NDArray:
        columns = object.__getattribute__(self, "_columns")
        if name in columns:
            return columns[name]
        raise AttributeError(f"{type(self).__name__} has no column {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")

    def __reduce__(self):
        return type(self)._from_published, (dict(self._columns), dict(self._symbols))

    def arrays(self) -> Mapping[str, NDArray]:
        r"""The read-only columns by name."""
        return self._columns

    def symbols(self) -> Mapping[str, tuple[str, ...]]:
        r"""The symbol families by name."""
        return self._symbols


def check_rows(
    argument: str, values, rows: int, *, low: int = 0, shape: tuple | None = None, code: str = "value_invalid"
) -> NDArray[np.int64]:
    r"""A caller's indices as integers in ``[low, rows)`` (a negative ``low`` admits an encoding its entry documents,
    ``-1`` none) and, with ``shape``, of exactly that rank and those dimensions (``None`` any length), else
    ``InvalidRequestError(code)``: never a negative index reading from the end, never a shape numpy would broadcast."""
    array = np.asarray(values)
    if shape is not None and (
        array.ndim != len(shape) or any(want is not None and want != got for want, got in zip(shape, array.shape))
    ):
        fail(code, RequestRef(argument, str(array.shape)), invariant="shape", expected=str(shape))
    if array.size and (array.dtype.kind not in "iu" or array.min() < low or array.max() >= rows):
        fail(code, RequestRef(argument, str(array.ravel()[:8].tolist())), invariant="index_range", low=low, rows=rows)
    return array.astype(np.int64)


def check_mask(argument: str, values, rows: int, *, code: str = "value_invalid") -> NDArray[np.bool_]:
    r"""A caller's mask as booleans, one per row, else ``InvalidRequestError(code)``: never indices read as a mask."""
    array = np.asarray(values)
    if array.dtype != bool or array.shape != (rows,):
        fail(code, RequestRef(argument, f"{array.dtype}{list(array.shape)}"), invariant="mask", rows=rows)
    return array


def fits(values, dtype) -> bool:
    r"""Whether an integer or flag column of ``dtype`` holds every value exactly: integral and within the target's own
    range (``np.iinfo``, a flag 0 or 1), decided on the values themselves, never by a cast and back. A float target,
    and a source that is no number, are the cast's to judge."""
    array, dtype = np.asarray(values), np.dtype(dtype)
    if dtype.kind not in "biu" or array.dtype.kind not in "biuf" or not array.size:
        return True
    if array.dtype.kind == "f" and not (np.isfinite(array).all() and (array == np.trunc(array)).all()):
        return False
    low, high = (0, 1) if dtype.kind == "b" else (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
    return low <= int(array.min()) and int(array.max()) <= high


def occupancy_in_range(
    occupancy: NDArray[np.float64], uncertainty: NDArray[np.float64], uncertainty_state: NDArray[np.uint8]
) -> NDArray[np.bool_]:
    r"""
    Whether each occupancy lies in −3u ≤ occupancy ≤ 1 + 3u (IUCr coreCIF ``_atom_site_occupancy``, its 0.0:1.0 range
    read with its su), u the standard uncertainty where it is KNOWN, else 0: an Unknown su admits no
    tolerance, a KNOWN one is finite and not negative. The comparison is in float64; where u is not 0, a value within
    rounding of a bound is checked again in decimal, on the shortest decimals (``repr``) of the value and of u.
    """
    stated = known(uncertainty_state)
    u = np.where(stated, uncertainty, 0.0)
    valid = ~stated | (np.isfinite(uncertainty) & (uncertainty >= 0))
    low, high = -3 * u, 1 + 3 * u
    inside = valid & (occupancy >= low) & (occupancy <= high)
    near = np.isclose(occupancy, low, rtol=1e-12, atol=0) | np.isclose(occupancy, high, rtol=1e-12, atol=0)
    near &= valid & (u != 0)
    for i in np.flatnonzero(near).tolist():
        value, su = Decimal(repr(float(occupancy[i]))), Decimal(repr(float(u[i])))
        inside[i] = -3 * su <= value <= 1 + 3 * su
    return inside


def publish(
    columns: Mapping[str, NDArray], symbols: Mapping[str, tuple[str, ...]], schema: Schema
) -> Mapping[str, NDArray]:
    r"""
    Fresh read-only, C-contiguous copies of ``columns`` in the schema's dtypes.

    A value its integer or flag column cannot hold exactly (``fits``) is ``UnsupportedError(precision_loss)``; a symbol
    family a code column declares that ``symbols`` lacks or holds as anything but a tuple of texts (other families are
    accepted), a column whose fixed dimensions are not the declared ones, or columns whose row counts disagree on one
    axis, are ``InvalidRequestError(value_invalid)``. The copy is always taken, so no buffer the caller keeps can change
    a published value.
    """

    for family in sorted({field.symbols for field in schema.fields if field.symbols is not None}):
        table = symbols.get(family)
        if not isinstance(table, tuple) or not all(isinstance(text, str) for text in table):
            fail("value_invalid", RequestRef("symbols", family), invariant="family")
    published: dict[str, NDArray] = {}
    rows: dict[str, int] = {}
    for field in schema.fields:
        array, dtype = np.asarray(columns[field.name]), np.dtype(field.dtype)
        lead, shape = len(field.leading), array.shape
        expected = (*field.leading, field.axis, *field.trailing)
        if len(shape) != len(expected) or shape[:lead] + shape[lead + 1 :] != field.leading + field.trailing:
            fail("value_invalid", RequestRef(field.name, str(shape)), expected=expected)
        if not fits(array, dtype):
            fail("precision_loss", RequestRef(field.name, str(array.dtype)), dtype=field.dtype)
        copy = np.array(array, dtype=dtype, order="C", copy=True)
        copy.flags.writeable = False
        published[field.name] = copy
        size = copy.shape[lead] - field.axis.endswith("+1")
        axis = field.axis.removesuffix("+1")
        if rows.setdefault(axis, size) != size:
            fail("value_invalid", RequestRef(field.name, str(copy.shape)), axis=axis, rows=rows[axis])
    return MappingProxyType(published)


# The tables below restate facts of the two schemas that the validators need (a test locks them to the schemas).
#: The symbol family each code column indexes.
_SYSTEM_CODES = {
    "entity_symbol": "entity",
    "entity_polymer_type": "polymer_type",
    "instance_symbol": "instance",
    "component_symbol": "component",
    "component_parent": "parent",
    "component_definition_parent": "parent",
    "atom_symbol": "atom",
    "bond_connection_type": "connection_type",
    "bond_declared_leaving": "atom",
    "external_partner_instance": "instance",
    "external_partner_atom": "atom",
    "external_connection_type": "connection_type",
    "external_declared_leaving": "atom",
}
_STRUCTURE_CODES = {
    "atom_altloc": "altloc",
    "component_author_text": "author_number",
    "component_insertion": "insertion",
    "instance_copy_operators": "operator",
    "instance_source_asym": "asym",
    "instance_author_chain": "chain",
    "space_group": "space_group",
    "method": "method",
    "operation_id": "operation",
}
#: Code columns that may hold ``-1`` (no symbol).
_NULLABLE = {
    "entity_polymer_type",
    "bond_connection_type",
    "external_connection_type",
    "atom_altloc",
    "component_author_text",
    "component_insertion",
    "instance_copy_operators",
    "instance_source_asym",
    "instance_author_chain",
    "space_group",
}
#: Each CSR pointer and a column of the axis it points into.
_SYSTEM_POINTERS = {
    "system_entity_ptr": "entity_symbol",
    "system_instance_ptr": "instance_symbol",
    "instance_component_ptr": "component_symbol",
    "component_atom_ptr": "atom_symbol",
    "component_parent_ptr": "component_parent",
    "component_definition_parent_ptr": "component_definition_parent",
    "bond_declared_leaving_ptr": "bond_declared_leaving",
    "external_declared_leaving_ptr": "external_declared_leaving",
    "stereogroup_ptr": "stereogroup_member",
}
_STRUCTURE_POINTERS = {
    "system_method_ptr": "method",
    "system_resolution_ptr": "resolution",
    "operation_ptr": "operation_id",
}
#: State columns that do not state the column ``<name>`` of ``<name>_state``.
_STATED = {
    "transform_state": ("transform_rotation", "transform_translation"),
    "stereocentre_state": ("stereocentre_parity", "stereocentre_neighbors"),
    "stereobond_state": ("stereobond_value", "stereobond_reference"),
}
#: Index columns whose canonical zero is ``-1``.
_INDEX_ZERO = {"stereocentre_neighbors", "stereobond_reference"}
#: The values of each enumerated column: its ``IntEnum`` (a direction is a sign, a declared leaving side 0 names the
#: first atom); a stated column's value is checked where it is KNOWN (canonical zero elsewhere).
_SYSTEM_ENUMS = {
    "entity_type": range(1, 6),  # EntityType
    "entity_nstd_linkage": range(1, 3),  # NstdLinkage
    "component_definition_scope": range(6),  # DefinitionScope
    "component_monomer_type": range(4),  # MonomerType
    "atom_fate": range(4),  # AtomFate
    "atom_endpoint_state": range(7),  # 0 (not an endpoint) or an EndpointState
    "atom_designation": range(3),  # Designation
    "reference_pool_source": range(4),  # ReferenceSource
    "bond_kind": range(1, 3),  # BondKind
    "bond_direction": range(-1, 2),
    "bond_origin": range(6),  # BondOrigin
    "bond_declared_leaving_side": range(2),
    "external_kind": range(1, 3),
    "external_direction": range(-1, 2),
    "external_origin": range(6),
    "external_declared_leaving_side": range(2),
    "stereogroup_kind": range(3),  # StereoGroupKind
}
_STRUCTURE_ENUMS = {
    "coordinate_evidence": range(5),  # CoordinateEvidence
    "instance_copy_kind": range(4),  # CopyKind
    "resolution_kind": range(5),  # ResolutionKind
    "operation_numbering": range(3),  # OperationNumbering
}

_LEGAL_STATE = np.zeros(256, dtype=bool)
_LEGAL_STATE[sorted(ALLOWED_STATES)] = True
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_UNKNOWN_NONE = state(Resolution.UNKNOWN, Basis.NONE)
_UNKNOWN_DERIVED = state(Resolution.UNKNOWN, Basis.DERIVED)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_LEFT, _UNDETERMINED, _UNOBSERVED_ENDPOINT = 3, 1, 2  # AtomFate
_OPEN_ENDPOINT = (5, 6)  # EndpointState UNKNOWN, CONFLICT
_DECIMAL = re.compile(r"[+-]?[0-9]+")


def _check(bad: NDArray[np.bool_], column: str, invariant: str) -> None:
    r"""Fail ``value_invalid`` at the first row of ``column`` where ``bad`` holds (in any element of the row)."""
    bad = np.asarray(bad, dtype=bool)
    if not bad.any():
        return
    if bad.ndim > 1:
        bad = bad.reshape(len(bad), -1).any(axis=1)
    row = int(np.flatnonzero(bad)[0])
    fail("value_invalid", StructureRef(item=column), invariant=invariant, row=row, count=int(bad.sum()))


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _repeated(*keys: NDArray) -> NDArray[np.bool_]:
    r"""Rows equal, on every key, to another row."""
    count = len(keys[0])
    out = np.zeros(count, dtype=bool)
    if count < 2:
        return out
    order = np.lexsort(tuple(np.asarray(key, dtype=np.int64) for key in keys[::-1]))
    same = np.ones(count - 1, dtype=bool)
    for key in keys:
        ordered = np.asarray(key)[order]
        same &= ordered[1:] == ordered[:-1]
    out[order[1:][same]] = True
    out[order[:-1][same]] = True
    return out


def _crystallographic(rotation: NDArray) -> NDArray[np.bool_]:
    r"""Integer matrices of determinant ±1 and finite order (R¹² = I), exactly (object arithmetic, no overflow)."""
    if not len(rotation):
        return np.ones(0, dtype=bool)
    exact = rotation.astype(np.int64).astype(object)
    square = exact @ exact
    fourth = square @ square
    twelfth = fourth @ fourth @ fourth
    finite = (twelfth == np.eye(3, dtype=np.int64).astype(object)).reshape(len(rotation), 9).all(axis=1)
    determinant = np.rint(np.linalg.det(rotation.astype(np.float64)))
    return np.isin(determinant, (-1, 1)) & finite.astype(bool)


def _unique_symbols(symbols: Mapping[str, tuple[str, ...]]) -> None:
    for family, table in symbols.items():
        if len(set(table)) != len(table):
            fail("value_invalid", RequestRef("symbols", family), invariant="unique_symbols")


def _csr(columns: Mapping[str, NDArray], pointers: Mapping[str, str]) -> None:
    for name, target in pointers.items():
        pointer = columns[name]
        if pointer[0] != 0 or pointer[-1] != len(columns[target]) or (np.diff(pointer) < 0).any():
            fail("value_invalid", StructureRef(item=name), invariant="csr")


def _facts(columns: Mapping[str, NDArray], symbols: Mapping[str, tuple[str, ...]], codes: Mapping[str, str]) -> None:
    r"""Legal state bytes; canonical zero where a fact is not KNOWN; every code names a symbol of its family."""
    for name, states in columns.items():
        if not name.endswith("_state") or name == "atom_endpoint_state":  # an EndpointState, not a fact state
            continue
        _check(~_LEGAL_STATE[states], name, "state_byte")
        unknown = resolution(states) != Resolution.KNOWN
        if name == "frame_state":  # an INAPPLICABLE frame may keep the transform it rests on (T7)
            unknown &= resolution(states) != Resolution.INAPPLICABLE
        for value in _STATED.get(name, (name.removesuffix("_state"),)):
            payload = columns[value].reshape(len(states), int(np.prod(columns[value].shape[1:])))
            zero = -1 if value in codes or value in _INDEX_ZERO else 0
            _check(unknown[:, None] & (payload != zero), value, "canonical_zero")
            if value in codes:
                _check(~unknown[:, None] & (payload < 0), value, "known_code")
    for name, family in codes.items():
        lowest = -1 if name in _NULLABLE else 0
        _check((columns[name] < lowest) | (columns[name] >= len(symbols[family])), name, "code_range")


def _enums(columns: Mapping[str, NDArray], enums: Mapping[str, range]) -> None:
    r"""Every enumerated value is one of its column's values (where its fact is KNOWN, for a stated column)."""
    for name, values in enums.items():
        value, states = columns[name].astype(np.int64), columns.get(f"{name}_state")
        scope = np.ones(len(value), dtype=bool) if states is None else known(states)
        _check(scope & ((value < values.start) | (value >= values.stop)), name, "enum")


def validate_system(columns: Mapping[str, NDArray], symbols: Mapping[str, tuple[str, ...]]) -> None:
    r"""The invariants of a ``MolecularSystem``."""

    _unique_symbols(symbols)
    _csr(columns, _SYSTEM_POINTERS)
    _facts(columns, symbols, _SYSTEM_CODES)
    _enums(columns, _SYSTEM_ENUMS)
    atom_component = _owner(columns["component_atom_ptr"])
    component_instance = _owner(columns["instance_component_ptr"])
    instance_system = _owner(columns["system_instance_ptr"])
    entity_system = _owner(columns["system_entity_ptr"])
    atom_system = instance_system[component_instance[atom_component]]
    n = len(atom_component)

    # S3: an instance's entity belongs to its system; entity and instance symbols are unique per system
    entity = columns["instance_entity"].astype(np.int64)
    outside = (entity < 0) | (entity >= len(entity_system))
    entity_of = np.append(entity_system, -1)[np.where(outside, -1, entity)]
    _check(entity_of != instance_system, "instance_entity", "own_system")
    _check(_repeated(entity_system, columns["entity_symbol"]), "entity_symbol", "unique_per_system")
    _check(_repeated(instance_system, columns["instance_symbol"]), "instance_symbol", "unique_per_system")

    # S7: atoms
    _check(_repeated(atom_component, columns["atom_symbol"]), "atom_symbol", "unique_per_component")
    z, z_known = columns["atomic_number"].astype(np.int64), known(columns["atomic_number_state"])
    _check(z_known & ((z < 1) | (z > 118)), "atomic_number", "element")
    isotope = columns["isotope"].astype(np.int64)
    _check((isotope > 0) & z_known & (isotope < z), "isotope", "mass_number")
    for name in ("implicit_hydrogens", "atom_valence"):
        _check(known(columns[f"{name}_state"]) & (columns[name] < 0), name, "non_negative")
    _check(columns["atom_fate"] == _LEFT, "atom_fate", "left_atom")  # left atoms do not exist in a value
    reference_row = columns["atom_reference_row"].astype(np.int64)
    pool_source = columns["reference_pool_source"].astype(np.int64)
    _check((reference_row < -1) | (reference_row >= len(pool_source)), "atom_reference_row", "range")

    # S8: edges are low → high, sorted, unique and inside one system
    first, second = (end.astype(np.int64) for end in columns["edge_index"])
    _check((first < 0) | (first >= second) | (second >= n), "edge_index", "edge_order")
    keys = first * max(n, 1) + second
    _check(np.diff(keys, prepend=-1) <= 0, "edge_index", "sorted_unique")
    _check(atom_system[first] != atom_system[second], "edge_index", "one_system")

    # S9: external connections
    external_atom = columns["external_atom"].astype(np.int64)
    transform = columns["external_transform"].astype(np.int64)
    _check((external_atom < 0) | (external_atom >= n), "external_atom", "range")
    _check((transform < 0) | (transform >= len(columns["transform_state"])), "external_transform", "range")
    _check(columns["external_partner_component"] < 0, "external_partner_component", "range")
    _externals(columns, external_atom, transform, atom_component, component_instance, instance_system)

    # S10: transforms: row 0 is the identity; a KNOWN rotation is a crystallographic operation
    rotation, translation = columns["transform_rotation"], columns["transform_translation"]
    states = columns["transform_state"]
    identity = len(states) and states[0] == _KNOWN_DEFINED and columns["transform_code"][0] == -1
    if not (identity and (rotation[0] == np.eye(3)).all() and not translation[0].any()):
        fail("value_invalid", StructureRef(item="transform_state"), invariant="identity_row")
    _check(known(states) & ~_crystallographic(rotation), "transform_rotation", "crystallographic")

    # S11: components
    count = len(columns["component_symbol"])
    port = columns["component_port"].astype(np.int64)
    start, stop = columns["component_atom_ptr"][:-1, None], columns["component_atom_ptr"][1:, None]
    _check((port != -1) & ((port < start) | (port >= stop)), "component_port", "own_atom")
    scope, digest = columns["component_definition_scope"], columns["component_definition_digest"]
    _check((scope == 0) != (digest == 0), "component_definition_digest", "scope")
    _claims(columns, atom_component, count, first, second, external_atom)

    # S12: reference conformers: one source per component, never NONE for a referenced row
    _check(~np.isfinite(columns["reference_pool_coordinates"]), "reference_pool_coordinates", "finite")
    referenced = reference_row >= 0
    row_source = np.append(pool_source, -1)[reference_row]
    _check(referenced & (row_source == 0), "atom_reference_row", "source_none")
    sources = np.zeros((count, _SYSTEM_ENUMS["reference_pool_source"].stop), dtype=bool)  # its range checked
    sources[atom_component[referenced], row_source[referenced]] = True
    mixed = sources.sum(axis=1) > 1
    _check(referenced & mixed[atom_component], "atom_reference_row", "one_source_per_component")

    # S13: stereo
    _stereo(columns, keys, n, external_atom)


def _externals(columns, external_atom, transform, atom_component, component_instance, instance_system) -> None:
    r"""S9: external rows are unique; an identity external whose partner is in its own system is an edge."""
    if not len(external_atom):
        return
    partner = (
        columns["external_partner_instance"],
        columns["external_partner_component"],
        columns["external_partner_atom"],
    )
    _check(_repeated(external_atom, *partner, transform), "external_atom", "duplicate")
    rotation, translation, states = (
        columns["transform_rotation"],
        columns["transform_translation"],
        columns["transform_state"],
    )
    identity = known(states) & (rotation == np.eye(3)).all(axis=(1, 2)) & (translation == 0).all(axis=1)
    atom_instance = component_instance[atom_component]
    ordinal = atom_component - columns["instance_component_ptr"][atom_instance]
    present = np.stack(
        (instance_system[atom_instance], columns["instance_symbol"][atom_instance], ordinal, columns["atom_symbol"]),
        axis=1,
    ).astype(np.int64)
    system = instance_system[atom_instance[external_atom]]
    named = np.stack((system, *partner), axis=1).astype(np.int64)
    _, inverse = np.unique(np.concatenate((present, named)), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    found = np.isin(inverse[len(present) :], inverse[: len(present)])
    _check(identity[transform] & found, "external_atom", "identity_partner_present")


def _claims(columns, atom_component, count, first, second, external_atom) -> None:
    r"""S11: a completeness claim needs a definition and no atom or connection that would leave it undetermined."""
    fate, heavy = columns["atom_fate"], columns["atomic_number"] != 1
    undetermined, unobserved = fate == _UNDETERMINED, fate == _UNOBSERVED_ENDPOINT
    unclassified = np.zeros(len(fate), dtype=bool)
    edge_unknown = columns["bond_kind_state"] == _UNKNOWN_NONE
    unclassified[first[edge_unknown]] = True
    unclassified[second[edge_unknown]] = True
    unclassified[external_atom[columns["external_kind_state"] == _UNKNOWN_NONE]] = True
    open_endpoint = np.isin(columns["atom_endpoint_state"], _OPEN_ENDPOINT)

    def any_atom(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return np.bincount(atom_component[mask], minlength=count) > 0

    undefined = columns["component_definition_scope"] == 0
    blocked = {
        "component_topology_complete": undefined | any_atom(undetermined | unobserved | open_endpoint | unclassified),
        "component_heavy_atoms_complete": undefined | any_atom((undetermined & heavy) | unobserved),
    }
    for name, mask in blocked.items():
        _check(columns[name] & mask, name, "claim")


def _slots_ok(slots, owners, keys, n, external_atom) -> NDArray[np.bool_]:
    r"""Per slot: an atom bonded to its owner, the implicit slot ``-1`` or an external row ``-(x + 2)`` of the owner."""
    slots, owners = slots.astype(np.int64), owners.astype(np.int64)
    atom = (slots >= 0) & (slots < n)
    query = np.where(atom, np.minimum(slots, owners) * max(n, 1) + np.maximum(slots, owners), -1)
    at = np.searchsorted(keys, query).clip(max=max(len(keys) - 1, 0))  # S8 has checked the keys sorted and unique
    bonded = atom & (keys[at] == query) if len(keys) else np.zeros(query.shape, dtype=bool)
    row = -slots - 2
    external = (slots <= -2) & (row < len(external_atom))
    external &= np.append(external_atom, -1)[np.where(external, row, -1)] == owners
    return bonded | (slots == -1) | external


def _stereo(columns, keys, n, external_atom) -> None:
    r"""S13: stereocentres, stereobonds and stereogroups."""
    centre, parity = columns["stereocentre_atom"], columns["stereocentre_parity"]
    slots, centre_known = columns["stereocentre_neighbors"], known(columns["stereocentre_state"])
    _check((centre < 0) | (centre >= n) | _repeated(centre), "stereocentre_atom", "range_unique")
    _check(centre_known != np.isin(parity, (1, 2)), "stereocentre_parity", "known_parity")
    owners = np.repeat(centre, 4).reshape(-1, 4)
    ordered = np.sort(slots, axis=1)
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != -1)
    bad = ~_slots_ok(slots, owners, keys, n, external_atom) | duplicate.any(axis=1, keepdims=True)
    bad |= ((slots == -1).sum(axis=1) > 1)[:, None]
    _check(centre_known[:, None] & bad, "stereocentre_neighbors", "slots")

    first, second = columns["edge_index"]
    edge, value = columns["stereobond_edge"].astype(np.int64), columns["stereobond_value"]
    references, bond_known = columns["stereobond_reference"], known(columns["stereobond_state"])
    _check((edge < 0) | (edge >= len(first)) | _repeated(edge), "stereobond_edge", "range_unique")
    _check(bond_known != np.isin(value, (1, 2)), "stereobond_value", "known_value")
    ends = np.stack((first[edge], second[edge]), axis=1)
    ok = _slots_ok(references, ends, keys, n, external_atom) & (references != ends[:, ::-1]) & (references != -1)
    _check(bond_known[:, None] & ~ok, "stereobond_reference", "references")

    member = columns["stereogroup_member"]
    _check((member < 0) | (member >= len(centre)) | _repeated(member), "stereogroup_member", "members")


def validate_structure(
    system: Mapping[str, NDArray], columns: Mapping[str, NDArray], symbols: Mapping[str, tuple[str, ...]]
) -> None:
    r"""The invariants of a ``Structure`` over its system's columns."""

    atom_component = _owner(system["component_atom_ptr"])
    component_instance = _owner(system["instance_component_ptr"])
    instance_system = _owner(system["system_instance_ptr"])
    atom_system = instance_system[component_instance[atom_component]]
    n, instances = len(atom_component), len(system["instance_symbol"])
    for name, size in (
        ("coordinates", n),
        ("component_author_text", len(component_instance)),
        ("instance_copy_kind", instances),
        ("system_model_number", len(system["system_instance_ptr"]) - 1),
    ):
        if len(columns[name]) != size:
            fail("value_invalid", StructureRef(item=name), invariant="layout", rows=len(columns[name]), expected=size)
    _unique_symbols(symbols)
    _csr(columns, _STRUCTURE_POINTERS)
    _facts(columns, symbols, _STRUCTURE_CODES)
    _enums(columns, _STRUCTURE_ENUMS)

    # T2: coordinates and their printed quantum are finite, canonical zero where their evidence is NONE; an OBSERVED
    # coordinate of a deposited instance is its source's token, so it has the token's printed unit
    coordinates, evidence = columns["coordinates"], columns["coordinate_evidence"]
    _check(~np.isfinite(coordinates), "coordinates", "finite")
    _check((evidence == 0)[:, None] & (coordinates != 0), "coordinates", "canonical_zero")
    quantum = columns["coordinate_quantum"]
    _check(~np.isfinite(quantum) | (quantum < 0), "coordinate_quantum", "finite")
    _check((evidence == 0) & (quantum != 0), "coordinate_quantum", "canonical_zero")
    deposited = columns["instance_copy_kind"][component_instance[atom_component]] == 0  # CopyKind.DEPOSITED
    _check((evidence == 1) & deposited & (quantum == 0), "coordinate_quantum", "printed")

    # T3: observations are in range, and never inherited by coordinates that were not observed; an atom
    # without coordinates states none in a system without a source document. The value stores no source row, so an
    # atom without coordinates beside a document is not checked
    no_document = ~columns["source_sha256"].any(axis=1)
    unobserved = np.isin(evidence, (2, 3, 4)) | ((evidence == 0) & no_document[atom_system])
    su, su_state = columns["occupancy_uncertainty"], columns["occupancy_uncertainty_state"]
    in_range = occupancy_in_range(columns["occupancy"], su, su_state)
    _check(known(columns["occupancy_state"]) & ~in_range, "occupancy", "range")
    _check(known(su_state) & ~(np.isfinite(su) & (su >= 0)), "occupancy_uncertainty", "range")
    _check(known(columns["b_factor_state"]) & ~np.isfinite(columns["b_factor"]), "b_factor", "range")
    for name in ("occupancy", "occupancy_uncertainty", "b_factor"):
        _check(unobserved & (columns[f"{name}_state"] != _INAPPLICABLE_DEFINED), name, "never_inherited")

    # T4: without a source document no atom_site charge is stated; beside one, the value stores no source row
    # to check a charge against
    charge_state = columns["atom_site_charge_state"]
    _check((charge_state != _UNKNOWN_NONE) & no_document[atom_system], "atom_site_charge_state", "no_document")

    # T5: the author number is the integer its source's grammar reads the author text as, decided where the source is
    # read (a decimal, a PDB hybrid-36 resSeq): a KNOWN number has its text, a text without a reading is
    # UNKNOWN×DERIVED, and a decimal text reads as its value in every grammar (hybrid-36 starts with no digit or sign)
    text, number_state = columns["component_author_text"], columns["component_author_number_state"]
    tokens = symbols["author_number"]
    decimal = np.array([_DECIMAL.fullmatch(t) is not None for t in tokens] + [False], dtype=bool)  # -1: no text
    readings = [int(t) if d and -(2**63) <= int(t) < 2**63 else None for t, d in zip(tokens, decimal)] + [None]
    fits = np.array([r is not None for r in readings], dtype=bool)
    value = np.array([0 if r is None else r for r in readings], dtype=np.int64)
    stated = text >= 0
    bad = known(number_state) & ~stated
    bad |= stated & ~known(number_state) & (number_state != _UNKNOWN_DERIVED)
    bad |= known(number_state) & decimal[text] & (~fits[text] | (columns["component_author_number"] != value[text]))
    _check(bad, "component_author_number", "author_text")

    # T6: a copy names operators exactly when an assembly or crystal generated it; copy ids are unique per system
    kind, operators, asym = (
        columns["instance_copy_kind"],
        columns["instance_copy_operators"],
        columns["instance_source_asym"],
    )
    plain = np.isin(kind, (0, 3))  # CopyKind DEPOSITED, RECIPE
    _check((plain & (operators != -1)) | (~plain & (operators < 0)), "instance_copy_operators", "copy_kind")
    named = asym >= 0
    duplicate = np.zeros(instances, dtype=bool)
    duplicate[named] = _repeated(instance_system[named], kind[named], operators[named], asym[named])
    _check(duplicate, "instance_source_asym", "copy_unique")

    # T7: metadata travelling with the coordinates
    resolution_values = columns["resolution"]
    bad = known(columns["resolution_state"]) & ~(np.isfinite(resolution_values) & (resolution_values > 0))
    _check(bad, "resolution", "positive")
    cell = columns["cell"]
    lengths, angles = cell[:, :3], cell[:, 3:]
    geometric = (lengths > 0).all(axis=1) & ((angles > 0) & (angles < 180)).all(axis=1)
    _check(known(columns["cell_state"]) & ~geometric, "cell", "cell")
    frame, frame_state = columns["frame"], columns["frame_state"]  # [A | u]: a lattice basis is right-handed
    finite = np.isfinite(frame).all(axis=(1, 2))
    linear = np.where(finite[:, None, None], frame[:, :, :3], 0.0)
    determinant = np.linalg.det(linear) if len(frame) else np.zeros(0)
    # held where KNOWN, and where INAPPLICABLE by the source's transform: no cell value stated (UNKNOWN or
    # INAPPLICABLE); else canonical zero
    inapplicable = resolution(frame_state) == Resolution.INAPPLICABLE
    unstated = np.isin(resolution(columns["cell_state"]), (Resolution.UNKNOWN, Resolution.INAPPLICABLE))
    held = known(frame_state) | inapplicable & unstated & (frame != 0).any(axis=(1, 2))
    _check(~held[:, None] & (frame.reshape(len(frame), -1) != 0), "frame", "canonical_zero")
    _check(held & ~(finite & (determinant > 0)), "frame", "frame")
    for name, stated in (("cell", known(columns["cell_state"])), ("frame", held)):
        quantum = columns[f"{name}_quantum"].reshape(len(cell), -1)
        _check(~np.isfinite(quantum) | (quantum < 0), f"{name}_quantum", "finite")
        _check(~stated[:, None] & (quantum != 0), f"{name}_quantum", "canonical_zero")
    _check(~_crystallographic(columns["operation_rotation"]), "operation_rotation", "crystallographic")
    operation_system = _owner(columns["operation_ptr"])
    _check(_repeated(operation_system, columns["operation_id"]), "operation_id", "unique_per_system")
    # rows are the file's own table (FILE); an ITA list is the setting the space group names, regenerated, not stored
    numbering, empty = columns["operation_numbering"], np.diff(columns["operation_ptr"]) == 0
    _check((numbering == 1) == empty, "operation_numbering", "segment")  # OperationNumbering FILE
    _check((numbering == 2) & ~known(columns["space_group_state"]), "operation_numbering", "setting")  # ITA
