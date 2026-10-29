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
Component definitions as values, the decoded-definition input type, the versioned artifact and its process-wide
library, and the one definition pass with scope precedence.

``ComponentItems`` arrive decoded by io's codec table: state bytes are the codec's ``read`` map
(KNOWN×DECLARED, ``?`` UNKNOWN×DECLARED, ``.`` INAPPLICABLE×DECLARED, KNOWN×DEFAULT for an absent item with a verified
default whose payload io fills, else UNKNOWN×NONE), numbers and elements are decoded, payloads are canonical zero, so
chemistry never parses a CIF token. A definition's items carry basis REFERENCE in any scope (a stated ``?`` or ``.``
becomes UNKNOWN×REFERENCE). Conformer, parity and cis/trans rows are derived once per definition, never per
occurrence; its ports and findings are derivations of it (``fragments``, memoised per definition). A definition
represents no radical (``smiles.read_smiles`` refuses a stated one and one RDKit assigns to a non-metal atom; a
metal's RDKit count is not consulted; the CCD states none).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import DefinitionScope, ReferenceSource
from ..data.stereo import bond_configuration, parity_from_coordinates, permutation_parity
from ..errors import CODE_NAMES, CODES, PayloadKind, RequestRef, SourceRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, basis, known, resolution, state
from ..reference.ccd.source import (
    ARTIFACT_FILENAME,
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    PROVENANCE_FILENAME,
    resolve_snapshot,
)
from .fragments import findings

if TYPE_CHECKING:
    from os import PathLike

    from .rdkit import StereoArbitration

ARTIFACT_SCHEMA = "ofoldx.ccd/7"

_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
_STATED = state(Resolution.KNOWN, Basis.REFERENCE)
_NULL = state(Resolution.UNKNOWN, Basis.REFERENCE)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_STATED_NULLS = (state(Resolution.UNKNOWN, Basis.DECLARED), state(Resolution.INAPPLICABLE, Basis.DECLARED))  # ? and .

# item state (the codec's read map) -> definition state; absent items and dictionary defaults keep their state
_DEFINITION = np.arange(256, dtype=np.uint8)
_DEFINITION[state(Resolution.KNOWN, Basis.DECLARED)] = _STATED
_DEFINITION[state(Resolution.UNKNOWN, Basis.DECLARED)] = _NULL
_DEFINITION[state(Resolution.INAPPLICABLE, Basis.DECLARED)] = _NULL

_FLAGS = {b"N": 1, b"Y": 2}
_LABELS = {b"R": "R", b"S": "S", b"E": "E", b"Z": "Z", b"N": "N"}  # N: no configuration declared, a fact unlike ?
_CONFORMERS = (
    ("ideal", "pdbx_model_Cartn_{}_ideal", ReferenceSource.IDEAL),
    ("model", "model_Cartn_{}", ReferenceSource.MODEL),
)

# draft column -> its dictionary items, named by file_scope_null_over_archive_value
_ITEMS = {
    "chem_comp_type": ("_chem_comp.type",),
    "one_letter_code": ("_chem_comp.one_letter_code",),
    "parents": ("_chem_comp.mon_nstd_parent_comp_id",),
    "element": ("_chem_comp_atom.type_symbol",),
    "isotope": ("_chem_comp_atom.type_symbol",),
    "charge": ("_chem_comp_atom.charge",),
    "aromatic": ("_chem_comp_atom.pdbx_aromatic_flag",),
    "leaving_flag": ("_chem_comp_atom.pdbx_leaving_atom_flag",),
    "terminal_flag": ("_chem_comp_atom.pdbx_n_terminal_atom_flag", "_chem_comp_atom.pdbx_c_terminal_atom_flag"),
    "cip": ("_chem_comp_atom.pdbx_stereo_config",),
    "bond_order_token": ("_chem_comp_bond.value_order",),
    "bond_aromatic": ("_chem_comp_bond.pdbx_aromatic_flag",),
    "label": ("_chem_comp_bond.pdbx_stereo_config",),
}

# Array fields of ChemicalComponent: dtype, trailing shape and the axis whose rows they hold (atom, bond, stereocentre
# row, cis/trans row, stereo group, group member; ``stereo_group_ptr`` holds one row more than its groups). The same
# table types the artifact columns.
_ARRAYS = {
    "element": (np.uint8, (), "atom"),
    "isotope": (np.uint16, (), "atom"),
    "charge": (np.int8, (), "atom"),
    "charge_state": (np.uint8, (), "atom"),
    "aromatic": (np.bool_, (), "atom"),
    "aromatic_state": (np.uint8, (), "atom"),
    "leaving_flag": (np.uint8, (), "atom"),
    "terminal_flag": (np.uint8, (2,), "atom"),
    "reference": (np.float32, (3,), "atom"),
    "reference_quantum": (np.float32, (), "atom"),
    "bond_atoms": (np.int16, (2,), "bond"),
    "bond_order": (np.uint8, (), "bond"),
    "bond_order_state": (np.uint8, (), "bond"),
    "bond_order_token": (np.uint8, (), "bond"),
    "bond_aromatic": (np.bool_, (), "bond"),
    "bond_aromatic_state": (np.uint8, (), "bond"),
    "stereo_centre": (np.int16, (), "stereocentre"),
    "stereo_parity": (np.int8, (), "stereocentre"),
    "stereo_neighbors": (np.int16, (4,), "stereocentre"),
    "stereo_basis": (np.uint8, (), "stereocentre"),
    "bond_stereo_bond": (np.int16, (), "cis_trans"),
    "bond_stereo_value": (np.int8, (), "cis_trans"),
    "bond_stereo_reference": (np.int16, (2,), "cis_trans"),
    "stereo_group_kind": (np.uint8, (), "group"),
    "stereo_group_ptr": (np.int64, (), "group"),
    "stereo_group_member": (np.int16, (), "member"),
}
_AXES = tuple(dict.fromkeys(axis for _, _, axis in _ARRAYS.values()))


class StereoBasis(IntEnum):
    r"""What a stereocentre row's parity rests on."""

    CONFORMER = 1  # the definition's coherent conformer
    DESCRIPTOR_VERIFIED = 2  # the stated label, which agrees with the component's InChI descriptor where the conformer
    # does not: a check of the configuration against that descriptor, no proof of the 3-D conformer
    STATED = 3  # SMILES/CX tags
    REFUSED = 4  # a configuration exists and nothing settles it (parity 0)


class OrderToken(IntEnum):
    r"""A stated bond-order token (``_struct_conn.pdbx_value_order``, ``value_order``): SING–QUAD carry their order;
    AROM, DELO, PI and POLY state no order."""

    NONE = 0
    SING = 1
    DOUB = 2
    TRIP = 3
    QUAD = 4
    AROM = 5
    DELO = 6
    PI = 7
    POLY = 8


_TOKENS = {token.name.encode(): int(token) for token in OrderToken if token}  # AROM–POLY: no order, never 1.5


def _invalid(where: str, found: str) -> NoReturn:
    fail("value_invalid", RequestRef(where, found))


@dataclass(frozen=True, eq=False)
class ComponentItems:
    r"""
    The decoded ``chem_comp*`` rows of one component in one scope, items keyed by name without the category prefix:
    ``component`` the ``_chem_comp`` tokens and states; ``atoms``/``bonds`` the ``_chem_comp_atom``/``_chem_comp_bond``
    text items as fixed-width bytes and states; ``numbers`` the numeric atom items (charge, coordinates) as value, state
    and printed quantum; ``element`` the atomic number, isotope and state of ``type_symbol`` (``D``/``T`` → H, isotope
    2/3); ``descriptors`` the ``_pdbx_chem_comp_descriptor`` rows; ``refs`` one reference per row: the ``_chem_comp``
    row, atom rows, bond rows, descriptor rows.
    """

    component: Mapping[str, tuple[str | None, int]]
    atoms: Mapping[str, tuple[NDArray[np.bytes_], NDArray[np.uint8]]]
    numbers: Mapping[str, tuple[NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64]]]
    element: tuple[NDArray[np.uint8], NDArray[np.uint16], NDArray[np.uint8]]
    bonds: Mapping[str, tuple[NDArray[np.bytes_], NDArray[np.uint8]]]
    descriptors: tuple[tuple[str, str, str], ...]
    refs: SourceRefs


@dataclass(frozen=True, eq=False)
class DefinitionRows:
    r"""The decoded definitions of one source block, keyed by component id."""

    components: Mapping[str, ComponentItems]


@dataclass(frozen=True, eq=False)
class ChemicalComponent:
    r"""
    One resolved definition, arrays read-only. Built by ``define_component``, ``resolve_definitions``, the library and
    ``smiles.read_smiles``; equal by ``(component_id, scope, digest)``. Flags are 0 null, 1 N, 2 Y; ``cip_declared`` is
    each atom's stated label (R, S, or N for none stated; None when unknown), ``bond_stereo_declared`` each stereobond
    row's E/Z label, both provenance only. ``bond_order_token`` is each bond's ``OrderToken``, the stated
    ``value_order`` or a known order's own (NONE for neither): SING–QUAD are the order, AROM–POLY give order 0
    UNKNOWN×REFERENCE, as ``?`` and ``.`` do.
    ``stereo_group_*`` are the enhanced stereo groups a SMILES/CX string states (``data.stereo.StereoGroupKind`` AND/OR,
    CSR over stereocentre rows); a CCD definition has none.
    """

    component_id: str
    scope: DefinitionScope
    chem_comp_type: str | None
    chem_comp_type_state: int
    one_letter_code: str | None
    one_letter_code_state: int
    parents: tuple[str, ...]
    parents_state: int
    atom_names: tuple[str, ...]
    element: NDArray[np.uint8]
    isotope: NDArray[np.uint16]
    charge: NDArray[np.int8]
    charge_state: NDArray[np.uint8]
    aromatic: NDArray[np.bool_]
    aromatic_state: NDArray[np.uint8]
    leaving_flag: NDArray[np.uint8]
    terminal_flag: NDArray[np.uint8]
    reference: NDArray[np.float32]
    reference_source: ReferenceSource
    reference_quantum: NDArray[np.float32]
    bond_atoms: NDArray[np.int16]
    bond_order: NDArray[np.uint8]
    bond_order_state: NDArray[np.uint8]
    bond_order_token: NDArray[np.uint8]
    bond_aromatic: NDArray[np.bool_]
    bond_aromatic_state: NDArray[np.uint8]
    stereo_centre: NDArray[np.int16]
    stereo_parity: NDArray[np.int8]
    stereo_neighbors: NDArray[np.int16]
    stereo_basis: NDArray[np.uint8]
    cip_declared: tuple[str | None, ...]
    bond_stereo_bond: NDArray[np.int16]
    bond_stereo_value: NDArray[np.int8]
    bond_stereo_reference: NDArray[np.int16]
    bond_stereo_declared: tuple[str | None, ...]
    stereo_group_kind: NDArray[np.uint8]
    stereo_group_ptr: NDArray[np.int64]
    stereo_group_member: NDArray[np.int16]
    digest: int

    def __post_init__(self) -> None:
        for name in _ARRAYS:
            getattr(self, name).flags.writeable = False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChemicalComponent) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple:
        return self.component_id, self.scope, self.digest

    def at_scope(self, scope: DefinitionScope) -> ChemicalComponent:
        r"""The same definition at ``scope`` (the digest excludes scope, so it stays)."""
        return ChemicalComponent(**{f.name: getattr(self, f.name) for f in fields(self)} | {"scope": scope})


def _digest(values: Mapping[str, object]) -> int:
    r"""The int64 prefix (little-endian) of sha256 over the component id and every field but scope and digest, in
    declared order; scope is excluded so identical resolved items compare equal across scopes."""
    sha = hashlib.sha256()
    for f in fields(ChemicalComponent):
        if f.name in ("scope", "digest"):
            continue
        value = values[f.name]
        sha.update(f.name.encode() + b"\0")
        if isinstance(value, np.ndarray):
            little = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
            sha.update(f"{little.dtype.str}{little.shape}".encode() + little.tobytes())
        else:
            sha.update(json.dumps(int(value) if isinstance(value, IntEnum) else value).encode())
    return int.from_bytes(sha.digest()[:8], "little", signed=True)


@dataclass
class _Draft:
    r"""
    One scope's statements, before derivation. ``texts`` holds the ``_chem_comp`` facts as (value, state); ``atoms``
    and ``bonds`` one column per fact, rows in the order of ``names`` and ``pairs`` (a stateful fact ``x`` has an
    ``x_state`` column; ``cip`` and ``label`` are object columns); ``kept`` holds an archive definition's stereocentre
    rows by centre name, as (parity, neighbour names, basis); ``groups`` its AND/OR groups, as (kind, centre names);
    ``kept_bonds`` its cis/trans rows by bond (the two atom names), as (value, reference name per bond atom, label);
    a bond's order is stated as its ``bond_order_token``, from which ``_finish`` derives the order and its state;
    ``said`` per fact of a parsed scope where its source states ``?`` or ``.`` (a text fact: one flag), never where the
    item is absent.
    """

    texts: dict[str, tuple[object, int]]
    names: list[str]
    atoms: dict[str, NDArray]
    reference: NDArray[np.float64]
    quantum: NDArray[np.float64]
    source: ReferenceSource
    pairs: list[tuple[int, int]]
    bonds: dict[str, NDArray]
    kept: dict[str, tuple[int, tuple[str | None, ...], int]] = field(default_factory=dict)
    groups: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    kept_bonds: dict[frozenset[str], tuple[int, dict[str, str], str | None]] = field(default_factory=dict)
    said: dict[str, NDArray[np.bool_]] = field(default_factory=dict)


def _said(columns: Mapping, item: str, rows: int) -> NDArray[np.bool_]:
    r"""Where the source states ``?`` or ``.`` for ``item``; an absent item states neither."""
    return np.isin(columns[item][1], _STATED_NULLS) if item in columns else np.zeros(rows, dtype=bool)


def _text_item(columns: Mapping, item: str, rows: int) -> tuple[list[bytes | None], NDArray[np.uint8]]:
    r"""Stripped tokens (None where not KNOWN) and definition states of one text item."""
    values, raw = (
        columns[item] if item in columns else (np.full(rows, b"", dtype="S1"), np.full(rows, _ABSENT, dtype=np.uint8))
    )
    states = _DEFINITION[raw]
    tokens = [value.strip() if stated else None for value, stated in zip(values.tolist(), known(states))]
    return tokens, states


def _enumerated(columns: Mapping, item: str, rows: int, table: Mapping, default: object) -> tuple[list, NDArray]:
    r"""An enumerated text item looked up in ``table`` case-insensitively (CIF enumerations ignore case)."""
    tokens, states = _text_item(columns, item, rows)
    return [table.get((token or b"").upper(), default) for token in tokens], states


def _flag(columns: Mapping, item: str, rows: int) -> NDArray[np.uint8]:
    return np.array(_enumerated(columns, item, rows, _FLAGS, 0)[0], dtype=np.uint8)


def _boolean(columns: Mapping, item: str, rows: int) -> tuple[NDArray[np.bool_], NDArray[np.uint8]]:
    r"""A Y/N flag as a stateful boolean; a KNOWN token outside the enumeration states nothing readable (``?``)."""
    codes, states = _enumerated(columns, item, rows, _FLAGS, 0)
    flags = np.array(codes, dtype=np.uint8)
    return flags == 2, np.where(known(states) & (flags == 0), _NULL, states).astype(np.uint8)


def _null(values: NDArray) -> NDArray[np.bool_]:
    r"""Where a stateless fact states nothing: None in an object column, 0 elsewhere."""
    return np.equal(values, None) if values.dtype == object else values == 0


# an isotope is stated with its element: isotope 0 is natural abundance, a fact
_NULL_OF = {"isotope": "element"}


def _rank(states: NDArray[np.uint8]) -> NDArray[np.int64]:
    r"""The weight of a statement within one scope: 3 stated, 2 dictionary default, 1 ``?``/``.``/Conflict, 0 absent."""
    stated = known(states)
    default = basis(states) == Basis.DEFAULT
    return np.select([stated & ~default, stated, states != _ABSENT], [3, 2, 1], 0)


def _unify(va: NDArray, sa: NDArray, vb: NDArray, sb: NDArray) -> tuple[NDArray, NDArray, NDArray[np.bool_]]:
    r"""Two statements of one stateful item in one scope: the higher rank stands; two different values of one rank 2
    or 3 are a Conflict (canonical zero)."""
    ra, rb = _rank(sa), _rank(sb)
    conflict = (sa == _CONFLICT) | (sb == _CONFLICT) | ((ra == rb) & (ra >= 2) & (va != vb))
    take = rb > ra
    values = np.where(take, vb, va)
    states = np.where(take, sb, sa).astype(np.uint8)
    values[conflict] = 0
    states[conflict] = _CONFLICT
    return values, states, conflict


def _merge(a: Mapping[str, NDArray], b: Mapping[str, NDArray]) -> tuple[dict, bool, NDArray[np.bool_]]:
    r"""Two aligned statement sets of one scope: stateful facts unified, a stateless fact (flag, element, label) taken
    where one is null. Returns the columns, whether a Conflict arose, and per row whether a stateless fact disagrees."""
    rows = len(next(iter(a.values())))
    out, conflict, disagree = {}, False, np.zeros(rows, dtype=bool)
    for name, va in a.items():
        if f"{name}_state" in a:
            out[name], out[f"{name}_state"], hit = _unify(va, a[f"{name}_state"], b[name], b[f"{name}_state"])
            conflict |= bool(hit.any())
        elif not name.endswith("_state"):
            null_a, null_b = _null(a[_NULL_OF.get(name, name)]), _null(b[_NULL_OF.get(name, name)])
            out[name] = np.where(null_a, b[name], va)
            differs = ~null_a & ~null_b & (va != b[name])
            disagree |= differs.reshape(rows, -1).any(axis=1)
    return out, conflict, disagree


def _collapse(keys: list, columns: dict[str, NDArray]) -> tuple[list, dict, list, tuple[int, int] | None]:
    r"""
    Rows sharing a key as one row (``_merge``), in first-occurrence order. Returns the keys, the columns, the row pairs
    whose statements are in Conflict, and the first row pair whose stateless facts disagree (None when none do).
    """
    first: dict = {}
    for row, key in enumerate(keys):
        first.setdefault(key, row)
    out = {name: values[list(first.values())] for name, values in columns.items()}
    position = {key: i for i, key in enumerate(first)}
    conflicts = []
    for row, key in enumerate(keys):
        if row == first[key]:
            continue
        i = position[key]
        head = {name: values[i : i + 1] for name, values in out.items()}
        merged, conflict, disagree = _merge(head, {name: values[row : row + 1] for name, values in columns.items()})
        if disagree.any():
            return list(first), out, [], (first[key], row)
        for name, values in merged.items():
            out[name][i : i + 1] = values
        if conflict:
            conflicts.append((first[key], row))
    return list(first), out, conflicts, None


def _parse(items: ComponentItems, component_id: str) -> tuple[_Draft | None, list[tuple]]:
    r"""
    The one interpretation of a scope's ``chem_comp*`` items. Returns the draft, or None when the statements cannot
    form one definition (an unkeyed atom, a bond naming an absent atom or itself, duplicate rows whose stateless facts
    disagree: withheld), and the record entries it produced.
    """
    offset = 1 if items.component else 0  # refs rows: the _chem_comp row, then atom rows, then bond rows
    n = len(items.element[-1])

    def withheld(*rows: int) -> tuple[None, list[tuple]]:
        return None, [("definition_conflict", {"id": component_id}, tuple(items.refs.ref(row) for row in rows))]

    texts: dict[str, tuple[object, int]] = {}
    said: dict[str, NDArray[np.bool_]] = {}
    for name, item in (("chem_comp_type", "type"), ("one_letter_code", "one_letter_code")):
        token, raw = items.component.get(item, (None, _ABSENT))
        texts[name], said[name] = (token, int(_DEFINITION[raw])), np.isin(raw, _STATED_NULLS)
    token, raw = items.component.get("mon_nstd_parent_comp_id", (None, _ABSENT))
    said["parents"] = np.isin(raw, _STATED_NULLS)
    parents = tuple(parent.strip() for parent in (token or "").split(",") if parent.strip())
    parents_state = int(_DEFINITION[raw])
    if known(parents_state) and not parents:
        parents_state = _NULL
    texts["parents"] = (parents, parents_state)

    names, name_states = _text_item(items.atoms, "atom_id", n)
    unkeyed = np.flatnonzero(~known(name_states))
    if len(unkeyed):
        return withheld(offset + int(unkeyed[0]))
    z, isotope, element_state = items.element
    absent = (np.zeros(n), np.full(n, _ABSENT, dtype=np.uint8), np.zeros(n))
    charge, charge_state, _ = items.numbers.get("charge", absent)
    if (charge != np.round(charge)).any() or (np.abs(charge) > 127).any():
        _invalid("ComponentItems.numbers['charge']", "a charge that is not an int8 integer")
    aromatic, aromatic_state = _boolean(items.atoms, "pdbx_aromatic_flag", n)
    n_terminal = _flag(items.atoms, "pdbx_n_terminal_atom_flag", n)
    c_terminal = _flag(items.atoms, "pdbx_c_terminal_atom_flag", n)
    labels, _ = _enumerated(items.atoms, "pdbx_stereo_config", n, _LABELS, None)
    columns: dict[str, NDArray] = {
        "element": np.where(known(element_state), z, 0).astype(np.uint8),
        "isotope": np.where(known(element_state), isotope, 0).astype(np.uint16),
        "charge": charge.astype(np.int8),
        "charge_state": _DEFINITION[charge_state],
        "aromatic": aromatic,
        "aromatic_state": aromatic_state,
        "leaving_flag": _flag(items.atoms, "pdbx_leaving_atom_flag", n),
        "terminal_flag": np.stack([n_terminal, c_terminal], axis=1).reshape(n, 2),
        "cip": np.array(labels, dtype=object),
    }
    stated_null = {
        "element": np.isin(element_state, _STATED_NULLS),
        "charge": _said(items.numbers, "charge", n),
        "aromatic": _said(items.atoms, "pdbx_aromatic_flag", n),
        "leaving_flag": _said(items.atoms, "pdbx_leaving_atom_flag", n),
        "terminal_flag": np.stack([_said(items.atoms, f"pdbx_{end}_terminal_atom_flag", n) for end in "nc"], axis=1),
        "cip": _said(items.atoms, "pdbx_stereo_config", n),
    }
    stated_null["isotope"] = stated_null["element"]
    columns |= {f"{name}?": mask.reshape(columns[name].shape) for name, mask in stated_null.items()}  # merged per key
    # the coordinates and their printed quanta, merged across duplicate rows like any stated fact
    for kind, template, _ in _CONFORMERS:
        for axis in "xyz":
            coordinate, coordinate_raw, coordinate_quantum = items.numbers.get(template.format(axis), absent)
            states = _DEFINITION[coordinate_raw]
            columns[f"{kind}{axis}"], columns[f"{kind}{axis}_state"] = coordinate, states
            columns[f"{kind}{axis}q"], columns[f"{kind}{axis}q_state"] = coordinate_quantum, states
    keys, atoms, conflicts, disagree = _collapse([cast(bytes, name).decode() for name in names], columns)
    if disagree:
        return withheld(*(offset + row for row in disagree))
    said |= {name: atoms.pop(f"{name}?") for name in stated_null}

    index = {key: i for i, key in enumerate(keys)}
    m = len(next(iter(items.bonds.values()))[-1]) if items.bonds else 0
    first_ends, _ = _text_item(items.bonds, "atom_id_1", m)
    second_ends, _ = _text_item(items.bonds, "atom_id_2", m)
    pairs = []
    for row, (x, y) in enumerate(zip(first_ends, second_ends)):
        i, j = index.get((x or b"").decode(), -1), index.get((y or b"").decode(), -1)
        if i < 0 or j < 0 or i == j:
            return withheld(offset + n + row)
        pairs.append((min(i, j), max(i, j)))
    order_values, token_states = _enumerated(items.bonds, "value_order", m, _TOKENS, 0)
    order_tokens = np.array(order_values, dtype=np.uint8)
    bond_aromatic, bond_aromatic_state = _boolean(items.bonds, "pdbx_aromatic_flag", m)
    bond_labels, _ = _enumerated(items.bonds, "pdbx_stereo_config", m, _LABELS, None)
    bonds: dict[str, NDArray] = {
        "bond_order_token": order_tokens,
        "bond_order_token_state": np.where(known(token_states) & (order_tokens == 0), _NULL, token_states).astype(
            np.uint8
        ),
        "bond_aromatic": bond_aromatic,
        "bond_aromatic_state": bond_aromatic_state,
        "label": np.array(bond_labels, dtype=object),
    }
    stated_null = {
        "bond_order_token": _said(items.bonds, "value_order", m),
        "bond_aromatic": _said(items.bonds, "pdbx_aromatic_flag", m),
        "label": _said(items.bonds, "pdbx_stereo_config", m),
    }
    bonds |= {f"{name}?": mask for name, mask in stated_null.items()}
    pairs, bonds, bond_conflicts, bond_disagree = _collapse(pairs, bonds)
    if bond_disagree:
        return withheld(*(offset + n + row for row in bond_disagree))
    said |= {name: bonds.pop(f"{name}?") for name in stated_null}

    # one coherent conformer: the first kind of which every atom states all three coordinates
    reference, quantum, source = np.zeros((len(keys), 3)), np.zeros(len(keys)), ReferenceSource.NONE
    for kind, _, candidate in _CONFORMERS:
        if keys and all(known(atoms[f"{kind}{axis}_state"]).all() for axis in "xyz"):
            reference = np.stack([atoms[f"{kind}{axis}"] for axis in "xyz"], axis=1)
            quantum = np.max([atoms[f"{kind}{axis}q"] for axis in "xyz"], axis=0)
            source = candidate
            break
    atoms = {name: values for name, values in atoms.items() if not name.startswith(("ideal", "model"))}

    rows = {offset + row for pair in conflicts for row in pair}
    rows |= {offset + n + row for pair in bond_conflicts for row in pair}
    entries = []
    if rows:
        entries.append(("definition_conflict", {"id": component_id}, tuple(items.refs.ref(r) for r in sorted(rows))))
    return _Draft(texts, keys, atoms, reference, quantum, source, pairs, bonds, said=said), entries


def _draft_of(c: ChemicalComponent) -> _Draft:
    r"""A definition back as a draft, its stereocentre rows and stereo groups kept by centre name."""
    names = list(c.atom_names)
    kept = {}
    for centre, parity, slots, stereo_basis in zip(
        c.stereo_centre.tolist(), c.stereo_parity.tolist(), c.stereo_neighbors.tolist(), c.stereo_basis.tolist()
    ):
        kept[names[centre]] = (parity, tuple(names[s] if s >= 0 else None for s in slots), stereo_basis)
    centres, ptr, members = [names[x] for x in c.stereo_centre.tolist()], c.stereo_group_ptr, c.stereo_group_member
    groups = [
        (kind, tuple(centres[m] for m in members[ptr[g] : ptr[g + 1]].tolist()))
        for g, kind in enumerate(c.stereo_group_kind.tolist())
    ]
    atom_columns: tuple[str, ...] = ("element", "isotope", "leaving_flag", "terminal_flag", "charge", "charge_state")
    atom_columns += ("aromatic", "aromatic_state")
    atoms = {name: getattr(c, name).copy() for name in atom_columns}
    atoms["cip"] = np.array(c.cip_declared, dtype=object)
    bonds = {name: getattr(c, name).copy() for name in ("bond_order_token", "bond_aromatic", "bond_aromatic_state")}
    # an AROM–POLY token is stated (its order 0 UNKNOWN×REFERENCE is derived); another token has its order's state
    typed = c.bond_order_token > OrderToken.QUAD
    bonds["bond_order_token_state"] = np.where(typed, _STATED, c.bond_order_state).astype(np.uint8)
    bonds["label"] = np.full(len(c.bond_atoms), None, dtype=object)
    bonds["label"][c.bond_stereo_bond.astype(np.int64)] = c.bond_stereo_declared
    texts: dict[str, tuple[object, int]] = {
        "chem_comp_type": (c.chem_comp_type, c.chem_comp_type_state),
        "one_letter_code": (c.one_letter_code, c.one_letter_code_state),
        "parents": (c.parents, c.parents_state),
    }
    reference, quantum = c.reference.astype(np.float64), c.reference_quantum.astype(np.float64)
    pairs = [tuple(pair) for pair in c.bond_atoms.tolist()]
    kept_bonds = {}
    for bond, value, (left, right), label in zip(
        c.bond_stereo_bond.tolist(),
        c.bond_stereo_value.tolist(),
        c.bond_stereo_reference.tolist(),
        c.bond_stereo_declared,
    ):
        i, j = pairs[bond]
        kept_bonds[frozenset((names[i], names[j]))] = (value, {names[i]: names[left], names[j]: names[right]}, label)
    return _Draft(texts, names, atoms, reference, quantum, c.reference_source, pairs, bonds, kept, groups, kept_bonds)


def _bond_keys(d: _Draft) -> list[frozenset[str]]:
    return [frozenset((d.names[i], d.names[j])) for i, j in d.pairs]


def _same_scope(a: _Draft, b: _Draft) -> tuple[_Draft | None, bool]:
    r"""
    Two drafts of one scope (a caller entry and the library entry of the same id): their rows are one statement set
    (``_collapse``); different atom or bond key sets, or a stateless disagreement, withhold the definition (None). Two
    conformers apart by more than the printed quantum leave none in force; one restated keeps ``a``'s source and
    arbitration. Stereo groups stated by one side are kept; two different group sets withhold, and so does a stated
    stereocentre or cis/trans row the other side's row contradicts. Returns the draft and whether a Conflict arose.
    """
    if set(a.names) != set(b.names) or set(_bond_keys(a)) != set(_bond_keys(b)):
        return None, True
    if a.groups and b.groups and a.groups != b.groups:
        return None, True
    if _stated_disagree(a, b) or _stated_disagree(b, a):
        return None, True
    atom_columns = {name: np.concatenate([a.atoms[name], b.atoms[name]]) for name in a.atoms}
    bond_columns = {name: np.concatenate([a.bonds[name], b.bonds[name]]) for name in a.bonds}
    _, atoms, atom_conflicts, atom_disagree = _collapse(a.names + b.names, atom_columns)
    _, bonds, bond_conflicts, bond_disagree = _collapse(_bond_keys(a) + _bond_keys(b), bond_columns)
    if atom_disagree or bond_disagree:
        return None, True
    conflict = bool(atom_conflicts or bond_conflicts)
    texts: dict[str, tuple[object, int]] = {}
    for name, (value, value_state) in a.texts.items():
        other, other_state = b.texts[name]
        if _CONFLICT in (value_state, other_state) or (known(value_state) and known(other_state) and value != other):
            texts[name] = (() if name == "parents" else None, _CONFLICT)
            conflict = True
        elif known(value_state) or other_state == _ABSENT:
            texts[name] = (value, value_state)
        else:
            texts[name] = (other, other_state)
    order = [b.names.index(x) for x in a.names]
    tolerance = np.maximum(a.quantum, b.quantum[order])[:, None]
    apart = (np.abs(a.reference - b.reference[order]) > tolerance).any()
    reference, quantum, source, kept = a.reference, a.quantum, a.source, a.kept
    if b.source and not a.source:
        reference, quantum, source, kept = b.reference[order], b.quantum[order], b.source, b.kept
    elif b.source and apart:  # two conformers in one scope: neither
        reference, quantum, source, kept = a.reference * 0, a.quantum * 0, ReferenceSource.NONE, {}
        conflict = True
    groups = a.groups or b.groups
    kept = _stated(b.kept) | _stated(a.kept) | kept  # a caller's stated rows survive the unification (SMILES tags)
    kept_bonds = b.kept_bonds | a.kept_bonds
    return _Draft(texts, a.names, atoms, reference, quantum, source, a.pairs, bonds, kept, groups, kept_bonds), conflict


def _stated_disagree(a: _Draft, b: _Draft) -> bool:
    r"""Whether a row ``a`` states (SMILES/CX tags or marks) contradicts ``b``'s determined row at the same centre or
    bond: other neighbours, or another configuration once carried through the neighbour order (a parity through the
    slot permutation; a cis/trans value flips per bond end whose reference atom differs, the end's other neighbour)."""
    for name, (parity, slots, _) in _stated(a.kept).items():
        other = b.kept.get(name)
        if other is None or other[0] not in (1, 2):
            continue
        if sorted(map(str, slots)) != sorted(map(str, other[1])):
            return True
        odd = int(permutation_parity(np.array([other[1].index(x) for x in slots], dtype=np.int64)))
        if (3 - other[0] if odd else other[0]) != parity:
            return True
    for key, (value, reference, label) in a.kept_bonds.items():
        other_bond = b.kept_bonds.get(key)
        if label is not None or other_bond is None or other_bond[0] not in (1, 2) or value not in (1, 2):
            continue
        flips = sum(reference[end] != other_bond[1][end] for end in key)
        if (3 - other_bond[0] if flips % 2 else other_bond[0]) != value:
            return True
    return False


def _stated(kept: dict[str, tuple[int, tuple[str | None, ...], int]]) -> dict[str, tuple]:
    r"""The rows of ``kept`` stated by tags (a SMILES/CX definition), which no label of the draft re-derives."""
    return {name: row for name, row in kept.items() if row[2] == StereoBasis.STATED}


def _in_force(states: object) -> NDArray[np.bool_]:
    r"""A file-scope statement that overrides the archive: KNOWN or CONFLICT, and not a dictionary default."""
    stated = np.isin(resolution(states), (Resolution.KNOWN, Resolution.CONFLICT))
    return stated & (basis(states) != Basis.DEFAULT)


def _gather(values: NDArray, rows: NDArray[np.int64], fill: object = 0) -> NDArray:
    r"""``values[rows]`` with ``fill`` (None in an object column) where a row is -1."""
    out = np.full((len(rows), *values.shape[1:]), None if values.dtype == object else fill, dtype=values.dtype)
    out[rows >= 0] = values[rows[rows >= 0]]
    return out


def _pick(file: dict, archive: dict, f: NDArray, a: NDArray, said: dict, nulls: list[str]) -> dict[str, NDArray]:
    r"""
    Per row and fact, the file's statement where it is in force, else the archive's where it states one, else the
    file's default or null; a stateless fact is the file's where stated. ``f``/``a`` index the rows (-1 where a
    scope lacks the row); the dictionary items where a file row's stated ``?`` or ``.`` (``said``) left an archive
    value in force are appended to ``nulls``.
    """
    out, stated = {}, f >= 0
    for name, values in file.items():
        if name.endswith("_state"):
            continue
        null = _gather(said[name], f, False)
        if f"{name}_state" in file:
            file_values, file_states = _gather(values, f), _gather(file[f"{name}_state"], f, _ABSENT)
            archive_values, archive_states = _gather(archive[name], a), _gather(archive[f"{name}_state"], a, _ABSENT)
            use = ~(_in_force(file_states) & stated) & (a >= 0) & (archive_states != _ABSENT)
            out[name] = np.where(use, archive_values, file_values)
            out[f"{name}_state"] = np.where(use, archive_states, file_states).astype(np.uint8)
            if (use & null & known(archive_states)).any():
                nulls += _ITEMS[name]
        else:
            file_null = _null(_gather(file[_NULL_OF.get(name, name)], f))
            archive_null = _null(_gather(archive[_NULL_OF.get(name, name)], a))
            out[name] = np.where(file_null, _gather(archive[name], a), _gather(values, f))
            hit = np.atleast_2d((file_null & ~archive_null & null).T).any(axis=1)  # per column of the fact
            nulls += [item for item, found in zip(_ITEMS[name], hit) if found]
    return out


def _override(file: _Draft, archive: _Draft) -> tuple[_Draft, list[str], bool]:
    r"""
    File scope over archive scope, per item: atom keys in archive order then file-only atoms, bonds likewise. Returns
    the merged draft, the items where a file-stated ``?`` or ``.`` left an archive KNOWN value in force (an absent item
    is no stated null), and whether the key sets of a table the file states (atoms, bonds) differ: a
    ``_chem_comp``-only row states none.
    """
    archive_index = {x: i for i, x in enumerate(archive.names)}
    file_index = {x: i for i, x in enumerate(file.names)}
    names = archive.names + [x for x in file.names if x not in archive_index]
    fr = np.array([file_index.get(x, -1) for x in names], dtype=np.int64)
    ar = np.array([archive_index.get(x, -1) for x in names], dtype=np.int64)

    nulls: list[str] = []
    texts: dict[str, tuple[object, int]] = {}
    for name, (value, value_state) in file.texts.items():
        other, other_state = archive.texts[name]
        if not _in_force(value_state) and other_state != _ABSENT:
            texts[name] = (other, other_state)
            if known(other_state) and file.said[name]:
                nulls += _ITEMS[name]
        elif name == "chem_comp_type" and known(other_state) and str(value).upper() == str(other).upper():
            texts[name] = (other, other_state)  # the type is case-insensitive: a restatement keeps the archive token
        else:
            texts[name] = (value, value_state)

    atoms = _pick(file.atoms, archive.atoms, fr, ar, file.said, nulls)

    archive_bonds, file_bonds = _bond_keys(archive), _bond_keys(file)
    keys = archive_bonds + [k for k in file_bonds if k not in set(archive_bonds)]
    fb = np.array([file_bonds.index(k) if k in file_bonds else -1 for k in keys], dtype=np.int64)
    ab = np.array([archive_bonds.index(k) if k in archive_bonds else -1 for k in keys], dtype=np.int64)
    bonds = _pick(file.bonds, archive.bonds, fb, ab, file.said, nulls)
    position = {x: i for i, x in enumerate(names)}
    pairs = []
    for key in keys:
        i, j = sorted(position[x] for x in key)
        pairs.append((i, j))

    reference, quantum, source, kept = np.zeros((len(names), 3)), np.zeros(len(names)), ReferenceSource.NONE, {}
    if archive.source and (ar >= 0).all():
        reference, quantum, source, kept = archive.reference[ar], archive.quantum[ar], archive.source, archive.kept
    # a file restating the archive conformer within the printed quantum keeps it with its source and arbitration
    if file.source and (fr >= 0).all():
        within = np.abs(file.reference[fr] - reference) <= np.maximum(file.quantum[fr], quantum)[:, None]
        if not (source and within.all()):
            reference, quantum, source, kept = file.reference[fr], file.quantum[fr], file.source, {}

    # the descriptor verification holds only for the graph it was made on: on another the verified centre is refused
    # again (its conformer stays contradicted, no descriptor arbitrates at a read)
    unchanged = (
        names == archive.names
        and pairs == archive.pairs
        and all(np.array_equal(atoms[x], archive.atoms[x]) for x in ("element", "charge", "charge_state"))
        and all(np.array_equal(bonds[x], archive.bonds[x]) for x in ("bond_order_token", "bond_order_token_state"))
    )
    if not unchanged:
        verified = StereoBasis.DESCRIPTOR_VERIFIED
        kept = {x: (0, row[1], StereoBasis.REFUSED) if row[2] == verified else row for x, row in kept.items()}

    atoms_differ = bool(file.names) and bool((fr < 0).any() or (ar < 0).any())
    bonds_differ = bool(file.pairs) and bool((fb < 0).any() or (ab < 0).any())
    kept = _stated(archive.kept) | kept
    merged = _Draft(
        texts, names, atoms, reference, quantum, source, pairs, bonds, kept, archive.groups, archive.kept_bonds
    )
    return merged, list(dict.fromkeys(nulls)), atoms_differ or bonds_differ


def _graph(d: _Draft) -> list[list[int]]:
    graph: list[list[int]] = [[] for _ in d.names]
    for i, j in d.pairs:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def _stereo_rows(d: _Draft) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""
    The one stereocentre rule: a row per atom labelled R or S whose degree is 3 or 4, neighbours in ascending
    definition order with a -1 slot appended for degree 3; parity from the conformer in force (0 within its printed
    quantum, or without one). An archive row is kept where its conformer is in force and its neighbours are the same
    atoms, so an arbitration survives; its parity is carried through the neighbour permutation.
    """
    graph = _graph(d)
    rows = []
    for atom, label in enumerate(d.atoms["cip"]):
        neighbours = sorted(graph[atom])
        if label in ("R", "S") and len(neighbours) in (3, 4):
            rows.append((atom, neighbours + [-1] * (4 - len(neighbours))))
    centre = np.array([atom for atom, _ in rows], dtype=np.int64)
    slots = np.array([row for _, row in rows], dtype=np.int64).reshape(-1, 4)
    parity = np.zeros(len(rows), dtype=np.int8)
    stereo_basis = np.full(len(rows), StereoBasis.CONFORMER, dtype=np.uint8)
    if d.source != ReferenceSource.NONE and rows:
        parity = parity_from_coordinates(d.reference, centre, slots, d.quantum).astype(np.int8)
    for r, (atom, row) in enumerate(rows):
        kept = d.kept.get(d.names[atom])
        current = tuple(d.names[s] if s >= 0 else None for s in row)
        if kept is None or sorted(map(str, kept[1])) != sorted(map(str, current)):
            continue
        kept_parity, kept_slots, kept_basis = kept
        odd = int(permutation_parity(np.array([kept_slots.index(x) for x in current], dtype=np.int64)))
        parity[r] = 3 - kept_parity if odd and kept_parity else kept_parity
        stereo_basis[r] = kept_basis
    # a stated row (SMILES/CX tags) stands where no label decides the atom and its neighbours are unchanged
    index, labelled = {name: i for i, name in enumerate(d.names)}, set(centre.tolist())
    stated = []
    for name, (kept_parity, kept_slots, kept_basis) in d.kept.items():
        kept_atom = index.get(name)
        if (
            kept_basis != StereoBasis.STATED
            or kept_atom is None
            or kept_atom in labelled
            or d.atoms["cip"][kept_atom] is not None
        ):
            continue
        named = [x for x in kept_slots if x is not None]
        if len(named) == len(graph[kept_atom]) and set(named) == {d.names[x] for x in graph[kept_atom]}:
            stated.append((kept_atom, [index[x] if x is not None else -1 for x in kept_slots], kept_parity))
    if stated:
        order = np.argsort(np.concatenate((centre, [atom for atom, _, _ in stated])), kind="stable")
        centre = np.concatenate((centre, [atom for atom, _, _ in stated]))[order]
        slots = np.concatenate((slots, np.array([row for _, row, _ in stated], dtype=np.int64)))[order]
        parity = np.concatenate((parity, [p for _, _, p in stated])).astype(np.int8)[order]
        stereo_basis = np.concatenate((stereo_basis, [StereoBasis.STATED] * len(stated))).astype(np.uint8)[order]
    return centre, parity, slots, stereo_basis


def _bond_stereo_rows(d: _Draft) -> tuple[NDArray, NDArray, NDArray, tuple[str | None, ...]]:
    r"""A row per bond labelled E or Z: reference pair = on each end the lowest-index neighbour other than the partner;
    value cis/trans from the conformer in force (0 within its quantum or without one); the label kept as provenance."""
    graph, rows, stated = _graph(d), [], {}
    index = {name: i for i, name in enumerate(d.names)}
    for bond, label in enumerate(d.bonds["label"]):
        i, j = d.pairs[bond]
        left, right = [x for x in graph[i] if x != j], [x for x in graph[j] if x != i]
        if label in ("E", "Z") and left and right:
            rows.append((bond, min(left), min(right), label))
        elif label is None and (kept := d.kept_bonds.get(frozenset((d.names[i], d.names[j])))) and kept[2] is None:
            # a stated row (SMILES/CX marks) stands while its reference atoms are still the bond atoms' neighbours
            ends = (index.get(kept[1][d.names[i]], -1), index.get(kept[1][d.names[j]], -1))
            if ends[0] in left and ends[1] in right:
                rows.append((bond, *ends, None))
                stated[len(rows) - 1] = kept[0]
    bond = np.array([row[0] for row in rows], dtype=np.int64)
    reference = np.array([row[1:3] for row in rows], dtype=np.int64).reshape(-1, 2)
    value = np.zeros(len(rows), dtype=np.int8)
    if d.source != ReferenceSource.NONE and rows:
        ends = np.array([d.pairs[b] for b in bond.tolist()], dtype=np.int64)
        value = bond_configuration(d.reference, ends, reference, d.quantum).astype(np.int8)
    for row, kept_value in stated.items():
        value[row] = kept_value
    return bond, value, reference, tuple(row[3] for row in rows)


def _finish(
    d: _Draft, component_id: str, scope: DefinitionScope, stereo: StereoArbitration | None = None
) -> ChemicalComponent:
    r"""The derived rows of one draft, as a definition. A stereo group keeps the members that are stereocentre rows of
    the result."""
    n = len(d.names)
    centre, parity, slots, stereo_basis = _stereo_rows(d)
    row_of = {d.names[atom]: r for r, atom in enumerate(centre.tolist())}
    groups = [(kind, [row_of[x] for x in members if x in row_of]) for kind, members in d.groups]
    groups = [(kind, rows) for kind, rows in groups if rows]
    source = d.source
    if stereo is not None:
        parity, stereo_basis = stereo.parity, stereo.basis
        if stereo.contradicted:
            source = ReferenceSource.CONTRADICTED
    bond, value, bond_reference, labels = _bond_stereo_rows(d)
    values = {name: values for name, values in (d.atoms | d.bonds).items() if name in _ARRAYS}
    token, typed = d.bonds["bond_order_token"], d.bonds["bond_order_token"] > OrderToken.QUAD
    values |= {
        "bond_order": np.where(typed, 0, token),
        "bond_order_state": np.where(typed, _NULL, d.bonds["bond_order_token_state"]),
    }
    values |= {
        "reference": np.reshape(d.reference, (n, 3)),
        "reference_quantum": np.reshape(d.quantum, n),
        "bond_atoms": np.reshape(d.pairs, (-1, 2)),
        "stereo_centre": centre,
        "stereo_parity": parity,
        "stereo_neighbors": slots,
        "stereo_basis": stereo_basis,
        "bond_stereo_bond": bond,
        "bond_stereo_value": value,
        "bond_stereo_reference": bond_reference,
        "stereo_group_kind": [kind for kind, _ in groups],
        "stereo_group_ptr": np.cumsum([0, *(len(rows) for _, rows in groups)]),
        "stereo_group_member": [r for _, rows in groups for r in rows],
    }
    values |= {
        "component_id": component_id,
        "chem_comp_type": d.texts["chem_comp_type"][0],
        "chem_comp_type_state": int(d.texts["chem_comp_type"][1]),
        "one_letter_code": d.texts["one_letter_code"][0],
        "one_letter_code_state": int(d.texts["one_letter_code"][1]),
        "parents": tuple(cast(tuple[str, ...], d.texts["parents"][0])),
        "parents_state": int(d.texts["parents"][1]),
        "atom_names": tuple(d.names),
        "reference_source": source,
        "cip_declared": tuple(d.atoms["cip"].tolist()),
        "bond_stereo_declared": labels,
    }
    return stated_component(values, scope=scope)


def stated_component(values: Mapping[str, object], *, scope: DefinitionScope) -> ChemicalComponent:
    r"""
    The definition of complete field values (every field but scope and digest; arrays in any array-like form) with its
    digest: the one constructor of a definition from values (the definition pass and ``smiles.read_smiles``). Values
    without ``bond_order_token`` state none: each bond's token is its order where that is known, else NONE.
    """
    values = {name: np.array(v, dtype=_ARRAYS[name][0]) if name in _ARRAYS else v for name, v in values.items()}
    if "bond_order_token" not in values:
        order, order_state = values["bond_order"], values["bond_order_state"]
        values["bond_order_token"] = np.where(known(order_state), order, 0).astype(np.uint8)
    # The field schema assembles keyword arguments dynamically.
    return cast(Callable[..., ChemicalComponent], ChemicalComponent)(scope=scope, digest=_digest(values), **values)


def component_findings(c: ChemicalComponent) -> list[tuple[str, Mapping[str, object]]]:
    r"""The artifact findings of a definition with their payloads: the reference-graph findings of
    ``fragments.findings``, a refused parity per row and a contradicted conformer."""
    out = list(findings(c))
    for atom in c.stereo_centre[c.stereo_basis == StereoBasis.REFUSED].tolist():
        out.append(("parity_refused", {"atom": c.atom_names[atom]}))
    if c.reference_source == ReferenceSource.CONTRADICTED:
        out.append(("reference_coordinates_contradicted", {}))
    return out


def define_component(
    items: ComponentItems, *, scope: DefinitionScope, stereo: StereoArbitration | None = None
) -> ChemicalComponent:
    r"""
    The definition stated by one scope's items (id ``_chem_comp.id``), with derived rows. ``stereo``
    (``rdkit.arbitrate_stereo``) replaces the parity and basis of the stereocentre rows, a contradicted conformer
    becoming CONTRADICTED. Items without an id, or that cannot form one definition (an unkeyed atom, a bond naming an
    absent atom, duplicate rows that disagree) → ``InvalidRequestError(value_invalid)``.
    """
    component_id, id_state = items.component.get("id", (None, _ABSENT))
    if not known(id_state):
        _invalid("ComponentItems.component['id']", "no _chem_comp.id")
    component_id = cast(str, component_id)
    draft, entries = _parse(items, component_id)
    if draft is None or entries:
        _invalid("ComponentItems", f"{component_id}: statements that do not form one definition")
    return _finish(draft, component_id, scope, stereo)


@dataclass(frozen=True)
class ArtifactProvenance:
    r"""The schema, content version and payload digest of a CCD artifact; the build's source digest, tool and
    counts are the tools' sidecar."""

    schema: str
    content_version: str
    sha256: str


@dataclass(frozen=True)
class Definitions:
    r"""The definitions a value is built from, after scope precedence: an id with no atoms in force (a missing or
    withheld definition, or an atom-less one) is not in ``by_id`` or has no atom names; ``records`` the pass's
    records and every finding in force;
    ``refs`` per id in force the statements it was resolved from, the ones its records name (file row, artifact entry,
    caller entry; none for a value no source states)."""

    by_id: Mapping[str, ChemicalComponent]
    records: Records
    library: ArtifactProvenance | None
    refs: Mapping[str, tuple[SourceRef, ...]] = MappingProxyType({})

    def __reduce__(self):  # a MappingProxyType does not pickle (e.g. across processes)
        return _definitions, (dict(self.by_id), self.records, self.library, dict(self.refs))


def _definitions(by_id: dict, records: Records, library: ArtifactProvenance | None, refs: dict) -> Definitions:
    return Definitions(MappingProxyType(by_id), records, library, MappingProxyType(refs))


def resolve_definitions(
    component_ids: Collection[str],
    file_rows: DefinitionRows,
    library: ComponentLibrary,
    caller: Mapping[str, ChemicalComponent] = MappingProxyType({}),
) -> Definitions:
    r"""
    The one definition pass. Per id, in sorted order: the archive scope (library entry and caller entry, one evidence
    class whose two statements are unified; a caller entry alone has scope CALLER whatever scope it was built with),
    then file scope over it per item; a file-scope definition is derived again as a whole, so its port record is
    re-verified. An id withheld by a same-scope disagreement is not in ``by_id`` and has a ``definition_conflict``
    record.
    """
    entries: list[tuple[str, Mapping[str, object], tuple[SourceRef, ...]]] = []
    by_id, sources = {}, {}
    block = next((items.refs.block for items in file_rows.components.values()), None)
    for cid in sorted(set(component_ids)):
        archive, given = library.definition(cid), caller.get(cid)
        refs = []
        if archive is not None:
            refs.append(SourceRef("artifact", library._index(cid)))
        if given is not None:
            given = given.at_scope(DefinitionScope.CALLER)
            refs.append(SourceRef("caller", 0, item=cid))
        definition, draft, scope = archive or given, None, DefinitionScope.ARCHIVE
        if archive is not None and given is not None:
            draft, conflict = _same_scope(_draft_of(archive), _draft_of(given))
            if conflict:
                entries.append(("definition_conflict", {"id": cid}, tuple(refs)))
            if draft is None:
                continue
        items = file_rows.components.get(cid)
        if items is not None:
            parsed, notes = _parse(items, cid)
            entries += notes
            if parsed is None:
                continue
            if len(items.refs):
                refs.insert(0, items.refs.ref(0))
            base = draft or (_draft_of(definition) if definition is not None else None)
            if base is not None:
                parsed, nulls, differ = _override(parsed, base)
                for item in nulls:
                    entries.append(("file_scope_null_over_archive_value", {"id": cid, "item": item}, tuple(refs)))
                if differ:
                    entries.append(("definition_atom_sets_differ", {"id": cid}, tuple(refs)))
            draft, scope = parsed, DefinitionScope.FILE
        if draft is not None:
            definition = _finish(draft, cid, scope)
        if definition is None:
            continue
        by_id[cid], sources[cid] = definition, tuple(refs)
        entries += [(code, {"id": cid, **payload}, sources[cid]) for code, payload in component_findings(definition)]
    records, provenance = _records(entries, block), library.provenance
    return Definitions(MappingProxyType(by_id), records, provenance, MappingProxyType(sources))


def _records(entries: list[tuple[str, Mapping[str, object], tuple[SourceRef, ...]]], block: str | None) -> Records:
    r"""Definition records as ``Records`` columns: subject (NONE, -1), payload slots per the code's row."""
    if not entries:
        return Records.empty()
    keys: dict[tuple[str, int, str | None], int] = {}  # (category, row, item) -> table row
    texts: dict[str, int] = {}  # text payload -> index
    rows, ptr = [], [0]
    payload = np.zeros((len(entries), 4), dtype=np.int64)
    for i, (code, values, refs) in enumerate(entries):
        for slot, (name, kind) in enumerate(CODES[code].payload):
            payload[i, slot] = (
                texts.setdefault(cast(str, values[name]), len(texts)) if kind == PayloadKind.TEXT else values[name]
            )
        rows += [keys.setdefault((ref.category, ref.row, ref.item), len(keys)) for ref in refs]
        ptr.append(len(rows))
    categories = sorted({key[0] for key in keys})
    item_names = sorted({key[2] for key in keys if key[2] is not None})
    table = SourceRefs(
        np.array([categories.index(key[0]) for key in keys], dtype=np.int16),
        np.array([key[1] for key in keys], dtype=np.int64),
        np.array([item_names.index(key[2]) if key[2] is not None else -1 for key in keys], dtype=np.int16),
        tuple(categories),
        tuple(item_names),
        block,
    )
    count = len(entries)
    return Records(
        np.array([CODE_NAMES.index(code) for code, _, _ in entries], dtype=np.uint16),
        np.zeros(count, dtype=np.uint8),
        np.full(count, -1, dtype=np.int64),
        np.array(ptr, dtype=np.int64),
        np.array(rows, dtype=np.int64),
        table,
        payload,
        np.zeros(count + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        tuple(texts),
    )


# Artifact layout: every array field of ChemicalComponent as one column concatenated over components, one
# `<axis>.ptr` (int64 [c+1]) per axis framing every column of the axis (`stereo_group_ptr`, one row more per component,
# is framed by `group.ptr` plus the component index); the sorted ids, and per component one JSON record of its other
# fields (scope is always ARCHIVE), as uint8 blobs with `.offset` columns.
_RECORD = tuple(f.name for f in fields(ChemicalComponent) if f.name not in (*_ARRAYS, "component_id", "scope"))
_REFERENCE_SOURCES = frozenset(int(source) for source in ReferenceSource)
_PROVENANCE = ("schema", "content_version")
_DTYPES = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype(np.uint16),
    "I16": np.dtype(np.int16),
    "I64": np.dtype(np.int64),
    "F32": np.dtype(np.float32),
}
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}


def _pack(texts: list[str]) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    encoded = [text.encode() for text in texts]
    return np.frombuffer(b"".join(encoded), np.uint8).copy(), np.cumsum([0, *map(len, encoded)], dtype=np.int64)


def _unpack(columns: Mapping[str, NDArray], name: str, start: int, stop: int) -> tuple[str, ...]:
    offsets = columns[f"{name}.offset"][start : stop + 1]
    blob = columns[name][offsets[0] : offsets[-1]].tobytes()
    return tuple(_utf8(blob[a - offsets[0] : b - offsets[0]], name) for a, b in zip(offsets[:-1], offsets[1:]))


def artifact_columns(components: Iterable[ChemicalComponent]) -> dict[str, NDArray]:
    r"""The artifact columns of ``components`` sorted by id (the layout ``ComponentLibrary`` reads; scope is not
    stored). Also the column codec of definitions inside native artifacts."""
    components = sorted(components, key=lambda c: c.component_id)
    columns = {}
    for name, (dtype, shape, axis) in _ARRAYS.items():
        parts = [getattr(c, name) for c in components]
        columns[name] = np.concatenate(parts) if parts else np.zeros((0, *shape), dtype=dtype)
        if name != "stereo_group_ptr":
            columns[f"{axis}.ptr"] = np.cumsum([0, *map(len, parts)], dtype=np.int64)
    records = [json.dumps({key: getattr(c, key) for key in _RECORD}, sort_keys=True) for c in components]
    columns["component_id"], columns["component_id.offset"] = _pack([c.component_id for c in components])
    columns["record"], columns["record.offset"] = _pack(records)
    return columns


def write_artifact(
    components: Iterable[ChemicalComponent], path: str | PathLike, provenance: ArtifactProvenance
) -> None:
    r"""
    Write archive definitions as one ``ofoldx.ccd/7`` safetensors file (tools only), atomically and deterministically:
    the same components, schema and content version give byte-identical files. The payload digest is computed here.
    """
    columns = artifact_columns(components)

    header, chunks, offset = {}, [], 0
    for name in sorted(columns):
        array = np.ascontiguousarray(columns[name])
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": _DTYPE_NAMES[array.dtype], "shape": list(array.shape)}
        header[name]["data_offsets"] = [offset, offset + len(data)]
        chunks.append(data)
        offset += len(data)
    payload = b"".join(chunks)
    # not safetensors.numpy.save: it writes the metadata keys in hash order, so two builds would differ
    metadata = {key: getattr(provenance, key) for key in _PROVENANCE}
    metadata["sha256"] = hashlib.sha256(payload).hexdigest()
    header["__metadata__"] = metadata
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)

    temporary = Path(f"{os.fspath(path)}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(struct.pack("<Q", len(text)) + text + payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


#: The artifact layout: every column with its dtype, trailing shape and pointer (an axis pointer or a text offset).
_LAYOUT = {name: (dtype, shape, f"{axis}.ptr") for name, (dtype, shape, axis) in _ARRAYS.items()}
_LAYOUT |= {name: (np.uint8, (), f"{name}.offset") for name in ("component_id", "record")}
_POINTERS = (*(f"{axis}.ptr" for axis in _AXES), "component_id.offset", "record.offset")


def _frame(columns: Mapping[str, NDArray], name: str) -> NDArray[np.int64]:
    r"""The rows of column ``name`` per definition, as a pointer."""
    pointer = columns[_LAYOUT[name][2]]
    return pointer + np.arange(len(pointer)) if name == "stereo_group_ptr" else pointer


def _check_layout(columns: Mapping[str, NDArray]) -> None:
    r"""
    ``native_corrupt`` unless every column and pointer is present, every column has the layout's dtype and trailing
    shape, every pointer is int64, one entry per definition plus one, starting at 0 and non-decreasing, and every
    column's frame ends at its length.
    """
    for name in (*_LAYOUT, *_POINTERS):
        if name not in columns:
            _corrupt(name)
    for name, (dtype, shape, _) in _LAYOUT.items():
        if columns[name].dtype != dtype or columns[name].ndim != 1 + len(shape) or columns[name].shape[1:] != shape:
            _corrupt(name)
    count = len(columns["record.offset"]) - 1
    for name in _POINTERS:
        pointer = columns[name]
        framed = count >= 0 and pointer.dtype == np.int64 and pointer.shape == (count + 1,)
        if not framed or pointer[0] != 0 or (np.diff(pointer) < 0).any():
            _corrupt(name)
    for name in _LAYOUT:
        if _frame(columns, name)[-1] != len(columns[name]):
            _corrupt(name)


class ComponentLibrary:
    r"""The definitions of one ``ofoldx.ccd/7`` artifact, looked up by id (O(log n) on the sorted key) and built once
    per id. Columns that are not the artifact layout, an id or record whose bytes are no UTF-8, or a definition whose
    stored record does not decode, whose stored digest is not the digest of its stored fields, that indexes an atom,
    bond or stereocentre it does not have or whose bond order tokens its orders do not carry, are ``native_corrupt``.
    Obtained from ``load_library`` or ``configured_library``."""

    def __init__(self, columns: Mapping[str, NDArray], provenance: ArtifactProvenance | None) -> None:
        self.provenance = provenance
        self._columns = columns
        _check_layout(columns)
        self._frames = {name: _frame(columns, name) for name in _ARRAYS}
        count = len(columns["record.offset"]) - 1
        self._ids = np.array(_unpack(columns, "component_id", 0, count), dtype=np.str_)
        if (self._ids[1:] <= self._ids[:-1]).any():  # a lookup miss means "not in the library" only on sorted ids
            _corrupt("component_id")
        self._memo: dict[int, ChemicalComponent] = {}

    def _index(self, component_id: str) -> int:
        index = int(np.searchsorted(self._ids, component_id))
        return index if index < len(self._ids) and self._ids[index] == component_id else -1

    def __contains__(self, component_id: str) -> bool:
        return self._index(component_id) >= 0

    def definition(self, component_id: str) -> ChemicalComponent | None:
        i = self._index(component_id)
        if i < 0:
            return None
        if i not in self._memo:
            columns = self._columns
            values = {}
            for name in _ARRAYS:
                ptr = self._frames[name]
                values[name] = columns[name][ptr[i] : ptr[i + 1]]
            offsets, item = columns["record.offset"], f"{self._ids[i]}.record"
            record = _json(columns["record"][offsets[i] : offsets[i + 1]].tobytes(), item)
            if (
                not isinstance(record, dict)
                or set(record) != set(_RECORD)
                or type(source := record["reference_source"]) is not int
                or source not in _REFERENCE_SOURCES
            ):
                _corrupt(item)  # not one definition's fields
            values |= {key: tuple(value) if isinstance(value, list) else value for key, value in record.items()}
            values["reference_source"] = ReferenceSource(source)
            values["component_id"] = str(self._ids[i])
            # The artifact schema assembles keyword arguments dynamically.
            digest, component = _digest(values), cast(Callable[..., ChemicalComponent], ChemicalComponent)(
                scope=DefinitionScope.ARCHIVE, **values
            )
            if digest != component.digest:
                _corrupt(f"{component.component_id}.digest")
            if (out_of_range := _out_of_range(component)) is not None:  # the reference bounds, checked once
                _corrupt(f"{component.component_id}.{out_of_range}")
            self._memo[i] = component
        return self._memo[i]


def _out_of_range(c: ChemicalComponent) -> str | None:
    r"""The first index column of ``c`` naming a row its definition does not have (a stereo neighbour slot may be the
    implicit ``-1``), a stereo group pointer that is not a frame over the members, or a bond order token its order
    does not carry (SING–QUAD exactly the known orders, AROM–POLY order 0 UNKNOWN×REFERENCE); None when every index is
    in range."""
    atoms, bonds, centres = len(c.element), len(c.bond_order), len(c.stereo_centre)
    token, order, order_state = c.bond_order_token, c.bond_order, c.bond_order_state
    integer, typed = (token >= OrderToken.SING) & (token <= OrderToken.QUAD), token > OrderToken.QUAD
    if (
        (token > OrderToken.POLY).any()
        or (integer != known(order_state)).any()
        or (order[integer] != token[integer]).any()
        or (order[typed] != 0).any()
        or (order_state[typed] != _NULL).any()
    ):
        return "bond_order_token"
    for name, low, high in (
        ("bond_atoms", 0, atoms),
        ("stereo_centre", 0, atoms),
        ("stereo_neighbors", -1, atoms),
        ("bond_stereo_bond", 0, bonds),
        ("bond_stereo_reference", 0, atoms),
        ("stereo_group_member", 0, centres),
    ):
        values = getattr(c, name)
        if len(values) and (values.min() < low or values.max() >= high):
            return name
    ptr = c.stereo_group_ptr
    if ptr[0] != 0 or ptr[-1] != len(c.stereo_group_member) or (np.diff(ptr) < 0).any():
        return "stereo_group_ptr"
    return None


def _corrupt(item: str, **detail: object) -> NoReturn:
    fail("native_corrupt", SourceRef("artifact", 0, item=item), **detail)


def _utf8(data: bytes, item: str) -> str:
    r"""The text of ``data``, a field the layout stores as UTF-8: ``native_corrupt`` (item ``item``, the decoder's
    reason) for bytes that are no UTF-8, never repaired."""
    try:
        return data.decode()
    except UnicodeDecodeError as error:
        _corrupt(item, reason=str(error))


def _json(data: bytes, item: str) -> object:
    r"""The JSON value of the UTF-8 field ``data``; None when its text is no JSON."""
    try:
        return json.loads(_utf8(data, item))
    except ValueError:  # json.JSONDecodeError
        return None


def _counts(value: object, length: int | None = None) -> TypeGuard[list[int]]:
    r"""Whether ``value`` is a JSON list of non-negative integers (of ``length`` entries when given)."""
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(type(v) is int and v >= 0 for v in value)
    )


def _tensor(payload: memoryview, name: str, entry: object) -> NDArray:
    r"""The array one header entry names: ``native_corrupt`` (item ``name``) unless the entry states a dtype of the
    layout, a shape of at most two dimensions, none past the payload, and the byte range of exactly that many values
    inside the payload."""
    entry = entry if isinstance(entry, dict) else {}
    dtype_name, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
    if not (isinstance(dtype_name, str) and dtype_name in _DTYPES and _counts(shape) and _counts(offsets, 2)):
        _corrupt(name)
    dtype, (start, stop) = _DTYPES[dtype_name], offsets
    if not start <= stop <= len(payload) or stop - start != dtype.itemsize * math.prod(shape):
        _corrupt(name)
    # a zero dimension passes the byte count whatever the others are: they are bounded by the payload too
    if len(shape) > 2 or dtype.itemsize * math.prod(filter(None, shape)) > len(payload):
        _corrupt(name)
    return np.frombuffer(payload[start:stop], dtype=dtype.newbyteorder("<")).reshape(shape).astype(dtype, copy=False)


def load_library(path: str | PathLike) -> ComponentLibrary:
    r"""
    Load an ``ofoldx.ccd/7`` artifact. Any content version is accepted (an older snapshot stays reproducible).

    Raises:
        UnsupportedError: ``artifact_schema_unsupported`` for another schema.
        ContradictionError: ``native_corrupt`` for recognised damage only: a header that is no JSON object with its
            metadata, a payload digest other than the header's, a tensor that does not decode from its own bytes of the
            payload, columns that are not the artifact layout, a text field whose bytes are no UTF-8 (the field and
            the decoder's reason named), or a definition that fails its checks on access. Any other exception
            propagates.
    """
    data = memoryview(Path(path).read_bytes())
    size = struct.unpack("<Q", data[:8])[0] if len(data) >= 8 else -1
    header = _json(bytes(data[8 : 8 + size]), "header") if 0 <= size <= len(data) - 8 else None
    metadata = header.pop("__metadata__", None) if isinstance(header, dict) else None
    if not isinstance(metadata, dict):
        _corrupt("header")
    if (schema := metadata.get("schema")) != ARTIFACT_SCHEMA:
        fail("artifact_schema_unsupported", SourceRef("artifact", 0, item="schema"), schema=schema)
    payload = data[8 + size :]
    if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
        _corrupt("sha256")
    if "content_version" not in metadata:
        _corrupt("header")
    columns = {name: _tensor(payload, name, entry) for name, entry in cast(dict[str, object], header).items()}
    provenance = ArtifactProvenance(**{key: str(metadata[key]) for key in (*_PROVENANCE, "sha256")})
    return ComponentLibrary(columns, provenance)


def validate_library(directory: Path) -> Path:
    r"""Check a CCD dataset's schema, payload, sidecar digest and content version."""
    artifact = directory / ARTIFACT_FILENAME
    sidecar = directory / PROVENANCE_FILENAME
    library = load_library(artifact)
    record = _json(sidecar.read_bytes(), "provenance")
    if not isinstance(record, dict):
        _corrupt("provenance")
    with artifact.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if record.get("sha256") != digest:
        _corrupt("provenance.sha256")
    if library.provenance is None or record.get("content_version") != library.provenance.content_version:
        _corrupt("provenance.content_version")
    return artifact


def download_library(
    repo_id: str = DEFAULT_REPOSITORY,
    *,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
    local_files_only: bool = False,
) -> Path:
    r"""Download a CCD artifact and provenance sidecar from one dataset snapshot.

    Args:
        repo_id: Dataset repository; defaults to `oteam/ccd`.
        revision: Dataset tag, branch, or full commit; defaults to main. Select a full
            commit when a run needs a fixed snapshot.
        token: Optional Hub authentication token.
        local_files_only: Whether to resolve files only from the existing Hub cache.

    Returns:
        The cached `components.safetensors` path after schema, payload, and sidecar checks.
        Both files are resolved at the same dataset commit.

    Raises:
        InvalidRequestError: The requested revision is empty (`ccd_revision_invalid`).
        UnsupportedError: The artifact schema is unsupported.
        ContradictionError: The artifact or its provenance sidecar is inconsistent.
    """
    directory = resolve_snapshot(repo_id, revision=revision, token=token, local_files_only=local_files_only)
    return validate_library(directory)


_CONFIGURED: dict[str, ComponentLibrary] = {}


def configured_library() -> ComponentLibrary:
    r"""Return the default CCD library, loaded once per process.

    ``OFOLDX_CCD_ARTIFACT`` overrides the artifact with a local path. Otherwise,
    download the official ``oteam/ccd`` dataset using the Hugging Face cache.
    An explicit local path is never replaced by a download if loading fails.
    """
    path = os.environ.get("OFOLDX_CCD_ARTIFACT", "")
    key = str(Path(path).resolve()) if path else ""
    if key not in _CONFIGURED:
        if not path:
            path = str(download_library())
        _CONFIGURED[key] = load_library(path)
    return _CONFIGURED[key]


def get_component(component_id: str, *, library: ComponentLibrary | None = None) -> ChemicalComponent:
    r"""The library definition of one id; a miss → ``InsufficientInformationError(component_not_in_library)`` with the
    id and the artifact's content version. ``library`` must be a versioned CCD library, as returned by
    ``load_library`` or ``configured_library``."""
    library = library or configured_library()
    definition = library.definition(component_id)
    if definition is None:
        version = cast(ArtifactProvenance, library.provenance).content_version
        ref = RequestRef("component_id", component_id)
        fail("component_not_in_library", ref, id=component_id, content_version=version)
    return definition
