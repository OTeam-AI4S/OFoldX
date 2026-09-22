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
Identity-preserving union, rigid replication under explicit ``CopyId``\ s with the one copy join of external
connections (array level and value level) and batching.

A copy repeats its source instances' graph exactly (the gather of ``projection.project_arrays``): an edge from a copied
instance to an instance outside the copy becomes an external connection of the copy (identity transform, the partner
named by its source identity). The copy join turns an external connection into an edge when its partner is placed by
another copy at the composed placement and its reverse stub, if any, composes back (rotation 1e-5, translation 1e-3 Å);
``union`` re-joins identity external connections that name each other. An external connection the join leaves keeps a
transform that is exact in the output: its partner is the instance the output names, so where the copy
or that instance is placed by another matrix the transform becomes the table row of the composed placement, or an
UNKNOWN×NONE row when the table holds none. Chemistry is never re-derived: every fact is copied.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, StructureRef, fail
from ..provenance import Basis, Resolution, known, state
from .molecular import MolecularSystem, SystemArrays
from .projection import ProjectionMap, project_arrays, project_instances
from .schema import Schema, check_rows
from .stereo import ParityClears, StereoColumns, remap_parity
from .structure import CopyId, CopyKind, Structure, StructureArrays

V = TypeVar("V", MolecularSystem, Structure)
_ROTATION_TOLERANCE = 1e-5
_TRANSLATION_TOLERANCE = 1e-3
_UNKNOWN_NONE = state(Resolution.UNKNOWN, Basis.NONE)
#: Index columns and the axis they index (shifted when parts are concatenated).
_INDEX = {
    "instance_entity": "entity",
    "component_port": "atom",
    "atom_reference_row": "reference",
    "edge_index": "atom",
    "external_atom": "atom",
    "external_transform": "transform",
    "stereocentre_atom": "atom",
    "stereobond_edge": "edge",
    "stereogroup_member": "stereocentre",
}
#: Slot columns: an atom, ``-1`` implicit, or ``-(x + 2)`` for external row ``x``.
_SLOTS = ("stereocentre_neighbors", "stereobond_reference")
#: Each CSR pointer and the axis it points into.
_CHILD = {
    "system_entity_ptr": "entity",
    "system_instance_ptr": "instance",
    "instance_component_ptr": "component",
    "component_atom_ptr": "atom",
    "component_parent_ptr": "component_parent",
    "component_definition_parent_ptr": "component_definition_parent",
    "bond_declared_leaving_ptr": "bond_declared_leaving",
    "external_declared_leaving_ptr": "external_declared_leaving",
    "stereogroup_ptr": "stereogroup_member",
    "system_method_ptr": "method",
    "system_resolution_ptr": "resolution",
    "operation_ptr": "operation",
}
_FACTS = ("order", "aromatic", "kind", "direction")
#: The per-system segments of a structure and their payload columns.
_SEGMENTS = (
    ("system_method_ptr", ("method",)),
    ("system_resolution_ptr", ("resolution", "resolution_kind", "resolution_state")),
    ("operation_ptr", ("operation_id", "operation_rotation", "operation_translation")),
)
_SYSTEM_ROWS = (
    "source_sha256",
    "system_model_number",
    "system_model_number_state",
    "cell",
    "cell_state",
    "cell_quantum",
    "frame",
    "frame_state",
    "frame_quantum",
    "space_group",
    "space_group_state",
    "operation_numbering",
)


@dataclass(frozen=True)
class Copy:
    r"""Source instances placed by one Cartesian 4×4 ``matrix`` and named by ``copy_id`` (its asym per instance)."""

    instances: NDArray[np.int64]
    copy_id: CopyId
    matrix: NDArray[np.float64]


@dataclass(frozen=True)
class ReplicaMap:
    r"""Per output atom and instance: its source row and its copy (``-1`` for a kept source)."""

    atom_source: NDArray[np.int64]
    atom_copy: NDArray[np.int32]
    instance_source: NDArray[np.int64]
    instance_copy: NDArray[np.int32]


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _ragged(pointer: NDArray[np.int64], rows: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The pointer of segments ``rows`` (in that order) and the old index of every value they hold."""
    counts = pointer[rows + 1] - pointer[rows]
    new = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return new, np.repeat(pointer[rows] - new[:-1], counts) + np.arange(new[-1], dtype=np.int64)


def _close(placement: NDArray[np.float64], other: NDArray[np.float64]) -> bool:
    r"""Two Cartesian 4×4 placements equal within the rotation (1e-5) and translation (1e-3 Å) tolerances."""
    rotation = np.abs(placement[:3, :3] - other[:3, :3]).max()
    translation = np.abs(placement[:3, 3] - other[:3, 3]).max()
    return bool(rotation <= _ROTATION_TOLERANCE and translation <= _TRANSLATION_TOLERANCE)


def _cartesian_matrix(frame: NDArray[np.float64], rotation: NDArray, translation: NDArray) -> NDArray[np.float64]:
    r"""The Cartesian 4×4 of a fractional operation (R, τ; τ in 24ths) in the frame [A | u] (f = A·x + u): x ↦
    O·[R·(A·x + u) + τ − u], O = A⁻¹."""
    a, u = frame[:, :3], frame[:, 3]
    o = np.linalg.inv(a)
    matrix = np.eye(4)
    matrix[:3, :3] = o @ rotation @ a
    matrix[:3, 3] = o @ (rotation @ u + translation / 24.0 - u)
    return matrix


def _sizes(columns: Mapping[str, NDArray], schema: Schema) -> dict[str, int]:
    r"""The row count of every axis of one part."""
    sizes = {}
    for field in schema.fields:
        rows = columns[field.name].shape[len(field.leading)]
        if field.axis.endswith("+1"):
            sizes[field.axis.removesuffix("+1")] = rows - 1
        else:
            sizes[field.axis] = rows
    return sizes


def _concat(
    parts: Sequence[tuple[Mapping[str, tuple[str, ...]], Mapping[str, NDArray]]], schema: Schema
) -> tuple[dict[str, tuple[str, ...]], dict[str, NDArray]]:
    r"""Concatenate ``(symbols, columns)`` parts of one schema: families merged, codes and indices remapped."""
    families: dict[str, dict[str, int]] = {}
    recodes = []
    for symbols, _ in parts:
        recode = {}
        for family, table in symbols.items():
            index = families.setdefault(family, {})
            codes = [index.setdefault(text, len(index)) for text in table]
            recode[family] = np.array(codes + [-1], dtype=np.int64)  # the code -1 stays -1
        recodes.append(recode)
    offsets: list[dict[str, int]] = []
    total: dict[str, int] = {}
    for _, columns in parts:
        offsets.append(dict(total))
        for axis, size in _sizes(columns, schema).items():
            total[axis] = total.get(axis, 0) + size
    out = {}
    for field in schema.fields:
        pieces: list[NDArray] = []
        for (_, columns), offset, recode in zip(parts, offsets, recodes):
            array = np.asarray(columns[field.name])
            if field.axis.endswith("+1"):
                if pieces:  # a pointer after the first continues from the previous total
                    array = array[1:] + offset.get(_CHILD[field.name], 0)
            elif field.symbols:
                array = recode[field.symbols][array]
            elif field.name in _INDEX:
                array = np.where(array >= 0, array + offset.get(_INDEX[field.name], 0), array)
            elif field.name in _SLOTS:
                array = np.where(array >= 0, array + offset.get("atom", 0), array)
                array = np.where(array <= -2, array - offset.get("external", 0), array)
            pieces.append(array)
        out[field.name] = np.concatenate(pieces, axis=len(field.leading))
    return {family: tuple(index) for family, index in families.items()}, out


def _dedupe_transforms(c: MutableMapping[str, NDArray]) -> None:
    r"""One transform row per distinct (rotation, translation, code, state), in order of first use."""
    rows = np.concatenate(
        (
            c["transform_rotation"].reshape(-1, 9),
            c["transform_translation"],
            c["transform_code"][:, None],
            c["transform_state"][:, None],
        ),
        axis=1,
    ).astype(np.int64)
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    new = np.empty(len(order), dtype=np.int64)
    new[order] = np.arange(len(order))
    for name in ("transform_rotation", "transform_translation", "transform_code", "transform_state"):
        c[name] = c[name][first[order]]
    c["external_transform"] = new[inverse.reshape(-1)][c["external_transform"]]


def _one_system(c: MutableMapping[str, NDArray], s: MutableMapping[str, NDArray] | None) -> None:
    r"""Collapse the systems of concatenated single-system parts into one; their system facts must agree."""
    c["system_entity_ptr"] = c["system_entity_ptr"][[0, -1]]
    c["system_instance_ptr"] = c["system_instance_ptr"][[0, -1]]
    if s is None:
        return
    for pointer, payload in _SEGMENTS:
        segments = s[pointer]
        for name in payload:
            pieces = np.split(s[name], segments[1:-1])
            if any(not np.array_equal(piece, pieces[0]) for piece in pieces):
                fail("value_invalid", RequestRef("values", "systems differ"), item=name)
            s[name] = pieces[0]
        s[pointer] = np.array([0, segments[1]], dtype=np.int64)
    for name in _SYSTEM_ROWS:
        if not (s[name] == s[name][:1]).all():
            fail("value_invalid", RequestRef("values", "systems differ"), item=name)
        s[name] = s[name][:1]


def _merge_entities(c: MutableMapping[str, NDArray], *, sequences: bool) -> None:
    r"""
    One entity row per symbol in a single system. Rows sharing a symbol must state the same facts; with ``sequences``
    (no source document numbers the entities) every instance of a merged entity must also have one component sequence.
    """

    symbol = c["entity_symbol"]
    _, first, inverse = np.unique(symbol, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    for name in ("entity_type", "entity_polymer_type", "entity_nstd_linkage"):
        for column in (name, f"{name}_state"):
            if (c[column] != c[column][first][inverse]).any():
                fail("value_invalid", StructureRef(item=column), invariant="entity_facts_differ")
    if sequences:
        merged = np.bincount(inverse) > 1
        pointer = c["instance_component_ptr"]
        seen: dict[int, tuple[int, ...]] = {}
        for instance, entity in enumerate(inverse[c["instance_entity"]]):
            sequence = tuple(c["component_symbol"][pointer[instance] : pointer[instance + 1]].tolist())
            if merged[entity] and seen.setdefault(int(entity), sequence) != sequence:
                fail("value_invalid", StructureRef(instance=instance), invariant="entity_sequences_differ")
    keep = np.sort(first)
    new = np.full(len(symbol), -1, dtype=np.int64)
    new[keep] = np.arange(len(keep))
    for field in MolecularSystem.SCHEMA.fields:
        if field.axis == "entity":
            c[field.name] = c[field.name][keep]
    c["instance_entity"] = new[first[inverse]][c["instance_entity"]]
    c["system_entity_ptr"] = np.array([0, len(keep)], dtype=np.int64)


def _cartesian(c: Mapping[str, NDArray], s: Mapping[str, NDArray] | None) -> list[NDArray[np.float64] | None]:
    r"""Per transform row its Cartesian 4×4 matrix, or ``None`` where the row or the frame is not KNOWN."""
    frame = None
    if s is not None and known(s["frame_state"][:1]).all():
        frame = s["frame"][0]
    out: list[NDArray[np.float64] | None] = []
    for rotation, translation, is_known in zip(
        c["transform_rotation"], c["transform_translation"], known(c["transform_state"])
    ):
        if is_known and (rotation == np.eye(3)).all() and not translation.any():
            out.append(np.eye(4))
        elif is_known and frame is not None:
            out.append(_cartesian_matrix(frame, rotation, translation))
        else:
            out.append(None)
    return out


def _keys(c: Mapping[str, NDArray]) -> NDArray[np.int64]:
    r"""Per atom its identity (instance symbol code, component ordinal, atom symbol code)."""
    atom_component = _owner(c["component_atom_ptr"])
    atom_instance = _owner(c["instance_component_ptr"])[atom_component]
    ordinal = atom_component - c["instance_component_ptr"][atom_instance]
    return np.stack((c["instance_symbol"][atom_instance], ordinal, c["atom_symbol"]), axis=1).astype(np.int64)


def _same_fields(c: Mapping[str, NDArray], e: int, f: int) -> bool:
    r"""Two stubs of one connection: equal facts and states, opposite direction, swapped leaving sides."""
    names = ["external_direction_state", "external_origin", "external_connection_type"]
    for fact in ("order", "aromatic", "kind"):
        names += [f"external_{fact}", f"external_{fact}_state"]
    if any(c[name][e] != c[name][f] for name in names) or c["external_direction"][e] != -c["external_direction"][f]:
        return False
    pointer = c["external_declared_leaving_ptr"]
    leaving, sides = c["external_declared_leaving"], c["external_declared_leaving_side"]
    left = set(zip(leaving[pointer[e] : pointer[e + 1]], sides[pointer[e] : pointer[e + 1]]))
    right = set(zip(leaving[pointer[f] : pointer[f + 1]], 1 - sides[pointer[f] : pointer[f + 1]]))
    return left == right


def _join(
    c: MutableMapping[str, NDArray], s: Mapping[str, NDArray] | None, group, keys, matrices, *, union: bool
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""
    Turn external connections into edges where their partner is placed at the composed placement.

    ``group`` is the copy of every atom, ``keys`` its source identity and ``matrices`` the placement of every copy. A
    reverse stub (at the partner, naming the atom) is consumed with its connection only when its own transform composes
    back, so a symmetry stub is never taken for an identity one; with ``union`` only identity connections join and each
    needs an equal reverse stub (``stub_mismatch`` otherwise). Stereo slots follow (an external slot becomes the partner
    atom). Returns the edge map (old row → new row) and the external map (old row → partner atom, or the new
    ``-(row + 2)``).
    """

    count = len(c["external_atom"])
    cartesian = _cartesian(c, s)
    atoms_of: dict[tuple, list[int]] = {}
    for atom, key in enumerate(map(tuple, keys.tolist())):
        atoms_of.setdefault(key, []).append(atom)
    partner = np.stack(
        (c["external_partner_instance"], c["external_partner_component"], c["external_partner_atom"]), axis=1
    ).astype(np.int64)
    consumed = np.zeros(count, dtype=bool)
    joined_to = np.full(count, -1, dtype=np.int64)
    pairs = []
    for e in range(count):
        a = int(c["external_atom"][e])
        transform = cartesian[int(c["external_transform"][e])]
        if consumed[e] or transform is None or (union and not _close(transform, np.eye(4))):
            continue
        placed = matrices[group[a]] @ transform
        b = -1
        for candidate in atoms_of.get(tuple(partner[e]), ()):
            if candidate != a and _close(matrices[group[candidate]], placed):
                b = candidate
                break
        if b < 0:
            continue
        reverse = None
        for f in np.flatnonzero((c["external_atom"] == b) & ~consumed):
            back = cartesian[int(c["external_transform"][f])]
            if (
                tuple(partner[f]) == tuple(keys[a])
                and back is not None
                and _close(matrices[group[b]] @ back, matrices[group[a]])
            ):
                reverse = f
                break
        if union and (reverse is None or not _same_fields(c, e, reverse)):
            fail("stub_mismatch", StructureRef(external=e), partner=int(b))
        consumed[e], joined_to[e] = True, b
        if reverse is not None:
            consumed[reverse], joined_to[reverse] = True, a
        pairs.append((e, a, b))
    edge_map = _edges_from(c, pairs, consumed)
    external_map = np.where(consumed, joined_to, -(np.cumsum(~consumed) - 1) - 2)
    n = len(c["atom_symbol"])
    stereo, _ = remap_parity(
        StereoColumns.read(c),
        np.arange(n, dtype=np.int64),
        removed_hydrogen=np.zeros(n, dtype=bool),
        substitutions=np.zeros((0, 3), dtype=np.int64),
        externalised=np.full(n, -1, dtype=np.int64),
        edge_map=edge_map,
        external_map=external_map,
    )
    stereo.write(c)
    return edge_map, external_map


def _edges_from(
    c: MutableMapping[str, NDArray], pairs: list[tuple[int, int, int]], consumed: NDArray[np.bool_]
) -> NDArray[np.int64]:
    r"""Append the joined connections as edges (oriented low → high), drop the consumed externals, re-sort the edges."""
    rows = np.array([e for e, _, _ in pairs], dtype=np.int64)
    a = np.array([a for _, a, _ in pairs], dtype=np.int64)
    b = np.array([b for _, _, b in pairs], dtype=np.int64)
    flip = a > b
    old_edges = len(c["edge_index"][0])
    c["edge_index"] = np.concatenate((c["edge_index"], np.stack((np.minimum(a, b), np.maximum(a, b)))), axis=1)
    for fact in _FACTS:
        value = c[f"external_{fact}"][rows]
        if fact == "direction":
            value = np.where(flip, -value, value)
        c[f"bond_{fact}"] = np.concatenate((c[f"bond_{fact}"], value))
        c[f"bond_{fact}_state"] = np.concatenate((c[f"bond_{fact}_state"], c[f"external_{fact}_state"][rows]))
    c["bond_origin"] = np.concatenate((c["bond_origin"], c["external_origin"][rows]))
    c["bond_connection_type"] = np.concatenate((c["bond_connection_type"], c["external_connection_type"][rows]))
    leaving_ptr, leaving = _ragged(c["external_declared_leaving_ptr"], rows)
    sides = c["external_declared_leaving_side"][leaving]
    sides = np.where(np.repeat(flip, np.diff(leaving_ptr)), 1 - sides, sides)
    pointer = c["bond_declared_leaving_ptr"]
    c["bond_declared_leaving_ptr"] = np.concatenate((pointer, pointer[-1] + leaving_ptr[1:]))
    c["bond_declared_leaving"] = np.concatenate((c["bond_declared_leaving"], c["external_declared_leaving"][leaving]))
    c["bond_declared_leaving_side"] = np.concatenate((c["bond_declared_leaving_side"], sides))

    first, second = c["edge_index"].astype(np.int64)
    order = np.lexsort((second, first))
    edge_map = np.empty(len(order), dtype=np.int64)
    edge_map[order] = np.arange(len(order))
    c["bond_declared_leaving_ptr"], leaving = _ragged(c["bond_declared_leaving_ptr"], order)
    c["bond_declared_leaving"] = c["bond_declared_leaving"][leaving]
    c["bond_declared_leaving_side"] = c["bond_declared_leaving_side"][leaving]
    c["edge_index"] = c["edge_index"][:, order]
    for name in list(c):
        if name.startswith("bond_") and "declared" not in name:
            c[name] = c[name][order]

    kept = np.flatnonzero(~consumed)
    c["external_declared_leaving_ptr"], leaving = _ragged(c["external_declared_leaving_ptr"], kept)
    c["external_declared_leaving"] = c["external_declared_leaving"][leaving]
    c["external_declared_leaving_side"] = c["external_declared_leaving_side"][leaving]
    for name in list(c):
        if name.startswith("external_") and "declared_leaving" not in name:
            c[name] = c[name][kept]
    return edge_map[:old_edges]


def _restate(c: MutableMapping[str, NDArray], s: Mapping[str, NDArray], group, instance_group, matrices) -> None:
    r"""
    Make every external connection the copy join left exact in the output (module docstring): the partner is the
    transform applied to the output instance carrying the partner symbol (or, if none, to the source partner).
    """

    cartesian, identity = _cartesian(c, s), np.eye(4)
    named = {int(code): instance for instance, code in enumerate(c["instance_symbol"])}
    unknown = None
    for e in range(len(c["external_atom"])):
        code = int(c["external_partner_instance"][e])
        here = matrices[group[c["external_atom"][e]]]
        there = matrices[instance_group[named[code]]] if code in named else identity
        if _close(here, identity) and _close(there, identity):
            continue
        source = cartesian[int(c["external_transform"][e])]
        found = []
        if source is not None:
            target = here @ source @ np.linalg.inv(there)
            found = [k for k, matrix in enumerate(cartesian) if matrix is not None and _close(matrix, target)]
        if found:
            c["external_transform"][e] = found[0]
            continue
        if unknown is None:  # one UNKNOWN×NONE row for every placement the table does not hold
            unknown = len(c["transform_state"])
            for name, value in (("rotation", 0), ("translation", 0), ("code", -1), ("state", _UNKNOWN_NONE)):
                column = c[f"transform_{name}"]
                row = np.full((1, *column.shape[1:]), value, column.dtype)
                c[f"transform_{name}"] = np.concatenate((column, row))
        c["external_transform"][e] = unknown


def union(values: Sequence[V]) -> tuple[V, tuple[ProjectionMap, ...]]:
    r"""
    One single-system value holding every instance of the single-system ``values``.

    Families are merged; entities are merged by symbol: they must state the same facts and, when no source document
    numbers them (``MolecularSystem`` values, structures without a document), their instances one component sequence
    (``value_invalid`` otherwise). External connections that name each other with identity transforms and equal fields
    are re-joined into one edge; an identity connection whose partner is present without such a stub is
    ``stub_mismatch``. A repeated instance symbol is ``instance_collision``. The values must agree on their source
    document and frame.
    """

    values = list(values)
    if not values:
        fail("value_invalid", RequestRef("values", "[]"), invariant="nonempty")
    systems = [value if isinstance(value, MolecularSystem) else value.system for value in values]
    system_symbols, c = _concat([(dict(v.symbols()), v.build().columns) for v in systems], MolecularSystem.SCHEMA)
    symbols = s = None
    if isinstance(values[0], Structure):
        symbols, s = _concat([(dict(v.symbols()), v.build().columns) for v in values], Structure.SCHEMA)
    _one_system(c, s)
    _merge_entities(c, sequences=s is None or not s["source_sha256"].any())
    _dedupe_transforms(c)
    if len(np.unique(c["instance_symbol"])) != len(c["instance_symbol"]):
        fail("instance_collision", RequestRef("values", "instance symbols"))
    group = np.zeros(len(c["atom_symbol"]), dtype=np.int64)
    edge_map, external_map = _join(c, s, group, _keys(c), np.eye(4)[None], union=True)
    system = MolecularSystem.from_arrays(SystemArrays(system_symbols, c))
    value = system if s is None else Structure.from_arrays(system, StructureArrays(symbols, s))

    no_clears = ParityClears(np.zeros(0, np.int64), np.zeros(0, np.uint16))
    maps, start = [], np.zeros(5, dtype=np.int64)
    for part in systems:
        sizes = [len(part.atom_symbol), len(part.component_symbol), len(part.instance_symbol), len(part.external_atom)]
        sizes.append(part.edge_index.shape[1])
        atom, component, instance, external, edge = (start[k] + np.arange(sizes[k]) for k in range(5))
        code = external_map[external]
        external = np.where(code <= -2, -code - 2, -1)
        maps.append(ProjectionMap(atom, component, instance, external, edge_map[edge], no_clears))
        start += sizes
    return value, tuple(maps)


def _projections(system: SystemArrays, structure: StructureArrays, instance_sets: Sequence[NDArray[np.int64]]) -> list:
    r"""
    Per instance set its ``project_arrays`` (every atom) and the source atom of each atom it holds, at the cost of about
    one projection of the source per halving of the distinct sets rather than one per set (9fqr: 2,606 asyms, 12 M
    atoms; a set copied under several operators is projected once). Each half is projected from a nested
    neighbourhood: the projection of instances T and of every instance bonded to one of them holds every edge, external
    row and stereo row of T's atoms in source order (its gather is monotone and the edges are sorted, S8), so
    projecting a subset of T from it is projecting that subset from the source.
    """

    c0 = system.columns
    count = len(c0["instance_symbol"])
    instance_of = _owner(c0["instance_component_ptr"])[_owner(c0["component_atom_ptr"])]
    ends = instance_of[c0["edge_index"].astype(np.int64)]
    ends = ends[:, ends[0] != ends[1]]
    ends = np.concatenate((ends, ends[::-1]), axis=1)
    ends = ends[:, np.argsort(ends[0], kind="stable")]
    bonded_ptr, bonded = np.searchsorted(ends[0], np.arange(count + 1)), ends[1]
    out: list = [None] * len(instance_sets)

    def project(system, structure, origin, instances):
        part, placed, projection = project_arrays(system, structure, instances, np.ones(len(origin), dtype=bool))
        kept = projection.atom >= 0
        source = np.empty(len(part.columns["atom_symbol"]), dtype=np.int64)
        source[projection.atom[kept]] = origin[kept]
        return part, placed, source

    def solve(system, structure, origin, local, members):  # origin: the source atoms, local: source → own instances
        if len(members) == 1:
            out[members[0]] = project(system, structure, origin, local[instance_sets[members[0]]])
            return
        for half in (members[: len(members) // 2], members[len(members) // 2 :]):
            target = np.unique(np.concatenate([instance_sets[k] for k in half]))
            needed = np.union1d(target, bonded[_ragged(bonded_ptr, target)[1]])
            if len(needed) == len(system.columns["instance_symbol"]):
                solve(system, structure, origin, local, half)
                continue
            inner = np.full(count, -1, dtype=np.int64)
            inner[needed] = np.arange(len(needed))
            solve(*project(system, structure, origin, local[needed]), inner, half)

    first: dict[tuple[int, ...], int] = {}
    for k, instances in enumerate(instance_sets):
        first.setdefault(tuple(instances.tolist()), k)
    if first:
        atoms = np.arange(len(c0["atom_symbol"]), dtype=np.int64)
        solve(system, structure, atoms, np.arange(count, dtype=np.int64), list(first.values()))
    return [out[first[tuple(instances.tolist())]] for instances in instance_sets]


def replicate_arrays(
    system: SystemArrays, structure: StructureArrays, copies: Sequence[Copy], *, keep_source: bool
) -> tuple[SystemArrays, StructureArrays, ReplicaMap]:
    r"""
    The one copy join, on build buffers: the source instances (when ``keep_source``) and then every copy in the
    given order, each repeating its instances' graph exactly with coordinates placed by its matrix (evidence ``NONE``
    stays at canonical zero). The n-th copy of asym ``A`` is ``A`` for n = 1 and ``A-n``
    otherwise (a kept source is copy 1 of its asym), a crystal copy ``A@<n_klm>``; a generated symbol equal to another
    asym id is ``UnsupportedError(copy_symbol_collision)``, an improper matrix ``UnsupportedError(improper_copy)``.
    """

    c0, s0 = system.columns, structure.columns
    count = len(c0["instance_symbol"])
    groups: list[tuple[NDArray[np.int64], NDArray[np.float64], CopyId | None]] = (
        [(np.arange(count, dtype=np.int64), np.eye(4), None)] if keep_source else []
    )
    for copy in copies:  # the raw copies, before any of them indexes: distinct source instances, one 4×4 placement
        instances = check_rows("Copy.instances", copy.instances, count, shape=(None,))
        matrix = np.asarray(copy.matrix, dtype=float)
        if len(np.unique(instances)) != len(instances):
            fail("value_invalid", RequestRef("Copy.instances", str(instances.tolist())), invariant="distinct_rows")
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            fail("value_invalid", RequestRef("Copy.matrix", str(matrix.shape)), expected=(4, 4))
        if np.linalg.det(matrix[:3, :3]) < 0:
            fail("improper_copy", RequestRef("Copy.matrix", "det < 0"), copy_id=repr(copy.copy_id))
        groups.append((instances, matrix, copy.copy_id))

    system_families = {family: list(table) for family, table in system.symbols.items()}
    families = {family: list(table) for family, table in structure.symbols.items()}
    asym_texts = structure.symbols["asym"]
    names = []  # the asym id of every source instance, else its instance symbol
    for asym, instance in zip(s0["instance_source_asym"], c0["instance_symbol"]):
        names.append(asym_texts[asym] if asym >= 0 else system.symbols["instance"][instance])
    counter = dict.fromkeys(names, 1) if keep_source else {}  # a kept source is copy n = 1 of its asym

    parts_system, parts_structure = [], []
    sources, atom_groups, instance_groups = [], [], []
    atom_copy, instance_source, instance_copy = [], [], []
    projections = _projections(system, structure, [instances for instances, _, _ in groups])
    for index, ((instances, matrix, copy_id), (part, placed, atoms)) in enumerate(zip(groups, projections)):
        c, s = dict(part.columns), dict(placed.columns)
        observed = s["coordinate_evidence"] != 0
        moved = s["coordinates"] @ matrix[:3, :3].T + matrix[:3, 3]
        s["coordinates"] = np.where(observed[:, None], moved, 0.0)
        if copy_id is not None:  # a copy's coordinates are derived values: no source printed them
            s["coordinate_quantum"] = np.zeros_like(s["coordinate_quantum"])
        if copy_id is not None:
            if copy_id.operators and copy_id.operator_symbol not in families["operator"]:
                families["operator"].append(copy_id.operator_symbol)
            codes = []
            for instance in instances:
                name = names[instance]
                counter[name] = counter.get(name, 0) + 1
                if copy_id.kind == CopyKind.CRYSTAL:
                    text = f"{name}@{copy_id.operators[-1]}"
                elif counter[name] > 1:
                    text = f"{name}-{counter[name]}"
                else:
                    text = name
                if text in asym_texts and text != name:
                    fail("copy_symbol_collision", RequestRef("instance", text))
                if text not in system_families["instance"]:
                    system_families["instance"].append(text)
                codes.append(system_families["instance"].index(text))
            c["instance_symbol"] = np.array(codes, dtype=np.int64)
            s["instance_copy_kind"] = np.full(len(instances), int(copy_id.kind), dtype=np.uint8)
            operator = families["operator"].index(copy_id.operator_symbol) if copy_id.operators else -1
            s["instance_copy_operators"] = np.full(len(instances), operator)
        number = index - keep_source if copy_id is not None else -1
        parts_system.append(c)
        parts_structure.append(s)
        sources.append(atoms)
        atom_groups.append(np.full(len(atoms), index))
        instance_groups.append(np.full(len(instances), index))
        atom_copy.append(np.full(len(atoms), number))
        instance_source.append(instances)
        instance_copy.append(np.full(len(instances), number))

    system_symbols = {family: tuple(table) for family, table in system_families.items()}
    symbols = {family: tuple(table) for family, table in families.items()}
    _, c = _concat([(system_symbols, part) for part in parts_system], MolecularSystem.SCHEMA)
    _, s = _concat([(symbols, part) for part in parts_structure], Structure.SCHEMA)
    _one_system(c, s)
    _merge_entities(c, sequences=False)  # every copy repeats one source entity row
    atom_source = np.concatenate(sources).astype(np.int64)
    c["atom_reference_row"] = c0["atom_reference_row"][atom_source]
    for name in ("reference_pool_coordinates", "reference_pool_source"):
        c[name] = c0[name]
    if len(np.unique(c["instance_symbol"])) != len(c["instance_symbol"]):
        fail("copy_symbol_collision", RequestRef("instance", "duplicate"))
    matrices = np.stack([matrix for _, matrix, _ in groups]) if groups else np.eye(4)[None]
    atom_group, instance_group = np.concatenate(atom_groups), np.concatenate(instance_groups)
    _join(c, s, atom_group, _keys(c0)[atom_source], matrices, union=False)
    _restate(c, s, atom_group, instance_group, matrices)
    _dedupe_transforms(c)
    replica = ReplicaMap(
        atom_source=atom_source,
        atom_copy=np.concatenate(atom_copy).astype(np.int32),
        instance_source=np.concatenate(instance_source).astype(np.int64),
        instance_copy=np.concatenate(instance_copy).astype(np.int32),
    )
    return SystemArrays(system_symbols, c), StructureArrays(symbols, s), replica


def replicate(structure: Structure, copies: Sequence[Copy], *, keep_source: bool) -> tuple[Structure, ReplicaMap]:
    r"""``replicate_arrays`` on a value: build, replicate, construct once."""
    system_arrays, structure_arrays, replica = replicate_arrays(
        structure.system.build(), structure.build(), copies, keep_source=keep_source
    )
    return Structure.from_arrays(MolecularSystem.from_arrays(system_arrays), structure_arrays), replica


def batch(structures: Sequence[Structure]) -> Structure:
    r"""One value whose systems are those of ``structures`` in order (families merged; no merge or join of systems)."""
    if not len(structures):
        fail("value_invalid", RequestRef("structures", "[]"), invariant="nonempty")
    systems = [(dict(v.system.symbols()), v.system.build().columns) for v in structures]
    system_symbols, c = _concat(systems, MolecularSystem.SCHEMA)
    symbols, s = _concat([(dict(v.symbols()), v.build().columns) for v in structures], Structure.SCHEMA)
    _dedupe_transforms(c)
    system = MolecularSystem.from_arrays(SystemArrays(system_symbols, c))
    return Structure.from_arrays(system, StructureArrays(symbols, s))


def unbatch(structure: Structure) -> tuple[Structure, ...]:
    r"""One value per system of ``structure`` (each an instance projection without the other systems)."""
    system = structure.system
    instance_system, entity_system = _owner(system.system_instance_ptr), _owner(system.system_entity_ptr)
    out = []
    for k in range(len(system.system_instance_ptr) - 1):
        value, _ = project_instances(structure, instance_system == k)
        arrays, built = value.system.build(), value.build()
        c, s = arrays.columns, built.columns
        keep = entity_system == k
        for field in MolecularSystem.SCHEMA.fields:
            if field.axis == "entity":
                c[field.name] = c[field.name][keep]
        c["instance_entity"] = (np.cumsum(keep) - 1)[c["instance_entity"]]
        c["system_entity_ptr"] = np.array([0, int(keep.sum())], dtype=np.int64)
        c["system_instance_ptr"] = c["system_instance_ptr"][[0, -1]]
        for field in Structure.SCHEMA.fields:
            if field.axis == "system":
                s[field.name] = s[field.name][k : k + 1]
        for pointer, payload in _SEGMENTS:
            start, stop = s[pointer][k], s[pointer][k + 1]
            for name in payload:
                s[name] = s[name][start:stop]
            s[pointer] = np.array([0, stop - start], dtype=np.int64)
        out.append(Structure.from_arrays(MolecularSystem.from_arrays(arrays), built))
    return tuple(out)
