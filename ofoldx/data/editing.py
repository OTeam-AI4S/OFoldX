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
The one declared chemical edit (``apply_edit``): remove atoms and edges, add edges, replace whole
components, set atom facts with their final states, and state the claims of every touched component.

An edit never infers chemistry: every final row and state comes from the recipe, and a hydrogen count the edit's change
rests on is withdrawn unless the recipe states it again. Untouched atoms and edges keep every fact; parity is carried by
``remap_parity`` (a new neighbour of a centre takes no vacated slot: its parity clears). A connection between components
is never dropped silently: removing an atom with such a connection needs the edge listed in ``remove_edges``; one with
an external connection is refused. Components are never added or removed, so component indices are stable across an
edit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, fail
from ..provenance import Basis, Resolution, known, state
from .molecular import MolecularSystem, SystemArrays
from .schema import check_mask, check_rows, fits
from .stereo import ParityClears, StereoColumns, remap_parity
from .structure import CoordinateEvidence, Structure, StructureArrays

V = TypeVar("V", MolecularSystem, Structure)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_UNKNOWN_NONE = state(Resolution.UNKNOWN, Basis.NONE)
_ATOM_COLUMNS = tuple(f.name for f in MolecularSystem.SCHEMA.fields if f.axis == "atom" and f.name != "atom_symbol")
_CLAIMS = np.zeros((0, 2), dtype=bool)  # a component's two claims (topology, heavy atoms), as one row
_BOND_COLUMNS = tuple(
    f.name
    for f in MolecularSystem.SCHEMA.fields
    if f.axis == "edge" and f.name not in ("edge_index", "bond_connection_type")
)


@dataclass(frozen=True)
class EdgeRows:
    r"""
    Complete final edge rows to add. ``atoms`` ``[2, m]``: old atom indices, each surviving the edit. ``columns`` holds
    every edge column but the index and connection type, with its state; the direction is relative to ``atoms``.
    ``leaving`` lists per edge its declared leaving ids and sides.
    """

    atoms: NDArray[np.int64]
    columns: Mapping[str, NDArray]
    connection_type: tuple[str | None, ...]
    leaving: tuple[tuple[tuple[str, int], ...], ...]


@dataclass(frozen=True)
class ComponentRows:
    r"""
    One final component: its atoms (names and every other atom column, ``atom_reference_row`` local to ``reference``),
    internal edges (local indices, every edge column but the index and connection type), stereo (local indices; the
    implicit slot ``-1``, no external slot), claims, parents (the occurrence's own annotation, then the
    definition's), monomer type, definition scope and digest, ports (local, ``-1``) and conformer rows.
    """

    symbol: str
    names: tuple[str, ...]
    atoms: Mapping[str, NDArray]
    edges: NDArray[np.int64]
    bonds: Mapping[str, NDArray]
    stereo: StereoColumns
    claims: tuple[bool, bool]
    parents: tuple[str, ...]
    definition_parents: tuple[str, ...]
    monomer_type: int
    monomer_type_state: int
    definition_scope: int
    definition_digest: int
    port: tuple[int, int]
    reference: NDArray[np.float32]
    reference_source: NDArray[np.uint8]


@dataclass(frozen=True)
class ChemicalEdit:
    r"""A declared edit; atom and edge indices are those of the value it applies to."""

    remove_atoms: NDArray[np.int64]
    replace_components: Mapping[int, ComponentRows]
    retained_atoms: Mapping[int, NDArray[np.int64]]
    remove_edges: NDArray[np.int64]
    add_edges: EdgeRows
    set_atoms: Mapping[str, tuple[NDArray[np.int64], NDArray, NDArray[np.uint8] | None]]
    claims: Mapping[int, tuple[bool, bool]]
    keep_coordinates: NDArray[np.bool_] | None
    identity: Literal["withdraw", "inherit"]


@dataclass(frozen=True)
class AtomMap:
    r"""Old atom → new atom, ``-1`` for an atom the edit removed."""

    old_to_new: NDArray[np.int64]


@dataclass(frozen=True)
class EditReport:
    r"""Parity rows cleared, claims set per component, atoms whose coordinates were invalidated, identities changed."""

    parity_clears: ParityClears
    claims_set: Mapping[int, tuple[bool, bool]]
    coordinates_invalidated: NDArray[np.int64]
    identity_changed: NDArray[np.int64]


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _pointer(counts) -> NDArray[np.int64]:
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def _ragged(pointer: NDArray[np.int64], rows: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    counts = pointer[rows + 1] - pointer[rows]
    new = _pointer(counts)
    return new, np.repeat(pointer[rows] - new[:-1], counts) + np.arange(new[-1], dtype=np.int64)


def _survive(atoms: NDArray[np.int64], atom_map: NDArray[np.int64]) -> bool:
    r"""Every atom is one the edit keeps."""
    return bool((atom_map[atoms] >= 0).all())


def _invalid(argument: str, value: object, **detail: object) -> None:
    fail("edit_invalid", RequestRef(argument, str(value)), **detail)


def _index(argument: str, values, rows: int, low: int = 0, shape: tuple = (None,)) -> None:
    check_rows(argument, values, rows, low=low, shape=shape, code="edit_invalid")


def _count(argument: str, name: object, values: tuple, count: int) -> None:
    if len(values) != count:
        _invalid(argument, name, invariant="row_count", rows=count)


def _column(argument: str, name: object, values, column: NDArray, count: int) -> None:
    r"""``count`` rows for ``column``: exactly its row shape (the schema's trailing dimensions, as the value holds them,
    else ``edit_invalid``) and values its dtype holds exactly (``schema.fits``, the publish rule, else
    ``precision_loss``); checked before the edit writes them."""
    array = np.asarray(values)
    if array.shape != (count, *column.shape[1:]):
        _invalid(argument, name, invariant="shape", expected=str((count, *column.shape[1:])))
    if not fits(array, column.dtype):
        fail("precision_loss", RequestRef(argument, str(name)), dtype=str(column.dtype))


def _check(edit: ChemicalEdit, c: Mapping[str, NDArray]) -> None:
    r"""The raw recipe, before any of it indexes or is cast: every index names a row it may name and every value is
    one its column holds."""
    n, m, components = len(c["atom_symbol"]), c["edge_index"].shape[1], len(c["component_symbol"])
    _index("remove_atoms", edit.remove_atoms, n)
    _index("remove_edges", edit.remove_edges, m)
    for name, (atoms, values, states) in edit.set_atoms.items():
        stated = f"{name}_state" in c
        if name not in _ATOM_COLUMNS or not (stated or name == "isotope") or (states is None) == stated:
            _invalid("set_atoms", name, invariant="atom_fact")
        _index("set_atoms", atoms, n)
        _column("set_atoms", name, values, c[name], np.asarray(atoms).size)
        if stated:
            _column("set_atoms", f"{name}_state", states, c[f"{name}_state"], np.asarray(atoms).size)
    new = edit.add_edges
    _index("add_edges", new.atoms, n, shape=(2, None))
    ends = np.asarray(new.atoms)
    for name in _BOND_COLUMNS:
        _column("add_edges", name, new.columns[name], c[name], ends.shape[1])
    _count("add_edges", "connection_type", new.connection_type, ends.shape[1])
    _count("add_edges", "leaving", new.leaving, ends.shape[1])
    _index("claims", list(edit.claims), components)
    for component, claim in edit.claims.items():
        _column("claims", component, [claim], _CLAIMS, 1)
    _index("replace_components", list(edit.replace_components), components)
    for component, rows in edit.replace_components.items():
        size, stereo = len(rows.names), rows.stereo
        _column("replace_components", "claims", [rows.claims], _CLAIMS, 1)
        _index("replace_components.edges", rows.edges, size, shape=(2, None))
        local = np.asarray(rows.edges)
        _index("retained_atoms", edit.retained_atoms.get(component, np.full(size, -1)), n, low=-1, shape=(size,))
        for name in _ATOM_COLUMNS:
            _column("replace_components", name, rows.atoms[name], c[name], size)
        _index("replace_components", rows.atoms["atom_reference_row"], len(rows.reference_source), low=-1)
        for name in _BOND_COLUMNS:
            _column("replace_components", name, rows.bonds[name], c[name], local.shape[1])
        centres, bonds = np.asarray(stereo.centre).size, np.asarray(stereo.bond_edge).size
        for argument, indices, rows_of, low, shape in (
            ("port", rows.port, size, -1, (2,)),
            ("stereo.centre", stereo.centre, size, 0, (None,)),
            ("stereo.neighbors", stereo.neighbors, size, -1, (centres, 4)),
            ("stereo.bond_edge", stereo.bond_edge, local.shape[1], 0, (None,)),
            ("stereo.bond_reference", stereo.bond_reference, size, -1, (bonds, 2)),
        ):
            _index(f"replace_components.{argument}", indices, rows_of, low, shape)
        for name in ("definition_scope", "definition_digest", "monomer_type", "monomer_type_state"):
            _column("replace_components", name, [getattr(rows, name)], c[f"component_{name}"], 1)
    if edit.keep_coordinates is not None:
        check_mask("keep_coordinates", edit.keep_coordinates, n, code="edit_invalid")
    if edit.identity not in ("withdraw", "inherit"):
        _invalid("identity", edit.identity)


def _code(family: list[str], text: str | None) -> int:
    if text is None:
        return -1
    if text not in family:
        family.append(text)
    return family.index(text)


def _drop_centres(stereo: StereoColumns, drop: NDArray[np.bool_]) -> StereoColumns:
    r"""The rows whose centre is not dropped; group members follow, emptied groups go."""
    keep = ~drop[stereo.centre]
    row = np.where(keep, np.cumsum(keep) - 1, -1)
    group = np.repeat(np.arange(len(stereo.group_kind)), np.diff(stereo.group_ptr))
    stays = row[stereo.group_member] >= 0 if len(stereo.group_member) else np.zeros(0, dtype=bool)
    sizes = np.bincount(group[stays], minlength=len(stereo.group_kind))
    return StereoColumns(
        stereo.centre[keep],
        stereo.parity[keep],
        stereo.neighbors[keep],
        stereo.centre_state[keep],
        stereo.bond_edge,
        stereo.bond_value,
        stereo.bond_reference,
        stereo.bond_state,
        stereo.group_kind[sizes > 0],
        _pointer(sizes[sizes > 0]),
        row[stereo.group_member[stays]].astype(np.int32),
    )


def apply_edit(value: V, edit: ChemicalEdit) -> tuple[V, AtomMap, EditReport]:
    r"""
    Apply one declared edit: untouched atoms and edges keep every fact; replaced components take their final rows
    (``retained_atoms`` names the old atom of each final atom, ``-1`` new); ``set_atoms`` writes values with the states
    the recipe states (``None`` for ``isotope``, stated exactly when non-zero); a kept atom whose neighbours, bonds,
    element or charge the edit changes has its hydrogen count withdrawn (UNKNOWN×NONE) unless ``set_atoms`` states it
    again; every touched component takes the claims ``claims`` states (required). On a ``Structure`` only
    ``keep_coordinates`` atoms of touched components keep their coordinates, with the evidence, occupancy, su and B
    stated for them; their other atoms and every new atom have none, with occupancy and B INAPPLICABLE×DEFINED, and
    the method and resolution are withdrawn when no OBSERVED coordinate remains. ``identity="withdraw"`` withdraws the
    author numbering of replaced components, ``"inherit"`` keeps it. An inconsistent recipe is
    ``InvalidRequestError(edit_invalid)``.
    """

    system = value if isinstance(value, MolecularSystem) else value.system
    arrays = system.build()
    c, families = arrays.columns, {k: list(v) for k, v in arrays.symbols.items()}
    _check(edit, c)
    n, components = len(c["atom_symbol"]), len(c["component_symbol"])
    atom_component = _owner(c["component_atom_ptr"])
    remove = np.zeros(n, dtype=bool)
    remove[np.asarray(edit.remove_atoms, dtype=np.int64)] = True
    replaced = np.zeros(components, dtype=bool)
    for component, rows in edit.replace_components.items():
        retained = np.asarray(edit.retained_atoms.get(component, np.full(len(rows.names), -1)), dtype=np.int64)
        own = retained[retained >= 0]
        if (atom_component[own] != component).any() or len(np.unique(own)) != len(own):
            _invalid("retained_atoms", component, invariant="own_distinct_atoms")
        replaced[component] = True
    # final atom order: components in order; kept old atoms; a replaced component's final rows
    kept = ~remove & ~replaced[atom_component]
    sources = [(atom_component[kept], np.flatnonzero(kept))]
    for component, rows in sorted(edit.replace_components.items()):
        sources.append((np.full(len(rows.names), component), np.arange(len(rows.names))))
    owner, position = (np.concatenate(parts) for parts in zip(*sources))
    final = np.lexsort((position, owner))
    total = len(final)
    new_index = np.empty(total, dtype=np.int64)
    new_index[final] = np.arange(total)
    atom_map = np.full(n, -1, dtype=np.int64)
    atom_map[np.flatnonzero(kept)] = new_index[: kept.sum()]
    local_index, cursor = {}, kept.sum()
    for component, rows in sorted(edit.replace_components.items()):
        local_index[component] = new_index[cursor : cursor + len(rows.names)]
        retained = np.asarray(edit.retained_atoms.get(component, np.full(len(rows.names), -1)), dtype=np.int64)
        atom_map[retained[retained >= 0]] = local_index[component][retained >= 0]
        cursor += len(rows.names)
    out = {}
    pool = len(c["reference_pool_source"])
    for name in ("atom_symbol", *_ATOM_COLUMNS):
        column = np.zeros(total, dtype=c[name].dtype)
        column[atom_map[kept]] = c[name][kept]
        for component, rows in edit.replace_components.items():
            if name == "atom_symbol":
                column[local_index[component]] = [_code(families["atom"], text) for text in rows.names]
            elif name == "atom_reference_row":
                local = np.asarray(rows.atoms[name], dtype=np.int64)
                column[local_index[component]] = np.where(local >= 0, local + pool, -1)
                pool += len(rows.reference_source)
            else:
                column[local_index[component]] = np.asarray(rows.atoms[name])
        out[name] = column
    for name, (atoms, values, states) in edit.set_atoms.items():
        atoms = np.asarray(atoms, dtype=np.int64)
        if not _survive(atoms, atom_map):
            _invalid("set_atoms", name, invariant="removed_atom")
        out[name][atom_map[atoms]] = values
        if states is not None:
            out[f"{name}_state"][atom_map[atoms]] = states
    touched = replaced.copy()
    touched[atom_component[remove]] = True
    for atoms, _, _ in edit.set_atoms.values():
        touched[atom_component[np.asarray(atoms, dtype=np.int64)]] = True
    # edges
    first, second = (end.astype(np.int64) for end in c["edge_index"])
    removed_edges = np.zeros(len(first), dtype=bool)
    removed_edges[np.asarray(edit.remove_edges, dtype=np.int64)] = True
    touched[atom_component[first[removed_edges]]] = touched[atom_component[second[removed_edges]]] = True
    inside = atom_component[first] == atom_component[second]
    lost = (atom_map[first] < 0) | (atom_map[second] < 0)
    if (lost & ~inside & ~removed_edges).any():
        _invalid(
            "remove_edges", int(np.flatnonzero(lost & ~inside & ~removed_edges)[0]), invariant="connection_dropped"
        )
    renewed = inside & replaced[atom_component[first]]
    keep_edges = ~removed_edges & ~lost & ~renewed
    rows_kept = np.flatnonzero(keep_edges)
    edge_a, edge_b = [atom_map[first[rows_kept]]], [atom_map[second[rows_kept]]]
    columns = {name: [c[name][rows_kept]] for name in _BOND_COLUMNS}
    connection = [c["bond_connection_type"][rows_kept]]
    leaving_ptr, leaving = _ragged(c["bond_declared_leaving_ptr"], rows_kept)
    leaving_rows = [
        (c["bond_declared_leaving"][leaving], c["bond_declared_leaving_side"][leaving], np.diff(leaving_ptr))
    ]
    new_edges = edit.add_edges
    ends = np.asarray(new_edges.atoms, dtype=np.int64).reshape(2, -1)
    if not _survive(ends, atom_map):
        _invalid("add_edges", ends.tolist(), invariant="removed_atom")
    resolved = atom_map[ends]
    edge_a.append(resolved[0])
    edge_b.append(resolved[1])
    for name in _BOND_COLUMNS:
        columns[name].append(np.asarray(new_edges.columns[name]))
    connection.append(np.array([_code(families["connection_type"], t) for t in new_edges.connection_type], np.int64))
    names = [_code(families["atom"], name) for row in new_edges.leaving for name, _ in row]
    sides = [side for row in new_edges.leaving for _, side in row]
    leaving_rows.append(
        (np.array(names, np.int64), np.array(sides, np.int64), np.array([len(r) for r in new_edges.leaving], np.int64))
    )
    touched[atom_component[ends]] = True
    for component, rows in sorted(edit.replace_components.items()):
        local = np.asarray(rows.edges, dtype=np.int64).reshape(2, -1)
        edge_a.append(local_index[component][local[0]])
        edge_b.append(local_index[component][local[1]])
        for name in _BOND_COLUMNS:
            columns[name].append(np.asarray(rows.bonds[name]))
        connection.append(np.full(local.shape[1], -1, np.int64))
        leaving_rows.append((np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(local.shape[1], np.int64)))
    a, b = np.concatenate(edge_a), np.concatenate(edge_b)
    flip = a > b
    a, b = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((b, a))
    c["edge_index"] = np.stack((a[order], b[order]))
    for name in _BOND_COLUMNS:
        values = np.concatenate(columns[name])
        c[name] = (np.where(flip, -values, values) if name == "bond_direction" else values)[order]
    c["bond_connection_type"] = np.concatenate(connection)[order]
    counts = np.concatenate([row[2] for row in leaving_rows])
    leaving_names = np.concatenate([row[0] for row in leaving_rows])
    leaving_sides = np.concatenate([row[1] for row in leaving_rows])
    leaving_sides = np.where(np.repeat(flip, counts), 1 - leaving_sides, leaving_sides)
    c["bond_declared_leaving_ptr"], leaving_order = _ragged(_pointer(counts), order)
    c["bond_declared_leaving"], c["bond_declared_leaving_side"] = (
        leaving_names[leaving_order],
        leaving_sides[leaving_order],
    )
    edge_map = np.full(len(first), -1, dtype=np.int64)
    position_of = np.empty(len(order), dtype=np.int64)
    position_of[order] = np.arange(len(order))
    edge_map[rows_kept] = position_of[: len(rows_kept)]
    # hydrogen counts: a count the edit's change rests on is withdrawn unless the recipe restates it
    changed = np.zeros(n, dtype=bool)
    changed[first[removed_edges | lost]] = changed[second[removed_edges | lost]] = True
    changed[ends] = True
    for name in ("atomic_number", "formal_charge"):
        if name in edit.set_atoms:
            changed[np.asarray(edit.set_atoms[name][0], dtype=np.int64)] = True
    if "implicit_hydrogens" in edit.set_atoms:
        changed[np.asarray(edit.set_atoms["implicit_hydrogens"][0], dtype=np.int64)] = False
    stale = atom_map[changed & kept]
    out["implicit_hydrogens"][stale], out["implicit_hydrogens_state"][stale] = 0, _UNKNOWN_NONE
    # externals
    if (atom_map[c["external_atom"]] < 0).any():
        _invalid("remove_atoms", "an atom with an external connection", invariant="connection_dropped")
    c["external_atom"] = atom_map[c["external_atom"]]
    # stereo: rows of replaced centres go (their component brings its own); the rest is remapped
    stereo = _drop_centres(StereoColumns.read(c), replaced[atom_component])
    centres = set(stereo.centre[known(stereo.centre_state)].tolist())
    new_neighbours = zip(
        np.concatenate((ends[0], ends[1])).tolist(), np.concatenate((resolved[1], resolved[0])).tolist()
    )
    substitutions = np.array(  # a new neighbour of a centre takes no vacated slot (-2)
        [(old, -2, new) for old, new in new_neighbours if old in centres], dtype=np.int64
    ).reshape(-1, 3)
    remapped, clears = remap_parity(
        stereo,
        atom_map,
        removed_hydrogen=np.zeros(n, dtype=bool),
        substitutions=substitutions,
        externalised=np.full(n, -1, dtype=np.int64),
        edge_map=edge_map,
        external_map=None,
        edges=np.stack((first, second)),
    )
    parts = [remapped]
    first_edge = len(rows_kept) + ends.shape[1]  # a replaced component's edges follow the kept and added ones
    for component, rows in sorted(edit.replace_components.items()):
        local_stereo, mapping = rows.stereo, local_index[component]
        bond_edge = position_of[first_edge + np.asarray(local_stereo.bond_edge, dtype=np.int64)]
        first_edge += np.asarray(rows.edges).reshape(2, -1).shape[1]
        slots = np.asarray(local_stereo.neighbors, dtype=np.int64)
        references = np.asarray(local_stereo.bond_reference, dtype=np.int64)
        parts.append(
            StereoColumns(
                mapping[np.asarray(local_stereo.centre, dtype=np.int64)],
                local_stereo.parity,
                np.where(slots >= 0, mapping[np.maximum(slots, 0)], slots),
                local_stereo.centre_state,
                bond_edge,
                local_stereo.bond_value,
                np.where(references >= 0, mapping[np.maximum(references, 0)], references),
                local_stereo.bond_state,
                local_stereo.group_kind,
                local_stereo.group_ptr,
                local_stereo.group_member,
            )
        )
    offsets = np.cumsum([0] + [len(p.centre) for p in parts])
    group_offsets = np.cumsum([0] + [p.group_ptr[-1] for p in parts])
    c["stereocentre_atom"] = np.concatenate([p.centre for p in parts])
    c["stereocentre_parity"] = np.concatenate([p.parity for p in parts])
    c["stereocentre_neighbors"] = np.concatenate([np.asarray(p.neighbors).reshape(-1, 4) for p in parts])
    c["stereocentre_state"] = np.concatenate([p.centre_state for p in parts])
    c["stereobond_edge"] = np.concatenate([p.bond_edge for p in parts])
    c["stereobond_value"] = np.concatenate([p.bond_value for p in parts])
    c["stereobond_reference"] = np.concatenate([np.asarray(p.bond_reference).reshape(-1, 2) for p in parts])
    c["stereobond_state"] = np.concatenate([p.bond_state for p in parts])
    c["stereogroup_kind"] = np.concatenate([p.group_kind for p in parts])
    c["stereogroup_ptr"] = np.concatenate([[0]] + [p.group_ptr[1:] + g for p, g in zip(parts, group_offsets)])
    c["stereogroup_member"] = np.concatenate([np.asarray(p.group_member) + o for p, o in zip(parts, offsets)])
    # atoms, components, pool
    for name, column in out.items():
        c[name] = column
    c["component_atom_ptr"] = _pointer(np.bincount(owner, minlength=components))
    c["component_port"] = np.append(atom_map, -1)[c["component_port"].astype(np.int64)]
    parent_rows = {
        name: [c[name][c[f"{name}_ptr"][k] : c[f"{name}_ptr"][k + 1]] for k in range(components)]
        for name in ("component_parent", "component_definition_parent")
    }
    for component, rows in edit.replace_components.items():
        c["component_symbol"][component] = _code(families["component"], rows.symbol)
        c["component_monomer_type"][component], c["component_monomer_type_state"][component] = (
            rows.monomer_type,
            rows.monomer_type_state,
        )
        for name, parents in (
            ("component_parent", rows.parents),
            ("component_definition_parent", rows.definition_parents),
        ):
            parent_rows[name][component] = np.array([_code(families["parent"], p) for p in parents], dtype=np.int64)
        c["component_definition_scope"][component] = rows.definition_scope
        c["component_definition_digest"][component] = rows.definition_digest
        c["component_port"][component] = np.append(local_index[component], -1)[np.asarray(rows.port, dtype=np.int64)]
        c["reference_pool_coordinates"] = np.concatenate((c["reference_pool_coordinates"], rows.reference))
        c["reference_pool_source"] = np.concatenate((c["reference_pool_source"], rows.reference_source))
    for name, segments in parent_rows.items():
        c[f"{name}_ptr"] = _pointer([len(row) for row in segments])
        c[name] = np.concatenate(segments).astype(np.int64) if segments else np.zeros(0, np.int64)
    claims = dict(edit.claims)
    for component, rows in edit.replace_components.items():
        claim = cast(tuple[bool, bool], tuple(rows.claims))  # _check checked the two claim values.
        if claims.setdefault(component, claim) != claim:
            _invalid("claims", component, invariant="differs_from_component_rows")
    missing = sorted(set(np.flatnonzero(touched).tolist()) - set(claims))
    if missing:
        _invalid("claims", missing, invariant="required_for_touched_components")
    for component, (topology, heavy) in claims.items():
        c["component_topology_complete"][component], c["component_heavy_atoms_complete"][component] = topology, heavy
    edited = MolecularSystem.from_arrays(SystemArrays({k: tuple(v) for k, v in families.items()}, c))
    if isinstance(value, MolecularSystem):
        nothing = np.zeros(0, np.int64)
        return edited, AtomMap(atom_map), EditReport(clears, claims, nothing, nothing)
    structure, invalidated, changed = _edit_structure(value, edited, edit, atom_map, atom_component, touched, replaced)
    return Structure.from_arrays(edited, structure), AtomMap(atom_map), EditReport(clears, claims, invalidated, changed)


def _edit_structure(value: Structure, edited, edit: ChemicalEdit, atom_map, atom_component, touched, replaced):
    r"""The structure columns of an edit: of touched components only ``keep_coordinates`` atoms keep coordinates."""
    arrays = value.build()
    s, n, total = arrays.columns, len(atom_map), len(edited.atom_symbol)
    given = np.zeros(n, dtype=bool) if edit.keep_coordinates is None else np.asarray(edit.keep_coordinates)
    keep = given | ~touched[atom_component]
    survivor = atom_map >= 0
    out = {}
    for f in Structure.SCHEMA.fields:
        if f.axis != "atom":
            continue
        column = np.zeros((total, *s[f.name].shape[1:]), dtype=s[f.name].dtype)
        column[atom_map[survivor]] = s[f.name][survivor]
        out[f.name] = column
    placed = np.zeros(total, dtype=bool)
    placed[atom_map[survivor & keep]] = True
    lost = (out["coordinate_evidence"] != CoordinateEvidence.NONE) & ~placed
    invalidated = np.flatnonzero(lost).astype(np.int64)
    new = np.ones(total, dtype=bool)
    new[atom_map[survivor]] = False
    for name, empty in (
        ("coordinates", 0),
        ("coordinate_evidence", CoordinateEvidence.NONE),
        ("coordinate_quantum", 0),
        ("occupancy", 0),
        ("occupancy_uncertainty", 0),
        ("b_factor", 0),
    ):
        out[name][~placed] = empty
    for name in ("occupancy_state", "occupancy_uncertainty_state", "b_factor_state"):
        out[name][~placed] = _INAPPLICABLE_DEFINED
    out["atom_altloc"][new] = -1
    out["atom_site_charge"][new], out["atom_site_charge_state"][new] = 0, _UNKNOWN_NONE
    s.update(out)
    atom_system = _owner(edited.system_instance_ptr)[_owner(edited.instance_component_ptr)]
    atom_system = atom_system[_owner(edited.component_atom_ptr)]
    observed = np.bincount(
        atom_system[s["coordinate_evidence"] == CoordinateEvidence.OBSERVED], minlength=len(s["system_model_number"])
    )
    for pointer, payload in (
        ("system_method_ptr", ("method",)),
        ("system_resolution_ptr", ("resolution", "resolution_kind", "resolution_state")),
    ):  # withdrawn per system where no OBSERVED coordinate remains
        rows = np.repeat(observed > 0, np.diff(s[pointer]))
        for name in payload:
            s[name] = s[name][rows]
        s[pointer] = _pointer(np.where(observed > 0, np.diff(s[pointer]), 0))
    changed = np.flatnonzero(replaced).astype(np.int64)
    if edit.identity == "withdraw":
        s["component_author_text"][changed], s["component_insertion"][changed] = -1, -1
        s["component_author_number"][changed], s["component_author_number_state"][changed] = 0, _UNKNOWN_NONE
    return StructureArrays(arrays.symbols, s), invalidated, changed
