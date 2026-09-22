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
Parity over ordered neighbour slots: the one remap rule every graph change calls, and parity and
cis/trans read from coordinates with a bound propagated from the printed quantum.

A stereocentre stores a tetrahedral parity over four ordered neighbour slots (OpenSMILES ordering): 1 is ``@``, 2 is
``@@``, 0 undetermined. A slot holds an atom index, ``-1`` for the one implicit hydrogen or lone pair, or ``-(x + 2)``
for external connection row ``x``. Slot positions never move, so a remap never changes the permutation parity.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from ..errors import CODE_NAMES, RequestRef, fail
from ..provenance import Basis, Resolution, known, state
from .schema import check_rows


class StereoGroupKind(IntEnum):
    r"""CXSMILES enhanced stereo groups; a bare ``r`` is one ``AND`` group."""

    ABS = 0
    AND = 1
    OR = 2


@dataclass(frozen=True)
class StereoColumns:
    r"""The stereo axes of a system as an operation value; validated when lowered into ``SystemArrays`` (S13)."""

    centre: NDArray[np.int32]
    parity: NDArray[np.int8]
    neighbors: NDArray[np.int32]
    centre_state: NDArray[np.uint8]
    bond_edge: NDArray[np.int32]
    bond_value: NDArray[np.int8]
    bond_reference: NDArray[np.int32]
    bond_state: NDArray[np.uint8]
    group_kind: NDArray[np.uint8]
    group_ptr: NDArray[np.int64]
    group_member: NDArray[np.int32]

    @classmethod
    def read(cls, columns: Mapping[str, NDArray]) -> StereoColumns:
        r"""The stereo rows of a ``SystemArrays`` column buffer."""
        return cls(**{key: columns[name] for key, name in _COLUMNS.items()})

    def write(self, columns: MutableMapping[str, NDArray]) -> None:
        r"""Store these rows into a ``SystemArrays`` column buffer."""
        for key, name in _COLUMNS.items():
            columns[name] = getattr(self, key)


#: Each ``StereoColumns`` field and the ``MolecularSystem`` column holding it.
_COLUMNS = {
    "centre": "stereocentre_atom",
    "parity": "stereocentre_parity",
    "neighbors": "stereocentre_neighbors",
    "centre_state": "stereocentre_state",
    "bond_edge": "stereobond_edge",
    "bond_value": "stereobond_value",
    "bond_reference": "stereobond_reference",
    "bond_state": "stereobond_state",
    "group_kind": "stereogroup_kind",
    "group_ptr": "stereogroup_ptr",
    "group_member": "stereogroup_member",
}


@dataclass(frozen=True)
class ParityClears:
    r"""Cleared stereocentre and stereobond rows of the remapped value and the ``CODE_NAMES`` index of
    each reason."""

    centre: NDArray[np.int64]
    reason: NDArray[np.uint16]
    bond: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    bond_reason: NDArray[np.uint16] = field(default_factory=lambda: np.zeros(0, dtype=np.uint16))


_HEAVY, _TWO, _NEW = (
    CODE_NAMES.index(f"parity_cleared_{name}")
    for name in ("heavy_neighbour_removed", "two_virtual_slots", "new_neighbour_without_vacated_slot")
)
_CLEARED = state(Resolution.UNKNOWN, Basis.DERIVED)
_TRANSPORTED = state(Resolution.KNOWN, Basis.DERIVED)


def permutation_parity(order: NDArray[np.int64]) -> NDArray[np.int8]:
    r"""
    The parity of each row's ordering: its inversion count modulo 2 (0 even, 1 odd).

    Values must be distinct per row (``-1`` is an ordinary key).

    Examples:
        >>> permutation_parity(np.array([[0, 1, 2, 3], [1, 0, 2, 3], [-1, 2, 1, 0]])).tolist()
        [0, 1, 1]
    """

    order = np.asarray(order, dtype=np.int64)
    pairs = order[..., :, None] - order[..., None, :]
    upper = np.triu(np.ones((order.shape[-1],) * 2, dtype=bool), k=1)
    return ((pairs[..., upper] > 0).sum(axis=-1) % 2).astype(np.int8)


def _points(coordinates: NDArray[np.float64], quantum: NDArray[np.float64]) -> int:
    r"""The atom count of ``coordinates`` ``[n, 3]`` with one ``quantum`` per atom (``value_invalid`` otherwise)."""
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or quantum.shape != coordinates.shape[:1]:
        fail("value_invalid", RequestRef("coordinates", f"{coordinates.shape} {quantum.shape}"), expected="[n, 3], [n]")
    return len(coordinates)


def parity_from_coordinates(
    coordinates: NDArray[np.float64],
    centre: NDArray[np.int64],
    neighbors: NDArray[np.int64],
    quantum: NDArray[np.float64],
) -> NDArray[np.int8]:
    r"""
    The parity of each centre read off its neighbours' positions: 1 (``@``), 2 (``@@``) or 0 (undetermined).

    Looking from the first slot towards the centre, ``@`` lists the other three anticlockwise. The implicit slot
    (``-1``, in any position) is placed at the centre. The signed volume ``V = a·(b×c)`` of the slot vectors decides
    only where ``|V|`` exceeds its bound under per-axis rounding of ``quantum / 2`` at every point (first-order term
    plus the closed-form higher-order terms), so no threshold is a plausibility pick. A row with an external slot or
    more than one implicit slot is 0.

    Examples:
        >>> points = np.array([[0, 0, 0], [0, 0, 1], [1, 0, -0.33], [-0.5, 0.87, -0.33], [-0.5, -0.87, -0.33]])
        >>> parity_from_coordinates(points, np.array([0]), np.array([[1, 2, 3, 4]]), np.full(5, 0.001)).tolist()
        [1]
    """

    coordinates, quantum = np.asarray(coordinates, dtype=np.float64), np.asarray(quantum, dtype=np.float64)
    centre, slots = np.asarray(centre), np.asarray(neighbors)
    if not len(centre):
        return np.zeros(0, dtype=np.int8)
    n = _points(coordinates, quantum)
    centre = check_rows("centre", centre, n, shape=(None,))
    slots = check_rows("neighbors", slots, n, low=np.iinfo(np.int64).min, shape=(len(centre), 4))  # ≤ -1: a slot code
    usable = ((slots >= -1).all(axis=1)) & ((slots == -1).sum(axis=1) <= 1)
    atoms = np.where(slots == -1, centre[:, None], np.maximum(slots, 0))
    points, error = coordinates[atoms], quantum[atoms] / 2
    a, b, c = (points[:, k] - points[:, 0] for k in (1, 2, 3))
    g1, g2, g3 = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    g0 = -(g1 + g2 + g3)
    volume = np.einsum("ij,ij->i", a, g1)
    bound = sum(error[:, k] * np.abs(g).sum(axis=1) for k, g in enumerate((g0, g1, g2, g3)))
    ea, eb, ec = (error[:, k] + error[:, 0] for k in (1, 2, 3))
    norm = lambda v: np.linalg.norm(v, axis=1)  # noqa: E731
    bound = bound + 3 * (ea * eb * norm(c) + ea * ec * norm(b) + eb * ec * norm(a)) + 3 * np.sqrt(3) * ea * eb * ec
    result = np.where(volume < -bound, 1, np.where(volume > bound, 2, 0))
    return np.where(usable, result, 0).astype(np.int8)


def bond_configuration(
    coordinates: NDArray[np.float64],
    bond: NDArray[np.int64],
    reference: NDArray[np.int64],
    quantum: NDArray[np.float64],
) -> NDArray[np.int8]:
    r"""
    The configuration of each double bond ``bond[k] = (b1, b2)`` relative to its references ``(r1, r2)``: 1 cis, 2
    trans, 0 undetermined.

    With ``t = b2 − b1``, ``u1 = r1 − b1`` and ``u2 = r2 − b2``, ``D = (u1×t)·(u2×t)`` decides only where ``|D|``
    exceeds its bound under per-axis rounding of ``quantum / 2`` at the four points (a sound closed form).

    Examples:
        >>> points = np.array([[0, 0, 0], [1.3, 0, 0], [-0.7, 1.2, 0], [2.0, 1.2, 0], [2.0, -1.2, 0]])
        >>> quantum = np.full(5, 0.001)
        >>> bond_configuration(points, np.array([[0, 1], [0, 1]]), np.array([[2, 3], [2, 4]]), quantum).tolist()
        [1, 2]
    """

    coordinates, quantum = np.asarray(coordinates, dtype=np.float64), np.asarray(quantum, dtype=np.float64)
    bond, reference = np.asarray(bond), np.asarray(reference)
    if not len(bond):
        return np.zeros(0, dtype=np.int8)
    n = _points(coordinates, quantum)
    bond = check_rows("bond", bond, n, shape=(None, 2))
    reference = check_rows("reference", reference, n, shape=(len(bond), 2))
    e = quantum / 2
    b1, b2, r1, r2 = (bond[:, 0], bond[:, 1], reference[:, 0], reference[:, 1])
    t = coordinates[b2] - coordinates[b1]
    p, q = np.cross(coordinates[r1] - coordinates[b1], t), np.cross(coordinates[r2] - coordinates[b2], t)
    value = np.einsum("ij,ij->i", p, q)
    root3, norm = np.sqrt(3), lambda v: np.linalg.norm(v, axis=1)  # noqa: E731
    et, eu1, eu2 = e[b1] + e[b2], e[r1] + e[b1], e[r2] + e[b2]
    u1, u2 = coordinates[r1] - coordinates[b1], coordinates[r2] - coordinates[b2]
    dp = root3 * eu1 * norm(t) + root3 * et * norm(u1) + 3 * eu1 * et
    dq = root3 * eu2 * norm(t) + root3 * et * norm(u2) + 3 * eu2 * et
    bound = norm(p) * dq + dp * norm(q) + dp * dq
    return np.where(value > bound, 1, np.where(value < -bound, 2, 0)).astype(np.int8)


def _lookup(substitutions: NDArray[np.int64], n_old: int):
    r"""Sorted (centre, departed) keys of the substitution rows with departed ≥ -1, and their partners."""
    rows = substitutions[substitutions[:, 1] >= -1] if len(substitutions) else substitutions.reshape(0, 3)
    keys = rows[:, 0] * (n_old + 2) + rows[:, 1] + 2
    order = np.argsort(keys, kind="stable")
    return keys[order], rows[order, 2]


def _find(keys, partners, query):
    position = np.clip(np.searchsorted(keys, query), 0, max(len(keys) - 1, 0))
    hit = (keys[position] == query) if len(keys) else np.zeros(query.shape, dtype=bool)
    return hit, np.where(hit, partners[position] if len(keys) else 0, 0)


def remap_parity(
    stereo: StereoColumns,
    atom_map: NDArray[np.int64],
    *,
    removed_hydrogen: NDArray[np.bool_],
    substitutions: NDArray[np.int64],
    externalised: NDArray[np.int64],
    edge_map: NDArray[np.int64] | None = None,
    external_map: NDArray[np.int64] | None = None,
    edges: NDArray[np.int64] | None = None,
) -> tuple[StereoColumns, ParityClears]:
    r"""
    Carry stereo rows across a graph change: the one remap rule of projections, edits and the lowering.

    Each slot of a KNOWN centre resolves by the first rule that applies: (a) a kept atom → its new index; a
    ``substitutions`` row ``(centre, departed, partner)`` for this slot (the implicit slot ``-1`` included) → the
    partner, the row transported (KNOWN×DERIVED; a partner ``≤ -2`` is external row ``-(partner + 2)``); (b) an atom in
    ``externalised`` → that external row, parity and state kept; (c) a hydrogen that becomes implicit
    (``removed_hydrogen``) → ``-1``, cleared ``two_virtual_slots`` when the row then has two; (d) any other removed
    atom → cleared ``heavy_neighbour_removed``; an external slot maps through ``external_map`` (``-1`` = removed →
    cleared). (e) A substitution whose departed is ``-2`` or no slot of its centre → cleared
    ``new_neighbour_without_vacated_slot``. A slot atom whose edge to its centre the change drops while both atoms
    stay (``edges`` with ``edge_map``) departs like a removed atom. A cleared row keeps its row with parity 0,
    UNKNOWN×DERIVED and neighbours ``-1``; a removed centre drops its row and its group membership. ``edge_map``
    drops or renumbers bond-stereo rows, whose references map like slots; a reference removed without replacement
    clears its row (``heavy_neighbour_removed``, value 0, UNKNOWN×DERIVED, references ``-1``).

    Args:
        stereo: The rows before the change.
        atom_map: Old atom → new atom, ``-1`` removed.
        removed_hydrogen: Per old atom: it leaves the graph and becomes an implicit hydrogen of its neighbour.
        substitutions: ``[s, 3]`` rows ``(centre, departed, partner)``: old centre, old departed atom (``-1`` implicit,
            ``-2`` no vacated slot), new partner.
        externalised: Per old atom: the new external row it became the partner of, or ``-1``.
        edge_map: Old edge → new edge, ``-1`` removed; ``None`` keeps every bond row.
        external_map: Old external row → new slot code (an atom when re-joined, ``-(x + 2)``, or ``-1`` removed).
        edges: The old edge table ``[2, m]``; with ``edge_map`` it names the edges dropped between kept atoms.

    Returns:
        The remapped rows and the rows cleared, by new row index and reason.
    """

    atom_map = np.asarray(atom_map, dtype=np.int64)
    removed_hydrogen = np.asarray(removed_hydrogen, dtype=bool)
    externalised = np.asarray(externalised, dtype=np.int64)
    substitutions = np.asarray(substitutions, dtype=np.int64).reshape(-1, 3)
    n_old = len(atom_map)
    keys, partners = _lookup(substitutions, n_old)
    edges = np.zeros((2, 0), dtype=np.int64) if edges is None else np.asarray(edges, dtype=np.int64)
    dropped = np.zeros(edges.shape[1], dtype=bool)
    if edge_map is not None and edges.shape[1]:
        dropped = (np.asarray(edge_map) < 0) & (atom_map[edges[0]] >= 0) & (atom_map[edges[1]] >= 0)
    cut = np.sort(edges[0, dropped] * n_old + edges[1, dropped])  # edges are stored low → high

    def slot_codes(slots: NDArray[np.int64], owner: NDArray[np.int64], adjacent: NDArray[np.int64]):
        r"""New slot codes and the kept, transported, heavy-removal and folded masks of one slot array."""
        atom = slots >= 0
        safe = np.clip(slots, 0, max(n_old - 1, 0))
        departs, _ = _find(cut, cut, np.minimum(slots, adjacent) * n_old + np.maximum(slots, adjacent))
        kept = atom & (atom_map[safe] >= 0) & ~(departs & (adjacent >= 0))
        hit, partner = _find(keys, partners, owner * (n_old + 2) + slots + 2)
        hit &= ~kept & (slots >= -1)
        out = np.where(kept, atom_map[safe], -1)
        out = np.where(hit, partner, out)
        removed = atom & ~kept & ~hit
        external = removed & (externalised[safe] >= 0)
        out = np.where(external, -(externalised[safe] + 2), out)
        folded = removed & ~external & removed_hydrogen[safe]
        heavy = removed & ~external & ~folded
        if external_map is None:
            mapped = slots
        else:
            table = np.asarray(external_map, dtype=np.int64)
            mapped = table[np.clip(-slots - 2, 0, len(table) - 1)] if len(table) else np.full(slots.shape, -1)
        out = np.where(slots <= -2, mapped, out)
        heavy |= (slots <= -2) & (mapped == -1)
        return out, kept, hit, heavy, folded

    centre = np.asarray(stereo.centre, dtype=np.int64)
    keep = atom_map[centre] >= 0
    row_map = np.where(keep, np.cumsum(keep) - 1, -1)
    owner = np.repeat(centre, 4).reshape(-1, 4)
    slots = np.asarray(stereo.neighbors, dtype=np.int64)
    new_slots, _, transported, heavy, _ = slot_codes(slots, owner, owner)
    is_known = known(stereo.centre_state)
    two = (new_slots == -1).sum(axis=1) > 1
    vacated = np.ones(len(centre), dtype=bool)
    if len(substitutions) and len(centre):
        sub_centre, departed = substitutions[:, 0], substitutions[:, 1]
        order = np.argsort(centre, kind="stable")
        rows = order[np.clip(np.searchsorted(centre[order], sub_centre), 0, len(centre) - 1)]
        present = centre[rows] == sub_centre
        missing = present & ((departed == -2) | ~(slots[rows] == departed[:, None]).any(axis=1))
        vacated[rows[missing]] = False
    reason = np.where(heavy.any(axis=1), _HEAVY, np.where(two, _TWO, np.where(~vacated, _NEW, 0)))
    cleared = is_known & keep & (reason > 0)
    carried = is_known & keep & ~cleared
    neighbors = np.where(is_known[:, None], np.where(carried[:, None], new_slots, -1), slots)
    parity = np.where(cleared, 0, stereo.parity)
    states = np.where(cleared, _CLEARED, np.where(carried & transported.any(axis=1), _TRANSPORTED, stereo.centre_state))
    # groups: members follow their rows; cleared and dropped rows leave; emptied groups are dropped
    member = np.asarray(stereo.group_member, dtype=np.int64)
    group = np.repeat(np.arange(len(stereo.group_kind)), np.diff(stereo.group_ptr))
    stays = (row_map[member] >= 0) & ~cleared[member] if len(member) else np.zeros(0, dtype=bool)
    sizes = np.bincount(group[stays], minlength=len(stereo.group_kind))
    groups = sizes > 0
    # bond rows
    edge = np.asarray(stereo.bond_edge, dtype=np.int64)
    new_edge = edge if edge_map is None else np.asarray(edge_map, dtype=np.int64)[edge]
    bond_keep = new_edge >= 0
    references = np.asarray(stereo.bond_reference, dtype=np.int64)
    ends = edges[:, edge].T if edges.shape[1] else np.full(references.shape, -1)  # reference k is adjacent to end k
    ref_codes, ref_kept, ref_hit, ref_heavy, ref_folded = slot_codes(references, np.full(references.shape, -3), ends)
    if len(references) and len(substitutions):
        by_departed = substitutions[substitutions[:, 1] >= 0]
        order = np.argsort(by_departed[:, 1], kind="stable")
        dk, dp = by_departed[order, 1], by_departed[order, 2]
        hit, partner = _find(dk, dp, references)
        hit &= (references >= 0) & ~ref_kept
        ref_codes, ref_hit = np.where(hit, partner, ref_codes), ref_hit | hit
        ref_heavy, ref_folded = ref_heavy & ~hit, ref_folded & ~hit
    bond_known = known(stereo.bond_state)
    bond_cleared = bond_keep & bond_known & (ref_heavy | ref_folded).any(axis=1)
    bond_carried = bond_keep & bond_known & ~bond_cleared
    bond_refs = np.where(bond_carried[:, None], ref_codes, -1)
    bond_refs = np.where((~bond_known)[:, None], references, bond_refs)
    bond_states = np.where(
        bond_cleared, _CLEARED, np.where(bond_carried & ref_hit.any(axis=1), _TRANSPORTED, stereo.bond_state)
    )
    out = StereoColumns(
        centre=atom_map[centre[keep]].astype(np.int32),
        parity=parity[keep].astype(np.int8),
        neighbors=neighbors[keep].astype(np.int32).reshape(-1, 4),
        centre_state=states[keep].astype(np.uint8),
        bond_edge=new_edge[bond_keep].astype(np.int32),
        bond_value=np.where(bond_cleared, 0, stereo.bond_value)[bond_keep].astype(np.int8),
        bond_reference=bond_refs[bond_keep].astype(np.int32).reshape(-1, 2),
        bond_state=bond_states[bond_keep].astype(np.uint8),
        group_kind=np.asarray(stereo.group_kind)[groups].astype(np.uint8),
        group_ptr=np.concatenate(([0], np.cumsum(sizes[groups]))).astype(np.int64),
        group_member=row_map[member[stays]].astype(np.int32),
    )
    bond_row = np.cumsum(bond_keep) - 1
    return out, ParityClears(
        row_map[cleared].astype(np.int64),
        reason[cleared].astype(np.uint16),
        bond_row[bond_cleared].astype(np.int64),
        np.full(int(bond_cleared.sum()), _HEAVY, dtype=np.uint16),
    )
