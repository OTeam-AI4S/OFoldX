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
Declarative per-family layouts and their one evaluation.

A family states its layout as data — which atoms it addresses, which components it tokenizes per residue, its centre,
frame and representative atom names — and these functions evaluate it over a ``MolecularSystem`` in one pass. The
model input token of every family is the canonical class (72 classes, specials included); a family never re-derives a
token, a centre, a frame or a slot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import EntityType, MolecularSystem, MonomerType
from ..data.tokenization import CLASS_MONOMER, Encoding, TokenLayout, token_classes, tokenize
from ..provenance import known
from .preparation import SlotMap


@dataclass(frozen=True)
class LayoutTable:
    r"""
    One family's layout as data.

    ``residue_slots`` (per component id) and ``monomer_slots`` (per canonical monomer type of the component's class)
    name the atoms a component is addressed by, in slot order; a component named by neither table is not addressed
    when either table is given, and every atom is addressed when both are ``None``. ``hydrogens="exclude"`` leaves
    atoms with atomic number 1 unaddressed; ``exclude_designated_leaving`` leaves the designated port fragments
    (OXT/HXT, OP3/HOP3) unaddressed even when retained. ``per_atom_components`` says which polymer components are one
    token per atom (``"nonstandard"``: those whose class encoding is not ``EXACT``); components of no polymer class
    always are.
    ``representative`` maps a canonical class to its representative (distogram) atom, ``representative_default`` a
    monomer type to the fallback name.
    """

    family: str
    monomer_types: Mapping[MonomerType, int]
    residue_slots: Mapping[str, tuple[str | None, ...]] | None
    monomer_slots: Mapping[MonomerType, tuple[str | None, ...]] | None
    hydrogens: Literal["exclude", "include"]
    per_atom_components: Literal["nonstandard", "none", "all"]
    exclude_designated_leaving: bool
    centre_atom: Mapping[MonomerType, str]
    frame_atoms: Mapping[MonomerType, tuple[str, str, str]]
    representative: Mapping[int, str]
    representative_default: Mapping[MonomerType, str]
    exclude_water: bool = False


@dataclass(frozen=True)
class EncodingLoss:
    r"""The positions (components) whose canonical class does not encode their identity exactly."""

    component: NDArray[np.int64]
    canonical: NDArray[np.uint8]
    encoding: NDArray[np.uint8]


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _slot_names(system: MolecularSystem, table: LayoutTable) -> list[tuple[str | None, ...] | None]:
    r"""Per component, the names it is addressed by in slot order, or ``None`` for every atom."""
    if table.residue_slots is None and table.monomer_slots is None:
        return [None] * len(system.component_symbol)
    monomer = CLASS_MONOMER[token_classes(system).token]
    ids = system.text("component", system.component_symbol).tolist()
    residue_slots, monomer_slots = table.residue_slots or {}, table.monomer_slots or {}
    return [
        residue_slots[name] if name in residue_slots else monomer_slots.get(MonomerType(kind), ())
        for name, kind in zip(ids, monomer.tolist())
    ]


def token_layout(
    system: MolecularSystem, table: LayoutTable, *, atom_mask: NDArray[np.bool_] | None = None
) -> TokenLayout:
    r"""
    The family's token axis over the atoms it addresses (and ``atom_mask``, default every atom).

    A polymer component the table tokenizes per residue is one token centred on its ``centre_atom`` (per atom when
    that atom is not addressed); every other addressed atom is its own token. Components with no addressed atom have
    no token.
    """

    s = system.arrays()
    count = len(s["component_symbol"])
    atom_component = _owner(s["component_atom_ptr"])
    classes = token_classes(system)
    monomer = CLASS_MONOMER[classes.token]
    names = system.text("atom", s["atom_symbol"])
    visible = np.ones(len(names), dtype=bool) if atom_mask is None else np.array(atom_mask, dtype=bool)
    if table.hydrogens == "exclude":
        visible &= s["atomic_number"] != 1
    if table.exclude_designated_leaving:
        visible &= s["atom_designation"] == 0
    if table.exclude_water:
        entity = s["instance_entity"][_owner(s["instance_component_ptr"])]
        water = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.WATER)
        visible &= ~water[atom_component]
    slot_names = _slot_names(system, table)
    if any(entry is not None for entry in slot_names):
        allowed = [frozenset(entry) if entry is not None else None for entry in slot_names]
        named = np.array(
            [allowed[c] is None or name in allowed[c] for c, name in zip(atom_component.tolist(), names.tolist())],
            dtype=bool,
        )
        visible &= named
    per_atom = {
        "nonstandard": classes.encoding != Encoding.EXACT,
        "none": np.zeros(count, dtype=bool),
        "all": np.ones(count, dtype=bool),
    }[table.per_atom_components]
    wanted = np.array([table.centre_atom.get(MonomerType(kind), "") for kind in monomer.tolist()] + [""])
    hit = visible & (names == wanted[np.where(monomer[atom_component] > 0, atom_component, count)])
    centre = np.full(count, -1, dtype=np.int64)
    first = np.flatnonzero(hit)[::-1]
    centre[atom_component[first]] = first
    residue = (monomer > 0) & ~per_atom & (centre >= 0)
    return tokenize(system, residue_token=residue, centre_atom=centre, atom_mask=visible)


def slot_map(system: MolecularSystem, layout: TokenLayout, table: LayoutTable) -> SlotMap:
    r"""
    The slot axis the family's tensors are indexed by.

    A residue token of a component the table names gets the table's names in order, each bound to its atom (``-1``
    when the token does not hold it); every other token gets its atoms in canonical order. Target atoms with no slot
    are ``unaddressed``.
    """

    slot_names = _slot_names(system, table)
    owner = _owner(layout.token_atom_ptr)
    if all(entry is None for entry in slot_names):
        atom, token = layout.token_atoms.copy(), owner
    else:
        names = system.text("atom", system.atom_symbol)
        counts = np.bincount(layout.token_component, minlength=len(slot_names))
        atoms, tokens = [], []
        for t, component in enumerate(layout.token_component.tolist()):
            held = layout.token_atoms[layout.token_atom_ptr[t] : layout.token_atom_ptr[t + 1]].tolist()
            entry = slot_names[component]
            if entry is not None and counts[component] == 1:
                by_name = {str(names[a]): a for a in held}
                held = [by_name.get(name, -1) for name in entry]
            atoms += held
            tokens += [t] * len(held)
        atom, token = np.array(atoms, dtype=np.int64), np.array(tokens, dtype=np.int64)
    addressed = np.zeros(len(system.atom_symbol), dtype=bool)
    addressed[atom[atom >= 0]] = True
    return SlotMap(atom=atom, token=token, layout=layout, unaddressed=np.flatnonzero(~addressed).astype(np.int64))


def model_tokens(system: MolecularSystem, layout: TokenLayout) -> tuple[NDArray[np.int64], EncodingLoss]:
    r"""The model input token of every token (its canonical class) and the positions with encoding loss."""
    classes = token_classes(system)
    components = np.unique(layout.token_component)
    lossy = components[classes.encoding[components] != Encoding.EXACT]
    return layout.token_class.astype(np.int64), EncodingLoss(lossy, classes.token[lossy], classes.encoding[lossy])


def _named(system: MolecularSystem, layout: TokenLayout, wanted: NDArray[np.str_]) -> NDArray[np.int64]:
    r"""Per token, its atom named ``wanted[token]``, else its centre atom."""
    owner = _owner(layout.token_atom_ptr)
    names = system.text("atom", system.atom_symbol)[layout.token_atoms]
    hit = names == wanted[owner]
    found = layout.token_centre_atom.copy()
    first = np.flatnonzero(hit)[::-1]
    found[owner[first]] = layout.token_atoms[first]
    return found


def token_representatives(system: MolecularSystem, layout: TokenLayout, table: LayoutTable) -> NDArray[np.int64]:
    r"""Per token, its representative (distogram) atom by the table's class and monomer rules, else its centre."""
    monomer = CLASS_MONOMER[layout.token_class]
    wanted = [
        table.representative.get(int(k), table.representative_default.get(MonomerType(m), ""))
        for k, m in zip(layout.token_class.tolist(), monomer.tolist())
    ]
    return _named(system, layout, np.array(wanted + [""])[: len(wanted)])


def token_frames(system: MolecularSystem, layout: TokenLayout, table: LayoutTable) -> NDArray[np.int64]:
    r"""Per token, its three frame atoms by the table's monomer rule (``[tokens, 3]``), each else its centre."""
    monomer = CLASS_MONOMER[layout.token_class].tolist()
    triples = [table.frame_atoms.get(MonomerType(m), ("", "", "")) for m in monomer]
    columns = [_named(system, layout, np.array([t[k] for t in triples] + [""])[: len(triples)]) for k in range(3)]
    return np.stack(columns, axis=1) if columns[0].size else np.zeros((0, 3), dtype=np.int64)
