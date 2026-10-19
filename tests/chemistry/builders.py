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
Builders shared by the chemistry and tool tests (a plain module, imported; fixtures live in ``conftest``).

``make_items`` builds synthetic ``ComponentItems`` the way io's codec delivers them (no decoder needed): tokens are
KNOWN, ``"?"``/``"."`` are the stated nulls, an item left out is absent. ``SUBSET`` is the byte-identical CCD subset of
``tests/fixtures/ccd``; ``glycine_columns`` are fresh, editable columns of CCD GLY.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ofoldx.chemistry.definitions import ARTIFACT_SCHEMA, ArtifactProvenance, ComponentItems
from ofoldx.data.molecular import ELEMENTS
from ofoldx.provenance import Basis, Resolution, SourceRefs, state

SUBSET = Path(__file__).resolve().parents[1] / "fixtures" / "ccd" / "components_subset.cif.gz"
DECLARED, DEFAULT = state(Resolution.KNOWN, Basis.DECLARED), state(Resolution.KNOWN, Basis.DEFAULT)
NULLS = {"?": state(Resolution.UNKNOWN, Basis.DECLARED), ".": state(Resolution.INAPPLICABLE, Basis.DECLARED)}
ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
NUMERIC = ("charge", *(f"{p}Cartn_{a}{s}" for p, s in (("model_", ""), ("pdbx_model_", "_ideal")) for a in "xyz"))
ISOTOPES = {"D": (1, 2), "T": (1, 3)}


def token_state(token: object) -> int:
    if token is None:
        return ABSENT
    if isinstance(token, tuple):  # ("default", value): an absent item whose codec default io filled
        return DEFAULT
    return NULLS.get(token, DECLARED)


def payload(token: object) -> object:
    return token[1] if isinstance(token, tuple) else None if token is None or token in NULLS else token


def make_items(
    component: Mapping[str, object],
    atoms: Mapping[str, Sequence[object]] | None = None,
    bonds: Mapping[str, Sequence[object]] | None = None,
    descriptors: Iterable[tuple[str, str, str]] = (),
) -> ComponentItems:
    r"""Synthetic items of one component. Numeric atom items take floats (quantum 0.001; charge quantum 1);
    ``type_symbol`` is decoded as io does (case-insensitive, ``D``/``T`` → H with isotope 2/3, ``X`` → 0)."""

    atoms, bonds = dict(atoms or {}), dict(bonds or {})
    n = len(next(iter(atoms.values()), ()))
    m = len(next(iter(bonds.values()), ()))
    symbols = atoms.pop("type_symbol", ["C"] * n)
    index = {symbol.upper(): z for z, symbol in enumerate(ELEMENTS)}
    pairs = [
        ISOTOPES.get(s.upper(), (index.get(s.upper(), 0), 0)) if token_state(s) == DECLARED else (0, 0) for s in symbols
    ]
    element = (
        np.array([z for z, _ in pairs], dtype=np.uint8),
        np.array([i for _, i in pairs], dtype=np.uint16),
        np.array([token_state(s) for s in symbols], dtype=np.uint8),
    )
    numbers = {}
    for item in [item for item in atoms if item in NUMERIC]:
        tokens = atoms.pop(item)
        values = np.array([float(payload(t) or 0.0) for t in tokens])
        states = np.array([token_state(t) for t in tokens], dtype=np.uint8)
        quantum = np.where(states == DECLARED, 1.0 if item == "charge" else 0.001, 0.0)
        numbers[item] = (values, states, quantum)

    def text(columns: dict) -> MappingProxyType:
        return MappingProxyType(
            {
                item: (
                    np.array([(payload(t) or "").encode() for t in tokens], dtype="S"),
                    np.array([token_state(t) for t in tokens], dtype=np.uint8),
                )
                for item, tokens in columns.items()
            }
        )

    descriptors = tuple(descriptors)
    rows = [("chem_comp", 0)] * bool(component) + [("chem_comp_atom", r) for r in range(n)]
    rows += [("chem_comp_bond", r) for r in range(m)] + [
        ("pdbx_chem_comp_descriptor", r) for r in range(len(descriptors))
    ]
    categories = ("chem_comp", "chem_comp_atom", "chem_comp_bond", "pdbx_chem_comp_descriptor")
    refs = SourceRefs(
        np.array([categories.index(c) for c, _ in rows], dtype=np.int16),
        np.array([r for _, r in rows], dtype=np.int64),
        np.full(len(rows), -1, dtype=np.int16),
        categories,
        (),
        "TEST",
    )
    fields = {item: (payload(token), token_state(token)) for item, token in component.items()}
    return ComponentItems(
        MappingProxyType(fields), text(atoms), MappingProxyType(numbers), element, text(bonds), descriptors, refs
    )


def edit_atom(items: ComponentItems, item: str, atom: str, token: bytes) -> ComponentItems:
    r"""``items`` with one ``_chem_comp_atom`` text token of one atom replaced."""
    values, states = items.atoms[item]
    names = [name.strip() for name in items.atoms["atom_id"][0].tolist()]
    values = values.astype(f"S{max(values.dtype.itemsize, len(token))}")
    values[names.index(atom.encode())] = token
    atoms = MappingProxyType({**items.atoms, item: (values, states.copy())})
    return ComponentItems(
        items.component, atoms, items.numbers, items.element, items.bonds, items.descriptors, items.refs
    )


def provenance(content_version: str = "test") -> ArtifactProvenance:
    return ArtifactProvenance(ARTIFACT_SCHEMA, content_version, "")


# CCD GLY (atom_id, type_symbol, charge, aromatic, leaving, stereo, n-terminal, c-terminal, ideal x y z) and bonds
GLY_ATOMS = """
N   N 0 N N N Y N  1.931  0.090 -0.034
CA  C 0 N N N N N  0.761 -0.799 -0.008
C   C 0 N N N N Y -0.498  0.029 -0.005
O   O 0 N N N N Y -0.429  1.235 -0.023
OXT O 0 N Y N N Y -1.697 -0.574  0.018
H   H 0 N N N Y N  1.910  0.738  0.738
H2  H 0 N Y N Y N  2.788 -0.442 -0.037
HA2 H 0 N N N N N  0.772 -1.440 -0.889
HA3 H 0 N N N N N  0.793 -1.415  0.891
HXT H 0 N Y N N Y -2.477 -0.002  0.019
"""
GLY_BONDS = "N CA SING, N H SING, N H2 SING, CA C SING, CA HA2 SING, CA HA3 SING, C O DOUB, C OXT SING, OXT HXT SING"
ATOM_ITEMS = ("atom_id", "type_symbol", "charge", "pdbx_aromatic_flag", "pdbx_leaving_atom_flag", "pdbx_stereo_config")
ATOM_ITEMS += ("pdbx_n_terminal_atom_flag", "pdbx_c_terminal_atom_flag")
ATOM_ITEMS += tuple(f"pdbx_model_Cartn_{axis}_ideal" for axis in "xyz")


def glycine_columns() -> dict:
    r"""Fresh, editable columns of CCD GLY: ``{"component": ..., "atoms": ..., "bonds": ...}`` for ``make_items``."""
    rows = [line.split() for line in GLY_ATOMS.strip().splitlines()]
    atoms = {item: [row[k] for row in rows] for k, item in enumerate(ATOM_ITEMS)}
    for item in ATOM_ITEMS[8:] + ("charge",):
        atoms[item] = [float(token) for token in atoms[item]]
    bonds = [bond.split() for bond in GLY_BONDS.split(", ")]
    bond_items = ("atom_id_1", "atom_id_2", "value_order")
    bonds = {item: [bond[k] for bond in bonds] for k, item in enumerate(bond_items)}
    bonds |= {"pdbx_aromatic_flag": ["N"] * 9, "pdbx_stereo_config": ["N"] * 9}
    component = {"id": "GLY", "type": "L-PEPTIDE LINKING", "one_letter_code": "G", "mon_nstd_parent_comp_id": "?"}
    return {"component": component, "atoms": atoms, "bonds": bonds}


# ---- attachment builders (tests of chemistry.declarations and chemistry.attachments)

KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
KNOWN_ASSUMED = state(Resolution.KNOWN, Basis.ASSUMED)
UNKNOWN_DECLARED = NULLS["?"]
UNKNOWN_DEFINED = state(Resolution.UNKNOWN, Basis.DEFINED)


def edge(order: int = 1, order_state: int = DECLARED, *, kind: int = 1, count: int = 1, retention: bool = False):
    r"""One new edge of an ``EndpointProblem``: covalent (kind 1) or coordination (2), a SING–QUAD order."""
    token = order if order_state == DECLARED else 0
    return (kind, KNOWN_DEFINED, order if order <= 4 else 0, order_state, token, count, retention)


def problem(component, atom: str, edges=(), *, observed=(), placed=(), bonded=(), declared=(), charge=None, **fields):
    r"""The ``EndpointProblem`` of ``atom`` of a definition: its terminal-fragment atoms with the given atom names
    observed, placed at occupancy 0 or bonded elsewhere; ``declared`` leaving ids; ``charge`` a stated charge."""
    from ofoldx.chemistry.attachments import EndpointProblem
    from ofoldx.chemistry.declarations import InUse
    from ofoldx.chemistry.fragments import terminal_fragments

    e = component.atom_names.index(atom)
    fragments = terminal_fragments(component, e)
    atoms = tuple(sorted(a for f in fragments for a in f.atoms))
    names = [component.atom_names[a] for a in atoms]
    designated = sum(
        1 << atoms.index(a)
        for f in fragments
        if component.leaving_flag[f.root] == 2
        for a in f.atoms
        if component.leaving_flag[a] == 2
    )
    bits = [
        (InUse.OBSERVED | InUse.PLACED) * (n in observed) | InUse.PLACED * (n in placed) | InUse.BONDED * (n in bonded)
        for n in names
    ]
    inside = [component.atom_names.index(n) for n in declared if n in names]
    outside = tuple(
        component.atom_names.index(n) if n in component.atom_names else -1 for n in declared if n not in names
    )
    return EndpointProblem(
        definition=0,
        atom=e,
        edges=tuple(sorted(edges)),
        fragment_atoms=atoms,
        declared=sum(1 << atoms.index(a) for a in inside),
        in_use=tuple(int(b) for b in bits),
        designated=designated,
        charge=0 if charge is None else charge,
        charge_state=ABSENT if charge is None else DECLARED,
        observed_only=fields.pop("observed_only", False),
        existence_unknown=fields.pop("existence_unknown", 0),
        symmetry_unknown=fields.pop("symmetry_unknown", False),
        assumed=fields.pop("assumed", ()),
        declared_outside=outside,
        **fields,
    )


def occurrences(*components, observed: set[str] = frozenset()):
    r"""Occurrences of the given definitions (rows 0..n-1 of ``definitions``) with the named atoms observed."""
    sizes = [len(c.atom_names) for c in components]
    ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    seen = np.array([name in observed for c in components for name in c.atom_names])
    n = int(ptr[-1])
    from ofoldx.chemistry.declarations import OccurrenceTable

    return OccurrenceTable(
        ordinal=np.arange(len(components), dtype=np.int32),
        definition=np.arange(len(components), dtype=np.int32),
        unusable=np.zeros(len(components), dtype=bool),
        unsupported=np.zeros(len(components), dtype=bool),
        atom_ptr=ptr,
        placed=seen.copy(),
        observed=seen,
        zero_occupancy=np.zeros(n, dtype=bool),
        atom_ref=np.full(n, -1, dtype=np.int64),
        charge=np.zeros(n, dtype=np.int8),
        charge_state=np.full(n, ABSENT, dtype=np.uint8),
        observed_only=np.zeros(n, dtype=np.uint8),
    )


def identity():
    r"""A transform table holding the identity only."""
    from ofoldx.data.molecular import Transforms

    return Transforms(
        np.eye(3, dtype=np.int8)[None],
        np.zeros((1, 3), dtype=np.int16),
        np.full(1, -1, np.int32),
        np.full(1, KNOWN_DEFINED, np.uint8),
    )
