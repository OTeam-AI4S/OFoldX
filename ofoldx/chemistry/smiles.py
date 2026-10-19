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
SMILES and CXSMILES import as a component definition.

The configuration facts of a string are read from the string itself. A ``@``/``@@`` tag is defined by OpenSMILES 3.8.2
over the neighbour order of the string — an implicit hydrogen immediately after the atom the centre was reached from,
first when the centre opens the string; a lone pair of a three-connected centre last, as RDKit reads it — and is stored
as a parity by permutation onto the definition's slot order (no CIP pass). RDKit supplies chemistry only:
sanitization, aromaticity and the hydrogen inventory. A definition represents no radical, checked atom by atom: a
count the string itself states (a ``^n:`` key) is a declaration, refused on any atom; a count RDKit's assignRadicals
computes is refused on a non-metal atom and not consulted on a metal atom, being a computation step, not an electron
state the source declared. Metal electronic states are outside the model: nothing claims a metal's radical count is
zero, and a molecule containing a metal is still checked on its non-metal atoms. The CX block is tokenized by key
before RDKit sees it, so an unsupported declaration is a typed refusal rather than a silently ignored one; agreement
between declarations that must say the same thing (a wedge and the tag it starts at, a ring ``c:``/``t:`` and a
``/``/``\`` mark on the same bond) is checked here from the geometry and the string.

The definition names its atoms by element and per-element count in string order (``C1``, ``N1``, ``C2``; the
hydrogens RDKit adds follow the string's atoms), has scope SMILES and no reference conformer. Its records stay in
``SmilesImport.records`` and never join a read report.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from math import dist
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import DefinitionScope, ReferenceSource
from ..data.stereo import StereoGroupKind, parity_from_coordinates, permutation_parity
from ..errors import CODE_NAMES, SourceRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, state
from .definitions import ChemicalComponent, StereoBasis, stated_component

# Every SMILES failure and record is about the one string it was read from.
_ROW = SourceRef("smiles", 0)
_STATED = state(Resolution.KNOWN, Basis.REFERENCE)
_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)

_TOKEN = re.compile(r"(\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]|\*)|([-=#$:/\\.~])|(%\d\d|\d)|([()])")
_CHIRALITY = re.compile(r"@(@)?(TH|AL|SP|TB|OH)?(\d*)")
_BRACKET_HYDROGEN = re.compile(r"H(\d*)")
_CX_KEY = re.compile(r"(\^\d+|@@|@|&\d+|[A-Za-z][A-Za-z0-9]*)(:?)")
_PAIR = re.compile(r"(\d+)\.(\d+)")

# CX keys whose meaning OFoldX does not represent: a typed refusal, never a silent omission.
_UNSUPPORTED_KEYS = frozenset(
    {"TLB", "THB", "TEB", "@", "@@", "labels", "atomProp", "LN", "m", "rb", "s", "u", "C", "H", "Z", "LP", "RG", "f"}
)
# Expected coordination of each non-tetrahedral OpenSMILES stereo class.
_CLASS_COORDINATION = {"SP": 4, "TB": 5, "OH": 6}
# The non-metal domain, metalloids included (the snapshot's NON_METAL_ELEMENTS without the unknown X): a four-fold
# coordination is tetrahedral, never square planar, and RDKit's computed radical count is consulted, and refused, on
# these atoms only.
_NON_METALS = frozenset(
    (
        "H",
        "He",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Sb",
        "Te",
        "I",
        "Xe",
        "At",
        "Rn",
    )
)
_ORDERS = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}


@dataclass(frozen=True)
class SmilesImport:
    r"""One component read from a SMILES or CXSMILES string, and the records of the import."""

    component: ChemicalComponent
    records: Records


class _Structure(NamedTuple):
    r"""The base string as written: atom order, neighbour order (``-1`` an open ring bond), ChemAxon bond order."""

    tokens: tuple[str, ...]
    neighbors: tuple[tuple[int, ...], ...]
    previous: tuple[int | None, ...]
    bonds: tuple[tuple[int, int, str], ...]


class _Declarations(NamedTuple):
    r"""The accepted content of one CX block."""

    groups: tuple[tuple[str, tuple[int, ...]], ...]
    relative: bool
    cis: tuple[int, ...]
    trans: tuple[int, ...]
    unknown_bonds: tuple[int, ...]
    wavy: tuple[tuple[int, int], ...]
    wedges: tuple[tuple[str, int, int], ...]
    coordinates: tuple[tuple[float, float, float], ...] | None


class _Parity(NamedTuple):
    r"""One declared tetrahedral configuration (1 ``@``, 2 ``@@``) and the neighbour order it is read over (``-1`` the
    implicit hydrogen or lone pair)."""

    parity: int
    order: tuple[int, ...]


def read_smiles(text: str, *, component_id: str) -> SmilesImport:
    r"""
    Read one SMILES or CXSMILES string as a definition of scope SMILES.

    Accepted CX keys: ``a:``, ``oN:``, ``&N:``, ``c:``, ``t:``, ``ctu:``, ``w:``, ``wU:``, ``wD:``, a coordinate block,
    ``^n:`` and a bare ``r`` (one AND group over every tagged centre outside an explicit group). ``ctu:`` leaves its
    double bond's geometry unstated, a ``w:`` wavy bond the configuration of the atom it starts at and the geometry of
    every double bond at its two atoms (either configuration, as RDKit reads it); a double bond marked on one side
    only states nothing and is recorded
    (``cx_incomplete_declaration``) when each of its atoms has another neighbour atom in the string. A wedge only draws
    the configuration of its atom's tag and is checked against it where coordinates are given.

    Args:
        text: SMILES or supported CXSMILES text defining one component.
        component_id: Caller-assigned identity for the imported component definition.

    Returns:
        The component definition and import records, including supported declarations whose information remains
            incomplete. The result can supply a caller definition to a structure read or recipe build.

    Raises:
        ContradictionError: ``smiles_syntax`` (malformed string, bracket field or CX key);
            ``cx_conflicting_declaration`` (two declarations of one configuration disagree); ``invalid_stereo_class``
            (a stereo class inconsistent with the atom's coordination); ``smiles_chemistry`` (no valence or Kekulé
            structure).
        UnsupportedError: ``cx_unsupported_declaration``, ``unsupported_stereo_class``, ``smiles_unsupported``;
            ``smiles_atom_mapping_unsupported`` (an atom-map number, read from the parsed atoms: a definition keeps no
            atom mapping; payload the string atoms and their map numbers); ``unsupported_radical`` (an atom a ``^n:``
            key states radical electrons for, or a non-metal atom RDKit assigns them to, as a bracket atom below its
            valence; a metal atom's computed count is not consulted; payload the string atoms and their radical
            electrons, the stated count where one is stated).

    Examples:
        >>> component = read_smiles("N[C@@H](C)C(=O)O", component_id="LIG1").component
        >>> component.atom_names[:6], component.stereo_centre.tolist()
        (('N1', 'C1', 'C2', 'C3', 'O1', 'O2'), [1])
    """

    base, block = _split_extension(text)
    declarations = _read_extension(block)
    structure = _parse_structure(base)
    parities = _declared_parities(structure)
    for atom, _ in declarations.wavy:
        parities.pop(atom, None)
    molecule, stated_orders, stated_charges, stated_electrons = _rdkit_molecule(text, structure)
    incomplete: list[tuple[int, int]] = []
    geometry = _ring_declarations(molecule, structure, declarations, _directional_geometry(structure, incomplete))
    _check_wedges(structure, declarations, parities)
    groups = _stereo_groups(declarations, len(structure.tokens), parities)
    component = _component(
        molecule,
        stated_orders,
        stated_charges,
        stated_electrons,
        component_id,
        parities,
        geometry,
        declarations,
        structure,
        groups,
    )
    pair_bond = {frozenset(pair): row for row, pair in enumerate(component.bond_atoms.tolist())}
    return SmilesImport(component, _records([pair_bond[frozenset(pair)] for pair in incomplete]))


def standardize_smiles(text: str) -> str:
    r"""
    The ChEMBL-standardised parent of a SMILES string as RDKit canonical SMILES: the ChEMBL exclusion flag, the
    largest fragment and ``standardize_mol`` (the rule of Boltz-2's affinity ligands, upstream ``schema.standardize``).

    Raises:
        ContradictionError: ``smiles_syntax`` (RDKit cannot parse the string); ``smiles_chemistry`` (the standardised
            molecule does not sanitise).
        UnsupportedError: ``smiles_excluded``.
    """

    from chembl_structure_pipeline.exclude_flag import exclude_flag
    from chembl_structure_pipeline.standardizer import standardize_mol
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(text, sanitize=False)
    if molecule is None:
        fail("smiles_syntax", _ROW, at=text)
    if exclude_flag(molecule, includeRDKitSanitization=False):
        fail("smiles_excluded", _ROW)
    molecule.UpdatePropertyCache(strict=False)
    standardized = Chem.MolToSmiles(standardize_mol(rdMolStandardize.LargestFragmentChooser().choose(molecule)))
    if Chem.MolFromSmiles(standardized) is None:
        fail("smiles_chemistry", _ROW)
    return standardized


def canonical_atom_ranks(text: str) -> NDArray[np.int64]:
    r"""
    RDKit's canonical rank of every atom of a SMILES string parsed and hydrogen-completed by RDKit (its atoms in string
    order, then the added hydrogens: ``read_smiles``'s order for a string without hydrogen atoms).

    Raises:
        ContradictionError: ``smiles_syntax`` (RDKit cannot parse the string).
    """

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        fail("smiles_syntax", _ROW, at=text)
    return np.array(list(Chem.CanonicalRankAtoms(Chem.AddHs(molecule))), dtype=np.int64)


def _split_extension(text: str) -> tuple[str, str | None]:
    r"""The base SMILES and its CX block."""

    stripped = text.strip()
    if "|" not in stripped:
        return stripped, None
    base, _, rest = stripped.partition("|")
    if not rest.endswith("|"):
        fail("smiles_syntax", _ROW, at="unterminated CX block")
    return base.strip(), rest[:-1]


def _read_extension(block: str | None) -> _Declarations:
    r"""Tokenize a CX block by key and accept only the keys OFoldX represents."""

    groups: list[tuple[str, tuple[int, ...]]] = []
    relative = False
    cis: list[int] = []
    trans: list[int] = []
    unknown_bonds: list[int] = []
    wavy: list[tuple[int, int]] = []
    wedges: list[tuple[str, int, int]] = []
    coordinates = None
    for key, payload in _cx_items(block or ""):
        if key == "coordinates":
            coordinates = _coordinates(payload)
        elif key == "r":
            relative = True
        elif key == "a" or re.fullmatch(r"[o&]\d+", key):
            groups.append((key, _indices(payload)))
        elif key in ("c", "t", "ctu"):
            {"c": cis, "t": trans, "ctu": unknown_bonds}[key].extend(_indices(payload))
        elif key == "w":
            wavy.extend(_pairs(payload))
        elif key in ("wU", "wD"):
            wedges.extend((key, atom, bond) for atom, bond in _pairs(payload))
        elif key.startswith("^"):
            continue  # the parsed molecule carries the stated radical electrons, refused on any atom
        elif key in _UNSUPPORTED_KEYS or key.startswith("Sg"):
            fail("cx_unsupported_declaration", _ROW, key=key)
        else:
            fail("smiles_syntax", _ROW, at=f"CX key {key!r}")
    return _Declarations(
        tuple(groups), relative, tuple(cis), tuple(trans), tuple(unknown_bonds), tuple(wavy), tuple(wedges), coordinates
    )


def _cx_items(block: str) -> list[tuple[str, str]]:
    r"""A CX block as ``(key, payload)`` items, payloads uninterpreted."""

    items: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(block):
        character = block[cursor]
        if character == ",":
            cursor += 1
            continue
        if character in "($":
            stop = block.find(")" if character == "(" else "$", cursor + 1)
            if stop < 0:
                fail("smiles_syntax", _ROW, at=f"unterminated CX {'coordinate' if character == '(' else 'label'} block")
            items.append(("coordinates" if character == "(" else "labels", block[cursor + 1 : stop]))
            cursor = stop + 1
            continue
        match = _CX_KEY.match(block, cursor)
        if match is None:
            fail("smiles_syntax", _ROW, at=f"CX block at {block[cursor:]!r}")
        cursor = match.end()
        if not match.group(2):
            items.append((match.group(1), ""))
            continue
        stop = cursor
        while stop < len(block) and not (block[stop] == "," and not block[stop + 1 : stop + 2].isdigit()):
            stop += 1
        items.append((match.group(1), block[cursor:stop]))
        cursor = stop
    return items


def _indices(payload: str) -> tuple[int, ...]:
    return tuple(int(value) for value in re.findall(r"\d+", payload))


def _pairs(payload: str) -> tuple[tuple[int, int], ...]:
    return tuple((int(first), int(second)) for first, second in _PAIR.findall(payload))


def _coordinates(payload: str) -> tuple[tuple[float, float, float], ...]:
    points = []
    for item in payload.split(";"):
        parts = (item.split(",") + ["", "", ""])[:3]
        try:
            x, y, z = (float(value) if value else 0.0 for value in parts)
            points.append((x, y, z))
        except ValueError:
            fail("smiles_syntax", _ROW, at=f"CX coordinate {item!r}")
    return tuple(points)


def _parse_structure(base: str) -> _Structure:
    r"""Atom order, per-atom neighbour order and ChemAxon bond order (a ring bond numbered when closed)."""

    tokens: list[str] = []
    neighbors: list[list[int]] = []
    previous: list[int | None] = []
    bonds: list[tuple[int, int, str]] = []
    ring_open: dict[str, tuple[int, int, str]] = {}
    branches: list[int] = []
    prior: int | None = None
    pending = ""
    cursor = 0
    for match in _TOKEN.finditer(base):
        if match.start() != cursor:
            break
        cursor = match.end()
        atom, bond, ring, bracket = match.groups()
        if atom is not None:
            index = len(tokens)
            tokens.append(atom)
            neighbors.append([])
            previous.append(None)
            if prior is not None and pending != ".":
                bonds.append((prior, index, pending))
                neighbors[prior].append(index)
                neighbors[index].append(prior)
                previous[index] = prior
            prior, pending = index, ""
        elif bond is not None:
            pending = bond
        elif ring is not None:
            if prior is None:
                fail("smiles_syntax", _ROW, at=f"ring closure {ring!r} without an atom")
            opened = ring_open.pop(ring, None)
            if opened is None:
                ring_open[ring] = (prior, len(neighbors[prior]), pending)
                neighbors[prior].append(-1)
            else:
                start, slot, symbol = opened
                if pending:  # a mark at the closing digit runs from this atom to the opening one
                    symbol = {"/": "\\", "\\": "/"}.get(pending, pending)
                bonds.append((start, prior, symbol))
                neighbors[start][slot] = prior
                neighbors[prior].append(start)
            pending = ""
        elif bracket == "(":
            if prior is None:
                fail("smiles_syntax", _ROW, at="branch without an atom")
            branches.append(prior)
        else:
            if not branches:
                fail("smiles_syntax", _ROW, at="unbalanced branch")
            prior = branches.pop()
    if cursor != len(base):
        fail("smiles_syntax", _ROW, at=base[cursor:])
    if ring_open or branches:
        fail("smiles_syntax", _ROW, at="unclosed ring bond or branch")
    return _Structure(tuple(tokens), tuple(map(tuple, neighbors)), tuple(previous), tuple(bonds))


def _declared_parities(structure: _Structure) -> dict[int, _Parity]:
    r"""Every tetrahedral tag, read over the OpenSMILES neighbour order of the string."""

    parities: dict[int, _Parity] = {}
    for index, token in enumerate(structure.tokens):
        if not token.startswith("[") or "@" not in token:
            continue
        inner = token[1:-1]
        match = cast(re.Match[str], _CHIRALITY.search(inner))
        stereo_class = match.group(2) or "TH"
        number = int(match.group(3)) if match.group(3) else (2 if match.group(1) else 1)
        order = _neighbor_order(structure, index, inner[match.end() :])
        if stereo_class in _CLASS_COORDINATION:
            element = _element(token)
            if len(order) != _CLASS_COORDINATION[stereo_class] or (stereo_class == "SP" and element in _NON_METALS):
                fail("invalid_stereo_class", _ROW, atom=index, stereo_class=stereo_class, coordination=len(order))
            fail("unsupported_stereo_class", _ROW, atom=index, stereo_class=stereo_class)
        if stereo_class == "AL":
            fail("unsupported_stereo_class", _ROW, atom=index, stereo_class=stereo_class)
        if number not in (1, 2):
            fail("smiles_syntax", _ROW, at=f"@TH{number} at atom {index}")
        if len(order) == 3:
            order = (*order, -1)  # the lone pair of a three-connected centre is the last slot
        if len(order) != 4:
            fail("invalid_stereo_class", _ROW, atom=index, stereo_class="TH", coordination=len(order))
        parities[index] = _Parity(number, order)
    return parities


def _neighbor_order(structure: _Structure, index: int, rest: str) -> tuple[int, ...]:
    r"""The neighbour order with the implicit hydrogen placed where OpenSMILES 3.8.2 defines it."""

    order = list(structure.neighbors[index])
    match = _BRACKET_HYDROGEN.match(rest)
    count = int(match.group(1) or 1) if match else 0
    if count > 1:
        fail("invalid_stereo_class", _ROW, atom=index, stereo_class="TH", implicit_hydrogens=count)
    if count:
        prior = structure.previous[index]
        order.insert(0 if prior is None else order.index(prior) + 1, -1)
    return tuple(order)


def _element(token: str) -> str:
    match = re.match(r"\[(\d*)([A-Z][a-z]?|[a-z]{1,2}|\*)", token)
    return "" if match is None else match.group(2).capitalize()


def _directional_geometry(structure: _Structure, incomplete: list[tuple[int, int]]) -> dict[int, tuple[int, int, bool]]:
    r"""
    ``/`` and ``\`` marks as same-side relations around each double bond: string bond → (reference at the first
    atom, reference at the second, same side). Two marks at one atom putting both substituents on one side state an
    impossible geometry (OpenSMILES 3.8.3); a double bond marked on one side only states nothing (OpenSMILES 5) and is
    appended to ``incomplete`` when each of its atoms has a neighbour atom in the string besides its partner (a bracket
    hydrogen is not counted: ``F/C=[CH2]`` and ``F/C=C`` are one molecule and give one answer).
    """

    side: dict[tuple[int, int], int] = {}
    for first, second, symbol in structure.bonds:
        if symbol in ("/", "\\"):
            value = 1 if symbol == "/" else -1
            side[second, first] = -value
            side[first, second] = value
    geometry: dict[int, tuple[int, int, bool]] = {}
    degree = [len(neighbors) for neighbors in structure.neighbors]
    for bond, (first, second, symbol) in enumerate(structure.bonds):
        if symbol != "=":
            continue
        marked_first = _marked_neighbor(structure, first, second, side)
        marked_second = _marked_neighbor(structure, second, first, side)
        if marked_first is None or marked_second is None:
            if (marked_first or marked_second) and degree[first] >= 2 and degree[second] >= 2:
                incomplete.append((first, second))
            continue
        geometry[bond] = (marked_first[0], marked_second[0], marked_first[1] == marked_second[1])
    return geometry


def _marked_neighbor(structure: _Structure, atom: int, other: int, side: dict) -> tuple[int, int] | None:
    r"""The marked substituent of one double-bond atom and the side it sits on."""

    marked = [(n, side[atom, n]) for n in structure.neighbors[atom] if n >= 0 and n != other and (atom, n) in side]
    if not marked:
        return None
    if len({value for _, value in marked}) != len(marked):
        fail("cx_conflicting_declaration", _ROW, atom=atom, declaration="directional marks on one side")
    return marked[0]


def _ring_declarations(molecule: Any, structure: _Structure, declarations: _Declarations, directions: dict) -> dict:
    r"""
    The ``c:``/``t:`` ring geometries added to the base marks and checked against them. ChemAxon states a ring double
    bond cis or trans along the ring, so its reference pair is the pair of ring neighbours; a base mark on the same bond
    states the same geometry over its own pair, and the two must agree once expressed over one pair.
    """

    geometry = dict(directions)
    for bonds, ring_same_side in ((declarations.cis, True), (declarations.trans, False)):
        for bond in bonds:
            first, second = _string_bond(structure, bond)
            if str(molecule.GetBondBetweenAtoms(first, second).GetBondType()) != "DOUBLE":
                fail("cx_conflicting_declaration", _ROW, bond=bond, declaration="ring geometry on a non-double bond")
            ring = _ring_neighbors(structure, bond)
            if ring is None:
                fail("cx_conflicting_declaration", _ROW, bond=bond, declaration="ring geometry on a non-ring bond")
            declared = directions.get(bond)
            if declared is None:
                geometry[bond] = (ring[0], ring[1], ring_same_side)
            elif (declared[2] != ((declared[0] != ring[0]) != (declared[1] != ring[1]))) != ring_same_side:
                fail("cx_conflicting_declaration", _ROW, bond=bond, declaration="ring geometry against marks")
    return geometry


def _string_bond(structure: _Structure, bond: int) -> tuple[int, int]:
    if bond >= len(structure.bonds):
        fail("smiles_syntax", _ROW, at=f"CX bond {bond} of {len(structure.bonds)}")
    return structure.bonds[bond][:2]


def _ring_neighbors(structure: _Structure, bond: int) -> tuple[int, int] | None:
    r"""The neighbours continuing the ring on each side of one ring bond (a shortest path around it)."""

    first, second, _ = structure.bonds[bond]
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(structure.tokens))}
    for index, (a, b, _) in enumerate(structure.bonds):
        if index != bond:
            adjacency[a].append(b)
            adjacency[b].append(a)
    came_from = {first: first}
    pending = deque([first])
    while pending:
        atom = pending.popleft()
        if atom == second:
            path = [atom]
            while path[-1] != first:
                path.append(came_from[path[-1]])
            return path[-2], path[1]
        for neighbor in adjacency[atom]:
            if neighbor not in came_from:
                came_from[neighbor] = atom
                pending.append(neighbor)
    return None


def _check_wedges(structure: _Structure, declarations: _Declarations, parities: dict[int, _Parity]) -> None:
    r"""
    Every wedge against the tag of the atom it starts at, by the parity of the 2D coordinates with the wedged
    neighbours raised or lowered (RDKit does not use wedges to set stereochemistry). A wedge is a drawing of the
    configuration the tag states, never a second source of it: a wedge at an untagged atom is one ChemAxon found to
    carry no configuration (it writes a tag for every wedge that does), and a wedge without a coordinate block cannot
    be checked; neither states anything.
    """

    points = declarations.coordinates
    if not declarations.wedges or points is None:
        return
    if len(points) != len(structure.tokens):
        fail("cx_conflicting_declaration", _ROW, declaration=f"{len(points)} points for {len(structure.tokens)} atoms")
    raised: dict[int, dict[int, float]] = {}
    for key, atom, bond in declarations.wedges:
        first, second = _string_bond(structure, bond)
        if atom not in (first, second):
            fail("cx_conflicting_declaration", _ROW, atom=atom, declaration=f"wedge on bond {bond} it does not touch")
        other = second if atom == first else first
        length = dist(points[atom][:2], points[other][:2]) or 1.0
        raised.setdefault(atom, {})[other] = 0.8 * length * (1.0 if key == "wU" else -1.0)
    for atom, offsets in raised.items():
        declared = parities.get(atom)
        if declared is None:
            continue
        coordinates = np.array(points, dtype=np.float64)
        for neighbor, offset in offsets.items():
            coordinates[neighbor, 2] = offset
        slots = np.array([declared.order])
        computed = int(parity_from_coordinates(coordinates, np.array([atom]), slots, np.zeros(len(points)))[0])
        if computed and computed != declared.parity:
            fail("cx_conflicting_declaration", _ROW, atom=atom, declaration="wedge against tag")


def _rdkit_molecule(text: str, structure: _Structure) -> tuple[Any, list[bool], list[bool], list[int]]:
    r"""
    The string parsed by RDKit in its own atom order, sanitized, Kekulé orders set and hydrogens added; per bond and
    per string atom whether its order or charge is stated; and per string atom the radical electrons its ``^n:`` key
    states (read before sanitization, which rewrites them). RDKit's cleanup rewrites hypervalent groups (nitro,
    azide, perchlorate) into charge-separated form before the Kekulé orders are assigned around the written ones, so
    charges and orders stay one structure; an order or charge that differs from the string's, or an aromatic bond's
    Kekulé order, is derived. Existing atom and bond indices are kept.
    """

    from rdkit import Chem

    parameters: Any = Chem.SmilesParserParams()
    parameters.allowCXSMILES = True
    parameters.strictCXSMILES = True
    parameters.parseName = False
    parameters.removeHs = False
    parameters.sanitize = False
    molecule = Chem.MolFromSmiles(text, parameters)
    if molecule is None or molecule.GetNumAtoms() != len(structure.tokens):
        fail("smiles_syntax", _ROW, at=text)
    # a valid annotation a definition does not keep: refused as unsupported, never dropped
    mapped = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in molecule.GetAtoms() if atom.GetAtomMapNum()]
    if mapped:
        atoms, numbers = zip(*mapped)
        fail("smiles_atom_mapping_unsupported", _ROW, atoms=atoms, map_numbers=numbers)
    written = [bond.GetBondType() for bond in molecule.GetBonds()]
    charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
    stated_electrons = [atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()]
    try:
        molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(molecule, Chem.SanitizeFlags.SANITIZE_CLEANUP)
        kekule = Chem.RWMol(molecule)  # Kekulé orders for the aromatic bonds only, around the written ones
        kekule.UpdatePropertyCache(strict=False)
        Chem.Kekulize(kekule, clearAromaticFlags=False)
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError):
        fail("smiles_chemistry", _ROW)
    complete = Chem.RWMol(Chem.AddHs(molecule))
    for bond, kekule_bond in zip(complete.GetBonds(), kekule.GetBonds()):  # the added bonds come last
        bond.SetBondType(kekule_bond.GetBondType())  # the perceived aromatic flag stays set
    stated_orders = [str(old) in _ORDERS and old == bond.GetBondType() for old, bond in zip(written, kekule.GetBonds())]
    stated_charges = [old == atom.GetFormalCharge() for old, atom in zip(charges, molecule.GetAtoms())]
    return complete, stated_orders, stated_charges, stated_electrons


def _stereo_groups(
    declarations: _Declarations, atoms: int, parities: dict[int, _Parity]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    r"""The AND/OR groups by string atom (a bare ``r`` is one AND group over the tagged atoms outside every group)."""

    members: set[int] = set()
    groups = []
    for label, indices in declarations.groups:
        if any(index >= atoms for index in indices):
            fail("smiles_syntax", _ROW, at=f"CX stereo group {label} names an atom outside the string")
        if members.intersection(indices) or any(index not in parities for index in indices):
            fail("cx_conflicting_declaration", _ROW, declaration=f"stereo group {label}")
        members.update(indices)
        if label != "a":
            groups.append((StereoGroupKind.OR if label.startswith("o") else StereoGroupKind.AND, tuple(indices)))
    ungrouped = tuple(sorted(set(parities) - members))
    if declarations.relative and ungrouped:
        groups.append((StereoGroupKind.AND, ungrouped))
    return tuple(sorted(groups, key=lambda group: min(group[1])))


def _component(
    molecule: Any,
    stated_orders: list[bool],
    stated_charges: list[bool],
    stated_electrons: list[int],
    component_id: str,
    parities: dict[int, _Parity],
    geometry: dict[int, tuple[int, int, bool]],
    declarations: _Declarations,
    structure: _Structure,
    groups: tuple[tuple[int, tuple[int, ...]], ...],
) -> ChemicalComponent:
    r"""The definition of the sanitized, hydrogen-complete molecule with the string's stereo rows and groups."""

    atoms = list(molecule.GetAtoms())
    if any(atom.GetAtomicNum() == 0 for atom in atoms):
        fail("smiles_unsupported", _ROW, feature="dummy atom")
    # per atom: the stated count, else RDKit's computed one on a non-metal atom; a metal's computed count is not
    # consulted (the 0 claims nothing of its electrons); never inferred from a charge
    electrons = [
        stated or (atom.GetNumRadicalElectrons() if atom.GetSymbol() in _NON_METALS else 0)
        for atom, stated in zip(atoms, stated_electrons + [0] * (len(atoms) - len(stated_electrons)))
    ]
    radicals = [(index, count) for index, count in enumerate(electrons) if count]
    if radicals:
        fail("unsupported_radical", _ROW, atoms=tuple(a for a, _ in radicals), electrons=tuple(e for _, e in radicals))
    counts: dict[str, int] = {}
    names = []
    for atom in atoms:
        element = atom.GetSymbol().upper()
        counts[element] = counts.get(element, 0) + 1
        names.append(f"{element}{counts[element]}")
    n = len(atoms)

    bonds = list(molecule.GetBonds())
    if any(str(bond.GetBondType()) not in _ORDERS for bond in bonds):
        fail("smiles_unsupported", _ROW, feature="bond type")
    pairs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bonds]
    bond_of = {frozenset(pair): row for row, pair in enumerate(pairs)}
    stated = stated_orders + [True] * (len(bonds) - len(stated_orders))  # the added hydrogens' bonds
    charge_stated = stated_charges + [True] * (n - len(stated_charges))  # the added hydrogens

    # stereocentres: the string's order with its implicit hydrogen as the added hydrogen atom, then sorted slots
    added_hydrogen = {atom.GetNeighbors()[0].GetIdx(): atom.GetIdx() for atom in atoms[len(structure.tokens) :]}
    centres, parity, slots = [], [], []
    for atom, declared in sorted(parities.items()):
        order = list(declared.order)
        if atom in added_hydrogen:  # a tagged bracket atom carries at most one hydrogen: the -1 slot
            order[order.index(-1)] = added_hydrogen[atom]
        stored = sorted(x for x in order if x >= 0) + [-1] * order.count(-1)
        odd = permutation_parity(np.array([order.index(x) for x in stored]))
        centres.append(atom)
        parity.append(3 - declared.parity if odd else declared.parity)
        slots.append(stored)
    row_of = {atom: row for row, atom in enumerate(centres)}

    unknown = {frozenset(_string_bond(structure, bond)) for bond in declarations.unknown_bonds}
    for _, bond in declarations.wavy:  # every double bond at a wavy bond's atoms
        ends = set(_string_bond(structure, bond))
        unknown |= {frozenset(structure.bonds[b][:2]) for b in geometry if ends & set(structure.bonds[b][:2])}
    stereo_bonds = []
    for bond, (first_reference, second_reference, same_side) in sorted(geometry.items()):
        first, second = structure.bonds[bond][:2]
        if frozenset((first, second)) in unknown:
            continue
        row = bond_of[frozenset((first, second))]
        references = (first_reference, second_reference)
        if pairs[row][0] != first:
            references = references[::-1]
        stereo_bonds.append((row, references, 1 if same_side else 2))

    values = {
        "component_id": component_id,
        "chem_comp_type": None,
        "chem_comp_type_state": _ABSENT,
        "one_letter_code": None,
        "one_letter_code_state": _ABSENT,
        "parents": (),
        "parents_state": _ABSENT,
        "atom_names": tuple(names),
        "element": [atom.GetAtomicNum() for atom in atoms],
        "isotope": [atom.GetIsotope() for atom in atoms],
        "charge": [atom.GetFormalCharge() for atom in atoms],
        "charge_state": [_STATED if known else _DERIVED for known in charge_stated],
        "aromatic": [atom.GetIsAromatic() for atom in atoms],
        "aromatic_state": [_DERIVED] * n,
        "leaving_flag": np.zeros(n),
        "terminal_flag": np.zeros((n, 2)),
        "reference": np.zeros((n, 3)),
        "reference_source": ReferenceSource.NONE,
        "reference_quantum": np.zeros(n),
        "bond_atoms": np.reshape(pairs, (-1, 2)),
        "bond_order": [_ORDERS[str(bond.GetBondType())] for bond in bonds],
        "bond_order_state": [_STATED if known else _DERIVED for known in stated],
        "bond_aromatic": [bond.GetIsAromatic() for bond in bonds],
        "bond_aromatic_state": [_DERIVED] * len(bonds),
        "stereo_centre": centres,
        "stereo_parity": parity,
        "stereo_neighbors": np.reshape(slots, (-1, 4)),
        "stereo_basis": [StereoBasis.STATED] * len(centres),
        "cip_declared": (None,) * n,
        "bond_stereo_bond": [row for row, _, _ in stereo_bonds],
        "bond_stereo_value": [value for _, _, value in stereo_bonds],
        "bond_stereo_reference": np.reshape([references for _, references, _ in stereo_bonds], (-1, 2)),
        "bond_stereo_declared": (None,) * len(stereo_bonds),
        "stereo_group_kind": [kind for kind, _ in groups],
        "stereo_group_ptr": np.cumsum([0, *(len(members) for _, members in groups)]),
        "stereo_group_member": [row_of[atom] for _, members in groups for atom in members],
    }
    return stated_component(values, scope=DefinitionScope.SMILES)


def _records(bonds: list[int]) -> Records:
    r"""One ``cx_incomplete_declaration`` record per bond row (subject none; the string as the reference)."""

    if not bonds:
        return Records.empty()
    count = len(bonds)
    refs = SourceRefs(
        np.zeros(1, dtype=np.int16),
        np.zeros(1, dtype=np.int64),
        np.full(1, -1, dtype=np.int16),
        ("smiles",),
        (),
        None,
    )
    payload = np.zeros((count, 4), dtype=np.int64)
    payload[:, 0] = bonds
    return Records(
        np.full(count, CODE_NAMES.index("cx_incomplete_declaration"), dtype=np.uint16),
        np.zeros(count, dtype=np.uint8),
        np.full(count, -1, dtype=np.int64),
        np.arange(count + 1, dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        refs,
        payload,
        np.zeros(count + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        (),
    )
