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
Exact crystallographic symmetry of one read: operations, settings, codes, frames, site stabilisers and the geometry of
declared edges.

An operation is an integer rotation and a translation in 24ths of a cell, the lattice part kept, so composition,
numbering and the fixed-point question are integer algebra. Floating point enters only where a measured position meets
an image, with every bound stated. Stab(E) is decided without coordinates wherever the algebra decides it; otherwise it
needs a placed endpoint and a known frame, and is undecided, never trivial, without them. An INAPPLICABLE
frame builds no lattice, so no site question arises and Stab(E) is trivial.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import BondKind, Transforms
from ..data.structure import ResolutionKind
from ..errors import CODE_NAMES, CODES, PayloadKind, RequestRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, known, state
from ..reference.space_groups import load_table as _load_space_group_table
from .binding import QuerySource

if TYPE_CHECKING:
    from .tables import CrystalTable, ExperimentTable

#: The stabiliser rule, carried in ``SourceInfo.reader_version``. An operation (R, τ) with fixed points is a member of
#: Stab(E) at a placed E with a known frame [A | u] when the distance r = ‖e‖, e = O·(R·f + τ + n* − f) (f = A·x + u,
#: O = A⁻¹, n* the lattice translation nearest to a fix), is within the inner bound ‖|M − I|·q⃗‖ +
#: ‖|O|·|δA|·|e| + |M − I|·|O|·(|δA|·|x| + |δu|)‖ + 16ε(‖x‖ + ‖O‖_F), with M = O·R·A, q the printed coordinate
#: half-unit and [δA | δu] the half-units of the frame in use (``CrystalFrame.quantum``); not a member at r ≥ 0.74 Å
#: (a versioned engineering constant); undecided between, and whenever the inner bound reaches the outer one.
#: The decided members with the identity must be closed under composition and inversion in the exact algebra, else
#: every numerically decided operation of E is undecided. An endpoint without a position or a known frame has every
#: operation with fixed points undecided.
STABILISER_RULE = "stabiliser/2"

_DEN = 24  # translation numerators are over 24: 1/2, 1/3, 1/4, 1/6, 1/8 and 1/12 are exact
_OUTER_BOUND = 0.74
_CONTACT_BOUND = 5.0  # one rejection bound for every bonding type
_PLACEHOLDER = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)  # the format's cell of coordinates that are not from a crystal
#: The ``_exptl.method`` / ``EXPDTA`` values of the PDBx enumeration naming an experiment without a crystal lattice.
#: X-ray, neutron, fiber and powder diffraction and electron crystallography name one; an unlisted text names neither.
_NON_CRYSTAL = frozenset(
    {
        "ELECTRON MICROSCOPY", "SOLUTION NMR", "SOLID-STATE NMR", "SOLUTION SCATTERING", "EPR",
        "FLUORESCENCE TRANSFER", "INFRARED SPECTROSCOPY", "THEORETICAL MODEL",
    }
)  # fmt: skip
_EPSILON = np.finfo(np.float64).eps
_MAX_COSETS = 192  # the most cosets modulo the lattice any space group has (F m -3 m in its centred cell)
_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
_SYMMETRY_CODE = re.compile(r"([1-9]|[1-9][0-9]|1[0-8][0-9]|19[0-2])(?:_([1-9])([1-9])([1-9]))?")
_VALUE = r"(?:[xyz]|\d+/0*[1-9]\d*|\d+\.\d*|\.\d+|\d+)"  # an axis or a constant (a p/q with q ≠ 0)
_TERM, _COMPONENT = re.compile(rf"([+-]?)({_VALUE})"), re.compile(rf"[+-]?{_VALUE}(?:[+-]{_VALUE})*")
_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_INAPPLICABLE = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_CODE_INDEX = {name: index for index, name in enumerate(CODE_NAMES)}
# the records that put the operation list in force in conflict, or leave it not known in full
_LIST_CONFLICTS = frozenset({"operation_table_vs_setting", "operation_table_not_group", "operation_id_conflict"})
_LIST_UNKNOWN = "operation_table_unparsable"
_NONE, _EDGE, _ATOM = SubjectKind.NONE, SubjectKind.EDGE, SubjectKind.ATOM


@dataclass(frozen=True, slots=True)
class SymOp:
    r"""
    One exact operation x ↦ R·x + t/24 on fractional coordinates.

    ``rotation`` is R row-major; ``translation`` holds the numerators of t over 24 with the lattice part kept, never
    wrapped into the cell. Equal fields are the same operation.

    Examples:
        >>> screw = SymOp((0, -1, 0, 1, 0, 0, 0, 0, 1), (0, 0, 12))  # -y,x,z+1/2
        >>> screw.compose(screw)
        SymOp(rotation=(-1, 0, 0, 0, -1, 0, 0, 0, 1), translation=(0, 0, 24))
        >>> screw.inverse()
        SymOp(rotation=(0, 1, 0, -1, 0, 0, 0, 0, 1), translation=(0, 0, -12))
    """

    rotation: tuple[int, int, int, int, int, int, int, int, int]
    translation: tuple[int, int, int]

    def compose(self, other: SymOp) -> SymOp:
        r"""``self ∘ other``: ``other`` is applied first."""
        moved = _apply(self.rotation, other.translation)
        translation = cast(tuple[int, int, int], tuple(a + b for a, b in zip(moved, self.translation)))
        return SymOp(_multiply(self.rotation, other.rotation), translation)

    def inverse(self) -> SymOp:
        r"""The exact inverse, lattice part included."""
        r = self.rotation
        adjugate = (
            *(r[4] * r[8] - r[5] * r[7], r[2] * r[7] - r[1] * r[8], r[1] * r[5] - r[2] * r[4]),
            *(r[5] * r[6] - r[3] * r[8], r[0] * r[8] - r[2] * r[6], r[2] * r[3] - r[0] * r[5]),
            *(r[3] * r[7] - r[4] * r[6], r[1] * r[6] - r[0] * r[7], r[0] * r[4] - r[1] * r[3]),
        )
        rotation = cast(
            tuple[int, int, int, int, int, int, int, int, int],
            tuple(_determinant(r) * value for value in adjugate),
        )  # det = ±1: the inverse is integral
        translation = cast(tuple[int, int, int], tuple(-value for value in _apply(rotation, self.translation)))
        return SymOp(rotation, translation)


def _determinant(r: Sequence[int]) -> int:
    return r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6])


def _multiply(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, int, int, int, int, int, int, int]:
    return cast(
        tuple[int, int, int, int, int, int, int, int, int],
        tuple(sum(a[3 * i + k] * b[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)),
    )


def _apply(a: Sequence[int], v: Sequence[int]) -> tuple[int, int, int]:
    return cast(tuple[int, int, int], tuple(sum(a[3 * i + k] * v[k] for k in range(3)) for i in range(3)))


_IDENTITY = SymOp((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0))


def _order(rotation: Sequence[int]) -> int:
    r"""The smallest k ≤ 6 with Rᵏ = I; 0 when R is not of crystallographic order."""
    power = tuple(rotation)
    for k in range(1, 7):
        if power == _IDENTITY.rotation:
            return k
        power = _multiply(power, rotation)
    return 0


def _coset(operation: SymOp) -> SymOp:
    r"""The operation modulo lattice translations: its translation wrapped into [0, 24)."""
    return SymOp(operation.rotation, cast(tuple[int, int, int], tuple(value % _DEN for value in operation.translation)))


def _shifted(operation: SymOp, cells: Sequence[int]) -> SymOp:
    translation = cast(tuple[int, int, int], tuple(int(t + _DEN * n) for t, n in zip(operation.translation, cells)))
    return SymOp(operation.rotation, translation)


def triplet(operation: SymOp) -> str:
    r"""The operation as an ``x,y,z`` triplet (record payloads)."""
    parts = []
    for i in range(3):
        row = operation.rotation[3 * i : 3 * i + 3]
        terms = "".join(f"{'+-'[c < 0]}{'' if abs(c) == 1 else abs(c)}{'xyz'[j]}" for j, c in enumerate(row) if c)
        constant = Fraction(operation.translation[i], _DEN)
        parts.append((terms + (f"{'+-'[constant < 0]}{abs(constant)}" if constant else "")).lstrip("+"))
    return ",".join(parts)


def _parse_triplet(text: str) -> SymOp | None:
    r"""An operation triplet (``-x+1/2,y,z``; any case and spacing, terms in any order) as an exact operation. A
    constant is an integer, ``p/q`` or a decimal and must be a whole number of 24ths; each axis appears at most once per
    component with coefficient ±1, and the translation must fit ``Transforms`` (int16). None when the text is not a
    crystallographic operation; the caller records it."""
    components = re.sub(r"\s+", "", text).lower().split(",")
    if len(components) != 3 or not all(map(_COMPONENT.fullmatch, components)):
        return None
    rotation: list[int] = []
    translation: list[int] = []
    for component in components:
        row, constant = [0, 0, 0], Fraction(0)
        for sign, term in _TERM.findall(component):
            if term in ("x", "y", "z") and row["xyz".index(term)]:
                return None
            if term in ("x", "y", "z"):
                row["xyz".index(term)] = -1 if sign == "-" else 1
            else:
                constant += (-1 if sign == "-" else 1) * Fraction(term)
        if (constant * _DEN).denominator != 1:
            return None
        rotation, translation = rotation + row, [*translation, int(constant * _DEN)]
    if not _order(rotation) or max(map(abs, translation)) >= 1 << 15:  # a finite order implies det(R) = ±1
        return None
    return SymOp(tuple(rotation), tuple(translation))  # type: ignore[arg-type]


def parse_symmetry_code(text: str) -> tuple[str, tuple[int, int, int]] | None:
    r"""
    One ``n_klm`` code by the PDBx ``symop`` type: ``([1-9]|[1-9][0-9]|1[0-8][0-9]|19[0-2])(_[1-9][1-9][1-9])?``.

    Returns:
        The operation id as written (resolved against the ids of the table in force, never a row position) and the
        lattice shift ``(k − 5, l − 5, m − 5)`` in cells; a bare ``n`` is ``n_555``. None for a grammar violation.

    Examples:
        >>> parse_symmetry_code("3_655"), parse_symmetry_code("12"), parse_symmetry_code("1555")
        (('3', (1, 0, 0)), ('12', (0, 0, 0)), None)
    """

    match = _SYMMETRY_CODE.fullmatch(text.strip())
    if match is None:
        return None
    n, *digits = match.groups()
    return n, (0, 0, 0) if digits[0] is None else tuple(int(digit) - 5 for digit in digits)  # type: ignore


def _alternative_reading(text: str) -> str | None:
    r"""The grammatical code the same digits spell when the lattice code starts three digits from the end."""
    digits = text.strip().replace("_", "", 1)
    candidate = f"{digits[:-3]}_{digits[-3:]}"
    if re.fullmatch("[0-9]{4,}", digits) and candidate != text.strip() and parse_symmetry_code(candidate):
        return candidate
    return None


def _pack(n: str, shift: Sequence[int]) -> int:
    r"""``Transforms.code`` ``n·1000 + 100k + 10l + m``; −1 for an id that is no number, or a shift beyond ±4
    cells."""
    if not re.fullmatch("[1-9][0-9]{0,5}", n) or any(abs(s) > 4 for s in shift):
        return -1
    return int(n) * 1000 + 100 * (shift[0] + 5) + 10 * (shift[1] + 5) + shift[2] + 5


@dataclass(frozen=True)
class SpaceGroup:
    r"""One setting of the shipped ITA table: its full name, its group number and its operations in ITA order."""

    name: str
    number: int
    operations: tuple[SymOp, ...]


def _setting_key(name: str) -> str:
    r"""A Hermann-Mauguin name up to spacing; never case-folded (lattice and glide letters differ by case)."""
    return " ".join(name.split())


@lru_cache(maxsize=1)
def _settings() -> Mapping[str, SpaceGroup]:
    r"""The shipped settings, keyed by every name and alias; CCP4 numbers alternative settings ``index·1000 + n``."""
    parse = cache(_parse_triplet)  # settings share most of their operators
    index: dict[str, SpaceGroup] = {}
    for entry in _load_space_group_table()["settings"]:
        operations = tuple(map(parse, entry["operators"]))
        group = SpaceGroup(entry["name"], entry["number"] % 1000, operations)  # type: ignore[arg-type]
        for name in (entry["name"], *entry["aliases"]):
            index[_setting_key(name)] = group
    return MappingProxyType(index)


def space_group_setting(name: str) -> SpaceGroup | None:
    r"""The shipped setting a Hermann-Mauguin name (up to spacing) names, ``None`` for a name none is shipped for."""
    return _settings().get(_setting_key(name))


def _closure(operations: tuple[SymOp, ...]) -> tuple[SymOp, ...] | None:
    r"""The cosets modulo the lattice of the group the operations generate, identity first; None when that is not a
    space group (an element of non-crystallographic order, or more cosets than any space group has)."""
    generators = tuple(dict.fromkeys(map(_coset, operations)))
    elements, frontier = {_IDENTITY: None}, [_IDENTITY]
    while frontier:
        found = []
        for product in (_coset(element.compose(generator)) for element in frontier for generator in generators):
            if product not in elements:
                if not _order(product.rotation) or len(elements) == _MAX_COSETS:
                    return None
                elements[product] = None
                found.append(product)
        frontier = found
    return tuple(elements)


@lru_cache(maxsize=None)
def _fixed_point_free(operation: SymOp) -> bool:
    r"""Whether no lattice translate of the operation fixes a point (exact): (R, t + 24n) has one iff
    S·(t + 24n) = 0 with S = Σ_{i<m} Rⁱ (its m-th power is that pure translation, and a finite-order affine map fixes
    the centroid of an orbit), so iff 24 divides S·t and S·n = −S·t/24 has an integer solution."""
    total: tuple[int, ...] = (0,) * 9
    power = _IDENTITY.rotation
    for _ in range(_order(operation.rotation)):
        total = tuple(a + b for a, b in zip(total, power))
        power = _multiply(power, operation.rotation)
    image = _apply(total, operation.translation)
    return any(value % _DEN for value in image) or not _integer_solvable(total, [-value // _DEN for value in image])


def _integer_solvable(matrix: Sequence[int], b: Sequence[int]) -> bool:
    r"""Whether A·n = b (A 3×3 row-major) has an integer solution: A is diagonalised by unimodular row and column
    operations, the row operations applied to b too, and each diagonal entry must divide its entry of b."""
    a, b = [list(matrix[3 * i : 3 * i + 3]) for i in range(3)], list(b)
    for k in range(3):
        while True:
            pivots = [(abs(a[i][j]), i, j) for i in range(k, 3) for j in range(k, 3) if a[i][j]]
            if not pivots:
                return not any(b[k:]) and all(b[i] % a[i][i] == 0 for i in range(k))
            _, i, j = min(pivots)
            a[k], a[i], b[k], b[i] = a[i], a[k], b[i], b[k]
            for row in a:
                row[k], row[j] = row[j], row[k]
            for i in range(k + 1, 3):
                q = a[i][k] // a[k][k]
                a[i], b[i] = [x - q * y for x, y in zip(a[i], a[k])], b[i] - q * b[k]
            for j in range(k + 1, 3):
                q = a[k][j] // a[k][k]
                for row in a:
                    row[j] -= q * row[k]
            if not any(a[i][k] for i in range(k + 1, 3)) and not any(a[k][j] for j in range(k + 1, 3)):
                break
    return all(b[k] % a[k][k] == 0 for k in range(3))


def _records(rows: Sequence[tuple], refs: SourceRefs) -> Records:
    r"""Records from rows ``(code, subject kind, subject, source rows, payload values)``: a TEXT slot takes a str, a
    source row −1 is dropped."""
    texts: dict[str, int] = {}
    payload = np.zeros((len(rows), 4), dtype=np.int64)
    for i, (code, _, _, _, values) in enumerate(rows):
        for j, ((_, kind), value) in enumerate(zip(CODES[code].payload, values, strict=True)):
            payload[i, j] = texts.setdefault(value, len(texts)) if kind == PayloadKind.TEXT else value
    sources = [[int(ref) for ref in row[3] if ref >= 0] for row in rows]
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(row) for row in sources], dtype=np.int64)
    empty = np.zeros(len(rows) + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    subject_kind = np.array([row[1] for row in rows], dtype=np.uint8)
    subject = np.array([row[2] for row in rows], dtype=np.int64)
    code = np.array([_CODE_INDEX[row[0]] for row in rows], dtype=np.uint16)
    flat = np.array([ref for row in sources for ref in row], dtype=np.int64)
    return Records(code, subject_kind, subject, ptr, flat, refs, payload, *empty, tuple(texts))


@dataclass(frozen=True, eq=False)
class CrystalFrame:
    r"""
    The crystal frame of one read: the affine map [A | u] (``fractionalisation``, 3×4) from
    Cartesian Å to fractional coordinates, f = A·x + u, and ``orthogonalisation`` O = A⁻¹; an operation (R, τ) moves x
    to O·[R·(A·x + u) + τ − u], u never dropped. NaN unless KNOWN or INAPPLICABLE by its transform.

    ``state``: KNOWN×DECLARED (the source's transform: ``_atom_sites.fract_transf_*``, ``SCALEn``, or the inverse of a
    lone ``_atom_sites.Cartn_transf_*``), KNOWN×DERIVED (none stated: [A | 0] of the cell by the PDB convention, **a**
    along x, **b** in the xy plane), INAPPLICABLE×DEFINED (``placeholder_frame`` on the cell, or with no cell known
    ``placeholder_transform`` on the source's transform, which it keeps with its half-units: no lattice, no crystal
    operation, no site question), UNKNOWN×NONE or CONFLICT×NONE. ``quantum``: the half-units of [A | u] every
    bound uses but the cell's metric check (which relaxes them, ``_coarsest``), those of the matrix in use — printed
    (a printed 0 too), or those of the statement it is derived from (the cell, a ``Cartn_transf``) propagated.
    ``refs``: what an INAPPLICABLE frame rests on (the cell, the transform and the methods; rows of the read's
    references). ``cell_spread``: the half-units of A a stated cell admits beside a stated transform, the cell's
    relaxed range (``_coarsest``) propagated (``_derived_quantum``), zero otherwise: wwPDB computes the
    deposited operator matrices from the cell, so the operator cross-check (``io.assemblies``) adds it to ``quantum``;
    no stabiliser decision uses it.
    """

    fractionalisation: NDArray[np.float64]
    orthogonalisation: NDArray[np.float64]
    state: int
    quantum: NDArray[np.float64]
    refs: tuple[int, ...]
    cell_spread: NDArray[np.float64]


def _orthogonalisation(cell: Sequence[float]) -> NDArray[np.float64] | None:
    r"""The orthogonalisation of a, b, c (Å), α, β, γ (degrees) by the PDB convention; None when the values describe no
    cell."""
    a, b, c, alpha, beta, gamma = cell
    if min(a, b, c) <= 0 or not all(0 < angle < 180 for angle in (alpha, beta, gamma)):
        return None
    cos_a, cos_b, cos_g = np.cos(np.deg2rad([alpha, beta, gamma]))
    sin_g = np.sin(np.deg2rad(gamma))
    determinant = 1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g
    if determinant <= 0:
        return None
    column_c = [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, c * np.sqrt(determinant) / sin_g]
    return np.array([[a, b * cos_g, column_c[0]], [0.0, b * sin_g, column_c[1]], [0.0, 0.0, column_c[2]]])


def placeholder_frame(cell: Sequence[float], methods: Sequence[str]) -> bool:
    r"""
    Whether the crystal frame is inapplicable: ``cell`` is the format's placeholder, a = b = c = 1 Å and
    α = β = γ = 90° as stated, and every stated experimental method names an experiment without a crystal lattice
    (electron microscopy, NMR, ...). Neither alone: no method stated, or one crystal method (X-ray beside NMR), is a
    crystal.

    Examples:
        >>> placeholder = (1, 1, 1, 90, 90, 90)
        >>> placeholder_frame(placeholder, ["ELECTRON MICROSCOPY"]), placeholder_frame(placeholder, [])
        (True, False)
    """
    stated = [method.strip().upper() for method in methods]
    return tuple(map(float, cell)) == _PLACEHOLDER and bool(stated) and all(m in _NON_CRYSTAL for m in stated)


def cell_state(crystal: CrystalTable) -> int:
    r"""The state of the stated cell, a fact of its own beside the frame: KNOWN×DECLARED for six values describing a
    cell, CONFLICT×NONE for six describing none, UNKNOWN×NONE for zero lengths (which declare no crystal) or a cell in
    part, else the state its six items share (absent, ``?``, ``.``)."""
    states, cell = crystal.cell_state, crystal.cell
    if known(states[:3]).all() and (cell[:3] == 0).all():
        return _UNKNOWN
    if known(states).all():
        return _KNOWN_DECLARED if _orthogonalisation(cell) is not None else _CONFLICT
    return int(states[0]) if not known(states).any() and (states == states[0]).all() else _UNKNOWN


def _undetermined(resolved: int, refs: tuple[int, ...] = ()) -> CrystalFrame:
    nan = np.full((3, 4), np.nan)
    return CrystalFrame(nan, nan[:, :3].copy(), resolved, np.zeros((3, 4)), refs, np.zeros((3, 3)))


def _inverse(transform: NDArray[np.float64], quantum: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    r"""The inverse [L⁻¹ | −L⁻¹·t] of an affine map [L | t] with its half-units to first order: |L⁻¹|·δL·|L⁻¹| and
    |L⁻¹|·δL·|L⁻¹|·|t| + |L⁻¹|·δt."""
    inverse = np.linalg.inv(transform[:, :3])
    spread = np.abs(inverse) @ quantum[:, :3] @ np.abs(inverse)
    origin = spread @ np.abs(transform[:, 3]) + np.abs(inverse) @ quantum[:, 3]
    return np.column_stack([inverse, -inverse @ transform[:, 3]]), np.column_stack([spread, origin])


def _derived_quantum(cell: NDArray[np.float64], quantum: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""The half-units of A = O(cell)⁻¹: Σᵢ |∂A/∂pᵢ|·qᵢ over the six cell parameters and their printed half-units
    (central differences); inf where a step leaves the cells."""
    total = np.zeros((3, 3))
    for i in np.flatnonzero(quantum):
        step, up, down = 1e-6 * max(1.0, abs(cell[i])), cell.copy(), cell.copy()
        up[i], down[i] = cell[i] + step, cell[i] - step
        high, low = _orthogonalisation(up), _orthogonalisation(down)
        if high is None or low is None:
            return np.full((3, 3), np.inf)
        total += np.abs(np.linalg.inv(high) - np.linalg.inv(low)) / (2 * step) * quantum[i]
    return total


_PARAMETERS = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")


def _parameters(orthogonalisation: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""a, b, c (Å) and α, β, γ (degrees) of the metric OᵀO: what a frame states of the cell, its origin and
    orientation aside."""
    metric = orthogonalisation.T @ orthogonalisation
    lengths = np.sqrt(np.diag(metric))
    cosines = [metric[1, 2] / (lengths[1] * lengths[2]), metric[0, 2] / (lengths[0] * lengths[2])]
    cosines.append(metric[0, 1] / (lengths[0] * lengths[1]))
    return np.concatenate([lengths, np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))])


def placeholder_transform(fractionalisation: NDArray[np.float64], methods: Sequence[str]) -> bool:
    r"""
    ``placeholder_frame`` on the metric of a transform [A | u] (right-handed): the lengths and angles of BᵀB
    (B = A⁻¹), u aside — the placeholder cell a source states through the matrix when it states no cell.

    Examples:
        >>> placeholder_transform(np.eye(3, 4), ["SOLUTION NMR"]), placeholder_transform(np.eye(3, 4) / 1000, ["EPR"])
        (True, False)
    """
    return placeholder_frame(_parameters(np.linalg.inv(fractionalisation[:, :3])), methods)


def _metric_spread(linear: NDArray[np.float64], quantum: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Σ |∂p/∂Aⱼₖ|·δAⱼₖ of the six metric parameters of A⁻¹ over the coefficients of A and their half-units (central
    differences)."""
    total, step = np.zeros(6), 1e-7 * np.abs(linear).max()
    for j, k in zip(*np.nonzero(quantum)):
        up, down = linear.copy(), linear.copy()
        up[j, k], down[j, k] = linear[j, k] + step, linear[j, k] - step
        moved = _parameters(np.linalg.inv(up)) - _parameters(np.linalg.inv(down))
        total += np.abs(moved) / (2 * step) * quantum[j, k]
    return total


def _coarsest(value: NDArray[np.float64], quantum: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    The relaxed rounding range of printed numbers in the cell's metric check, and in that check only:
    ``quantum`` (half the printed unit) ten times over per trailing zero after the decimal point, never into the
    integer part (112.260 → 0.005, 90.000 → 0.5); a printed 0, with no digit to strip back to, keeps its printed
    half-unit. This is a conservative compatibility-check policy for how far incompatibility must be proven, not a
    factual claim that trailing zeros carry no precision: no ``frame_conflict`` means incompatibility cannot be proven
    within this range, never that the cell and the matrix agree exactly. The nominal values and the [A | u] in use
    stay as stated. The stabiliser decisions and every other bound (the corner check, those on
    ``CrystalFrame.quantum``) keep the printed half-unit. Printed digits are representation resolution, not
    experimental standard uncertainty, which CIF states separately (``_cell.*_esd``, a parenthesised su).
    """
    unit = 2 * quantum
    place = np.rint(np.log10(np.where(unit > 0, unit, 1.0)))
    digits, stripped = np.rint(np.abs(value) / 10.0**place), place.copy()
    while (zero := (stripped < 0) & (digits != 0) & (np.fmod(digits, 10) == 0)).any():
        digits, stripped = np.where(zero, digits / 10, digits), stripped + zero
    return quantum * 10.0 ** (stripped - place)


def _judged(log: list[tuple], items: Sequence[str], measured, allowed, refs: tuple[int, ...]) -> bool:
    r"""Whether a measured difference exceeds its bound anywhere; one ``frame_conflict`` naming the worst one, in
    thousandths of its unit, the difference rounded up and the bound down, so the record shows the one beyond the
    other."""
    worst = int(np.argmax(np.asarray(measured) / np.asarray(allowed)))
    if measured[worst] <= allowed[worst]:
        return False
    values = (items[worst], int(np.ceil(measured[worst] * 1000)), int(np.floor(allowed[worst] * 1000)))
    log.append(("frame_conflict", _NONE, -1, refs, values))
    return True


def _frame(crystal: CrystalTable, methods: Sequence[tuple[str, int]], log: list[tuple]) -> CrystalFrame | None:
    r"""
    The frame of the read, its records logged; ``methods`` the stated experimental methods
    with their references. The placeholder cell in a non-crystal experiment makes it INAPPLICABLE, whatever else is
    stated. Otherwise a stated transform is the frame — the ``fract`` group, else the inverse of the ``Cartn`` group;
    each stated group complete (else ``frame_incomplete``) with a linear part of positive determinant (else
    ``frame_conflict``), never replaced by the cell's — checked against the ``Cartn`` group as its inverse at the unit
    cell's corners, and against the cell through the metric alone: the lengths and angles of BᵀB (B = A⁻¹), each
    within the cell's half-unit plus the matrix's half-units propagated, both relaxed to their last non-zero decimal
    (``_coarsest``: how far incompatibility must be proven; no ``frame_conflict`` is no proof of exact
    agreement), where the corner check takes the printed half-units; the values themselves are used as stated.
    With no cell known (``cell_state``: absent, ``?`` or ``.``, in part, or of zero lengths), a transform whose metric
    is the placeholder in a non-crystal experiment (``placeholder_transform``) makes it INAPPLICABLE too, the transform
    kept. Only when no transform is stated is the frame derived from the cell, by the PDB convention; a stated
    ``Cartn_transform_axes`` (free text, not interpreted) leaves it UNKNOWN. None when no cell value and no transform
    is stated (zero lengths declare no crystal).
    """
    cell, stated, cell_known = crystal.cell, known(crystal.cell_state), cell_state(crystal)
    groups = [g for g in range(2) if known(crystal.deposited_state[g]).any()]
    refs = (crystal.cell_ref, *crystal.deposited_ref[groups].tolist())
    texts = [text for text, _ in methods]
    if cell_known == _KNOWN_DECLARED and placeholder_frame(cell, texts):
        return _undetermined(_INAPPLICABLE, (crystal.cell_ref, *(ref for _, ref in methods)))
    if stated[:3].all() and (cell[:3] == 0).all():
        stated = np.zeros(6, dtype=bool)  # zero lengths declare no crystal
    if not groups:
        if not stated.any():
            return None
        if cell_known == _CONFLICT:
            log.append(("frame_conflict", _NONE, -1, refs[:1], ("", -1, -1)))
            return _undetermined(_CONFLICT)
        if not stated.all() or crystal.axes_ref >= 0:  # in part, or by an axis convention not interpreted
            log.append(("frame_incomplete", _NONE, -1, (crystal.cell_ref, crystal.axes_ref), ()))
            return _undetermined(_UNKNOWN)
        orthogonalisation = _orthogonalisation(cell)
        derived = np.column_stack([np.linalg.inv(orthogonalisation), np.zeros(3)])
        quantum = np.column_stack([_derived_quantum(cell, crystal.cell_quantum), np.zeros(3)])
        return CrystalFrame(derived, orthogonalisation, _KNOWN_DERIVED, quantum, (), np.zeros((3, 3)))

    found = len(log)
    for g, ref in zip(groups, refs[1:]):
        matrix = crystal.deposited_frame[g]
        if not known(crystal.deposited_state[g]).all():
            log.append(("frame_incomplete", _NONE, -1, (refs[0], ref), ()))
        elif not np.linalg.det(matrix[:, :3]) > 0:  # singular, or a left-handed lattice basis
            log.append(("frame_conflict", _NONE, -1, (refs[0], ref), ("", -1, -1)))
    if len(log) > found:
        conflict = any(row[0] == "frame_conflict" for row in log[found:])
        return _undetermined(_CONFLICT if conflict else _UNKNOWN)
    stated_frame, quanta = crystal.deposited_frame, crystal.deposited_quantum
    if groups[0] == 0:
        fractionalisation, quantum = stated_frame[0], quanta[0]
    else:
        fractionalisation, quantum = _inverse(stated_frame[1], quanta[1])
    orthogonalisation = np.linalg.inv(fractionalisation[:, :3])
    conflict, cell_spread = False, np.zeros((3, 3))
    if groups == [0, 1]:  # the Cartn group is the inverse of [A | u]: the unit cell's corners land alike
        inverse, spread = _inverse(fractionalisation, quantum)
        cartn = stated_frame[1]
        moved = _CORNERS @ (cartn[:, :3] - inverse[:, :3]).T + cartn[:, 3] - inverse[:, 3]
        allowed = _CORNERS @ (quanta[1][:, :3] + spread[:, :3]).T + quanta[1][:, 3] + spread[:, 3]
        scale = np.linalg.norm(_CORNERS @ cartn[:, :3].T + cartn[:, 3], axis=1)
        allowed = np.linalg.norm(allowed, axis=1) + 16 * _EPSILON * (scale + 1)
        corners = ("Cartn_transf",) * len(_CORNERS)
        conflict = _judged(log, corners, np.linalg.norm(moved, axis=1), allowed, (refs[1], refs[2]))
    if cell_known == _CONFLICT:
        log.append(("frame_conflict", _NONE, -1, refs[:2], ("", -1, -1)))
        conflict = True
    elif cell_known == _KNOWN_DECLARED:  # the metric alone: an origin or an orientation is free
        stated, g = _parameters(orthogonalisation), groups[0]
        coarse = _coarsest(stated_frame[g], quanta[g])
        coarse = coarse if g == 0 else _inverse(stated_frame[1], coarse)[1]
        relaxed = _coarsest(cell, crystal.cell_quantum)
        allowed = relaxed + _metric_spread(fractionalisation[:, :3], coarse[:, :3]) + 16 * _EPSILON * np.abs(stated)
        conflict |= _judged(log, _PARAMETERS, np.abs(stated - cell), allowed, refs[:2])
        cell_spread = _derived_quantum(cell, relaxed)
    if conflict:
        return _undetermined(_CONFLICT)
    if not known(np.uint8(cell_known)) and placeholder_transform(fractionalisation, texts):  # 1ho7: A = I, no _cell
        resting = (*refs, *(ref for _, ref in methods))
        return CrystalFrame(fractionalisation.copy(), orthogonalisation, _INAPPLICABLE, quantum, resting, cell_spread)
    return CrystalFrame(fractionalisation.copy(), orthogonalisation, _KNOWN_DECLARED, quantum, (), cell_spread)


@dataclass(frozen=True, eq=False)
class Symmetry:
    r"""The operation list in force of one read, its setting, frame and records. ``numbering``: ``"file"`` (the rows
    of the file's table, every one decoded, its ids in row order; an id which names two operations is not among them),
    ``"ita"`` (the resolved setting, ids ``"1"…"N"``; a wwPDB entry's table of its operations too) or
    ``"none"`` (no id resolves; ``operations`` the identity when
    nothing is declared, empty when a space group, a lattice or a table is stated but neither a setting resolves nor
    the table is in force: a row of it states no operation, or every id names two).
    ``operation_refs``: per operation its row of ``records.refs``, −1 for ITA. ``disagreement``: the file ids
    whose operation is not in the setting's set. ``space_group``, ``space_group_state``: the H-M name its statements
    decide (a ``space_group`` symbol, −1 unless KNOWN); ``space_group_ref``: the statement that decides it, and so the
    setting (its row of ``records.refs``, −1 none). ``contested``: the statements of the setting contradict one
    another (``space_group_name_vs_number``, ``space_group_restatement_conflict``) and no file table governs, so every
    code resolved, and every stabiliser decided, through the setting is CONFLICT."""

    operations: tuple[SymOp, ...]
    operation_ids: tuple[str, ...]
    numbering: Literal["file", "ita", "none"]
    setting: SpaceGroup | None
    frame: CrystalFrame | None
    disagreement: frozenset[str]
    records: Records
    operation_refs: tuple[int, ...]
    space_group: int
    space_group_state: int
    space_group_ref: int
    contested: bool


def _space_group(crystal: CrystalTable, log: list[tuple]) -> tuple:
    r"""The space group of the read from every statement of it (``_symmetry`` and ``_space_group``; ``CRYST1``): an
    unstated one (``?``, ``.``, absent) agrees with any; stated H-M names agree when they name one setting (an alias
    included); names of two settings, or two numbers of two groups (modulo 1000), are
    ``space_group_restatement_conflict``; names that do not all resolve cannot be compared, and leave the setting
    unresolved and the name Unknown. Returns the name (the one an unresolved setting reports), its setting, the
    number, the name and number items deciding them, the name as the value states it (code, state, item) and whether
    the statements contradict one another."""
    told, counted = np.flatnonzero(known(crystal.space_group_state)), np.flatnonzero(known(crystal.number_state))
    names = [crystal.symbols["space_group"][c] for c in crystal.space_group[told].tolist()]
    settings = [_settings().get(_setting_key(name)) for name in names]
    numbers = crystal.number[counted].tolist()
    contradicted, value = False, (-1, _UNKNOWN, -1)
    if not names:
        name, setting = None, None
        present = np.flatnonzero(crystal.space_group_state != _UNKNOWN)  # stated as ? or .: its state
        if len(present):
            value = (-1, int(crystal.space_group_state[present[0]]), int(crystal.space_group_ref[present[0]]))
    elif len(set(map(_setting_key, names))) == 1 or (None not in settings and len(set(settings)) == 1):
        name, setting, i = names[0], settings[0], told[0]
        value = (int(crystal.space_group[i]), int(crystal.space_group_state[i]), int(crystal.space_group_ref[i]))
    elif None not in settings:  # two settings
        k = next(k for k, other in enumerate(settings) if other != settings[0])
        basis = crystal.space_group_ref[told[[0, k]]].tolist()
        log.append(("space_group_restatement_conflict", _NONE, -1, basis, (names[0], names[k])))
        name, setting, contradicted, value = None, None, True, (-1, _CONFLICT, -1)
    else:  # an unresolved name may name any setting
        name, setting = names[settings.index(None)], None
    number = numbers[0] if numbers else None
    if any(other % 1000 != numbers[0] % 1000 for other in numbers):
        k = next(k for k, other in enumerate(numbers) if other % 1000 != numbers[0] % 1000)
        basis = crystal.number_ref[counted[[0, k]]].tolist()
        log.append(("space_group_restatement_conflict", _NONE, -1, basis, (str(number), str(numbers[k]))))
        number, contradicted = None, True
    named = (value[2], int(crystal.number_ref[counted[0]]) if numbers else -1)  # the statements deciding them
    return name, setting, number, named, value, contradicted


def symmetry(crystal: CrystalTable, experiment: ExperimentTable, refs: SourceRefs) -> Symmetry:
    r"""
    The operation list in force, its setting and its frame.

    The file's own table governs by its ids; else the setting of the full H-M name (never ``Int_Tables_number``, which
    only cross-checks) completes in ITA order. A file table is compared with a resolved, uncontested setting as coset
    sets, else must itself be a group. A wwPDB entry (marked by ``CrystalTable.wwpdb_ref``, a recognised archive site)
    numbers its codes, and the ``n_klm`` names of its crystal ``_pdbx_struct_oper_list`` rows, in the setting's ITA
    order: a complete table of the setting's operations, one row per coset in any order, then only cross-checks, and the
    setting governs (the operator ids ``oper_expression`` names are another id space, never renumbered). The
    marker is not proof that the file is unmodified, so every other condition still applies. Any other file keeps its
    own ids: a legal table may number the same operations otherwise, its references with it. The table is one statement:
    a row of it that states no operation (``operation_table_unparsable``) puts none of it in force, so numbering is
    ``none`` and every code naming one of its ids is Unknown; the stated table still governs its ids, so no setting
    replaces it. The space group stays as stated under an INAPPLICABLE frame, which states no
    lattice. ``crystal``'s symbol families are ``space_group``, ``operation_id``, ``operation_text`` and ``text`` (the
    methods of ``experiment``); its ref columns index ``refs``, the read's references, which the records use.
    """
    log: list[tuple] = []
    methods = np.flatnonzero((experiment.resolution_kind == ResolutionKind.NONE) & (experiment.method >= 0))
    stated_methods = [(crystal.symbols["text"][experiment.method[k]], int(experiment.ref[k])) for k in methods]
    framing: list[tuple] = []
    frame = _frame(crystal, stated_methods, framing)
    stating = np.concatenate([crystal.space_group_state, crystal.number_state])
    group = bool((stating != _UNKNOWN).any())  # stated, even as ? or .
    tabled = len(crystal.operation_id) > 0  # the file states a table, whether or not a row of it is decoded
    # a crystal stated without its operations; an INAPPLICABLE frame states none
    lattice = frame is not None and frame.state != _INAPPLICABLE and not tabled
    stated: dict[str, tuple[SymOp, int]] = {}
    conflicted: set[str] = set()
    unparsed: set[str | None] = set()  # ids a row of which states no operation: their operation is Unknown
    unstated: dict[str, int] = {}  # id -> its first row stating no operation, which agrees with any
    for i, t, ref in zip(crystal.operation_id, crystal.operation_text, crystal.operation_ref.tolist()):
        text = crystal.symbols["operation_text"][t] if t >= 0 else ""
        id_ = crystal.symbols["operation_id"][i] if i >= 0 else None
        if t < 0 and id_ is not None:
            unstated.setdefault(id_, ref)
            continue
        if (operation := _parse_triplet(text)) is None or id_ is None:
            log.append(("operation_table_unparsable", _NONE, -1, (ref,), (id_ or "", text)))
            unparsed.add(id_)
            continue
        if id_ in stated and stated[id_][0] != operation:
            first, at = stated[id_]
            log.append(("operation_id_conflict", _NONE, -1, (at, ref), (id_, triplet(first), triplet(operation))))
            conflicted.add(id_)
        stated.setdefault(id_, (operation, ref))
    for id_, ref in unstated.items():  # no row states the id's operation
        if id_ not in stated and id_ not in unparsed:
            log.append(("operation_table_unparsable", _NONE, -1, (ref,), (id_, "")))
            unparsed.add(id_)
    stated = {} if unparsed else {id_: row for id_, row in stated.items() if id_ not in conflicted}
    complete = not conflicted and not unparsed

    name, setting, number, named, value, contested = _space_group(crystal, log)
    space_group, space_group_state, space_group_ref = value
    if setting is None and (group or lattice) and space_group_state != _CONFLICT:
        unresolved = (-1, name or "", -1 if number is None else number, "")
        items = (*crystal.space_group_ref.tolist(), *crystal.number_ref.tolist())  # every statement
        basis = items if group else (crystal.cell_ref, *crystal.deposited_ref.tolist())
        log.append(("space_group_setting_unresolved", _NONE, -1, basis, unresolved))
    elif setting is not None and number is not None and number % 1000 != setting.number:
        log.append(("space_group_name_vs_number", _NONE, -1, named, (name, number, setting.number)))
        contested = True

    disagreement: set[str] = set()
    operations = tuple(row[0] for row in stated.values())
    operation_refs = tuple(row[1] for row in stated.values())
    numbering: Literal["file", "ita", "none"] = "file"
    ids = tuple(stated)
    if stated and setting is not None and not contested:
        ita, listed = setting.operations, {_coset(operation) for operation in operations}
        cosets = {_coset(operation) for operation in ita}
        for id_, (operation, ref) in stated.items():
            if _coset(operation) not in cosets:
                disagreement.add(id_)
                same = triplet(ita[int(id_) - 1]) if id_.isdecimal() and 0 < int(id_) <= len(ita) else ""
                vs = (-1, triplet(operation), same, setting.name)
                log.append(("operation_table_vs_setting", _NONE, -1, (ref, named[0]), vs))
        for coset in dict.fromkeys(c for c in map(_coset, ita) if c not in listed) if complete else ():
            vs = (-1, "", triplet(coset), setting.name)
            log.append(("operation_table_vs_setting", _NONE, -1, (named[0],), vs))
        if crystal.wwpdb_ref >= 0 and complete and len(operations) == len(ita) and listed == cosets:
            # wwPDB numbers its codes in the setting's ITA order; a table of its operations, one row each in
            # another order (the cctbx order PHENIX writes, 6efz's centred rows unreduced), only cross-checks them
            numbering, operations = "ita", ita
            ids, operation_refs = tuple(str(k) for k in range(1, len(ita) + 1)), (-1,) * len(ita)
    elif stated:
        listed = {_coset(operation) for operation in operations}
        if complete and set(closure := _closure(operations) or ()) != listed:
            log.append(("operation_table_not_group", _NONE, -1, operation_refs, (len(listed), len(closure) or -1)))
    elif setting is not None and not tabled:  # a stated table governs its ids even when no row of it is decoded
        numbering, operations = "ita", setting.operations
        ids, operation_refs = tuple(str(k) for k in range(1, len(operations) + 1)), (-1,) * len(operations)
    else:
        numbering, ids, operations = "none", (), () if group or lattice or tabled else (_IDENTITY,)
        operation_refs = (-1,) * len(operations)
    records = _records(log + framing, refs)
    return Symmetry(
        operations, ids, numbering, setting, frame, frozenset(disagreement), records, operation_refs,
        space_group, space_group_state, space_group_ref, contested and not tabled,
    )  # fmt: skip


@dataclass(frozen=True, eq=False)
class StabiliserTable:
    r"""Stab(E) per endpoint that needs it, placed or not, sorted by ``endpoint``. Decided members (CSR
    ``ptr``) are rows k of ``Symmetry.operations`` with a lattice ``shift``: exactly ``(R_k, t_k + 24·shift)``;
    the identity is one unless the list in force is empty. ``undecided`` (CSR ``undecided_ptr``) rows have unknown
    membership. ``state``: KNOWN×DERIVED (all decided), UNKNOWN×NONE (some not, or the list in force not known in
    full), CONFLICT×NONE (list in conflict, or E placed under a CONFLICT frame). ``records`` name
    endpoint ids (subject ATOM); the caller remaps them."""

    endpoint: NDArray[np.int32]
    ptr: NDArray[np.int64]
    operation: NDArray[np.int32]
    shift: NDArray[np.int32]
    undecided_ptr: NDArray[np.int64]
    undecided: NDArray[np.int32]
    state: NDArray[np.uint8]
    records: Records


def _list_status(symmetry: Symmetry) -> tuple[str, tuple[SymOp, ...]]:
    r"""How the list in force decides stabilisers (``conflict``, ``unknown``, ``trivial``, ``group``), and the cosets
    with fixed points whose membership needs a position."""
    found = symmetry.records.counts().keys()
    if found & _LIST_CONFLICTS or symmetry.contested:
        return "conflict", ()
    if symmetry.frame is not None and symmetry.frame.state == _INAPPLICABLE:
        return "trivial", ()  # no lattice, so no site question
    cosets = tuple(dict.fromkeys(map(_coset, symmetry.operations)))
    fixed = tuple(c for c in cosets if c != _IDENTITY and not _fixed_point_free(c))
    if not cosets or _LIST_UNKNOWN in found:
        return "unknown", fixed
    return ("group" if fixed else "trivial"), fixed  # a list in force that is no group carries a conflict record


def stabilisers(
    symmetry: Symmetry,
    endpoints: NDArray[np.int32],
    sites: NDArray[np.float64],
    placed: NDArray[np.bool_],
    quantum: NDArray[np.float64],
) -> StabiliserTable:
    r"""
    Stab(E) of each endpoint (unique ids) under the operation list in force (``STABILISER_RULE``).

    ``sites`` are Cartesian Å, read where ``placed``; ``quantum`` is the printed coordinate half-unit in Å (the largest
    of the three tokens). One ``special_position_unplaced`` per endpoint undecided for want of a position or a frame
    (a placed one under a CONFLICT frame is CONFLICT, its record the frame's), one ``special_position_undecided`` per
    placed endpoint with an undecided operation (``group_property_failed`` 0 | 1).
    """
    e = len(endpoints)
    order = np.argsort(endpoints, kind="stable")
    endpoints, sites, placed, quantum = endpoints[order], sites[order], placed[order], quantum[order]

    status, fixed = _list_status(symmetry)
    rows = {coset: row for row, coset in reversed(list(enumerate(map(_coset, symmetry.operations))))}  # first row wins
    candidates = [rows.get(_IDENTITY, -1), *(rows[coset] for coset in fixed)]
    member, shift = np.zeros((e, len(candidates)), dtype=bool), np.zeros((e, len(candidates), 3), dtype=np.int64)
    if candidates[0] >= 0:
        member[:, 0], shift[:, 0] = True, [-value // _DEN for value in symmetry.operations[candidates[0]].translation]
    undecided, failed = np.zeros_like(member), np.zeros(e, dtype=bool)
    undecided[:, 1:] = status == "unknown"
    frame, unplaced, clash = symmetry.frame, np.zeros(e, dtype=bool), np.zeros(e, dtype=bool)
    if status == "group":  # a placed E under a CONFLICT frame inherits the conflict, not an unknown
        framing = _UNKNOWN if frame is None else frame.state
        unplaced, clash = ~placed | (framing == _UNKNOWN), placed & (framing == _CONFLICT)
        undecided[unplaced | clash, 1:] = True
    if status == "group" and len(at := np.flatnonzero(~unplaced & ~clash)):
        active_frame = cast(CrystalFrame, frame)  # An absent frame makes every endpoint unplaced above.
        o, a, u = (
            active_frame.orthogonalisation,
            active_frame.fractionalisation[:, :3],
            active_frame.fractionalisation[:, 3],
        )
        delta_a, delta_u = active_frame.quantum[:, :3], active_frame.quantum[:, 3]
        x = sites[at]
        fractional = x @ a.T + u
        floating = 16 * _EPSILON * (np.linalg.norm(x, axis=1) + np.linalg.norm(o))
        reach = np.abs(x) @ delta_a.T + delta_u  # |δA|·|x| + |δu|: how far the frame's rounding moves f
        for k, coset in enumerate(fixed, start=1):  # per coset over the placed endpoints
            rotation = np.array(coset.rotation, dtype=np.float64).reshape(3, 3)
            d = fractional @ rotation.T + np.array(coset.translation) / _DEN - fractional
            nearest = -np.rint(d)
            residual = (d + nearest) @ o.T  # e, Cartesian
            r = np.linalg.norm(residual, axis=1)
            excess = np.abs(o @ rotation @ a - np.eye(3))
            inner = quantum[at] * np.linalg.norm(excess.sum(axis=1)) + floating
            rounding = np.abs(residual) @ (np.abs(o) @ delta_a).T + reach @ (excess @ np.abs(o)).T  # the frame's
            inner += np.linalg.norm(rounding, axis=1)
            member[at, k] = (inner < _OUTER_BOUND) & (r <= inner)
            undecided[at, k] = ~member[at, k] & ~((inner < _OUTER_BOUND) & (r >= _OUTER_BOUND))
            lattice = (np.array(coset.translation) - symmetry.operations[candidates[k]].translation) // _DEN
            shift[at, k] = nearest.astype(np.int64) + lattice  # relative to the stated row, not to the coset
        failed[at] = _group_property_fails(symmetry, candidates, member[at], shift[at])
        undecided[failed, 1:], member[failed, 1:] = True, False

    resolved = {"conflict": _CONFLICT, "unknown": _UNKNOWN}.get(status)
    states = np.where(undecided.any(axis=1), _UNKNOWN, _KNOWN_DERIVED) if resolved is None else np.full(e, resolved)
    states[clash] = _CONFLICT
    reported = np.flatnonzero(undecided.any(axis=1) & (status == "group") & ~clash)
    log: list[tuple] = [("special_position_unplaced", _ATOM, endpoints[i], (), ()) for i in reported if unplaced[i]]
    log += [
        ("special_position_undecided", _ATOM, endpoints[i], (), (int(failed[i]),)) for i in reported if not unplaced[i]
    ]
    log.sort(key=lambda row: row[2])
    member_rows, undecided_rows = _csr(member, candidates), _csr(undecided, candidates)
    shifts, records = shift[member].astype(np.int32), _records(log, symmetry.records.refs)
    return StabiliserTable(endpoints, *member_rows, shifts, *undecided_rows, states.astype(np.uint8), records)


def _group_property_fails(
    symmetry: Symmetry, candidates: Sequence[int], member: NDArray[np.bool_], shift: NDArray[np.int64]
) -> NDArray[np.bool_]:
    r"""Per endpoint, whether its decided members with the identity fail closure under composition and inversion in the
    exact algebra; endpoints sharing members and shifts are checked once."""
    fails = np.zeros(len(member), dtype=bool)
    keys = np.concatenate([member, np.where(member[..., None], shift, 0).reshape(len(member), -1)], axis=1)
    patterns, inverse = np.unique(keys, axis=0, return_inverse=True)
    for index, pattern in enumerate(patterns):
        offsets, chosen = pattern[len(candidates) :].reshape(-1, 3), np.flatnonzero(pattern[1 : len(candidates)]) + 1
        group = {_IDENTITY} | {_shifted(symmetry.operations[candidates[k]], offsets[k]) for k in chosen}
        closed = all(a.compose(b) in group for a in group for b in group) and all(a.inverse() in group for a in group)
        fails[inverse.ravel() == index] = not closed
    return fails


def _csr(mask: NDArray[np.bool_], candidates: Sequence[int]) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    ptr = np.zeros(len(mask) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(mask.sum(axis=1))
    return ptr, np.broadcast_to(np.array(candidates, dtype=np.int32), mask.shape)[mask]


@dataclass(frozen=True, eq=False)
class TransformTable:
    r"""Resolved codes and relative transforms of one read; states are final when returned. Row 0 of
    ``transforms`` is the identity, KNOWN×DEFINED, every other row some side's or relative's or the inverse of a KNOWN
    relative; UNKNOWN and CONFLICT rows carry zero payload."""

    transforms: Transforms
    per_side: NDArray[np.int32]  # [d, 2] code row per declaration side
    relative: NDArray[np.int32]  # [d] row of op(s1)⁻¹ ∘ op(s2)
    records: Records


def resolve_transforms(
    symmetry: Symmetry,
    codes: NDArray[np.int32],
    code_state: NDArray[np.uint8],
    code_text: tuple[str, ...],
    origin: NDArray[np.uint8],
    refs: NDArray[np.int64],
    first_rows: NDArray[np.int64],
    second_rows: NDArray[np.int64],
    coordinates: NDArray[np.float64],
) -> TransformTable:
    r"""
    Each side's stated code and each declaration's relative transform g = op(s1)⁻¹ ∘ op(s2), exact.

    A side is stated (KNOWN×DECLARED, ``codes`` into ``code_text``), ``.``/blank LINK (KNOWN×DEFINED), ``?``
    (UNKNOWN×DECLARED), absent (UNKNOWN×NONE, ``symmetry_absent``; ``ssbond_symmetry_absent`` for SSBOND) or CONFLICT
    (an OFoldX basis row restoring a written conflict). A code resolves by its id in the numbering in force; an
    id no row states is UNKNOWN, ``space_group_setting_unresolved``; every code is UNKNOWN when a row of the file's
    table states no operation, the read's ``operation_table_unparsable`` naming the rows (the table is not in force,
    ``symmetry``); equal parsed codes give the identity. A
    KNOWN g ≠ I placing the partner (declared-altloc rows) beyond 5 Å of E is born CONFLICT, ``contact_implausible``;
    a side without a stated position (row −1) is never measured. Under an INAPPLICABLE frame no crystal operation
    applies: a KNOWN g ≠ I is born CONFLICT, ``crystal_operation_inapplicable`` naming the declaration and what the
    frame rests on (``CrystalFrame.refs``). The table also holds the inverse g⁻¹ of every KNOWN relative
    transform (the second endpoint's external connection).
    Records name ``refs[d]`` (rows of ``symmetry.records.refs``) under subject (EDGE, d).
    """
    d = len(refs)
    stated = code_state == _KNOWN_DECLARED

    rows: dict[tuple, int] = {(_IDENTITY, -1, _KNOWN_DEFINED): 0}  # (operation | None, code, state) -> row
    log: list[tuple] = []
    index = {id_: k for k, id_ in enumerate(symmetry.operation_ids)}
    contested = symmetry.contested
    unusable = {record.detail["id"] for record in symmetry.records.where(code="operation_id_conflict")}
    unparsed = {record.detail["id"] for record in symmetry.records.where(code="operation_table_unparsable")}
    setting = "" if symmetry.setting is None else symmetry.setting.name
    read = symmetry.records.where(code="space_group_setting_unresolved")  # its name and number as stated
    statement = (read[0].detail["name"], read[0].detail["number"]) if len(read) else ("", -1)
    text_row = np.zeros(len(code_text) + 1, dtype=np.int32)  # the last entry stands for code −1
    parsed_key = np.full(len(code_text) + 1, -1, dtype=np.int64)
    problems: dict[int, tuple] = {}  # code index -> (record code, payload after the side, operation row...)
    parsed_codes: dict[tuple, int] = {}
    for c in np.unique(codes[stated]).tolist():
        text, parsed = code_text[c], parse_symmetry_code(code_text[c])
        key: tuple[SymOp | None, int, int] = (None, -1, _CONFLICT)
        if parsed is None:
            reading = _alternative_reading(text)
            issue = ("symmetry_code_ambiguous", (text, reading)) if reading else ("symmetry_code_grammar", (text,))
            problems[c] = issue
            key = (None, -1, _UNKNOWN if reading else _CONFLICT)
        elif unparsed:  # the table is not in force, the read's records name its rows: Unknown
            key = (None, -1, _UNKNOWN)
        elif (n := parsed[0]) not in index and not contested and n not in unusable:
            problems[c], key = ("space_group_setting_unresolved", (*statement, text)), (None, -1, _UNKNOWN)
        elif n in symmetry.disagreement and not contested:  # a disagreement exists only against a resolved setting
            operation_index, ita = index[n], cast(SpaceGroup, symmetry.setting).operations
            same = triplet(ita[int(n) - 1]) if 0 < int(n) <= len(ita) else ""
            problems[c] = (
                "operation_table_vs_setting",
                (triplet(symmetry.operations[operation_index]), same, setting),
                operation_index,
            )
        elif not contested and n not in unusable:
            key = (_shifted(symmetry.operations[index[n]], parsed[1]), _pack(n, parsed[1]), _KNOWN_DERIVED)
        parsed_key[c] = -1 if parsed is None else parsed_codes.setdefault(parsed, len(parsed_codes))
        text_row[c] = rows.setdefault(key, len(rows))

    per_side = np.where(stated, text_row[codes], 0).astype(np.int32)
    per_side[code_state == _UNKNOWN_DECLARED] = rows.setdefault((None, -1, _UNKNOWN_DECLARED), len(rows))
    per_side[absent := code_state == _UNKNOWN] = rows.setdefault((None, -1, _UNKNOWN), len(rows))
    per_side[code_state == _CONFLICT] = rows.setdefault((None, -1, _CONFLICT), len(rows))  # an OFoldX basis row
    for i, side in zip(*np.nonzero(absent | (stated & np.isin(codes, list(problems))))):
        if absent[i, side]:
            code = "ssbond_symmetry_absent" if origin[i] == QuerySource.SSBOND else "symmetry_absent"
            log.append((code, _EDGE, i, (refs[i],), (side,)))
        else:
            code, values, *operation_rows = problems[codes[i, side]]
            log.append(
                (code, _EDGE, i, (refs[i], *(symmetry.operation_refs[j] for j in operation_rows)), (side, *values))
            )

    first_key, second_key = parsed_key[codes[:, 0]], parsed_key[codes[:, 1]]
    equal = stated.all(axis=1) & (first_key == second_key) & (first_key >= 0)
    table = list(rows)
    pairs, pair_inverse = np.unique(np.column_stack([per_side, equal]), axis=0, return_inverse=True)
    relative = [_relative(symmetry, table[a], table[b], bool(e), rows) for a, b, e in pairs.tolist()]
    relative = np.array(relative, np.int32)[pair_inverse.ravel()] if d else np.zeros(0, dtype=np.int32)

    frame, table = symmetry.frame, list(rows)
    moved = np.array([key[0] not in (None, _IDENTITY) for key in table])[relative]
    if frame is not None and frame.state == _INAPPLICABLE:  # no lattice, no crystal operation
        for i in np.flatnonzero(moved).tolist():
            relative[i] = rows.setdefault((None, -1, _CONFLICT), len(rows))
            log.append(("crystal_operation_inapplicable", _EDGE, i, (refs[i], *frame.refs), ()))
    moved &= (first_rows >= 0) & (second_rows >= 0)
    for i in np.flatnonzero(moved if frame is not None and known(frame.state) else []).tolist():
        active_frame = cast(CrystalFrame, frame)  # The loop's selection requires a known frame.
        operation, e, f = table[relative[i]][0], coordinates[first_rows[i]], coordinates[second_rows[i]]
        rotation = np.array(operation.rotation, dtype=np.float64).reshape(3, 3)
        a, u = active_frame.fractionalisation[:, :3], active_frame.fractionalisation[:, 3]
        image = active_frame.orthogonalisation @ (rotation @ (a @ f + u) + np.array(operation.translation) / _DEN - u)
        if (distance := float(np.linalg.norm(image - e))) > _CONTACT_BOUND:  # NaN (unplaced) never exceeds
            relative[i] = rows.setdefault((None, -1, _CONFLICT), len(rows))
            log.append(("contact_implausible", _EDGE, i, (refs[i],), (round(distance * 1000),)))

    # the inverse of a KNOWN relative transform places the first endpoint seen from the second (its external connection)
    inverse_rows = []
    for operation, _, resolved in [list(rows)[k] for k in np.unique(relative).tolist()]:
        if operation is not None and operation != _IDENTITY:
            image = operation.inverse()
            inverse_rows.append(rows.setdefault((image, _code(symmetry, image), resolved), len(rows)))
    table_keys = list(rows)
    used = np.unique(np.concatenate([[0], per_side.ravel(), relative, inverse_rows])).astype(np.int64)  # rows in use
    used = np.array(sorted(used.tolist(), key=lambda k: _table_order(table_keys[k])), dtype=np.int64)
    table, renumber = [table_keys[k] for k in used.tolist()], np.zeros(len(table_keys), np.int32)
    renumber[used] = np.arange(len(used))
    per_side, relative = renumber[per_side], renumber[relative]
    rotation = np.array([key[0].rotation if key[0] else (0,) * 9 for key in table], np.int8).reshape(-1, 3, 3)
    translation = np.array([key[0].translation if key[0] else (0,) * 3 for key in table], np.int16).reshape(-1, 3)
    code, resolved = np.array([key[1] for key in table], np.int32), np.array([key[2] for key in table], np.uint8)
    transforms = Transforms(rotation, translation, code, resolved)
    return TransformTable(transforms, per_side, relative, _records(log, symmetry.records.refs))


def _table_order(key: tuple) -> tuple:
    r"""The canonical order of transform rows: identity first (row 0, KNOWN×DEFINED, leads), then by operation, state
    and code, so that neither the order nor the numbering of the source's codes orders the table."""
    operation, code, state = key
    exact = () if operation is None else (operation.rotation, operation.translation)
    return operation is None, operation != _IDENTITY, exact, state, code


def _relative(symmetry: Symmetry, first: tuple, second: tuple, equal: bool, rows: dict[tuple, int]) -> int:
    r"""The row of g = op(s1)⁻¹ ∘ op(s2) for two side rows (keys ``(operation, code, state)``)."""
    (a, _, s1), (b, _, s2) = first, second
    key: tuple[SymOp | None, int, int]
    if equal:  # op(n)⁻¹ ∘ op(n) = I exactly, whatever op(n) is, contested included; the sides keep their states
        key = (_IDENTITY, _code(symmetry, _IDENTITY), _KNOWN_DERIVED)
    elif _CONFLICT in (s1, s2):
        key = (None, -1, _CONFLICT)
    elif not (known(s1) and known(s2)):
        key = (None, -1, _UNKNOWN_DECLARED if _UNKNOWN_DECLARED in (s1, s2) else _UNKNOWN)
    elif (g := a.inverse().compose(b)) == _IDENTITY and s1 == s2 == _KNOWN_DEFINED:
        return 0
    else:  # a composition of symmetry is a derivation, identity included
        key = (g, _code(symmetry, g), _KNOWN_DERIVED)
    return rows.setdefault(key, len(rows))


def _code(symmetry: Symmetry, operation: SymOp) -> int:
    r"""The packed code of an operation in the numbering in force: the id of the first listed operation of its coset and
    the lattice shift from it; −1 when no listed operation has its coset."""
    for k, listed in enumerate(symmetry.operations if symmetry.numbering != "none" else ()):
        if _coset(listed) == _coset(operation):
            cells = [(t - u) // _DEN for t, u in zip(operation.translation, listed.translation)]
            return _pack(symmetry.operation_ids[k], cells)
    return -1


@dataclass(frozen=True, eq=False)
class EdgeGeometry:
    r"""Per declaration: its relative transform row (``TransformTable.relative``), its physical edge, the neighbours it
    contributes at each endpoint (``count`` [d, 2]; −1 undeterminable, 0 for a coordination row), whether it is
    E–E under the identity (``self_bond``) and ``reading``, whether the rows naming its atom pair have one reading as
    physical edges: KNOWN×DERIVED, UNKNOWN×NONE (several) or CONFLICT×NONE (none)."""

    transform: NDArray[np.int32]
    count: NDArray[np.int8]
    physical_edge: NDArray[np.int32]
    self_bond: NDArray[np.bool_]
    reading: NDArray[np.uint8]


class _Sites:
    r"""Exact questions about the stabilisers of the endpoints of a set of declarations."""

    def __init__(self, symmetry: Symmetry, table: StabiliserTable, transforms: TransformTable, ends: NDArray[np.int32]):
        self.operations, self.table, self.status = symmetry.operations, table, _list_status(symmetry)[0]
        self.transforms, self.relative = transforms.transforms, transforms.relative
        at = np.minimum(np.searchsorted(table.endpoint, ends), max(len(table.endpoint) - 1, 0))
        present = (ends >= 0) & (np.append(table.endpoint, -1)[at] == ends) if len(table.endpoint) else ends < -1
        self.rows = np.where(present, at, -1)  # each side's row of the table, −1 none

    def operation(self, i: int, side: int = 0) -> SymOp:
        r"""Declaration i's relative transform read from its ``side`` endpoint (the inverse at side 1)."""
        row, t = self.relative[i], self.transforms
        g = SymOp(tuple(t.rotation[row].ravel().tolist()), tuple(t.translation[row].tolist()))
        return g if side == 0 else g.inverse()

    def members(self, r: int) -> set[SymOp]:
        r"""The decided members of Stab of table row r, the identity always one: the table holds it as a row of the
        list in force, which an INAPPLICABLE frame's trivial Stab(E) need not have (an unresolved space group)."""
        span = slice(self.table.ptr[r], self.table.ptr[r + 1]) if r >= 0 else slice(0)
        rows = zip(self.table.operation[span].tolist(), self.table.shift[span].tolist())
        return {_IDENTITY, *(_shifted(self.operations[k], s) for k, s in rows)}

    def member(self, r: int, x: SymOp) -> bool | None:
        r"""Whether x ∈ Stab of table row r; None when undecided."""
        if x in self.members(r):
            return True
        if x.rotation == _IDENTITY.rotation or _fixed_point_free(_coset(x)):
            return False  # a lattice translation, a screw or a glide fixes no point (coordinate-free)
        if r < 0 or self.status in ("conflict", "unknown"):
            return None
        span = self.table.undecided[self.table.undecided_ptr[r] : self.table.undecided_ptr[r + 1]]
        return None if _coset(x) in {_coset(self.operations[k]) for k in span.tolist()} else False

    def same_edge(self, i: int, j: int, ends: NDArray[np.int32], alike: bool) -> bool | None:
        r"""Whether rows i, j of one native pair (lo, hi) are one physical edge: g_j ∈ Stab(lo)·g_i·Stab(hi), for a
        self-link also Stab(E)·g_i⁻¹·Stab(E) (E–E^g ≡ E–E^{g⁻¹}); None when undecided. Beyond g_j = g_i (g_i⁻¹) the
        join rests on a member other than the identity, decided only within a tolerance: it is decided for rows that
        state ``alike``, undecided for rows it would set against each other."""
        side = int(ends[i, 0] > ends[i, 1])
        lo, hi = self.rows[i, side], self.rows[i, 1 - side]
        gi, gj = self.operation(i, side), self.operation(j, int(ends[j, 0] > ends[j, 1]))
        both = (gi, gi.inverse()) if ends[i, 0] == ends[i, 1] else (gi,)
        if gj in both:
            return True
        answer: bool | None = lo >= 0 and self.table.state[lo] == _KNOWN_DERIVED  # a decided Stab(lo) can say no
        for g in both:
            for a in self.members(lo):
                if found := self.member(hi, a.compose(g).inverse().compose(gj)):
                    return True if alike else None
                answer = None if found is None else answer
        return False if answer else None

    def count(self, i: int, side: int, resolved: int, self_link: bool, doubtful: bool) -> int:
        r"""The neighbours row i contributes at its ``side`` endpoint E, ignoring topology; −1 when undeterminable."""
        e, f = self.rows[i, side], self.rows[i, 1 - side]
        if e < 0 or self.table.state[e] != _KNOWN_DERIVED or resolved == _CONFLICT:
            return -1
        stabiliser = self.members(e)
        if not known(resolved):
            return 1 if len(stabiliser) == 1 and not self_link else -1
        if doubtful:
            return -1
        g = self.operation(i, side)
        if self_link:  # the Stab(E)-orbit count of {E^g, E^{g⁻¹}}: distinct left cosets x·Stab(E)
            partners = (a.compose(h) for a in stabiliser for h in (g, g.inverse()))
            return len({frozenset(x.compose(s) for s in stabiliser) for x in partners})
        fixed = [self.member(f, g.inverse().compose(s).compose(g)) for s in stabiliser]
        return -1 if None in fixed else len(stabiliser) // sum(cast(list[bool], fixed))


def edge_geometry(
    symmetry: Symmetry,
    transforms: TransformTable,
    codes: NDArray[np.int32],
    code_state: NDArray[np.uint8],
    stabilisers: StabiliserTable,
    first: NDArray[np.int32],
    second: NDArray[np.int32],
    kind: NDArray[np.uint8],
    topology_fixed: NDArray[np.bool_],
    statement: NDArray[np.int64],
) -> EdgeGeometry:
    r"""
    Physical edges, neighbour counts and self-bonds of the declarations.

    ``(E, F, g)``, ``(E, F, g')`` (oriented ``E^I–F^g ≡ F^I–E^{g⁻¹}``) are one physical edge iff g' ∈ Stab(E)·g·Stab(F)
    (for E = F also Stab(E)·g⁻¹·Stab(E)); undecided → apart, every count of both −1. A member of Stab(E) other
    than the identity is only ever decided within a geometric tolerance, an inference that may count and merge rows but
    never sets declarations against each other as a source conflict: it joins rows that state alike (``statement``
    [d, k], what each states of its edge oriented (lo, hi), equal in every column), and leaves rows that state
    otherwise undecided; nor does it put a fixed relation a read establishes in conflict, and only E–E under the exact
    identity is a self-bond (an image E^g within tolerance of E is not E). Rows stating the same code tokens per side
    (``codes``, every side KNOWN×DECLARED in ``code_state``, oriented) are one physical edge whatever their transform:
    within a file one code names one operation. Under an INAPPLICABLE frame no crystal operation applies,
    so rows naming one atom pair are one physical edge unless one has a CONFLICT transform. Several rows naming one
    atom pair have one reading (``reading``) when every pair of them is so decided one edge or apart;
    otherwise (an unstated code, an undecided pair) one restated edge and distinct edges are both compatible (UNKNOWN),
    unless a CONFLICT transform among them admits none (CONFLICT). A covalent or unknown-kind (``kind`` 0) row counts
    |Stab(E)| / |Stab(E) ∩ g·Stab(F)·g⁻¹| at E, a self-link the Stab(E)-orbit count of {E^g, E^{g⁻¹}}, an UNKNOWN
    transform 1 only at a decided trivial Stab(E); −1 wherever E's value depends on an undecided operation, a CONFLICT
    or a missing endpoint. A topology-fixed row counts 1 at both sides. Every present endpoint of a covalent row
    that is not topology-fixed, or of a row with a KNOWN g ≠ I, needs a row of ``stabilisers``.
    """
    d, relative = len(first), transforms.relative
    ends = np.stack([first, second], axis=1)
    sites, resolved = _Sites(symmetry, stabilisers, transforms, ends), transforms.transforms.state[relative]
    covalent = np.isin(kind, (0, BondKind.COVALENT))

    inapplicable = symmetry.frame is not None and symmetry.frame.state == _INAPPLICABLE
    lo, hi = ends.min(axis=1), ends.max(axis=1)  # physical edges: per atom pair, the union of the rows decided one
    token = np.where(code_state == _KNOWN_DECLARED, codes, -1)  # the stated code tokens, oriented (lo, hi)
    token = np.where((first > second)[:, None], token[:, ::-1], token)
    token[lo == hi] = np.sort(token[lo == hi], axis=1)  # a self-link's sides are unordered
    two_ended = np.flatnonzero(lo >= 0)
    two_ended = two_ended[np.lexsort((hi[two_ended], lo[two_ended]))]
    label, doubtful = np.arange(d), np.zeros(d, dtype=bool)
    reading = np.full(d, _KNOWN_DERIVED, dtype=np.uint8)
    for group in np.split(two_ended, np.flatnonzero(np.diff(lo[two_ended]) | np.diff(hi[two_ended])) + 1):
        undecided = []
        for x, i in enumerate(group.tolist()):
            for j in group[x + 1 :].tolist():
                decidable = bool(known(resolved[i]) and known(resolved[j]))
                alike = bool((statement[i] == statement[j]).all())
                same = sites.same_edge(i, j, ends, alike) if decidable else None
                doubtful[[i, j]] |= decidable and same is None
                if same is None and (token[i] >= 0).all() and (token[i] == token[j]).all():
                    same = True  # one code names one operation within a file: one statement made twice
                if same is None and inapplicable and _CONFLICT not in (resolved[i], resolved[j]):
                    same = True  # only the identity relates them
                if same:
                    a, b = _root(label, i), _root(label, j)
                    label[max(a, b)] = min(a, b)
                elif same is None:
                    undecided.append((i, j))
        if any(_root(label, i) != _root(label, j) for i, j in undecided):  # no one reading: none with a CONFLICT
            reading[group] = _CONFLICT if (resolved[group] == _CONFLICT).any() else _UNKNOWN
    roots = np.array([_root(label, i) for i in range(d)], dtype=np.int64)
    physical_edge = np.unique(roots, return_inverse=True)[1].ravel().astype(np.int32)

    count = np.repeat(covalent[:, None], 2, axis=1).astype(np.int8)
    self_link = (first == second) & (first >= 0)
    for i, side in zip(*np.nonzero((covalent & ~topology_fixed)[:, None] & (ends >= 0))):
        count[i, side] = sites.count(i, side, resolved[i], self_link[i], doubtful[i])
    count[covalent[:, None] & (ends < 0)] = -1

    identity = (transforms.transforms.rotation == np.eye(3)).all(axis=(1, 2))
    identity &= ~transforms.transforms.translation.any(axis=1)
    self_bond = self_link & known(resolved) & identity[relative]
    return EdgeGeometry(relative, count, physical_edge, self_bond, reading)


def _root(label: NDArray[np.int64], i: int) -> int:
    while label[i] != i:
        label[i] = label[label[i]]
        i = label[i]
    return i


def cartesian(frame: CrystalFrame, transforms: Transforms) -> NDArray[np.float64]:
    r"""Each transform (R, τ = t/24) as the Cartesian 4×4 matrix of x ↦ O·[R·(A·x + u) + τ − u] in the frame [A | u]:
    O·R·A and O·(R·u + τ − u); NaN for a row that is not KNOWN. Raises
    ``InsufficientInformationError(frame_unknown)`` without a KNOWN frame (an INAPPLICABLE one included)."""
    if frame is None or not known(frame.state):
        fail("frame_unknown", RequestRef("frame", "none" if frame is None else f"state {frame.state}"))
    o, rotation = frame.orthogonalisation, transforms.rotation.astype(np.float64)
    a, u = frame.fractionalisation[:, :3], frame.fractionalisation[:, 3]
    matrices = np.zeros((len(transforms.state), 4, 4))
    matrices[:, :3, :3] = o @ rotation @ a
    matrices[:, :3, 3] = (rotation @ u + transforms.translation / _DEN - u) @ o.T
    matrices[:, 3, 3] = 1.0
    matrices[~known(transforms.state)] = np.nan
    return matrices
