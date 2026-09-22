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
Declared operators and the copies of a requested assembly.

A ``_pdbx_struct_oper_list`` row, or a ``REMARK 350 BIOMT`` matrix, is applied as deposited, except a crystal symmetry
row named ``n_klm``: its placement is the operation in force of that code in the crystal frame, and the deposited matrix
only cross-checks it. Placements are measured, never repaired: the rigidity defect, the largest deviation of a singular
value of the linear part from one, is recorded where it is not negligible. The copies of an assembly are ``Copy``
values in generator, product and asym order; realising them, the copy join included, is
``data.assembly.replicate_arrays``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..data.assembly import Copy
from ..data.molecular import Transforms
from ..data.structure import CopyId, CopyKind
from ..errors import RequestRef, SourceRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, known, resolution, state
from .symmetry import CrystalFrame, Symmetry, SymOp, cartesian, parse_symmetry_code
from .tables import AssemblyTable, OperatorTable, TableBuilder

_CRYSTAL = "crystal symmetry operation"
_UNSUPPORTED = frozenset({"2D crystal symmetry operation", "3D crystal symmetry operation"})  # never applied
_ROTATION_TOLERANCE, _TRANSLATION_TOLERANCE = 1e-5, 1e-3  # floors under what two printings account for
_RIGID, _RIGIDITY_LIMIT = 1e-5, 1e-3  # the limit: a floor under what the printing accounts for
_IDENTITY_ROTATION = (1, 0, 0, 0, 1, 0, 0, 0, 1)
_GROUPS = re.compile(r"(?:\([^()]+\))+")
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)


@dataclass(frozen=True, eq=False)
class DeclaredOperator:
    r"""
    One declared operator and the Cartesian 4×4 placement it applies; ``type`` is ``""`` and ``name`` None where the
    source states none (``REMARK 350`` states neither).

    ``status``: ``rigid`` (defect ≤ 1e-5), ``defect_recorded`` (a larger defect, recorded), ``unsupported`` (a 2D/3D
    crystal operation: ``matrix`` as deposited, never applied) or ``undetermined`` (``matrix`` NaN where unknown,
    ``rigidity_defect`` NaN). ``defect_origin`` names the statement a defect, or the indeterminacy, comes from:
    ``crystal_frame`` when it is the operation in force in the read's frame (an undetermined crystal row: no KNOWN
    frame), ``declared_matrix`` when it is the deposited matrix (an undetermined row: a coefficient not stated).
    """

    id: str
    type: str
    name: str | None
    matrix: NDArray[np.float64]
    rigidity_defect: float
    defect_origin: Literal["crystal_frame", "declared_matrix"]
    status: Literal["rigid", "defect_recorded", "unsupported", "undetermined"]
    ref: SourceRef


def declared_operators(
    symmetry: Symmetry,
    table: OperatorTable,
    symbols: Mapping[str, tuple[str, ...]],
    refs: SourceRefs,
    ids: Collection[str],
) -> tuple[tuple[DeclaredOperator, ...], Records]:
    r"""
    The operators of ``table`` whose id is in ``ids`` (those a request applies) in row order, and their records
    (subject NONE, each naming its row); a row of another id is neither placed nor recorded.

    A crystal symmetry row whose name is a code resolving in the numbering in force is placed by that operation in the
    frame (the identity needs none; no frame, or an UNKNOWN or CONFLICT one, leaves it ``undetermined``), with
    ``operator_matrix_deviates`` when the deposited matrix differs beyond what the two printings account for, never
    less than 1e-5 in rotation and 1e-3 Å in translation. Every other row is applied as deposited, a crystal row whose
    code does not resolve included (``symmetry`` records why), and so is every crystal row under an INAPPLICABLE frame,
    where no crystal operation applies. Records: ``operator_2d3d_present`` per 2D/3D row,
    ``operator_defect_recorded`` for a defect above 1e-5 and ``non_isometric_operator`` beyond max(1e-3, ‖q‖_F), q the
    printed half-units of the deposited linear part when the defect is its own (none for a computed placement's),
    past which no rotation R lies within them: |σᵢ(R + E) − 1| ≤ ‖E‖₂ ≤ ‖q‖_F for |E| ≤ q (Weyl).
    ``symbols`` are the read's families (``operator``, ``text``); ``table.ref`` indexes ``refs``.
    """

    builder = TableBuilder(refs.block)
    operators = []
    for i, ref in enumerate(table.ref.tolist()):
        if (id_ := symbols["operator"][table.id[i]]) not in ids:
            continue
        type_ = symbols["text"][table.type[i]] if table.type[i] >= 0 else ""
        name = symbols["text"][table.name[i]] if table.name[i] >= 0 else None
        matrix = np.vstack([table.matrix[i], [0.0, 0.0, 0.0, 1.0]])
        defect = _defect(matrix)
        origin: Literal["crystal_frame", "declared_matrix"] = "declared_matrix"
        status: Literal["rigid", "defect_recorded", "unsupported", "undetermined"] | None = None
        if type_ in _UNSUPPORTED:
            builder.record("operator_2d3d_present", [ref])
            status = "unsupported"
        elif (operation := _named_operation(symmetry, type_, name)) is not None:
            identity = operation.rotation == _IDENTITY_ROTATION and not any(operation.translation)
            frame = None if identity else symmetry.frame
            if not identity and (frame is None or not known(frame.state)):
                matrix, origin, status = np.full((4, 4), np.nan), "crystal_frame", "undetermined"
            else:
                computed = np.eye(4) if identity else cartesian(cast(CrystalFrame, frame), _transforms(operation))[0]
                if _deviates(computed, matrix, table.quantum[i], frame):
                    builder.record("operator_matrix_deviates", [ref])
                if not defect > _defect(computed) + _RIGID:  # the deposited matrix carries no worse defect
                    defect, origin = _defect(computed), "crystal_frame"
                matrix = computed
        if status is None and not np.isfinite(matrix).all():
            status = "undetermined"
        elif status is None:
            status = "rigid" if defect <= _RIGID else "defect_recorded"
            if defect > _RIGID:
                printed = np.linalg.norm(table.quantum[i][:, :3]) if origin == "declared_matrix" else 0.0
                limit = max(_RIGIDITY_LIMIT, printed)
                builder.record("operator_defect_recorded" if defect <= limit else "non_isometric_operator", [ref])
        if status == "undetermined":
            defect = np.nan
        operators.append(DeclaredOperator(id_, type_, name, matrix, defect, origin, status, refs.ref(ref)))
    return tuple(operators), builder.records(refs)


def _defect(matrix: NDArray[np.float64]) -> float:
    r"""The largest deviation of a singular value of the linear part from one; NaN for an incomplete matrix."""
    if not np.isfinite(matrix[:3, :3]).all():
        return np.nan
    return float(np.abs(np.linalg.svd(matrix[:3, :3], compute_uv=False) - 1.0).max())


def _named_operation(symmetry: Symmetry, type_: str, name: str | None) -> SymOp | None:
    r"""The operation in force a crystal row's ``n_klm`` name states, lattice shift included; None when the row is not
    such a row, its id does not resolve (an id absent from the numbering in force, an id whose file operation the
    setting disagrees with, or an ITA list contested by ``space_group_name_vs_number``) or no crystal operation
    applies (an INAPPLICABLE frame: the row is applied as deposited)."""
    if type_ != _CRYSTAL or name is None or (code := parse_symmetry_code(name)) is None:
        return None
    if symmetry.frame is not None and resolution(np.uint8(symmetry.frame.state)) == Resolution.INAPPLICABLE:
        return None
    n, shift = code
    if symmetry.contested or n not in symmetry.operation_ids or n in symmetry.disagreement:
        return None
    operation = symmetry.operations[symmetry.operation_ids.index(n)]
    translation = cast(tuple[int, int, int], tuple(t + 24 * s for t, s in zip(operation.translation, shift)))
    return SymOp(operation.rotation, translation)  # 24ths


def _transforms(operation: SymOp) -> Transforms:
    return Transforms(
        np.array(operation.rotation, dtype=np.int8).reshape(1, 3, 3),
        np.array([operation.translation], dtype=np.int16),
        np.array([-1], dtype=np.int32),
        np.array([_KNOWN_DERIVED], dtype=np.uint8),
    )


def _deviates(
    computed: NDArray[np.float64],
    deposited: NDArray[np.float64],
    quantum: NDArray[np.float64],
    frame: CrystalFrame | None,
) -> bool:
    r"""Whether the stated coefficients of the deposited matrix differ from the computed placement [M | t] beyond what
    the statements account for: the deposit's printed half-units, and the frame's [δA | δu] propagated through
    M = O·R·A and t = O·(R·u + τ − u) (O = A⁻¹; δM bounded by |O|·|δA|·|M| + |M|·|O|·|δA|, δt by
    |O|·|δA|·|t| + |M − I|·|O|·|δu|), δA the printed half-units of A plus what the cell's relaxed range admits
    (``CrystalFrame.cell_spread``: wwPDB computes the deposited matrices from the cell, 2zex)."""
    stated = np.isfinite(deposited[:3])
    if not stated.any():
        return False
    difference = np.where(stated, np.abs(computed[:3] - deposited[:3]), 0.0)
    printed = np.where(stated, quantum, 0.0)
    rotation_bound, translation_bound = printed[:, :3].max(), printed[:, 3].max()
    if frame is not None:
        o, delta_u = np.abs(frame.orthogonalisation), frame.quantum[:, 3]
        delta_a = frame.quantum[:, :3] + frame.cell_spread
        m, t = np.abs(computed[:3, :3]), np.abs(computed[:3, 3])
        rotation_bound += (o @ delta_a @ m + m @ o @ delta_a).max()
        translation_bound += (o @ delta_a @ t + np.abs(computed[:3, :3] - np.eye(3)) @ o @ delta_u).max()
    return bool(
        difference[:, :3].max() > max(_ROTATION_TOLERANCE, rotation_bound)
        or difference[:, 3].max() > max(_TRANSLATION_TOLERANCE, translation_bound)
    )


def parse_operator_expression(text: str, ref: SourceRef) -> tuple[tuple[str, ...], ...]:
    r"""
    The operator products of one ``oper_expression`` in product order, left to right: a comma list of operator ids
    and ``m-n`` ranges, or a product of parenthesised lists. Each product holds its ids as written; PDBx applies the
    rightmost first.

    Raises:
        ContradictionError: ``operator_expression_grammar`` for an empty list or item, an unbalanced parenthesis or a
            descending range.

    Examples:
        >>> parse_operator_expression("(1,2)(X0)", SourceRef("pdbx_struct_assembly_gen", 0))
        (('1', 'X0'), ('2', 'X0'))
        >>> parse_operator_expression("1-3", SourceRef("pdbx_struct_assembly_gen", 0))
        (('1',), ('2',), ('3',))
    """

    compact = re.sub(r"\s+", "", text)
    if "(" not in compact and ")" not in compact:
        groups = [compact]
    elif _GROUPS.fullmatch(compact):
        groups = re.findall(r"\(([^()]+)\)", compact)
    else:
        fail("operator_expression_grammar", ref, expression=text)
    lists = []
    for group in groups:
        ids: list[str] = []
        for item in group.split(","):
            first, dash, last = item.partition("-")
            if dash and first.isdecimal() and last.isdecimal():
                if int(last) < int(first):
                    fail("operator_expression_grammar", ref, expression=text)
                ids += [str(n) for n in range(int(first), int(last) + 1)]
            elif item:
                ids.append(item)
            else:
                fail("operator_expression_grammar", ref, expression=text)
        lists.append(ids)
    return tuple(product(*lists))


def assembly_copies(
    symmetry: Symmetry,
    operators: OperatorTable,
    assemblies: AssemblyTable,
    symbols: Mapping[str, tuple[str, ...]],
    refs: SourceRefs,
    assembly: str,
    instances: Mapping[str, NDArray[np.int64]],
) -> tuple[tuple[Copy, ...], Records]:
    r"""
    The copies of ``assembly``, and the records of the operators its expressions name
    (``declared_operators``; an operator it never applies refuses nothing): per generator row in row order, per
    operator product of its expression left to right, per listed asym in the key order of ``instances`` (asym id →
    instance rows, in ``_struct_asym`` order), one ``Copy`` of that asym's instances named
    ``CopyId(ASSEMBLY, product, asym)`` and placed by the product of its operators' matrices (the rightmost applied
    first). The asym of an id that names several instances (a PDB author chain) is None. A copy generated twice is made
    once, an asym without instance rows none.

    No copy is ``DEPOSITED``: the deposited instances are kept only as the copy under a listed identity operator,
    ``CopyId(ASSEMBLY, ("1",), asym)`` in wwPDB files, so the caller replicates with ``keep_source=False``.
    ``symbols`` are the read's families (``assembly``, ``operator``, ``text``); ``assemblies.gen_ref`` and
    ``operators.ref`` index ``refs``.

    Raises:
        InvalidRequestError: ``assembly_unknown`` when no generator row names ``assembly``.
        ContradictionError: ``operator_expression_grammar``; ``operator_unresolved`` for an operator id no row declares;
            ``asym_undeclared`` for a listed asym ``instances`` does not name; ``copy_identity_conflict`` for two
            products placing one asym at one placement (within the matrix-comparison tolerances).
        UnsupportedError: ``operator_2d3d`` for a 2D or 3D crystal operation.
        InsufficientInformationError: ``frame_unknown`` (crystal row without a KNOWN frame) or
            ``operator_matrix_unknown`` (a deposited coefficient not stated) for an undetermined operator.
    """

    rows = [r for r, code in enumerate(assemblies.gen_assembly.tolist()) if symbols["assembly"][code] == assembly]
    if not rows:
        fail("assembly_unknown", RequestRef("assembly", assembly))
    order = {asym: k for k, asym in enumerate(instances)}
    generators = []
    for r in rows:
        ref = refs.ref(int(assemblies.gen_ref[r]))
        listed = {asym.strip() for asym in symbols["text"][assemblies.gen_asyms[r]].split(",")}
        for asym in sorted(listed - order.keys()):
            fail("asym_undeclared", ref, asym=asym)
        generators.append((ref, listed, parse_operator_expression(symbols["text"][assemblies.gen_expression[r]], ref)))
    named = {id_ for *_, products in generators for product in products for id_ in product}
    applied, records = declared_operators(symmetry, operators, symbols, refs, named)
    declared = {operator.id: operator for operator in applied}
    placements: dict[tuple[str, ...], tuple[NDArray[np.float64], set[str], SourceRef]] = {}
    copies: dict[tuple[tuple[str, ...], str], Copy] = {}
    for ref, listed, products in generators:
        for operator_ids in products:
            if operator_ids not in placements:
                placements[operator_ids] = (_placement(declared, operator_ids, ref), set(), ref)
            matrix, placed, _ = placements[operator_ids]
            for asym in sorted(listed, key=order.__getitem__):
                rows = np.asarray(instances[asym], dtype=np.int64)
                if len(rows) and (operator_ids, asym) not in copies:
                    copy_id = CopyId(CopyKind.ASSEMBLY, operator_ids, asym if len(rows) == 1 else None)
                    copies[operator_ids, asym] = Copy(rows, copy_id, matrix)
                    placed.add(asym)
    _distinct(placements)
    return tuple(copies.values()), records


def _placement(
    declared: Mapping[str, DeclaredOperator], operator_ids: tuple[str, ...], ref: SourceRef
) -> NDArray[np.float64]:
    matrix = np.eye(4)
    for id_ in operator_ids:
        if (operator := declared.get(id_)) is None:
            fail("operator_unresolved", ref, operator=id_)
        if operator.status == "unsupported":
            fail("operator_2d3d", operator.ref, operator=id_)
        if operator.status == "undetermined":
            code = "frame_unknown" if operator.defect_origin == "crystal_frame" else "operator_matrix_unknown"
            fail(code, operator.ref)
        matrix = matrix @ operator.matrix
    return matrix


def _distinct(placements: Mapping[tuple[str, ...], tuple[NDArray[np.float64], set[str], SourceRef]]) -> None:
    r"""Refuse two operator products that place a common asym at one placement."""
    products = list(placements)
    matrices = np.stack([placements[p][0] for p in products])
    for j, first in enumerate(products[:-1]):
        rotation = np.abs(matrices[j + 1 :, :3, :3] - matrices[j, :3, :3]).max(axis=(1, 2))
        translation = np.abs(matrices[j + 1 :, :3, 3] - matrices[j, :3, 3]).max(axis=1)
        close = (rotation <= _ROTATION_TOLERANCE) & (translation <= _TRANSLATION_TOLERANCE)
        for second in (products[j + 1 + k] for k in np.flatnonzero(close)):
            if placements[first][1] & placements[second][1]:
                fail("copy_identity_conflict", placements[first][2], placements[second][2], first=first, second=second)
