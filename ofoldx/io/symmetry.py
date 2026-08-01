r"""Source operation declarations and reusable placement relationships."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from functools import cached_property
from itertools import product
from typing import cast

import numpy as np

from .cif import CifBlock as CIFBlock
from .cif import column_strings as _cif_column


def _affine_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _assembly_operation_steps(expression: str) -> tuple[tuple[str, ...], ...]:
    r"""Expand one PDBx assembly operation expression in application order."""

    groups = [group.strip() for group in expression.replace(")", "").split("(") if group.strip()]
    groups.reverse()
    choices: list[tuple[str, ...]] = []
    for group in groups:
        operation_ids: list[str] = []
        for item in group.split(","):
            item = item.strip()
            endpoints = item.split("-", maxsplit=1)
            if len(endpoints) == 2 and all(endpoint.isdecimal() for endpoint in endpoints):
                first, last = (int(endpoint) for endpoint in endpoints)
                if last < first:
                    raise ValueError(f"invalid mmCIF assembly operation range {item!r}")
                operation_ids.extend(str(operation_id) for operation_id in range(first, last + 1))
            elif item:
                operation_ids.append(item)
        if not operation_ids:
            raise ValueError(f"invalid mmCIF assembly operation expression {expression!r}")
        choices.append(tuple(operation_ids))
    if not choices:
        raise ValueError(f"invalid mmCIF assembly operation expression {expression!r}")
    return tuple(product(*choices))


def _mmcif_assembly_placements(block: object, assembly_id: str) -> dict[tuple[str | None, str], np.ndarray]:
    r"""Assign one global copy ID to each biological assembly operation tuple."""

    cif_block = cast("CIFBlock", block)
    try:
        assembly = cif_block["pdbx_struct_assembly_gen"]
        operators = cif_block["pdbx_struct_oper_list"]
    except KeyError as error:
        raise ValueError("mmCIF biological assembly lacks its operation categories") from error

    transformations: dict[str, np.ndarray] = {}
    operator_ids = _cif_column(operators, "id")
    try:
        matrices = {
            (row, column): operators[f"matrix[{row}][{column}]"].as_array(float)
            for row in (1, 2, 3)
            for column in (1, 2, 3)
        }
        vectors = {row: operators[f"vector[{row}]"].as_array(float) for row in (1, 2, 3)}
    except KeyError as error:
        raise ValueError("mmCIF biological assembly has an incomplete operation") from error
    for index, operator_id in enumerate(operator_ids):
        if operator_id is None:
            raise ValueError("mmCIF biological assembly operation lacks an id")
        transform = _affine_transform(
            np.asarray(
                [[matrices[row, column][index] for column in (1, 2, 3)] for row in (1, 2, 3)],
                dtype=np.float64,
            ),
            np.asarray([vectors[row][index] for row in (1, 2, 3)], dtype=np.float64),
        )
        previous = transformations.setdefault(operator_id, transform)
        if not np.array_equal(previous, transform):
            raise ValueError(f"mmCIF biological assembly operation {operator_id!r} is defined more than once")

    placements: dict[tuple[str | None, str], np.ndarray] = {}
    symmetry_id_by_operation: dict[tuple[str, ...], str] = {}
    for declared_assembly, expression, asym_list in zip(
        _cif_column(assembly, "assembly_id"),
        _cif_column(assembly, "oper_expression"),
        _cif_column(assembly, "asym_id_list"),
    ):
        if declared_assembly != assembly_id:
            continue
        if expression is None or asym_list is None:
            raise ValueError(f"mmCIF assembly {assembly_id!r} has an incomplete generation rule")
        operations = _assembly_operation_steps(expression)
        composed: list[tuple[tuple[str, ...], np.ndarray]] = []
        for operation in operations:
            transform = np.eye(4, dtype=np.float64)
            for operator_id in operation:
                try:
                    transform = transformations[operator_id] @ transform
                except KeyError:
                    raise ValueError(
                        f"mmCIF assembly {assembly_id!r} references unknown operation {operator_id!r}"
                    ) from None
            composed.append((operation, transform))
        asym_ids = tuple(item.strip() for item in asym_list.split(","))
        for asym_id in asym_ids:
            if not asym_id:
                raise ValueError(f"mmCIF assembly {assembly_id!r} has an empty asym id")
        for operation, transform in composed:
            symmetry_id = symmetry_id_by_operation.setdefault(operation, str(len(symmetry_id_by_operation)))
            for asym_id in asym_ids:
                key = symmetry_id, asym_id
                previous = placements.setdefault(key, transform)
                if not np.allclose(previous, transform, rtol=0.0, atol=1e-12):
                    raise ValueError(f"mmCIF assembly placement {key!r} has conflicting transforms")
    if not placements:
        raise ValueError(f"mmCIF assembly {assembly_id!r} has no generation rules")
    return placements


def _fractional_symmetry_operation(expression: str) -> np.ndarray:
    r"""Parse one official ``operation_xyz`` affine transform in fractional space."""

    coordinates = expression.lower().replace(" ", "").split(",")
    if len(coordinates) != 3:
        raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for output_axis, coordinate in enumerate(coordinates):
        terms = coordinate.replace("-", "+-").split("+")
        for term in (term for term in terms if term):
            variables = [axis for axis in "xyz" if axis in term]
            if variables:
                if len(variables) != 1 or term not in {variables[0], f"-{variables[0]}"}:
                    raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
                input_axis = "xyz".index(variables[0])
                rotation[output_axis, input_axis] += -1.0 if term.startswith("-") else 1.0
            else:
                try:
                    translation[output_axis] += float(Fraction(term))
                except (ValueError, ZeroDivisionError):
                    raise ValueError(f"invalid mmCIF space-group operation {expression!r}") from None
    determinant = np.linalg.det(rotation)
    if not np.isclose(abs(determinant), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
    return _affine_transform(rotation, translation)


def _mmcif_symmetry_operators(block: CIFBlock) -> dict[str, np.ndarray]:
    r"""Read current and legacy official fractional space-group operation tables."""

    operators: dict[str, np.ndarray] = {}
    for category_name, expression_name in (
        ("space_group_symop", "operation_xyz"),
        ("symmetry_equiv", "pos_as_xyz"),
    ):
        if category_name not in block:
            continue
        category = block[category_name]
        for operator_id, expression in zip(
            _cif_column(category, "id"),
            _cif_column(category, expression_name),
        ):
            if operator_id is None or expression is None:
                raise ValueError(f"mmCIF {category_name} row lacks its id or operation")
            transform = _fractional_symmetry_operation(expression)
            previous = operators.setdefault(operator_id, transform)
            if not np.allclose(previous, transform, rtol=0.0, atol=1e-12):
                raise ValueError(f"mmCIF space-group operation {operator_id!r} has conflicting definitions")
    return operators


class _UnresolvedCrystallographicSymmetry(ValueError):
    r"""A ``struct_conn`` partner references a symmetry operator the file does not resolve.

    A space-group name does not specify a source file's operator numbering.
    The raw source remains readable; a consumer requiring this transformation
    must report that it cannot construct the requested structure.
    """


def _mmcif_symmetry_operator(operators: dict[str, np.ndarray], operator_id: str) -> np.ndarray:
    r"""Resolve one ``n_klm`` operator id to its fractional affine transform.

    The mmCIF ``_space_group_symop`` / ``_symmetry_equiv`` loop is optional: real PDB
    files routinely omit it and give only the space-group name and number, yet still
    reference ``1_555`` / ``1_klm`` partner symmetry codes in ``struct_conn``. Operator
    id ``"1"`` is, by the International Tables / PDB convention, always the identity
    ``x,y,z`` and is listed first in every table, so a reference to it resolves to the
    identity even when the file ships no symop loop. Any other missing operator is
    genuinely unresolvable here -- gemmi's operator ordering does not match the PDB
    ``n_klm`` numbering for centred or higher-symmetry groups -- and is reported via
    :class:`_UnresolvedCrystallographicSymmetry` so the consumer cannot silently omit the selected connection.
    """

    if operator_id in operators:
        return operators[operator_id].copy()
    if operator_id == "1":
        return np.eye(4, dtype=np.float64)
    raise _UnresolvedCrystallographicSymmetry(
        f"mmCIF crystallographic symmetry references unknown operation {operator_id!r}"
    )


def _mmcif_fractionalization(block: CIFBlock) -> np.ndarray:
    r"""Return the homogeneous Cartesian-to-fractional transform declared by mmCIF."""

    if "atom_sites" not in block:
        raise ValueError("mmCIF crystallographic symmetry requires _atom_sites coordinate transforms")
    category = block["atom_sites"]

    def declared_transform(prefix: str) -> np.ndarray | None:
        matrix_names = [f"{prefix}_transf_matrix[{row}][{column}]" for row in (1, 2, 3) for column in (1, 2, 3)]
        vector_names = [f"{prefix}_transf_vector[{row}]" for row in (1, 2, 3)]
        names = matrix_names + vector_names
        values = [(_cif_column(category, name) or (None,))[0] for name in names]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"mmCIF _atom_sites has an incomplete {prefix}_transf transform")
        numeric = np.asarray([category[name].as_array(np.float64)[0] for name in names], dtype=np.float64)
        matrix = numeric[:9].reshape(3, 3)
        vector = numeric[9:]
        return _affine_transform(matrix, vector)

    fractional = declared_transform("fract")
    cartesian = declared_transform("Cartn")
    if fractional is None and cartesian is None:
        raise ValueError("mmCIF crystallographic symmetry requires _atom_sites coordinate transforms")
    if fractional is None:
        assert cartesian is not None
        try:
            fractional = np.linalg.inv(cartesian)
        except np.linalg.LinAlgError:
            raise ValueError("mmCIF _atom_sites Cartesian transform is singular") from None
    elif cartesian is not None and not np.allclose(cartesian @ fractional, np.eye(4), rtol=0.0, atol=1e-5):
        raise ValueError("mmCIF _atom_sites Cartesian and fractional transforms disagree")
    try:
        np.linalg.inv(fractional)
    except np.linalg.LinAlgError:
        raise ValueError("mmCIF _atom_sites fractional transform is singular") from None
    return fractional


def _mmcif_crystallographic_code(symmetry: str | None) -> tuple[str | None, str]:
    r"""Parse one official ``n_klm`` partner-operation code."""

    if symmetry is None:
        return None, "555"
    operator_id, separator, translation_code = symmetry.partition("_")
    if (
        not operator_id.isascii()
        or not operator_id.isdecimal()
        or operator_id != str(int(operator_id))
        or not 1 <= int(operator_id) <= 192
        or (separator and (len(translation_code) != 3 or any(value not in "123456789" for value in translation_code)))
    ):
        raise ValueError(f"invalid mmCIF crystallographic symmetry {symmetry!r}")
    if not separator:
        translation_code = "555"
    return operator_id, translation_code


class SymmetryContext:
    """Resolve each source operation once, keeping source numbering authoritative."""

    def __init__(self, block: CIFBlock) -> None:
        self.block = block
        self._transforms: dict[tuple[str | None, str], tuple[str | None, np.ndarray]] = {}

    @cached_property
    def operators(self) -> dict[str, np.ndarray]:
        return _mmcif_symmetry_operators(self.block)

    @cached_property
    def fractionalization(self) -> np.ndarray:
        return _mmcif_fractionalization(self.block)

    @cached_property
    def orthogonalization(self) -> np.ndarray:
        return np.linalg.inv(self.fractionalization)

    def transform(self, symmetry: str | None) -> tuple[str | None, np.ndarray]:
        code = _mmcif_crystallographic_code(symmetry)
        if code not in self._transforms:
            operator, translation = code
            operation = np.eye(4) if operator is None else _mmcif_symmetry_operator(self.operators, operator)
            operation[:3, 3] += [int(value) - 5 for value in translation]
            result: tuple[str | None, np.ndarray]
            if np.allclose(operation, np.eye(4), rtol=0, atol=1e-12):
                result = None, np.eye(4)
            else:
                transform = self.orthogonalization @ operation @ self.fractionalization
                if not np.isfinite(transform).all():
                    raise ValueError(f"mmCIF crystallographic symmetry {symmetry!r} produces a non-finite transform")
                result = f"{operator}_{translation}", transform
            result[1].setflags(write=False)
            self._transforms[code] = result
        return self._transforms[code]

    def pair(
        self, first: str | None, second: str | None
    ) -> tuple[tuple[str | None, np.ndarray], tuple[str | None, np.ndarray]]:
        a, b = _mmcif_crystallographic_code(first), _mmcif_crystallographic_code(second)
        if a == b:
            if (
                a[0] not in {None, "1"}
                and ("space_group_symop" in self.block or "symmetry_equiv" in self.block)
                and a[0] not in self.operators
            ):
                raise _UnresolvedCrystallographicSymmetry(
                    f"mmCIF crystallographic symmetry references unknown operation {a[0]!r}"
                )
            identity = self.transform(None)
            return identity, identity
        return self.transform(first), self.transform(second)


class PlacementIndex:
    """Join connection partners to selected copies by unique source operations.

    The common case joins equal copy IDs in linear time. Different crystallographic
    frames require a numerical transform comparison, performed once per unique
    asym/frame pair, not once per residue or bond.
    """

    def __init__(self, block: CIFBlock, placements: Mapping[tuple[str | None, str], np.ndarray]) -> None:
        self.frames = SymmetryContext(block)
        self.by_asym: defaultdict[str, dict[str | None, np.ndarray]] = defaultdict(dict)
        for (copy, asym), transform in placements.items():
            self.by_asym[asym][copy] = transform
        self._pairs: dict[tuple[str, str, str | None, str | None], tuple[tuple[str | None, str | None], ...]] = {}
        self._inverses: dict[int, np.ndarray] = {}

    def pairs(
        self, asym_a: str, asym_b: str, symmetry_a: str | None, symmetry_b: str | None
    ) -> tuple[tuple[str | None, str | None], ...]:
        key = asym_a, asym_b, symmetry_a, symmetry_b
        if key not in self._pairs:
            (_, a), (_, b) = self.frames.pair(symmetry_a, symmetry_b)
            left, right = self.by_asym[asym_a], self.by_asym[asym_b]
            if np.allclose(a, b, rtol=0, atol=1e-8):
                # Invert once even here: the same-frame comparison uses the
                # same numerical expression/tolerance as distinct frames.
                inverse_a, inverse_b = self._inverse(a), self._inverse(b)
                pairs = tuple(
                    (copy, copy)
                    for copy in left
                    if copy in right and np.allclose(left[copy] @ inverse_a, right[copy] @ inverse_b, rtol=0, atol=1e-5)
                )
            else:
                inv_a, inv_b = self._inverse(a), self._inverse(b)
                transformed_left = [(copy, transform @ inv_a) for copy, transform in left.items()]
                transformed_right = [(copy, transform @ inv_b) for copy, transform in right.items()]
                pairs = tuple(
                    (copy_a, copy_b)
                    for copy_a, transform_a in transformed_left
                    for copy_b, transform_b in transformed_right
                    if np.allclose(transform_a, transform_b, rtol=0, atol=1e-5)
                )
            self._pairs[key] = pairs
        return self._pairs[key]

    def _inverse(self, matrix: np.ndarray) -> np.ndarray:
        key = id(matrix)
        if key not in self._inverses:
            self._inverses[key] = np.linalg.inv(matrix)
        return self._inverses[key]
