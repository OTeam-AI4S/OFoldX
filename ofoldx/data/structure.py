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

r"""Molecular structures, coordinate evidence, and coordinate replacement.

`Structure` combines a `MolecularSystem` with coordinates, observations, source
identity, copy identity, and an experimental frame. `CoordinateEvidence` identifies
observed, predicted, generated, reference, or unavailable coordinates; unavailable
coordinates have zero payload. Method, resolution, cell, frame, and applied operations
retain the source of each coordinate set.

The cell `(a, b, c, alpha, beta, gamma)` and the Cartesian-to-fractional frame
`f = A @ x + u` have separate states. A frame can be declared by the source,
derived from its cell, or inapplicable for a noncrystal experiment's placeholder
cell. Its transform and unit are retained when no cell value is known.

`coordinate_quantum`, `cell_quantum`, and `frame_quantum` record the unit of the last
printed source digit. Coordinate quantum uses the largest unit across XYZ. These
values describe representation rounding, rather than experimental precision, and
are zero when no source token supplied the value. Moved, replaced, derived, and
predicted values therefore receive no inherited printed precision. A declared
frame obtained by inverting `_atom_sites.Cartn_transf_*` retains the printed units
propagated through that inverse for subsequent numerical decisions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, fail
from ..provenance import Basis, Resolution, known, state
from . import projection
from .molecular import EntityType, MolecularSystem
from .schema import ArrayRecord, Field, Schema, check_mask, check_rows, publish, validate_structure


class CoordinateEvidence(IntEnum):
    NONE = 0
    OBSERVED = 1
    PREDICTED = 2
    GENERATED = 3
    REFERENCE = 4


class CopyKind(IntEnum):
    DEPOSITED = 0
    ASSEMBLY = 1
    CRYSTAL = 2
    RECIPE = 3


class ResolutionKind(IntEnum):
    r"""The source category a resolution value comes from; ``NONE`` names no source category."""

    NONE = 0
    REFINEMENT = 1  # _refine.ls_d_res_high
    REFLECTIONS = 2  # _reflns.d_resolution_high
    EM_RECONSTRUCTION = 3  # _em_3d_reconstruction.resolution
    PDB_REMARK_2 = 4


class OperationNumbering(IntEnum):
    r"""
    Which operation list numbers the source's symmetry codes. ``FILE``: the file's own table, the only list the
    operation rows store. ``ITA``: the full setting the KNOWN space group names (origin choice and basis included,
    never the bare number), which ``io.symmetry.space_group_setting`` regenerates, so no row is stored; a
    wwPDB entry whose table lists the setting's operations in another order numbers its codes so too.
    ``NONE``: no list in force.
    """

    NONE = 0
    FILE = 1
    ITA = 2


@dataclass(frozen=True, slots=True)
class CopyId:
    r"""
    What generated one physical copy of one asym; equality is by value, the display symbol is derived, never parsed.

    Examples:
        >>> CopyId(CopyKind.CRYSTAL, ("3_655",), "A") == CopyId(CopyKind.CRYSTAL, ("3_655",), "A")
        True
    """

    kind: CopyKind
    operators: tuple[str, ...]
    asym: str | None

    @property
    def operator_symbol(self) -> str:
        r"""The exact encoding of ``operators`` in the ``operator`` symbol family, decoded by ``copy_id``."""
        return _OPERATOR_SEPARATOR.join(self.operators)


@dataclass(frozen=True, slots=True)
class StructureArrays:
    r"""The mutable build buffer of a ``Structure``: every column and every symbol family."""

    symbols: Mapping[str, tuple[str, ...]]
    columns: MutableMapping[str, NDArray]

    def gather(
        self, atoms: NDArray[np.int64], components: NDArray[np.int64], instances: NDArray[np.int64]
    ) -> StructureArrays:
        r"""
        New buffers holding the rows ``atoms``, ``components`` and ``instances`` (in those orders) of the per-atom,
        -component and -instance columns; system rows are kept. The columns of ``self`` are never written.
        """

        s = dict(self.columns)
        for field in Structure.SCHEMA.fields:
            rows = {"atom": atoms, "component": components, "instance": instances}.get(field.axis)
            if rows is not None:
                s[field.name] = s[field.name][rows]
        return StructureArrays(self.symbols, s)


_FIELDS = (
    Field("coordinates", "float64", "atom", (3,)),
    Field("coordinate_evidence", "uint8", "atom"),
    Field("coordinate_quantum", "float64", "atom"),
    *Field("occupancy", "float64", "atom").stated(),
    *Field("occupancy_uncertainty", "float64", "atom").stated(),  # its standard uncertainty u
    *Field("b_factor", "float64", "atom").stated(),
    *Field("atom_site_charge", "int8", "atom").stated(),
    Field("atom_altloc", "int32", "atom", symbols="altloc"),
    *Field("component_sequence_number", "int32", "component").stated(),
    Field("component_author_text", "int32", "component", symbols="author_number"),
    *Field("component_author_number", "int64", "component").stated(),
    Field("component_insertion", "int32", "component", symbols="insertion"),
    Field("instance_copy_kind", "uint8", "instance"),
    Field("instance_copy_operators", "int32", "instance", symbols="operator"),
    Field("instance_source_asym", "int32", "instance", symbols="asym"),
    *Field("instance_author_chain", "int32", "instance", symbols="chain").stated(),
    Field("source_sha256", "uint8", "system", (32,)),
    *Field("system_model_number", "int32", "system").stated(),
    *Field("cell", "float64", "system", (6,)).stated(),
    Field("cell_quantum", "float64", "system", (6,)),
    *Field("frame", "float64", "system", (3, 4)).stated(),  # [A | u]: f = A·x + u
    Field("frame_quantum", "float64", "system", (3, 4)),
    *Field("space_group", "int32", "system", symbols="space_group").stated(),
    Field("operation_numbering", "uint8", "system"),
    Field("system_method_ptr", "int64", "system+1"),
    Field("system_resolution_ptr", "int64", "system+1"),
    Field("operation_ptr", "int64", "system+1"),
    Field("method", "int32", "method", symbols="method"),
    Field("resolution", "float64", "resolution"),
    Field("resolution_kind", "uint8", "resolution"),
    Field("resolution_state", "uint8", "resolution", state=True),
    Field("operation_id", "int32", "operation", symbols="operation"),
    Field("operation_rotation", "int8", "operation", (3, 3)),
    Field("operation_translation", "int16", "operation", (3,)),
)


_AXES = ("atom", "component", "instance", "system", "method", "resolution", "operation")
_FAMILIES = ("asym", "chain", "author_number", "insertion", "altloc", "operator", "method", "space_group", "operation")
_OPERATOR_SEPARATOR = "\x1f"  # no CIF 1.1/2.0 or PDB token contains it: an exact encoding of an operator tuple
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_UNKNOWN_NONE = state(Resolution.UNKNOWN, Basis.NONE)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)


class Structure(ArrayRecord):
    r"""
    A frozen, validated structure: ``system`` plus its coordinate, observation and frame columns.

    ``from_arrays`` is the only construction; ``build`` returns writable copies of the structure columns (operations
    rebuild the system and the structure together).
    """

    SCHEMA: ClassVar[Schema] = Schema(_FIELDS, _AXES)
    __slots__ = ("system",)
    system: MolecularSystem

    @classmethod
    def _with_system(cls, system: MolecularSystem, columns, symbols) -> Structure:
        value = cls._from_published(columns, symbols)
        object.__setattr__(value, "system", system)
        return value

    @classmethod
    def from_arrays(cls, system: MolecularSystem, arrays: StructureArrays) -> Structure:
        r"""Publish read-only copies of ``arrays``, validate them against ``system`` once and wrap them."""
        columns = publish(arrays.columns, arrays.symbols, cls.SCHEMA)
        validate_structure(system.arrays(), columns, arrays.symbols)
        return cls._with_system(system, columns, arrays.symbols)

    def __reduce__(self):
        return type(self)._with_system, (self.system, dict(self.arrays()), dict(self.symbols()))

    @classmethod
    def unobserved(cls, system: MolecularSystem) -> Structure:
        r"""
        A structure of ``system`` that states no observation: every coordinate ``NONE``, no source document, no frame;
        OFoldX numbers the components of polymer and branched entities 1..k per instance.
        """

        s = system.arrays()
        n, c, i = len(s["atom_symbol"]), len(s["component_symbol"]), len(s["instance_symbol"])
        systems = len(s["system_instance_ptr"]) - 1
        component_instance = np.repeat(np.arange(i), np.diff(s["instance_component_ptr"]))
        entity = s["instance_entity"][component_instance]
        typed = known(s["entity_type_state"][entity])
        numbered = typed & np.isin(s["entity_type"][entity], (EntityType.POLYMER, EntityType.BRANCHED))
        ordinal = np.arange(c) - s["instance_component_ptr"][component_instance] + 1
        sequence_state = np.where(numbered, _KNOWN_DERIVED, _INAPPLICABLE_DEFINED)
        sequence_state = np.where(typed, sequence_state, _UNKNOWN_NONE)
        known_number = sequence_state == _KNOWN_DERIVED

        def full(count, value, dtype, shape=()):
            return np.full((count, *shape), value, dtype=dtype)

        columns = {
            "coordinates": full(n, 0, np.float64, (3,)),
            "coordinate_evidence": full(n, CoordinateEvidence.NONE, np.uint8),
            "coordinate_quantum": full(n, 0, np.float64),
            "occupancy": full(n, 0, np.float64),
            "occupancy_state": full(n, _INAPPLICABLE_DEFINED, np.uint8),
            "occupancy_uncertainty": full(n, 0, np.float64),
            "occupancy_uncertainty_state": full(n, _INAPPLICABLE_DEFINED, np.uint8),
            "b_factor": full(n, 0, np.float64),
            "b_factor_state": full(n, _INAPPLICABLE_DEFINED, np.uint8),
            "atom_site_charge": full(n, 0, np.int8),
            "atom_site_charge_state": full(n, _UNKNOWN_NONE, np.uint8),
            "atom_altloc": full(n, -1, np.int32),
            "component_sequence_number": np.where(known_number, ordinal, 0).astype(np.int32),
            "component_sequence_number_state": sequence_state.astype(np.uint8),
            "component_author_text": full(c, -1, np.int32),
            "component_author_number": full(c, 0, np.int64),
            "component_author_number_state": full(c, _UNKNOWN_NONE, np.uint8),
            "component_insertion": full(c, -1, np.int32),
            "instance_copy_kind": full(i, CopyKind.RECIPE, np.uint8),
            "instance_copy_operators": full(i, -1, np.int32),
            "instance_source_asym": full(i, -1, np.int32),
            "instance_author_chain": full(i, -1, np.int32),
            "instance_author_chain_state": full(i, _UNKNOWN_NONE, np.uint8),
            "source_sha256": full(systems, 0, np.uint8, (32,)),
            "system_model_number": full(systems, 0, np.int32),
            "system_model_number_state": full(systems, _UNKNOWN_NONE, np.uint8),
            "cell": full(systems, 0, np.float64, (6,)),
            "cell_state": full(systems, _UNKNOWN_NONE, np.uint8),
            "cell_quantum": full(systems, 0, np.float64, (6,)),
            "frame": full(systems, 0, np.float64, (3, 4)),
            "frame_state": full(systems, _UNKNOWN_NONE, np.uint8),
            "frame_quantum": full(systems, 0, np.float64, (3, 4)),
            "space_group": full(systems, -1, np.int32),
            "space_group_state": full(systems, _UNKNOWN_NONE, np.uint8),
            "operation_numbering": full(systems, OperationNumbering.NONE, np.uint8),
            "system_method_ptr": full(systems + 1, 0, np.int64),
            "system_resolution_ptr": full(systems + 1, 0, np.int64),
            "operation_ptr": full(systems + 1, 0, np.int64),
            "method": full(0, 0, np.int32),
            "resolution": full(0, 0, np.float64),
            "resolution_kind": full(0, 0, np.uint8),
            "resolution_state": full(0, 0, np.uint8),
            "operation_id": full(0, 0, np.int32),
            "operation_rotation": full(0, 0, np.int8, (3, 3)),
            "operation_translation": full(0, 0, np.int16, (3,)),
        }
        return cls.from_arrays(system, StructureArrays({family: () for family in _FAMILIES}, columns))

    def build(self) -> StructureArrays:
        r"""Writable, independent copies of the structure columns and families (the system is not included)."""
        return StructureArrays(dict(self.symbols()), {name: array.copy() for name, array in self.arrays().items()})

    def copy_id(self, instance: int) -> CopyId:
        r"""The ``CopyId`` of one instance."""
        instance = int(check_rows("instance", instance, len(self.instance_copy_kind), shape=()))
        operators = int(self.instance_copy_operators[instance])
        asym = int(self.instance_source_asym[instance])
        return CopyId(
            CopyKind(int(self.instance_copy_kind[instance])),
            () if operators < 0 else tuple(self.symbols()["operator"][operators].split(_OPERATOR_SEPARATOR)),
            None if asym < 0 else self.symbols()["asym"][asym],
        )

    def digest(self) -> str:
        r"""sha256 over both schema digests, every column (name, dtype, shape, little-endian bytes) and every family."""
        if "digest" not in self._cache:
            hasher = hashlib.sha256()
            for record in (self.system, self):
                hasher.update(record.SCHEMA.digest().encode())
                for name in sorted(record.arrays()):
                    array = record.arrays()[name]
                    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
                    hasher.update(f"{name}\x1f{array.dtype.str}\x1f{array.shape}\x1e".encode())
                    hasher.update(np.ascontiguousarray(little).tobytes())
                for family in sorted(record.symbols()):
                    table = record.symbols()[family]
                    hasher.update(f"{family}\x1f{len(table)}\x1e".encode())
                    for text in table:
                        encoded = text.encode()
                        hasher.update(len(encoded).to_bytes(8, "little") + encoded)
            self._cache["digest"] = hasher.hexdigest()
        return self._cache["digest"]

    @property
    def coordinate_mask(self) -> NDArray[np.bool_]:
        r"""``coordinate_evidence != NONE``."""
        return self.coordinate_evidence != CoordinateEvidence.NONE

    def select(
        self,
        *,
        chains: Sequence[str] | None = None,
        namespace: Literal["author", "label"] = "author",
        water: bool = True,
        hydrogens: bool = True,
    ) -> Structure:
        r"""Select chains and choose whether to retain water and explicit hydrogens.

        Args:
            chains: Author chain names, or label asym IDs when `namespace="label"`.
                None retains all chains. An author chain can match several label instances.
            namespace: Namespace used to interpret `chains`.
            water: Whether to retain water entities.
            hydrogens: Whether to retain explicit hydrogens. False folds them into counts.

        Returns:
            A structure with the selected molecular view. The source remains unchanged.

        Raises:
            InvalidRequestError: A requested chain is absent or the namespace is invalid.

        Notes:
            Selection preserves canonical component identities, external connections, and
            applicable stereochemistry. See `ofoldx.data.projection.select` for the index
            and completeness rules.
        """
        return projection.select(self, chains=chains, namespace=namespace, water=water, hydrogens=hydrogens)


@dataclass(frozen=True)
class CoordinateReport:
    r"""What a coordinate replacement wrote, invalidated and withdrew."""

    written: NDArray[np.int64]
    invalidated_unwritten: NDArray[np.int64]
    observations_withdrawn: int
    resolution_withdrawn: bool
    method_withdrawn: bool
    frame_withdrawn: bool


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _segments(pointer: NDArray[np.int64], keep: NDArray[np.bool_]) -> NDArray[np.int64]:
    r"""The pointer after emptying the segments of the systems not in ``keep``."""
    return np.concatenate(([0], np.cumsum(np.where(keep, np.diff(pointer), 0)))).astype(np.int64)


def replace_coordinates(
    structure: Structure,
    atoms: NDArray[np.int64],
    coordinates: NDArray[np.float64],
    *,
    scope: NDArray[np.bool_],
    evidence: CoordinateEvidence,
    frame: Literal["source", "new"] = "source",
) -> tuple[Structure, CoordinateReport]:
    r"""
    Regenerate the coordinates of the target range ``scope`` from one output.

    The listed ``atoms`` take ``coordinates`` and ``evidence``; scope atoms not listed become ``NONE`` (old coordinates
    are never refilled) with printed quantum 0; occupancy and B of scope atoms become INAPPLICABLE×DEFINED (never
    inherited). ``OBSERVED`` copies another
    source's observation into an instance that is not deposited: no source printed a deposited instance's regenerated
    coordinate (``value_invalid``, T2). Atoms outside scope keep everything. Resolution is withdrawn where an OBSERVED
    coordinate was in scope, method where no OBSERVED coordinate remains; ``frame="new"`` (scope = every atom) also
    withdraws cell, frame, space group and operation table. The graph is untouched.
    """

    n = len(structure.coordinates)
    # the raw request, before any of it indexes: distinct atoms of the value, one point each, a mask of every atom
    atoms, scope = check_rows("atoms", atoms, n, shape=(None,)), check_mask("scope", scope, n)
    coordinates = np.asarray(coordinates)
    if len(np.unique(atoms)) != len(atoms):
        fail("value_invalid", RequestRef("atoms", str(atoms.tolist()[:8])), invariant="distinct_rows")
    if coordinates.shape != (len(atoms), 3):
        fail("value_invalid", RequestRef("coordinates", str(coordinates.shape)), expected=(len(atoms), 3))
    if evidence not in tuple(CoordinateEvidence) or evidence == CoordinateEvidence.NONE:
        fail("value_invalid", RequestRef("evidence", str(evidence)))
    if frame not in ("source", "new"):
        fail("value_invalid", RequestRef("frame", str(frame)))
    if not scope[atoms].all():
        fail("scope_violation", RequestRef("atoms", str(atoms[~scope[atoms]][:8].tolist())))
    if frame == "new" and not scope.all():
        fail("scope_violation", RequestRef("frame", "new"), unscoped=int((~scope).sum()))
    arrays = structure.build()
    cols, old = arrays.columns, structure.coordinate_evidence
    cols["coordinates"][scope] = 0
    cols["coordinate_evidence"][scope] = CoordinateEvidence.NONE
    cols["coordinate_quantum"][scope] = 0
    cols["coordinates"][atoms] = coordinates
    cols["coordinate_evidence"][atoms] = evidence
    withdrawn = scope & (
        (cols["occupancy_state"] != _INAPPLICABLE_DEFINED) | (cols["b_factor_state"] != _INAPPLICABLE_DEFINED)
    )
    for name in ("occupancy", "occupancy_uncertainty", "b_factor"):
        cols[name][scope] = 0
        cols[name + "_state"][scope] = _INAPPLICABLE_DEFINED
    s = structure.system.arrays()
    atom_system = _owner(s["system_instance_ptr"])[_owner(s["instance_component_ptr"])[_owner(s["component_atom_ptr"])]]
    systems = len(cols["system_model_number"])
    observed = CoordinateEvidence.OBSERVED
    lose_resolution = np.bincount(atom_system[scope & (old == observed)], minlength=systems) > 0
    lose_method = np.bincount(atom_system[cols["coordinate_evidence"] == observed], minlength=systems) == 0
    lose_resolution &= np.diff(cols["system_resolution_ptr"]) > 0  # only a stated row can be withdrawn
    lose_method &= np.diff(cols["system_method_ptr"]) > 0
    for pointer, payload, lose in (
        ("system_resolution_ptr", ("resolution", "resolution_kind", "resolution_state"), lose_resolution),
        ("system_method_ptr", ("method",), lose_method),
    ):
        rows = np.repeat(~lose, np.diff(cols[pointer]))
        for name in payload:
            cols[name] = cols[name][rows]
        cols[pointer] = _segments(cols[pointer], ~lose)
    if frame == "new":
        for name in ("cell", "frame"):
            cols[name][:] = 0
            cols[name + "_state"][:] = _UNKNOWN_NONE
            cols[name + "_quantum"][:] = 0
        cols["space_group"][:] = -1
        cols["space_group_state"][:] = _UNKNOWN_NONE
        cols["operation_numbering"][:] = OperationNumbering.NONE
        for name in ("operation_id", "operation_rotation", "operation_translation"):
            cols[name] = cols[name][:0]
        cols["operation_ptr"][:] = 0
    replaced = Structure.from_arrays(structure.system, arrays)
    report = CoordinateReport(
        written=atoms.copy(),
        invalidated_unwritten=np.flatnonzero(scope & ~np.isin(np.arange(n), atoms)).astype(np.int64),
        observations_withdrawn=int(withdrawn.sum()),
        resolution_withdrawn=bool(lose_resolution.any()),
        method_withdrawn=bool(lose_method.any()),
        frame_withdrawn=frame == "new",
    )
    return replaced, report
