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
The ``MolecularSystem`` value: a coordinate-free molecular graph with external connections carrying
exact transforms, the reference-conformer pool, its single constructor and sorted-key lookups in both directions.

Axes: system → entity → instance → component → atom; edge; external; transform; stereocentre, stereobond, stereogroup;
reference pool. Every fact that can be unknown carries a ``<column>_state`` byte (``ofoldx.provenance``); symbol codes
index per-vocabulary families that projections never renumber. ``MolecularSystem`` itself is the bond graph: no second
graph object is stored.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, fail
from ..provenance import Resolution, resolution
from .schema import ArrayRecord, Field, Schema, check_rows, publish, validate_system

#: Element symbols indexed by atomic number; ``X`` (0) is a source atom whose element is explicitly unknown.
ELEMENTS: list[str] = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og"
).split()
#: The groups of each period in atomic-number order (the lanthanides and actinides are group 3).
_PERIOD_GROUPS = (
    (1, 18),
    (1, 2, *range(13, 19)),
    (1, 2, *range(13, 19)),
    tuple(range(1, 19)),
    tuple(range(1, 19)),
    (1, 2, *(3,) * 15, *range(4, 19)),
    (1, 2, *(3,) * 15, *range(4, 19)),
)
#: Periodic-table group and period of each element, indexed by atomic number like ``ELEMENTS`` (0 for ``X``).
GROUP_BY_ATOMIC_NUMBER: tuple[int, ...] = (0, *(group for groups in _PERIOD_GROUPS for group in groups))
PERIOD_BY_ATOMIC_NUMBER: tuple[int, ...] = (0, *(k for k, groups in enumerate(_PERIOD_GROUPS, 1) for _ in groups))


class BondKind(IntEnum):
    r"""The kind of a connection; 0 is the canonical zero of an undetermined kind."""

    COVALENT = 1
    COORDINATION = 2


class BondOrigin(IntEnum):
    r"""What produced an edge or external connection."""

    REFERENCE = 0
    JUNCTION = 1
    BRANCH_LINK = 2
    CONNECTION = 3
    CAP = 4
    RECIPE = 5


class AtomFate(IntEnum):
    r"""Atom fates; ``LEFT`` exists only in an ``AttachmentResult`` (left atoms do not exist in a value)."""

    RETAINED = 0
    UNDETERMINED = 1
    UNOBSERVED_ENDPOINT = 2
    LEFT = 3


class EndpointState(IntEnum):
    r"""The attachment outcome of a native endpoint (not a fact state byte); 0 in a column means "not an endpoint"."""

    DECLARED_BY_SOURCE = 1
    DERIVED_FROM_REFERENCE = 2
    DERIVED_FROM_VALENCE = 3
    INAPPLICABLE = 4
    UNKNOWN = 5
    CONFLICT = 6


class MonomerType(IntEnum):
    NONE = 0
    PROTEIN = 1
    RNA = 2
    DNA = 3


class ReferenceSource(IntEnum):
    r"""The conformer a definition supplies: ideal if complete, else model; never mixed per atom."""

    NONE = 0
    IDEAL = 1
    MODEL = 2
    CONTRADICTED = 3


class DefinitionScope(IntEnum):
    r"""The scope of the definition used for a component; ``NONE`` is a component without a definition."""

    NONE = 0
    ARCHIVE = 1
    CALLER = 2
    FILE = 3
    SMILES = 4
    RECIPE = 5


class EntityType(IntEnum):
    r"""``_entity.type`` (stated with ``entity_type_state``); 0 is the canonical zero of an undetermined type."""

    POLYMER = 1
    NON_POLYMER = 2
    BRANCHED = 3
    MACROLIDE = 4
    WATER = 5


class NstdLinkage(IntEnum):
    r"""``_entity_poly.nstd_linkage`` as stated (``entity_nstd_linkage_state``; no dictionary default)."""

    NO = 1
    YES = 2


class Designation(IntEnum):
    r"""Whether an atom lies in the designated leaving fragment of a port (a definition fact)."""

    NONE = 0
    PREVIOUS = 1
    NEXT = 2


_ENDPOINT_RESOLUTION = np.array(
    [Resolution.INAPPLICABLE, Resolution.KNOWN, Resolution.KNOWN, Resolution.KNOWN]
    + [Resolution.INAPPLICABLE, Resolution.UNKNOWN, Resolution.CONFLICT],
    dtype=np.uint8,
)


def endpoint_resolution(states: NDArray[np.uint8]) -> NDArray[np.uint8]:
    r"""
    The one projection of ``EndpointState`` onto ``Resolution``.

    Declared and derived outcomes are KNOWN, ``INAPPLICABLE`` and "not an endpoint" (0) are INAPPLICABLE, ``UNKNOWN``
    and ``CONFLICT`` keep their resolution.

    Examples:
        >>> endpoint_resolution(np.array([0, 2, 5, 6], dtype=np.uint8)).tolist()
        [2, 0, 1, 3]
    """

    return _ENDPOINT_RESOLUTION[check_rows("states", states, len(_ENDPOINT_RESOLUTION))]


@dataclass(frozen=True)
class Transforms:
    r"""
    Exact operations in the source's fractional basis.

    ``translation`` holds numerators over 24 with the lattice shift included. ``code`` is the stated or derived
    operation code packed as ``n·1000 + 100k + 10l + m`` (``n_klm``; ``-1`` none); it is independent of whether
    the matrix resolved. In the ``MolecularSystem`` table, row 0 is the identity (KNOWN×DEFINED) and the states are
    KNOWN×{DEFINED, DERIVED}, UNKNOWN×{NONE, DECLARED} or CONFLICT×NONE; an operation table of a ``Structure`` keeps
    file order and may hold KNOWN×DECLARED rows.
    """

    rotation: NDArray[np.int8]
    translation: NDArray[np.int16]
    code: NDArray[np.int32]
    state: NDArray[np.uint8]
    IDENTITY: ClassVar[int] = 0


@dataclass(frozen=True, slots=True)
class SystemArrays:
    r"""The mutable build buffer of a ``MolecularSystem``: every column and every symbol family, no defaults."""

    symbols: Mapping[str, tuple[str, ...]]
    columns: MutableMapping[str, NDArray]


@dataclass(frozen=True, slots=True)
class AtomSite:
    r"""An atom named by instance symbol, component ordinal within the instance and atom name."""

    instance: str
    component: int
    atom: str


@dataclass(frozen=True, slots=True)
class ComponentRef:
    r"""A component named by instance symbol and its ordinal within the instance."""

    instance: str
    component: int


_FIELDS = (
    Field("system_entity_ptr", "int64", "system+1"),
    Field("system_instance_ptr", "int64", "system+1"),
    Field("entity_symbol", "int32", "entity", symbols="entity"),
    *Field("entity_type", "uint8", "entity").stated(),
    *Field("entity_polymer_type", "int32", "entity", symbols="polymer_type").stated(),
    *Field("entity_nstd_linkage", "uint8", "entity").stated(),
    Field("instance_symbol", "int32", "instance", symbols="instance"),
    Field("instance_entity", "int32", "instance"),
    Field("instance_connections_complete", "bool", "instance"),
    Field("instance_component_ptr", "int64", "instance+1"),
    Field("component_symbol", "int32", "component", symbols="component"),
    Field("component_definition_scope", "uint8", "component"),
    Field("component_definition_digest", "int64", "component"),
    *Field("component_monomer_type", "uint8", "component").stated(),
    Field("component_port", "int32", "component", (2,)),
    Field("component_topology_complete", "bool", "component"),
    Field("component_heavy_atoms_complete", "bool", "component"),
    Field("component_atom_ptr", "int64", "component+1"),
    Field("component_parent_ptr", "int64", "component+1"),
    Field("component_parent", "int32", "component_parent", symbols="parent"),
    Field("component_definition_parent_ptr", "int64", "component+1"),
    Field("component_definition_parent", "int32", "component_definition_parent", symbols="parent"),
    Field("atom_symbol", "int32", "atom", symbols="atom"),
    *Field("atomic_number", "uint8", "atom").stated(),
    Field("isotope", "uint16", "atom"),  # the mass number, 0 for none stated
    *Field("formal_charge", "int8", "atom").stated(),
    *Field("implicit_hydrogens", "int8", "atom").stated(),
    *Field("atom_valence", "int8", "atom").stated(),
    *Field("atom_aromatic", "bool", "atom").stated(),
    Field("atom_fate", "uint8", "atom"),
    Field("atom_endpoint_state", "uint8", "atom"),
    Field("atom_designation", "uint8", "atom"),
    Field("atom_reference_row", "int32", "atom"),
    Field("reference_pool_coordinates", "float32", "reference", (3,)),
    Field("reference_pool_source", "uint8", "reference"),
    Field("edge_index", "int32", "edge", leading=(2,)),
    *Field("bond_order", "uint8", "edge").stated(),
    *Field("bond_aromatic", "bool", "edge").stated(),
    *Field("bond_kind", "uint8", "edge").stated(),
    *Field("bond_direction", "int8", "edge").stated(),
    Field("bond_origin", "uint8", "edge"),
    Field("bond_connection_type", "int32", "edge", symbols="connection_type"),
    Field("bond_declared_leaving_ptr", "int64", "edge+1"),
    Field("bond_declared_leaving", "int32", "bond_declared_leaving", symbols="atom"),
    Field("bond_declared_leaving_side", "uint8", "bond_declared_leaving"),
    Field("external_atom", "int32", "external"),
    Field("external_partner_instance", "int32", "external", symbols="instance"),
    Field("external_partner_component", "int32", "external"),
    Field("external_partner_atom", "int32", "external", symbols="atom"),
    Field("external_transform", "int32", "external"),
    *Field("external_order", "uint8", "external").stated(),
    *Field("external_aromatic", "bool", "external").stated(),
    *Field("external_kind", "uint8", "external").stated(),
    *Field("external_direction", "int8", "external").stated(),
    Field("external_origin", "uint8", "external"),
    Field("external_connection_type", "int32", "external", symbols="connection_type"),
    Field("external_declared_leaving_ptr", "int64", "external+1"),
    Field("external_declared_leaving", "int32", "external_declared_leaving", symbols="atom"),
    Field("external_declared_leaving_side", "uint8", "external_declared_leaving"),
    Field("transform_rotation", "int8", "transform", (3, 3)),
    Field("transform_translation", "int16", "transform", (3,)),
    Field("transform_code", "int32", "transform"),
    Field("transform_state", "uint8", "transform", state=True),
    Field("stereocentre_atom", "int32", "stereocentre"),
    Field("stereocentre_parity", "int8", "stereocentre"),
    Field("stereocentre_neighbors", "int32", "stereocentre", (4,)),
    Field("stereocentre_state", "uint8", "stereocentre", state=True),
    Field("stereobond_edge", "int32", "stereobond"),
    Field("stereobond_value", "int8", "stereobond"),
    Field("stereobond_reference", "int32", "stereobond", (2,)),
    Field("stereobond_state", "uint8", "stereobond", state=True),
    Field("stereogroup_kind", "uint8", "stereogroup"),
    Field("stereogroup_ptr", "int64", "stereogroup+1"),
    Field("stereogroup_member", "int32", "stereogroup_member"),
)
_AXES: tuple[str, ...] = (
    "system",
    "entity",
    "instance",
    "component",
    "component_parent",
    "component_definition_parent",
    "atom",
)
_AXES += ("reference", "edge")
_AXES += ("bond_declared_leaving", "external", "external_declared_leaving", "transform", "stereocentre")
_AXES += ("stereobond", "stereogroup", "stereogroup_member")


class MolecularSystem(ArrayRecord):
    r"""
    A frozen, validated molecular graph.

    ``from_arrays`` is the only construction; ``build`` returns a writable copy for the operations of ``ofoldx.data``.
    Lookups return ``-1`` for a key that names nothing in the value; each caller raises its own condition.
    """

    SCHEMA: ClassVar[Schema] = Schema(_FIELDS, _AXES)
    __slots__ = ()

    @classmethod
    def from_arrays(cls, arrays: SystemArrays) -> MolecularSystem:
        r"""Publish read-only copies of ``arrays``, validate them once and wrap them."""
        columns = publish(arrays.columns, arrays.symbols, cls.SCHEMA)
        validate_system(columns, arrays.symbols)
        return cls._from_published(columns, arrays.symbols)

    def build(self) -> SystemArrays:
        r"""Writable, independent copies of every column and family."""
        return SystemArrays(dict(self.symbols()), {name: array.copy() for name, array in self.arrays().items()})

    @property
    def component_hydrogens_complete(self) -> NDArray[np.bool_]:
        r"""
        Per component, whether every hydrogen-counting atom (any but a hydrogen) has a decided hydrogen count now: its
        ``implicit_hydrogens_state`` is KNOWN, or INAPPLICABLE (no count applies to decide); an UNKNOWN or
        CONFLICT count is not decided. A reading of the counts the value holds, not of how it was made: no claim of
        materialised hydrogens or of a full reference atom set, and no edit or projection history denies it.
        """

        if "hydrogens" not in self._cache:
            count = len(self.component_symbol)
            owner = np.repeat(np.arange(count, dtype=np.int64), np.diff(self.component_atom_ptr))
            decided = np.isin(resolution(self.implicit_hydrogens_state), (Resolution.KNOWN, Resolution.INAPPLICABLE))
            undecided = (self.atomic_number != 1) & ~decided
            complete = np.bincount(owner[undecided], minlength=count) == 0
            complete.flags.writeable = False
            self._cache["hydrogens"] = complete
        return self._cache["hydrogens"]

    def _codes(self, family: str) -> dict[str, int]:
        key = ("codes", family)
        if key not in self._cache:
            self._cache[key] = {text: code for code, text in enumerate(self.symbols()[family])}
        return self._cache[key]

    def instance_index(self, symbols: Sequence[str]) -> NDArray[np.int64]:
        r"""
        The instance carrying each symbol, ``-1`` for a symbol of no instance.

        A symbol carried by instances of several systems (a batch) is ambiguous: ``value_invalid``.
        """

        if "instance" not in self._cache:
            codes = self.instance_symbol.astype(np.int64)
            table = np.full(len(self.symbols()["instance"]) + 1, -1, dtype=np.int64)
            counts = np.bincount(codes, minlength=len(table))
            table[codes] = np.arange(len(codes))
            table[counts > 1] = -2
            self._cache["instance"] = table
        family = self._codes("instance")
        query = np.array([family.get(symbol, -1) for symbol in symbols], dtype=np.int64)
        found = self._cache["instance"][query]
        if (found == -2).any():
            ambiguous = symbols[int(np.flatnonzero(found == -2)[0])]
            fail("value_invalid", RequestRef("instance", ambiguous), invariant="instance_symbol_ambiguous")
        return found

    def component_index(self, refs: Sequence[ComponentRef]) -> NDArray[np.int64]:
        r"""The component of each reference, ``-1`` when the instance is unknown or the ordinal is past it."""
        instance = self.instance_index([ref.instance for ref in refs])
        ordinal = np.array([ref.component for ref in refs], dtype=np.int64)
        pointer = self.instance_component_ptr
        start = np.append(pointer[:-1], 0)[instance]
        count = np.append(np.diff(pointer), 0)[instance]
        valid = (instance >= 0) & (ordinal >= 0) & (ordinal < count)
        return np.where(valid, start + ordinal, -1)

    def component_refs(self, components: NDArray[np.int64]) -> tuple[ComponentRef, ...]:
        r"""The ``ComponentRef`` of each component: the exact inverse of ``component_index``."""
        components = check_rows("components", components, len(self.component_symbol), shape=(None,))
        pointer = self.instance_component_ptr
        instance = np.searchsorted(pointer, components, side="right") - 1
        texts = self.text("instance", self.instance_symbol[instance])
        return tuple(ComponentRef(str(t), int(o)) for t, o in zip(texts, components - pointer[instance]))

    def atom_index(self, sites: Sequence[AtomSite]) -> NDArray[np.int64]:
        r"""The atom at each site, ``-1`` when the instance, ordinal or atom name names nothing."""

        radix = len(self.symbols()["atom"]) + 1
        if "atoms" not in self._cache:
            count = len(self.component_symbol)
            owner = np.repeat(np.arange(count, dtype=np.int64), np.diff(self.component_atom_ptr))
            keys = owner * radix + self.atom_symbol
            order = np.argsort(keys, kind="stable")
            self._cache["atoms"] = keys[order], order
        keys, order = self._cache["atoms"]
        component = self.component_index([ComponentRef(site.instance, site.component) for site in sites])
        family = self._codes("atom")
        code = np.array([family.get(site.atom, -1) for site in sites], dtype=np.int64)
        query = component * radix + code
        position = np.clip(np.searchsorted(keys, query), 0, max(len(keys) - 1, 0))
        hit = (component >= 0) & (code >= 0) & (len(keys) > 0)
        hit &= keys[position] == query if len(keys) else False
        return np.where(hit, order[position] if len(keys) else -1, -1)

    def reference_coordinates(self) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
        r"""Per atom, its reference-conformer coordinates and source (zeros, ``ReferenceSource.NONE`` without one)."""
        row = self.atom_reference_row.astype(np.int64)
        present = row >= 0
        coordinates = np.zeros((len(row), 3), dtype=np.float64)
        source = np.zeros(len(row), dtype=np.uint8)
        coordinates[present] = self.reference_pool_coordinates[row[present]]
        source[present] = self.reference_pool_source[row[present]]
        return coordinates, source

    def text(self, family: str, codes: NDArray[np.int32]) -> NDArray[np.str_]:
        r"""The symbols of ``codes`` in ``family``, ``""`` for ``-1`` (diagnostics and writers only)."""
        table = self.symbols()[family]
        return np.asarray((*table, ""), dtype=np.str_)[check_rows("codes", codes, len(table), low=-1)]
