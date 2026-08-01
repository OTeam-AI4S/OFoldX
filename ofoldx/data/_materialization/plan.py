"""Observed residue membership and validated molecular occurrence plans."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple, TypeAlias, TypeVar, cast

import numpy as np

from ..._arrays import immutable_array
from ...io.records import AtomColumns, unique_identifier
from .._placed_atoms import PlacedAtoms
from ..indexing import AtomSiteTable, AtomSiteTableBuilder
from ..molecular import AtomSite, Molecule, PolymerSequence, SystemBond
from ..references import ComponentReference

AtomRows: TypeAlias = np.recarray | PlacedAtoms
Entity: TypeAlias = PolymerSequence | Molecule
EntityT = TypeVar("EntityT", bound=Entity)


@dataclass(frozen=True)
class ResolvedConnections:
    """Retained edges and reactions whose other endpoint was projected away."""

    bonds: tuple[SystemBond, ...] = ()
    boundary_leaving: Mapping[AtomSite, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_leaving", MappingProxyType(dict(self.boundary_leaving)))


class ResidueKey(NamedTuple):
    symmetry_id: str | None
    chain_id: str
    residue_id: int | str | None
    author_residue_id: str | None
    insertion_code: str | None
    component_id: str


class Occurrence(NamedTuple):
    order: int
    instance_id: str
    entity_id: str
    value: Entity
    observed: tuple[tuple[int, ResidueKey], ...]


class ResidueTable:
    """Index borrowed columns without gathering coordinates or changing row IDs.

    Positions address the selected table. Its row_index column identifies physical
    occurrences, including assembly copies, independently of these positions.
    """

    def __init__(self, rows: AtomColumns) -> None:
        self.rows = rows
        positions: dict[ResidueKey, list[int]] = {}
        chains: dict[tuple[str | None, str], list[ResidueKey]] = {}
        columns = (
            rows.symmetry_id,
            rows.chain_id,
            rows.residue_id,
            rows.auth_residue_id,
            rows.insertion_code,
            rows.component_id,
        )
        for position, values in enumerate(zip(*columns, strict=True)):
            key = ResidueKey(*values)
            if key not in positions:
                positions[key] = []
                chains.setdefault((key.symmetry_id, key.chain_id), []).append(key)
            positions[key].append(position)
        self.positions = MappingProxyType({key: immutable_array(value, np.int64) for key, value in positions.items()})
        self.chains = MappingProxyType({key: tuple(value) for key, value in chains.items()})


@dataclass(frozen=True, kw_only=True)
class OccurrencePlan:
    """Owned plan whose instance and component references are valid at publication.

    Format readers establish molecular definitions and occurrence variants. This
    boundary validates references, not chemical equivalence between variants.
    Atom-name and bond validation belong to molecular graph construction.
    """

    entities: Mapping[str, Entity]
    occurrences: tuple[Occurrence, ...]
    site_by_row_index: AtomSiteTable
    residue_numbers: Mapping[ComponentReference, tuple[str, str]]
    standard_components: Mapping[ComponentReference, str]
    source_asym_by_instance: Mapping[str, str]
    instances: Mapping[str, str] = field(init=False)
    instance_values: Mapping[str, Entity] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", MappingProxyType(dict(self.entities)))
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        object.__setattr__(self, "residue_numbers", MappingProxyType(dict(self.residue_numbers)))
        object.__setattr__(self, "standard_components", MappingProxyType(dict(self.standard_components)))
        object.__setattr__(self, "source_asym_by_instance", MappingProxyType(dict(self.source_asym_by_instance)))
        if not self.occurrences:
            raise ValueError("selected structure contains no molecular instances")
        instances = {item.instance_id: item.entity_id for item in self.occurrences}
        if len(instances) != len(self.occurrences):
            raise ValueError("molecular occurrences must have unique instance identifiers")
        if any(entity_id not in self.entities for entity_id in instances.values()):
            raise ValueError("molecular occurrence references an undeclared entity")
        values = {item.instance_id: item.value for item in self.occurrences}
        object.__setattr__(self, "instances", MappingProxyType(instances))
        object.__setattr__(self, "instance_values", MappingProxyType(values))
        for item in self.occurrences:
            observed = [index for index, _ in item.observed]
            if len(observed) != len(set(observed)):
                raise ValueError(f"instance {item.instance_id!r} maps multiple residues to one component")
            for index in observed:
                self._validate_component((item.instance_id, index))
        for references in (self.residue_numbers, self.standard_components, self.site_by_row_index.components):
            for reference in references:
                self._validate_component(reference)
        if not self.source_asym_by_instance.keys() <= instances.keys():
            raise ValueError("source asym mapping references an undeclared instance")

    def _validate_component(self, reference: ComponentReference) -> None:
        instance, index = reference
        try:
            index = operator.index(index)
        except TypeError:
            raise ValueError(f"component index must be an integer: {reference!r}") from None
        value = self.instance_values.get(instance)
        if value is None or not 0 <= index < len(value.components):
            raise ValueError(f"invalid molecular component reference {reference!r}")


class PlanBuilder:
    """Collect source facts, then publish a validated, independent snapshot.

    Numbering and atom bindings can precede occurrence registration. Cross
    references are checked together at build(), so no caller ordering is needed.
    Only occurrence registration and naming affect identifier allocation.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._occurrences: dict[str, Occurrence] = {}
        self._residue_numbers: dict[ComponentReference, tuple[str, str]] = {}
        self._standard_components: dict[ComponentReference, str] = {}
        self._source_asym_by_instance: dict[str, str] = {}
        self._source_sites = AtomSiteTableBuilder()
        self._used_instance_ids: set[str] = set()

    @property
    def occurrences(self) -> tuple[Occurrence, ...]:
        return tuple(sorted(self._occurrences.values(), key=lambda item: item.order))

    def name_instance(self, seed: str) -> str:
        return unique_identifier(seed, self._used_instance_ids)

    def register_entity(self, entity_id: str, value: EntityT, *, allow_variants: bool = False) -> EntityT:
        """Register a definition; format-validated variants keep their own value."""
        previous = self._entities.get(entity_id)
        if previous is None:
            self._entities[entity_id] = value
            return value
        if allow_variants:
            return value
        if previous != value:
            raise ValueError(f"source entity ID {entity_id!r} has conflicting molecular definitions")
        return cast(EntityT, previous)

    def add_occurrence(self, occurrence: Occurrence) -> None:
        if occurrence.instance_id in self._occurrences:
            raise ValueError("molecular occurrences must have unique instance identifiers")
        self._occurrences[occurrence.instance_id] = occurrence
        self._used_instance_ids.add(occurrence.instance_id)

    def set_residue_number(self, reference: ComponentReference, value: tuple[str, str]) -> None:
        self._residue_numbers[reference] = value

    def set_standard_component(self, reference: ComponentReference, component_id: str) -> None:
        self._standard_components[reference] = component_id

    def set_source_asym(self, instance_id: str, asym_id: str) -> None:
        self._source_asym_by_instance[instance_id] = asym_id

    def bind_atoms(self, reference: ComponentReference, row_ids: Iterable[int], atom_ids: Iterable[str]) -> None:
        self._source_sites.add_component(reference, row_ids, atom_ids)

    def build(self) -> OccurrencePlan:
        return OccurrencePlan(
            entities=self._entities,
            occurrences=self.occurrences,
            site_by_row_index=self._source_sites.build(),
            residue_numbers=self._residue_numbers,
            standard_components=self._standard_components,
            source_asym_by_instance=self._source_asym_by_instance,
        )
