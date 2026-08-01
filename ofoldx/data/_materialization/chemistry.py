"""Resolve observed, declared and reference component chemistry in one place."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...io.records import AtomColumns
from ..components import complete_component, get_component
from ..elements import ATOMIC_NUMBER_BY_ELEMENT
from ..molecular import ChemicalAtom, ChemicalComponent


@dataclass(frozen=True, kw_only=True)
class ResolvedComponents:
    components: Mapping[str, ChemicalComponent]
    topology_complete: Mapping[str, bool]
    formal_charge_atoms: Mapping[str, frozenset[str]]
    non_leaving_atoms: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        for name in ("components", "topology_complete", "formal_charge_atoms", "non_leaving_atoms"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _observed_components(rows: AtomColumns) -> dict[str, ChemicalComponent]:
    r"""Return the positive atom chemistry stated by coordinate records."""

    atoms_by_component: dict[str, dict[str, ChemicalAtom]] = {}
    for raw_component_id, raw_atom_name, raw_element in zip(
        rows.component_id,
        rows.atom_name,
        rows.element,
        strict=True,
    ):
        component_id = str(raw_component_id)
        atom_name = str(raw_atom_name)
        element = str(raw_element)
        if element not in ATOMIC_NUMBER_BY_ELEMENT:
            raise ValueError(f"source atom {atom_name!r} in component {component_id!r} has invalid element {element!r}")
        atom = ChemicalAtom(
            atom_id=atom_name,
            element=element,
        )
        component_atoms = atoms_by_component.setdefault(component_id, {})
        previous = component_atoms.get(atom_name)
        if previous is None:
            component_atoms[atom_name] = atom
            continue
        if previous.element != atom.element:
            raise ValueError(f"component {component_id!r} atom {atom_name!r} has conflicting source elements")
    return {
        component_id: ChemicalComponent(
            component_id=component_id,
            atoms=tuple(atoms.values()),
            bonds=None,
        )
        for component_id, atoms in atoms_by_component.items()
    }


def resolve_components(
    source_rows: AtomColumns,
    declared_component_ids: Iterable[str],
    *,
    source_components: Mapping[str, ChemicalComponent],
    source_component_stereo_known: Mapping[str, tuple[frozenset[str], frozenset[frozenset[str]]]],
    components: Mapping[str, ChemicalComponent] | None,
    component_resolver: Callable[[str], ChemicalComponent] | None,
) -> ResolvedComponents:
    source_non_leaving_atom_ids = {key: value.non_leaving_atom_ids for key, value in source_components.items()}
    source_formal_charge_atoms = {
        key: frozenset(atom.atom_id for atom in value.atoms if atom.formal_charge is not None)
        for key, value in source_components.items()
    }
    component_ids = {*source_rows.component_id, *declared_component_ids}
    component_table: dict[str, ChemicalComponent] = {}
    explicit_component_ids = set() if components is None else set(components)
    observed_components = _observed_components(source_rows)
    component_topology_complete: dict[str, bool] = {}
    for raw_component_id in component_ids:
        component_id = str(raw_component_id)
        if component_id in explicit_component_ids:
            continue
        source_component = source_components.get(component_id)
        source_topology_complete = source_component is not None and source_component.bonds is not None
        observed_component = observed_components.get(component_id)
        if observed_component is not None:
            if source_component is None:
                source_component = observed_component
                source_topology_complete = observed_component.bonds is not None
            else:
                source_component, source_topology_complete = complete_component(
                    observed_component, source_component, occurrence_topology_complete=False
                )
        reference_component: ChemicalComponent | None = None
        if component_resolver is not None:
            with contextlib.suppress(KeyError):
                reference_component = component_resolver(component_id)
        with contextlib.suppress(KeyError):
            if reference_component is None:
                reference_component = get_component(component_id)
        if reference_component is not None and reference_component.component_id != component_id:
            raise ValueError("component resolver keys must equal ChemicalComponent.component_id")
        if source_component is not None and reference_component is not None:
            atom_stereo_known, bond_stereo_known = source_component_stereo_known.get(
                component_id, (frozenset(), frozenset())
            )
            component_table[component_id], component_topology_complete[component_id] = complete_component(
                source_component,
                reference_component,
                occurrence_topology_complete=source_topology_complete,
                atom_stereo_known=atom_stereo_known,
                bond_stereo_known=bond_stereo_known,
            )
        elif source_component is not None:
            component_table[component_id] = source_component
            component_topology_complete[component_id] = source_topology_complete
        elif reference_component is not None:
            component_table[component_id] = reference_component
            component_topology_complete[component_id] = reference_component.bonds is not None
    if components is not None:
        for component_id, component in components.items():
            if component.component_id != component_id:
                raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
            component_table[component_id] = component
            component_topology_complete[component_id] = component.bonds is not None
            source_non_leaving_atom_ids[component_id] = component.non_leaving_atom_ids
    for component_id, component in observed_components.items():
        component_table.setdefault(component_id, component)
    for component_id, component in component_table.items():
        component_topology_complete.setdefault(component_id, component.bonds is not None)
    return ResolvedComponents(
        components=component_table,
        topology_complete=component_topology_complete,
        formal_charge_atoms=source_formal_charge_atoms,
        non_leaving_atoms=source_non_leaving_atom_ids,
    )
