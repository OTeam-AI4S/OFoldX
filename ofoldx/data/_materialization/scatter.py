"""Bind a completed molecular graph to source coordinates and experimental metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...io.records import AtomColumns
from ..elements import ATOMIC_NUMBER_BY_ELEMENT
from ..indexing import MolecularAtomIndex
from ..molecular import AtomSite, MolecularSystem
from ..structure import Structure
from .plan import OccurrencePlan


@dataclass(frozen=True, kw_only=True)
class StructureMetadata:
    component_parents: Mapping[str, tuple[str, ...]]
    resolutions: tuple[tuple[float, int], ...]
    formal_charge_atoms: Mapping[str, frozenset[str]]
    formal_charge_provenance: frozenset[AtomSite] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_parents", MappingProxyType(dict(self.component_parents)))
        object.__setattr__(self, "formal_charge_atoms", MappingProxyType(dict(self.formal_charge_atoms)))
        object.__setattr__(self, "resolutions", tuple(self.resolutions))
        if self.formal_charge_provenance is not None:
            object.__setattr__(self, "formal_charge_provenance", frozenset(self.formal_charge_provenance))


def scatter_structure(
    rows: AtomColumns, plan: OccurrencePlan, system: MolecularSystem, metadata: StructureMetadata
) -> Structure:
    graph_index = MolecularAtomIndex(system)
    source_atom_index = _map_source_atoms(rows, plan, system, graph_index)
    _apply_source_atom_chemistry(rows, source_atom_index, system)
    source_formal_charge_mask = _formal_charge_evidence(rows, source_atom_index, system, graph_index, plan, metadata)
    annotations = _component_annotations(plan, system, graph_index, metadata)
    atom_count = system.atom_symbol.numel()
    source_resolutions = metadata.resolutions
    coordinates = torch.zeros((atom_count, 3), dtype=torch.float64)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    occupancy = torch.zeros(atom_count, dtype=torch.float64)
    occupancy_mask = torch.zeros(atom_count, dtype=torch.bool)
    b_factor = torch.zeros(atom_count, dtype=torch.float64)
    b_factor_mask = torch.zeros(atom_count, dtype=torch.bool)
    source_coordinate_mask = torch.from_numpy(np.array(rows.coordinate_present, dtype=np.bool_, copy=True))
    coordinate_mask[source_atom_index] = source_coordinate_mask
    coordinates[source_atom_index] = torch.from_numpy(
        np.array(rows.coordinates, dtype=np.float64, copy=True, order="C")
    )
    source_occupancy_mask = torch.from_numpy(_present(rows.occupancy))
    occupancy_mask[source_atom_index] = source_occupancy_mask
    occupancy[source_atom_index] = torch.from_numpy(
        np.asarray(np.where(_present(rows.occupancy), rows.occupancy, 0.0), dtype=np.float64)
    )
    source_b_factor_mask = torch.from_numpy(_present(rows.b_factor))
    b_factor_mask[source_atom_index] = source_b_factor_mask
    b_factor[source_atom_index] = torch.from_numpy(
        np.asarray(np.where(_present(rows.b_factor), rows.b_factor, 0.0), dtype=np.float64)
    )
    return Structure(
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=annotations.symbols,
        residue_id=annotations.residue_ids,
        insertion_code=annotations.insertion_codes,
        component_parent_ptr=annotations.parent_ptr,
        parent_component_id=annotations.parent_ids,
        source_formal_charge_mask=source_formal_charge_mask,
        system_resolution_ptr=torch.tensor([0, len(source_resolutions)], dtype=torch.int64, device=coordinates.device),
        resolution=torch.tensor(
            [value for value, _ in source_resolutions], dtype=torch.float64, device=coordinates.device
        ),
        resolution_kind=torch.tensor(
            [kind for _, kind in source_resolutions], dtype=torch.uint8, device=coordinates.device
        ),
        batch_size=[],
    )


def _present(values: NDArray[np.object_]) -> NDArray[np.bool_]:
    """Preserve the distinction between an absent measurement and a measured zero."""
    return np.not_equal(values, np.asarray(None, dtype=object))


def _map_source_atoms(
    rows: AtomColumns, plan: OccurrencePlan, system: MolecularSystem, graph_index: MolecularAtomIndex
) -> Tensor:
    site_by_row_index = plan.site_by_row_index
    atomic_number = system.atomic_number
    atom_indices = graph_index.map_rows(site_by_row_index, rows.row_index)
    ordered = np.sort(atom_indices)
    duplicates = ordered[1:][ordered[1:] == ordered[:-1]]
    if len(duplicates):
        raise ValueError(f"selected source rows map more than once to {system.atom_site(int(duplicates[0]))!r}")
    source_atom_index = torch.from_numpy(atom_indices)
    source_atomic_number = torch.from_numpy(
        np.fromiter(map(ATOMIC_NUMBER_BY_ELEMENT.__getitem__, rows.element), dtype=np.int64)
    ).to(atomic_number.dtype)
    element_mismatch = atomic_number[source_atom_index].cpu() != source_atomic_number
    if bool(element_mismatch.any()):
        source_index = int(torch.nonzero(element_mismatch, as_tuple=False)[0])
        site = system.atom_site(atom_indices[source_index])
        raise ValueError(f"source element {rows.element[source_index]!r} disagrees with component atom at {site!r}")
    return source_atom_index


def _apply_source_atom_chemistry(rows: AtomColumns, source_atom_index: Tensor, system: MolecularSystem) -> None:
    isotope = system.isotope
    isotope_known = system.isotope_known
    formal_charge = system.formal_charge
    formal_charge_known = system.formal_charge_known
    source_isotope_mask = torch.from_numpy(_present(rows.isotope))
    isotope[source_atom_index[source_isotope_mask]] = torch.from_numpy(
        np.asarray(rows.isotope[_present(rows.isotope)], dtype=np.int64)
    ).to(isotope.dtype)
    isotope_known[source_atom_index[source_isotope_mask]] = True
    source_charge_mask = torch.from_numpy(_present(rows.charge))
    formal_charge[source_atom_index[source_charge_mask]] = torch.from_numpy(
        np.asarray(rows.charge[_present(rows.charge)], dtype=np.int64)
    ).to(formal_charge.dtype)
    formal_charge_known[source_atom_index[source_charge_mask]] = True


def _formal_charge_evidence(
    rows: AtomColumns,
    source_atom_index: Tensor,
    system: MolecularSystem,
    graph_index: MolecularAtomIndex,
    plan: OccurrencePlan,
    metadata: StructureMetadata,
) -> Tensor:
    source_formal_charge_atoms = metadata.formal_charge_atoms
    source_formal_charge_provenance = metadata.formal_charge_provenance
    source_asym_by_instance = plan.source_asym_by_instance
    component_index_by_reference = graph_index.component_indices
    atom_count = system.atom_symbol.numel()
    symbols = system.symbols
    component_symbol = system.component_symbol.tolist()
    source_charge_mask = torch.from_numpy(_present(rows.charge))
    source_formal_charge_mask = torch.zeros(atom_count, dtype=torch.bool)
    charge_components: list[int] = []
    charge_names: list[str] = []
    for component_index, component_id in enumerate(component_symbol):
        declared_names = source_formal_charge_atoms.get(symbols[component_id], ())
        charge_components.extend([component_index] * len(declared_names))
        charge_names.extend(declared_names)
    declared_charges = graph_index.atom_indices(charge_components, charge_names, allow_missing=True)
    source_formal_charge_mask[torch.from_numpy(declared_charges[declared_charges >= 0])] = True
    source_formal_charge_mask[source_atom_index[source_charge_mask]] = True
    if source_formal_charge_provenance is not None:
        copies_by_asym: dict[str, list[str]] = {}
        for instance, asym in source_asym_by_instance.items():
            copies_by_asym.setdefault(asym, []).append(instance)
        provenance = tuple(source_formal_charge_provenance)
        references, names, owners = ([], [], [])
        for owner, site in enumerate(provenance):
            for instance in copies_by_asym.get(site.instance_id, ()):
                references.append(component_index_by_reference.get((instance, site.component_index), -1))
                names.append(site.atom_id)
                owners.append(owner)
        realized = graph_index.atom_indices(references, names, allow_missing=True)
        found = {owners[i] for i in np.flatnonzero(realized >= 0)}
        missing = [site for i, site in enumerate(provenance) if i not in found]
        if missing:
            raise ValueError(
                f"mmCIF _ofoldx_source_formal_charge references unrealized atom {min(missing, key=repr)!r}"
            )
        source_formal_charge_mask.zero_()
        source_formal_charge_mask[torch.from_numpy(realized[realized >= 0])] = True
    return source_formal_charge_mask


class ComponentAnnotations(NamedTuple):
    symbols: tuple[str, ...]
    residue_ids: Tensor
    insertion_codes: Tensor
    parent_ptr: Tensor
    parent_ids: Tensor


def _component_annotations(
    plan: OccurrencePlan, system: MolecularSystem, graph_index: MolecularAtomIndex, metadata: StructureMetadata
) -> ComponentAnnotations:
    source_component_parents = metadata.component_parents
    residue_numbers = plan.residue_numbers
    standard_components = plan.standard_components
    symbols = system.symbols
    component_symbol = system.component_symbol.tolist()
    component_count = len(component_symbol)
    component_index_by_reference = graph_index.component_indices
    component_references = graph_index.component_references
    source_symbols: list[str] = []
    source_symbol_index: dict[str, int] = {}

    def intern(value: str) -> int:
        index = source_symbol_index.get(value)
        if index is None:
            index = len(source_symbols)
            source_symbol_index[value] = index
            source_symbols.append(value)
        return index

    residue_id_values: np.ndarray = np.full(component_count, -1, dtype=np.int64)
    insertion_code_values: np.ndarray = np.full(component_count, -1, dtype=np.int64)
    parent_ptr = [0]
    parent_ids: list[int] = []
    for reference, (number, code) in residue_numbers.items():
        component_index = component_index_by_reference[reference]
        residue_id_values[component_index] = intern(number)
        if code:
            insertion_code_values[component_index] = intern(code)
    for component_index in range(component_count):
        reference = component_references[component_index]
        standard_component = standard_components.get(reference)
        component_id = symbols[int(component_symbol[component_index])]
        parents = (
            (standard_component,) if standard_component is not None else source_component_parents.get(component_id, ())
        )
        parent_ids.extend(intern(parent) for parent in parents)
        parent_ptr.append(len(parent_ids))
    return ComponentAnnotations(
        tuple(source_symbols),
        torch.from_numpy(residue_id_values),
        torch.from_numpy(insertion_code_values),
        torch.tensor(parent_ptr, dtype=torch.int64),
        torch.tensor(parent_ids, dtype=torch.int64),
    )
