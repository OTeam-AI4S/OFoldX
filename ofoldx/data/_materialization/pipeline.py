"""Graph allocation follows successful occurrence and connection resolution."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...io.records import AtomColumns
from ..indexing import MolecularAtomIndex
from ..molecular import BOND_ROLE_COMPONENT, AtomSite, MolecularSystem, SystemBond, atom_subgraph
from ..structure import Structure
from .chemistry import ResolvedComponents
from .plan import OccurrencePlan
from .scatter import StructureMetadata, scatter_structure


def realize_structure(
    rows: AtomColumns,
    plan: OccurrencePlan,
    bonds: tuple[SystemBond, ...],
    chemistry: ResolvedComponents,
    metadata: StructureMetadata,
    *,
    boundary_leaving: Mapping[AtomSite, frozenset[str]] | None = None,
) -> Structure:
    system = MolecularSystem.from_entities(plan.entities, plan.instances, bonds, instance_values=plan.instance_values)
    if boundary_leaving:
        index = MolecularAtomIndex(system)
        removed: set[int] = set()
        endpoints = {system.atom_index(site) for site in boundary_leaving}
        endpoints.update(system.edge_index[:, system.bond_role != BOND_ROLE_COMPONENT].flatten().tolist())
        endpoints.update(system.atom_index(site) for bond in bonds for site in (bond.site_a, bond.site_b))
        for site, names in boundary_leaving.items():
            component = index.component_indices[site.instance_id, site.component_index]
            atoms = index.atom_indices([component] * len(names), list(names), allow_missing=True)
            removed.update(int(atom) for atom in atoms if atom >= 0)
        if removed & endpoints:
            raise ValueError("a bond endpoint cannot also be a leaving atom")
        if removed:
            keep = torch.ones(system.atom_symbol.numel(), dtype=torch.bool)
            keep[list(removed)] = False
            system, _ = atom_subgraph(system, keep)
    structure = scatter_structure(rows, plan, system, metadata)
    topology_complete = system.component_topology_complete.new_tensor(
        [
            chemistry.topology_complete.get(system.symbols[symbol], complete)
            for symbol, complete in zip(
                system.component_symbol.tolist(), system.component_topology_complete.tolist(), strict=True
            )
        ]
    )
    return structure.replace(system=structure.system.replace(component_topology_complete=topology_complete))
