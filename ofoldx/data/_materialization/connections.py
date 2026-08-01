"""Resolve external bond declarations against realized occurrence addresses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..molecular import AtomSite, Molecule, PolymerSequence, SystemBond
from .chemistry import ResolvedComponents
from .cif_catalog import CifCatalog
from .cif_planner import CifPlan
from .pdb_catalog import PdbDeclarations
from .pdb_planner import PdbPlan
from .plan import AtomRows, ResolvedConnections

if TYPE_CHECKING:
    from .._cif_chemistry import _CifSourceSites


def resolve_cif_connections(
    catalog: CifCatalog,
    source_rows: AtomRows,
    rows: AtomRows,
    planned: CifPlan,
    chemistry: ResolvedComponents,
    placements: Mapping[tuple[str | None, str], NDArray[np.float64]],
    *,
    model_rows: AtomRows,
    source_sites: _CifSourceSites,
    model_occurrences: frozenset[int],
) -> ResolvedConnections:
    plan = planned.molecular
    instance_values = plan.instance_values
    from .._cif_chemistry import read_mmcif_bonds

    source_bonds = read_mmcif_bonds(
        catalog.view,
        source_rows,
        rows,
        plan.site_by_row_index,
        planned.sites,
        chemistry.components,
        placements=placements,
        model_rows=model_rows,
        source_sites=source_sites,
        model_occurrences=model_occurrences,
        occurrence_components={
            (instance, index): component
            for instance, entity in plan.instance_values.items()
            for index, component in enumerate(entity.components)
        },
        component_topology_complete=chemistry.topology_complete,
        source_non_leaving_atom_ids=chemistry.non_leaving_atoms,
        declared_branch_bonds={
            frozenset((site_a, site_b)): (
                SystemBond(
                    site_a,
                    site_b,
                    order=bond.order,
                    aromatic=bond.aromatic,
                    stereo=bond.stereo,
                    stereo_sites=(
                        None
                        if bond.stereo_atoms is None
                        else (
                            AtomSite(instance_id, *bond.stereo_atoms[0]),
                            AtomSite(instance_id, *bond.stereo_atoms[1]),
                        )
                    ),
                    role="entity",
                    leaving_a=bond.leaving_a,
                    leaving_b=bond.leaving_b,
                    kind=bond.kind,
                    direction=bond.direction,
                ),
                leaving_a_known,
                leaving_b_known,
            )
            for instance_id, entity in instance_values.items()
            if isinstance(entity, Molecule) and entity.molecule_type == "branched"
            for bond, leaving_a_known, leaving_b_known in catalog.branch_bonds(
                plan.instances[instance_id], plan.source_asym_by_instance[instance_id]
            )
            for site_a, site_b in (
                (
                    AtomSite(instance_id, bond.component_a, bond.atom_a),
                    AtomSite(instance_id, bond.component_b, bond.atom_b),
                ),
            )
        },
    )
    lowered_bonds: list[SystemBond] = []
    for bond in source_bonds.bonds:
        bond_entity = instance_values.get(bond.site_a.instance_id)
        if (
            bond.kind == "covalent"
            and bond.site_a.instance_id == bond.site_b.instance_id
            and isinstance(bond_entity, PolymerSequence)
            and (abs(bond.site_a.component_index - bond.site_b.component_index) == 1)
            and (bond.order in {None, 1.0})
            and (bond.aromatic in {None, False})
        ):
            if bond.site_a.component_index < bond.site_b.component_index:
                left_site, right_site = (bond.site_a, bond.site_b)
                left_leaving, right_leaving = (bond.leaving_a, bond.leaving_b)
            else:
                left_site, right_site = (bond.site_b, bond.site_a)
                left_leaving, right_leaving = (bond.leaving_b, bond.leaving_a)
            left = bond_entity.components[left_site.component_index]
            right = bond_entity.components[right_site.component_index]
            left_attachment = left.next_attachment
            right_attachment = right.previous_attachment
            if (
                not bond_entity.is_linked(left_site.component_index)
                and left_attachment is not None
                and (right_attachment is not None)
                and (left_attachment.atom_id == left_site.atom_id)
                and (right_attachment.atom_id == right_site.atom_id)
                and (len(left_leaving) == len(left_attachment.leaving_atom_ids))
                and (len(right_leaving) == len(right_attachment.leaving_atom_ids))
            ):
                bond = replace(bond, order=1.0, aromatic=False, role="polymer")
        lowered_bonds.append(bond)
    return replace(source_bonds, bonds=tuple(lowered_bonds))


def resolve_pdb_connections(
    declarations: PdbDeclarations,
    source_rows: np.recarray,
    rows: np.recarray,
    planned: PdbPlan,
    chemistry: ResolvedComponents,
    *,
    include_water: bool,
    model_rows: np.recarray,
    selected_model_rows: np.recarray,
    missing_residues: tuple[tuple[str, str, int, str], ...],
) -> ResolvedConnections:
    from ...io.pdb import read_pdb_bonds

    return read_pdb_bonds(
        declarations.lines,
        source_rows,
        rows,
        planned.molecular.site_by_row_index,
        planned.sites,
        planned.ambiguous_sites,
        chemistry.components,
        include_water=include_water,
        assembly_placements=declarations.assembly_placements,
        model_rows=model_rows,
        selected_model_rows=selected_model_rows,
        missing_residues=missing_residues,
    )
