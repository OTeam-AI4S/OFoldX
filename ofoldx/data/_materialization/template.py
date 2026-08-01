"""Protein template projection has its own entry point and evidence requirements."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...io.selection import (
    _ALTLOC_POLICIES,
    _polymer_declarations,
    _select_altloc,
    _select_mmcif_polymer_microheterogeneity,
)
from ...io.source import StructureView, _require_cif_source
from ..molecular import ChemicalComponent
from ..structure import Structure
from .chemistry import ResolvedComponents, _observed_components
from .cif_catalog import CifCatalog
from .cif_planner import plan_cif
from .pipeline import realize_structure
from .scatter import StructureMetadata


def materialize_template(
    view: StructureView, *, altloc: Literal["highest_occupancy", "first"] = "highest_occupancy"
) -> Structure:
    """Project CIF sequence and observed coordinates without completing chemistry.

    Parent relationships retain every declared component identifier. Selection of
    one standard residue, if needed, belongs to the consuming model.
    """
    source = _require_cif_source(view.source)
    if view.asym_ids is None or len(view.asym_ids) != 1:
        raise ValueError("template projection requires exactly one explicit label-asym chain")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if view.assembly_id is not None:
        raise ValueError("template coordinate projection uses deposited chains")
    monomers = _polymer_declarations(view).polymer_monomers
    if monomers is None:
        raise ValueError("template chain requires a declared polymer sequence")
    rows = _select_altloc(_select_mmcif_polymer_microheterogeneity(view.rows, monomers, altloc), altloc)
    rows = rows[rows.element != "H"].view(np.recarray)
    if not rows.coordinate_present.any():
        raise ValueError("template chain has no observed heavy-atom coordinates")
    components = _observed_components(rows)
    for positions in monomers.values():
        for _, candidates in positions:
            for identifier in candidates:
                components.setdefault(identifier, ChemicalComponent(identifier, atoms=(), bonds=None))
    parents = source.component_parents
    chemistry = ResolvedComponents(
        components=components,
        topology_complete=dict.fromkeys(components, False),
        formal_charge_atoms={},
        non_leaving_atoms={},
    )
    catalog = CifCatalog.for_template(view, rows)
    planned = plan_cif(
        catalog,
        rows,
        len(view.rows),
        components,
        placements={(None, chain): np.eye(4) for chain in view.asym_ids},
        topology_placements=set(),
        include_water=False,
        include_declared_instances=True,
    )
    plan = planned.molecular
    return realize_structure(
        rows, plan, (), chemistry, StructureMetadata(component_parents=parents, resolutions=(), formal_charge_atoms={})
    )
