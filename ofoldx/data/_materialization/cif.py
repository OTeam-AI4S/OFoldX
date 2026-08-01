"""mmCIF evidence preparation and explicit molecular materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ...io.declarations import _read_mmcif_atom_provenance, read_mmcif_resolutions
from ...io.selection import _select_altloc, _select_cif_declarations, _select_mmcif_polymer_microheterogeneity
from ...io.source import StructureView, _require_cif_source
from ...io.symmetry import _mmcif_assembly_placements
from .._cif_chemistry import (
    _is_graph_connection_type,
    _materialize_mmcif_crystallographic_copies,
    _mmcif_source_occurrences,
    _mmcif_source_sites,
    _resolve_mmcif_occurrences,
    _validate_mmcif_atom_provenance,
    read_mmcif_branch_entities,
    read_mmcif_components,
)
from .._placed_atoms import PlacedAtoms
from ..molecular import ChemicalComponent
from ..structure import Structure
from .chemistry import resolve_components
from .cif_catalog import CifCatalog
from .cif_planner import plan_cif
from .connections import resolve_cif_connections
from .pipeline import realize_structure
from .plan import AtomRows
from .scatter import StructureMetadata
from .selection import _exclude_water_rows


def materialize_cif(
    view: StructureView,
    *,
    altloc: Literal["highest_occupancy", "first"],
    include_water: bool,
    include_hydrogens: bool,
    components: Mapping[str, ChemicalComponent] | None,
    component_resolver: Callable[[str], ChemicalComponent] | None,
) -> Structure:
    source = _require_cif_source(view.source)
    source_rows: AtomRows = view.rows
    declarations = _select_cif_declarations(view)
    placements: dict[tuple[str | None, str], NDArray[np.float64]]
    if view.assembly_id is None:
        asym_ids = declarations.asym_entities or tuple(dict.fromkeys(source_rows.chain_id))
        placements = {(None, asym): np.eye(4) for asym in asym_ids}
    else:
        placements = _mmcif_assembly_placements(source.block, view.assembly_id)
        if view.asym_ids is not None:
            placements = {key: value for key, value in placements.items() if key[1] in view.asym_ids}
        view = replace(view, asym_ids=frozenset(asym for _, asym in placements))
        source_rows = view.rows
        declarations = _select_cif_declarations(view)
    selected_entities = set(declarations.asym_entities.values()) | {
        str(entity) for entity in source_rows.entity_id if entity is not None
    }
    needed = set(source_rows.component_id) | set(declarations.nonpolymer_components.values())
    for sequence in (declarations.polymer_monomers or {}).values():
        needed.update(component for _, candidates in sequence for component in candidates)
    branch = source.block.get("pdbx_entity_branch_list")
    if branch is not None:
        needed.update(
            component
            for entity, component in zip(branch.strings("entity_id"), branch.strings("comp_id"))
            if entity in selected_entities and component is not None
        )
    deposited_rows = source.atoms.by_index(view.model_index)
    source_occurrences = (
        _mmcif_source_occurrences(source, view.model_index) if source.connections or branch is not None else ()
    )
    for connection in source.connections:
        if not _is_graph_connection_type(connection.kind):
            continue
        for endpoint in (connection.first, connection.second):
            declared = _resolve_mmcif_occurrences(source_occurrences, endpoint)
            needed.update(str(occurrence.label[3]) for occurrence in declared)
    source_components, _, parents, stereo = read_mmcif_components(
        source.document, owned=() if components is None else tuple(components), selected=needed
    )
    chemistry = resolve_components(
        cast(np.recarray, deposited_rows[np.isin(deposited_rows.component_id, tuple(needed))]),
        needed,
        source_components=source_components,
        source_component_stereo_known=stereo,
        components=components,
        component_resolver=component_resolver,
    )
    source_sites = _mmcif_source_sites(source, view.model_index, chemistry.components, source_occurrences)
    provenance = _read_mmcif_atom_provenance(source.block)
    if provenance is not None:
        _validate_mmcif_atom_provenance(source.document, provenance, chemistry.components)

    def select(rows: AtomRows) -> AtomRows:
        rows = _select_altloc(
            _select_mmcif_polymer_microheterogeneity(rows, declarations.polymer_monomers, altloc), altloc
        )
        return rows if include_water else _exclude_water_rows(rows, chemistry.components, declarations.entity_types)

    model_rows = _select_altloc(
        _select_mmcif_polymer_microheterogeneity(deposited_rows, source.declarations.polymer_monomers, altloc), altloc
    )
    model_row_ids = set(model_rows.row_index)
    model_components: dict[tuple[object, ...], set[object]] = {}
    for row in model_rows:
        if row.label_seq_id is not None:
            model_components.setdefault((row.chain_id, row.label_seq_id), set()).add(row.component_id)
    model_occurrences = frozenset(
        index
        for index, occurrence in enumerate(source_occurrences)
        if (
            not occurrence.rows
            or any(deposited_rows.row_index[position] in model_row_ids for position in occurrence.rows)
        )
        and (
            occurrence.label[:2] not in model_components
            or occurrence.label[3] in model_components[occurrence.label[:2]]
        )
    )
    rows = select(source_rows)
    topology_placements = set()
    if view.assembly_id is None:
        old_count = len(source_rows)
        source_rows, placements, topology_placements = _materialize_mmcif_crystallographic_copies(
            view,
            source_rows,
            rows,
            placements,
            source_sites=source_sites,
            model_occurrences=model_occurrences,
            include_water=include_water,
        )
        if len(source_rows) > old_count:
            rows = select(source_rows)
    else:
        source_rows = PlacedAtoms.assembly(view.rows, placements)
        rows = source_rows.select_source_rows(rows.row_index)

    catalog = CifCatalog._validated(
        view,
        rows,
        *read_mmcif_branch_entities(
            source.document,
            chemistry.components,
            rows,
            source_rows=deposited_rows,
            source_occurrences=source_occurrences,
            selected_entities=selected_entities,
        ),
    )
    planned = plan_cif(
        catalog,
        rows,
        len(source_rows),
        chemistry.components,
        placements=placements,
        topology_placements=topology_placements,
        include_water=include_water,
        include_declared_instances=view.assembly_id is None,
    )
    plan = planned.molecular
    bonds = resolve_cif_connections(
        catalog,
        source_rows,
        rows,
        planned,
        chemistry,
        placements,
        model_rows=model_rows,
        source_sites=source_sites,
        model_occurrences=model_occurrences,
    )
    if provenance is not None:
        retained_asym_ids = set(plan.source_asym_by_instance.values())
        provenance = frozenset(site for site in provenance if site.instance_id in retained_asym_ids)
    metadata = StructureMetadata(
        component_parents=parents,
        resolutions=read_mmcif_resolutions(source.block),
        formal_charge_atoms=chemistry.formal_charge_atoms,
        formal_charge_provenance=provenance,
    )
    structure = realize_structure(rows, plan, bonds.bonds, chemistry, metadata, boundary_leaving=bonds.boundary_leaving)
    return structure if include_hydrogens else structure.suppress_hydrogens()
