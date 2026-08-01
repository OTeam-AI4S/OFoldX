"""PDB evidence preparation, with PDB-only missing-residue and crystal rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal

from ...io.selection import _select_altloc
from ...io.source import PdbSource, StructureView
from ..molecular import ChemicalComponent
from ..structure import Structure
from .chemistry import resolve_components
from .connections import resolve_pdb_connections
from .pdb_catalog import PdbDeclarations
from .pdb_planner import plan_pdb
from .pipeline import realize_structure
from .scatter import StructureMetadata
from .selection import _exclude_water_rows


def materialize_pdb(
    view: StructureView,
    *,
    altloc: Literal["highest_occupancy", "first"],
    include_water: bool,
    include_hydrogens: bool,
    components: Mapping[str, ChemicalComponent] | None,
    component_resolver: Callable[[str], ChemicalComponent] | None,
) -> Structure:
    from ...io.pdb import (
        _materialize_pdb_crystallographic_copies,
        _pdb_conect_row_pairs,
        _pdb_connection_occurrences,
        _pdb_connection_symmetry_endpoints,
        _pdb_row_occurrence,
        prepare_pdb_view,
        read_pdb_component_bonds,
        read_pdb_resolutions,
    )

    if not isinstance(view.source, PdbSource):
        raise TypeError("PDB materialization requires PdbSource")
    model_rows = view.source.atoms.by_index(view.model_index)
    _pdb_conect_row_pairs(view.source.lines, model_rows, complete_model=True)
    selected_model_rows = _select_altloc(model_rows, altloc)

    assembly_id = view.assembly_id
    source_rows, lines, missing, placements = prepare_pdb_view(view)
    if not len(source_rows):
        raise ValueError("structure source contains no atoms for the requested model/assembly")
    declarations = PdbDeclarations.read(lines, missing, placements)
    if view.asym_ids is not None:
        declarations = declarations.select_chains(view.asym_ids)
    if placements is not None:
        declarations = declarations.select_chains({chain for _, chains in placements.values() for chain in chains})
    declared_ids = {name for sequence in declarations.sequences.values() for name in sequence}
    model_occurrences = {_pdb_row_occurrence(row) for row in model_rows}
    model_occurrences.update(
        (chain or None, number, insertion or None, component) for chain, component, number, insertion in missing
    )
    for line in lines:
        endpoints = _pdb_connection_occurrences(line)
        if endpoints is None:
            continue
        for endpoint in endpoints:
            if endpoint not in model_occurrences:
                raise ValueError(
                    f"PDB {line[:6].strip()} endpoint identity is missing from the selected model: {endpoint!r}"
                )
            declared_ids.add(endpoint[-1])
    chemistry = resolve_components(
        source_rows,
        declared_ids,
        source_components={},
        source_component_stereo_known={},
        components=components,
        component_resolver=component_resolver,
    )
    selected = _select_altloc(source_rows, altloc)
    rows = selected if include_water else _exclude_water_rows(selected, chemistry.components)
    if placements is None:
        symmetry_endpoints = _pdb_connection_symmetry_endpoints(
            lines,
            selected_model_rows,
            model_rows=model_rows,
            include_water=include_water,
            components=chemistry.components,
        )
        if symmetry_endpoints:
            old_count = len(source_rows)
            source_rows = _materialize_pdb_crystallographic_copies(
                source_rows,
                rows,
                lines,
                symmetry_endpoints,
                include_water=include_water,
                components=chemistry.components,
                model_rows=model_rows,
                selected_model_rows=selected_model_rows,
            )
            if len(source_rows) > old_count:
                rows = _select_altloc(source_rows, altloc)
                if not include_water:
                    rows = _exclude_water_rows(rows, chemistry.components)
    bonded_components = dict(chemistry.components)
    read_pdb_component_bonds(lines, rows, bonded_components)
    chemistry = replace(chemistry, components=bonded_components)
    planned = plan_pdb(
        declarations, rows, len(source_rows), chemistry.components, include_declared_instances=assembly_id is None
    )
    plan = planned.molecular
    bonds = resolve_pdb_connections(
        declarations,
        source_rows,
        rows,
        planned,
        chemistry,
        include_water=include_water,
        model_rows=model_rows,
        selected_model_rows=selected_model_rows,
        missing_residues=missing,
    )
    metadata = StructureMetadata(
        component_parents={}, resolutions=read_pdb_resolutions(lines), formal_charge_atoms=chemistry.formal_charge_atoms
    )
    structure = realize_structure(rows, plan, bonds.bonds, chemistry, metadata, boundary_leaving=bonds.boundary_leaving)
    if not bool(structure.coordinate_mask.any()):
        raise ValueError("selected structure contains no observed coordinates")
    return structure if include_hydrogens else structure.suppress_hydrogens()
