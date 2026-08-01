from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ...io.identity import SiteIndex
from ...io.records import AtomColumns, BranchResidueKey
from ...io.source import _require_cif_source
from ..components import get_builtin_component_polymer_type
from ..molecular import (
    AtomSite,
    ChemicalComponent,
    Molecule,
    MoleculeType,
    PolymerSequence,
    PolymerType,
    is_water_component,
)
from .cif_catalog import CifCatalog
from .plan import Occurrence, OccurrencePlan, PlanBuilder, ResidueKey, ResidueTable
from .polymers import (
    _ENTITY_POLY_TYPE_TO_POLYMER,
    _declared_polymer,
    _resolve_declared_component,
    _source_author_numbering,
)

_MMCIF_MOLECULE_TYPES: dict[str, MoleculeType] = {
    "macrolide": "macrolide",
    "non-polymer": "non-polymer",
    "water": "water",
}


class CifPolymer(NamedTuple):
    entity_id: str
    value: PolymerSequence
    observed: tuple[tuple[int, ResidueKey], ...]
    sequence_declared: bool


class CifPlan(NamedTuple):
    molecular: OccurrencePlan
    sites: SiteIndex


def plan_cif(
    catalog: CifCatalog,
    rows: AtomColumns,
    source_row_count: int,
    components: Mapping[str, ChemicalComponent],
    *,
    placements: Mapping[tuple[str | None, str], NDArray[np.float64]],
    topology_placements: set[tuple[str | None, str, str | None]],
    include_water: bool,
    include_declared_instances: bool,
) -> CifPlan:
    """Interpret CIF evidence with state owned by this invocation."""
    return _CifPlanner(
        catalog,
        rows,
        source_row_count,
        components,
        placements=placements,
        topology_placements=topology_placements,
        include_water=include_water,
        include_declared_instances=include_declared_instances,
    ).run()


class _CifPlanner:
    """Interpret selected CIF entities into occurrences and connection addresses."""

    def __init__(
        self,
        catalog: CifCatalog,
        rows: AtomColumns,
        source_row_count: int,
        components: Mapping[str, ChemicalComponent],
        *,
        placements: Mapping[tuple[str | None, str], NDArray[np.float64]],
        topology_placements: set[tuple[str | None, str, str | None]],
        include_water: bool,
        include_declared_instances: bool,
    ) -> None:
        self.catalog = catalog
        self.residues = ResidueTable(rows)
        self.source_row_count = source_row_count
        self.components = components
        self.placements = placements
        self.topology_placements = topology_placements
        self.include_water = include_water
        self.include_declared_instances = include_declared_instances
        self.builder = PlanBuilder()
        self.has_connections = "struct_conn" in _require_cif_source(catalog.view.source).block
        self.aliases = SiteIndex()
        self.matched_nonpoly_occurrences: set[tuple[str, str]] = set()

    def run(self) -> CifPlan:
        for (symmetry_id, chain_id), residue_order in self.residues.chains.items():
            self._plan_chain(symmetry_id, chain_id, residue_order)
        self._plan_unobserved_nonpolymers()
        self._plan_unobserved_entities()
        for item in self.builder.occurrences:
            self._bind_observed(item)
        plan = self.builder.build()
        self.aliases.seal()
        return CifPlan(plan, self.aliases)

    def _plan_chain(self, symmetry_id: str | None, chain_id: str, residue_order: Sequence[ResidueKey]) -> None:
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        polymer_residues = []
        for key in residue_order:
            position = self.residues.positions[key][0]
            source_entity_id = self.residues.rows.entity_id[position]
            entity_type = None if source_entity_id is None else self.catalog.entity_types.get(str(source_entity_id))
            if entity_type == "polymer" or (
                entity_type is None
                and get_builtin_component_polymer_type(str(self.residues.rows.component_id[position])) is not None
            ):
                polymer_residues.append(key)
        polymer_segments = [polymer_residues] if polymer_residues else []
        for segment_ordinal, segment_residues in enumerate(polymer_segments):
            self._plan_polymer(symmetry_id, chain_id, base_instance_id, segment_ordinal, segment_residues)
        polymer_residue_set = set(polymer_residues)
        nonpolymer_residues = [key for key in residue_order if key not in polymer_residue_set]
        molecule_groups: list[tuple[ResidueKey, ...]] = []
        grouped_by_entity: dict[str, list[ResidueKey]] = {}
        for residue_key in nonpolymer_residues:
            source_entity_id = self.residues.rows.entity_id[self.residues.positions[residue_key][0]]
            if source_entity_id is not None and source_entity_id in self.catalog.branch_entities:
                grouped_by_entity.setdefault(str(source_entity_id), []).append(residue_key)
            else:
                molecule_groups.append((residue_key,))
        molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
        molecule_groups.sort(
            key=lambda group: min(
                int(self.residues.rows.row_index[position])
                for key in group
                for position in self.residues.positions[key]
            )
        )
        for group_residues in molecule_groups:
            self._plan_molecule(symmetry_id, chain_id, base_instance_id, group_residues)

    def _resolve_polymer(
        self,
        symmetry_id: str | None,
        chain_id: str,
        base_instance_id: str,
        segment_ordinal: int,
        segment_residues: Sequence[ResidueKey],
    ) -> CifPolymer:
        segment_first_position = None if not segment_residues else self.residues.positions[segment_residues[0]][0]
        source_entity_id = (
            None
            if segment_first_position is None or self.residues.rows.entity_id[segment_first_position] is None
            else str(self.residues.rows.entity_id[segment_first_position])
        )
        declared = None
        if source_entity_id is not None:
            declared = _declared_polymer(
                source_entity_id,
                self.catalog.declarations.polymer_monomers,
                self.catalog.declarations.polymer_types,
                self.components,
                segment_residues,
                self.residues.positions,
                self.residues.rows,
            )
        if declared is not None:
            declared_entity, indexed = declared
            assert source_entity_id is not None
            entity_id = source_entity_id
            self.builder.register_entity(entity_id, declared_entity, allow_variants=True)
            entity = declared_entity
        else:
            classified_component_ids = tuple(
                str(self.residues.rows.component_id[self.residues.positions[key][0]]) for key in segment_residues
            )
            classified_values: set[PolymerType | None] = set()
            for component_id in classified_component_ids:
                classification = get_builtin_component_polymer_type(component_id)
                classified_values.add(classification)
            classified = {polymer_type for polymer_type in classified_values if polymer_type is not None}
            if classified == {"dna", "rna"}:
                chain_polymer_type: PolymerType = "dna_rna_hybrid"
            elif len(classified) == 1:
                chain_polymer_type = next(iter(classified))
            elif not classified and source_entity_id is not None:
                mapped = _ENTITY_POLY_TYPE_TO_POLYMER.get(
                    self.catalog.declarations.polymer_types.get(source_entity_id, "")
                )
                if mapped is None:
                    raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
                chain_polymer_type = mapped
            else:
                raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
            if source_entity_id is not None:
                entity_id = source_entity_id
            elif segment_ordinal == 0:
                entity_id = f"polymer_{chain_id}"
            else:
                entity_id = f"polymer_{chain_id}#{segment_ordinal}"
            resolved = PolymerSequence(
                chain_polymer_type,
                tuple(
                    self.components[str(self.residues.rows.component_id[self.residues.positions[key][0]])]
                    for key in segment_residues
                ),
            )
            indexed = tuple(enumerate(segment_residues))
            entity = self.builder.register_entity(entity_id, resolved)
        return CifPolymer(entity_id=entity_id, value=entity, observed=indexed, sequence_declared=declared is not None)

    def _plan_polymer(
        self,
        symmetry_id: str | None,
        chain_id: str,
        base_instance_id: str,
        segment_ordinal: int,
        segment_residues: Sequence[ResidueKey],
    ) -> None:
        resolved = self._resolve_polymer(symmetry_id, chain_id, base_instance_id, segment_ordinal, segment_residues)
        entity_id = resolved.entity_id
        entity = resolved.value
        indexed = resolved.observed
        sequence_declared = resolved.sequence_declared
        instance_id = self.builder.name_instance(base_instance_id)
        self.builder.set_source_asym(instance_id, chain_id)
        if sequence_declared:
            observed_indices = {index for index, _ in indexed}
            for component_index, component in enumerate(entity.components):
                if component_index in observed_indices:
                    continue
                author = _source_author_numbering(
                    self.catalog.declarations.author_numbering, chain_id, component_index, component.component_id
                )
                missing_insertion_code = None if author is None else author[1]
                for atom in component.atoms if self.has_connections else ():
                    self.aliases.add(
                        "label",
                        (
                            symmetry_id,
                            chain_id,
                            component_index + 1,
                            missing_insertion_code,
                            component.component_id,
                            atom.atom_id,
                        ),
                        AtomSite(instance_id, component_index, atom.atom_id),
                    )
                for atom in component.atoms if self.has_connections else ():
                    self.aliases.add(
                        "auth",
                        (
                            symmetry_id,
                            self.catalog.author_chains.get(chain_id),
                            None if author is None else author[0],
                            missing_insertion_code,
                            component.component_id,
                            None,
                        ),
                        AtomSite(instance_id, component_index, atom.atom_id),
                    )
                if author is not None and author[0] is not None:
                    self.builder.set_residue_number((instance_id, component_index), (str(author[0]), author[1] or ""))
        self.builder.add_occurrence(
            Occurrence(
                order=(
                    min(
                        int(self.residues.rows.row_index[position])
                        for key in segment_residues
                        for position in self.residues.positions[key]
                    )
                    if segment_residues
                    else self.source_row_count + segment_ordinal
                ),
                instance_id=instance_id,
                entity_id=entity_id,
                value=entity,
                observed=indexed,
            )
        )

    def _plan_molecule(
        self, symmetry_id: str | None, chain_id: str, base_instance_id: str, group_residues: Sequence[ResidueKey]
    ) -> None:
        first_position = self.residues.positions[group_residues[0]][0]
        first_component = self.components[str(self.residues.rows.component_id[first_position])]
        residue_label = self.residues.rows.auth_residue_id[first_position]
        if residue_label is None:
            residue_label = self.residues.rows.residue_id[first_position]
        residue_identity = f"{residue_label}{self.residues.rows.insertion_code[first_position] or ''}"
        source_entity_id = (
            None
            if self.residues.rows.entity_id[first_position] is None
            else str(self.residues.rows.entity_id[first_position])
        )
        if source_entity_id is not None and source_entity_id in self.catalog.branch_entities:
            entity = self.catalog.branch_entity(source_entity_id, chain_id)
            entity_id = source_entity_id
            instance_seed = base_instance_id
            indexed_residues: list[tuple[int, ResidueKey]] = []
            for residue_key in group_residues:
                position = self.residues.positions[residue_key][0]
                branch_component_index: int | None = None
                source_keys = (
                    BranchResidueKey(
                        self.residues.rows.chain_id[position],
                        self.residues.rows.entity_id[position],
                        self.residues.rows.component_id[position],
                        self.residues.rows.chain_id[position],
                        None,
                        self.residues.rows.insertion_code[position],
                    ),
                    BranchResidueKey(
                        self.residues.rows.chain_id[position],
                        self.residues.rows.entity_id[position],
                        self.residues.rows.auth_component_id[position],
                        self.residues.rows.auth_chain_id[position],
                        self.residues.rows.auth_residue_id[position],
                        self.residues.rows.insertion_code[position],
                    ),
                )
                for branch_lookup_key in source_keys:
                    branch_component_index = self.catalog.branch_component_indices.get(branch_lookup_key)
                    if branch_component_index is not None:
                        break
                if branch_component_index is None:
                    raise ValueError(f"cannot map branched-entity residue {residue_key!r} to its declaration")
                indexed_residues.append((branch_component_index, residue_key))
            if len({index for index, _ in indexed_residues}) != len(indexed_residues):
                raise ValueError(f"branched entity {source_entity_id!r} maps multiple residues to one component")
            indexed_residues.sort()
        elif source_entity_id is not None:
            source_entity_type = self.catalog.entity_types.get(source_entity_id)
            if source_entity_type == "branched":
                raise ValueError(f"branched entity {source_entity_id!r} has no branch declaration")
            if source_entity_type is not None and source_entity_type not in _MMCIF_MOLECULE_TYPES:
                raise ValueError(
                    f"mmCIF entity {source_entity_id!r} is declared {source_entity_type!r} but observed as a molecule"
                )
            source_number = (
                None
                if self.residues.rows.auth_residue_id[first_position] is None
                else str(self.residues.rows.auth_residue_id[first_position])
            )
            source_insertion = str(self.residues.rows.insertion_code[first_position] or "")
            source_chain = str(
                self.residues.rows.auth_chain_id[first_position] or self.residues.rows.chain_id[first_position]
            )
            source_monomers = {
                str(self.residues.rows.component_id[first_position]),
                str(
                    self.residues.rows.auth_component_id[first_position]
                    or self.residues.rows.component_id[first_position]
                ),
            }
            declared_occurrences = self.catalog.declarations.nonpolymer_occurrences.get(
                str(self.residues.rows.chain_id[first_position]), ()
            )
            occurrence_matches = [
                ndb_code
                for ndb_code, occurrence_entity, monomer, author_number, insertion, pdb_chain in declared_occurrences
                if occurrence_entity == source_entity_id
                and monomer in source_monomers
                and (author_number is None or source_number == author_number)
                and (source_insertion == (insertion or ""))
                and (pdb_chain is None or pdb_chain == source_chain)
            ]
            if len(occurrence_matches) > 1:
                raise ValueError(f"cannot uniquely match non-polymer occurrence {residue_identity!r}")
            if declared_occurrences and (not occurrence_matches):
                raise ValueError(f"cannot match non-polymer occurrence {residue_identity!r} to its declaration")
            if occurrence_matches:
                self.matched_nonpoly_occurrences.add(
                    (str(self.residues.rows.chain_id[first_position]), occurrence_matches[0])
                )
            molecule_type = (
                _MMCIF_MOLECULE_TYPES[source_entity_type]
                if source_entity_type is not None
                else "water" if is_water_component(first_component) else "non-polymer"
            )
            entity = Molecule((first_component,), molecule_type=molecule_type)
            entity_id = source_entity_id
            instance_seed = base_instance_id
            indexed_residues = [(0, group_residues[0])]
        else:
            entity = Molecule(
                (first_component,), molecule_type="water" if is_water_component(first_component) else "non-polymer"
            )
            entity_id = f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
            instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
            indexed_residues = [(0, group_residues[0])]
        if source_entity_id is not None and source_entity_id in self.catalog.branch_entities:
            self.builder.register_entity(entity_id, entity, allow_variants=True)
        else:
            entity = self.builder.register_entity(entity_id, entity)
        instance_id = self.builder.name_instance(instance_seed)
        self.builder.set_source_asym(instance_id, chain_id)
        self.builder.add_occurrence(
            Occurrence(
                order=min(
                    int(self.residues.rows.row_index[position])
                    for key in group_residues
                    for position in self.residues.positions[key]
                ),
                instance_id=instance_id,
                entity_id=entity_id,
                value=entity,
                observed=tuple(indexed_residues),
            )
        )

    def _plan_unobserved_nonpolymers(self) -> None:
        nonpoly_declaration_placements: tuple[tuple[str | None, str, str | None], ...]
        nonpoly_declaration_placements = (
            (
                *tuple((None, asym_id, None) for asym_id in self.catalog.declarations.nonpolymer_occurrences),
                *tuple(sorted(placement for placement in self.topology_placements if placement[2] is not None)),
            )
            if self.include_declared_instances
            else tuple(
                (
                    (symmetry_id, asym_id, None)
                    for symmetry_id, asym_id in self.placements
                    if asym_id in self.catalog.declarations.nonpolymer_occurrences
                )
            )
        )
        if nonpoly_declaration_placements:
            declaration_index = 0
            for symmetry_id, asym_id, target_ndb_code in nonpoly_declaration_placements:
                occurrences = self.catalog.declarations.nonpolymer_occurrences[asym_id]
                for ndb_code, source_entity_id, monomer, author_number, insertion, pdb_chain in occurrences:
                    if target_ndb_code is not None and ndb_code != target_ndb_code:
                        continue
                    if (
                        target_ndb_code is None
                        and (asym_id, ndb_code) in self.matched_nonpoly_occurrences
                        and ((symmetry_id, asym_id) in self.residues.chains)
                    ):
                        continue
                    source_entity_type = self.catalog.entity_types.get(source_entity_id)
                    if not self.include_water and source_entity_type == "water":
                        continue
                    if source_entity_type is not None and source_entity_type not in _MMCIF_MOLECULE_TYPES:
                        raise ValueError(
                            f"mmCIF entity {source_entity_id!r} is declared {source_entity_type!r} "
                            "but used as a molecule"
                        )
                    component = _resolve_declared_component(monomer, self.components)
                    if not self.include_water and source_entity_type is None and is_water_component(component):
                        continue
                    molecule_type = (
                        _MMCIF_MOLECULE_TYPES[source_entity_type]
                        if source_entity_type is not None
                        else "water" if is_water_component(component) else "non-polymer"
                    )
                    entity = self.builder.register_entity(
                        source_entity_id, Molecule((component,), molecule_type=molecule_type)
                    )
                    residue_identity = author_number or ndb_code
                    instance_seed = asym_id if symmetry_id is None else f"{asym_id}#{symmetry_id}"
                    occurrence_id = (
                        f"{instance_seed}:{residue_identity}:{monomer}" if len(occurrences) > 1 else instance_seed
                    )
                    instance_id = self.builder.name_instance(occurrence_id)
                    self.builder.set_source_asym(instance_id, asym_id)
                    if author_number is not None:
                        self.builder.set_residue_number((instance_id, 0), (author_number, insertion or ""))
                    for atom in component.atoms if self.has_connections else ():
                        site = AtomSite(instance_id, 0, atom.atom_id)
                        self.aliases.add("label", (symmetry_id, asym_id, None, insertion, monomer, atom.atom_id), site)
                        self.aliases.add(
                            "auth", (symmetry_id, pdb_chain, author_number, insertion, monomer, None), site
                        )
                    self.builder.add_occurrence(
                        Occurrence(
                            order=self.source_row_count + declaration_index,
                            instance_id=instance_id,
                            entity_id=source_entity_id,
                            value=entity,
                            observed=(),
                        )
                    )
                    declaration_index += 1

    def _plan_unobserved_entities(self) -> None:
        entity: PolymerSequence | Molecule
        observed_asym_placements = set(self.residues.chains)
        declared_instances: tuple[tuple[str | None, str, str], ...]
        if self.include_declared_instances:
            values: list[tuple[str | None, str, str]] = [
                (None, asym_id, entity_id) for asym_id, entity_id in self.catalog.declarations.asym_entities.items()
            ]
            values.extend(
                (
                    (symmetry_id, asym_id, self.catalog.declarations.asym_entities[asym_id])
                    for symmetry_id, asym_id, ndb_code in sorted(self.topology_placements)
                    if ndb_code is None
                )
            )
            declared_instances = tuple(values)
        else:
            declared_instances_list: list[tuple[str | None, str, str]] = []
            for symmetry_id, asym_id in self.placements:
                source_entity_id = self.catalog.declarations.asym_entities.get(asym_id)
                if source_entity_id is None:
                    raise ValueError(f"mmCIF assembly references undeclared asym {asym_id!r}")
                declared_instances_list.append((symmetry_id, asym_id, source_entity_id))
            declared_instances = tuple(declared_instances_list)
        for declaration_index, (symmetry_id, asym_id, source_entity_id) in enumerate(declared_instances):
            if (
                symmetry_id,
                asym_id,
            ) in observed_asym_placements or asym_id in self.catalog.declarations.nonpolymer_occurrences:
                continue
            source_entity_type = self.catalog.entity_types.get(source_entity_id)
            if not self.include_water and source_entity_type == "water":
                continue
            if source_entity_type == "polymer":
                declared = _declared_polymer(
                    source_entity_id,
                    self.catalog.declarations.polymer_monomers,
                    self.catalog.declarations.polymer_types,
                    self.components,
                    [],
                    self.residues.positions,
                    self.residues.rows,
                )
                if declared is None:
                    raise ValueError(
                        f"unobserved mmCIF polymer entity {source_entity_id!r} has no sequence declaration"
                    )
                resolved, _ = declared
                entity_id = source_entity_id
                self.builder.register_entity(entity_id, resolved, allow_variants=True)
                entity = resolved
            elif source_entity_id is not None and source_entity_id in self.catalog.branch_entities:
                entity_id = source_entity_id
                entity = self.catalog.branch_entity(source_entity_id, asym_id)
                self.builder.register_entity(entity_id, entity, allow_variants=True)
            else:
                declared_component_id = self.catalog.declarations.nonpolymer_components.get(source_entity_id)
                if declared_component_id is None:
                    if source_entity_type in _MMCIF_MOLECULE_TYPES:
                        raise ValueError(
                            f"unobserved mmCIF {source_entity_type} entity {source_entity_id!r} "
                            "has no component declaration"
                        )
                    continue
                component = _resolve_declared_component(declared_component_id, self.components)
                inferred_entity_type = (
                    "water"
                    if source_entity_type is None and is_water_component(component)
                    else source_entity_type or "non-polymer"
                )
                if not self.include_water and inferred_entity_type == "water":
                    continue
                entity_id = source_entity_id
                entity = self.builder.register_entity(
                    entity_id, Molecule((component,), molecule_type=_MMCIF_MOLECULE_TYPES[inferred_entity_type])
                )
            instance_seed = asym_id if symmetry_id is None else f"{asym_id}#{symmetry_id}"
            instance_id = self.builder.name_instance(instance_seed)
            self.builder.set_source_asym(instance_id, asym_id)
            self._bind_unobserved_entity(
                instance_id, entity, asym_id, symmetry_id, source_entity_id, source_entity_type
            )
            self.builder.add_occurrence(
                Occurrence(
                    order=self.source_row_count + declaration_index,
                    instance_id=instance_id,
                    entity_id=entity_id,
                    value=entity,
                    observed=(),
                )
            )

    def _bind_observed(self, item: Occurrence) -> None:
        for component_index, residue_key in item.observed:
            self._bind_component(item, component_index, residue_key)
        if isinstance(item.value, Molecule) and item.observed:
            self._bind_missing_branch_components(item)

    def _bind_component(self, item: Occurrence, component_index: int, residue_key: ResidueKey) -> None:
        instance_id, entity = item.instance_id, item.value
        residue_positions = self.residues.positions[residue_key]
        first_position = residue_positions[0]
        source_number = self.residues.rows.auth_residue_id[first_position]
        source_insertion = str(self.residues.rows.insertion_code[first_position] or "")
        author = _source_author_numbering(
            self.catalog.declarations.author_numbering,
            str(self.residues.rows.chain_id[first_position]),
            component_index,
            entity.components[component_index].component_id,
        )
        if author is not None:
            if source_number is None:
                source_number = author[0]
            if self.residues.rows.insertion_code[first_position] is None:
                source_insertion = author[1] or ""
        if source_number is not None:
            self.builder.set_residue_number((instance_id, component_index), (str(source_number), source_insertion))
        component = entity.components[component_index]
        label_prefix = (
            self.residues.rows.symmetry_id[first_position],
            self.residues.rows.chain_id[first_position],
            self.residues.rows.label_seq_id[first_position],
            self.residues.rows.insertion_code[first_position],
            component.component_id,
        )
        auth_prefix = (
            self.residues.rows.symmetry_id[first_position],
            self.residues.rows.auth_chain_id[first_position],
            self.residues.rows.auth_residue_id[first_position],
            self.residues.rows.insertion_code[first_position],
            self.residues.rows.auth_component_id[first_position],
        )
        for atom in component.atoms if self.has_connections else ():
            site = AtomSite(instance_id, component_index, atom.atom_id)
            self.aliases.add("label", (*label_prefix, atom.atom_id), site)
            self.aliases.add("auth", (*auth_prefix, None), site)
        self.builder.bind_atoms(
            (instance_id, component_index),
            self.residues.rows.row_index[residue_positions],
            self.residues.rows.atom_name[residue_positions],
        )
        if self.has_connections:
            for position in residue_positions:
                site = AtomSite(instance_id, component_index, str(self.residues.rows.atom_name[position]))
                self.aliases.add(
                    "label",
                    (
                        self.residues.rows.symmetry_id[position],
                        self.residues.rows.chain_id[position],
                        self.residues.rows.label_seq_id[position],
                        self.residues.rows.insertion_code[position],
                        self.residues.rows.component_id[position],
                        self.residues.rows.atom_name[position],
                    ),
                    site,
                )
                self.aliases.add(
                    "auth",
                    (
                        self.residues.rows.symmetry_id[position],
                        self.residues.rows.auth_chain_id[position],
                        self.residues.rows.auth_residue_id[position],
                        self.residues.rows.insertion_code[position],
                        self.residues.rows.auth_component_id[position],
                        self.residues.rows.auth_atom_name[position],
                    ),
                    site,
                )

    def _bind_missing_branch_components(self, item: Occurrence) -> None:
        instance_id, entity, indexed = item.instance_id, item.value, item.observed
        first_position = self.residues.positions[indexed[0][1]][0]
        source_entity_id = self.residues.rows.entity_id[first_position]
        observed_components = {component_index for component_index, _ in indexed}
        for branch_source_key, component_index in self.catalog.branch_component_indices.items():
            if (
                branch_source_key.entity_id != source_entity_id
                or branch_source_key.label_asym_id != self.residues.rows.chain_id[first_position]
                or component_index in observed_components
            ):
                continue
            author = self.catalog.declarations.author_numbering.get(
                (str(branch_source_key.label_asym_id), component_index + 1)
            )
            branch_insertion_code = (
                None if branch_source_key.insertion_code is None else str(branch_source_key.insertion_code)
            )
            insertion = branch_insertion_code if author is None else author[1] or branch_insertion_code
            if author is not None and author[0] is not None:
                self.builder.set_residue_number((instance_id, component_index), (author[0], insertion or ""))
            component = entity.components[component_index]
            for atom in component.atoms if self.has_connections else ():
                site = AtomSite(instance_id, component_index, atom.atom_id)
                self.aliases.add(
                    "label",
                    (
                        self.residues.rows.symmetry_id[first_position],
                        branch_source_key.label_asym_id,
                        None,
                        insertion,
                        branch_source_key.component_id,
                        atom.atom_id,
                    ),
                    site,
                )
                self.aliases.add(
                    "auth",
                    (
                        self.residues.rows.symmetry_id[first_position],
                        branch_source_key.source_asym_id,
                        branch_source_key.source_residue_id,
                        insertion,
                        branch_source_key.component_id,
                        atom.atom_id,
                    ),
                    site,
                )

    def _bind_unobserved_entity(
        self,
        instance_id: str,
        entity: PolymerSequence | Molecule,
        asym_id: str,
        symmetry_id: str | None,
        source_entity_id: str,
        source_entity_type: str | None,
    ) -> None:
        for component_index, component in enumerate(entity.components):
            author = _source_author_numbering(
                self.catalog.declarations.author_numbering, asym_id, component_index, component.component_id
            )
            source_author_number = None if author is None else author[0]
            insertion = None if author is None else author[1]
            if source_author_number is not None:
                self.builder.set_residue_number((instance_id, component_index), (source_author_number, insertion or ""))
            if source_entity_type == "polymer":
                label_sequence: int | None = component_index + 1
            else:
                label_sequence = None
            for atom in component.atoms if self.has_connections else ():
                site = AtomSite(instance_id, component_index, atom.atom_id)
                self.aliases.add(
                    "label",
                    (symmetry_id, asym_id, label_sequence, insertion, component.component_id, atom.atom_id),
                    site,
                )
                for branch_source_key, branch_index in self.catalog.branch_component_indices.items():
                    if (
                        branch_index != component_index
                        or branch_source_key.label_asym_id != asym_id
                        or branch_source_key.entity_id != source_entity_id
                    ):
                        continue
                    self.aliases.add(
                        "label",
                        (
                            symmetry_id,
                            branch_source_key.label_asym_id,
                            None,
                            branch_source_key.insertion_code,
                            branch_source_key.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
                    self.aliases.add(
                        "auth",
                        (
                            symmetry_id,
                            branch_source_key.source_asym_id,
                            branch_source_key.source_residue_id,
                            branch_source_key.insertion_code,
                            branch_source_key.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
                self.aliases.add(
                    "auth",
                    (
                        symmetry_id,
                        self.catalog.author_chains.get(asym_id),
                        source_author_number,
                        insertion,
                        component.component_id,
                        None,
                    ),
                    site,
                )
