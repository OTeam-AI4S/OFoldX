from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import NamedTuple

from ...io.records import AtomColumns, PdbSiteKey
from ..components import get_builtin_component_polymer_type
from ..molecular import AtomSite, ChemicalComponent, Molecule, PolymerSequence, PolymerType, is_water_component
from .pdb_catalog import PdbDeclarations
from .plan import Occurrence, OccurrencePlan, PlanBuilder, ResidueKey, ResidueTable
from .polymers import _pdb_component_polymer_type, _pdb_polymer_segments, _pdb_polymer_with_declared_missing_residues


class PdbPolymer(NamedTuple):
    entity_id: str
    value: PolymerSequence
    observed: tuple[tuple[int, ResidueKey], ...]
    unobserved: tuple[tuple[int, str, str], ...]


class PdbPlan(NamedTuple):
    """Retained graph plan and canonical addresses, including unobserved atoms."""

    molecular: OccurrencePlan
    sites: Mapping[PdbSiteKey, AtomSite]
    ambiguous_sites: frozenset[PdbSiteKey]


def plan_pdb(
    declarations: PdbDeclarations,
    rows: AtomColumns,
    source_row_count: int,
    components: Mapping[str, ChemicalComponent],
    *,
    include_declared_instances: bool,
) -> PdbPlan:
    """Interpret PDB evidence with state owned by this invocation."""
    return _PdbPlanner(
        declarations, rows, source_row_count, components, include_declared_instances=include_declared_instances
    ).run()


class _PdbPlanner:
    """Resolve PDB sequence, segment and modified-residue evidence into occurrences."""

    def __init__(
        self,
        declarations: PdbDeclarations,
        rows: AtomColumns,
        source_row_count: int,
        components: Mapping[str, ChemicalComponent],
        *,
        include_declared_instances: bool,
    ) -> None:
        if not len(rows):
            raise ValueError("alternate-location selection contains no atoms")
        self.declarations = declarations
        self.residues = ResidueTable(rows)
        self.source_row_count = source_row_count
        self.components = components
        self.include_declared_instances = include_declared_instances
        self.builder = PlanBuilder()
        self.sites: dict[PdbSiteKey, AtomSite] = {}
        self.ambiguous_sites: set[PdbSiteKey] = set()

    def run(self) -> PdbPlan:
        chain_items = list(self.residues.chains.items())
        declared_pdb_chains: tuple[tuple[str | None, str], ...]
        if self.include_declared_instances:
            declared_pdb_chains = tuple((None, chain_id) for chain_id in self.declarations.sequences)
        else:
            assert self.declarations.assembly_placements is not None
            declared_pdb_chains = tuple(
                (
                    (symmetry_id, chain_id)
                    for symmetry_id, (_, chains) in self.declarations.assembly_placements.items()
                    for chain_id in self.declarations.sequences
                    if chain_id in chains
                )
            )
        chain_items.extend(
            (placement, ()) for placement in declared_pdb_chains if placement not in self.residues.chains
        )
        for (symmetry_id, chain_id), residue_order in chain_items:
            self._plan_chain(symmetry_id, chain_id, residue_order)
        for item in self.builder.occurrences:
            self._bind_observed(item)
        return PdbPlan(self.builder.build(), MappingProxyType(dict(self.sites)), frozenset(self.ambiguous_sites))

    def _add_alias(self, key: PdbSiteKey, site: AtomSite) -> None:
        if key in self.ambiguous_sites:
            return
        previous = self.sites.get(key)
        if previous is not None and previous != site:
            self.sites.pop(key)
            self.ambiguous_sites.add(key)
        else:
            self.sites[key] = site

    def _plan_chain(self, symmetry_id: str | None, chain_id: str, residue_order: Sequence[ResidueKey]) -> None:
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        declared_pdb_sequence = self.declarations.sequences.get(chain_id)
        polymer_residues = []
        for key in residue_order:
            residue_positions = self.residues.positions[key]
            first_position = residue_positions[0]
            source_number = self.residues.rows.auth_residue_id[first_position]
            if source_number is None:
                source_number = self.residues.rows.residue_id[first_position]
            if (
                any(self.residues.rows.group[position] == "ATOM" for position in residue_positions)
                or (
                    chain_id,
                    int(source_number),
                    str(self.residues.rows.insertion_code[first_position] or ""),
                    str(self.residues.rows.component_id[first_position]),
                )
                in self.declarations.modified_residues
            ):
                polymer_residues.append(key)
        polymer_segments = (
            [polymer_residues]
            if declared_pdb_sequence is not None
            else _pdb_polymer_segments(
                polymer_residues,
                self.residues.positions,
                self.residues.rows.serial,
                self.declarations.segment_by_serial,
            )
        )
        if declared_pdb_sequence is None and any(
            (declared_chain == chain_id for declared_chain, *_ in self.declarations.missing_residues)
        ):
            raise ValueError(f"PDB REMARK 465 for chain {chain_id!r} requires a SEQRES declaration")
        segment_declarations = [self.declarations.missing_residues for _ in polymer_segments]
        if len(polymer_segments) > 1:
            assigned: list[list[tuple[str, str, int, str]]] = [[] for _ in polymer_segments]
            segment_positions: list[list[tuple[int, str]]] = []
            for segment_residues in polymer_segments:
                segment_position_values: list[tuple[int, str]] = []
                for residue_key in segment_residues:
                    first_position = self.residues.positions[residue_key][0]
                    number = self.residues.rows.auth_residue_id[first_position]
                    if number is None:
                        number = self.residues.rows.residue_id[first_position]
                    if number is None:
                        raise ValueError("PDB polymer residue lacks author numbering")
                    segment_position_values.append(
                        (int(number), str(self.residues.rows.insertion_code[first_position] or ""))
                    )
                segment_positions.append(segment_position_values)
            for declaration in self.declarations.missing_residues:
                declared_chain, _, number, declared_insertion_code = declaration
                if declared_chain != chain_id:
                    continue
                declared_position = (number, declared_insertion_code)
                candidates = [
                    index
                    for index, segment_position_values in enumerate(segment_positions)
                    if declared_position in segment_position_values
                    or segment_position_values[0] < declared_position < segment_position_values[-1]
                ]
                if len(candidates) == 1:
                    assigned[candidates[0]].append(declaration)
            segment_declarations = [tuple(values) for values in assigned]
        for segment_ordinal, segment_residues in enumerate(polymer_segments):
            self._plan_polymer(
                symmetry_id,
                chain_id,
                base_instance_id,
                segment_ordinal,
                segment_residues,
                declared_pdb_sequence,
                segment_declarations[segment_ordinal],
                len(polymer_segments) == 1,
            )
        polymer_residue_set = set(polymer_residues)
        nonpolymer_residues = [key for key in residue_order if key not in polymer_residue_set]
        molecule_groups: list[tuple[ResidueKey, ...]] = []
        molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)
        for group_residues in molecule_groups:
            self._plan_molecule(symmetry_id, chain_id, base_instance_id, group_residues)

    def _resolve_polymer(
        self,
        symmetry_id: str | None,
        chain_id: str,
        base_instance_id: str,
        segment_ordinal: int,
        segment_residues: Sequence[ResidueKey],
        declared_pdb_sequence: tuple[str, ...] | None,
        missing_residues: tuple[tuple[str, str, int, str], ...],
        include_terminal: bool,
    ) -> PdbPolymer:
        pdb_unmodeled: tuple[tuple[int, str, str], ...] = ()
        classified_component_ids = (
            declared_pdb_sequence
            if declared_pdb_sequence is not None
            else tuple(
                str(self.residues.rows.component_id[self.residues.positions[key][0]]) for key in segment_residues
            )
        )
        chain_polymer_type = self._classify_polymer(
            chain_id, base_instance_id, classified_component_ids, segment_residues
        )
        if chain_id in self.declarations.entity_by_chain:
            entity_id = self.declarations.entity_by_chain[chain_id]
        elif segment_ordinal == 0:
            entity_id = f"polymer_{chain_id}"
        else:
            entity_id = f"polymer_{chain_id}#{segment_ordinal}"
        resolved, indexed, pdb_unmodeled = _pdb_polymer_with_declared_missing_residues(
            chain_polymer_type,
            chain_id,
            segment_residues,
            self.residues.positions,
            self.residues.rows,
            self.components,
            missing_residues,
            self.declarations.modified_residues,
            declared_sequence=declared_pdb_sequence,
            include_terminal=include_terminal,
        )
        if chain_id in self.declarations.entity_by_chain:
            self.builder.register_entity(entity_id, resolved, allow_variants=True)
            entity = resolved
        else:
            entity = self.builder.register_entity(entity_id, resolved)
        return PdbPolymer(entity_id=entity_id, value=entity, observed=indexed, unobserved=pdb_unmodeled)

    def _plan_polymer(
        self,
        symmetry_id: str | None,
        chain_id: str,
        base_instance_id: str,
        segment_ordinal: int,
        segment_residues: Sequence[ResidueKey],
        declared_pdb_sequence: tuple[str, ...] | None,
        missing_residues: tuple[tuple[str, str, int, str], ...],
        include_terminal: bool,
    ) -> None:
        resolved = self._resolve_polymer(
            symmetry_id,
            chain_id,
            base_instance_id,
            segment_ordinal,
            segment_residues,
            declared_pdb_sequence,
            missing_residues,
            include_terminal,
        )
        entity_id = resolved.entity_id
        entity = resolved.value
        indexed = resolved.observed
        pdb_unmodeled = resolved.unobserved
        instance_id = self.builder.name_instance(base_instance_id)
        for component_index, residue_id, insertion_code in pdb_unmodeled:
            self.builder.set_residue_number((instance_id, component_index), (residue_id, insertion_code))
        source_positions: dict[int, tuple[int, str]] = {
            component_index: (int(residue_id), insertion_code)
            for component_index, residue_id, insertion_code in pdb_unmodeled
        }
        source_component_ids = dict(enumerate(declared_pdb_sequence)) if declared_pdb_sequence is not None else {}
        for component_index, residue_key in indexed:
            first_position = self.residues.positions[residue_key][0]
            residue_number = self.residues.rows.auth_residue_id[first_position]
            if residue_number is None:
                residue_number = self.residues.rows.residue_id[first_position]
            if residue_number is None:
                raise ValueError("PDB polymer residue lacks author numbering")
            source_positions[component_index] = (
                int(residue_number),
                str(self.residues.rows.insertion_code[first_position] or ""),
            )
            source_component_ids[component_index] = str(self.residues.rows.component_id[first_position])
        for component_index, (residue_number, source_insertion) in source_positions.items():
            component = entity.components[component_index]
            source_component_id = source_component_ids.get(component_index, component.component_id)
            standard_component = self.declarations.modified_residues.get(
                (chain_id, residue_number, source_insertion, source_component_id)
            )
            if standard_component is not None:
                self.builder.set_standard_component((instance_id, component_index), standard_component)
            for atom in component.atoms:
                site = AtomSite(instance_id, component_index, atom.atom_id)
                source_key = PdbSiteKey(
                    symmetry_id,
                    chain_id or None,
                    residue_number,
                    source_insertion or None,
                    source_component_id,
                    atom.atom_id,
                )
                self._add_alias(source_key, site)
                canonical_key = source_key._replace(component_id=component.component_id)
                if canonical_key != source_key:
                    self._add_alias(canonical_key, site)
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
        entity = Molecule(
            (first_component,), molecule_type="water" if is_water_component(first_component) else "non-polymer"
        )
        entity_id = f"molecule_{first_component.component_id}"
        instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
        indexed_residues = [(0, group_residues[0])]
        entity = self.builder.register_entity(entity_id, entity)
        instance_id = self.builder.name_instance(instance_seed)
        for atom in first_component.atoms:
            self._add_alias(
                PdbSiteKey(
                    symmetry_id,
                    chain_id or None,
                    int(residue_label),
                    self.residues.rows.insertion_code[first_position] or None,
                    first_component.component_id,
                    atom.atom_id,
                ),
                AtomSite(instance_id, 0, atom.atom_id),
            )
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

    def _bind_observed(self, item: Occurrence) -> None:
        instance_id, indexed = (item.instance_id, item.observed)
        for component_index, residue_key in indexed:
            residue_positions = self.residues.positions[residue_key]
            first_position = residue_positions[0]
            source_number = self.residues.rows.auth_residue_id[first_position]
            if source_number is None:
                source_number = self.residues.rows.residue_id[first_position]
            source_insertion = str(self.residues.rows.insertion_code[first_position] or "")
            if source_number is not None:
                self.builder.set_residue_number((instance_id, component_index), (str(source_number), source_insertion))
            self.builder.bind_atoms(
                (instance_id, component_index),
                self.residues.rows.row_index[residue_positions],
                self.residues.rows.atom_name[residue_positions],
            )

    def _classify_polymer(
        self,
        chain_id: str,
        base_instance_id: str,
        classified_component_ids: Sequence[str],
        segment_residues: Sequence[ResidueKey],
    ) -> PolymerType:
        classified_values: set[PolymerType | None] = set()
        classification: PolymerType | None
        for component_id in classified_component_ids:
            if component_id == "N":
                continue
            component = self.components.get(component_id)
            if component is None and component_id not in {"I", "DI", "DU"}:
                raise ValueError(
                    f"PDB SEQRES component {component_id!r} has no chemical definition; supply it through components="
                )
            if component_id == "I":
                classification = "rna"
            elif component_id in {"DI", "DU"}:
                classification = "dna"
            else:
                assert component is not None
                classification = _pdb_component_polymer_type(component)
            if classification is None:
                parent_types: set[PolymerType] = set()
                for (modified_chain, _, _, modified_id), parent_id in self.declarations.modified_residues.items():
                    if modified_chain != chain_id or modified_id != component_id:
                        continue
                    parent_type = get_builtin_component_polymer_type(parent_id)
                    if parent_type is not None:
                        parent_types.add(parent_type)
                if len(parent_types) > 1:
                    raise ValueError(
                        f"PDB MODRES declarations give component {component_id!r} conflicting polymer types"
                    )
                if parent_types:
                    classification = next(iter(parent_types))
            classified_values.add(classification)
        classified = {polymer_type for polymer_type in classified_values if polymer_type is not None}
        if "N" in classified_component_ids:
            if classified - {"dna", "rna"}:
                raise ValueError(f"PDB chain {base_instance_id!r} mixes nucleotide N with a non-nucleic polymer")
            if classified == {"dna", "rna"}:
                raise ValueError(
                    f"PDB chain {base_instance_id!r} uses N in a DNA/RNA hybrid without identifying its sugar"
                )
            if not classified:
                n_atom_ids = {
                    str(self.residues.rows.atom_name[position])
                    for key in segment_residues
                    for position in self.residues.positions[key]
                    if str(self.residues.rows.component_id[position]) == "N"
                }
                rna_identified = bool({"O2'", "HO2'"} & n_atom_ids)
                dna_identified = "H2''" in n_atom_ids
                if rna_identified == dna_identified:
                    raise ValueError(f"PDB chain {base_instance_id!r} contains only ambiguous N nucleotides")
                classified.add("rna" if rna_identified else "dna")
        if classified == {"dna", "rna"}:
            chain_polymer_type: PolymerType = "dna_rna_hybrid"
        elif len(classified) == 1:
            chain_polymer_type = next(iter(classified))
        else:
            raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
        return chain_polymer_type
