r"""Canonical molecular lowering for ODesign."""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.evidence import MSA, MSACollection
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    SystemBond,
)

# Hotspot, SymbolicBond, ODesignPlacements, and ReferenceCoordinates are the exact controls
# this Processor consumes. Hotspot and SymbolicBond reuse shared biological values from
# ``ofoldx.data``; the Processor validates their ODesign semantics inline. The
# generation-topology placement records (GeneratedSegment / ExistingSpan / SymbolicResidueSite
# / SymbolicAtomSite) and resolution helpers come from
# ``ofoldx.data.placement``.
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
    PlacementMap,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import instance_of as _placement_instance_of
from ...data.placement import resolve as _resolve_site
from ...data.structure import Structure
from ...io.structure import read_structure
from .config import ODesignProcessorConfig
from .model import (
    ODESIGN_RESTYPE_INDEX,
    ODESIGN_STANDARD_POLYMER_COMPONENTS,
    Hotspot,
    ODesignInput,
    ODesignLayout,
    ODesignPlacements,
    ReferenceCoordinates,
    build_odesign_cyclic_period,
    build_odesign_token_bonds,
    build_odesign_token_indices,
    build_odesign_tokens,
    materialize_odesign_cyclic_links,
)

# The released ODesign generation surface realizes a de-novo polymer GeneratedSegment as a run
# of placeholder components of the segment's polymer type (protein -> UNK, rna -> N, dna -> DN).
# There is NO honest de-novo *ligand* realization: the released path emits placeholder ``-L`` atom
# rows with fake bonds and binary NONE/ANY bond-type logits, not a real chemical graph. A generated
# ligand segment therefore has no entry here and is rejected at the realization boundary below:
# a de-novo molecule has no PlacementValue. Existing or fixed ligand context is unaffected.
_GENERATED_COMPONENT_ID = {"protein": "UNK", "rna": "N", "dna": "DN"}

# MSA cells are uppercase one-letter symbols or the ``-`` gap, not ChemicalComponent IDs.
# Standard symbols resolve to residue tokens, gaps to the gap token, and ambiguity symbols to ``UNK``.
_MSA_UNKNOWN = ODESIGN_RESTYPE_INDEX["UNK"]
_MSA_GAP = ODESIGN_RESTYPE_INDEX["-"]
_MSA_SYMBOL_TOKENS = {
    "protein": {
        symbol: ODESIGN_RESTYPE_INDEX[component_id]
        for symbol, component_id in {
            "A": "ALA",
            "R": "ARG",
            "N": "ASN",
            "D": "ASP",
            "C": "CYS",
            "Q": "GLN",
            "E": "GLU",
            "G": "GLY",
            "H": "HIS",
            "I": "ILE",
            "L": "LEU",
            "K": "LYS",
            "M": "MET",
            "F": "PHE",
            "P": "PRO",
            "S": "SER",
            "T": "THR",
            "W": "TRP",
            "Y": "TYR",
            "V": "VAL",
        }.items()
    },
    "rna": {symbol: ODESIGN_RESTYPE_INDEX[symbol] for symbol in ("A", "G", "C", "U", "N")},
    "dna": {symbol: ODESIGN_RESTYPE_INDEX["D" + symbol] for symbol in ("A", "G", "C", "T", "N")},
}


def _msa_symbol_token(symbol: str, polymer_type: str) -> int:
    r"""Lower one uppercase MSA symbol (or ``-`` gap) to an ODesign residue token."""

    if symbol == "-":
        return _MSA_GAP
    return _MSA_SYMBOL_TOKENS[polymer_type].get(symbol, _MSA_UNKNOWN)


class ODesignProcessor:
    r"""Lower canonical systems, evidence, and supported Conditions to ODesignInput."""

    def __init__(
        self,
        config: ODesignProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = ODesignProcessorConfig() if config is None else config
        self.config = config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update runtime tensor placement and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ReferenceCoordinates] = (),
        receptor_mode: str | None = None,
    ) -> ODesignInput:
        r"""Validate and lower one unbatched canonical ODesign example."""

        condition_values = tuple(conditions)
        for condition in condition_values:
            if not isinstance(condition, (ReferenceCoordinates, Hotspot, SymbolicBond)):
                raise TypeError(f"ODesign cannot consume {type(condition).__name__}")

        normalized = _normalize_input(input)
        # ODesign validates condition semantics inline: type and intent above, endpoints and
        # referenced atoms below.
        structure = normalized if isinstance(normalized, Structure) else None
        source_system = structure.system if structure is not None else normalized
        # A GeneratedSegment carries no embedded id; its identity is the
        # globally-unique ``placement_id`` inner key of the PlacementSpec mapping.
        segments = {
            placement_id: part
            for chain in (() if placement is None else placement.placements.values())
            for placement_id, part in chain.items()
            if isinstance(part, GeneratedSegment)
        }
        for condition in condition_values:
            if not isinstance(condition, Hotspot):
                continue
            for reference in condition.residues:
                if isinstance(reference, (AtomSite, SymbolicAtomSite)):
                    continue
                if isinstance(reference, str):
                    # A whole-placement reference is the bare globally-unique placement_id.
                    polymer_type = segments[reference].polymer_type
                elif isinstance(reference, SymbolicResidueSite):
                    polymer_type = segments[reference.placement_id].polymer_type
                else:
                    if source_system is None:
                        raise ValueError("existing Hotspot residues require molecular input")
                    reference_entity = source_system.entities[source_system.instances[reference[0]]]
                    if not isinstance(reference_entity, PolymerSequence):
                        raise ValueError("non-protein Hotspot residues require exact atom references")
                    polymer_type = reference_entity.polymer_type
                if polymer_type != "protein":
                    raise ValueError("non-protein Hotspot residues require exact atom references")
        if placement is not None:
            (
                system,
                sites,
                placement_map,
                instance_of,
                source_residue_map,
                source_atom_map,
            ) = _materialize_placement(placement, source_system)
        else:
            if source_system is None:
                raise ValueError("input=None requires one PlacementSpec that determines generated topology")
            system = source_system
            sites = _sites_with_reference_coordinates(system)
            placement_map = {}
            instance_of = {}
            source_residue_map = {
                (instance_id, component_index): (instance_id, component_index)
                for instance_id, entity_id in system.instances.items()
                if isinstance(entity := system.entities[entity_id], PolymerSequence)
                for component_index in range(len(entity))
            }
            source_atom_map = {site: site for site in sites}

        system, cyclic_instances, cyclic_links = materialize_odesign_cyclic_links(
            system,
            condition_values,
            placement_map,
            instance_of,
            source_atom_map,
        )

        unsupported_components = {
            component.component_id
            for entity in system.entities.values()
            if isinstance(entity, PolymerSequence)
            for component in entity.components
            if component.component_id not in ODESIGN_STANDARD_POLYMER_COMPONENTS
        }
        if unsupported_components:
            raise ValueError(
                "released ODesign has no residue token for components " f"{sorted(unsupported_components)!r}"
            )
        msa_value = _normalize_msa(msa, system)

        if not sites:
            raise ValueError("ODesign requires at least one heavy atom")
        generated_residues = {
            occurrence
            for placements in placement_map.values()
            for occurrences in placements.values()
            for occurrence in occurrences
        }
        token_sites, token_residues, token_entities = build_odesign_tokens(system, sites)
        site_to_token = {site: token_index for token_index, group in enumerate(token_sites) for site in group}
        site_to_atom = {site: index for index, site in enumerate(sites)}
        observed_positions = torch.zeros(len(sites), 3, dtype=torch.float32)
        observed_mask = torch.zeros(len(sites), dtype=torch.bool)
        if structure is not None:
            observed = {site: index for index, site in enumerate(structure.sites)}
            source_by_output = {output: source for source, output in source_atom_map.items()}
            for atom_index, output_site in enumerate(sites):
                source_site = source_by_output.get(output_site)
                if source_site is None:
                    continue
                source_index = observed.get(source_site)
                if source_index is None or not bool(structure.coordinate_mask[source_index]):
                    continue
                observed_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())
                observed_mask[atom_index] = True

        reference_positions = torch.empty_like(observed_positions)
        reference_space_id = []
        reference_spaces: dict[tuple[str, int], int] = {}
        for atom_index, site in enumerate(sites):
            entity_id = system.instances[site.instance_id]
            entity = system.entities[entity_id]
            component = system.component((site.instance_id, site.component_index))
            if isinstance(entity, PolymerSequence):
                if (
                    site.instance_id,
                    site.component_index,
                ) in generated_residues and entity.polymer_type == "protein":
                    reference_component = get_component("GLY")
                    owner = "the generated-protein GLY template"
                else:
                    reference_component = component
                    owner = f"component {component.component_id!r}"
            else:
                reference_component = component
                owner = f"molecule {entity_id!r}"
            coordinates = reference_component.reference_coordinates
            atoms = reference_component.atoms
            if not coordinates:
                raise ValueError(f"ODesign requires reference coordinates for {owner}")
            coordinate = coordinates[next(index for index, atom in enumerate(atoms) if atom.atom_id == site.atom_id)]
            if coordinate is None:
                raise ValueError(f"ODesign requires a reference coordinate for {owner} atom {site.atom_id!r}")
            reference_positions[atom_index] = torch.tensor(coordinate)

            reference_space = (site.instance_id, site.component_index)
            if reference_space not in reference_spaces:
                reference_spaces[reference_space] = len(reference_spaces)
            reference_space_id.append(reference_spaces[reference_space])

        reference_space_id = torch.tensor(reference_space_id, dtype=torch.int64)
        for space_id in range(len(reference_spaces)):
            space_mask = reference_space_id == space_id
            reference_positions[space_mask] -= reference_positions[space_mask].mean(dim=0)
        reference_mask = torch.ones(len(sites), dtype=torch.bool)

        hotspot = _hotspot_tokens(
            condition_values,
            placement_map,
            instance_of,
            source_residue_map,
            source_atom_map,
            site_to_token,
        )
        fixed_coordinate_mask = torch.zeros(len(sites), dtype=torch.bool)
        constraint_coordinate_mask = torch.zeros(len(sites), dtype=torch.bool)
        output_atom_by_source = source_atom_map
        for condition in condition_values:
            if not isinstance(condition, ReferenceCoordinates):
                continue
            for site in condition.sites:
                condition_site = output_atom_by_source.get(site)
                if condition_site is None or condition_site not in site_to_atom:
                    raise ValueError(f"ODesign atom layout does not contain reference site {site!r}")
                atom_index = site_to_atom[condition_site]
                constraint_coordinate_mask[atom_index] = True
                if condition.intent == "hard":
                    fixed_coordinate_mask[atom_index] = True
                elif condition.intent != "soft":
                    raise ValueError(f"unknown ReferenceCoordinates intent: {condition.intent!r}")
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("ReferenceCoordinates must refer to observed input coordinates")
        # ``receptor_mode="rigid"`` realizes the ``diffusion`` condition by holding
        # every existing (non-generated) receptor heavy atom rigid — it is marked as a hard fixed
        # coordinate that the EDM sampler pins (model.sample reads ``fixed_coordinate_mask``).
        # This is additive over the per-atom ReferenceCoordinates route above and only fixes atoms
        # that are actually observed (a receptor without coordinates cannot be pinned). "flex"
        # regenerates the receptor and leaves this untouched, so only explicit hard
        # ReferenceCoordinates pin atoms.
        mode = self.config.receptor_mode if receptor_mode is None else receptor_mode
        if mode == "rigid":
            for atom_index, site in enumerate(sites):
                is_generated = (site.instance_id, site.component_index) in generated_residues
                if not is_generated and bool(observed_mask[atom_index]):
                    fixed_coordinate_mask[atom_index] = True
        elif mode != "flex":
            raise ValueError(f"unknown receptor_mode: {mode!r}")

        msa_features, msa_mask, profile, deletion_mean = _msa_features(
            msa_value,
            system,
            token_residues,
            token_entities,
            generated_residues,
        )
        token_features = torch.zeros(len(token_sites), 453, dtype=torch.float32)
        for token_index, (residue, entity) in enumerate(zip(token_residues, token_entities)):
            if residue is None:
                restype = "-L"
            elif residue in generated_residues:
                restype = "-P" if cast(PolymerSequence, entity).polymer_type == "protein" else "-N"
            else:
                sequence = cast(PolymerSequence, system.entities[system.instances[residue[0]]])
                restype = sequence.components[residue[1]].component_id
            restype_index = ODESIGN_RESTYPE_INDEX.get(restype, 20)
            token_features[token_index, 384 + restype_index] = 1.0
            token_features[token_index, 419 : 419 + 32] = profile[token_index]
            token_features[token_index, 451] = deletion_mean[token_index]
            token_features[token_index, 452] = float(hotspot[token_index])

        atom_to_token = torch.tensor(
            [site_to_token[site] for site in sites],
            dtype=torch.int64,
        )
        atom_features = _atom_features(system, sites, reference_mask)
        token_bonds = build_odesign_token_bonds(
            system,
            token_sites,
            site_to_token,
            excluded=cyclic_links,
        )
        constraint_distogram = _constraint_distogram(
            system,
            sites,
            site_to_token,
            observed_positions,
            constraint_coordinate_mask,
        )
        chain_index, entity_index, symmetry_index, residue_index = build_odesign_token_indices(
            system,
            token_sites,
            token_residues,
            token_entities,
        )
        return ODesignInput(
            token_mask=torch.ones(1, len(token_sites), dtype=torch.bool, device=self.device),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(len(token_sites), dtype=torch.int64, device=self.device).unsqueeze(0),
            cyclic_period=build_odesign_cyclic_period(token_sites, cyclic_instances).unsqueeze(0).to(self.device),
            token_features=token_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            token_bonds=token_bonds.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_features=atom_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(1, len(sites), dtype=torch.bool, device=self.device),
            reference_positions=reference_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=torch.tensor(
                [system.atom(site).formal_charge for site in sites],
                dtype=self.dtype,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(-1),
            observed_positions=observed_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            observed_mask=observed_mask.unsqueeze(0).to(self.device),
            fixed_coordinate_mask=fixed_coordinate_mask.unsqueeze(0).to(self.device),
            constraint_distogram=constraint_distogram.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_features=msa_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            layout=ODesignLayout(
                system=system,
                sites=sites,
                placement=placement,
                conditions=condition_values,
                placement_map=placement_map,
                instance_of=instance_of,
                source_residue_map=source_residue_map,
                source_atom_map=source_atom_map,
                msa=msa_value,
            ),
        )


def _normalize_input(input):
    if input is None or isinstance(input, (MolecularSystem, Structure)):
        return input
    if isinstance(input, str | PathLike):
        return read_structure(Path(input).expanduser())
    raise TypeError("ODesign input must be a MolecularSystem, Structure, local path, or None")


def _sites_with_reference_coordinates(system: MolecularSystem) -> tuple[AtomSite, ...]:
    r"""Select released model atoms; canonical-only terminal atoms stay unresolved in Results."""

    sites = []
    for site, atom in system.iter_atoms():
        if atom.element == "H":
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence):
            component = entity.components[site.component_index]
            atom_index = next(index for index, member in enumerate(component.atoms) if member.atom_id == site.atom_id)
            if component.reference_coordinates and component.reference_coordinates[atom_index] is None:
                continue
        sites.append(site)
    return tuple(sites)


def _materialize_placement(
    placement: ODesignPlacements,
    source_system: MolecularSystem | None,
) -> tuple[
    MolecularSystem,
    tuple[AtomSite, ...],
    PlacementMap,
    dict[str, str],
    dict[tuple[str, int], tuple[str, int]],
    dict[AtomSite, AtomSite],
]:
    r"""Materialize one exact mixed PlacementSpec and its source-to-output identity maps.

    Returns the realized ``placement_map`` (the output surface for generated
    placements: ``declaration_group -> placement_id -> ordered ComponentReference tuple``)
    and its ``instance_of`` index (``placement_id -> declaration_group``, derived once from the
    PlacementSpec), alongside the source-residue and source-atom maps.
    """

    molecule_entities: dict[str, Molecule] = (
        {entity_id: entity for entity_id, entity in source_system.entities.items() if isinstance(entity, Molecule)}
        if source_system is not None
        else {}
    )
    molecule_instances: dict[str, str] = (
        {
            instance_id: entity_id
            for instance_id, entity_id in source_system.instances.items()
            if isinstance(source_system.entities[entity_id], Molecule)
        }
        if source_system is not None
        else {}
    )
    used_entity_ids = set(molecule_entities)
    used_instance_ids = set(molecule_instances)
    polymer_entities: dict[str, PolymerSequence] = {}
    polymer_instances: dict[str, str] = {}
    placement_map: PlacementMap = {}
    source_residue_map: dict[tuple[str, int], tuple[str, int]] = {}
    generated_residues: set[tuple[str, int]] = set()
    generated_component_ids: set[str] = set()
    new_entities_by_sequence: dict[tuple[PolymerType, tuple[ChemicalComponent, ...]], tuple[str, PolymerSequence]] = {}

    # The shared declaration-order allocator mints the realized instance IDs:
    # canonical ``chain_{n}`` (uniform and 0-based) once per realized output
    # chain that has >=1 occurrence. A declaration group whose parts all realize to zero length
    # mints NO chain and consumes NO index — it is recorded as () and skipped.
    instance_ids = InstanceIdAllocator()
    for chain_index, (group, chain) in enumerate(placement.placements.items()):
        if all(
            part.max_length == 0 if isinstance(part, GeneratedSegment) else part.stop == part.start
            for part in chain.values()
        ):
            for placement_id, part in chain.items():
                if isinstance(part, GeneratedSegment):
                    placement_map.setdefault(group, {})[placement_id] = ()
            continue
        instance_id = instance_ids.allocate()
        while instance_id in used_instance_ids:
            instance_id += "_"
        used_instance_ids.add(instance_id)

        components: list[ChemicalComponent] = []
        segment_offsets: list[tuple[str, GeneratedSegment, int, int]] = []
        polymer_type: str | None = None
        for placement_id, part in chain.items():
            if isinstance(part, GeneratedSegment):
                part_polymer_type = part.polymer_type
                start = len(components)
                length = part.max_length
                if part_polymer_type not in _GENERATED_COMPONENT_ID:
                    raise NotImplementedError(
                        "ODesign cannot realize one dna_rna_hybrid segment without per-residue polymer "
                        "identity; use separate DNA and RNA generated segments"
                    )
                component_id = _GENERATED_COMPONENT_ID[part_polymer_type]
                components.extend([get_component(component_id)] * length)
                generated_component_ids.add(component_id)
                segment_offsets.append((placement_id, part, start, length))
            else:
                part = cast(ExistingSpan, part)
                source_system = cast(MolecularSystem, source_system)
                source_entity = cast(PolymerSequence, source_system.entities[source_system.instances[part.instance_id]])
                part_polymer_type = source_entity.polymer_type
                for source_index in range(part.start, part.stop):
                    output_index = len(components)
                    components.append(source_entity.components[source_index])
                    source_residue_map[(part.instance_id, source_index)] = (instance_id, output_index)

            if polymer_type is not None and polymer_type != part_polymer_type:
                raise ValueError("one ODesign output chain cannot mix polymer types")
            polymer_type = part_polymer_type

        polymer_type = cast(PolymerType, polymer_type)

        reused_entity_id: str | None = None
        chain_parts = tuple(chain.values())
        if len(chain_parts) == 1 and isinstance(chain_parts[0], ExistingSpan) and source_system is not None:
            span = chain_parts[0]
            source_entity_id = source_system.instances[span.instance_id]
            candidate = cast(PolymerSequence, source_system.entities[source_entity_id])
            if span.start == 0 and span.stop == len(candidate):
                reused_entity_id = source_entity_id
                sequence = candidate
        if reused_entity_id is not None:
            entity_id = reused_entity_id
        else:
            signature = (polymer_type, tuple(components))
            existing_entity = new_entities_by_sequence.get(signature)
            if existing_entity is None:
                entity_id = f"odesign_entity_{chain_index + 1}"
                while entity_id in used_entity_ids:
                    entity_id += "_"
                sequence = PolymerSequence(polymer_type, tuple(components))
                new_entities_by_sequence[signature] = (entity_id, sequence)
            else:
                entity_id, sequence = existing_entity
        if entity_id not in used_entity_ids:
            polymer_entities[entity_id] = sequence
            used_entity_ids.add(entity_id)
        polymer_instances[instance_id] = entity_id

        for placement_id, _segment, start, length in segment_offsets:
            occurrences = tuple((instance_id, start + offset) for offset in range(length))
            generated_residues.update(occurrences)
            placement_map.setdefault(group, {})[placement_id] = occurrences

    source_components = (
        {
            component.component_id: component
            for entity in source_system.entities.values()
            for component in entity.components
        }
        if source_system is not None
        else {}
    )
    for component_id in generated_component_ids:
        source_component = source_components.get(component_id)
        if source_component is not None and source_component != get_component(component_id):
            raise ValueError(f"generated {component_id} conflicts with the input system's component graph")

    preliminary = MolecularSystem(
        {**polymer_entities, **molecule_entities},
        {**polymer_instances, **molecule_instances},
    )
    complete_sites = _sites_with_reference_coordinates(preliminary)
    source_residue_by_output = {output: source for source, output in source_residue_map.items()}
    source_heavy_sites = (
        [site for site, atom in source_system.iter_atoms() if atom.element != "H"] if source_system is not None else []
    )
    source_sites = set(source_heavy_sites)
    sites = []
    for site in complete_sites:
        residue = (site.instance_id, site.component_index)
        entity = preliminary.entities[preliminary.instances[site.instance_id]]
        if residue in generated_residues:
            if cast(PolymerSequence, entity).polymer_type == "protein" and site.atom_id not in {"N", "CA", "C", "O"}:
                continue
        elif isinstance(entity, PolymerSequence):
            source_residue = source_residue_by_output[residue]
            if AtomSite(source_residue[0], source_residue[1], site.atom_id) not in source_sites:
                continue
        sites.append(site)
    site_tuple = tuple(sites)
    output_sites = set(site_tuple)
    source_atom_map: dict[AtomSite, AtomSite] = {}
    if source_system is not None:
        source_sites_by_residue: dict[tuple[str, int], list[AtomSite]] = {}
        for source_site in source_heavy_sites:
            source_sites_by_residue.setdefault((source_site.instance_id, source_site.component_index), []).append(
                source_site
            )
        for source_residue, output_residue in source_residue_map.items():
            for source_site in source_sites_by_residue.get(source_residue, ()):
                output_site = AtomSite(output_residue[0], output_residue[1], source_site.atom_id)
                if output_site in output_sites:
                    source_atom_map[source_site] = output_site
        for source_site in source_heavy_sites:
            if source_site.instance_id in molecule_instances and source_site in output_sites:
                source_atom_map[source_site] = source_site

    mapped_atoms = source_atom_map
    mapped_bonds = []
    if source_system is not None:
        for bond in source_system.bonds:
            site_a = mapped_atoms.get(bond.site_a)
            site_b = mapped_atoms.get(bond.site_b)
            if (site_a is None) != (site_b is None):
                raise ValueError(f"ODesign placement cuts exactly one endpoint of source bond {bond!r}")
            if site_a is None or site_b is None:
                continue
            stereo_sites = (
                None
                if bond.stereo_sites is None
                else (mapped_atoms[bond.stereo_sites[0]], mapped_atoms[bond.stereo_sites[1]])
            )
            mapped_bonds.append(
                SystemBond(
                    site_a,
                    site_b,
                    order=bond.order,
                    aromatic=bond.aromatic,
                    stereo=bond.stereo,
                    stereo_sites=stereo_sites,
                    origin=bond.origin,
                )
            )
    system = MolecularSystem(
        preliminary.entities,
        preliminary.instances,
        frozenset(mapped_bonds),
    )
    return (
        system,
        site_tuple,
        placement_map,
        _placement_instance_of(placement.placements),
        source_residue_map,
        source_atom_map,
    )


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return msa
    if isinstance(msa, MSA | MSACollection):
        values = (msa,) if isinstance(msa, MSA) else msa.msas
        # Evidence holds no PolymerSequence: each column names one entity by ``entity_id`` and closes
        # over this system, so the Processor resolves the entity and checks the query width (the
        # MSA validates neither against an entity). Column ``i`` binds to ``entity.components[i]``.
        for value in values:
            entity = system.entities.get(value.entity_id)
            if not isinstance(entity, PolymerSequence):
                raise ValueError("MSA entity disagrees with ODesign input")
            if len(value.unpaired_sequences[0]) != len(entity):
                raise ValueError(
                    f"MSA for entity {value.entity_id!r} spans {len(value.unpaired_sequences[0])} columns "
                    f"but the entity has {len(entity)} components"
                )
        return msa
    path = Path(msa).expanduser()
    polymers = [
        (entity_id, entity) for entity_id, entity in system.entities.items() if isinstance(entity, PolymerSequence)
    ]
    if len(polymers) != 1:
        raise ValueError("a FASTA MSA path is unambiguous only for a single-polymer system; pass MSACollection")
    from multimolecule.io import read_fasta_records

    entity_id, entity = polymers[0]
    # The released path reads a plain aligned FASTA (row 0 is the ungapped query) and carries no
    # insertion state, so every row becomes a query-width one-letter symbol string and both pools'
    # deletion counts stay zero. Column ``i`` binds to ``entity.components[i]`` positionally.
    rows = tuple(_aligned_symbols(record.sequence) for record in read_fasta_records(path))
    width = len(rows[0]) if rows else len(entity)
    if width != len(entity):
        raise ValueError("MSA query width disagrees with the ODesign entity length")
    return MSA(
        entity_id=entity_id,
        unpaired_sequences=rows,
        unpaired_deletion_counts=np.zeros((len(rows), width), dtype=np.uint32),
        paired_sequences=(rows[0],),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )


def _aligned_symbols(sequence: str) -> str:
    r"""Project one aligned FASTA row to an uppercase one-letter symbol string (``.`` folds to ``-``)."""

    return "".join("-" if symbol in "-." else symbol for symbol in sequence.upper())


def _atom_features(system: MolecularSystem, sites: tuple[AtomSite, ...], reference_mask: Tensor) -> Tensor:
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    element = torch.zeros(len(sites), 129)
    for index, site in enumerate(sites):
        atom = system.atom(site)
        element[index, int(periodic_table.GetAtomicNumber(atom.element)) - 1] = 1.0
    characters = torch.zeros(len(sites), 4, 64)
    for atom_index, site in enumerate(sites):
        name = site.atom_id
        if len(name) > 4:
            raise ValueError(f"ODesign cannot encode atom name longer than four characters: {name!r}")
        for char_index, character in enumerate(name.ljust(4)):
            value = ord(character) - 32
            if not 0 <= value < 64:
                raise ValueError(f"ODesign cannot encode atom-name character {character!r}")
            characters[atom_index, char_index, value] = 1.0
    return torch.cat([reference_mask.float().unsqueeze(-1), element, characters.flatten(-2)], dim=-1)


def _hotspot_tokens(
    conditions,
    placement_map,
    instance_of,
    source_residue_map,
    source_atom_map,
    site_to_token,
):
    hotspot = torch.zeros(max(site_to_token.values()) + 1, dtype=torch.bool)
    for condition in conditions:
        if not isinstance(condition, Hotspot):
            continue
        selected = False
        for reference in condition.residues:
            # Skip generated endpoints: a whole-placement ``placement_id`` (str), one symbolic
            # residue, or one symbolic atom — hotspots mark EXISTING input residues only.
            if isinstance(reference, (str, SymbolicResidueSite, SymbolicAtomSite)):
                continue
            for site in _reference_sites(
                reference,
                placement_map,
                instance_of,
                source_residue_map,
                source_atom_map,
                site_to_token,
            ):
                hotspot[site_to_token[site]] = True
                selected = True
        if not selected:
            raise ValueError("ODesign Hotspot must select at least one input residue")
    return hotspot


def _constraint_distogram(
    system,
    sites,
    site_to_token,
    observed_positions,
    constraint_coordinate_mask,
):
    size = max(site_to_token.values()) + 1
    values = observed_positions.new_zeros(size, size, 1)
    atom_by_token = {}
    for atom_index, site in enumerate(sites):
        if not bool(constraint_coordinate_mask[atom_index]):
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence):
            component_id = entity.components[site.component_index].component_id
            if component_id in ODESIGN_STANDARD_POLYMER_COMPONENTS:
                center_atom = "CA" if entity.polymer_type == "protein" else "C1'"
                if site.atom_id != center_atom:
                    continue
        atom_by_token[site_to_token[site]] = atom_index
    for left_index, left_atom in atom_by_token.items():
        for right_index, right_atom in atom_by_token.items():
            if left_index != right_index and values[left_index, right_index, 0] == 0:
                distance = torch.linalg.vector_norm(observed_positions[left_atom] - observed_positions[right_atom])
                values[left_index, right_index, 0] = distance
    return values


def _reference_sites(
    reference,
    placement_map,
    instance_of,
    source_residue_map,
    source_atom_map,
    available_sites,
):
    if isinstance(reference, AtomSite):
        return (source_atom_map[reference],)
    if isinstance(reference, SymbolicAtomSite):
        residue = _resolve_site(reference.residue, placement_map, instance_of)
        return (AtomSite(residue[0], residue[1], reference.atom_id),)
    if isinstance(reference, str):
        # A bare globally-unique placement_id names the whole generated placement, in order.
        residues = placement_map[instance_of[reference]][reference]
    elif isinstance(reference, SymbolicResidueSite):
        residues = (_resolve_site(reference, placement_map, instance_of),)
    else:
        residues = (source_residue_map[reference],)
    sites = []
    for residue in residues:
        beta = AtomSite(residue[0], residue[1], "CB")
        sites.append(beta if beta in available_sites else AtomSite(residue[0], residue[1], "CA"))
    return tuple(sites)


def _msa_features(msa, system, token_residues, token_entities, generated_residues):
    token_count = len(token_residues)
    profile = torch.zeros(token_count, 32)
    deletion_mean = torch.zeros(token_count)
    if msa is None:
        row = torch.full((token_count,), 31, dtype=torch.int64)
        for index, residue in enumerate(token_residues):
            component = (
                "-"
                if residue is None or residue in generated_residues
                else token_entities[index].components[residue[1]].component_id
            )
            token = min(ODESIGN_RESTYPE_INDEX.get(component, 20), 31)
            profile[index, token] = 1.0
            row[index] = token
        features = torch.cat(
            [F.one_hot(row, 32).float(), torch.zeros(token_count, 2)],
            dim=-1,
        ).unsqueeze(0)
        return features, torch.ones(1, token_count, dtype=torch.bool), profile, deletion_mean
    msas = (msa,) if isinstance(msa, MSA) else msa.msas
    # The MSA holds no PolymerSequence: column ``c`` binds ``msas[c].entity_id`` to a system entity and
    # ``query_to_entity`` is the identity map, so a token's component index is the aligned-row column.
    # ``polymer_types[c]`` resolves the one-letter-symbol lowering; ``tokens_by_entity`` groups every
    # token (across all instances of an entity, matching the old value-equality broadcast) by entity_id.
    polymer_types = [cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in msas]
    tokens_by_entity: dict[str, list[tuple[int, int]]] = {}
    for token_index, residue in enumerate(token_residues):
        if residue is None:
            continue
        tokens_by_entity.setdefault(system.instances[residue[0]], []).append((token_index, residue[1]))
    # A bare MSA replays its unpaired pool (row 0 is the query); a collection replays its dense
    # row_pairing/row_is_paired plan, selecting the paired or unpaired pool per column (``-1`` = gap).
    if isinstance(msa, MSA):
        plan: list[tuple[tuple[int, int, bool], ...]] = [
            ((0, source_row, False),) for source_row in range(len(msa.unpaired_sequences))
        ]
    else:
        row_pairing = msa.row_pairing
        row_is_paired = msa.row_is_paired
        plan = [
            tuple(
                (column, int(row_pairing[plan_index, column]), bool(row_is_paired[plan_index, column]))
                for column in range(len(msas))
                if row_pairing[plan_index, column] != -1
            )
            for plan_index in range(row_pairing.shape[0])
        ]
    rows = []
    deletions = []
    for selections in plan:
        token_row = torch.full((token_count,), 31, dtype=torch.int64)
        deletion_row = torch.zeros(token_count)
        for column_index, source_row, is_paired in selections:
            value = msas[column_index]
            polymer_type = polymer_types[column_index]
            aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
            aligned_deletions = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                source_row
            ]
            for token_index, column in tokens_by_entity.get(value.entity_id, ()):
                token_row[token_index] = _msa_symbol_token(aligned[column], polymer_type)
                deletion_row[token_index] = float(aligned_deletions[column])
        rows.append(token_row)
        deletions.append(deletion_row)
    tokens = torch.stack(rows)
    deletion_counts = torch.stack(deletions)
    one_hot = F.one_hot(tokens, 32).float()
    profile = one_hot.mean(dim=0)
    deletion_mean = deletion_counts.mean(dim=0)
    has_deletion = (deletion_counts > 0).float()
    deletion_value = (2 / math.pi) * torch.atan(deletion_counts / 3)
    features = torch.cat([one_hot, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
    mask = torch.ones(tokens.shape, dtype=torch.bool)
    return features, mask, profile, deletion_mean
