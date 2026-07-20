r"""Canonical target-and-binder lowering for the released PXDesign-d model."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from typing_extensions import Self

# The placement core is frozen (DESIGN §3.7 "placement core", ratified
# 2026-07-18). PXDesign imports the shared records — the nested ``PlacementSpec`` mapping,
# ``GeneratedSegment`` / ``ExistingSpan`` values, the two-field ``SymbolicResidueSite``
# reference, and the ``validate_placement`` global-uniqueness check — from ``ofoldx.data``.
# ``Hotspot`` is the frozen §3.7.4 distance-free per-residue target label PXDesign consumes.
from ...data import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
    PlacementSpec,
    SymbolicResidueSite,
    validate_placement,
)
from ...data.components import get_component
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import AtomSite, ChemicalComponent, MolecularSystem, Molecule, PolymerSequence, SystemBond
from ...data.structure import Structure
from ...io.structure import read_structure
from .config import PxDesignProcessorConfig
from .model import Hotspot, PxDesignInput, PxDesignLayout

_RESTYPE_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
}
_GENERATED_RESTYPE = 32
_GENERATED_ATOM_IDS = frozenset(atom.atom_id for atom in get_component("GLY").atoms if atom.element != "H")


class PxDesignProcessor:
    r"""Lower one released target-conditioned binder-design problem to tensors."""

    config_class = PxDesignProcessorConfig

    def __init__(
        self,
        config: PxDesignProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = PxDesignProcessorConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update runtime tensor placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        # Frozen §3.7 placement core (``PlacementSpec``) plus the frozen §3.7.4 ``Hotspot``
        # target-conditioning label; PXDesign consumes both directly from ``ofoldx.data``.
        conditions: Sequence[PlacementSpec | Hotspot] = (),
    ) -> PxDesignInput:
        r"""Encode the released single, fixed-length protein-binder design contract."""

        if msa is not None:
            raise ValueError("PXDesign-d does not consume MSA evidence; MSA belongs to the downstream filter")
        if tuple(templates):
            raise ValueError("PXDesign-d does not consume Template evidence")
        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if input is not None and type(input) not in (MolecularSystem, Structure):
            raise TypeError("input must be a MolecularSystem, Structure, local path, or None")

        condition_values = tuple(conditions)
        # A frozen PlacementSpec is the nested ``group -> placement_id -> value`` mapping (a plain
        # dict); a frozen ``Hotspot`` is the distance-free per-residue label. Discriminate
        # structurally. The released hotspot channel is unweighted, soft, and distance-free, so
        # the frozen ``Hotspot`` (no ``weight`` / ``intent`` / ``cutoff`` fields) makes the former
        # weighted/hard/cutoff rejections structurally impossible rather than runtime checks.
        if any(not isinstance(condition, (dict, Hotspot)) for condition in condition_values):
            unsupported = next(
                condition for condition in condition_values if not isinstance(condition, (dict, Hotspot))
            )
            raise TypeError(f"PXDesign cannot consume {type(unsupported).__name__}")
        placements = tuple(condition for condition in condition_values if isinstance(condition, dict))
        if len(placements) != 1:
            raise ValueError("PXDesign requires exactly one PlacementSpec containing its generated binder")
        # The remaining family-owned semantic checks live inline in ``_hotspots`` (§3.7.1).

        structure = input if type(input) is Structure else None
        source = input.system if structure is not None else input
        # Released PXDesign applies the AlphaFold3-SI-§2.1 modified-polymer transform BEFORE
        # tokenization/featurization: MSE (selenomethionine) is canonicalized to MET. Do it
        # here, on the input, so every downstream check and feature sees the standard residue.
        source, structure = _canonicalize_modified_polymers(source, structure)
        placement = placements[0]
        # Enforce the frozen construction-time invariant: placement_id globally unique.
        validate_placement(placement)
        # Each outer key is a declaration group (a chain); each inner key is a globally
        # unique placement_id. Identity rides the inner key, not a value field: the binder
        # placement is KEYED by ``generated_id``, while its realized instance id is minted by the
        # shared allocator (``chain_0``, below) per the uniform realized-id convention.
        generated_items = [
            (group, placement_id, part)
            for group, placements in placement.items()
            for placement_id, part in placements.items()
            if type(part) is GeneratedSegment
        ]
        if len(generated_items) != 1:
            raise ValueError("released PXDesign generates exactly one binder chain")
        generated_group, generated_id, generated = generated_items[0]
        if generated.polymer_type != "protein":
            raise ValueError("released PXDesign generates protein binders only")
        if generated.min_length != generated.max_length:
            raise ValueError("released PXDesign requires one fixed binder length")
        if tuple(placement[generated_group].values()) != (generated,):
            raise ValueError("the PXDesign binder must be a separate generated chain")
        if next(reversed(placement)) != generated_group:
            raise ValueError("released PXDesign requires the generated binder after its target chains")
        existing_chains = tuple(
            tuple(placements.values()) for group, placements in placement.items() if group != generated_group
        )
        if any(not all(type(part) is ExistingSpan for part in chain) for chain in existing_chains):
            raise ValueError("every non-binder PXDesign PlacementSpec chain must contain only ExistingSpan values")
        if source is None and existing_chains:
            raise ValueError("ExistingSpan target chains require molecular input")
        if structure is None and source is not None:
            raise ValueError("PXDesign target conditioning requires observed Structure coordinates")
        if source is not None:
            if not existing_chains:
                raise ValueError("PXDesign target input must be declared by its PlacementSpec")
            declared_instances = set()
            for chain in existing_chains:
                if len(chain) != 1:
                    raise ValueError("PXDesign target PlacementSpec chains currently require one complete ExistingSpan")
                span = chain[0]
                entity = cast(PolymerSequence, source.entities[source.instances[span.instance_id]])
                if span.start != 0 or span.stop != len(entity):
                    raise ValueError(
                        "PXDesign target cropping requires canonical chain materialization before lowering"
                    )
                if span.instance_id in declared_instances:
                    raise ValueError("a PXDesign target instance may appear only once in PlacementSpec")
                declared_instances.add(span.instance_id)

        system, sites, generated_map, selected_residues = _materialize_problem(
            source,
            structure,
            existing_chains,
            generated,
            generated_id,
        )
        modeled_residues = {
            (site.instance_id, site.component_index)
            for site in sites
            if type(system.entities[system.instances[site.instance_id]]) is not Molecule
        }
        unsupported_components = set()
        for instance_id, component_index in modeled_residues:
            entity = cast(PolymerSequence, system.entities[system.instances[instance_id]])
            component_id = entity.components[component_index].component_id
            if component_id not in _RESTYPE_INDEX:
                unsupported_components.add(component_id)
        if unsupported_components:
            raise ValueError(
                "PXDesign modified-polymer lowering canonicalizes MSE to MET (AlphaFold3 SI §2.1) but "
                "otherwise requires the CCD canonical-parent mapping the package component value does "
                f"not carry; unsupported components: {sorted(unsupported_components)!r}"
            )
        token_sites, token_residues, token_components, representative_sites = _tokens(
            system,
            sites,
            frozenset(generated_map.values()),
        )
        site_to_token = {site: token_index for token_index, group in enumerate(token_sites) for site in group}
        hotspot_residues = _hotspots(
            condition_values,
            selected_residues,
            token_residues,
        )

        reference_positions, reference_mask, reference_space_id = _reference_features(
            system,
            sites,
            frozenset(generated_map.values()),
        )
        atom_features, reference_charges = _atom_features(system, sites, reference_mask)
        atom_to_token = torch.tensor([site_to_token[site] for site in sites], dtype=torch.int64)
        token_count = len(token_sites)
        restype = torch.zeros(token_count, 36, dtype=torch.float32)
        generated_residues = frozenset(generated_map.values())
        for token_index, (residue, component_id) in enumerate(zip(token_residues, token_components)):
            index = (
                _GENERATED_RESTYPE
                if residue in generated_residues
                else _RESTYPE_INDEX.get(component_id, _RESTYPE_INDEX["UNK"])
            )
            restype[token_index, index] = 1.0

        observed = {} if structure is None else {site: row for row, site in enumerate(structure.sites)}
        representative_coordinates = torch.zeros(token_count, 3, dtype=torch.float32)
        representative_mask = torch.zeros(token_count, dtype=torch.bool)
        for token_index, site in enumerate(representative_sites):
            row = observed.get(site)
            if row is None or not bool(structure.coordinate_mask[row]):  # type: ignore[union-attr]
                continue
            coordinates = cast(Structure, structure).coordinates[row]
            representative_coordinates[token_index] = torch.from_numpy(coordinates.copy())
            representative_mask[token_index] = True
        conditional_templ = torch.zeros(token_count, token_count, dtype=torch.int64)
        conditional_templ_mask = representative_mask[:, None] & representative_mask[None, :]
        if bool(representative_mask.any()):
            rows = torch.nonzero(representative_mask, as_tuple=True)[0]
            distances = torch.cdist(
                representative_coordinates.index_select(0, rows),
                representative_coordinates.index_select(0, rows),
            )
            bins = torch.sum(
                distances.unsqueeze(-1) > torch.linspace(2.0, 22.0, 63),
                dim=-1,
            )
            ii, jj = torch.meshgrid(rows, rows, indexing="ij")
            conditional_templ[ii, jj] = bins

        chain_index, entity_index, symmetry_index, residue_index = _token_indices(
            system,
            token_sites,
            token_residues,
        )
        return PxDesignInput(
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(token_count, dtype=torch.int64, device=self.device).unsqueeze(0),
            atom_features=atom_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(1, len(sites), dtype=torch.bool, device=self.device),
            reference_positions=reference_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=reference_charges.reshape(1, len(sites), 1).to(
                device=self.device,
                dtype=self.dtype,
            ),
            restype=restype.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            plddt=torch.zeros(1, token_count, 1, dtype=self.dtype, device=self.device),
            hotspot=torch.tensor(
                [residue in hotspot_residues for residue in token_residues],
                dtype=self.dtype,
                device=self.device,
            ).reshape(1, token_count, 1),
            conditional_templ=conditional_templ.unsqueeze(0).to(self.device),
            conditional_templ_mask=conditional_templ_mask.unsqueeze(0).to(self.device),
            layout=PxDesignLayout(
                system=system,
                sites=sites,
                conditions=condition_values,
                # Frozen output surface: group -> placement_id -> ordered realized
                # occurrence tuple (offset-indexed; negative offsets resolve natively).
                # Exhaustive coverage (X1-B witness rule): the single binder placement covers
                # every realized generated residue (offsets 0..max_length-1) with no
                # unaddressable tail. Target ExistingSpans stay source-addressed and out of
                # the map, matching the sibling generated-only convention (odesign/boltzgen).
                # Each ComponentReference[0] carries the shared-allocator realized id
                # (``chain_0``), not the ``placement_id``.
                placement_map={generated_group: {generated_id: tuple(generated_map.values())}},
            ),
        )


def _canonicalize_modified_polymers(
    source: MolecularSystem | None,
    structure: Structure | None,
) -> tuple[MolecularSystem | None, Structure | None]:
    r"""Apply the released AlphaFold3-SI-§2.1 MSE->MET modified-polymer transform.

    Released PXDesign canonicalizes selenomethionine before it tokenizes or featurizes
    (reference ``pxdesign/data/parser.py::mse_to_met`` and
    ``json_to_feature.py::mse_to_met``): every ``MSE`` residue becomes ``MET``, its
    selenium ``SE`` atom is renamed to the sulfur ``SD`` (element ``S``), and the residue
    is relabeled a standard (non-hetero) MET. ``MSE`` and ``MET`` are otherwise the exact
    same heavy-atom graph in the wwPDB CCD (``N CA C O CB CG {SE|SD} CE`` plus terminal
    ``OXT``), so this is an exact canonicalization, not a lossy guess: rewriting the
    residue to the audited ``MET`` component and renaming its ``SE`` occurrence to ``SD``
    makes an ``MSE`` residue byte-identical to the equivalent ``MET`` residue.

    Any *other* modified polymer residue would need the CCD canonical-parent mapping that
    the package :class:`ChemicalComponent` value does not carry; those are left unchanged
    here and rejected precisely by the modeled-residue guard in :meth:`PxDesignProcessor.__call__`.
    """

    system = structure.system if structure is not None else source
    if system is None:
        return source, structure

    met = get_component("MET")
    new_entities: dict[str, PolymerSequence | Molecule] = {}
    modified_entities: set[str] = set()
    for entity_id, entity in system.entities.items():
        if type(entity) is PolymerSequence and any(component.component_id == "MSE" for component in entity.components):
            new_entities[entity_id] = PolymerSequence(
                entity.polymer_type,
                tuple(met if component.component_id == "MSE" else component for component in entity.components),
            )
            modified_entities.add(entity_id)
        else:
            new_entities[entity_id] = entity
    if not modified_entities:
        return source, structure

    # ``(instance_id, component_index)`` of every residue rewritten from MSE to MET; only the
    # selenium ``SE`` occurrence in exactly those residues is renamed to the sulfur ``SD``.
    modified_residues = {
        (instance_id, index)
        for instance_id, entity_id in system.instances.items()
        if entity_id in modified_entities
        for index, component in enumerate(system.entities[entity_id].components)
        if component.component_id == "MSE"
    }

    def rename(site: AtomSite) -> AtomSite:
        if site.atom_id == "SE" and (site.instance_id, site.component_index) in modified_residues:
            return AtomSite(site.instance_id, site.component_index, "SD")
        return site

    new_bonds = frozenset(
        SystemBond(
            rename(bond.site_a),
            rename(bond.site_b),
            order=bond.order,
            aromatic=bond.aromatic,
            stereo=bond.stereo,
            stereo_sites=(
                None if bond.stereo_sites is None else (rename(bond.stereo_sites[0]), rename(bond.stereo_sites[1]))
            ),
            origin=bond.origin,
        )
        for bond in system.bonds
    )
    new_system = MolecularSystem(new_entities, dict(system.instances), new_bonds)
    if structure is None:
        return new_system, None
    new_structure = Structure(
        new_system,
        tuple(rename(site) for site in structure.sites),
        structure.coordinates,
        structure.coordinate_mask,
        structure.occupancy,
        structure.residue_numbers,
    )
    return new_system, new_structure


def _materialize_problem(
    source: MolecularSystem | None,
    structure: Structure | None,
    existing_chains: tuple[tuple[ExistingSpan, ...], ...],
    generated: GeneratedSegment,
    generated_id: str,
) -> tuple[
    MolecularSystem,
    tuple[AtomSite, ...],
    dict[SymbolicResidueSite, tuple[str, int]],
    frozenset[tuple[str, int]],
]:
    explicit_instances = [part.instance_id for chain in existing_chains for part in chain]
    if source is None:
        selected_instances: list[tuple[str, str]] = []
    elif explicit_instances:
        selected_ids = set(explicit_instances)
        selected_instances = [
            (instance_id, source.instances[instance_id]) for instance_id in dict.fromkeys(explicit_instances)
        ]
        selected_instances.extend(
            (instance_id, entity_id)
            for instance_id, entity_id in source.instances.items()
            if type(source.entities[entity_id]) is Molecule and instance_id not in selected_ids
        )
    else:
        selected_instances = list(source.instances.items())

    selected_instance_ids = {instance_id for instance_id, _ in selected_instances}
    selected_entity_ids = {entity_id for _, entity_id in selected_instances}
    entities: dict[str, PolymerSequence | Molecule] = (
        {} if source is None else {entity_id: source.entities[entity_id] for entity_id in selected_entity_ids}
    )

    # Uniform realized-id convention (owner "one correct way"): the realized INSTANCE id is
    # minted by the single shared allocator (``ofoldx.data.placement.InstanceIdAllocator`` ->
    # ``chain_{n}``) and surfaces only inside ``ComponentReference[0]``; all four families call
    # the same helper. Entity ids remain family-owned (``{family}_entity_{N}``). PXDesign
    # realizes exactly one generated chain — the binder — so it allocates ``chain_0`` (below).
    # The placement is still KEYED by its globally-unique ``generated_id``; only the realized id
    # is minted.
    entity_id = "pxdesign_entity_0"
    suffix = 1
    while entity_id in selected_entity_ids:
        entity_id = f"pxdesign_entity_0_{suffix}"
        suffix += 1
    entities[entity_id] = PolymerSequence("protein", (get_component("UNK"),) * generated.max_length)

    # Realized instance ids are minted by the shared declaration-order allocator
    # (``ofoldx.data.placement``) per the canonical ``chain_{n}`` convention. PXDesign realizes
    # exactly one generated output chain — the binder — so it allocates a single id (``chain_0``).
    # The placement stays KEYED by its globally-unique ``generated_id``; only the realized id is
    # minted here. ``chain_{n}`` is disjoint by construction from every SOURCE instance id
    # (author chain ids, symmetry ``A#1``, ``molecule_...``), so no collision guard is needed.
    instance_id = InstanceIdAllocator().allocate()
    instances = dict(selected_instances)
    instances[instance_id] = entity_id

    bonds = (
        frozenset()
        if source is None
        else frozenset(
            bond
            for bond in source.bonds
            if bond.site_a.instance_id in selected_instance_ids and bond.site_b.instance_id in selected_instance_ids
        )
    )
    system = MolecularSystem(entities, instances, bonds)

    if existing_chains:
        selected_residues = frozenset(
            (part.instance_id, monomer_index)
            for chain in existing_chains
            for part in chain
            for monomer_index in range(part.start, part.stop)
        )
    else:
        selected_residues = frozenset(
            (instance_id, monomer_index)
            for instance_id, entity_id in selected_instances
            if type(entity := system.entities[entity_id]) is PolymerSequence
            for monomer_index in range(len(entity))
        )
    if structure is None:
        observed_residues: frozenset[tuple[str, int]] = frozenset()
        observed_molecules: frozenset[str] = frozenset()
    else:
        molecule_instances = {
            instance_id
            for instance_id, entity_id in structure.system.instances.items()
            if type(structure.system.entities[entity_id]) is Molecule
        }
        observed_residues = frozenset(
            (site.instance_id, site.component_index)
            for site, present in zip(structure.sites, structure.coordinate_mask)
            if present and site.instance_id not in molecule_instances
        )
        observed_molecules = frozenset(
            site.instance_id
            for site, present in zip(structure.sites, structure.coordinate_mask)
            if present and site.instance_id in molecule_instances
        )
    # One canonical, offset-indexed key per generated residue (offsets 0..max_length-1, in
    # order). The former positive/negative double-key alias table is deleted: a
    # from-the-end reference is resolved by native negative indexing into the ordered
    # placement_map occurrence tuple, not by pre-keying a second SymbolicResidueSite.
    generated_map = {}
    for offset in range(generated.max_length):
        residue = (instance_id, offset)
        generated_map[SymbolicResidueSite(generated_id, offset)] = residue
    generated_residues = frozenset(generated_map.values())
    sites = []
    for site, atom in system.iter_atoms():
        if atom.element == "H":
            continue
        entity = system.entities[system.instances[site.instance_id]]
        residue = None if type(entity) is Molecule else (site.instance_id, site.component_index)
        if residue in generated_residues:
            if site.atom_id in _GENERATED_ATOM_IDS:
                sites.append(site)
        elif residue is not None:
            if residue in selected_residues and residue in observed_residues:
                sites.append(site)
        elif site.instance_id in observed_molecules:
            sites.append(site)
    if not sites:
        raise ValueError("PXDesign requires at least one modeled atom")
    return system, tuple(sites), generated_map, selected_residues


def _tokens(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...],
    generated_residues: frozenset[tuple[str, int]],
) -> tuple[
    tuple[tuple[AtomSite, ...], ...],
    tuple[tuple[str, int] | None, ...],
    tuple[str, ...],
    tuple[AtomSite, ...],
]:
    groups: list[tuple[AtomSite, ...]] = []
    residues: list[tuple[str, int] | None] = []
    components: list[str] = []
    representatives: list[AtomSite] = []
    by_residue: dict[tuple[str, int], list[AtomSite]] = {}
    for site in sites:
        if type(system.entities[system.instances[site.instance_id]]) is not Molecule:
            by_residue.setdefault((site.instance_id, site.component_index), []).append(site)
    emitted_residues: set[tuple[str, int]] = set()
    for site in sites:
        entity = system.entities[system.instances[site.instance_id]]
        if type(entity) is Molecule:
            groups.append((site,))
            residues.append(None)
            components.append("UNK")
            representatives.append(site)
            continue
        polymer = cast(PolymerSequence, entity)
        residue = (site.instance_id, site.component_index)
        if residue in emitted_residues:
            continue
        emitted_residues.add(residue)
        component_id = polymer.components[site.component_index].component_id
        group = tuple(by_residue[residue])
        if polymer.polymer_type == "protein":
            representative_id = "CA" if component_id == "GLY" or residue in generated_residues else "CB"
        elif component_id in {"DA", "DG", "A", "G"}:
            representative_id = "C4"
        elif component_id in {"DC", "DT", "C", "U"}:
            representative_id = "C2"
        else:
            representative_id = "C1'"
        representative = next(atom_site for atom_site in group if atom_site.atom_id == representative_id)
        groups.append(group)
        residues.append(residue)
        components.append(component_id)
        representatives.append(representative)
    return tuple(groups), tuple(residues), tuple(components), tuple(representatives)


def _hotspots(
    conditions: tuple[PlacementSpec | Hotspot, ...],
    selected_residues: frozenset[tuple[str, int]],
    token_residues: tuple[tuple[str, int] | None, ...],
) -> frozenset[tuple[str, int]]:
    r"""Collect the released per-token hotspot target residues (frozen §3.7.4 ``Hotspot``).

    The frozen :class:`~ofoldx.data.Hotspot` is single-sided and distance-free: ``residues``
    name the EXISTING target residues to condition on and the generated partner is implicit
    (the whole binder). Each named residue must be a materialized target
    ``ComponentReference`` — a ``(instance_id, component_index)`` tuple present in the selected,
    tokenized target — so a generated-binder endpoint (a bare ``placement_id`` / a
    :class:`~ofoldx.data.SymbolicResidueSite` / a :class:`~ofoldx.data.SymbolicAtomSite`) is
    rejected: PXDesign hotspots its target, not its binder. This mirrors the previous
    two-sided ``Contact`` lowering (which required one all-existing target side); the cutoff
    it validated-then-ignored is gone with the record.
    """

    hotspots: set[tuple[str, int]] = set()
    token_residue_set = {residue for residue in token_residues if residue is not None}
    for condition in conditions:
        if type(condition) is not Hotspot:
            continue
        for reference in condition.residues:
            if type(reference) is not tuple:
                raise ValueError(
                    "PXDesign hotspot names an existing target residue (ComponentReference); "
                    f"the generated binder is implicit and cannot be a hotspot: {reference!r}"
                )
            residue = cast(tuple[str, int], reference)
            if residue not in selected_residues or residue not in token_residue_set:
                raise ValueError(f"PXDesign hotspot residue is absent from the modeled target: {residue!r}")
            hotspots.add(residue)
    return frozenset(hotspots)


def _component_reference(component: ChemicalComponent) -> dict[str, tuple[float, float, float]]:
    return {
        atom.atom_id: coordinate
        for atom, coordinate in zip(component.atoms, component.reference_coordinates)
        if coordinate is not None
    }


def _reference_features(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...],
    generated_residues: frozenset[tuple[str, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    glycine_coordinates = _component_reference(get_component("GLY"))
    positions = torch.zeros(len(sites), 3, dtype=torch.float32)
    mask = torch.zeros(len(sites), dtype=torch.bool)
    spaces: dict[tuple[str, int | None], int] = {}
    space_ids: list[int] = []
    for row, site in enumerate(sites):
        entity = system.entities[system.instances[site.instance_id]]
        residue = (site.instance_id, site.component_index)
        if type(entity) is not Molecule and residue in generated_residues:
            coordinate = glycine_coordinates.get(site.atom_id)
        else:
            coordinate = _component_reference(entity.components[site.component_index]).get(site.atom_id)
        if coordinate is not None:
            positions[row] = torch.tensor(coordinate)
            mask[row] = True
        space: tuple[str, int | None] = (
            (site.instance_id, None) if type(entity) is Molecule else (site.instance_id, site.component_index)
        )
        if space not in spaces:
            spaces[space] = len(spaces)
        space_ids.append(spaces[space])
    space_id = torch.tensor(space_ids, dtype=torch.int64)
    for value in range(len(spaces)):
        selected = space_id == value
        positions[selected] -= positions[selected].mean(dim=0)
    return positions, mask, space_id


def _atom_features(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...],
    reference_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    element = torch.zeros(len(sites), 128)
    names = torch.zeros(len(sites), 4, 64)
    charges = torch.zeros(len(sites), dtype=torch.float32)
    for row, site in enumerate(sites):
        atom = system.atom(site)
        element[row, int(periodic_table.GetAtomicNumber(atom.element)) - 1] = 1.0
        charges[row] = atom.formal_charge
        if len(site.atom_id) > 4:
            raise ValueError(f"PXDesign cannot encode atom name longer than four characters: {site.atom_id!r}")
        for column, character in enumerate(site.atom_id.ljust(4)):
            index = ord(character) - 32
            if not 0 <= index < 64:
                raise ValueError(f"PXDesign cannot encode atom-name character {character!r}")
            names[row, column, index] = 1.0
    features = torch.cat((reference_mask.float().unsqueeze(-1), element, names.flatten(-2)), dim=-1)
    return features, charges


def _token_indices(
    system: MolecularSystem,
    token_sites: tuple[tuple[AtomSite, ...], ...],
    token_residues: tuple[tuple[str, int] | None, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    chain = {instance_id: index for index, instance_id in enumerate(system.instances)}
    entity = {value: index for index, value in enumerate(sorted(set(system.instances.values())))}
    seen: dict[str, int] = {}
    symmetry: dict[str, int] = {}
    for instance_id, entity_id in system.instances.items():
        symmetry[instance_id] = seen.get(entity_id, 0)
        seen[entity_id] = symmetry[instance_id] + 1
    chain_values = []
    entity_values = []
    symmetry_values = []
    residue_values = []
    for group, residue in zip(token_sites, token_residues):
        instance_id = group[0].instance_id
        entity_id = system.instances[instance_id]
        chain_values.append(chain[instance_id])
        entity_values.append(entity[entity_id])
        symmetry_values.append(symmetry[instance_id])
        residue_values.append(0 if residue is None else residue[1])
    return tuple(
        torch.tensor(values, dtype=torch.int64)
        for values in (chain_values, entity_values, symmetry_values, residue_values)
    )


# ---------------------------------------------------------------------------
# Binder ⇄ placement-core projection (folded from projection.py;
# CONTIG_GENERALITY_AUDIT §3, "pxdesign").
#
# The frozen §3.7 placement core (``PlacementSpec``) is the shared canonical form; each
# family keeps its own native surface and needs a lossless declaration-side projection
# to/from it. PXDesign's native surface is a *target-and-binder* problem: a fixed protein
# binder of one length designed against a set of existing target chains.
#
# * ``from_binder`` lowers the native binder problem to a ``PlacementSpec``: every target
#   chain becomes a declaration group of ordered ``ExistingSpan`` values (a discontinuous
#   crop → several half-open spans), and the binder becomes ONE trailing ``GeneratedSegment``
#   group (the released processor requires the binder last).
# * ``to_binder`` recovers ``(binder_length, target chains, count)`` from a ``PlacementSpec``.
#
# **Named losses (both by construction, per the audit):**
#
# * ``count`` (N symmetric copies) has **no home in the declaration** — it is a family
#   realization argument (parallel to how length re-homes), recovered only from a realized
#   ``placement_map`` where one declared binder placement spans N occurrences. A bare
#   ``PlacementSpec`` therefore always round-trips to ``count == 1``; ``from_binder`` accepts
#   ``count`` for API symmetry but deliberately does NOT encode it.
# * The released ``PxDesignProcessor`` narrows *further* than the core: it consumes one
#   complete span per target chain and no interleaved binder. The core (and this projection)
#   represent a discontinuous crop losslessly; the released path simply chooses not to
#   consume it yet. So ``from_binder`` can emit a multi-span target that round-trips through
#   ``to_binder`` even though the processor would reject it — an impl narrowing, not a core
#   loss.
# ---------------------------------------------------------------------------

#: One target chain: its source ``instance_id`` and an ordered list of half-open
#: ``[start, stop)`` crop intervals. A single complete-chain crop is the common (released)
#: case; several intervals express a discontinuous crop the core represents losslessly.
TargetChain = tuple[str, list[tuple[int, int]]]

_BINDER_LABEL = "binder"


def from_binder(
    binder_length: int,
    target: Sequence[TargetChain] = (),
    *,
    count: int = 1,
) -> PlacementSpec:
    r"""Project a native target-and-binder problem onto the shared :data:`PlacementSpec`.

    ``binder_length`` fixes the single generated protein binder (``min == max``). Each
    ``target`` chain becomes a declaration group of ordered :class:`~ofoldx.data.ExistingSpan`
    values; the binder is appended as the trailing generated group. ``count`` is a family
    realization argument and is validated but NOT stored (see the module named-loss note).
    """

    if binder_length < 0:
        raise ValueError("binder_length must be non-negative")
    if count < 1:
        raise ValueError("count (symmetric copies) must be a positive integer")

    placement: PlacementSpec = {}
    for instance_id, crops in target:
        crop_intervals = tuple(crops)
        if not crop_intervals:
            raise ValueError(f"target chain {instance_id!r} needs at least one crop interval")
        if instance_id in placement:
            raise ValueError(f"target instance {instance_id!r} declared more than once")
        group: dict[str, ExistingSpan] = {}
        for start, stop in crop_intervals:
            if not 0 <= start < stop:
                raise ValueError(
                    f"target chain {instance_id!r} crop must be a half-open [start, stop) with "
                    f"0 <= start < stop; got ({start}, {stop})"
                )
            placement_id = instance_id if len(crop_intervals) == 1 else f"{instance_id}#{start}_{stop}"
            if placement_id in group:
                raise ValueError(f"target chain {instance_id!r} has a duplicate crop {(start, stop)!r}")
            group[placement_id] = ExistingSpan(instance_id, start, stop)
        placement[instance_id] = group

    existing_ids = {placement_id for chain in placement.values() for placement_id in chain}
    binder_label = _BINDER_LABEL
    while binder_label in placement or binder_label in existing_ids:
        binder_label += "_"
    placement[binder_label] = {binder_label: GeneratedSegment("protein", binder_length, binder_length)}

    # Enforce the frozen construction-time invariant (globally-unique placement_id).
    validate_placement(placement)
    return placement


def to_binder(placement: PlacementSpec) -> tuple[int, list[TargetChain], int]:
    r"""Recover ``(binder_length, target chains, count)`` from a :data:`PlacementSpec`.

    The trailing declaration group must be exactly one fixed-length protein
    :class:`~ofoldx.data.GeneratedSegment` (the binder); every earlier group must contain
    only :class:`~ofoldx.data.ExistingSpan` values addressing one source instance. ``count``
    is not declaration-recoverable and is always returned as ``1`` (module named-loss note).
    """

    validate_placement(placement)
    groups = list(placement.items())
    if not groups:
        raise ValueError("an empty PlacementSpec declares no binder")

    _, binder_placements = groups[-1]
    binder_values = tuple(binder_placements.values())
    if len(binder_values) != 1 or type(binder_values[0]) is not GeneratedSegment:
        raise ValueError("the trailing PlacementSpec group must be exactly one generated binder")
    binder = binder_values[0]
    if binder.polymer_type != "protein":
        raise ValueError("PXDesign binder must be a protein GeneratedSegment")
    if binder.min_length != binder.max_length:
        raise ValueError("PXDesign binder must have one fixed length (min_length == max_length)")
    binder_length = binder.max_length

    targets: list[TargetChain] = []
    for group, placements in groups[:-1]:
        instance_id: str | None = None
        crops: list[tuple[int, int]] = []
        for part in placements.values():
            if type(part) is not ExistingSpan:
                raise ValueError(f"target group {group!r} must contain only ExistingSpan values")
            if instance_id is None:
                instance_id = part.instance_id
            elif part.instance_id != instance_id:
                raise ValueError(f"target group {group!r} must address one source instance")
            crops.append((part.start, part.stop))
        targets.append((instance_id, crops))  # type: ignore[arg-type]

    # count is a realization argument with no home in the declaration; a bare PlacementSpec
    # declares one binder placement, so it recovers as 1.
    return binder_length, targets, 1
