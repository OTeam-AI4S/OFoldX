r"""Canonical Structure lowering for MpnnPacker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
import torch
from typing_extensions import Self

from ...data import ATOMIC_NUMBER_BY_ELEMENT, is_water_component
from ...data.components import get_component
from ...data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules import residue_constants as rc
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ..mpnn_pdb import read_mpnn_pdb
from .config import MpnnPackerProcessorConfig
from .model import MpnnPackerInput


class MpnnPackerProcessor:
    r"""Lower one canonical Structure to the released MpnnPacker."""

    def __init__(
        self,
        config: MpnnPackerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = MpnnPackerProcessorConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update runtime placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> MpnnPackerInput:
        r"""Lower fixed identity, selected residues, coordinates, and nearest-atom context."""

        structure = read_mpnn_pdb(input) if isinstance(input, (str, PathLike)) else input
        system = structure.system
        protein_instances: list[tuple[str, PolymerSequence]] = []
        for instance_id, entity_id in system.instances.items():
            entity = system.entities[entity_id]
            if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                protein_instances.append((instance_id, entity))
        if isinstance(sequence, str):
            if len(protein_instances) != 1:
                raise ValueError("a bare packing sequence requires exactly one protein instance")
            sequence = {protein_instances[0][0]: sequence}

        target_components = {instance_id: entity.components for instance_id, entity in protein_instances}
        if sequence is not None:
            for instance_id, value in sequence.items():
                entity = system.entities[system.instances[instance_id]]
                if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                    raise ValueError(f"packing sequence instance {instance_id!r} is not a protein")
                if isinstance(value, str):
                    requested = PolymerSequence.from_one_letter(value).components
                else:
                    requested = tuple(get_component(component_id) for component_id in value)
                if len(requested) != len(entity):
                    raise ValueError(
                        f"packing sequence for instance {instance_id!r} has length {len(requested)}, "
                        f"expected {len(entity)}"
                    )
                target_components[instance_id] = tuple(
                    original if original.component_id == replacement.component_id else replacement
                    for original, replacement in zip(entity.components, requested)
                )

        changed_references = {
            (instance_id, component_index)
            for instance_id, entity in protein_instances
            for component_index, (original, target) in enumerate(zip(entity.components, target_components[instance_id]))
            if original.component_id != target.component_id
        }
        if changed_references:
            source_atom_ids: dict[ComponentReference, list[str]] = {}
            for site in structure.sites:
                source_atom_ids.setdefault((site.instance_id, site.component_index), []).append(site.atom_id)
            entities: dict[str, PolymerSequence | Molecule] = {}
            instance_entities: dict[str, str] = {}
            allocated_ids = set(system.entities)
            for entity_id, entity in system.entities.items():
                entity_instances = tuple(
                    instance_id
                    for instance_id, instance_entity_id in system.instances.items()
                    if instance_entity_id == entity_id
                )
                if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                    entities[entity_id] = entity
                    for instance_id in entity_instances:
                        instance_entities[instance_id] = entity_id
                    continue

                groups: dict[tuple[str, ...], list[str]] = {}
                for instance_id in entity_instances:
                    instance_components = target_components[instance_id]
                    component_ids = tuple(component.component_id for component in instance_components)
                    groups.setdefault(component_ids, []).append(instance_id)
                original_ids = tuple(component.component_id for component in entity.components)
                ordered_ids = tuple(groups)
                retained_ids = original_ids if original_ids in groups else ordered_ids[0]
                next_suffix = 2
                for component_ids in ordered_ids:
                    if component_ids == retained_ids:
                        target_entity_id = entity_id
                    else:
                        while f"{entity_id}_{next_suffix}" in allocated_ids:
                            next_suffix += 1
                        target_entity_id = f"{entity_id}_{next_suffix}"
                        allocated_ids.add(target_entity_id)
                        next_suffix += 1
                    group_instances = groups[component_ids]
                    requested_components = target_components[group_instances[0]]
                    realized_components: list[ChemicalComponent] = []
                    for component_index, (original, target) in enumerate(zip(entity.components, requested_components)):
                        if original.component_id == target.component_id:
                            realized_components.append(original)
                            continue

                        previous_attachment = (
                            target.previous_attachment if original.previous_attachment is not None else None
                        )
                        next_attachment = target.next_attachment if original.next_attachment is not None else None
                        if original.bonds is not None:
                            realized_components.append(
                                ChemicalComponent(
                                    component_id=target.component_id,
                                    atoms=target.atoms,
                                    bonds=target.bonds,
                                    previous_attachment=previous_attachment,
                                    next_attachment=next_attachment,
                                    reference_coordinates=target.reference_coordinates,
                                )
                            )
                            continue

                        target_atoms = {atom.atom_id: atom for atom in target.atoms}
                        atom_ids: list[str] = []
                        seen_atom_ids: set[str] = set()
                        for atom_id in rc.restype_name_to_atom14_names.get(target.component_id, ()):
                            if atom_id and atom_id not in seen_atom_ids:
                                atom_ids.append(atom_id)
                                seen_atom_ids.add(atom_id)
                        for instance_id in group_instances:
                            for atom_id in source_atom_ids.get((instance_id, component_index), ()):
                                if atom_id in target_atoms and atom_id not in seen_atom_ids:
                                    atom_ids.append(atom_id)
                                    seen_atom_ids.add(atom_id)
                        attachment_atom_ids: set[str] = set()
                        for attachment in (previous_attachment, next_attachment):
                            if attachment is not None:
                                attachment_atom_ids.add(attachment.atom_id)
                                attachment_atom_ids.update(attachment.leaving_atom_ids)
                        for atom in target.atoms:
                            if atom.atom_id in attachment_atom_ids and atom.atom_id not in seen_atom_ids:
                                atom_ids.append(atom.atom_id)
                                seen_atom_ids.add(atom.atom_id)
                        realized_components.append(
                            ChemicalComponent(
                                component_id=target.component_id,
                                atoms=tuple(
                                    ChemicalAtom(atom_id=atom_id, element=target_atoms[atom_id].element)
                                    for atom_id in atom_ids
                                ),
                                bonds=None,
                                previous_attachment=previous_attachment,
                                next_attachment=next_attachment,
                            )
                        )

                    entities[target_entity_id] = PolymerSequence("protein", tuple(realized_components))
                    for instance_id in group_instances:
                        instance_entities[instance_id] = target_entity_id

            instances = {instance_id: instance_entities[instance_id] for instance_id in system.instances}
            system = MolecularSystem(entities, instances, system.bonds)
            retained_sites = {site for site, _atom in system.iter_atoms()}
            indices = tuple(index for index, site in enumerate(structure.sites) if site in retained_sites)
            structure = Structure(
                system,
                tuple(structure.sites[index] for index in indices),
                structure.coordinates[np.asarray(indices)],
                structure.coordinate_mask[np.asarray(indices)],
                None if structure.occupancy is None else structure.occupancy[np.asarray(indices)],
                structure.residue_numbers,
            )

        normalized_entities = dict(system.entities)
        normalized = False
        for entity_id, entity in system.entities.items():
            if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                continue
            components: list[ChemicalComponent] = []
            entity_normalized = False
            for component in entity.components:
                if component.component_id not in AMINO_ACID_COMPONENT_IDS or component.bonds is not None:
                    components.append(component)
                    continue
                declared_atom_ids = {atom.atom_id for atom in component.atoms}
                missing_atom_ids = tuple(
                    atom_id
                    for atom_id in rc.restype_name_to_atom14_names[component.component_id]
                    if atom_id and atom_id not in declared_atom_ids
                )
                if not missing_atom_ids:
                    components.append(component)
                    continue
                audited_atoms = {atom.atom_id: atom for atom in get_component(component.component_id).atoms}
                reference_coordinates = component.reference_coordinates
                if reference_coordinates:
                    reference_coordinates += (None,) * len(missing_atom_ids)
                components.append(
                    ChemicalComponent(
                        component_id=component.component_id,
                        atoms=component.atoms
                        + tuple(
                            ChemicalAtom(atom_id=atom_id, element=audited_atoms[atom_id].element)
                            for atom_id in missing_atom_ids
                        ),
                        bonds=None,
                        previous_attachment=component.previous_attachment,
                        next_attachment=component.next_attachment,
                        reference_coordinates=reference_coordinates,
                    )
                )
                entity_normalized = True
            if entity_normalized:
                normalized_entities[entity_id] = PolymerSequence("protein", tuple(components))
                normalized = True

        if normalized:
            system = MolecularSystem(normalized_entities, system.instances, system.bonds)
            structure = Structure(
                system,
                structure.sites,
                structure.coordinates,
                structure.coordinate_mask,
                structure.occupancy,
                structure.residue_numbers,
            )

        observed = structure.coordinate_mask
        if filter_by_occupancy and structure.occupancy is not None:
            observed = observed & (structure.occupancy > 0)

        rows: list[tuple[ComponentReference, str, int, int]] = []
        chain_indices: dict[str, int] = {}
        for site, present in zip(structure.sites, observed):
            if not present or site.atom_id != "CA":
                continue
            entity = system.entities[system.instances[site.instance_id]]
            if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                continue
            if site.instance_id not in chain_indices:
                chain_indices[site.instance_id] = len(chain_indices)
            reference = (site.instance_id, site.component_index)
            component = entity.components[site.component_index]
            residue_index = (
                site.component_index
                if structure.residue_numbers is None
                else structure.residue_numbers[reference].number
            )
            rows.append((reference, component.component_id, chain_indices[site.instance_id], residue_index))

        adjusted_rows: list[tuple[ComponentReference, str, int, int]] = []
        duplicate_count = 0
        previous_residue_index: int | None = None
        for reference, component_id, chain_index, residue_index in rows:
            if previous_residue_index == residue_index:
                duplicate_count += 1
            adjusted_rows.append((reference, component_id, chain_index, residue_index + duplicate_count))
            previous_residue_index = residue_index
        rows = adjusted_rows
        unresolved_expected_atoms: list[tuple[str, ...]] = []
        for reference, component_id, _chain_index, _residue_index in rows:
            if component_id not in AMINO_ACID_COMPONENT_IDS:
                unresolved_expected_atoms.append(())
                continue
            unresolved: list[str] = []
            for atom_id in rc.restype_name_to_atom14_names[component_id]:
                if not atom_id:
                    continue
                try:
                    system.atom(AtomSite(reference[0], reference[1], atom_id))
                except KeyError:
                    unresolved.append(atom_id)
            unresolved_expected_atoms.append(tuple(unresolved))
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        coordinate_dtype = structure.coordinates.dtype
        backbone: list[np.ndarray] = []
        all_atom = np.zeros((len(rows), len(rc.atom_types), 3), dtype=coordinate_dtype)
        all_atom_mask = np.zeros((len(rows), len(rc.atom_types)), dtype=np.bool_)
        valid: list[bool] = []
        for row_index, (residue, _component_id, _chain_index, _residue_index) in enumerate(rows):
            for atom_index, atom_name in enumerate(rc.atom_types):
                if atom_name == "OXT":
                    continue
                site = AtomSite(residue[0], residue[1], atom_name)
                index = coordinate_index.get(site)
                if index is not None and bool(observed[index]):
                    all_atom[row_index, atom_index] = structure.coordinates[index]
                    all_atom_mask[row_index, atom_index] = True
            backbone.append(all_atom[row_index, (0, 1, 2, 4)])
            valid.append(bool(all_atom_mask[row_index, (0, 1, 2, 4)].all()))
        row_references = tuple(row[0] for row in rows)
        if positions is None:
            selected = tuple(
                is_valid and component_id in AMINO_ACID_COMPONENT_IDS and not unresolved
                for (_residue, component_id, _chain_index, _residue_index), is_valid, unresolved in zip(
                    rows,
                    valid,
                    unresolved_expected_atoms,
                )
            )
        else:
            selected_references = set(positions)
            unknown = selected_references - set(row_references)
            if unknown:
                raise ValueError(f"packing positions are outside the observed protein layout: {sorted(unknown)!r}")
            selected = tuple(residue in selected_references for residue in row_references)
            for row, is_selected, is_valid, unresolved_atom_ids in zip(
                rows, selected, valid, unresolved_expected_atoms
            ):
                if not is_selected:
                    continue
                if row[1] not in AMINO_ACID_COMPONENT_IDS:
                    raise ValueError(f"MpnnPacker cannot pack residue identity {row[1]!r} at {row[0]!r}")
                if unresolved_atom_ids:
                    raise ValueError(
                        f"MpnnPacker cannot pack residue {row[0]!r} because expected atom14 sites "
                        f"do not resolve: {unresolved_atom_ids!r}"
                    )
                if not is_valid:
                    raise ValueError(f"MpnnPacker cannot pack residue {row[0]!r} without complete backbone coordinates")

        unselected_changes = changed_references - {
            reference for reference, is_selected in zip(row_references, selected) if is_selected
        }
        if unselected_changes:
            raise ValueError(f"changed identities must be selected for packing: {sorted(unselected_changes)!r}")

        backbone_tensor = torch.as_tensor(np.stack(backbone), device=self.device, dtype=self.dtype).unsqueeze(0)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=self.device).unsqueeze(0)
        num_context_atoms = self.config.num_context_atoms
        selected_coords = torch.zeros(len(rows), num_context_atoms, 3, device=self.device, dtype=self.dtype)
        selected_types = torch.zeros(len(rows), num_context_atoms, device=self.device, dtype=torch.int64)
        selected_mask = torch.zeros(len(rows), num_context_atoms, device=self.device, dtype=torch.bool)
        if use_atom_context:
            context_coords: list[np.ndarray] = []
            context_types: list[int] = []
            for site, coordinates, present in zip(structure.sites, structure.coordinates, observed):
                if not present:
                    continue
                entity = system.entities[system.instances[site.instance_id]]
                if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                    continue
                if is_water_component(entity.components[site.component_index]):
                    continue
                element = system.atom(site).element
                if element == "H":
                    continue
                context_coords.append(coordinates)
                context_types.append(ATOMIC_NUMBER_BY_ELEMENT[element])

            if context_coords:
                context = torch.as_tensor(np.stack(context_coords), device=self.device, dtype=self.dtype)
                context_type = torch.tensor(context_types, dtype=torch.int64, device=self.device)
                context_mask = context_type != 0
                n = backbone_tensor[0, :, 0]
                ca = backbone_tensor[0, :, 1]
                c = backbone_tensor[0, :, 2]
                cb = (
                    -0.58273431 * torch.cross(ca - n, c - ca, dim=-1)
                    + 0.56802827 * (ca - n)
                    - 0.54067466 * (c - ca)
                    + ca
                )
                squared_distance = ((cb[:, None] - context[None]) ** 2).sum(dim=-1)
                pair_mask = valid_tensor[0, :, None] * context_mask[None]
                adjusted_distance = squared_distance * pair_mask + (~pair_mask) * 1000.0
                atom_count = min(num_context_atoms, context.shape[0])
                nearest = torch.topk(
                    adjusted_distance,
                    atom_count,
                    dim=-1,
                    largest=False,
                    sorted=True,
                ).indices
                selected_coords[:, :atom_count] = context[nearest]
                selected_types[:, :atom_count] = context_type[nearest]
                selected_mask[:, :atom_count] = context_mask[nearest]

        return MpnnPackerInput(
            backbone_coordinates=backbone_tensor,
            sequence_tokens=torch.tensor(
                [AMINO_ACID_TOKEN_BY_COMPONENT.get(row[1], AMINO_ACID_UNKNOWN_TOKEN) for row in rows],
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            residue_mask=valid_tensor,
            residue_index=torch.tensor([row[3] for row in rows], device=self.device, dtype=torch.int64).unsqueeze(0),
            chain_index=torch.tensor([row[2] for row in rows], device=self.device, dtype=torch.int64).unsqueeze(0),
            context_coordinates=selected_coords.unsqueeze(0),
            context_atomic_numbers=selected_types.unsqueeze(0),
            context_mask=selected_mask.unsqueeze(0),
            atom37_coordinates=torch.as_tensor(all_atom, device=self.device, dtype=self.dtype).unsqueeze(0),
            atom37_mask=torch.as_tensor(all_atom_mask, device=self.device).unsqueeze(0),
            pack_mask=torch.tensor(selected, device=self.device, dtype=torch.bool).unsqueeze(0),
            structure=structure,
            component_references=row_references,
        )
