r"""
MPNN Packer input processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import Tensor

from ...data import ATOMIC_NUMBER_BY_ELEMENT, water_component_mask
from ...data.components import get_component
from ...data.molecular import (
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    ENTITY_PROTEIN,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    PolymerAttachment,
    PolymerSequence,
    replace_components,
)
from ...data.structure import Structure
from ...model import Model
from ...modules import residue_constants as rc
from ...modules.geometry import virtual_beta_carbon
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...processor import Processor
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import MpnnPackerInput

_BACKBONE_ATOM37_INDICES = (0, 1, 2, 4)


def _repartition_entities(
    system: MolecularSystem,
    target_component_ids: Sequence[str],
) -> MolecularSystem:
    r"""
    Split copies of each original entity by their complete target sequence.
    """

    symbols = list(system.symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

    def intern(symbol: str) -> int:
        index = symbol_index.get(symbol)
        if index is None:
            index = len(symbols)
            symbols.append(symbol)
            symbol_index[symbol] = index
        return index

    instance_entity = system.instance_entity.detach().cpu().tolist()
    instances_by_entity: list[list[int]] = [[] for _ in range(system.entity_symbol.numel())]
    for instance, entity in enumerate(instance_entity):
        instances_by_entity[entity].append(instance)

    used_ids = {system.symbols[int(symbol)] for symbol in system.entity_symbol.detach().cpu().tolist()}
    output_entity_symbol: list[int] = []
    output_entity_type: list[int] = []
    output_entity_molecule_type: list[int] = []
    output_instance_entity = [-1] * system.instance_symbol.numel()
    current_component_ids = [system.symbols[int(symbol)] for symbol in system.component_symbol.detach().cpu().tolist()]
    for entity, instances in enumerate(instances_by_entity):
        entity_id = system.entity_id(entity)
        entity_type = int(system.entity_type[entity])
        entity_molecule_type = int(system.entity_molecule_type[entity])
        if not instances:
            output_entity_symbol.append(intern(entity_id))
            output_entity_type.append(entity_type)
            output_entity_molecule_type.append(entity_molecule_type)
            continue

        def signature(instance: int, values: Sequence[str]) -> tuple[str, ...]:
            start = int(system.instance_component_ptr[instance])
            stop = int(system.instance_component_ptr[instance + 1])
            return tuple(values[start:stop])

        original_signature = signature(
            instances[0],
            current_component_ids,
        )
        groups: dict[tuple[str, ...], list[int]] = {}
        for instance in instances:
            groups.setdefault(
                signature(instance, target_component_ids),
                [],
            ).append(instance)
        primary = original_signature if original_signature in groups else next(iter(groups))
        ordered = (
            primary,
            *(value for value in groups if value != primary),
        )
        suffix = 2
        for partition, value in enumerate(ordered):
            if partition == 0:
                output_id = entity_id
            else:
                output_id = f"{entity_id}_{suffix}"
                while output_id in used_ids:
                    suffix += 1
                    output_id = f"{entity_id}_{suffix}"
                used_ids.add(output_id)
                suffix += 1
            output_entity = len(output_entity_symbol)
            output_entity_symbol.append(intern(output_id))
            output_entity_type.append(entity_type)
            output_entity_molecule_type.append(entity_molecule_type)
            for instance in groups[value]:
                output_instance_entity[instance] = output_entity

    return system.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        system_entity_ptr=system.system_entity_ptr.new_tensor([0, len(output_entity_symbol)]),
        entity_symbol=system.entity_symbol.new_tensor(output_entity_symbol),
        entity_type=system.entity_type.new_tensor(output_entity_type),
        entity_molecule_type=system.entity_molecule_type.new_tensor(output_entity_molecule_type),
        instance_entity=system.instance_entity.new_tensor(output_instance_entity),
    )


def _open_replacement(
    system: MolecularSystem,
    component: int,
    target: ChemicalComponent,
    polymer_neighbors: Mapping[int, set[int]],
) -> ChemicalComponent:
    r"""
    Build the smallest target component justified by an open source occurrence.
    """

    audited_atoms = {atom.atom_id: atom for atom in target.atoms}
    atom_ids: list[str] = []
    retained = set()
    for atom_id in rc.AMINO_ACID_ATOM14_NAMES.get(target.component_id, ()):
        if atom_id:
            atom_ids.append(atom_id)
            retained.add(atom_id)

    atom_start = int(system.component_atom_ptr[component])
    atom_stop = int(system.component_atom_ptr[component + 1])
    for symbol in system.atom_symbol[atom_start:atom_stop].detach().cpu().tolist():
        atom_id = system.symbols[symbol]
        if atom_id in audited_atoms and atom_id not in retained:
            atom_ids.append(atom_id)
            retained.add(atom_id)

    neighbors = polymer_neighbors.get(component, set())
    previous = target.previous_attachment if any(neighbor < component for neighbor in neighbors) else None
    following = target.next_attachment if any(neighbor > component for neighbor in neighbors) else None
    for attachment in (previous, following):
        if attachment is not None and attachment.atom_id not in retained:
            atom_ids.append(attachment.atom_id)
            retained.add(attachment.atom_id)

    return ChemicalComponent(
        component_id=target.component_id,
        atoms=tuple(
            ChemicalAtom(
                atom_id=atom_id,
                element=audited_atoms[atom_id].element,
            )
            for atom_id in atom_ids
        ),
        bonds=None,
        previous_attachment=(None if previous is None else PolymerAttachment(previous.atom_id)),
        next_attachment=(None if following is None else PolymerAttachment(following.atom_id)),
    )


def _supplement_incomplete_atom14(
    system: MolecularSystem,
) -> tuple[MolecularSystem, Tensor]:
    r"""
    Append missing standard atom14 heavy nodes to incomplete components.
    """

    additions: list[tuple[str, ...]] = []
    audited_components: list[ChemicalComponent | None] = []
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.component_symbol.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    protein_instance = system.entity_type[system.instance_entity] == ENTITY_PROTEIN
    for component in range(system.component_symbol.numel()):
        component_id = system.symbols[int(system.component_symbol[component])]
        if (
            not bool(protein_instance[component_instance[component]])
            or component_id not in AMINO_ACID_COMPONENT_IDS
            or bool(system.component_topology_complete[component])
        ):
            additions.append(())
            audited_components.append(None)
            continue
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        present = {system.symbols[symbol] for symbol in system.atom_symbol[start:stop].detach().cpu().tolist()}
        missing = tuple(
            atom_id for atom_id in rc.AMINO_ACID_ATOM14_NAMES[component_id] if atom_id and atom_id not in present
        )
        additions.append(missing)
        audited_components.append(None if not missing else get_component(component_id))

    added_count = sum(map(len, additions))
    if not added_count:
        return (
            system,
            torch.arange(
                system.atom_symbol.numel(),
                dtype=torch.int64,
                device=system.atom_symbol.device,
            ),
        )

    symbols = list(system.symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

    def intern(symbol: str) -> int:
        index = symbol_index.get(symbol)
        if index is None:
            index = len(symbols)
            symbols.append(symbol)
            symbol_index[symbol] = index
        return index

    old_count = system.atom_symbol.numel()
    new_count = old_count + added_count
    old_to_new = torch.empty(
        old_count,
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    component_atom_ptr = [0]
    added_atoms: list[tuple[int, str, int]] = []
    for component, missing in enumerate(additions):
        old_start = int(system.component_atom_ptr[component])
        old_stop = int(system.component_atom_ptr[component + 1])
        new_start = component_atom_ptr[-1]
        old_to_new[old_start:old_stop] = torch.arange(
            new_start,
            new_start + old_stop - old_start,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        audited = audited_components[component]
        if audited is not None:
            audited_atoms = {atom.atom_id: atom for atom in audited.atoms}
            for offset, atom_id in enumerate(missing):
                atom = audited_atoms[atom_id]
                added_atoms.append(
                    (
                        new_start + old_stop - old_start + offset,
                        atom_id,
                        ATOMIC_NUMBER_BY_ELEMENT[atom.element],
                    )
                )
        component_atom_ptr.append(new_start + old_stop - old_start + len(missing))

    retained_index = old_to_new

    def copied(field: Tensor, fill: int | float | bool = 0) -> Tensor:
        output = field.new_full((new_count, *field.shape[1:]), fill)
        output[retained_index] = field
        return output

    atom_symbol = copied(system.atom_symbol)
    atomic_number = copied(system.atomic_number)
    for index, atom_id, number in added_atoms:
        atom_symbol[index] = intern(atom_id)
        atomic_number[index] = number

    bond_stereo_index = torch.where(
        system.bond_stereo_index < 0,
        system.bond_stereo_index,
        old_to_new[system.bond_stereo_index.clamp_min(0)],
    )
    return (
        system.replace(  # type: ignore[attr-defined]
            symbols=tuple(symbols),
            component_atom_ptr=system.component_atom_ptr.new_tensor(component_atom_ptr),
            atom_symbol=atom_symbol,
            atomic_number=atomic_number,
            isotope=copied(system.isotope),
            isotope_known=copied(system.isotope_known),
            formal_charge=copied(system.formal_charge),
            formal_charge_known=copied(system.formal_charge_known),
            radical_electrons=copied(system.radical_electrons),
            radical_electrons_known=copied(system.radical_electrons_known),
            explicit_hydrogens=copied(system.explicit_hydrogens),
            explicit_hydrogens_known=copied(system.explicit_hydrogens_known),
            implicit_hydrogens=copied(system.implicit_hydrogens),
            implicit_hydrogens_known=copied(system.implicit_hydrogens_known),
            implicit_hydrogens_allowed=copied(system.implicit_hydrogens_allowed),
            implicit_hydrogens_allowed_known=copied(system.implicit_hydrogens_allowed_known),
            atom_aromatic=copied(system.atom_aromatic),
            atom_aromatic_known=copied(system.atom_aromatic_known),
            atom_stereo=copied(system.atom_stereo, -1),
            reference_coordinates=copied(system.reference_coordinates),
            reference_coordinate_mask=copied(system.reference_coordinate_mask),
            edge_index=old_to_new[system.edge_index],
            bond_stereo_index=bond_stereo_index,
        ),
        old_to_new,
    )


class MpnnPackerProcessor(Processor):
    r"""
    Prepare structures and target sequences for side-chain packing.

    Args:
        config: Side-chain context options. Defaults to ``MpnnPackerProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = MpnnPackerProcessorConfig
    config: MpnnPackerProcessorConfig

    def __init__(
        self,
        config: MpnnPackerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = MpnnPackerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def validate_model_input_layout(self, model: Model) -> None:
        model_config = cast(MpnnPackerConfig, model.config)
        if self.config.num_context_atoms != model_config.num_context_atoms:
            raise ValueError(
                "MPNN Packer Model and Processor must use the same num_context_atoms layout; "
                f"got model={model_config.num_context_atoms}, processor={self.config.num_context_atoms}"
            )

    def __call__(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        pack_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> MpnnPackerInput:
        r"""
        Prepare a structure and optional target sequence for side-chain packing.

        Args:
            structure: Coordinate-bearing source structure.
            sequence: Optional replacement protein sequence. A bare string requires one protein
                instance; a mapping addresses instances explicitly.
            pack_positions: Residue occurrences to repack. ``None`` selects every eligible
                protein residue.
            use_atom_context: Whether nonprotein heavy atoms are visible to the packer.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.

        Returns:
            A single-example ``MpnnPackerInput`` with a canonical target Structure.
        """

        if structure.system.system_instance_ptr.numel() != 2:
            raise ValueError("MpnnPackerProcessor requires one molecular system")
        structure, changed_components = self._realize_target(
            structure,
            sequence,
        )
        return self._lower(
            structure,
            changed_components,
            pack_positions=pack_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def _realize_target(
        self,
        structure: Structure,
        sequence: str | Mapping[str, str | Sequence[str]] | None,
    ) -> tuple[Structure, set[int]]:
        system = structure.system
        protein_instances = torch.nonzero(
            system.entity_type[system.instance_entity] == ENTITY_PROTEIN,
            as_tuple=False,
        ).flatten()
        if not protein_instances.numel():
            raise ValueError("MpnnPacker requires at least one protein instance")

        if isinstance(sequence, str):
            if protein_instances.numel() != 1:
                raise ValueError("a bare packing sequence requires exactly one protein instance")
            sequence = {system.instance_id(int(protein_instances[0])): sequence}

        target_component_ids = [
            system.symbols[int(symbol)] for symbol in system.component_symbol.detach().cpu().tolist()
        ]
        if sequence:
            for instance_id, value in sequence.items():
                instance = system.instance_index(instance_id)
                entity = int(system.instance_entity[instance])
                if int(system.entity_type[entity]) != ENTITY_PROTEIN:
                    raise ValueError(f"packing sequence instance {instance_id!r} is not a protein")
                start = int(system.instance_component_ptr[instance])
                stop = int(system.instance_component_ptr[instance + 1])
                requested = (
                    tuple(component.component_id for component in PolymerSequence.from_one_letter(value).components)
                    if isinstance(value, str)
                    else tuple(value)
                )
                if len(requested) != stop - start:
                    raise ValueError(
                        f"packing sequence for instance {instance_id!r} has length "
                        f"{len(requested)}, expected {stop - start}"
                    )
                target_component_ids[start:stop] = requested

        source_component_ids = [
            system.symbols[int(symbol)] for symbol in system.component_symbol.detach().cpu().tolist()
        ]
        changed_components = {
            component
            for component, (source, target) in enumerate(
                zip(
                    source_component_ids,
                    target_component_ids,
                    strict=True,
                )
            )
            if source != target
        }

        old_to_current = torch.arange(
            system.atom_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        if changed_components:
            system = _repartition_entities(
                system,
                target_component_ids,
            )
            atom_component = torch.repeat_interleave(
                torch.arange(
                    system.component_symbol.numel(),
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
                system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
            )
            polymer_neighbors: dict[int, set[int]] = {}
            polymer_edges = torch.nonzero(
                system.bond_role == BOND_ROLE_POLYMER,
                as_tuple=False,
            ).flatten()
            for left, right in atom_component[system.edge_index[:, polymer_edges]].T.detach().cpu().tolist():
                polymer_neighbors.setdefault(left, set()).add(right)
                polymer_neighbors.setdefault(right, set()).add(left)

            ordered_components = sorted(changed_components)
            open_components = {
                component for component in ordered_components if not bool(system.component_topology_complete[component])
            }
            component_edge = (
                (system.bond_role == BOND_ROLE_COMPONENT)
                & torch.isin(
                    atom_component[system.edge_index[0]],
                    system.component_symbol.new_tensor(
                        sorted(open_components),
                        dtype=torch.int64,
                    ),
                )
                & torch.isin(
                    atom_component[system.edge_index[1]],
                    system.component_symbol.new_tensor(
                        sorted(open_components),
                        dtype=torch.int64,
                    ),
                )
            )
            source_component_edges = torch.nonzero(
                component_edge,
                as_tuple=False,
            ).flatten()
            source_system = system
            replacements = []
            for component in ordered_components:
                target = get_component(target_component_ids[component])
                replacements.append(
                    target
                    if bool(system.component_topology_complete[component])
                    else _open_replacement(
                        system,
                        component,
                        target,
                        polymer_neighbors,
                    )
                )
            system, old_to_current, _ = replace_components(
                system,
                torch.tensor(
                    ordered_components,
                    dtype=torch.int64,
                    device=system.component_symbol.device,
                ),
                replacements,
            )
            if source_component_edges.numel():
                mapped_endpoints = old_to_current[source_system.edge_index[:, source_component_edges]]
                surviving = (mapped_endpoints >= 0).all(dim=0)
                source_component_edges = source_component_edges[surviving]
                mapped_endpoints = mapped_endpoints[:, surviving]
                if source_component_edges.numel():
                    swapped = mapped_endpoints[1] < mapped_endpoints[0]
                    mapped_endpoints = torch.sort(
                        mapped_endpoints,
                        dim=0,
                    ).values
                    direction = source_system.bond_direction[source_component_edges].clone()
                    direction[swapped] = -direction[swapped]

                    stereo_index = source_system.bond_stereo_index[
                        :,
                        source_component_edges,
                    ]
                    stereo_present = (stereo_index >= 0).all(dim=0)
                    mapped_stereo = torch.full_like(stereo_index, -1)
                    mapped_stereo[:, stereo_present] = old_to_current[stereo_index[:, stereo_present]]
                    stereo_retained = stereo_present & (mapped_stereo >= 0).all(dim=0)
                    mapped_stereo[:, ~stereo_retained] = -1
                    mapped_stereo[:, swapped] = mapped_stereo[[1, 0],][:, swapped]

                    symbols = list(system.symbols)
                    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
                    stereo = []
                    for source_stereo, retained_stereo in zip(
                        source_system.bond_stereo[source_component_edges].detach().cpu().tolist(),
                        stereo_retained.detach().cpu().tolist(),
                        strict=True,
                    ):
                        if source_stereo < 0 or not retained_stereo:
                            stereo.append(-1)
                            continue
                        descriptor = source_system.symbols[source_stereo]
                        index = symbol_index.get(descriptor)
                        if index is None:
                            index = len(symbols)
                            symbols.append(descriptor)
                            symbol_index[descriptor] = index
                        stereo.append(index)

                    edge_index = torch.cat(
                        (system.edge_index, mapped_endpoints),
                        dim=1,
                    )
                    fields = {
                        "bond_order": torch.cat(
                            (
                                system.bond_order,
                                source_system.bond_order[source_component_edges],
                            )
                        ),
                        "bond_order_known": torch.cat(
                            (
                                system.bond_order_known,
                                source_system.bond_order_known[source_component_edges],
                            )
                        ),
                        "bond_aromatic": torch.cat(
                            (
                                system.bond_aromatic,
                                source_system.bond_aromatic[source_component_edges],
                            )
                        ),
                        "bond_aromatic_known": torch.cat(
                            (
                                system.bond_aromatic_known,
                                source_system.bond_aromatic_known[source_component_edges],
                            )
                        ),
                        "bond_kind": torch.cat(
                            (
                                system.bond_kind,
                                source_system.bond_kind[source_component_edges],
                            )
                        ),
                        "bond_direction": torch.cat((system.bond_direction, direction)),
                        "bond_role": torch.cat(
                            (
                                system.bond_role,
                                source_system.bond_role[source_component_edges],
                            )
                        ),
                        "bond_stereo": torch.cat(
                            (
                                system.bond_stereo,
                                system.bond_stereo.new_tensor(stereo),
                            )
                        ),
                    }
                    bond_stereo_index = torch.cat(
                        (system.bond_stereo_index, mapped_stereo),
                        dim=1,
                    )
                    order = torch.argsort(edge_index[0] * system.atom_symbol.numel() + edge_index[1])
                    system = system.replace(  # type: ignore[attr-defined]
                        symbols=tuple(symbols),
                        edge_index=edge_index[:, order],
                        bond_stereo_index=bond_stereo_index[:, order],
                        **{name: value[order] for name, value in fields.items()},
                    )

        system, current_to_target = _supplement_incomplete_atom14(system)
        retained = old_to_current >= 0
        old_to_target = torch.full_like(old_to_current, -1)
        old_to_target[retained] = current_to_target[old_to_current[retained]]

        atom_count = system.atom_symbol.numel()

        def transferred(field: Tensor) -> Tensor:
            output = field.new_zeros((atom_count, *field.shape[1:]))
            output[old_to_target[retained]] = field[retained]
            return output

        standard_component_id = structure.standard_component_id.clone()
        standard_component_id[list(changed_components)] = -1
        structure = Structure(  # type: ignore[call-arg]
            system=system,
            coordinates=transferred(structure.coordinates),
            coordinate_mask=transferred(structure.coordinate_mask),
            occupancy=transferred(structure.occupancy),
            occupancy_mask=transferred(structure.occupancy_mask),
            b_factor=transferred(structure.b_factor),
            b_factor_mask=transferred(structure.b_factor_mask),
            source_symbols=structure.source_symbols,
            residue_id=structure.residue_id,
            insertion_code=structure.insertion_code,
            standard_component_id=standard_component_id,
            batch_size=[],
        )
        return structure, changed_components

    def _lower(
        self,
        structure: Structure,
        changed_components: set[int],
        *,
        pack_positions: Sequence[ComponentReference] | None,
        use_atom_context: bool,
        filter_by_occupancy: bool,
    ) -> MpnnPackerInput:
        system = structure.system
        device = system.atom_symbol.device
        observed = structure.coordinate_mask
        if filter_by_occupancy:
            observed = observed & (~structure.occupancy_mask | (structure.occupancy > 0))

        component_count = system.component_symbol.numel()
        component_instance = torch.repeat_interleave(
            torch.arange(
                system.instance_symbol.numel(),
                dtype=torch.int64,
                device=device,
            ),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        atom_component = torch.repeat_interleave(
            torch.arange(
                component_count,
                dtype=torch.int64,
                device=device,
            ),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        atom_instance = component_instance[atom_component]
        protein_instance = system.entity_type[system.instance_entity] == ENTITY_PROTEIN
        symbol_index = {symbol: index for index, symbol in enumerate(system.symbols)}
        ca_symbol = symbol_index.get("CA")
        row_atom = (
            system.atom_symbol.new_empty(0)
            if ca_symbol is None
            else torch.nonzero(
                observed & protein_instance[atom_instance] & (system.atom_symbol == ca_symbol),
                as_tuple=False,
            ).flatten()
        )
        row_component = atom_component[row_atom]
        if not row_component.numel():
            raise ValueError("MpnnPacker requires at least one protein residue with an observed CA")

        length = row_component.numel()
        row_by_component = torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        row_by_component[row_component] = torch.arange(
            length,
            dtype=torch.int64,
            device=device,
        )
        atom37 = structure.coordinates.new_zeros((length, len(rc.atom_types), 3))
        atom37_mask = torch.zeros(
            (length, len(rc.atom_types)),
            dtype=torch.bool,
            device=device,
        )
        for atom37_index, atom_id in enumerate(rc.atom_types):
            if atom_id == "OXT" or atom_id not in symbol_index:
                continue
            matching_atom = torch.nonzero(
                (system.atom_symbol == symbol_index[atom_id]) & (row_by_component[atom_component] >= 0),
                as_tuple=False,
            ).flatten()
            matching_row = row_by_component[atom_component[matching_atom]]
            atom37[matching_row, atom37_index] = structure.coordinates[matching_atom]
            atom37_mask[matching_row, atom37_index] = observed[matching_atom]

        backbone_index = torch.tensor(
            _BACKBONE_ATOM37_INDICES,
            dtype=torch.int64,
            device=device,
        )
        backbone = atom37[:, backbone_index]
        valid = atom37_mask[:, backbone_index].all(dim=-1)
        row_instance = component_instance[row_component]
        chain_by_instance: dict[int, int] = {}
        chain_index = []
        for instance in row_instance.detach().cpu().tolist():
            if instance not in chain_by_instance:
                chain_by_instance[instance] = len(chain_by_instance)
            chain_index.append(chain_by_instance[instance])

        source_id = structure.residue_id[row_component]
        has_source_id = source_id >= 0
        if not bool(has_source_id.any()):
            residue_index = row_component.detach().cpu().tolist()
        elif bool(has_source_id.all()):
            source_numbers = [int(structure.source_symbols[index]) for index in source_id.detach().cpu().tolist()]
            residue_index = []
            duplicate_count = 0
            previous_number: int | None = None
            for number in source_numbers:
                if number == previous_number:
                    duplicate_count += 1
                residue_index.append(number + duplicate_count)
                previous_number = number
        else:
            raise ValueError("MpnnPacker requires source residue IDs for every retained row or none")

        component_ids = [
            system.symbols[index] for index in system.component_symbol[row_component].detach().cpu().tolist()
        ]
        unresolved: list[tuple[str, ...]] = []
        for component, component_id in zip(
            row_component.detach().cpu().tolist(),
            component_ids,
            strict=True,
        ):
            if component_id not in AMINO_ACID_COMPONENT_IDS:
                unresolved.append(())
                continue
            start = int(system.component_atom_ptr[component])
            stop = int(system.component_atom_ptr[component + 1])
            atom_ids = {system.symbols[index] for index in system.atom_symbol[start:stop].detach().cpu().tolist()}
            unresolved.append(
                tuple(
                    atom_id
                    for atom_id in rc.AMINO_ACID_ATOM14_NAMES[component_id]
                    if atom_id and atom_id not in atom_ids
                )
            )

        row_components = row_component.detach().cpu().tolist()
        if pack_positions is None:
            selected = [
                bool(is_valid) and component_id in AMINO_ACID_COMPONENT_IDS
                for is_valid, component_id in zip(
                    valid.detach().cpu().tolist(),
                    component_ids,
                    strict=True,
                )
            ]
        else:
            requested = {system.component_index(reference) for reference in pack_positions}
            unknown = requested - set(row_components)
            if unknown:
                references = sorted(system.component_reference(component) for component in unknown)
                raise ValueError(f"packing positions are outside the observed protein layout: {references!r}")
            selected = [component in requested for component in row_components]

        for component, component_id, is_selected, is_valid, missing in zip(
            row_components,
            component_ids,
            selected,
            valid.detach().cpu().tolist(),
            unresolved,
            strict=True,
        ):
            if not is_selected:
                continue
            reference = system.component_reference(component)
            if component_id not in AMINO_ACID_COMPONENT_IDS:
                raise ValueError(f"MpnnPacker cannot pack residue identity {component_id!r} at {reference!r}")
            if missing:
                raise ValueError(
                    f"MpnnPacker cannot pack residue {reference!r} "
                    f"because expected atom14 sites do not resolve: {missing!r}"
                )
            if not is_valid:
                raise ValueError(f"MpnnPacker cannot pack residue {reference!r} without complete backbone coordinates")

        selected_components = {
            component
            for component, is_selected in zip(
                row_components,
                selected,
                strict=True,
            )
            if is_selected
        }
        unselected_changes = changed_components - selected_components
        if unselected_changes:
            references = sorted(system.component_reference(component) for component in unselected_changes)
            raise ValueError(f"changed identities must be selected for packing: {references!r}")

        backbone_tensor = backbone.to(
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        valid_tensor = valid.to(device=self.device).unsqueeze(0)
        num_context_atoms = self.config.num_context_atoms
        context_coordinates = torch.zeros(
            length,
            num_context_atoms,
            3,
            device=self.device,
            dtype=self.dtype,
        )
        context_atomic_numbers = torch.zeros(
            length,
            num_context_atoms,
            device=self.device,
            dtype=torch.int64,
        )
        context_mask = torch.zeros(
            length,
            num_context_atoms,
            device=self.device,
            dtype=torch.bool,
        )
        if use_atom_context:
            retained_context = (
                observed
                & ~protein_instance[atom_instance]
                & ~water_component_mask(system)[atom_component]
                & (system.atomic_number != 1)
            )
            if bool(retained_context.any()):
                context = structure.coordinates[retained_context].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                context_type = system.atomic_number[retained_context].to(device=self.device, dtype=torch.int64)
                n = backbone_tensor[0, :, 0]
                ca = backbone_tensor[0, :, 1]
                c = backbone_tensor[0, :, 2]
                cb = virtual_beta_carbon(n, ca, c)
                squared_distance = ((cb[:, None] - context[None]) ** 2).sum(dim=-1)
                pair_mask = valid_tensor[0, :, None]
                adjusted_distance = torch.where(
                    pair_mask,
                    squared_distance,
                    torch.full_like(squared_distance, 1000.0),
                )
                atom_count = min(num_context_atoms, context.shape[0])
                nearest = torch.topk(
                    adjusted_distance,
                    atom_count,
                    dim=-1,
                    largest=False,
                    sorted=True,
                ).indices
                context_coordinates[:, :atom_count] = context[nearest]
                context_atomic_numbers[:, :atom_count] = context_type[nearest]
                context_mask[:, :atom_count] = True

        return MpnnPackerInput(  # type: ignore[call-arg]
            backbone_coordinates=backbone_tensor,
            sequence_tokens=torch.tensor(
                [
                    AMINO_ACID_TOKEN_BY_COMPONENT.get(
                        component_id,
                        AMINO_ACID_UNKNOWN_TOKEN,
                    )
                    for component_id in component_ids
                ],
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            residue_mask=valid_tensor,
            residue_index=torch.tensor(
                residue_index,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            chain_index=torch.tensor(
                chain_index,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            context_coordinates=context_coordinates.unsqueeze(0),
            context_atomic_numbers=context_atomic_numbers.unsqueeze(0),
            context_mask=context_mask.unsqueeze(0),
            atom37_coordinates=atom37.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            atom37_mask=atom37_mask.to(device=self.device).unsqueeze(0),
            pack_positions=torch.tensor(
                selected,
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(0),
            structure=structure,
            component_index=row_component.to(
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            batch_size=[],
        )
