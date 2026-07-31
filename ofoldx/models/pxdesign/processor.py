r"""
Input processing for PXDesign target-binder generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from ...data import (
    ExistingSpan,
    GeneratedSegment,
    PlacementSpec,
    validate_placement,
)
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.molecular import (
    BOND_ROLE_COMPONENT,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    copy_components,
    disjoint_union,
)
from ...data.structure import Structure
from ...processor import Processor
from .model import PxDesignInput

_RESIDUE_TYPE_INDEX = {
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
_GENERATED_RESIDUE_TYPE = 32
_GENERATED_ATOM_IDS = frozenset(atom.atom_id for atom in get_component("GLY").atoms if atom.element != "H")


class PxDesignProcessor(Processor):
    r"""
    Convert one target-binder placement into tensor-native PXDesign input.
    """

    def __call__(
        self,
        input: MolecularSystem | Structure | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
    ) -> PxDesignInput:
        r"""
        Materialize one canonical design graph and its model tensors.

        Args:
            input: Optional target system or observed target structure.
            placement: Target groups followed by one fixed-length protein binder.
            conditions: Existing target components labelled as hotspots.

        Returns:
            A tensor-native [`PxDesignInput`][] with explicit model batch dimension
            one and an empty tensorclass batch size.
        """

        if isinstance(input, Structure):
            structure: Structure | None = input
            source: MolecularSystem | None = input.system
        else:
            structure = None
            source = input

        target_groups, binder_length = _placement_groups(placement, source)
        if target_groups and structure is None:
            raise ValueError("PXDesign target conditioning requires observed Structure coordinates")
        if source is not None and source.system_entity_ptr.numel() != 2:
            raise ValueError("PXDesign processes one molecular system at a time")

        (
            system,
            source_atom_index,
            source_component_index,
        ) = _materialize_problem(source, target_groups, binder_length)
        atom_index = _model_atom_index(
            system,
            source_atom_index,
            structure,
            source_component_index,
        )
        (
            token_component_index,
            token_atom_index,
            representative_atom_index,
            atom_to_token,
            residue_type_features,
        ) = _tokenize(system, atom_index, source_component_index)

        hotspot = _hotspot_features(
            tuple(conditions),
            source,
            source_component_index,
            token_component_index,
        )
        reference_positions, reference_mask, reference_space_id = _reference_features(
            system,
            atom_index,
            source_component_index,
        )
        atom_features, reference_charges = _atom_features(
            system,
            atom_index,
            reference_mask,
        )
        template_distance_bins, template_pair_mask = _template_features(
            structure,
            source_atom_index,
            representative_atom_index,
        )
        chain_index, entity_index, symmetry_index, residue_index = _token_indices(
            system,
            token_component_index,
        )
        token_count = token_component_index.numel()

        return PxDesignInput(  # type: ignore[call-arg]
            system=system.to(self.device),
            atom_index=atom_index.unsqueeze(0).to(self.device),
            token_component_index=token_component_index.unsqueeze(0).to(self.device),
            token_atom_index=token_atom_index.unsqueeze(0).to(self.device),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(
                token_count,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0),
            atom_features=atom_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(
                1,
                atom_index.numel(),
                dtype=torch.bool,
                device=self.device,
            ),
            reference_positions=reference_positions.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=reference_charges.reshape(
                1,
                atom_index.numel(),
                1,
            ).to(device=self.device, dtype=self.dtype),
            residue_type_features=residue_type_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            plddt=torch.zeros(
                1,
                token_count,
                1,
                dtype=self.dtype,
                device=self.device,
            ),
            hotspot=hotspot.reshape(1, token_count, 1).to(
                device=self.device,
                dtype=self.dtype,
            ),
            template_distance_bins=template_distance_bins.unsqueeze(0).to(self.device),
            template_pair_mask=template_pair_mask.unsqueeze(0).to(self.device),
            batch_size=[],
        )


def _placement_groups(
    placement: PlacementSpec,
    source: MolecularSystem | None,
) -> tuple[
    tuple[tuple[str, dict[str, ExistingSpan], int, tuple[int, ...]], ...],
    int,
]:
    r"""
    Resolve the family-owned target groups without retaining them in Model input.
    """

    binder_length, target_specs = to_binder(placement)
    groups = tuple(placement.items())

    targets = []
    target_source_instances: set[int] = set()
    for (group, values), (instance_id, crops) in zip(
        groups[:-1],
        target_specs,
        strict=True,
    ):
        spans = {placement_id: cast(ExistingSpan, value) for placement_id, value in values.items()}
        if source is None:
            raise ValueError("ExistingSpan target groups require molecular input")
        source_instance = source.instance_index(instance_id)
        if source_instance in target_source_instances:
            raise ValueError("one PXDesign source instance cannot define two target groups")
        target_source_instances.add(source_instance)
        component_start = int(source.instance_component_ptr[source_instance].item())
        component_stop = int(source.instance_component_ptr[source_instance + 1].item())
        component_count = component_stop - component_start
        selected = sorted({component for start, stop in crops for component in range(start, stop)})
        if any(start < 0 or start >= stop or stop > component_count for start, stop in crops):
            raise ValueError(f"PXDesign target group {group!r} has a crop outside its source instance")
        targets.append(
            (
                group,
                spans,
                source_instance,
                tuple(selected),
            )
        )

    return tuple(targets), binder_length


def _materialize_problem(
    source: MolecularSystem | None,
    target_groups: tuple[tuple[str, dict[str, ExistingSpan], int, tuple[int, ...]], ...],
    binder_length: int,
) -> tuple[MolecularSystem, Tensor, Tensor]:
    r"""
    Build the sole canonical output-problem graph.

    The returned source maps are transient Processor evidence. They are consumed
    while lowering coordinates and never enter [`PxDesignInput`][].
    """

    source_instance_ids = (
        set() if source is None else {source.instance_id(index) for index in range(source.instance_symbol.numel())}
    )
    source_entity_ids = (
        set() if source is None else {source.entity_id(index) for index in range(source.entity_symbol.numel())}
    )
    used_instances: set[str] = set()
    used_entities: set[str] = set()
    next_instance = 0
    next_entity = 0

    def instance_id() -> str:
        nonlocal next_instance
        while True:
            value = f"chain_{next_instance}"
            next_instance += 1
            if value not in source_instance_ids and value not in used_instances:
                used_instances.add(value)
                return value

    def entity_id() -> str:
        nonlocal next_entity
        while True:
            value = f"pxdesign_entity_{next_entity}"
            next_entity += 1
            if value not in source_entity_ids and value not in used_entities:
                used_entities.add(value)
                return value

    pieces = []
    source_atom_maps = []
    source_component_maps = []
    group_maps = []
    for group_index, (_, _, source_instance, selected) in enumerate(target_groups):
        assert source is not None
        piece, atom_map, component_map = _target_piece(
            source,
            source_instance,
            selected,
            entity_id(),
            instance_id(),
        )
        pieces.append(piece)
        source_atom_maps.append(atom_map)
        source_component_maps.append(component_map)
        group_maps.append(torch.full_like(atom_map, group_index, dtype=torch.int64))

    binder = MolecularSystem.from_entities(
        {
            (entity := entity_id()): PolymerSequence(
                "protein",
                (get_component("UNK"),) * binder_length,
            )
        },
        {instance_id(): entity},
        device=None if source is None else source.atom_symbol.device,
    )
    pieces.append(binder)
    source_atom_maps.append(torch.full_like(binder.atom_symbol, -1, dtype=torch.int64))
    source_component_maps.append(torch.full_like(binder.component_symbol, -1, dtype=torch.int64))
    group_maps.append(torch.full_like(binder.atom_symbol, -1, dtype=torch.int64))

    system, component_maps, atom_maps = disjoint_union(pieces)
    source_atom_index = torch.full_like(
        system.atom_symbol,
        -1,
        dtype=torch.int64,
    )
    source_component_index = torch.full_like(
        system.component_symbol,
        -1,
        dtype=torch.int64,
    )
    output_group_index = torch.full_like(
        system.atom_symbol,
        -1,
        dtype=torch.int64,
    )
    for atom_map, source_map, group_map in zip(
        atom_maps,
        source_atom_maps,
        group_maps,
        strict=True,
    ):
        source_atom_index[atom_map] = source_map
        output_group_index[atom_map] = group_map
    for component_map, source_map in zip(
        component_maps,
        source_component_maps,
        strict=True,
    ):
        source_component_index[component_map] = source_map
    if source is not None and target_groups:
        system = _restore_source_edges(
            system,
            source,
            source_atom_index,
            output_group_index,
        )
    return (
        system,
        source_atom_index,
        source_component_index,
    )


def _target_piece(
    source: MolecularSystem,
    source_instance: int,
    selected_local_components: tuple[int, ...],
    entity_id: str,
    instance_id: str,
) -> tuple[MolecularSystem, Tensor, Tensor]:
    r"""
    Copy one target group's selected components without reconstructing objects.
    """

    device = source.atom_symbol.device
    source_component_start = int(source.instance_component_ptr[source_instance].item())
    component_index = torch.tensor(
        [source_component_start + local for local in selected_local_components],
        dtype=torch.int64,
        device=device,
    )
    placeholder = MolecularSystem.from_entities(
        {entity_id: Molecule((get_component("UNK"),) * len(selected_local_components))},
        {instance_id: entity_id},
        device=device,
    )
    target_component_index = torch.arange(
        len(selected_local_components),
        dtype=torch.int64,
        device=device,
    )
    piece, _, source_to_new = copy_components(
        placeholder,
        target_component_index,
        source,
        component_index,
    )
    source_entity = int(source.instance_entity[source_instance].item())
    piece = piece.replace(  # type: ignore[attr-defined]
        entity_type=source.entity_type[source_entity : source_entity + 1],
        entity_molecule_type=source.entity_molecule_type[source_entity : source_entity + 1],
    )

    symbols = list(piece.symbols)

    def symbol(value: str) -> int:
        try:
            return symbols.index(value)
        except ValueError:
            symbols.append(value)
            return len(symbols) - 1

    component_symbol = piece.component_symbol.clone()
    atom_symbol = piece.atom_symbol.clone()
    atomic_number = piece.atomic_number.clone()
    for output_component, source_component in enumerate(component_index.tolist()):
        component_id = str(source.symbols[int(source.component_symbol[source_component].item())])
        if component_id != "MSE":
            continue
        component_symbol[output_component] = symbol("MET")
        for source_atom in range(
            int(source.component_atom_ptr[source_component].item()),
            int(source.component_atom_ptr[source_component + 1].item()),
        ):
            atom_id = str(source.symbols[int(source.atom_symbol[source_atom].item())])
            if atom_id != "SE":
                continue
            output_atom = int(source_to_new[source_atom].item())
            atom_symbol[output_atom] = symbol("SD")
            atomic_number[output_atom] = 16
    piece = piece.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        component_symbol=component_symbol,
        atom_symbol=atom_symbol,
        atomic_number=atomic_number,
    )

    source_atom_index = torch.full(
        (piece.atom_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=device,
    )
    copied_source_atom = torch.nonzero(source_to_new >= 0, as_tuple=True)[0]
    source_atom_index[source_to_new[copied_source_atom]] = copied_source_atom
    return piece, source_atom_index, component_index


def _restore_source_edges(
    system: MolecularSystem,
    source: MolecularSystem,
    source_atom_index: Tensor,
    output_group_index: Tensor,
) -> MolecularSystem:
    r"""
    Remap every surviving source edge onto its copied target occurrence.
    """

    source_atom_component = _atom_component(source)
    source_component_instance = _component_instance(source)
    source_atom_instance = source_component_instance[source_atom_component]
    output_by_source_group = {
        (int(source_atom), int(group)): output_atom
        for output_atom, (source_atom, group) in enumerate(
            zip(
                source_atom_index.tolist(),
                output_group_index.tolist(),
                strict=True,
            )
        )
        if source_atom >= 0
    }
    groups_by_source: dict[int, list[int]] = {}
    for source_atom, group in output_by_source_group:
        groups_by_source.setdefault(source_atom, []).append(group)

    edge_rows = []
    source_edge_rows = []
    stereo_rows = []
    directions = []
    for edge, (source_a, source_b) in enumerate(source.edge_index.transpose(0, 1).tolist()):
        if int(source.bond_role[edge].item()) == BOND_ROLE_COMPONENT:
            continue
        groups_a = groups_by_source.get(source_a, ())
        groups_b = groups_by_source.get(source_b, ())
        if int(source_atom_instance[source_a].item()) == int(source_atom_instance[source_b].item()):
            group_pairs = ((group, group) for group in set(groups_a) & set(groups_b))
        else:
            group_pairs = ((group_a, group_b) for group_a in groups_a for group_b in groups_b)
        for group_a, group_b in group_pairs:
            output_a = output_by_source_group[(source_a, group_a)]
            output_b = output_by_source_group[(source_b, group_b)]
            swapped = output_a > output_b
            edge_rows.append((output_b, output_a) if swapped else (output_a, output_b))
            source_edge_rows.append(edge)
            direction = int(source.bond_direction[edge].item())
            directions.append(-direction if swapped else direction)

            stereo = []
            for stereo_atom in source.bond_stereo_index[:, edge].tolist():
                if stereo_atom < 0:
                    stereo.append(-1)
                    continue
                stereo_instance = int(source_atom_instance[stereo_atom].item())
                group = group_a if stereo_instance == int(source_atom_instance[source_a].item()) else group_b
                stereo.append(output_by_source_group.get((stereo_atom, group), -1))
            stereo_rows.append(tuple(stereo))

    if not edge_rows:
        return system

    source_rows = torch.tensor(
        source_edge_rows,
        dtype=torch.int64,
        device=source.edge_index.device,
    )
    copied_edge_index = torch.tensor(
        edge_rows,
        dtype=torch.int64,
        device=system.edge_index.device,
    ).transpose(0, 1)
    copied_stereo_index = torch.tensor(
        stereo_rows,
        dtype=torch.int64,
        device=system.edge_index.device,
    ).transpose(0, 1)
    copied_bond_stereo = source.bond_stereo[source_rows].clone()
    symbols = list(system.symbols)
    for index, source_symbol in enumerate(copied_bond_stereo.tolist()):
        if source_symbol < 0:
            continue
        value = source.symbols[source_symbol]
        try:
            copied_bond_stereo[index] = symbols.index(value)
        except ValueError:
            symbols.append(value)
            copied_bond_stereo[index] = len(symbols) - 1
        if (copied_stereo_index[:, index] < 0).any():
            copied_bond_stereo[index] = -1

    edge_index = torch.cat((system.edge_index, copied_edge_index), dim=1)
    bond_order = torch.cat((system.bond_order, source.bond_order[source_rows]))
    bond_order_known = torch.cat((system.bond_order_known, source.bond_order_known[source_rows]))
    bond_aromatic = torch.cat((system.bond_aromatic, source.bond_aromatic[source_rows]))
    bond_aromatic_known = torch.cat(
        (
            system.bond_aromatic_known,
            source.bond_aromatic_known[source_rows],
        )
    )
    bond_kind = torch.cat((system.bond_kind, source.bond_kind[source_rows]))
    bond_direction = torch.cat(
        (
            system.bond_direction,
            torch.tensor(
                directions,
                dtype=system.bond_direction.dtype,
                device=system.bond_direction.device,
            ),
        )
    )
    bond_role = torch.cat((system.bond_role, source.bond_role[source_rows]))
    bond_stereo = torch.cat((system.bond_stereo, copied_bond_stereo))
    bond_stereo_index = torch.cat(
        (system.bond_stereo_index, copied_stereo_index),
        dim=1,
    )
    order = (edge_index[0] * (system.atom_symbol.numel() + 1) + edge_index[1]).argsort()
    return system.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        edge_index=edge_index[:, order],
        bond_order=bond_order[order],
        bond_order_known=bond_order_known[order],
        bond_aromatic=bond_aromatic[order],
        bond_aromatic_known=bond_aromatic_known[order],
        bond_kind=bond_kind[order],
        bond_direction=bond_direction[order],
        bond_role=bond_role[order],
        bond_stereo=bond_stereo[order],
        bond_stereo_index=bond_stereo_index[:, order],
    )


def _model_atom_index(
    system: MolecularSystem,
    source_atom_index: Tensor,
    structure: Structure | None,
    source_component_index: Tensor,
) -> Tensor:
    atom_component = _atom_component(system)
    observed_atom = torch.zeros(
        system.atom_symbol.numel(),
        dtype=torch.bool,
        device=system.atom_symbol.device,
    )
    if structure is not None:
        copied = source_atom_index >= 0
        observed_atom[copied] = structure.coordinate_mask[source_atom_index[copied]]

    atom_names = [system.symbols[int(symbol)] for symbol in system.atom_symbol.tolist()]
    component_instance = _component_instance(system)
    component_type = system.entity_type[system.instance_entity[component_instance]]
    component_ids = [system.symbols[int(symbol)] for symbol in system.component_symbol.tolist()]
    generated_atom = torch.tensor(
        [
            source_component_index[int(component)] < 0 and name in _GENERATED_ATOM_IDS
            for component, name in zip(
                atom_component.tolist(),
                atom_names,
                strict=True,
            )
        ],
        dtype=torch.bool,
        device=system.atom_symbol.device,
    )
    standard_representative = torch.tensor(
        [
            int(component_type[int(component)].item()) != ENTITY_MOLECULE
            and component_ids[int(component)] in _RESIDUE_TYPE_INDEX
            and name == ("CA" if int(component_type[int(component)].item()) == ENTITY_PROTEIN else "C1'")
            for component, name in zip(
                atom_component.tolist(),
                atom_names,
                strict=True,
            )
        ],
        dtype=torch.bool,
        device=system.atom_symbol.device,
    )
    selected = (generated_atom | observed_atom | standard_representative) & (system.atomic_number != 1)
    atom_index = torch.nonzero(selected, as_tuple=True)[0]
    if atom_index.numel() == 0:
        raise ValueError("PXDesign requires at least one modeled atom")
    return atom_index


def _tokenize(
    system: MolecularSystem,
    atom_index: Tensor,
    source_component_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    atom_component = _atom_component(system)
    component_instance = _component_instance(system)
    component_entity = system.instance_entity[component_instance]
    component_type = system.entity_type[component_entity]
    model_component = atom_component[atom_index]
    model_rows_by_component: dict[int, list[int]] = {}
    for model_atom, component in enumerate(model_component.tolist()):
        model_rows_by_component.setdefault(component, []).append(model_atom)

    token_components = []
    token_atoms = []
    representatives = []
    token_model_atoms = []
    residue_feature_indices = []
    for component in range(system.component_symbol.numel()):
        model_rows = model_rows_by_component.get(component)
        if not model_rows:
            continue
        component_id = system.symbols[int(system.component_symbol[component].item())]
        generated = int(source_component_index[component].item()) < 0
        standard_polymer = int(component_type[component].item()) != ENTITY_MOLECULE and (
            generated or component_id in _RESIDUE_TYPE_INDEX
        )
        representative_id = "CA" if int(component_type[component].item()) == ENTITY_PROTEIN else "C1'"
        representative = next(
            (
                int(atom_index[model_row].item())
                for model_row in model_rows
                if system.symbols[int(system.atom_symbol[int(atom_index[model_row].item())].item())]
                == representative_id
            ),
            None,
        )
        if standard_polymer:
            if representative is None:
                raise ValueError(f"standard polymer component {component_id!r} " f"has no {representative_id} atom")
            token_components.append(component)
            token_atoms.append(-1)
            representatives.append(representative)
            token_model_atoms.append(model_rows)
            residue_feature_indices.append(_GENERATED_RESIDUE_TYPE if generated else _RESIDUE_TYPE_INDEX[component_id])
            continue

        unknown = _unknown_residue_type(int(component_type[component].item()))
        for model_row in model_rows:
            canonical_atom = int(atom_index[model_row].item())
            token_components.append(component)
            token_atoms.append(canonical_atom)
            representatives.append(canonical_atom)
            token_model_atoms.append([model_row])
            residue_feature_indices.append(unknown)

    token_count = len(token_components)
    atom_to_token = torch.empty_like(atom_index)
    for token, model_atoms in enumerate(token_model_atoms):
        atom_to_token[torch.tensor(model_atoms, device=atom_index.device)] = token
    residue_type_features = torch.zeros(
        token_count,
        36,
        dtype=torch.float32,
        device=atom_index.device,
    )
    residue_type_features[
        torch.arange(token_count, device=atom_index.device),
        torch.tensor(residue_feature_indices, device=atom_index.device),
    ] = 1
    return (
        torch.tensor(
            token_components,
            dtype=torch.int64,
            device=atom_index.device,
        ),
        torch.tensor(
            token_atoms,
            dtype=torch.int64,
            device=atom_index.device,
        ),
        torch.tensor(
            representatives,
            dtype=torch.int64,
            device=atom_index.device,
        ),
        atom_to_token,
        residue_type_features,
    )


def _unknown_residue_type(entity_type: int) -> int:
    if entity_type == ENTITY_PROTEIN:
        return _RESIDUE_TYPE_INDEX["UNK"]
    # Canonical tensor entity codes are stable: RNA=2, DNA=3, hybrid=4.
    if entity_type == 2:
        return _RESIDUE_TYPE_INDEX["N"]
    if entity_type in (3, 4):
        return _RESIDUE_TYPE_INDEX["DN"]
    return _RESIDUE_TYPE_INDEX["UNK"]


def _hotspot_features(
    conditions: tuple[Hotspot, ...],
    source: MolecularSystem | None,
    source_component_index: Tensor,
    token_component_index: Tensor,
) -> Tensor:
    selected_source_components = {int(component) for component in source_component_index.tolist() if component >= 0}
    hotspots: set[int] = set()
    for condition in conditions:
        for reference in condition.residues:
            if source is None or not isinstance(reference, tuple) or len(reference) != 2:
                raise ValueError("PXDesign hotspots name existing target components")
            source_component = source.component_index(cast(tuple[str, int], reference))
            if source_component not in selected_source_components:
                raise ValueError(f"PXDesign hotspot is absent from the target crop: {reference!r}")
            hotspots.add(source_component)
    return torch.tensor(
        [
            int(source_component_index[int(component)].item()) in hotspots
            for component in token_component_index.tolist()
        ],
        dtype=torch.float32,
        device=token_component_index.device,
    )


def _reference_features(
    system: MolecularSystem,
    atom_index: Tensor,
    source_component_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    glycine = get_component("GLY")
    glycine_coordinates = {
        atom.atom_id: coordinate
        for atom, coordinate in zip(
            glycine.atoms,
            glycine.reference_coordinates,
            strict=True,
        )
        if coordinate is not None
    }
    atom_component = _atom_component(system)
    component_instance = _component_instance(system)
    positions = system.reference_coordinates[atom_index].clone()
    mask = system.reference_coordinate_mask[atom_index].clone()
    spaces: dict[tuple[int, int], int] = {}
    space_ids = []
    for model_atom, canonical_atom in enumerate(atom_index.tolist()):
        component = int(atom_component[canonical_atom].item())
        instance = int(component_instance[component].item())
        entity = int(system.instance_entity[instance].item())
        if int(source_component_index[component].item()) < 0:
            atom_id = system.symbols[int(system.atom_symbol[canonical_atom].item())]
            coordinate = glycine_coordinates.get(atom_id)
            if coordinate is None:
                mask[model_atom] = False
            else:
                positions[model_atom] = positions.new_tensor(coordinate)
                mask[model_atom] = True
        space = (instance, -1) if int(system.entity_type[entity].item()) == ENTITY_MOLECULE else (instance, component)
        if space not in spaces:
            spaces[space] = len(spaces)
        space_ids.append(spaces[space])

    space_id = torch.tensor(
        space_ids,
        dtype=torch.int64,
        device=atom_index.device,
    )
    for value in range(len(spaces)):
        selected = (space_id == value) & mask
        if selected.any():
            positions[selected] -= positions[selected].mean(dim=0)
    return positions, mask, space_id


def _atom_features(
    system: MolecularSystem,
    atom_index: Tensor,
    reference_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    element = torch.zeros(
        atom_index.numel(),
        128,
        device=atom_index.device,
    )
    element[
        torch.arange(atom_index.numel(), device=atom_index.device),
        system.atomic_number[atom_index].to(torch.int64) - 1,
    ] = 1
    names = torch.zeros(
        atom_index.numel(),
        4,
        64,
        device=atom_index.device,
    )
    for row, canonical_atom in enumerate(atom_index.tolist()):
        atom_id = system.symbols[int(system.atom_symbol[canonical_atom].item())]
        if len(atom_id) > 4:
            raise ValueError(f"PXDesign cannot encode atom name longer than four characters: {atom_id!r}")
        for column, character in enumerate(atom_id.ljust(4)):
            value = ord(character) - 32
            if not 0 <= value < 64:
                raise ValueError(f"PXDesign cannot encode atom-name character {character!r}")
            names[row, column, value] = 1
    features = torch.cat(
        (
            reference_mask.to(torch.float32).unsqueeze(-1),
            element,
            names.flatten(-2),
        ),
        dim=-1,
    )
    return features, system.formal_charge[atom_index].to(torch.float32)


def _template_features(
    structure: Structure | None,
    source_atom_index: Tensor,
    representative_atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    token_count = representative_atom_index.numel()
    positions = torch.zeros(
        token_count,
        3,
        dtype=torch.float32,
        device=representative_atom_index.device,
    )
    mask = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=representative_atom_index.device,
    )
    if structure is not None:
        source_atoms = source_atom_index[representative_atom_index]
        selected = source_atoms >= 0
        selected_rows = torch.nonzero(selected, as_tuple=True)[0]
        selected_source = source_atoms[selected]
        valid = structure.coordinate_mask[selected_source]
        valid_rows = selected_rows[valid]
        valid_source = selected_source[valid]
        positions[valid_rows] = structure.coordinates[valid_source].to(positions)
        mask[valid_rows] = True

    bins = torch.zeros(
        token_count,
        token_count,
        dtype=torch.int64,
        device=positions.device,
    )
    pair_mask = mask[:, None] & mask[None, :]
    rows = torch.nonzero(mask, as_tuple=True)[0]
    if rows.numel():
        distances = torch.cdist(
            positions.index_select(0, rows),
            positions.index_select(0, rows),
        )
        values = torch.sum(
            distances.unsqueeze(-1)
            > torch.linspace(
                2.0,
                22.0,
                63,
                device=positions.device,
            ),
            dim=-1,
        )
        first, second = torch.meshgrid(rows, rows, indexing="ij")
        bins[first, second] = values
    return bins, pair_mask


def _token_indices(
    system: MolecularSystem,
    token_component_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    component_instance = _component_instance(system)
    instance = component_instance[token_component_index]
    entity = system.instance_entity[instance]
    residue = token_component_index - system.instance_component_ptr[instance]
    # PXDesign deliberately repartitions every target group into its own entity.
    symmetry = torch.zeros_like(instance)
    return instance, entity, symmetry, residue


def _atom_component(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _component_instance(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.component_symbol.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )


# One target chain: source instance id and ordered half-open crop intervals.
TargetChain = tuple[str, list[tuple[int, int]]]

_BINDER_LABEL = "binder"


def from_binder(
    binder_length: int,
    target: Sequence[TargetChain] = (),
) -> PlacementSpec:
    r"""
    Build target groups followed by one exact protein binder.
    """

    if binder_length <= 0:
        raise ValueError("binder_length must be positive")
    placement: PlacementSpec = {}
    for instance_id, crops in target:
        crop_intervals = tuple(crops)
        if not crop_intervals:
            raise ValueError(f"target chain {instance_id!r} needs at least one crop interval")
        if instance_id in placement:
            raise ValueError(f"target instance {instance_id!r} declared more than once")
        group: dict[str, GeneratedSegment | ExistingSpan] = {}
        for start, stop in crop_intervals:
            if not 0 <= start < stop:
                raise ValueError(
                    f"target chain {instance_id!r} crop must be a half-open " f"[start, stop); got ({start}, {stop})"
                )
            placement_id = instance_id if len(crop_intervals) == 1 else f"{instance_id}#{start}_{stop}"
            if placement_id in group:
                raise ValueError(f"target chain {instance_id!r} has duplicate crop {(start, stop)!r}")
            group[placement_id] = ExistingSpan(instance_id, start, stop)
        placement[instance_id] = group

    placement_ids = {placement_id for chain in placement.values() for placement_id in chain}
    binder_label = _BINDER_LABEL
    while binder_label in placement or binder_label in placement_ids:
        binder_label += "_"
    placement[binder_label] = {
        binder_label: GeneratedSegment(
            "protein",
            binder_length,
            binder_length,
        )
    }
    return placement


def to_binder(placement: PlacementSpec) -> tuple[int, list[TargetChain]]:
    r"""
    Decode a target-and-binder placement declaration.
    """

    validate_placement(placement)
    groups = list(placement.items())
    if not groups:
        raise ValueError("an empty PlacementSpec declares no binder")
    _, binder_placements = groups[-1]
    binder_values = tuple(binder_placements.values())
    if len(binder_values) != 1 or not isinstance(
        binder_values[0],
        GeneratedSegment,
    ):
        raise ValueError("the trailing PlacementSpec group must be exactly one generated binder")
    binder = binder_values[0]
    if binder.polymer_type != "protein":
        raise ValueError("PXDesign binder must be protein")
    if binder.min_length != binder.max_length or binder.max_length <= 0:
        raise ValueError("PXDesign binder must have one positive fixed length")

    targets: list[TargetChain] = []
    for group, placements in groups[:-1]:
        instance_id: str | None = None
        crops = []
        for value in placements.values():
            if not isinstance(value, ExistingSpan):
                raise ValueError(f"target group {group!r} must contain ExistingSpan values")
            if instance_id is None:
                instance_id = value.instance_id
            elif value.instance_id != instance_id:
                raise ValueError(f"target group {group!r} must address one source instance")
            crops.append((value.start, value.stop))
        if instance_id is None:
            raise ValueError(f"target group {group!r} must contain at least one span")
        targets.append((instance_id, crops))
    return binder.max_length, targets
