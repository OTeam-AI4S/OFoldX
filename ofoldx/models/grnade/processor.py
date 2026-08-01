r"""
Canonical RNA ``Structure`` lowering for gRNAde.

The feature equations follow geometric-rna-design v1.0.0 revision
``2453b18778a1981ac25f314790e5395f1e9ab218``. ODesign's released wrapper uses
the same equations with an all-unpaired secondary-structure graph.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ENTITY_RNA, ComponentReference
from ...data.structure import Structure
from ...data.tokenization import TokenLayout, select_tokens, tokenize_molecular_system
from ...processor import Processor
from .config import GrnadeV032ProcessorConfig
from .model import (
    GRNADE_TOKEN_BY_COMPONENT,
    GRNADE_UNKNOWN_TOKEN,
    GrnadeInput,
)

_FILL_VALUE = 1e-5
_DISTANCE_EPS = 1e-5
_NUM_NEIGHBORS = 32
_NUM_RBF = 32
_NUM_POSITIONAL_EMBEDDINGS = 32
_V032_DISTANCE_EPS = 1e-3


class GrnadeProcessor(Processor):
    r"""
    Lower aligned canonical RNA conformations to the released gRNAde tensor graph.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __call__(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        base_pairs: Sequence[tuple[ComponentReference, ComponentReference]] = (),
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeInput:
        r"""
        Lower one RNA sequence with optional 2D and multi-conformation 3D context.

        Args:
            structure: One canonical RNA structure, or aligned conformations of
                the same RNA sequence. The first supplies the returned canonical
                target and component-reference axis.
            instance_id: Target RNA instance. It may be omitted when unambiguous.
            design_positions: RNA component occurrences allowed to change.
                ``None`` selects every residue.
            base_pairs: Undirected secondary-structure pairs. Crossing pairs
                express pseudoknots directly.
            use_coordinates: Whether to include P, C4', and base-nitrogen
                geometry. ``False`` builds the released 1D+2D-only graph.
            filter_by_occupancy: Whether zero-occupancy coordinates are unobserved.
            cropper: Optional structural-token training view.
            generator: Random generator consumed by the crop policy.

        Returns:
            One typed gRNAde tensor graph.
        """

        structures = (structure,) if isinstance(structure, Structure) else tuple(structure)
        if not structures:
            raise ValueError("gRNAde requires at least one RNA conformation")
        target = structures[0]
        layout = _rna_token_layout(target, instance_id)
        full_component_index = layout.token_component_index

        component_ids = tuple(
            target.system.symbols[symbol]
            for symbol in target.system.component_symbol[full_component_index].detach().cpu().tolist()
        )
        tokens: list[int] = []
        for component_id in component_ids:
            component_id = "U" if component_id in ("T", "DT") else component_id
            tokens.append(GRNADE_TOKEN_BY_COMPONENT.get(component_id, GRNADE_UNKNOWN_TOKEN))

        full_component_rows = full_component_index.detach().cpu().tolist()
        component_set = set(full_component_rows)
        requested_components = (
            component_set
            if design_positions is None
            else {target.system.component_index(reference) for reference in design_positions}
        )
        if requested_components - component_set:
            raise ValueError("gRNAde design positions must belong to the selected RNA instance")

        base_pair_components: list[tuple[int, int]] = []
        for first, second in base_pairs:
            first_component = target.system.component_index(first)
            second_component = target.system.component_index(second)
            if first_component not in component_set or second_component not in component_set:
                raise ValueError("gRNAde base-pair residues must belong to the selected RNA instance")
            if first_component == second_component:
                raise ValueError("a gRNAde base pair must connect two distinct residues")
            base_pair_components.append((first_component, second_component))

        coordinates: list[Tensor] = []
        atom_masks: list[Tensor] = []
        for conformation in structures:
            conformation_components = _rna_component_index(conformation, instance_id)
            conformation_ids = tuple(
                conformation.system.symbols[symbol]
                for symbol in conformation.system.component_symbol[conformation_components].detach().cpu().tolist()
            )
            if conformation_ids != component_ids:
                raise ValueError("gRNAde conformations must contain the same RNA sequence")
            conformation_coordinates, conformation_atom_mask = _backbone_coordinates(
                conformation,
                conformation_components,
                filter_by_occupancy=filter_by_occupancy,
            )
            coordinates.append(conformation_coordinates)
            atom_masks.append(conformation_atom_mask)

        if cropper is not None:
            required = _component_mask(layout, (component for pair in base_pair_components for component in pair))
            anchor = _component_mask(
                layout,
                (
                    component
                    for component, token in zip(full_component_rows, tokens, strict=True)
                    if component in requested_components and token != GRNADE_UNKNOWN_TOKEN
                ),
            )
            layout = crop_token_layout(
                target,
                layout,
                cropper,
                required_token_mask=required,
                anchor_token_mask=anchor,
                generator=generator,
            )

        component_index = layout.token_component_index
        component_rows = component_index.detach().cpu().tolist()
        full_row = {component: index for index, component in enumerate(full_component_rows)}
        component_row = {component: index for index, component in enumerate(component_rows)}
        selected_rows = torch.tensor(
            [full_row[component] for component in component_rows],
            dtype=torch.long,
            device=structures[0].coordinates.device,
        )
        tokens = [tokens[index] for index in selected_rows.detach().cpu().tolist()]
        base_pair_indices = [(component_row[first], component_row[second]) for first, second in base_pair_components]
        coordinate_tensor = torch.stack(coordinates).to(
            device=self.device,
            dtype=self.dtype,
        )[:, selected_rows.to(self.device)]
        atom_mask = torch.stack(atom_masks).to(
            device=self.device,
            dtype=torch.bool,
        )[:, selected_rows.to(self.device)]
        conformation_mask = atom_mask.any(dim=-1).any(dim=-1)
        if not use_coordinates:
            coordinate_tensor = torch.zeros_like(coordinate_tensor)
            atom_mask = torch.zeros_like(atom_mask)
        sequence_tokens = torch.tensor(tokens, device=self.device, dtype=torch.long)
        if base_pair_indices:
            base_pair_index = torch.tensor(
                base_pair_indices,
                device=self.device,
                dtype=torch.long,
            ).t()
        else:
            base_pair_index = torch.empty(
                2,
                0,
                device=self.device,
                dtype=torch.long,
            )
        (
            node_scalar,
            node_vector,
            edge_scalar,
            edge_vector,
            edge_index,
            conformation_mask,
        ) = _rna_graph_features(
            coordinate_tensor,
            atom_mask,
            sequence_tokens,
            base_pair_index,
            conformation_mask,
            layout.token_residue_index.to(self.device),
            num_neighbors=_NUM_NEIGHBORS,
            num_rbf=_NUM_RBF,
            num_positional_embeddings=_NUM_POSITIONAL_EMBEDDINGS,
            distance_epsilon=_DISTANCE_EPS,
        )
        return GrnadeInput(  # type: ignore[call-arg]
            node_scalar_features=node_scalar,
            node_vector_features=node_vector,
            edge_scalar_features=edge_scalar,
            edge_vector_features=edge_vector,
            edge_index=edge_index,
            sequence_tokens=sequence_tokens,
            conformation_mask=conformation_mask,
            design_mask=torch.tensor(
                [component in requested_components for component in component_rows],
                device=self.device,
                dtype=torch.bool,
            ),
            structure=target,
            component_index=component_index.to(device=self.device),
            batch_size=[],
        )


class GrnadeV032Processor(Processor):
    r"""
    Lower canonical RNA conformations with the released v0.3.2 graph equation.

    Args:
        config: Released conformation count carried by the paired checkpoint.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = GrnadeV032ProcessorConfig
    config: GrnadeV032ProcessorConfig

    def __init__(
        self,
        config: GrnadeV032ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = GrnadeV032ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeInput:
        r"""
        Build the legacy spatial-only graph from selected RNA conformations.

        When fewer conformations are available than the checkpoint count, all
        are retained in input order and the remaining rows are sampled with
        replacement. Otherwise a uniform subset without replacement is used. The
        same generator drives an optional structural-token crop.
        """

        structures = (structure,) if isinstance(structure, Structure) else tuple(structure)
        if not structures:
            raise ValueError("gRNAde requires at least one RNA conformation")
        target = structures[0]
        layout = _rna_token_layout(target, instance_id)
        full_component_index = layout.token_component_index
        component_ids = tuple(
            target.system.symbols[symbol]
            for symbol in target.system.component_symbol[full_component_index].detach().cpu().tolist()
        )
        sequence_tokens = torch.tensor(
            [GRNADE_TOKEN_BY_COMPONENT.get(component_id, GRNADE_UNKNOWN_TOKEN) for component_id in component_ids],
            device=self.device,
            dtype=torch.long,
        )
        all_component_rows = full_component_index.detach().cpu().tolist()
        all_components = set(all_component_rows)
        requested_components = (
            all_components
            if design_positions is None
            else {target.system.component_index(reference) for reference in design_positions}
        )
        if requested_components - all_components:
            raise ValueError("gRNAde design positions must belong to the selected RNA instance")

        coordinates: list[Tensor] = []
        atom_masks: list[Tensor] = []
        for conformation in structures:
            conformation_components = _rna_component_index(conformation, instance_id)
            conformation_ids = tuple(
                conformation.system.symbols[symbol]
                for symbol in conformation.system.component_symbol[conformation_components].detach().cpu().tolist()
            )
            if conformation_ids != component_ids:
                raise ValueError("gRNAde conformations must contain the same RNA sequence")
            conformation_coordinates, conformation_atom_mask = _backbone_coordinates(
                conformation,
                conformation_components,
                filter_by_occupancy=filter_by_occupancy,
            )
            coordinates.append(conformation_coordinates)
            atom_masks.append(conformation_atom_mask)

        coordinate_tensor = torch.stack(coordinates).to(device=self.device, dtype=self.dtype)
        atom_mask = torch.stack(atom_masks).to(device=self.device, dtype=torch.bool)
        count = self.config.max_num_conformations
        available = coordinate_tensor.shape[0]
        if count > available:
            selected = torch.cat(
                (
                    torch.arange(available, device=self.device),
                    torch.randint(
                        available,
                        (count - available,),
                        device=self.device,
                        generator=generator,
                    ),
                )
            )
        else:
            selected = torch.randperm(
                available,
                device=self.device,
                generator=generator,
            )[:count]
        coordinate_tensor = coordinate_tensor[selected]
        atom_mask = atom_mask[selected]
        selected_conformation_mask = atom_mask.any(dim=-1).any(dim=-1)

        keep = atom_mask.all(dim=0).all(dim=-1) & (sequence_tokens < GRNADE_UNKNOWN_TOKEN)
        layout = select_tokens(layout, torch.nonzero(keep.to(layout.token_component_index.device)).flatten())
        valid_components = set(layout.token_component_index.detach().cpu().tolist())
        if design_positions is not None and requested_components - valid_components:
            raise ValueError("gRNAde design positions cannot be residues removed for missing coordinates")
        if cropper is not None:
            layout = crop_token_layout(
                target,
                layout,
                cropper,
                anchor_token_mask=_component_mask(layout, requested_components),
                generator=generator,
            )

        component_index = layout.token_component_index
        full_row = {component: index for index, component in enumerate(all_component_rows)}
        selected_rows = torch.tensor(
            [full_row[component] for component in component_index.detach().cpu().tolist()],
            dtype=torch.long,
            device=self.device,
        )
        coordinate_tensor = coordinate_tensor[:, selected_rows]
        sequence_tokens = sequence_tokens[selected_rows]
        component_index = component_index.to(device=self.device)
        if not use_coordinates:
            coordinate_tensor = torch.zeros_like(coordinate_tensor)

        component_rows = component_index.detach().cpu().tolist()
        node_scalar, node_vector, edge_scalar, edge_vector, edge_index = _v032_graph_features(
            coordinate_tensor,
            layout.token_residue_index.to(self.device),
            num_neighbors=_NUM_NEIGHBORS,
            num_rbf=_NUM_RBF,
            num_positional_embeddings=_NUM_POSITIONAL_EMBEDDINGS,
            use_coordinates=use_coordinates,
        )
        return GrnadeInput(  # type: ignore[call-arg]
            node_scalar_features=node_scalar,
            node_vector_features=node_vector,
            edge_scalar_features=edge_scalar,
            edge_vector_features=edge_vector,
            edge_index=edge_index,
            sequence_tokens=sequence_tokens,
            conformation_mask=selected_conformation_mask.unsqueeze(0).expand(sequence_tokens.shape[0], -1),
            design_mask=torch.tensor(
                [component in requested_components for component in component_rows],
                device=self.device,
                dtype=torch.bool,
            ),
            structure=target,
            component_index=component_index,
            batch_size=[],
        )


def _rna_component_index(
    structure: Structure,
    instance_id: str | None,
) -> Tensor:
    system = structure.system
    rna_instances = torch.nonzero(
        system.entity_type[system.instance_entity] == ENTITY_RNA,
        as_tuple=False,
    ).flatten()
    if instance_id is None:
        if rna_instances.numel() != 1:
            raise ValueError("gRNAde requires exactly one RNA instance when instance_id is omitted")
        instance_index = int(rna_instances[0])
    else:
        instance_index = system.instance_index(instance_id)
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_RNA:
            raise ValueError(f"{instance_id!r} is not an RNA instance in the input structure")
    return torch.arange(
        int(system.instance_component_ptr[instance_index]),
        int(system.instance_component_ptr[instance_index + 1]),
        dtype=torch.long,
        device=system.instance_component_ptr.device,
    )


def _rna_token_layout(structure: Structure, instance_id: str | None) -> TokenLayout:
    component_index = _rna_component_index(structure, instance_id)
    component_mask = torch.zeros_like(structure.system.component_symbol, dtype=torch.bool)
    component_mask[component_index] = True
    return tokenize_molecular_system(
        structure.system,
        atomize_nonstandard_polymers=False,
        component_mask=component_mask,
    )


def _component_mask(layout: TokenLayout, components: Iterable[int]) -> Tensor:
    selected = torch.tensor(
        tuple(components),
        dtype=torch.long,
        device=layout.token_component_index.device,
    )
    return torch.isin(layout.token_component_index, selected)


def _backbone_coordinates(
    structure: Structure,
    component_index: Tensor,
    *,
    filter_by_occupancy: bool,
) -> tuple[Tensor, Tensor]:
    system = structure.system
    observed = structure.coordinate_mask
    if filter_by_occupancy:
        observed = observed & (~structure.occupancy_mask | (structure.occupancy > 0))

    coordinates: list[Tensor] = []
    atom_masks: list[Tensor] = []
    for component in component_index.detach().cpu().tolist():
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        atom_rows = range(start, stop)
        atom_index = {system.symbols[int(system.atom_symbol[atom])]: atom for atom in atom_rows}
        component_id = system.symbols[int(system.component_symbol[component])]
        atom_ids = (
            "P",
            "C4'",
            _base_nitrogen(
                component_id,
                frozenset(atom_index),
            ),
        )
        row = torch.full(
            (3, 3),
            _FILL_VALUE,
            dtype=structure.coordinates.dtype,
            device=structure.coordinates.device,
        )
        row_mask = torch.zeros(
            3,
            dtype=torch.bool,
            device=structure.coordinates.device,
        )
        for feature_index, atom_id in enumerate(atom_ids):
            coordinate_position = atom_index.get(atom_id)
            if coordinate_position is None or not bool(observed[coordinate_position]):
                continue
            row[feature_index] = structure.coordinates[coordinate_position]
            row_mask[feature_index] = True
        coordinates.append(row)
        atom_masks.append(row_mask)
    return torch.stack(coordinates), torch.stack(atom_masks)


def _base_nitrogen(
    component_id: str,
    atom_ids: frozenset[str],
) -> str:
    if component_id in ("A", "G"):
        return "N9"
    if component_id in ("C", "U"):
        return "N1"
    if "N9" in atom_ids:
        return "N9"
    if "N1" in atom_ids:
        return "N1"
    return "N"


def _rna_graph_features(
    coordinates: Tensor,
    atom_mask: Tensor,
    sequence_tokens: Tensor,
    base_pair_index: Tensor,
    valid_conformations: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    num_rbf: int,
    num_positional_embeddings: int,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    coordinate_residue_mask = atom_mask.any(dim=-1)
    conformation_mask = valid_conformations.unsqueeze(0).expand(sequence_tokens.shape[0], -1)
    sequence_mask = sequence_tokens != GRNADE_UNKNOWN_TOKEN
    residue_mask = coordinate_residue_mask & sequence_mask.unsqueeze(0)
    dihedral, angle, length = _internal_coordinates(
        coordinates,
        residue_mask,
        residue_index,
        distance_epsilon=_DISTANCE_EPS,
    )
    angular = torch.cat((dihedral, angle), dim=-1)
    node_scalar = torch.cat(
        (
            torch.cos(angular),
            torch.sin(angular),
            torch.log(length + distance_epsilon),
        ),
        dim=-1,
    )
    node_scalar = node_scalar * residue_mask.unsqueeze(-1)
    node_vector = _internal_vectors(
        coordinates,
        atom_mask,
        residue_index,
        distance_epsilon=_DISTANCE_EPS,
    )

    edge_index, edge_type = _rna_edges(
        coordinates,
        atom_mask,
        residue_mask,
        sequence_mask,
        base_pair_index,
        residue_index,
        num_neighbors=num_neighbors,
        distance_epsilon=distance_epsilon,
    )
    coordinates_by_residue = coordinates.permute(1, 0, 2, 3)
    atom_mask_by_residue = atom_mask.permute(1, 0, 2)
    source, destination = edge_index
    edge_displacements = coordinates_by_residue[source] - coordinates_by_residue[destination]
    edge_atom_mask = atom_mask_by_residue[source] & atom_mask_by_residue[destination]
    edge_displacements = edge_displacements * edge_atom_mask.unsqueeze(-1)
    edge_lengths = torch.sqrt(edge_displacements.square().sum(dim=-1) + distance_epsilon)
    edge_lengths = edge_lengths * edge_atom_mask
    edge_rbf = _rbf_expansion(edge_lengths, num_rbf=num_rbf)
    edge_position = _positional_encoding(
        (residue_index[source] - residue_index[destination]).unsqueeze(-1),
        num_positional_embeddings,
        dtype=coordinates.dtype,
    ).unsqueeze(1)
    edge_position = edge_position.expand(-1, coordinates.shape[0], -1)
    edge_type_features = F.one_hot(edge_type, num_classes=4).to(coordinates.dtype).unsqueeze(1)
    edge_type_features = edge_type_features.expand(-1, coordinates.shape[0], -1)
    edge_scalar = torch.cat((edge_type_features, edge_rbf, edge_position), dim=-1)
    edge_vector = _normalize(edge_displacements, distance_epsilon=_DISTANCE_EPS)

    return (
        torch.nan_to_num(node_scalar.permute(1, 0, 2)),
        torch.nan_to_num(node_vector.permute(1, 0, 2, 3)),
        torch.nan_to_num(edge_scalar),
        torch.nan_to_num(edge_vector),
        edge_index,
        conformation_mask,
    )


def _v032_graph_features(
    coordinates: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    num_rbf: int,
    num_positional_embeddings: int,
    use_coordinates: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_conformations, num_residues = coordinates.shape[:2]
    if not use_coordinates:
        return (
            coordinates.new_zeros(num_residues, num_conformations, 15),
            coordinates.new_zeros(num_residues, num_conformations, 4, 3),
            coordinates.new_zeros(0, num_conformations, 131),
            coordinates.new_zeros(0, num_conformations, 3, 3),
            torch.empty(2, 0, device=coordinates.device, dtype=torch.long),
        )
    residue_mask = torch.ones(
        num_conformations,
        num_residues,
        device=coordinates.device,
        dtype=torch.bool,
    )
    dihedral, angle, length = _internal_coordinates(
        coordinates,
        residue_mask,
        residue_index,
        distance_epsilon=_V032_DISTANCE_EPS,
    )
    angular = torch.cat((dihedral, angle), dim=-1)
    node_scalar = torch.cat(
        (
            angular.cos(),
            angular.sin(),
            torch.log(length + _V032_DISTANCE_EPS),
        ),
        dim=-1,
    )
    node_vector = _internal_vectors(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        residue_index,
        distance_epsilon=_V032_DISTANCE_EPS,
    )

    edges = [
        _knn_edges(conformation.mean(dim=1), residue_mask[index], num_neighbors)
        for index, conformation in enumerate(coordinates)
    ]
    edge_index = _undirected_union(edges, num_residues)
    coordinates_by_residue = coordinates.permute(1, 0, 2, 3)
    source, destination = edge_index
    displacement = coordinates_by_residue[source] - coordinates_by_residue[destination]
    edge_length = torch.sqrt(displacement.square().sum(dim=-1) + _V032_DISTANCE_EPS)
    edge_rbf = _rbf_expansion(edge_length, num_rbf=num_rbf)
    edge_position = _positional_encoding(
        (residue_index[source] - residue_index[destination]).unsqueeze(-1),
        num_positional_embeddings,
        dtype=coordinates.dtype,
    ).unsqueeze(1)
    edge_position = edge_position.expand(-1, num_conformations, -1)
    edge_scalar = torch.cat((edge_rbf, edge_position, torch.log(edge_length)), dim=-1)
    edge_vector = _normalize(displacement, distance_epsilon=_V032_DISTANCE_EPS)
    return (
        torch.nan_to_num(node_scalar.permute(1, 0, 2)),
        torch.nan_to_num(node_vector.permute(1, 0, 2, 3)),
        torch.nan_to_num(edge_scalar),
        torch.nan_to_num(edge_vector),
        edge_index,
    )


def _undirected_union(edges: list[Tensor], num_residues: int) -> Tensor:
    edge_index = torch.cat(edges, dim=1)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    if not edge_index.shape[1]:
        return edge_index
    code = torch.unique(edge_index[0] * num_residues + edge_index[1], sorted=True)
    return torch.stack(
        (
            code.div(num_residues, rounding_mode="floor"),
            code.remainder(num_residues),
        )
    )


def _rna_edges(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_mask: Tensor,
    sequence_mask: Tensor,
    base_pair_index: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor]:
    edge_indices: list[Tensor] = []
    edge_types: list[Tensor] = []
    for conformation_coordinates, conformation_atom_mask, conformation_residue_mask in zip(
        coordinates,
        atom_mask,
        residue_mask,
        strict=True,
    ):
        centroids = conformation_coordinates.sum(dim=1) / (
            conformation_atom_mask.sum(dim=1, keepdim=True) + distance_epsilon
        )
        spatial_edges = _knn_edges(
            centroids,
            conformation_residue_mask,
            num_neighbors,
        )
        edge_indices.append(spatial_edges)
        edge_types.append(
            torch.full(
                (spatial_edges.shape[1],),
                2,
                device=coordinates.device,
                dtype=torch.long,
            )
        )

    edge_indices.append(base_pair_index)
    edge_types.append(
        torch.full(
            (base_pair_index.shape[1],),
            3,
            device=coordinates.device,
            dtype=torch.long,
        )
    )

    sequence_length = sequence_mask.shape[0]
    separation = (residue_index[:, None] - residue_index[None]).abs()
    sequence_edges = (
        ((separation > 0) & (separation <= num_neighbors) & sequence_mask[:, None] & sequence_mask[None])
        .nonzero(as_tuple=False)
        .t()
    )
    edge_indices.append(sequence_edges)
    edge_types.append(
        torch.ones(
            sequence_edges.shape[1],
            device=coordinates.device,
            dtype=torch.long,
        )
    )

    edge_index = torch.cat(edge_indices, dim=1)
    edge_type = torch.cat(edge_types)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    edge_type = torch.cat((edge_type, edge_type))
    if edge_index.shape[1] == 0:
        return edge_index, edge_type

    code = edge_index[0] * sequence_length + edge_index[1]
    unique_code, inverse = torch.unique(code, sorted=True, return_inverse=True)
    merged_type = torch.zeros(
        unique_code.shape[0],
        device=edge_type.device,
        dtype=edge_type.dtype,
    )
    merged_type.scatter_reduce_(
        0,
        inverse,
        edge_type,
        reduce="amax",
        include_self=True,
    )
    edge_index = torch.stack(
        (
            unique_code.div(sequence_length, rounding_mode="floor"),
            unique_code.remainder(sequence_length),
        )
    )
    keep = sequence_mask[edge_index[0]] & sequence_mask[edge_index[1]]
    return edge_index[:, keep], merged_type[keep]


def _knn_edges(
    coordinates: Tensor,
    mask: Tensor,
    num_neighbors: int,
) -> Tensor:
    num_residues = coordinates.shape[0]
    neighbor_count = min(num_neighbors, max(num_residues - 1, 0))
    if neighbor_count == 0:
        return torch.empty(2, 0, device=coordinates.device, dtype=torch.long)
    distance = torch.cdist(coordinates, coordinates)
    distance.fill_diagonal_(torch.inf)
    nearest = distance.topk(neighbor_count, dim=-1, largest=False).indices
    source = nearest.reshape(-1)
    destination = torch.arange(
        num_residues,
        device=coordinates.device,
    ).repeat_interleave(neighbor_count)
    keep = mask[source] & mask[destination]
    return torch.stack((source[keep], destination[keep]))


def _internal_coordinates(
    coordinates: Tensor,
    residue_mask: Tensor,
    residue_index: Tensor,
    *,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, num_residues = coordinates.shape[:2]
    linear = coordinates[:, :, :2].reshape(batch_size, 2 * num_residues, 3)
    linear_length = _length(
        linear[:, 1:],
        linear[:, :-1],
        distance_epsilon=distance_epsilon,
    )
    linear_angle = math.pi - _angle(
        linear[:, :-2],
        linear[:, 1:-1],
        linear[:, 2:],
        distance_epsilon=distance_epsilon,
    )
    linear_dihedral = _dihedral(
        linear[:, :-3],
        linear[:, 1:-2],
        linear[:, 2:-1],
        linear[:, 3:],
        distance_epsilon=distance_epsilon,
    )

    phosphorus, carbon4, nitrogen = coordinates.unbind(dim=2)
    nitrogen_length = _length(
        carbon4,
        nitrogen,
        distance_epsilon=distance_epsilon,
    )
    nitrogen_angle = math.pi - _angle(
        phosphorus,
        carbon4,
        nitrogen,
        distance_epsilon=distance_epsilon,
    )
    nitrogen_dihedral = _dihedral(
        phosphorus[:, 1:],
        nitrogen[:, :-1],
        carbon4[:, :-1],
        phosphorus[:, :-1],
        distance_epsilon=distance_epsilon,
    )

    linear_mask = torch.stack((residue_mask, residue_mask), dim=-1).reshape(batch_size, -1)
    linear_position = torch.stack((residue_index * 2, residue_index * 2 + 1), dim=-1).flatten()
    length_mask = linear_mask[:, :-1] & linear_mask[:, 1:] & (linear_position[1:] - linear_position[:-1] == 1)
    angle_mask = (
        linear_mask[:, :-2]
        & linear_mask[:, 1:-1]
        & linear_mask[:, 2:]
        & (linear_position[1:-1] - linear_position[:-2] == 1)
        & (linear_position[2:] - linear_position[1:-1] == 1)
    )
    dihedral_mask = (
        linear_mask[:, :-3]
        & linear_mask[:, 1:-2]
        & linear_mask[:, 2:-1]
        & linear_mask[:, 3:]
        & (linear_position[1:-2] - linear_position[:-3] == 1)
        & (linear_position[2:-1] - linear_position[1:-2] == 1)
        & (linear_position[3:] - linear_position[2:-1] == 1)
    )
    nitrogen_dihedral_mask = residue_mask[:, :-1] & residue_mask[:, 1:] & (residue_index[1:] - residue_index[:-1] == 1)

    dihedral = _pad_internal(
        linear_dihedral,
        nitrogen_dihedral,
        left=1,
        right=2,
        num_residues=num_residues,
    )
    angle = _pad_internal(
        linear_angle,
        nitrogen_angle,
        left=1,
        right=1,
        num_residues=num_residues,
    )
    length = _pad_internal(
        linear_length,
        nitrogen_length,
        left=0,
        right=1,
        num_residues=num_residues,
    )
    dihedral_mask = _pad_internal(
        dihedral_mask.to(coordinates.dtype),
        nitrogen_dihedral_mask.to(coordinates.dtype),
        left=1,
        right=2,
        num_residues=num_residues,
    )
    angle_mask = _pad_internal(
        angle_mask.to(coordinates.dtype),
        residue_mask.to(coordinates.dtype),
        left=1,
        right=1,
        num_residues=num_residues,
    )
    length_mask = _pad_internal(
        length_mask.to(coordinates.dtype),
        residue_mask.to(coordinates.dtype),
        left=0,
        right=1,
        num_residues=num_residues,
    )
    residue_float = residue_mask.unsqueeze(-1).to(coordinates.dtype)
    return (
        dihedral * dihedral_mask * residue_float,
        angle * angle_mask * residue_float,
        length * length_mask * residue_float,
    )


def _pad_internal(
    linear: Tensor,
    branched: Tensor,
    *,
    left: int,
    right: int,
    num_residues: int,
) -> Tensor:
    linear = F.pad(linear, (left, right)).reshape(linear.shape[0], num_residues, 2)
    branched = F.pad(branched, (0, num_residues - branched.shape[1]))
    return torch.cat((linear, branched.unsqueeze(-1)), dim=-1)


def _internal_vectors(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_index: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    phosphorus, carbon4, nitrogen = coordinates.unbind(dim=2)
    phosphorus_mask, carbon4_mask, nitrogen_mask = atom_mask.unbind(dim=2)
    nitrogen = (nitrogen - carbon4) * (nitrogen_mask & carbon4_mask).unsqueeze(-1)
    phosphorus = (phosphorus - carbon4) * (phosphorus_mask & carbon4_mask).unsqueeze(-1)
    adjacent = residue_index[1:] - residue_index[:-1] == 1
    adjacent_mask = carbon4_mask[:, 1:] & carbon4_mask[:, :-1] & adjacent
    forward = F.pad((carbon4[:, 1:] - carbon4[:, :-1]) * adjacent_mask.unsqueeze(-1), (0, 0, 0, 1))
    backward = F.pad((carbon4[:, :-1] - carbon4[:, 1:]) * adjacent_mask.unsqueeze(-1), (0, 0, 1, 0))
    return torch.stack(
        (
            _normalize(phosphorus, distance_epsilon=distance_epsilon),
            _normalize(nitrogen, distance_epsilon=distance_epsilon),
            _normalize(forward, distance_epsilon=distance_epsilon),
            _normalize(backward, distance_epsilon=distance_epsilon),
        ),
        dim=-2,
    )


def _length(
    first: Tensor,
    second: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    return torch.sqrt((second - first).square().sum(dim=-1) + distance_epsilon)


def _angle(
    first: Tensor,
    center: Tensor,
    last: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    first_direction = _normalize(first - center, distance_epsilon=distance_epsilon)
    last_direction = _normalize(last - center, distance_epsilon=distance_epsilon)
    return torch.acos((first_direction * last_direction).sum(dim=-1).clamp(-1, 1))


def _dihedral(
    first: Tensor,
    second: Tensor,
    third: Tensor,
    fourth: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    first_direction = _normalize(second - first, distance_epsilon=distance_epsilon)
    middle_direction = _normalize(third - second, distance_epsilon=distance_epsilon)
    last_direction = _normalize(fourth - third, distance_epsilon=distance_epsilon)
    first_normal = _normalize(
        torch.cross(first_direction, middle_direction, dim=-1),
        distance_epsilon=distance_epsilon,
    )
    last_normal = _normalize(
        torch.cross(middle_direction, last_direction, dim=-1),
        distance_epsilon=distance_epsilon,
    )
    cosine = (first_normal * last_normal).sum(dim=-1).clamp(-1, 1)
    sign = torch.sign((first_direction * last_normal).sum(dim=-1))
    return sign * torch.acos(cosine)


def _normalize(
    vectors: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    return vectors / torch.sqrt(vectors.square().sum(dim=-1, keepdim=True) + distance_epsilon)


def _rbf_expansion(
    values: Tensor,
    *,
    num_rbf: int,
) -> Tensor:
    centers = torch.linspace(
        0,
        30,
        num_rbf,
        device=values.device,
        dtype=values.dtype,
    )
    standard_deviation = centers[1] - centers[0]
    expanded = torch.exp(-((values.unsqueeze(-1) - centers) / standard_deviation).square())
    return expanded.reshape(*values.shape[:-1], values.shape[-1] * num_rbf)


def _positional_encoding(
    values: Tensor,
    size: int,
    *,
    dtype: torch.dtype,
) -> Tensor:
    frequencies = size // 2
    periods = torch.logspace(
        0,
        3,
        frequencies,
        device=values.device,
        dtype=dtype,
    )
    angular_frequency = 2 * math.pi / periods
    encoded = values.unsqueeze(-1) * angular_frequency
    return torch.cat((encoded.cos(), encoded.sin()), dim=-1).reshape(
        *values.shape[:-1],
        values.shape[-1] * 2 * frequencies,
    )
