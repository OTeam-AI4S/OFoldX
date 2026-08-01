r"""
Canonical Structure lowering for OInvFold polymer and ligand graphs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_DNA,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    water_component_mask,
)
from ...data.structure import Structure
from ...processor import Processor
from .config import OInvFoldPolymerProcessorConfig
from .model import (
    OINVFOLD_NUM_GLOBAL_NODES,
    OINVFOLD_POLYMER_TOKENS,
    OInvFoldGraph,
    OInvFoldLigandInput,
    OInvFoldPolymerInput,
)

_POLYMER_ATOM_SLOTS = {
    "protein": ("N", "CA", "C", "O"),
    "dna": ("O5'", "C5'", "C4'", "P", "C3'", "O3'", "N"),
    "rna": ("P", "C4'", "N", "O5'", "C3'", "O3'", "C5'"),
}


class OInvFoldPolymerProcessor(Processor):
    r"""
    Lower one protein, DNA, or RNA chain to the released OInvFold graph.

    Args:
        config: Polymer modality stored with the paired repository.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = OInvFoldPolymerProcessorConfig
    config: OInvFoldPolymerProcessorConfig

    def __init__(
        self,
        config: OInvFoldPolymerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = OInvFoldPolymerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        structure: Structure,
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldPolymerInput:
        r"""
        Lower one complete polymer chain while retaining its design mask.

        Args:
            structure: Canonical coordinate-bearing structure.
            instance_id: Target polymer instance. It may be omitted when exactly
                one instance has the configured modality.
            design_positions: Component occurrences whose decoded identities may
                replace the input. ``None`` selects every complete row.
            filter_by_occupancy: Whether zero-occupancy atoms are unobserved.

        Returns:
            One full-chain OInvFold input.
        """

        component_axis = self._component_axis(structure, instance_id)
        system = structure.system
        observed = structure.coordinate_mask
        if filter_by_occupancy:
            observed = observed & (~structure.occupancy_mask | (structure.occupancy > 0))

        component_indices: list[int] = []
        coordinate_atom_indices: list[tuple[int, ...]] = []
        coordinates: list[Tensor] = []
        sequence_tokens: list[int | None] = []
        for component_index in component_axis.detach().cpu().tolist():
            component_id = system.symbols[int(system.component_symbol[component_index])]
            atom_start = int(system.component_atom_ptr[component_index])
            atom_stop = int(system.component_atom_ptr[component_index + 1])
            atom_index = {
                system.symbols[int(system.atom_symbol[index])]: index for index in range(atom_start, atom_stop)
            }
            atom_ids = _polymer_atom_ids(
                self.config.modality,
                frozenset(atom_index),
            )
            if any(atom_id not in atom_index for atom_id in atom_ids):
                continue
            exact_indices = tuple(atom_index[atom_id] for atom_id in atom_ids)
            index_tensor = torch.tensor(
                exact_indices,
                dtype=torch.long,
                device=structure.coordinates.device,
            )
            if not bool(observed[index_tensor].all()):
                continue
            component_indices.append(component_index)
            coordinate_atom_indices.append(exact_indices)
            coordinates.append(structure.coordinates[index_tensor])
            sequence_tokens.append(
                _polymer_token(
                    self.config.modality,
                    component_id,
                )
            )

        if len(component_indices) < 2:
            raise ValueError("OInvFold polymer inference requires at least two complete coordinate rows")

        if design_positions is None:
            selected = set(component_indices)
        else:
            selected = {system.component_index(reference) for reference in design_positions}
            if selected - set(component_indices):
                raise ValueError("OInvFold design positions must be complete rows of the selected polymer instance")

        coordinate_tensor = torch.stack(coordinates).to(
            dtype=self.dtype,
            device=self.device,
        )
        graph = _lower_polymer_graph(
            coordinate_tensor,
            num_neighbors=32,
            node_feature_size=133,
            edge_feature_size=310,
            dtype=self.dtype,
        )
        return OInvFoldPolymerInput(  # type: ignore[call-arg]
            graph=graph,
            structure=structure,
            component_index=torch.tensor(
                component_indices,
                dtype=torch.long,
                device=self.device,
            ),
            coordinate_atom_index=torch.tensor(
                coordinate_atom_indices,
                dtype=torch.long,
                device=self.device,
            ),
            design_mask=torch.tensor(
                [component_index in selected for component_index in component_indices],
                dtype=torch.bool,
                device=self.device,
            ),
            chain_index=torch.zeros(
                len(component_indices),
                dtype=torch.long,
                device=self.device,
            ),
            sequence_tokens=(
                torch.tensor(
                    [token for token in sequence_tokens if token is not None],
                    dtype=torch.long,
                    device=self.device,
                )
                if all(token is not None for token in sequence_tokens)
                else None
            ),
            batch_size=[],
        )

    def _component_axis(
        self,
        structure: Structure,
        instance_id: str | None,
    ) -> Tensor:
        system = structure.system
        entity_type = {
            "protein": ENTITY_PROTEIN,
            "dna": ENTITY_DNA,
            "rna": ENTITY_RNA,
        }[self.config.modality]
        candidates = torch.nonzero(
            system.entity_type[system.instance_entity] == entity_type,
            as_tuple=False,
        ).flatten()
        if instance_id is None:
            if candidates.numel() != 1:
                raise ValueError(f"instance_id is required unless exactly one input instance is {self.config.modality}")
            instance_index = int(candidates[0])
        else:
            instance_index = system.instance_index(instance_id)
            if int(system.entity_type[int(system.instance_entity[instance_index])]) != entity_type:
                raise ValueError(f"OInvFold {self.config.modality} processor cannot lower instance {instance_id!r}")
        component_index = torch.arange(
            int(system.instance_component_ptr[instance_index]),
            int(system.instance_component_ptr[instance_index + 1]),
            dtype=torch.long,
            device=system.instance_component_ptr.device,
        )
        return component_index


class OInvFoldLigandProcessor(Processor):
    r"""
    Lower ordered coordinate-bearing atom sites to the released ligand graph.
    """

    def __call__(
        self,
        coordinates: Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        vocabulary_logit_mask: Tensor | NDArray | None = None,
    ) -> OInvFoldLigandInput:
        r"""
        Prepare an atom-element assignment graph.

        Args:
            coordinates: Provisional atom coordinates in slot order.
            slot_ids: Stable caller-owned identifiers in coordinate order.
            context: Optional canonical context retained with decoded candidates.
            vocabulary_logit_mask: Optional additive logits mask aligned to slots.

        Returns:
            One atom graph retaining the ordered slot alignment.
        """

        slot_ids = tuple(slot_ids)
        if len(slot_ids) < 3:
            raise ValueError("OInvFold ligand inference requires at least three coordinate slots")
        coordinates = torch.as_tensor(
            coordinates,
            dtype=self.dtype,
            device=self.device,
        )
        if coordinates.shape[0] != len(slot_ids):
            raise ValueError("OInvFold ligand coordinates and slot IDs must have the same length")
        local_slots = _ligand_local_slots(coordinates)
        graph = _lower_graph(
            local_slots,
            num_neighbors=48,
            node_feature_size=114,
            edge_feature_size=272,
            dtype=self.dtype,
        )
        return OInvFoldLigandInput(  # type: ignore[call-arg]
            graph=graph,
            context=context,
            slot_ids=slot_ids,
            coordinates=coordinates,
            vocabulary_logit_mask=(
                None
                if vocabulary_logit_mask is None
                else torch.as_tensor(
                    vocabulary_logit_mask,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
            batch_size=[],
        )

    def from_structure(
        self,
        context: Structure,
        *,
        atom_sites: Sequence[AtomSite] | None = None,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandInput:
        r"""
        Lower ordered sites from an existing canonical ligand Structure.

        This convenience path is for known molecular inputs. Provisional ODesign
        atoms use :meth:`__call__` directly and are never fabricated as
        ``AtomSite`` occurrences.
        """

        if atom_sites is None:
            atom_sites = self._default_atom_sites(context)
        sites = tuple(atom_sites)
        indices = torch.tensor(
            [context.system.atom_index(site) for site in sites],
            dtype=torch.long,
            device=context.coordinates.device,
        )
        observed = context.coordinate_mask
        if filter_by_occupancy:
            observed = observed & (~context.occupancy_mask | (context.occupancy > 0))
        if not bool(observed[indices].all()):
            raise ValueError("OInvFold ligand atom sites require observed coordinates")
        return self(
            context.coordinates[indices],
            slot_ids=tuple(f"{site.instance_id}:{site.component_index}:{site.atom_id}" for site in sites),
            context=context,
        )

    @staticmethod
    def _default_atom_sites(context: Structure) -> tuple[AtomSite, ...]:
        system = context.system
        water = water_component_mask(system)
        instance_indices = []
        for instance_index, entity_index in enumerate(system.instance_entity.detach().cpu().tolist()):
            if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
                continue
            start = int(system.instance_component_ptr[instance_index])
            stop = int(system.instance_component_ptr[instance_index + 1])
            if not bool(water[start:stop].all()):
                instance_indices.append(instance_index)
        if len(instance_indices) != 1:
            raise ValueError(
                "atom_sites is required unless the context contains exactly one non-water Molecule instance"
            )
        instance_index = instance_indices[0]
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        atom_start = int(system.component_atom_ptr[component_start])
        atom_stop = int(system.component_atom_ptr[component_stop])
        return tuple(system.atom_site(atom_index) for atom_index in range(atom_start, atom_stop))


def _polymer_atom_ids(
    modality: str,
    atom_ids: frozenset[str],
) -> tuple[str, ...]:
    slots = _POLYMER_ATOM_SLOTS[modality]
    if modality == "protein":
        return slots
    if "N9" in atom_ids:
        nitrogen = "N9"
    elif "N1" in atom_ids:
        nitrogen = "N1"
    else:
        nitrogen = next(
            (atom_id for atom_id in atom_ids if atom_id.upper().startswith("N")),
            "N1",
        )
    return tuple(nitrogen if atom_id == "N" else atom_id for atom_id in slots)


def _polymer_token(
    modality: str,
    component_id: str,
) -> int | None:
    if modality == "protein":
        letter = AMINO_ACID_THREE_TO_ONE.get(component_id)
        return None if letter is None else OINVFOLD_POLYMER_TOKENS.index(letter)
    if modality == "dna":
        return {
            "DA": 20,
            "DC": 21,
            "DG": 22,
            "DT": 23,
            "DU": 23,
            "DN": 20,
            "DI": 20,
        }.get(component_id)
    return {
        "A": 20,
        "C": 21,
        "G": 22,
        "T": 24,
        "U": 24,
        "I": 20,
        "N": 20,
    }.get(component_id)


def _ligand_local_slots(coordinates: Tensor) -> Tensor:
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(torch.inf)
    neighbors = torch.topk(distances, k=2, largest=False).indices
    slots = coordinates.new_zeros((coordinates.shape[0], 6, 3))
    slots[:, 0] = coordinates[neighbors[:, 0]]
    slots[:, 1] = coordinates
    slots[:, 2] = coordinates[neighbors[:, 1]]
    return slots


def _lower_polymer_graph(
    coordinates: Tensor,
    *,
    num_neighbors: int,
    node_feature_size: int,
    edge_feature_size: int,
    dtype: torch.dtype,
) -> OInvFoldGraph:
    graph = _lower_graph(
        coordinates,
        num_neighbors=num_neighbors,
        node_feature_size=node_feature_size,
        edge_feature_size=edge_feature_size,
        dtype=dtype,
    )
    # Exact revision cc95c2f, featurizer.py:434-435, indexes each
    # global edge by min(endpoint) on both sides and therefore composes one
    # transform with its own inverse. Published polymer checkpoints consumed
    # these identity relative transforms.
    global_edge_rotations = torch.eye(
        3,
        dtype=dtype,
        device=coordinates.device,
    ).expand(graph.global_edge_index.shape[1], -1, -1)
    global_edge_translations = coordinates.new_zeros((graph.global_edge_index.shape[1], 3)).to(dtype)
    return graph.replace(  # type: ignore[attr-defined]
        global_edge_rotations=global_edge_rotations,
        global_edge_translations=global_edge_translations,
        global_edge_rbf=_rbf(
            global_edge_translations.norm(dim=-1),
            0.0,
            50.0,
            16,
        ),
    )


def _lower_graph(
    coordinates: Tensor,
    *,
    num_neighbors: int,
    node_feature_size: int,
    edge_feature_size: int,
    dtype: torch.dtype,
) -> OInvFoldGraph:
    num_nodes = coordinates.shape[0]
    centers = coordinates[:, 1]
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(torch.inf)
    neighbors = torch.topk(
        distances,
        k=min(num_neighbors, num_nodes - 1),
        largest=False,
    ).indices
    source = torch.arange(num_nodes, device=coordinates.device).unsqueeze(1).expand_as(neighbors)
    edge_index = torch.stack((source.reshape(-1), neighbors.reshape(-1)))
    order = torch.argsort(edge_index[1] * (int(edge_index[0].max()) + 1) + edge_index[0])
    edge_index = edge_index[:, order]

    node_rotations, node_translations = _reference_frames(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
    )
    source, destination = edge_index
    edge_rotations, edge_translations = _relative_transforms(
        node_rotations[source],
        node_translations[source],
        node_rotations[destination],
        node_translations[destination],
    )
    node_features, edge_features = _interaction_features(
        coordinates,
        node_rotations,
        node_translations,
        edge_rotations,
        edge_translations,
        edge_index,
    )
    node_features = F.pad(node_features, (0, node_feature_size - node_features.shape[-1]))
    edge_features = F.pad(edge_features, (0, edge_feature_size - edge_features.shape[-1]))

    global_rotations, global_translations = _global_frames(
        node_translations,
        OINVFOLD_NUM_GLOBAL_NODES,
    )
    global_nodes = torch.arange(
        num_nodes,
        num_nodes + OINVFOLD_NUM_GLOBAL_NODES,
        device=coordinates.device,
    )
    global_source = global_nodes.repeat_interleave(num_nodes)
    global_destination = torch.arange(num_nodes, device=coordinates.device).repeat(OINVFOLD_NUM_GLOBAL_NODES)
    global_edge_index = torch.stack((global_destination, global_source))
    global_edge_index = torch.cat((global_edge_index, global_edge_index.flip(0)), dim=1)

    all_rotations = torch.cat((node_rotations, global_rotations), dim=0)
    all_translations = torch.cat((node_translations, global_translations), dim=0)
    source, destination = global_edge_index
    global_edge_rotations, global_edge_translations = _relative_transforms(
        all_rotations[source],
        all_translations[source],
        all_rotations[destination],
        all_translations[destination],
    )

    return OInvFoldGraph(  # type: ignore[call-arg]
        node_features=node_features.to(dtype),
        edge_features=edge_features.to(dtype),
        edge_rotations=edge_rotations.to(dtype),
        edge_translations=edge_translations.to(dtype),
        edge_rbf=_rbf(edge_translations.norm(dim=-1), 0.0, 50.0, 16).to(dtype),
        global_edge_rotations=global_edge_rotations.to(dtype),
        global_edge_translations=global_edge_translations.to(dtype),
        global_edge_rbf=_rbf(
            global_edge_translations.norm(dim=-1),
            0.0,
            50.0,
            16,
        ).to(dtype),
        edge_index=edge_index,
        global_edge_index=global_edge_index,
        batch_index=torch.zeros(num_nodes, dtype=torch.long, device=coordinates.device),
        global_batch_index=torch.zeros(
            OINVFOLD_NUM_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
        virtual_indices=torch.arange(
            OINVFOLD_NUM_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
        batch_size=[],
    )


def _reference_frames(
    first: Tensor,
    center: Tensor,
    third: Tensor,
) -> tuple[Tensor, Tensor]:
    first = first - center
    third = third - center
    third_x, third_y, third_z = third.unbind(dim=-1)

    norm = torch.sqrt(1e-20 + third_x.square() + third_y.square())
    sin_first = -third_y / norm
    cos_first = third_x / norm
    rotation_first = first.new_zeros((*first.shape[:-1], 3, 3))
    rotation_first[..., 0, 0] = cos_first
    rotation_first[..., 0, 1] = -sin_first
    rotation_first[..., 1, 0] = sin_first
    rotation_first[..., 1, 1] = cos_first
    rotation_first[..., 2, 2] = 1

    norm = torch.sqrt(1e-20 + third_x.square() + third_y.square() + third_z.square())
    sin_second = third_z / norm
    cos_second = torch.sqrt(third_x.square() + third_y.square()) / norm
    rotation_second = first.new_zeros((*first.shape[:-1], 3, 3))
    rotation_second[..., 0, 0] = cos_second
    rotation_second[..., 0, 2] = sin_second
    rotation_second[..., 1, 1] = 1
    rotation_second[..., 2, 0] = -sin_second
    rotation_second[..., 2, 2] = cos_second

    third_rotation = rotation_second @ rotation_first
    first = torch.einsum("...ij,...j->...i", third_rotation, first)
    _, first_y, first_z = first.unbind(dim=-1)
    norm = torch.sqrt(1e-20 + first_y.square() + first_z.square())
    sin_third = -first_z / norm
    cos_third = first_y / norm
    rotation_third = first.new_zeros((*first.shape[:-1], 3, 3))
    rotation_third[..., 0, 0] = 1
    rotation_third[..., 1, 1] = cos_third
    rotation_third[..., 1, 2] = -sin_third
    rotation_third[..., 2, 1] = sin_third
    rotation_third[..., 2, 2] = cos_third
    rotations = (rotation_third @ third_rotation).transpose(-1, -2)
    return rotations, center


def _relative_transforms(
    source_rotations: Tensor,
    source_translations: Tensor,
    destination_rotations: Tensor,
    destination_translations: Tensor,
) -> tuple[Tensor, Tensor]:
    inverse_destination = destination_rotations.transpose(-1, -2)
    rotations = inverse_destination @ source_rotations
    translations = torch.einsum(
        "...ij,...j->...i",
        inverse_destination,
        source_translations - destination_translations,
    )
    return rotations, translations


def _interaction_features(
    coordinates: Tensor,
    node_rotations: Tensor,
    node_translations: Tensor,
    edge_rotations: Tensor,
    edge_translations: Tensor,
    edge_index: Tensor,
) -> tuple[Tensor, Tensor]:
    differences = F.pad(
        coordinates.reshape(-1, 3).diff(dim=0),
        (0, 0, 1, 0),
    ).reshape_as(coordinates)
    first_n, first_c = coordinates[0, 0], coordinates[0, 2]
    second_n, second_c = coordinates[1, 0], coordinates[1, 2]
    direction = 1 if (first_n - second_c).norm() > (first_c - second_n).norm() else -1
    local_differences = torch.einsum(
        "nij,naj->nai",
        node_rotations.transpose(-1, -2),
        differences * direction,
    )
    node_features = _decouple(local_differences).reshape(coordinates.shape[0], -1)
    node_features = node_features.masked_fill(torch.isnan(node_features), 0)

    source, destination = edge_index
    paired = torch.cat((coordinates[source], coordinates[destination]), dim=1)
    local_paired = torch.einsum(
        "eij,eaj->eai",
        node_rotations[source].transpose(-1, -2),
        paired - node_translations[source].unsqueeze(1),
    )
    edge_geometry = _decouple(local_paired).reshape(edge_index.shape[1], -1)
    rotation_geometry = edge_rotations.transpose(-1, -2).reshape(
        edge_index.shape[1],
        9,
    )
    translation_geometry = _decouple(edge_translations)
    position_geometry = _positional_embeddings(
        edge_index,
        16,
        dtype=coordinates.dtype,
    )
    edge_features = torch.cat(
        (
            edge_geometry,
            rotation_geometry,
            translation_geometry,
            position_geometry,
        ),
        dim=-1,
    )
    return node_features, edge_features


def _global_frames(
    translations: Tensor,
    num_global_nodes: int,
) -> tuple[Tensor, Tensor]:
    mean = translations.mean(dim=0, keepdim=True)
    centered = translations - mean
    left, _, right = torch.svd(centered.transpose(0, 1) @ centered)
    correction = torch.eye(3, dtype=translations.dtype, device=translations.device)
    if torch.det(left @ right.transpose(0, 1)) < 0:
        correction[2, 2] = -1
    rotation = left @ correction @ right.transpose(0, 1)
    return (
        rotation.unsqueeze(0).expand(num_global_nodes, -1, -1),
        mean.expand(num_global_nodes, -1),
    )


def _decouple(vectors: Tensor) -> Tensor:
    norms = vectors.norm(dim=-1, keepdim=True)
    return torch.cat(
        (
            vectors / (norms + 1e-6),
            _rbf(norms[..., 0], 0.0, 20.0, 16),
        ),
        dim=-1,
    )


def _rbf(
    values: Tensor,
    minimum: float,
    maximum: float,
    num_bins: int,
) -> Tensor:
    centers = torch.linspace(
        minimum,
        maximum,
        num_bins,
        dtype=values.dtype,
        device=values.device,
    )
    width = (maximum - minimum) / num_bins
    return torch.exp(-((values.unsqueeze(-1) - centers) / width).square())


def _positional_embeddings(
    edge_index: Tensor,
    num_embeddings: int,
    *,
    dtype: torch.dtype,
) -> Tensor:
    differences = edge_index[0] - edge_index[1]
    frequencies = torch.exp(
        torch.arange(
            0,
            num_embeddings,
            2,
            dtype=dtype,
            device=edge_index.device,
        )
        * -(math.log(10000.0) / num_embeddings)
    )
    angles = differences.unsqueeze(-1) * frequencies
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
