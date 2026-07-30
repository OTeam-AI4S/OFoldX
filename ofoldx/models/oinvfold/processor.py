r"""
Canonical Structure lowering for OInvFold polymer and ligand graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    Molecule,
    PolymerSequence,
    is_water_component,
)
from ...data.structure import Structure
from ...processor import Processor
from .config import OInvFoldPolymerProcessorConfig
from .model import (
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
_NUM_GLOBAL_NODES = 3


@dataclass(eq=False)
class OInvFoldLigandSlots:
    r"""
    Ordered coordinate slots for ligand atom-element assignment.

    Args:
        slot_ids: Stable caller-owned identifiers in coordinate order.
        coordinates: Real array with shape ``(num_slots, 3)``.
        context: Optional canonical context Structure. The slots need not be
            atoms in that Structure.
        vocabulary_logit_mask: Optional additive logits mask with one row per
            slot and one column per released ligand vocabulary token.
    """

    slot_ids: tuple[str, ...]
    coordinates: NDArray
    context: Structure | None = None
    vocabulary_logit_mask: NDArray | None = None

    def __post_init__(self) -> None:
        if self.coordinates.shape != (len(self.slot_ids), 3):
            raise ValueError("OInvFold ligand coordinates must have shape " f"{(len(self.slot_ids), 3)}")
        if self.coordinates.dtype.kind not in "iuf":
            raise TypeError("OInvFold ligand coordinates must have a real numerical dtype")
        if len(self.slot_ids) != len(set(self.slot_ids)):
            raise ValueError("OInvFold ligand slot IDs must be unique")


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

        instance_id, entity = self._resolve_entity(structure, instance_id)

        observed = structure.coordinate_mask
        if filter_by_occupancy and structure.occupancy is not None:
            observed = observed & (np.isnan(structure.occupancy) | (structure.occupancy > 0))
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}

        references: list[ComponentReference] = []
        coordinate_slots: list[tuple[AtomSite, ...]] = []
        coordinates: list[np.ndarray] = []
        sequence_tokens: list[int | None] = []
        for component_index, component in enumerate(entity.components):
            atom_ids = _polymer_atom_ids(self.config.modality, component)
            sites = tuple(AtomSite(instance_id, component_index, atom_id) for atom_id in atom_ids)
            indices = tuple(coordinate_index.get(site) for site in sites)
            if any(index is None for index in indices):
                continue
            exact_indices = tuple(int(index) for index in indices if index is not None)
            if not all(observed[index] and np.isfinite(structure.coordinates[index]).all() for index in exact_indices):
                continue
            references.append((instance_id, component_index))
            coordinate_slots.append(sites)
            coordinates.append(structure.coordinates[np.asarray(exact_indices, dtype=np.int64)])
            sequence_tokens.append(
                _polymer_token(
                    self.config.modality,
                    component,
                )
            )

        if len(references) < 2:
            raise ValueError("OInvFold polymer inference requires at least two complete coordinate rows")

        if design_positions is None:
            selected = set(references)
        else:
            selected = set(design_positions)
            if selected - set(references):
                raise ValueError("OInvFold design positions must be complete rows of the selected polymer instance")

        coordinate_tensor = torch.as_tensor(
            np.stack(coordinates),
            dtype=torch.float32,
            device=self.device,
        )
        graph = _lower_polymer_graph(
            coordinate_tensor,
            num_neighbors=32,
            node_feature_size=133,
            edge_feature_size=310,
            dtype=self.dtype,
        )
        return OInvFoldPolymerInput(
            graph=graph,
            structure=structure,
            component_references=tuple(references),
            coordinate_slots=tuple(coordinate_slots),
            design_mask=torch.tensor(
                [reference in selected for reference in references],
                dtype=torch.bool,
                device=self.device,
            ),
            chain_index=torch.zeros(
                len(references),
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
        )

    def _resolve_entity(
        self,
        structure: Structure,
        instance_id: str | None,
    ) -> tuple[str, PolymerSequence]:
        candidates: list[tuple[str, PolymerSequence]] = []
        for candidate, entity_id in structure.system.instances.items():
            entity = structure.system.entities[entity_id]
            if isinstance(entity, PolymerSequence) and entity.polymer_type == self.config.modality:
                candidates.append((candidate, entity))

        if instance_id is None:
            if len(candidates) != 1:
                raise ValueError(
                    "instance_id is required unless exactly one input instance " f"is {self.config.modality}"
                )
            return candidates[0]

        entity = structure.system.entities[structure.system.instances[instance_id]]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type != self.config.modality:
            raise ValueError(f"OInvFold {self.config.modality} processor cannot lower instance {instance_id!r}")
        return instance_id, entity


class OInvFoldLigandProcessor(Processor):
    r"""
    Lower ordered coordinate-bearing atom sites to the released ligand graph.
    """

    def __call__(
        self,
        input: OInvFoldLigandSlots,
    ) -> OInvFoldLigandInput:
        r"""
        Prepare an atom-element assignment graph.

        Args:
            input: Ordered provisional slots and optional canonical context.

        Returns:
            One atom graph retaining the ordered slot alignment.
        """

        if len(input.slot_ids) < 3:
            raise ValueError("OInvFold ligand inference requires at least three coordinate slots")
        if not np.isfinite(input.coordinates).all():
            raise ValueError("OInvFold ligand coordinates must be finite")
        coordinates = torch.as_tensor(
            input.coordinates,
            dtype=torch.float32,
            device=self.device,
        )
        local_slots = _ligand_local_slots(coordinates)
        graph = _lower_graph(
            local_slots,
            num_neighbors=48,
            node_feature_size=114,
            edge_feature_size=272,
            dtype=self.dtype,
        )
        return OInvFoldLigandInput(
            graph=graph,
            context=input.context,
            slot_ids=input.slot_ids,
            coordinates=coordinates.to(self.dtype),
            vocabulary_logit_mask=(
                None
                if input.vocabulary_logit_mask is None
                else torch.as_tensor(
                    input.vocabulary_logit_mask,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
        )

    def slots_from_structure(
        self,
        context: Structure,
        *,
        atom_sites: Sequence[AtomSite] | None = None,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandSlots:
        r"""
        Derive ordered slots from an existing canonical ligand Structure.

        This convenience path is for known molecular inputs. Provisional ODesign
        atoms use [`OInvFoldLigandSlots`][] directly and are never fabricated as
        ``AtomSite`` occurrences.
        """

        if atom_sites is None:
            atom_sites = self._default_atom_sites(context)
        sites = tuple(atom_sites)
        observed = context.coordinate_mask
        if filter_by_occupancy and context.occupancy is not None:
            observed = observed & (np.isnan(context.occupancy) | (context.occupancy > 0))
        coordinate_index = {site: index for index, site in enumerate(context.sites)}
        indices = tuple(coordinate_index[site] for site in sites)
        if not all(observed[index] and np.isfinite(context.coordinates[index]).all() for index in indices):
            raise ValueError("OInvFold ligand atom sites require observed finite coordinates")
        return OInvFoldLigandSlots(
            tuple(f"{site.instance_id}:{site.component_index}:{site.atom_id}" for site in sites),
            context.coordinates[np.asarray(indices, dtype=np.int64)],
            context,
        )

    @staticmethod
    def _default_atom_sites(context: Structure) -> tuple[AtomSite, ...]:
        instance_ids = []
        for instance_id, entity_id in context.system.instances.items():
            entity = context.system.entities[entity_id]
            if isinstance(entity, Molecule) and not all(
                is_water_component(component) for component in entity.components
            ):
                instance_ids.append(instance_id)
        if len(instance_ids) != 1:
            raise ValueError(
                "atom_sites is required unless the context contains exactly one non-water Molecule instance"
            )
        instance_id = instance_ids[0]
        return tuple(site for site in context.sites if site.instance_id == instance_id)


def _polymer_atom_ids(
    modality: str,
    component: ChemicalComponent,
) -> tuple[str, ...]:
    slots = _POLYMER_ATOM_SLOTS[modality]
    if modality == "protein":
        return slots
    atoms = component.atoms
    atom_ids = {atom.atom_id for atom in atoms}
    if "N9" in atom_ids:
        nitrogen = "N9"
    elif "N1" in atom_ids:
        nitrogen = "N1"
    else:
        nitrogen = next(
            (atom.atom_id for atom in atoms if atom.atom_id.upper().startswith("N")),
            "N1",
        )
    return tuple(nitrogen if atom_id == "N" else atom_id for atom_id in slots)


def _polymer_token(
    modality: str,
    component: ChemicalComponent,
) -> int | None:
    if modality == "protein":
        letter = AMINO_ACID_THREE_TO_ONE.get(component.component_id)
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
        }.get(component.component_id)
    return {
        "A": 20,
        "C": 21,
        "G": 22,
        "T": 24,
        "U": 24,
        "I": 20,
        "N": 20,
    }.get(component.component_id)


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
    return replace(
        graph,
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
        _NUM_GLOBAL_NODES,
    )
    global_nodes = torch.arange(
        num_nodes,
        num_nodes + _NUM_GLOBAL_NODES,
        device=coordinates.device,
    )
    global_source = global_nodes.repeat_interleave(num_nodes)
    global_destination = torch.arange(num_nodes, device=coordinates.device).repeat(_NUM_GLOBAL_NODES)
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

    return OInvFoldGraph(
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
            _NUM_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
        virtual_indices=torch.arange(
            _NUM_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
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
    position_geometry = _positional_embeddings(edge_index, 16)
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
) -> Tensor:
    differences = edge_index[0] - edge_index[1]
    frequencies = torch.exp(
        torch.arange(
            0,
            num_embeddings,
            2,
            dtype=torch.float32,
            device=edge_index.device,
        )
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = differences.unsqueeze(-1) * frequencies
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
