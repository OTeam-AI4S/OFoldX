r"""
Canonical RNA ``Structure`` lowering for gRNAde.

The feature equations follow geometric-rna-design v1.0.0 revision
``2453b18778a1981ac25f314790e5395f1e9ab218``. ODesign's released wrapper uses
the same equations with an all-unpaired secondary-structure graph.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.molecular import AtomSite, ComponentReference, PolymerSequence
from ...data.structure import Structure
from ...processor import Processor
from .config import GrnadeProcessorConfig
from .model import (
    GRNADE_TOKEN_BY_COMPONENT,
    GRNADE_UNKNOWN_TOKEN,
    GrnadeInput,
)

_FILL_VALUE = 1e-5
_DISTANCE_EPS = 1e-5


class GrnadeProcessor(Processor):
    r"""
    Lower aligned canonical RNA conformations to the released gRNAde tensor graph.

    Args:
        config: RNA graph construction configuration.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = GrnadeProcessorConfig
    config: GrnadeProcessorConfig

    def __init__(
        self,
        config: GrnadeProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = GrnadeProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        base_pairs: Sequence[tuple[ComponentReference, ComponentReference]] = (),
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
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

        Returns:
            One typed gRNAde tensor graph.
        """

        structures = (structure,) if isinstance(structure, Structure) else tuple(structure)
        if not structures:
            raise ValueError("gRNAde requires at least one RNA conformation")
        target = structures[0]
        target_instance_id, sequence = self._resolve_sequence(target, instance_id)

        references: list[ComponentReference] = []
        tokens: list[int] = []
        for component_index, component in enumerate(sequence.components):
            reference = (target_instance_id, component_index)
            references.append(reference)
            component_id = "U" if component.component_id in ("T", "DT") else component.component_id
            tokens.append(GRNADE_TOKEN_BY_COMPONENT.get(component_id, GRNADE_UNKNOWN_TOKEN))

        reference_set = set(references)
        selected = reference_set if design_positions is None else set(design_positions)
        if selected - reference_set:
            raise ValueError("gRNAde design positions must belong to the selected RNA instance")
        reference_index = {reference: index for index, reference in enumerate(references)}
        base_pair_indices: list[tuple[int, int]] = []
        for first, second in base_pairs:
            first_index = reference_index[first]
            second_index = reference_index[second]
            if first_index == second_index:
                raise ValueError("a gRNAde base pair must connect two distinct residues")
            base_pair_indices.append((first_index, second_index))

        if use_coordinates:
            coordinates: list[np.ndarray] = []
            atom_masks: list[np.ndarray] = []
            for conformation in structures:
                conformation_instance_id, conformation_sequence = self._resolve_sequence(
                    conformation,
                    instance_id,
                )
                if conformation_sequence.components != sequence.components:
                    raise ValueError("gRNAde conformations must contain the same RNA sequence")
                conformation_coordinates, conformation_atom_mask = _backbone_coordinates(
                    conformation,
                    conformation_instance_id,
                    conformation_sequence,
                    filter_by_occupancy=filter_by_occupancy,
                )
                coordinates.append(conformation_coordinates)
                atom_masks.append(conformation_atom_mask)
            coordinate_tensor = torch.as_tensor(
                np.stack(coordinates),
                device=self.device,
                dtype=self.dtype,
            )
            atom_mask = torch.as_tensor(
                np.stack(atom_masks),
                device=self.device,
                dtype=torch.bool,
            )
            conformation_mask = atom_mask.any(dim=-1).any(dim=-1)
        else:
            coordinate_tensor = torch.full(
                (1, len(sequence.components), 3, 3),
                _FILL_VALUE,
                device=self.device,
                dtype=self.dtype,
            )
            atom_mask = torch.zeros(
                (1, len(sequence.components), 3),
                device=self.device,
                dtype=torch.bool,
            )
            conformation_mask = torch.zeros(
                1,
                device=self.device,
                dtype=torch.bool,
            )
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
            num_neighbors=self.config.num_neighbors,
            num_rbf=self.config.num_rbf,
            num_positional_embeddings=self.config.num_positional_embeddings,
            distance_epsilon=self.config.distance_epsilon,
        )
        return GrnadeInput(
            node_scalar_features=node_scalar,
            node_vector_features=node_vector,
            edge_scalar_features=edge_scalar,
            edge_vector_features=edge_vector,
            edge_index=edge_index,
            sequence_tokens=sequence_tokens,
            conformation_mask=conformation_mask,
            design_mask=torch.tensor(
                [reference in selected for reference in references],
                device=self.device,
                dtype=torch.bool,
            ),
            structure=target,
            component_references=tuple(references),
        )

    @staticmethod
    def _resolve_sequence(
        structure: Structure,
        instance_id: str | None,
    ) -> tuple[str, PolymerSequence]:
        candidates = tuple(
            (candidate_id, entity)
            for candidate_id, entity_id in structure.system.instances.items()
            if isinstance(entity := structure.system.entities[entity_id], PolymerSequence)
            and entity.polymer_type == "rna"
        )
        if instance_id is None:
            if len(candidates) != 1:
                raise ValueError("gRNAde requires exactly one RNA instance when instance_id is omitted")
            return candidates[0]
        for candidate_id, entity in candidates:
            if candidate_id == instance_id:
                return candidate_id, entity
        raise ValueError(f"{instance_id!r} is not an RNA instance in the input structure")


def _backbone_coordinates(
    structure: Structure,
    instance_id: str,
    sequence: PolymerSequence,
    *,
    filter_by_occupancy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    observed = structure.coordinate_mask & np.isfinite(structure.coordinates).all(axis=-1)
    if filter_by_occupancy and structure.occupancy is not None:
        observed = observed & (np.isnan(structure.occupancy) | (structure.occupancy > 0))
    coordinate_index = {site: index for index, site in enumerate(structure.sites)}

    coordinates: list[np.ndarray] = []
    atom_masks: list[np.ndarray] = []
    for component_index, component in enumerate(sequence.components):
        atom_ids = (
            "P",
            "C4'",
            _base_nitrogen(
                component.component_id,
                frozenset(atom.atom_id for atom in component.atoms),
            ),
        )
        row = np.full((3, 3), _FILL_VALUE, dtype=np.float32)
        row_mask = np.zeros(3, dtype=np.bool_)
        for atom_index, atom_id in enumerate(atom_ids):
            coordinate_position = coordinate_index.get(AtomSite(instance_id, component_index, atom_id))
            if coordinate_position is None or not bool(observed[coordinate_position]):
                continue
            row[atom_index] = structure.coordinates[coordinate_position]
            row_mask[atom_index] = True
        coordinates.append(row)
        atom_masks.append(row_mask)
    return np.stack(coordinates), np.stack(atom_masks)


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
        distance_epsilon=_DISTANCE_EPS,
    )

    edge_index, edge_type = _rna_edges(
        coordinates,
        atom_mask,
        residue_mask,
        sequence_mask,
        base_pair_index,
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
        (source - destination).unsqueeze(-1),
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


def _rna_edges(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_mask: Tensor,
    sequence_mask: Tensor,
    base_pair_index: Tensor,
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
    residue_index = torch.arange(sequence_length, device=coordinates.device)
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

    chain = residue_mask.long()
    linear_chain = torch.stack((chain, chain), dim=-1).reshape(batch_size, -1)
    length_mask = linear_chain[:, :-1] == linear_chain[:, 1:]
    angle_mask = (linear_chain[:, :-2] == linear_chain[:, 1:-1]) & (linear_chain[:, :-2] == linear_chain[:, 2:])
    dihedral_mask = (
        (linear_chain[:, :-3] == linear_chain[:, 1:-2])
        & (linear_chain[:, :-3] == linear_chain[:, 2:-1])
        & (linear_chain[:, :-3] == linear_chain[:, 3:])
    )
    nitrogen_dihedral_mask = chain[:, :-1] == chain[:, 1:]

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
    *,
    distance_epsilon: float,
) -> Tensor:
    phosphorus, carbon4, nitrogen = coordinates.unbind(dim=2)
    phosphorus_mask, carbon4_mask, nitrogen_mask = atom_mask.unbind(dim=2)
    nitrogen = (nitrogen - carbon4) * (nitrogen_mask & carbon4_mask).unsqueeze(-1)
    phosphorus = (phosphorus - carbon4) * (phosphorus_mask & carbon4_mask).unsqueeze(-1)
    forward = F.pad(carbon4[:, 1:] - carbon4[:, :-1], (0, 0, 0, 1))
    backward = F.pad(carbon4[:, :-1] - carbon4[:, 1:], (0, 0, 1, 0))
    forward = forward * carbon4_mask.unsqueeze(-1)
    backward = backward * carbon4_mask.unsqueeze(-1)
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
    return expanded.reshape(*values.shape[:-1], -1)


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
    return torch.cat((encoded.cos(), encoded.sin()), dim=-1).reshape(*values.shape[:-1], -1)
