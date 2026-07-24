r"""
Reusable graph message-passing components for MPNN models.
"""

from __future__ import annotations

import operator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backend import LayerNorm


def gather_edges(edges: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""
    Gather each residue's selected neighbors from a dense pair tensor.

    Args:
        edges: Dense pair features with shape ``(B, N, N, C)``.
        neighbor_indices: Selected neighbor indices with shape ``(B, N, K)``.

    Returns:
        Neighbor-aligned edge features with shape ``(B, N, K, C)``.
    """

    indices = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, edges.shape[-1])
    return torch.gather(edges, 2, indices)


def gather_nodes(nodes: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""
    Gather node features along every residue's neighbor list.

    Args:
        nodes: Node features with shape ``(B, N, C)``.
        neighbor_indices: Selected neighbor indices with shape ``(B, N, K)``.

    Returns:
        Neighbor-aligned node features with shape ``(B, N, K, C)``.
    """

    indices = neighbor_indices.reshape(neighbor_indices.shape[0], -1, 1).expand(
        -1,
        -1,
        nodes.shape[-1],
    )
    return torch.gather(nodes, 1, indices).reshape(*neighbor_indices.shape, nodes.shape[-1])


def cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""
    Concatenate gathered neighbor nodes onto aligned edge or context states.

    Args:
        node_states: Node features with shape ``(B, N, C_node)``.
        neighbor_states: Neighbor-aligned features with shape ``(B, N, K, C_neighbor)``.
        neighbor_indices: Selected neighbor indices with shape ``(B, N, K)``.

    Returns:
        Concatenated neighbor and node features with shape ``(B, N, K, C_neighbor + C_node)``.
    """

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_indices)), dim=-1)


def nearest_neighbors(
    coordinates: Tensor,
    residue_mask: Tensor,
    num_neighbors: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""
    Return masked nearest-neighbor distances and indices.

    Args:
        coordinates: Residue coordinates with shape ``(B, N, 3)``.
        residue_mask: Valid-residue mask with shape ``(B, N)``.
        num_neighbors: Maximum number of neighbors selected per residue.
        eps: Additive epsilon inside the distance square root.

    Returns:
        Neighbor distances and indices, each with shape ``(B, N, min(num_neighbors, N))``.
    """

    pair_mask = residue_mask[:, :, None] * residue_mask[:, None, :]
    distances = torch.sqrt(torch.sum((coordinates[:, :, None] - coordinates[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    maximum = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * maximum
    return torch.topk(
        adjusted,
        min(num_neighbors, coordinates.shape[1]),
        dim=-1,
        largest=False,
    )


def radial_basis(
    distances: Tensor,
    num_bins: int = 16,
    lower_bound: float = 2.0,
    upper_bound: float = 22.0,
) -> Tensor:
    r"""
    Expand distances over evenly spaced Gaussian radial-basis centers.

    Args:
        distances: Input distances of any shape.
        num_bins: Number of radial-basis centers.
        lower_bound: First radial-basis center.
        upper_bound: Last radial-basis center.

    Returns:
        Radial-basis features with ``num_bins`` appended as the final axis.
    """

    centers = torch.linspace(
        lower_bound,
        upper_bound,
        num_bins,
        device=distances.device,
        dtype=distances.dtype,
    )
    centers = centers.view(*((1,) * distances.ndim), num_bins)
    width = (upper_bound - lower_bound) / num_bins
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def virtual_beta_carbon(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""
    Reconstruct the idealized protein Cβ coordinate from N, Cα, and C.

    Args:
        nitrogen: Backbone nitrogen coordinates with final dimension 3.
        alpha_carbon: Alpha-carbon coordinates with final dimension 3.
        carbon: Backbone carbon coordinates with final dimension 3.

    Returns:
        Virtual beta-carbon coordinates with the same shape as each input.
    """

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    return -0.58273431 * torch.cross(b, c, dim=-1) + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def local_backbone_frame_features(
    nitrogen: Tensor,
    alpha_carbon: Tensor,
    carbon: Tensor,
    points: Tensor,
) -> Tensor:
    r"""
    Express context-point directions in each residue's local backbone frame.

    Args:
        nitrogen: Nitrogen coordinates with shape ``(B, L, 3)``.
        alpha_carbon: Alpha-carbon coordinates with shape ``(B, L, 3)``.
        carbon: Carbon coordinates with shape ``(B, L, 3)``.
        points: Context coordinates with shape ``(B, L, A, 3)``.

    Returns:
        Scale-invariant direction features with shape ``(B, L, A, 4)``.
    """

    first_axis = F.normalize(nitrogen - alpha_carbon, dim=-1)
    carbon_vector = carbon - alpha_carbon
    projected = torch.einsum("bli,bli->bl", first_axis, carbon_vector)[..., None]
    second_axis = F.normalize(carbon_vector - first_axis * projected, dim=-1)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    frame = torch.stack((first_axis, second_axis, third_axis), dim=-1)
    local = torch.einsum("blqp,blyq->blyp", frame, points - alpha_carbon[:, :, None])
    planar_radius = torch.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-8)
    spatial_radius = torch.norm(local, dim=-1) + 1e-8
    return torch.stack(
        (
            local[..., 0] / planar_radius,
            local[..., 1] / planar_radius,
            planar_radius / spatial_radius,
            local[..., 2] / spatial_radius,
        ),
        dim=-1,
    )


def validate_positive_integer(value: int, name: str) -> int:
    r"""
    Return an integer-like value after requiring it to be positive.

    Args:
        value: Integer-like value to validate.
        name: Parameter name used in the error message.

    Returns:
        The validated built-in integer.

    Raises:
        ValueError: If ``value`` is not positive.
    """

    result = operator.index(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def decoding_order_to_ranks(decoding_order: Tensor) -> Tensor:
    r"""
    Convert a residue permutation to ranks where smaller values decode earlier.

    Args:
        decoding_order: Residue indices ordered by decoding step with shape ``(..., N)``.

    Returns:
        Per-residue decoding ranks with the same shape and dtype.
    """

    ranks = torch.empty_like(decoding_order)
    ranks.scatter_(
        -1,
        decoding_order,
        torch.arange(decoding_order.shape[-1], device=decoding_order.device).expand_as(decoding_order),
    )
    return ranks


def tied_groups_from_ids(
    tied_group_ids: Tensor,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    r"""
    Reconstruct per-batch tied member groups from dense causal-unit IDs.

    Every returned group contains more than one residue and lists members in
    ascending residue order. Untied singleton units are omitted.

    Args:
        tied_group_ids: Integer group ID for each residue with shape ``(B, N)``.

    Returns:
        Per-batch tuples of tied residue-index groups.
    """

    batches: list[tuple[tuple[int, ...], ...]] = []
    for row in tied_group_ids.tolist():
        members_by_id: dict[int, list[int]] = {}
        for residue_index, group_id in enumerate(row):
            members_by_id.setdefault(group_id, []).append(residue_index)
        batches.append(tuple(tuple(members) for members in members_by_id.values() if len(members) > 1))
    return tuple(batches)


def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor:
    r"""
    Compute unreduced negative log likelihood for sequence tokens.

    Args:
        sequence_tokens: Target token IDs with shape ``(..., N)``.
        log_probabilities: Token log probabilities with shape ``(..., N, V)``.

    Returns:
        Per-position negative log likelihood with shape ``(..., N)``.
    """

    return -log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1)).squeeze(-1)


def label_smoothed_nll_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor:
    r"""
    Compute unreduced uniformly label-smoothed negative log likelihood.

    Args:
        sequence_tokens: Target token IDs with shape ``(..., N)``.
        log_probabilities: Token log probabilities with shape ``(..., N, V)``.
        weight: Nonnegative smoothing weight added uniformly across the vocabulary.

    Returns:
        Per-position losses with shape ``(..., N)``.
    """

    num_classes = log_probabilities.shape[-1]
    target = F.one_hot(sequence_tokens, num_classes).to(log_probabilities.dtype)
    target = target + weight / num_classes
    target = target / target.sum(dim=-1, keepdim=True)
    return -(target * log_probabilities).sum(dim=-1)


def autoregressive_masks(
    decoding_ranks: Tensor,
    neighbor_indices: Tensor,
    residue_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build backward and forward neighbor masks directly from aligned causal ranks.

    Args:
        decoding_ranks: Per-residue decoding ranks with shape ``(B, N)``.
        neighbor_indices: Selected neighbor indices with shape ``(B, N, K)``.
        residue_mask: Valid-residue mask with shape ``(B, N)``.

    Returns:
        Backward and forward masks with shape ``(B, N, K, 1)``.
    """

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = residue_mask.unsqueeze(-1).unsqueeze(-1)
    return (
        query_mask * backward.unsqueeze(-1),
        query_mask * (~backward).unsqueeze(-1),
    )


class MpnnMessage(nn.Module):
    r"""
    Compute neighbor messages with a three-projection GELU network.

    Args:
        input_size: Width of the assembled neighbor context.
        hidden_size: Width of hidden messages and the returned update.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply the message network to an aligned context tensor.

        Args:
            x: Aligned context features with ``input_size`` channels.

        Returns:
            Message features with ``hidden_size`` channels.
        """

        hidden = self.activation(self.input_projection(x))
        hidden = self.activation(self.hidden_projection(hidden))
        return self.output_projection(hidden)


class MpnnFeedForward(nn.Module):
    r"""
    Position-wise projection, GELU activation, and output projection.

    Residual addition, dropout, and normalization belong to the containing block.

    Args:
        hidden_size: Width of input and output states.
        intermediate_size: Width of the activated intermediate state.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.output_projection = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply the feed-forward computation.

        Args:
            x: Hidden states with ``hidden_size`` channels.

        Returns:
            Feed-forward updates with the same shape as ``x``.
        """

        return self.output_projection(self.activation(self.input_projection(x)))


class MpnnRelativePositionEncoding(nn.Module):
    r"""
    Embed signed residue offsets with one distinct inter-chain bucket.

    Args:
        num_embeddings: Width of the returned relative-position features.
        max_relative_feature: Largest signed within-chain offset represented separately.
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.projection = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""
        Project bucketed residue offsets to edge-position features.

        Args:
            offset: Signed residue offsets.
            same_chain: Binary mask with the same shape as ``offset``.

        Returns:
            Relative-position features with ``num_embeddings`` channels.
        """

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        features = F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.projection.weight.dtype)
        return self.projection(features)


class MpnnEncoderBlock(nn.Module):
    r"""
    Apply one node-message, transition, and edge-message update.

    The block owns residual order, dropout, and post-normalization. Its
    [`MpnnFeedForward`][ofoldx.modules.mpnn.MpnnFeedForward] member computes only
    the transition update.

    Args:
        hidden_size: Width of node, edge, and message states.
        edge_input_size: Width of gathered neighbor context before the source node is appended.
        dropout: Dropout probability on each residual update.
        scale: Divisor applied to the summed neighbor messages.
    """

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = LayerNorm(hidden_size)
        self.transition = MpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)
        self.edge_message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        node_mask: Tensor | None,
        attend_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Update aligned node and edge states from one message-passing round.

        Args:
            node_states: Node states ``[batch, residues, hidden_size]``.
            edge_states: Neighbor-aligned edge states ``[batch, residues, neighbors, hidden_size]``.
            neighbor_indices: Source indices ``[batch, residues, neighbors]`` for each neighbor slot.
            node_mask: Optional valid-node mask ``[batch, residues]``.
            attend_mask: Optional valid-neighbor mask ``[batch, residues, neighbors]``.

        Returns:
            Updated node and edge states with the same shapes as their inputs.
        """

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        node_message = self.node_message(torch.cat((source_states, neighbor_states), dim=-1))
        if attend_mask is not None:
            node_message = node_message * attend_mask.unsqueeze(-1)
        node_states = self.node_norm(node_states + self.node_dropout(node_message.sum(dim=-2) / self.scale))
        node_states = self.transition_norm(node_states + self.transition_dropout(self.transition(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        edge_message = self.edge_message(torch.cat((source_states, neighbor_states), dim=-1))
        edge_states = self.edge_norm(edge_states + self.edge_dropout(edge_message))
        return node_states, edge_states


class MpnnDecoderBlock(nn.Module):
    r"""
    Apply one masked neighbor-message update followed by a transition.

    The block owns both residual paths, dropout, and post-normalization; the
    feed-forward module held as ``transition`` only computes its additive update.

    Args:
        hidden_size: Width of node states and messages.
        edge_input_size: Width of caller-assembled neighbor context.
        dropout: Dropout probability on each residual update.
        scale: Divisor applied to the summed neighbor messages.
    """

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = LayerNorm(hidden_size)
        self.transition = MpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Update node states from caller-assembled neighbor context.

        Args:
            node_states: Node states ``[..., residues, hidden_size]``.
            edge_context: Neighbor context ``[..., residues, neighbors, edge_input_size]``.
            node_mask: Optional valid-node mask ``[..., residues]``.
            attend_mask: Optional valid-neighbor mask ``[..., residues, neighbors]``.

        Returns:
            Updated node states with the same shape as ``node_states``.
        """

        source_states = node_states.unsqueeze(-2).expand(*node_states.shape[:-1], edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.transition_norm(node_states + self.transition_dropout(self.transition(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states


class MpnnProteinFeatures(nn.Module):
    r"""
    Build a 25-distance full-backbone edge graph for MPNN designers.

    Each residue selects nearest neighbors by Cα distance. The edge representation
    combines relative residue position with radial-basis features for 25 ordered
    atom-pair distances over N, Cα, C, O, and virtual Cβ coordinates.

    Args:
        edge_size: Width of returned edge features.
        num_neighbors: Maximum number of neighbors selected per residue.
        num_positional_embeddings: Width of relative-position features.
        num_rbf: Number of radial-basis bins per atom-pair distance.
    """

    def __init__(
        self,
        edge_size: int,
        num_neighbors: int,
        *,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_rbf = num_rbf
        self.relative_position = MpnnRelativePositionEncoding(num_positional_embeddings)
        self.edge_projection = nn.Linear(num_positional_embeddings + 25 * num_rbf, edge_size, bias=False)
        self.edge_norm = LayerNorm(edge_size)

    def forward(
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Build neighbor-aligned geometric and relative-position edge features.

        Args:
            backbone_coordinates: Backbone coordinates ``[batch, residues, 4, 3]`` in N, Cα, C, O order.
            residue_mask: Valid-residue mask ``[batch, residues]``.
            residue_index: Residue indices ``[batch, residues]`` used for signed offsets.
            chain_index: Chain indices ``[batch, residues]`` used to separate inter-chain offsets.

        Returns:
            Edge features ``[batch, residues, neighbors, edge_size]`` and their residue indices
            ``[batch, residues, neighbors]``.
        """

        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        atoms = torch.stack((alpha_carbon, nitrogen, carbon, oxygen, beta_carbon), dim=2)
        neighbor_distances, neighbor_indices = nearest_neighbors(
            alpha_carbon,
            residue_mask,
            self.num_neighbors,
        )

        neighbor_atoms = gather_nodes(atoms.flatten(start_dim=-2), neighbor_indices).reshape(
            *neighbor_indices.shape,
            5,
            3,
        )
        pair_distances = torch.sqrt(
            torch.sum(
                (atoms[:, :, None, :, None] - neighbor_atoms[:, :, :, None]) ** 2,
                dim=-1,
            )
            + 1e-6
        )
        pair_distances = pair_distances[
            ...,
            (0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2),
            (0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3),
        ]
        pair_distances = torch.cat((neighbor_distances.unsqueeze(-1), pair_distances[..., 1:]), dim=-1)
        distance_features = radial_basis(pair_distances, self.num_rbf).flatten(start_dim=-2)

        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices


def periodic_group_period(*, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
    r"""
    Build the periodic-table group/period lookup used for atom element features.

    LigandMPNN one-hot-encodes each context atom by its periodic group and period; this returns the two
    lookup tables indexed by atomic number so ``group[z]`` / ``period[z]`` give the placement of element
    ``z``. Index ``0`` (no atom) and any atomic number past the tabulated main-group elements map to ``0``.

    Args:
        device: Optional device for the lookup tensors.

    Returns:
        Tuple ``(group, period)`` of two `[120]` long tensors indexed by atomic number.
    """

    groups = [0] * 120
    periods = [0] * 120
    rows = [
        (1, [1, 18]),
        (2, [1, 2, 13, 14, 15, 16, 17, 18]),
        (3, [1, 2, 13, 14, 15, 16, 17, 18]),
        (4, list(range(1, 19))),
        (5, list(range(1, 19))),
        (6, [1, 2] + [3] * 15 + list(range(4, 19))),
        (7, [1, 2] + [3] * 15 + list(range(4, 19))),
    ]
    atomic_number = 1
    for period, row_groups in rows:
        for group in row_groups:
            groups[atomic_number] = group
            periods[atomic_number] = period
            atomic_number += 1
    return torch.tensor(groups, dtype=torch.long, device=device), torch.tensor(periods, dtype=torch.long, device=device)


#: Standard amino-acid component IDs in MPNN token order.
AMINO_ACID_COMPONENT_IDS = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)
#: MPNN token index for each standard amino-acid component ID.
AMINO_ACID_TOKEN_BY_COMPONENT = {name: index for index, name in enumerate(AMINO_ACID_COMPONENT_IDS)}
#: MPNN token index reserved for an unknown amino acid.
AMINO_ACID_UNKNOWN_TOKEN = len(AMINO_ACID_COMPONENT_IDS)
