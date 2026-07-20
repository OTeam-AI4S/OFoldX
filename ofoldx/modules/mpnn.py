r"""
Concrete graph primitives shared by the released MPNN families.

This module owns the exact message-passing blocks, feature graphs, aligned tensor
carriers, and helper functions reused by more than one released MPNN family.
Concrete Models, Inputs, Configs, samplers, Pipelines, and family-specific feature
graphs remain with their families.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(eq=False)
class MpnnOutput:
    r"""Raw logits and exact causal ranks from an MPNN sequence graph."""

    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""Return vocabulary log probabilities without storing a duplicate tensor."""

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""Return vocabulary probabilities without storing a duplicate tensor."""

        return self.logits.softmax(dim=-1)


@dataclass(eq=False)
class MpnnSamplingOutput:
    r"""Complete aligned tensor output from an MPNN sequence sampler."""

    sequences: Tensor
    log_probabilities: Tensor
    sampling_probabilities: Tensor
    decoding_ranks: Tensor


def gather_edges(edges: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""Gather each residue's selected neighbors from a dense pair tensor."""

    indices = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, edges.shape[-1])
    return torch.gather(edges, 2, indices)


def gather_nodes(nodes: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""Gather node features along every residue's neighbor list."""

    indices = neighbor_indices.reshape(neighbor_indices.shape[0], -1, 1).expand(
        -1,
        -1,
        nodes.shape[-1],
    )
    return torch.gather(nodes, 1, indices).reshape(*neighbor_indices.shape, nodes.shape[-1])


def cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""Concatenate gathered neighbor nodes onto aligned edge or context states."""

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_indices)), dim=-1)


def _nearest_neighbors(
    coordinates: Tensor,
    valid_mask: Tensor,
    num_neighbors: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""Return the released masked nearest-neighbor distances and indices."""

    pair_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
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


def _radial_basis(
    distances: Tensor,
    num_bins: int = 16,
    lower_bound: float = 2.0,
    upper_bound: float = 22.0,
) -> Tensor:
    r"""Expand distances over evenly spaced Gaussian radial-basis centers."""

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


def _virtual_beta_carbon(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""Reconstruct the idealized protein Cβ coordinate from N, Cα, and C."""

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    return -0.58273431 * torch.cross(b, c, dim=-1) + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def _decoding_order_to_ranks(decoding_order: Tensor) -> Tensor:
    r"""Convert a residue permutation to aligned lower-is-earlier causal ranks."""

    ranks = torch.empty_like(decoding_order)
    ranks.scatter_(
        -1,
        decoding_order,
        torch.arange(decoding_order.shape[-1], device=decoding_order.device).expand_as(decoding_order),
    )
    return ranks


def _autoregressive_masks(
    decoding_ranks: Tensor,
    neighbor_indices: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Build backward and forward neighbor masks directly from aligned causal ranks."""

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)
    return (
        query_mask * backward.unsqueeze(-1),
        query_mask * (~backward).unsqueeze(-1),
    )


class MpnnMessage(nn.Module):
    r"""Three-layer GELU network used to compute neighbor messages."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.middle = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the message network to an aligned context tensor."""

        return self.output(self.activation(self.middle(self.activation(self.input(x)))))


class MpnnFeedForward(nn.Module):
    r"""Position-wise two-layer GELU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the feed-forward block."""

        return self.output(self.activation(self.input(x)))


class MpnnRelativePositionEncoding(nn.Module):
    r"""Embed signed residue offsets with one distinct inter-chain bucket."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""Project bucketed residue offsets to edge-position features."""

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        return self.linear(F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.linear.weight.dtype))


class MpnnEncoderLayer(nn.Module):
    r"""One neighbor-message node, feed-forward, and edge update."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = MpnnFeedForward(hidden_size, hidden_size * 4)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.edge_message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        node_mask: Tensor | None,
        attend_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        r"""Update aligned node and edge states from one message-passing round."""

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        node_message = self.node_message(torch.cat((source_states, neighbor_states), dim=-1))
        if attend_mask is not None:
            node_message = node_message * attend_mask.unsqueeze(-1)
        node_states = self.node_norm(node_states + self.node_dropout(node_message.sum(dim=-2) / self.scale))
        node_states = self.feed_forward_norm(node_states + self.feed_forward_dropout(self.feed_forward(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        edge_message = self.edge_message(torch.cat((source_states, neighbor_states), dim=-1))
        edge_states = self.edge_norm(edge_states + self.edge_dropout(edge_message))
        return node_states, edge_states


class MpnnDecoderLayer(nn.Module):
    r"""One masked neighbor-message node update followed by feed-forward."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.message = MpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = MpnnFeedForward(hidden_size, hidden_size * 4)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Update node states from caller-assembled neighbor context."""

        source_states = node_states.unsqueeze(-2).expand(*node_states.shape[:-1], edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.feed_forward_norm(node_states + self.feed_forward_dropout(self.feed_forward(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states


class MpnnProteinFeatures(nn.Module):
    r"""The shared 25-pair full-protein edge graph used by MPNN designers."""

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
        self.edge_norm = nn.LayerNorm(edge_size)

    def forward(
        self,
        backbone_coordinates: Tensor,
        valid_mask: Tensor,
        residue_indices: Tensor,
        chain_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Build vectorized atom-pair and directly gathered identity edge features."""

        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = _virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        atoms = torch.stack((alpha_carbon, nitrogen, carbon, oxygen, beta_carbon), dim=2)
        neighbor_distances, neighbor_indices = _nearest_neighbors(
            alpha_carbon,
            valid_mask,
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
        distance_features = _radial_basis(pair_distances, self.num_rbf).flatten(start_dim=-2)

        neighbor_residue_indices = gather_nodes(residue_indices.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_indices.unsqueeze(-1) - neighbor_residue_indices
        neighbor_chain_indices = gather_nodes(chain_indices.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_indices.unsqueeze(-1) == neighbor_chain_indices).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices


def periodic_group_period(*, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
    r"""
    Build the periodic-table group/period lookup used for atom element features.

    LigandMPNN one-hot-encodes each context atom by its periodic group and period; this returns the two
    lookup tables indexed by atomic number so ``group[z]`` / ``period[z]`` give the placement of element
    ``z``. Index ``0`` (no atom) and any atomic number past the tabulated main-group elements map to ``0``.

    Returns:
        Tuple ``(group, period)`` of two `[120]` long tensors indexed by atomic number.

    Examples:
        >>> group, period = periodic_group_period()
        >>> group.shape
        torch.Size([120])
        >>> (group[6].item(), period[6].item())  # carbon
        (14, 2)
        >>> (group[8].item(), period[8].item())  # oxygen
        (16, 2)
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
AMINO_ACID_TOKEN_BY_COMPONENT = {name: index for index, name in enumerate(AMINO_ACID_COMPONENT_IDS)}
AMINO_ACID_UNKNOWN_TOKEN = len(AMINO_ACID_COMPONENT_IDS)
