r"""
Reusable graph message-passing components for MPNN models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backend import LayerNorm
from .geometry import virtual_beta_carbon

if TYPE_CHECKING:
    from danling import NestedTensor


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

    channels = nodes.shape[-1]
    neighbors = neighbor_indices.shape[-1]
    # Gather along the residue axis (ragged for a NestedTensor batch) without flattening ``N*K`` -- that
    # flatten assumes a rectangular batch. ``expand(-1, ...)`` keeps the batch and residue axes as they are,
    # so one formulation serves both dense and ragged inputs: out[b, i, k, c] = nodes[b, neighbor[b, i, k], c].
    return torch.gather(
        nodes[:, :, None, :].expand(-1, -1, neighbors, -1),
        1,
        neighbor_indices[..., None].expand(-1, -1, -1, channels),
    )


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
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return masked nearest-neighbor distances, indices, and a real-neighbor mask.

    Exactly ``num_neighbors`` neighbors are selected per residue. A structure shorter than ``num_neighbors``
    fills the remaining slots with masked padding rows -- never a real neighbor -- so the neighbor count is
    uniform and a residue's real neighbors are its own nearest valid residues regardless of what shared the
    batch (topology-invariant). The selection is computed densely (a ragged ``NestedTensor`` batch is unpacked
    to dense rows padded to at least ``num_neighbors``, its own padding masked out) and re-nested; the message
    passing stays ragged and masks the filler slots via the returned ``neighbor_valid``.

    Args:
        coordinates: Residue coordinates ``(B, N, 3)`` -- a ragged ``NestedTensor`` batch or a dense tensor.
        residue_mask: Valid-residue mask ``(B, N)``.
        num_neighbors: Number of neighbors selected per residue.
        eps: Additive epsilon inside the distance square root.

    Returns:
        Neighbor distances, indices, and validity, each ``(B, N, num_neighbors)``. ``neighbor_valid`` is 1 for
        a real neighbor and 0 for a filler slot -- it is exactly the encoder's ``attend_mask``.
    """

    from danling import NestedTensor as RuntimeNestedTensor

    ragged = isinstance(coordinates, RuntimeNestedTensor)
    if ragged:
        nested_coordinates = cast("NestedTensor", coordinates)
        nested_mask = cast("NestedTensor", residue_mask)
        lengths = [int(row.shape[0]) for row in nested_coordinates]
        coords, mask = nested_coordinates.tensor, nested_mask.tensor
    else:
        coords, mask = coordinates, residue_mask
    # Guarantee at least ``num_neighbors`` candidate columns per structure so the single top-k always fits;
    # the extra rows are masked, so they can only ever be selected as filler.
    shortfall = num_neighbors - coords.shape[1]
    if shortfall > 0:
        coords = F.pad(coords, (0, 0, 0, shortfall))
        mask = F.pad(mask, (0, shortfall))
    pair_mask = mask[:, :, None] * mask[:, None, :]
    distances = torch.sqrt(torch.sum((coords[:, :, None] - coords[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    # Push filler (masked) pairs strictly past the farthest real neighbor (``+ 1`` Angstrom), never merely
    # tie with it, so ``topk`` can never displace a genuine neighbor with a filler slot.
    maximum = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * (maximum + 1.0)
    neighbor_distances, neighbor_indices = torch.topk(adjusted, num_neighbors, dim=-1, largest=False)
    # A neighbor is real only when both it and the query residue are valid; a filler (masked padding row) is 0.
    neighbor_valid = gather_nodes(mask.unsqueeze(-1), neighbor_indices).squeeze(-1) * mask[:, :, None]
    if ragged:
        count = len(lengths)
        neighbor_distances = RuntimeNestedTensor([neighbor_distances[i, : lengths[i]] for i in range(count)])
        # Real neighbors index into the structure's own residues; clamp filler indices back in range (they are
        # masked by ``neighbor_valid``, so the clamped position never contributes).
        neighbor_indices = RuntimeNestedTensor(
            [neighbor_indices[i, : lengths[i]].clamp(max=max(lengths[i] - 1, 0)) for i in range(count)]
        )
        neighbor_valid = RuntimeNestedTensor([neighbor_valid[i, : lengths[i]] for i in range(count)])
    elif shortfall > 0:
        # A dense structure shorter than ``num_neighbors``: drop the padded query rows and clamp filler indices
        # back into the real residue range (masked by ``neighbor_valid``).
        length = coordinates.shape[1]
        neighbor_distances = neighbor_distances[:, :length]
        neighbor_indices = neighbor_indices[:, :length].clamp(max=length - 1)
        neighbor_valid = neighbor_valid[:, :length]
    return neighbor_distances, neighbor_indices, neighbor_valid


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
    width = (upper_bound - lower_bound) / num_bins
    # ``centers`` stays 1-D and broadcasts against the appended trailing axis of ``distances[..., None]``.
    # An explicit reshape to ``distances.ndim`` dims would build a dense tensor that cannot broadcast against
    # a ragged NestedTensor's packed values; the trailing broadcast is equivalent for a dense ``distances``.
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


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
    neighbor_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build backward and forward neighbor masks directly from aligned causal ranks.

    Args:
        decoding_ranks: Per-residue decoding ranks with shape ``(B, N)``.
        neighbor_indices: Selected neighbor indices with shape ``(B, N, K)``.
        residue_mask: Valid-residue mask with shape ``(B, N)``.
        neighbor_valid: Real-neighbor mask with shape ``(B, N, K)`` (0 for a filler slot padded in when a
            structure is shorter than ``num_neighbors``). Filler slots must contribute no causal message,
            exactly as they are masked in the encoder. For a structure at least ``num_neighbors`` long this is
            all ones, so the masks are unchanged.

    Returns:
        Backward and forward masks with shape ``(B, N, K, 1)``.
    """

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = residue_mask.unsqueeze(-1).unsqueeze(-1) * neighbor_valid.unsqueeze(-1)
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
        self.activation = nn.GELU()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

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
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(intermediate_size, hidden_size)

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

        source_states = node_states.unsqueeze(-2).expand(*([-1] * (node_states.ndim - 1)), edge_context.shape[-2], -1)
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Build neighbor-aligned geometric and relative-position edge features.

        Args:
            backbone_coordinates: Backbone coordinates ``[batch, residues, 4, 3]`` in N, Cα, C, O order.
            residue_mask: Valid-residue mask ``[batch, residues]``.
            residue_index: Residue indices ``[batch, residues]`` used for signed offsets.
            chain_index: Chain indices ``[batch, residues]`` used to separate inter-chain offsets.

        Returns:
            Edge features ``[batch, residues, neighbors, edge_size]``, their residue indices
            ``[batch, residues, neighbors]``, and the real-neighbor mask ``[batch, residues, neighbors]`` (the
            encoder's ``attend_mask``: 1 for a real neighbor, 0 for a filler slot).
        """

        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        atoms = torch.stack((alpha_carbon, nitrogen, carbon, oxygen, beta_carbon), dim=2)
        neighbor_distances, neighbor_indices, neighbor_valid = nearest_neighbors(
            alpha_carbon,
            residue_mask,
            self.num_neighbors,
        )

        neighbor_atoms = gather_nodes(atoms.flatten(start_dim=-2), neighbor_indices).unflatten(-1, (5, 3))
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
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices, neighbor_valid


def periodic_group_period(*, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
    r"""
    Build the periodic-table group/period lookup used for atom element features.

    LigandMPNN and MPNN Packer one-hot-encode context atoms by periodic group and period; this returns
    the two lookup tables indexed by atomic number so ``group[z]`` / ``period[z]`` give the placement of
    element ``z``. Index ``0`` (no atom) and any atomic number past the tabulated main-group elements map
    to ``0``.

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
