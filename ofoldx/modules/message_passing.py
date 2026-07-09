r"""
ProteinMPNN-style message-passing graph blocks.

The neighbor message-passing encoder/decoder layers, the relative-position edge encoding, and the
index-gather helpers shared by the MPNN inverse-folding families (proteinmpnn, nampnn). Kept here -- not
in one family's ``model.py`` -- because they are used by >=2 families; ligandmpnn keeps its own copies by
design. Pure ``torch``/``nn`` with no operator-backend or family dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProteinEncoderLayer(nn.Module):
    r"""
    One ProteinMPNN encoder layer: neighbor-message node update, feed-forward, then edge update.

    Aggregates messages from each residue's neighbors (scaled by ``scale``) with residual LayerNorm updates, applies
    a position-wise feed-forward block, and finally refreshes the edge states from the updated node states.

    Args:
        hidden_size: Node and edge hidden width.
        edge_input_size: Combined neighbor-context width fed to the message MLPs.
        dropout: Dropout probability on the message and feed-forward paths.
        scale: Divisor applied to the summed neighbor messages (upstream neighbor-count normalizer).
    """

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = MessageMLP(hidden_size + edge_input_size, hidden_size)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size * 4)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.edge_message = MessageMLP(hidden_size + edge_input_size, hidden_size)
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        node_mask: Tensor | None,
        attend_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Update node and edge states from one round of neighbor message passing.

        Args:
            node_states: Node states `[B, L, hidden_size]`.
            edge_states: Edge states `[B, L, K, hidden_size]`.
            neighbor_idx: Neighbor index `[B, L, K]`.
            node_mask: Optional valid-residue mask `[B, L]` zeroing padded nodes.
            attend_mask: Optional edge attend mask `[B, L, K]` zeroing messages from invalid neighbors.

        Returns:
            Updated ``(node_states, edge_states)`` with the same shapes as the inputs.
        """

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_idx)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        node_message = self.node_message(torch.cat((source_states, neighbor_states), dim=-1))
        if attend_mask is not None:
            node_message = node_message * attend_mask.unsqueeze(-1)
        node_states = self.node_norm(node_states + self.node_dropout(node_message.sum(dim=-2) / self.scale))
        node_states = self.feed_forward_norm(node_states + self.feed_forward_dropout(self.feed_forward(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)

        neighbor_states = cat_neighbors_nodes(node_states, edge_states, neighbor_idx)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        edge_message = self.edge_message(torch.cat((source_states, neighbor_states), dim=-1))
        edge_states = self.edge_norm(edge_states + self.edge_dropout(edge_message))
        return node_states, edge_states


class ProteinDecoderLayer(nn.Module):
    r"""
    One ProteinMPNN decoder layer: masked neighbor-message node update then feed-forward.

    Consumes the caller-assembled autoregressive edge context (backward decoder context plus forward encoder
    context), aggregates neighbor messages with a residual LayerNorm update, and applies a feed-forward block.

    Args:
        hidden_size: Node hidden width.
        edge_input_size: Combined edge-context width fed to the message MLP.
        dropout: Dropout probability on the message and feed-forward paths.
        scale: Divisor applied to the summed neighbor messages.
    """

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.message = MessageMLP(hidden_size + edge_input_size, hidden_size)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size * 4)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Update node states from the autoregressive edge context.

        Args:
            node_states: Node states `[B, L, hidden_size]`.
            edge_context: Assembled per-neighbor context `[B, L, K, edge_input_size]`.
            node_mask: Optional valid-residue mask `[B, L]` zeroing padded nodes.
            attend_mask: Optional per-neighbor mask zeroing disallowed messages.

        Returns:
            Updated node states `[B, L, hidden_size]`.
        """

        source_states = node_states.unsqueeze(-2).expand(-1, -1, edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.feed_forward_norm(node_states + self.feed_forward_dropout(self.feed_forward(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states


class MessageMLP(nn.Module):
    r"""
    Three-layer GELU message network used to compute neighbor messages.

    Args:
        input_size: Width of the concatenated source/neighbor context input.
        hidden_size: Width of the two hidden layers and the message output.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.middle = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply the three-layer GELU MLP to a message context tensor.

        Args:
            x: Message context `[..., input_size]`.

        Returns:
            The message `[..., hidden_size]`.
        """

        return self.output(self.activation(self.middle(self.activation(self.input(x)))))


class FeedForward(nn.Module):
    r"""
    Position-wise two-layer GELU feed-forward block.

    Args:
        hidden_size: Input and output width.
        intermediate_size: Width of the expanded hidden layer.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply the feed-forward block.

        Args:
            x: Input `[..., hidden_size]`.

        Returns:
            Output `[..., hidden_size]`.
        """

        return self.output(self.activation(self.input(x)))


class RelativePositionEncoding(nn.Module):
    r"""
    Encode the signed residue offset between edge endpoints, with a distinct inter-chain bucket.

    Clamps the offset into ``[-max_relative_feature, max_relative_feature]`` (one-hot bucketed), routes cross-chain
    pairs to a dedicated bucket, and linearly projects to the positional-embedding width.

    Args:
        num_embeddings: Output width of the positional embedding.
        max_relative_feature: Maximum absolute residue offset represented before clamping.
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""
        Embed per-edge residue offsets, using the inter-chain bucket where ``same_chain`` is zero.

        Args:
            offset: Signed residue-index offsets `[B, L, K]`.
            same_chain: Binary same-chain indicator `[B, L, K]`.

        Returns:
            Positional embeddings `[B, L, K, num_embeddings]`.
        """

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        return self.linear(F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.linear.weight.dtype))


def gather_edges(edges: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Select each residue's edge features to its neighbors from a dense `[B, L, L, C]` edge tensor.

    Args:
        edges: Dense pairwise edge features `[B, L, L, C]`.
        neighbor_idx: Neighbor index `[B, L, K]` with values in ``[0, L)``.

    Returns:
        Neighbor edge features `[B, L, K, C]`.

    Examples:
        >>> import torch
        >>> edges = torch.arange(18.0).reshape(1, 3, 3, 2)
        >>> neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        >>> gather_edges(edges, neighbor_idx).shape
        torch.Size([1, 3, 2, 2])
    """

    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.shape[-1])
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Gather neighbor node features for every residue.

    Args:
        nodes: Node features `[B, L, C]`.
        neighbor_idx: Neighbor index `[B, L, K]` with values in ``[0, L)``.

    Returns:
        Neighbor node features `[B, L, K, C]`.

    Examples:
        >>> import torch
        >>> nodes = torch.arange(6.0).reshape(1, 3, 2)
        >>> neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        >>> gather_nodes(nodes, neighbor_idx).shape
        torch.Size([1, 3, 2, 2])
    """

    flat_idx = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    flat_idx = flat_idx.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
    gathered = torch.gather(nodes, 1, flat_idx)
    return gathered.view(*neighbor_idx.shape, nodes.shape[-1])


def cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Concatenate gathered neighbor node states onto per-neighbor edge/context states.

    Args:
        node_states: Node states `[B, L, C_n]` to gather over neighbors.
        neighbor_states: Existing per-neighbor states `[B, L, K, C_e]`.
        neighbor_idx: Neighbor index `[B, L, K]` with values in ``[0, L)``.

    Returns:
        Concatenated per-neighbor context `[B, L, K, C_e + C_n]`.

    Examples:
        >>> import torch
        >>> nodes = torch.arange(6.0).reshape(1, 3, 2)
        >>> neighbor_states = torch.zeros(1, 3, 2, 2)
        >>> neighbor_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        >>> cat_neighbors_nodes(nodes, neighbor_states, neighbor_idx).shape
        torch.Size([1, 3, 2, 4])
    """

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_idx)), dim=-1)
