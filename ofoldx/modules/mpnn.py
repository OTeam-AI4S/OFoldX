# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Message-passing neural-network graph of the MPNN protein-design lineage.

The released ProteinMPNN graph and its successors (ligand-aware, nucleic-acid, and side-chain
packing) run the same encoder and decoder over a k-nearest-neighbour residue graph. This module owns
that architecture and ProteinMPNN's full-backbone residue graph, which the ligand-aware model reuses
unchanged. A family supplies any other graph — its neighbour selection, edge features, and decoding
order — from its own ``model.py``.

Every value the released ProteinMPNN fixes is a constructor argument here, so the classes are the
architecture and the defaults carry the lineage.
"""

from collections.abc import Callable, Iterable
from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from .atom.coordinates import project_coordinates
from .backend import LayerNorm, gather_nodes, masked_knn, resolve_config_backend
from .feed_forward import FeedForward
from .functional import radial_basis, weighted_mean
from .geometry import neighbor_atom_distances, virtual_beta_carbon


def cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""
    Append the gathered neighbour-node states to an aligned edge context.

    Args:
        node_states: Node states `[B, L, H]`.
        neighbor_states: Edge states aligned to ``neighbor_indices``, `[B, L, K, C]`.
        neighbor_indices: Neighbour indices `[B, L, K]`.

    Returns:
        The edge context `[B, L, K, C + H]`.
    """

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_indices)), dim=-1)


def validate_decoding_ranks(
    decoding_ranks: Tensor,
    tied_group_ids: Tensor,
) -> None:
    r"""
    Assert the exact explicit-rank tensor and its causal-unit identity.
    """

    if decoding_ranks.shape != tied_group_ids.shape:
        raise ValueError("decoding_ranks has the wrong exact shape")
    if decoding_ranks.dtype != torch.long:
        raise TypeError("decoding_ranks must have dtype torch.long")

    group_order = torch.argsort(tied_group_ids, dim=1)
    groups_by_group = torch.gather(tied_group_ids, 1, group_order)
    ranks_by_group = torch.gather(decoding_ranks, 1, group_order)
    same_group = groups_by_group[:, 1:] == groups_by_group[:, :-1]
    torch._assert(
        torch.all(~same_group | (ranks_by_group[:, 1:] == ranks_by_group[:, :-1])),
        "tied members must have equal decoding ranks",
    )

    rank_order = torch.argsort(decoding_ranks, dim=1)
    ranks_by_rank = torch.gather(decoding_ranks, 1, rank_order)
    groups_by_rank = torch.gather(tied_group_ids, 1, rank_order)
    same_rank = ranks_by_rank[:, 1:] == ranks_by_rank[:, :-1]
    torch._assert(
        torch.all(~same_rank | (groups_by_rank[:, 1:] == groups_by_rank[:, :-1])),
        "distinct causal units must have distinct decoding ranks",
    )


def autoregressive_masks(
    decoding_ranks: Tensor,
    neighbor_indices: Tensor,
    residue_mask: Tensor,
    neighbor_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the backward and forward neighbour masks implied by a decoding order.

    A neighbour is *backward* when it decodes before the query residue, so its already-chosen token
    may be conditioned on; the forward mask is the complement over valid neighbours.

    Args:
        decoding_ranks: Autoregressive rank of each residue `[B, L]`.
        neighbor_indices: Neighbour indices `[B, L, K]`.
        residue_mask: Valid-residue mask `[B, L]`.
        neighbor_valid: Real-neighbour mask `[B, L, K]`, zero for filler slots.

    Returns:
        backward: Mask `[B, L, K, 1]` selecting already-decoded neighbours.
        forward: Mask `[B, L, K, 1]` selecting not-yet-decoded neighbours.
    """

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = residue_mask.unsqueeze(-1).unsqueeze(-1) * neighbor_valid.unsqueeze(-1)
    return (
        query_mask * backward.unsqueeze(-1),
        query_mask * (~backward).unsqueeze(-1),
    )


def decode_sequence_free(
    node_states: Tensor,
    edge_states: Tensor,
    neighbor_indices: Tensor,
    neighbor_valid: Tensor,
    residue_mask: Tensor,
    decoding_ranks: Tensor,
    *,
    decoder_blocks: Iterable[nn.Module],
    output_projection: nn.Module,
    checkpoint: Callable[..., object] | None = None,
) -> Tensor:
    r"""
    Decode vocabulary logits from rank-masked encoder context without sequence identities.

    Each block sees the same initial encoder context. Only valid neighbours whose rank is at least
    the query rank contribute; all-equal ranks retain every valid encoder neighbour. The supplied
    blocks and output projection remain registered on their owning model.

    Args:
        node_states: Encoded nodes ``[B, L, H]``.
        edge_states: Encoded edges ``[B, L, K, H]``.
        neighbor_indices: Neighbour indices ``[B, L, K]``.
        neighbor_valid: Real-neighbour mask ``[B, L, K]``.
        residue_mask: Valid-residue mask ``[B, L]``.
        decoding_ranks: Causal ranks ``[B, L]``; equal ranks belong to the same causal unit.
        decoder_blocks: The model's learned decoder blocks, in order.
        output_projection: The model's learned vocabulary projection.
        checkpoint: The owner's active checkpoint callable, or ``None`` for ordinary execution.

    Returns:
        Vocabulary logits ``[B, L, V]``.
    """

    mask = residue_mask.to(dtype=node_states.dtype)
    encoder_edges = cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_indices)
    encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
    _, forward_mask = autoregressive_masks(
        decoding_ranks, neighbor_indices, mask, neighbor_valid.to(dtype=node_states.dtype)
    )
    forward_context = forward_mask * encoder_context
    states = node_states
    for block in decoder_blocks:
        if checkpoint is None:
            states = block(states, forward_context, mask, neighbor_valid)
        else:
            states = cast(Tensor, checkpoint(block.__call__, states, forward_context, mask, neighbor_valid))
    return output_projection(states)


class MpnnSequenceLoss(nn.Module):
    r"""
    Label-smoothed cross entropy of the observed sequence, averaged over the scored residues.

    This is ProteinMPNN's training loss, which LigandMPNN reuses: the one-hot target gains
    ``label_smoothing / V`` on every class and is renormalized before the cross entropy. The mean runs
    over the scored residues instead of upstream's fixed ``/ 2000``, so a batch's loss does not depend
    on how many residues it holds.

    Args:
        label_smoothing: Smoothing mass added to every class before renormalization.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.mpnn import MpnnSequenceLoss
        >>> loss = MpnnSequenceLoss()
        >>> logits, tokens = torch.zeros(1, 3, 21), torch.tensor([[0, 1, 2]])
        >>> loss(logits, tokens, torch.ones(1, 3, dtype=torch.bool))
        tensor(3.0445)
        >>> loss(logits, tokens, torch.zeros(1, 3, dtype=torch.bool))
        tensor(nan)
    """

    def __init__(self, *, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, sequence_tokens: Tensor, mask: Tensor) -> Tensor:
        r"""
        Args:
            logits: Sequence logits `[B, L, V]`.
            sequence_tokens: Observed token IDs `[B, L]`.
            mask: Boolean scored-residue availability `[B, L]`.

        Returns:
            The mean loss over the scored residues; NaN when there is none.
        """

        if mask.dtype != torch.bool:
            raise TypeError("sequence loss mask must be boolean scored-residue availability")
        observed = mask
        logits = torch.where(observed.unsqueeze(-1), logits, 0.0)
        sequence_tokens = torch.where(observed, sequence_tokens, 0)
        in_vocabulary = ((sequence_tokens >= 0) & (sequence_tokens < logits.shape[-1])).all()
        finite = torch.isfinite(logits)
        available = ((finite | torch.isneginf(logits)).all(dim=-1) & finite.any(dim=-1)).all()
        if torch.compiler.is_compiling():
            torch._assert_async(in_vocabulary, "scored sequence identities must be known vocabulary tokens")
            torch._assert_async(available, "scored logits require at least one finite class and no NaN/+Inf")
        else:
            if not bool(in_vocabulary):
                raise ValueError("scored sequence identities must be known vocabulary tokens")
            if not bool(available):
                raise ValueError("scored logits require at least one finite class and no NaN/+Inf")
        supported = finite.all() if self.label_smoothing > 0 else finite.gather(-1, sequence_tokens.unsqueeze(-1)).all()
        if torch.compiler.is_compiling():
            torch._assert_async(supported, "positive target mass cannot use an excluded class")
        elif not bool(supported):
            raise ValueError("positive target mass cannot use an excluded class")
        logit_values = logits.concat if isinstance(logits, NestedTensor) else logits
        token_values = sequence_tokens.concat if isinstance(sequence_tokens, NestedTensor) else sequence_tokens
        weights = observed.concat if isinstance(observed, NestedTensor) else observed
        per_residue = F.cross_entropy(
            logit_values.reshape(-1, logits.shape[-1]),
            token_values.reshape(-1),
            label_smoothing=self.label_smoothing / (1.0 + self.label_smoothing),
            reduction="none",
        )
        return weighted_mean(per_residue, weights.reshape(-1).to(per_residue.dtype))


class MpnnRelativePositionEncoding(nn.Module):
    r"""
    Embed signed residue offsets, with one extra bucket for inter-chain pairs.

    Offsets are clipped into ``2 * max_relative_feature + 1`` buckets; every pair whose neighbour
    sits on another chain takes the final bucket instead of its numeric offset.

    Args:
        embedding_size: Width of the produced edge-position features.
        max_relative_feature: Largest signed offset represented without clipping.
    """

    def __init__(self, embedding_size: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.projection = nn.Linear(2 * max_relative_feature + 2, embedding_size)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""
        Args:
            offset: Signed residue offset to each neighbour `[B, L, K]`.
            same_chain: ``1`` where the neighbour shares the query's chain, else ``0`` `[B, L, K]`.

        Returns:
            Edge-position features `[B, L, K, embedding_size]`.
        """

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        features = F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.projection.weight.dtype)
        return self.projection(features)


class MpnnProteinFeatureEncoder(nn.Module):
    r"""
    Build ProteinMPNN's k-nearest-neighbour residue graph with full-backbone edge features.

    Each residue contributes five atoms: N, C-alpha, C, O, and a virtual C-beta. An edge embeds the 25
    distances between the two residues' atoms with radial basis functions, next to their relative
    position, and neighbours are the ``num_neighbors`` nearest C-alpha atoms.

    Args:
        edge_size: Width of the produced edge features.
        num_neighbors: Number of nearest-neighbour residues in the graph.
        positional_embedding_size: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance.
        backend: Backend used by neighbour selection and geometry.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.mpnn import MpnnProteinFeatureEncoder
        >>> encoder = MpnnProteinFeatureEncoder(8, 3).eval()
        >>> residue_index = torch.arange(5).unsqueeze(0)
        >>> chain_index = torch.zeros(1, 5, dtype=torch.long)
        >>> edges, neighbors, valid = encoder(torch.randn(1, 5, 4, 3), torch.ones(1, 5), residue_index, chain_index)
        >>> edges.shape, neighbors.shape, valid.shape
        (torch.Size([1, 5, 3, 8]), torch.Size([1, 5, 3]), torch.Size([1, 5, 3]))
    """

    distance_indices: Tensor

    def __init__(
        self,
        edge_size: int,
        num_neighbors: int,
        *,
        positional_embedding_size: int = 16,
        num_rbf: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.num_neighbors = num_neighbors
        self.num_rbf = num_rbf
        self.relative_position = MpnnRelativePositionEncoding(positional_embedding_size)
        self.edge_projection = nn.Linear(positional_embedding_size + 25 * num_rbf, edge_size, bias=False)
        self.edge_norm = LayerNorm(edge_size, backend=backend)
        source_atom_indices = (0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2)
        target_atom_indices = (0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3)
        # A model built on the meta device still needs real indices, since loading never fills them.
        buffer_device = self.edge_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "distance_indices",
            torch.tensor(
                tuple(
                    source * 5 + target for source, target in zip(source_atom_indices, target_atom_indices, strict=True)
                ),
                device=buffer_device,
            ),
            persistent=False,
        )

    def forward(
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Args:
            backbone_coordinates: N, C-alpha, C, and O coordinates `[B, L, 4, 3]`.
            residue_mask: Valid-residue mask `[B, L]`.
            residue_index: Residue indices `[B, L]`.
            chain_index: Chain indices `[B, L]`.

        Returns:
            edges: Edge features `[B, L, K, edge_size]`.
            neighbor_indices: Neighbour indices `[B, L, K]`.
            neighbor_valid: Real-neighbour mask `[B, L, K]`.
        """

        observed = residue_mask.to(torch.bool)[..., None] & torch.ones_like(
            backbone_coordinates[..., 0], dtype=torch.bool
        )
        backbone_coordinates = project_coordinates(
            backbone_coordinates, observed, dtype=self.edge_projection.weight.dtype
        )
        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        atoms = torch.stack((alpha_carbon, nitrogen, carbon, oxygen, beta_carbon), dim=2)
        neighbor_distances, neighbor_indices, neighbor_valid = masked_knn(
            alpha_carbon,
            residue_mask,
            self.num_neighbors,
            backend=self.backend,
        )

        pair_distances = neighbor_atom_distances(atoms, neighbor_indices, eps=1e-6, backend=self.backend)
        pair_distances = pair_distances.flatten(start_dim=-2).index_select(-1, self.distance_indices)
        pair_distances = torch.cat((neighbor_distances.unsqueeze(-1), pair_distances[..., 1:]), dim=-1)
        distance_features = radial_basis(pair_distances, self.num_rbf).flatten(start_dim=-2)

        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features), dim=-1)
        return (
            self.edge_norm(self.edge_projection(edges.to(self.edge_projection.weight.dtype))),
            neighbor_indices,
            neighbor_valid,
        )


class MpnnEncoderBlock(nn.Module):
    r"""
    One encoder block: a node update, a feed-forward update, then an edge update.

    Each message is summed over neighbours and divided by ``scale`` rather than by the neighbour
    count, which is the released normalization.

    Args:
        hidden_size: Width of the node and edge states.
        edge_input_size: Width of the caller's edge context, excluding the source-node state.
        dropout: Dropout probability on each residual update.
        expansion: Transition expansion factor.
        scale: Divisor applied to the summed neighbour messages.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.mpnn import MpnnEncoderBlock
        >>> block = MpnnEncoderBlock(8, 16, dropout=0.0).eval()
        >>> nodes = torch.zeros(1, 4, 8)
        >>> edges = torch.zeros(1, 4, 3, 8)
        >>> neighbors = torch.tensor([[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]])
        >>> nodes, edges = block(nodes, edges, neighbors, None, None)
        >>> nodes.shape, edges.shape
        (torch.Size([1, 4, 8]), torch.Size([1, 4, 3, 8]))
    """

    def __init__(
        self,
        hidden_size: int,
        edge_input_size: int,
        *,
        dropout: float,
        expansion: int = 4,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = FeedForward(
            hidden_size + edge_input_size,
            hidden_size,
            hidden_size,
            num_hidden_layers=2,
            activation="gelu",
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size * expansion, activation="gelu")
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = LayerNorm(hidden_size)
        self.edge_message = FeedForward(
            hidden_size + edge_input_size,
            hidden_size,
            hidden_size,
            num_hidden_layers=2,
            activation="gelu",
        )
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
        Update aligned node and edge states.

        Args:
            node_states: Node states `[B, L, hidden_size]`.
            edge_states: Edge states `[B, L, K, C]`.
            neighbor_indices: Neighbour indices `[B, L, K]`.
            node_mask: Optional valid-residue mask `[B, L]`.
            attend_mask: Optional real-neighbour mask `[B, L, K]`.

        Returns:
            node_states: Updated node states `[B, L, hidden_size]`.
            edge_states: Updated edge states `[B, L, K, hidden_size]`.
        """

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


class MpnnDecoderBlock(nn.Module):
    r"""
    One decoder block: a masked neighbour-message update, then a feed-forward update.

    The caller assembles the neighbour context, so this block is agnostic to whether the context
    carries already-decoded tokens, ligand atoms, or side-chain state.

    Args:
        hidden_size: Width of the node states.
        edge_input_size: Width of the caller-assembled neighbour context.
        dropout: Dropout probability on each residual update.
        expansion: Transition expansion factor.
        scale: Divisor applied to the summed neighbour messages.
    """

    def __init__(
        self,
        hidden_size: int,
        edge_input_size: int,
        *,
        dropout: float,
        expansion: int = 4,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.message = FeedForward(
            hidden_size + edge_input_size, hidden_size, hidden_size, num_hidden_layers=2, activation="gelu"
        )
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size * expansion, activation="gelu")
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Update node states from a caller-assembled neighbour context.

        Args:
            node_states: Node states `[B, L, hidden_size]`.
            edge_context: Neighbour context `[B, L, K, edge_input_size]`.
            node_mask: Optional valid-residue mask `[B, L]`.
            attend_mask: Optional neighbour mask `[B, L, K]`, typically autoregressive.

        Returns:
            Updated node states `[B, L, hidden_size]`.
        """

        source_states = node_states.unsqueeze(-2).expand(*([-1] * (node_states.ndim - 1)), edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.feed_forward_norm(node_states + self.feed_forward_dropout(self.feed_forward(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states
