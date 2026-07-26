r"""
Released OInvFold polymer and ligand graph models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ...data.molecular import AtomSite, ComponentReference
from ...data.structure import Structure
from ...model import Model
from ...modules.backend import LayerNorm
from .config import OInvFoldLigandConfig, OInvFoldPolymerConfig

#: Unified OInvFold polymer vocabulary in checkpoint-column order.
OINVFOLD_POLYMER_TOKENS = tuple("ACDEFGHIKLMNPQRSTVWY") + ("A", "C", "G", "T", "U")

#: OInvFold ligand element vocabulary in checkpoint-column order.
OINVFOLD_LIGAND_ELEMENTS = (
    "C",
    "O",
    "N",
    "P",
    "S",
    "Fe",
    "Os",
    "F",
    "Mg",
    "Cl",
    "Br",
    "W",
    "B",
    "Co",
    "I",
    "Mo",
    "As",
    "Be",
    "Mn",
    "Cu",
    "Ta",
    "V",
    "Al",
    "Ir",
    "Hg",
    "Se",
    "Ni",
    "Ru",
    "D",
    "Pt",
    "Ca",
    "Re",
    "Zn",
    "Si",
    "RARE",
    "UNK",
    "*",
)


class OInvFoldPolymerModel(Model):
    r"""
    OInvFold inverse folding for protein, DNA, or RNA backbones.
    """

    config_class = OInvFoldPolymerConfig
    config: OInvFoldPolymerConfig

    def __init__(self, config: OInvFoldPolymerConfig | None = None) -> None:
        config = OInvFoldPolymerConfig() if config is None else config
        super().__init__(config)
        self.node_embedding = OInvFoldNormalizedReluFeedForward(
            2,
            config.node_feature_size,
            config.hidden_size,
            config.hidden_size,
        )
        self.edge_embedding = OInvFoldNormalizedReluFeedForward(
            2,
            config.edge_feature_size,
            config.hidden_size,
            config.hidden_size,
        )
        self.virtual_embedding = nn.Embedding(
            config.num_virtual_embeddings,
            config.hidden_size,
        )
        self.encoder = OInvFoldEncoder(
            config.hidden_size,
            config.num_encoder_blocks,
            config.num_virtual_atoms,
            config.num_attention_heads,
            config.num_geometric_layers,
            config.num_edge_layers,
            config.dropout,
        )
        self.decoder = OInvFoldPolymerDecoder(
            config.hidden_size,
            config.decoder_hidden_size,
            config.vocabulary_size,
        )
        self.autoregressive_decoder = OInvFoldAutoregressiveDecoder(
            config.hidden_size,
            config.vocabulary_size,
            num_blocks=4,
            num_heads=8,
            dropout=config.dropout,
            max_positions=8192,
        )
        self.post_init()

    def initialize_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        input: OInvFoldPolymerInput,
        *,
        mask_probability: float = 0.1,
    ) -> Tensor:
        r"""
        Encode one polymer graph, applying the released corruption in training mode.
        """

        return _encode_graph(
            input.graph,
            self.node_embedding,
            self.edge_embedding,
            self.virtual_embedding,
            self.encoder,
            mask_probability,
        )

    def forward(
        self,
        input: OInvFoldPolymerInput,
        *,
        mask_probability: float = 0.1,
    ) -> OInvFoldOutput:
        node = self.encode(input, mask_probability=mask_probability)
        return self.decoder(node)

    def autoregressive_probabilities(
        self,
        input: OInvFoldPolymerInput,
        *,
        mask_probability: float = 0.1,
    ) -> OInvFoldOutput:
        r"""
        Predict polymer identities with the released teacher-forced causal decoder.
        """

        sequence_tokens = input.sequence_tokens
        if sequence_tokens is None:
            raise ValueError("OInvFold autoregressive decoding requires canonical sequence tokens")
        node = self.encode(input, mask_probability=mask_probability)
        segment_start = torch.ones_like(input.chain_index, dtype=torch.bool)
        segment_start[1:] = (input.graph.batch_index[1:] != input.graph.batch_index[:-1]) | (
            input.chain_index[1:] != input.chain_index[:-1]
        )
        starts = torch.nonzero(segment_start, as_tuple=False).flatten()
        ends = torch.cat((starts[1:], starts.new_tensor((node.shape[0],))))
        boundaries = tuple(zip(starts.tolist(), ends.tolist(), strict=True))
        node_segments = [node[start:end] for start, end in boundaries]
        token_segments = [sequence_tokens[start:end] for start, end in boundaries]
        padded_node = pad_sequence(node_segments, batch_first=True)
        padded_tokens = pad_sequence(token_segments, batch_first=True)
        lengths = starts.new_tensor([end - start for start, end in boundaries])
        valid_mask = torch.arange(
            padded_node.shape[1],
            device=node.device,
        ).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        padded_logits = self.autoregressive_decoder(
            padded_node,
            padded_tokens,
            valid_mask,
        )
        logits = padded_logits[valid_mask]
        return OInvFoldOutput(
            logits=logits,
            log_probabilities=F.log_softmax(logits, dim=-1),
        )


class OInvFoldLigandModel(Model):
    r"""
    OInvFold atom-element assignment for a coordinate-defined ligand graph.
    """

    config_class = OInvFoldLigandConfig
    config: OInvFoldLigandConfig

    def __init__(self, config: OInvFoldLigandConfig | None = None) -> None:
        config = OInvFoldLigandConfig() if config is None else config
        super().__init__(config)
        self.node_embedding = OInvFoldNormalizedReluFeedForward(
            2,
            config.node_feature_size,
            config.hidden_size,
            config.hidden_size,
        )
        self.edge_embedding = OInvFoldNormalizedReluFeedForward(
            2,
            config.edge_feature_size,
            config.hidden_size,
            config.hidden_size,
        )
        self.virtual_embedding = nn.Embedding(
            config.num_virtual_embeddings,
            config.hidden_size,
        )
        self.encoder = OInvFoldEncoder(
            config.hidden_size,
            config.num_encoder_blocks,
            config.num_virtual_atoms,
            config.num_attention_heads,
            config.num_geometric_layers,
            config.num_edge_layers,
            config.dropout,
        )
        self.decoder = OInvFoldLigandDecoder(
            config.hidden_size,
            config.vocabulary_size,
        )
        self.post_init()

    def initialize_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        input: OInvFoldLigandInput,
        *,
        mask_probability: float = 0.1,
    ) -> Tensor:
        r"""
        Encode one ligand graph, applying the released corruption in training mode.
        """

        return _encode_graph(
            input.graph,
            self.node_embedding,
            self.edge_embedding,
            self.virtual_embedding,
            self.encoder,
            mask_probability,
        )

    def forward(
        self,
        input: OInvFoldLigandInput,
        *,
        mask_probability: float = 0.1,
    ) -> OInvFoldOutput:
        node = self.encode(input, mask_probability=mask_probability)
        output = self.decoder(node)
        if input.vocabulary_logit_mask is None:
            return output
        logits = output.logits + input.vocabulary_logit_mask.to(output.logits.dtype)
        return OInvFoldOutput(
            logits=logits,
            log_probabilities=F.log_softmax(logits, dim=-1),
        )


class OInvFoldEncoder(nn.Module):
    r"""
    Homogeneous graph-block stack with released learned layer aggregation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        num_virtual_atoms: int,
        num_attention_heads: int,
        num_geometric_layers: int,
        num_edge_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                OInvFoldEncoderBlock(
                    hidden_size,
                    num_virtual_atoms,
                    num_attention_heads,
                    num_geometric_layers,
                    num_edge_layers,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_gate = nn.Linear(hidden_size, 1)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_rotations: Tensor,
        edge_translations: Tensor,
        edge_rbf: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        num_real_nodes: int,
        mask_probability: float,
    ) -> Tensor:
        outputs = []
        for block in self.blocks:
            node, edge = block(
                node,
                edge,
                edge_rotations,
                edge_translations,
                edge_rbf,
                edge_index,
                batch_index,
                mask_probability,
            )
            outputs.append(node[:num_real_nodes])
        stacked = torch.stack(outputs, dim=1)
        weights = torch.sigmoid(self.layer_gate(stacked))
        return (stacked * weights).sum(dim=1)


class OInvFoldEncoderBlock(nn.Module):
    r"""
    One homogeneous OInvFold graph-encoder block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_virtual_atoms: int,
        num_attention_heads: int,
        num_geometric_layers: int,
        num_edge_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Embedding(2, hidden_size)
        self.geometric_edge_update = OInvFoldGeometricEdgeUpdate(
            hidden_size,
            num_virtual_atoms,
            num_geometric_layers,
            dropout,
        )
        self.attention = OInvFoldGraphAttention(hidden_size, num_attention_heads)
        self.node_update = OInvFoldNodeUpdate(hidden_size)
        self.edge_update = OInvFoldEdgeUpdate(
            hidden_size,
            num_edge_layers,
            dropout,
        )

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_rotations: Tensor,
        edge_translations: Tensor,
        edge_rbf: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        mask_probability: float,
    ) -> tuple[Tensor, Tensor]:
        if self.training:
            node = node.clone()
            edge = edge.clone()
            node_indices = torch.randperm(
                node.shape[0],
                device=node.device,
            )[: int(node.shape[0] * mask_probability)]
            edge_indices = torch.randperm(
                edge.shape[0],
                device=edge.device,
            )[: int(edge.shape[0] * mask_probability)]
            node[node_indices] = self.mask_token.weight[0]
            edge[edge_indices] = self.mask_token.weight[1]
        edge = self.geometric_edge_update(
            node,
            edge,
            edge_rotations,
            edge_translations,
            edge_rbf,
            edge_index,
        )
        node = self.attention(node, edge, edge_index)
        node = self.node_update(node, batch_index)
        edge = self.edge_update(node, edge, edge_index)
        return node, edge


class OInvFoldGeometricEdgeUpdate(nn.Module):
    r"""
    Update edge states from learned virtual geometry.
    """

    def __init__(
        self,
        hidden_size: int,
        num_virtual_atoms: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_virtual_atoms = num_virtual_atoms
        self.node_vector_projection = nn.Linear(
            hidden_size,
            num_virtual_atoms * 3,
        )
        self.edge_vector_projection = nn.Linear(
            hidden_size,
            num_virtual_atoms * 3,
        )
        self.geometry_transition = OInvFoldNormalizedReluFeedForward(
            num_layers,
            4 * num_virtual_atoms * 3 + 9 + 16 + num_virtual_atoms,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_rotations: Tensor,
        edge_translations: Tensor,
        edge_rbf: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        num_edges = source.shape[0]
        virtual_node = self.node_vector_projection(node).view(
            node.shape[0],
            self.num_virtual_atoms,
            3,
        )
        virtual_edge = self.edge_vector_projection(edge).view(
            num_edges,
            self.num_virtual_atoms,
            3,
        )

        source_vectors = torch.cat((virtual_edge, virtual_node[source]), dim=1)
        transformed_vectors = _apply_transform(edge_rotations, edge_translations, source_vectors)
        rotated_source = torch.einsum("eij,enj->eni", edge_rotations, virtual_node[source])
        directional_correlation = torch.einsum(
            "enj,enj->en",
            virtual_node[destination],
            rotated_source,
        )
        geometric = torch.cat(
            (
                source_vectors.reshape(num_edges, -1),
                transformed_vectors.reshape(num_edges, -1),
                edge_rotations.reshape(num_edges, 9),
                edge_rbf,
                directional_correlation,
            ),
            dim=-1,
        )
        update = self.geometry_transition(geometric)
        return self.output_projection(torch.cat((edge, update), dim=-1))


class OInvFoldGraphAttention(nn.Module):
    r"""
    Source-grouped multi-head edge attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.logit_input_projection = nn.Linear(
            3 * hidden_size,
            hidden_size,
        )
        self.logit_hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.logit_output_projection = nn.Linear(
            hidden_size,
            num_attention_heads,
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_gate = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        head_dim = self.hidden_size // self.num_attention_heads
        logit_hidden = F.relu(self.logit_input_projection(torch.cat((node[source], edge, node[destination]), dim=-1)))
        logit_hidden = F.relu(self.logit_hidden_projection(logit_hidden))
        attention_logits = self.logit_output_projection(logit_hidden).view(
            edge.shape[0],
            self.num_attention_heads,
            1,
        )
        attention = _scatter_softmax(
            attention_logits / math.sqrt(head_dim),
            source,
            node.shape[0],
        )
        values = F.gelu(self.value_projection(edge)).view(
            edge.shape[0],
            self.num_attention_heads,
            head_dim,
        )
        aggregate = _scatter_sum(attention * values, source, node.shape[0]).reshape(
            node.shape[0],
            self.hidden_size,
        )
        update = self.output_projection(aggregate) * torch.sigmoid(self.output_gate(aggregate))
        return node + update


class OInvFoldNodeUpdate(nn.Module):
    r"""
    Residual node transition followed by graph-level gating.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.transition_input_norm = nn.BatchNorm1d(hidden_size)
        self.transition_hidden_projection = nn.Linear(
            hidden_size,
            4 * hidden_size,
        )
        self.transition_output_projection = nn.Linear(
            4 * hidden_size,
            hidden_size,
        )
        self.transition_output_norm = nn.BatchNorm1d(hidden_size)
        self.graph_gate_input_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_gate_hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_gate_output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, node: Tensor, batch_index: Tensor) -> Tensor:
        update = self.transition_input_norm(node)
        update = F.relu(self.transition_hidden_projection(update))
        update = self.transition_output_projection(update)
        node = node + self.transition_output_norm(update)
        num_graphs = int(batch_index.max()) + 1
        graph_sum = _scatter_sum(node, batch_index, num_graphs)
        graph_count = _scatter_sum(
            node.new_ones((node.shape[0], 1)),
            batch_index,
            num_graphs,
        )
        graph_mean = graph_sum / graph_count
        graph_gate = F.relu(self.graph_gate_input_projection(graph_mean))
        graph_gate = F.relu(self.graph_gate_hidden_projection(graph_gate))
        graph_gate = self.graph_gate_output_projection(graph_gate)
        return node * torch.sigmoid(graph_gate)[batch_index]


class OInvFoldEdgeUpdate(nn.Module):
    r"""
    Residual edge transition conditioned on both endpoint nodes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.transition = OInvFoldGeluTransition(
            num_layers,
            3 * hidden_size,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )
        self.output_norm = nn.BatchNorm1d(hidden_size)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        update = self.transition(torch.cat((node[source], edge, node[destination]), dim=-1))
        return self.output_norm(edge + update)


class OInvFoldNormalizedReluFeedForward(nn.Module):
    r"""
    Batch-normalized ReLU feed-forward computation used by OInvFold embeddings
    and geometric edge features.
    """

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.intermediate_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)
        )
        self.intermediate_norms = nn.ModuleList(nn.BatchNorm1d(hidden_size) for _ in range(num_layers - 2))
        self.intermediate_dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 2))
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, features: Tensor) -> Tensor:
        hidden = F.relu(self.input_dropout(self.input_norm(self.input_projection(features))))
        for projection, norm, dropout in zip(
            self.intermediate_projections,
            self.intermediate_norms,
            self.intermediate_dropouts,
            strict=True,
        ):
            hidden = F.relu(dropout(norm(projection(hidden))))
        return self.output_projection(hidden)


class OInvFoldGeluTransition(nn.Module):
    r"""
    Unnormalized GELU transition used by OInvFold edge updates.
    """

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.intermediate_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)
        )
        self.intermediate_dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 2))
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, features: Tensor) -> Tensor:
        hidden = F.gelu(self.input_dropout(self.input_projection(features)))
        for projection, dropout in zip(
            self.intermediate_projections,
            self.intermediate_dropouts,
            strict=True,
        ):
            hidden = F.gelu(dropout(projection(hidden)))
        return self.output_projection(hidden)


class OInvFoldPolymerDecoder(nn.Module):
    r"""
    Released residual transition decoder for the unified polymer vocabulary.
    """

    def __init__(
        self,
        hidden_size: int,
        decoder_hidden_size: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.transition_norm = LayerNorm(hidden_size)
        self.transition_input_projection = nn.Linear(
            hidden_size,
            decoder_hidden_size,
        )
        self.transition_hidden_dropout = nn.Dropout(0.1)
        self.transition_output_projection = nn.Linear(
            decoder_hidden_size,
            hidden_size,
        )
        self.transition_output_dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, node: Tensor) -> OInvFoldOutput:
        update = self.transition_norm(node)
        update = F.gelu(self.transition_input_projection(update))
        update = self.transition_hidden_dropout(update)
        update = self.transition_output_projection(update)
        update = self.transition_output_dropout(update)
        logits = self.output_projection(node + update)
        return OInvFoldOutput(
            logits=logits,
            log_probabilities=F.log_softmax(logits, dim=-1),
        )


class OInvFoldAutoregressiveDecoder(nn.Module):
    r"""
    Teacher-forced causal decoder conditioned on OInvFold polymer states.
    """

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        *,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_positions: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.bos_embedding = nn.Parameter(torch.randn(hidden_size))
        self.structure_projection = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList(
            OInvFoldAutoregressiveDecoderBlock(
                hidden_size,
                num_heads,
                dropout,
            )
            for _ in range(num_blocks)
        )
        self.output_bias = nn.Parameter(torch.empty(vocabulary_size))
        nn.init.uniform_(
            self.output_bias,
            -1 / math.sqrt(hidden_size),
            1 / math.sqrt(hidden_size),
        )

    def forward(
        self,
        structure: Tensor,
        target_tokens: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch_size, length, _ = structure.shape
        previous_tokens = torch.roll(target_tokens, shifts=1, dims=1)
        token = self.token_embedding(previous_tokens)
        token = torch.cat(
            (
                self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1),
                token[:, 1:],
            ),
            dim=1,
        )
        positions = torch.arange(length, device=structure.device)
        states = token + self.position_embedding(positions) + self.structure_projection(structure)
        attention_mask = torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=structure.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            states = block(states, attention_mask, ~valid_mask)
        return F.linear(
            states,
            self.token_embedding.weight,
            self.output_bias,
        )


class OInvFoldAutoregressiveDecoderBlock(nn.Module):
    r"""
    One pre-normalized causal self-attention and feed-forward decoder block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transition_norm = LayerNorm(hidden_size)
        self.transition_input_projection = nn.Linear(
            hidden_size,
            4 * hidden_size,
        )
        self.transition_hidden_dropout = nn.Dropout(dropout)
        self.transition_output_projection = nn.Linear(
            4 * hidden_size,
            hidden_size,
        )
        self.transition_output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: Tensor,
        attention_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.attention_norm(states)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        states = states + update
        update = self.transition_norm(states)
        update = F.gelu(self.transition_input_projection(update))
        update = self.transition_hidden_dropout(update)
        update = self.transition_output_projection(update)
        update = self.transition_output_dropout(update)
        return states + update


class OInvFoldLigandDecoder(nn.Module):
    r"""
    Released linear element-category readout.
    """

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, node: Tensor) -> OInvFoldOutput:
        logits = self.output_projection(node)
        return OInvFoldOutput(
            logits=logits,
            log_probabilities=F.log_softmax(logits, dim=-1),
        )


def _encode_graph(
    graph: OInvFoldGraph,
    node_embedding: nn.Module,
    edge_embedding: nn.Module,
    virtual_embedding: nn.Module,
    encoder: OInvFoldEncoder,
    mask_probability: float,
) -> Tensor:
    num_real_nodes = graph.node_features.shape[0]
    node = node_embedding(graph.node_features)
    edge = edge_embedding(graph.edge_features)
    global_node = virtual_embedding(graph.virtual_indices)
    global_edge = edge.new_zeros((graph.global_edge_index.shape[1], edge.shape[-1]))
    return encoder(
        torch.cat((node, global_node), dim=0),
        torch.cat((edge, global_edge), dim=0),
        torch.cat((graph.edge_rotations, graph.global_edge_rotations), dim=0),
        torch.cat((graph.edge_translations, graph.global_edge_translations), dim=0),
        torch.cat((graph.edge_rbf, graph.global_edge_rbf), dim=0),
        torch.cat((graph.edge_index, graph.global_edge_index), dim=1),
        torch.cat((graph.batch_index, graph.global_batch_index), dim=0),
        num_real_nodes,
        mask_probability,
    )


def _scatter_sum(values: Tensor, index: Tensor, num_groups: int) -> Tensor:
    output = values.new_zeros((num_groups, *values.shape[1:]))
    expanded = index.view(index.shape[0], *((1,) * (values.ndim - 1))).expand_as(values)
    output.scatter_add_(0, expanded, values)
    return output


def _scatter_softmax(values: Tensor, index: Tensor, num_groups: int) -> Tensor:
    expanded = index.view(index.shape[0], *((1,) * (values.ndim - 1))).expand_as(values)
    maxima = values.new_full((num_groups, *values.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    exponentials = (values - maxima.gather(0, expanded)).exp()
    denominators = _scatter_sum(exponentials, index, num_groups).gather(0, expanded)
    return exponentials / denominators


def _apply_transform(rotations: Tensor, translations: Tensor, points: Tensor) -> Tensor:
    return torch.einsum("...ij,...nj->...ni", rotations, points) + translations.unsqueeze(-2)


@dataclass(eq=False, repr=False)
class OInvFoldGraph:
    r"""
    One lowered OInvFold graph.

    Tensor rows follow the exact real-node, local-edge, global-node, and
    global-edge orders supplied by the Processor.
    """

    node_features: Tensor
    edge_features: Tensor
    edge_rotations: Tensor
    edge_translations: Tensor
    edge_rbf: Tensor
    global_edge_rotations: Tensor
    global_edge_translations: Tensor
    global_edge_rbf: Tensor
    edge_index: Tensor
    global_edge_index: Tensor
    batch_index: Tensor
    global_batch_index: Tensor
    virtual_indices: Tensor


@dataclass(eq=False, repr=False)
class OInvFoldPolymerInput:
    r"""
    Lowered polymer graph and canonical decode alignment.

    ``chain_index`` and ``sequence_tokens`` align to real graph nodes.
    ``sequence_tokens`` is absent when a component has no released vocabulary
    identity and is required only by the teacher-forced autoregressive operation.
    """

    graph: OInvFoldGraph
    structure: Structure
    component_references: tuple[ComponentReference, ...]
    coordinate_slots: tuple[tuple[AtomSite, ...], ...]
    design_mask: Tensor
    chain_index: Tensor
    sequence_tokens: Tensor | None = None


@dataclass(eq=False, repr=False)
class OInvFoldLigandInput:
    r"""
    Lowered atom graph and ordered atom-slot decode alignment.
    """

    graph: OInvFoldGraph
    context: Structure | None
    slot_ids: tuple[str, ...]
    coordinates: Tensor
    vocabulary_logit_mask: Tensor | None = None


@dataclass(eq=False, repr=False)
class OInvFoldOutput:
    r"""
    Raw OInvFold categorical predictions.

    Args:
        logits: Unnormalized values with shape ``(num_nodes, vocabulary_size)``.
        log_probabilities: Log-softmax values aligned with ``logits``.
    """

    logits: Tensor
    log_probabilities: Tensor
