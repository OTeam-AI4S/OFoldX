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
Released OInvFold polymer and ligand graph models.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.backend import LayerNorm
from ...modules.feed_forward import FeedForward
from ...modules.functional import grouped_softmax, indexed_sum
from ...modules.geometry import apply_rigid_transform, rotate_vectors
from ..processing import project_model_features
from .config import OInvFoldLigandConfig, OInvFoldPolymerConfig

#: Fixed number of learned global nodes appended to each OInvFold graph.
NUM_OINVFOLD_GLOBAL_NODES = 3


def _output_classes(modality: str) -> tuple[int, ...]:
    r"""One class per released output column: the 20 amino acids, the modality's A, C, G (both nucleic types share
    those columns; protein reads them as RNA), ``DNA_T`` and ``RNA_U``."""
    shared = "DNA" if modality == "dna" else "RNA"
    return (
        tuple(TOKEN_INDEX[letter] for letter in "ACDEFGHIKLMNPQRSTVWY")
        + tuple(TOKEN_INDEX[f"{shared}_{letter}"] for letter in "ACG")
        + (TOKEN_INDEX["DNA_T"], TOKEN_INDEX["RNA_U"])
    )


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


def normalize_oinvfold_graph(graph: OInvFoldGraph, *, dtype: torch.dtype | None = None) -> OInvFoldGraph:
    """Validate the finite features of an explicitly fully observed graph."""
    result = cast(OInvFoldGraph, cast(TensorClass, graph).clone(recurse=False))
    for name in (
        "node_features",
        "edge_features",
        "edge_rotations",
        "edge_translations",
        "edge_rbf",
        "global_edge_rotations",
        "global_edge_translations",
        "global_edge_rbf",
    ):
        values = getattr(graph, name)
        setattr(
            result,
            name,
            project_model_features(
                values, torch.ones_like(values, dtype=torch.bool), dtype=dtype, feature=f"OInvFold {name}"
            ),
        )
    return result


def encode_oinvfold_graph(
    graph: OInvFoldGraph,
    node_embedding: nn.Module,
    edge_embedding: nn.Module,
    virtual_embedding: nn.Embedding,
    encoder: OInvFoldEncoder,
    *,
    mask_probability: float,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Project and encode the ordered real/global axes of one OInvFold graph."""

    graph = normalize_oinvfold_graph(graph, dtype=next(node_embedding.parameters()).dtype)
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
        graph.virtual_indices.numel() // NUM_OINVFOLD_GLOBAL_NODES,
        mask_probability,
        generator,
    )


class OInvFoldPolymerModel(Model):
    r"""
    OInvFold inverse folding for protein, DNA, or RNA backbones.

    Logits are over the 72 canonical classes; every class outside ``trained_classes`` (one class per released output
    column, by the configured modality) is ``-inf``, so log probabilities equal the released ones.
    """

    config_class = OInvFoldPolymerConfig
    config: OInvFoldPolymerConfig

    def __init__(self, config: OInvFoldPolymerConfig | None = None) -> None:
        config = OInvFoldPolymerConfig() if config is None else config
        super().__init__(config)
        self.node_embedding = FeedForward(
            133, config.hidden_size, config.hidden_size, num_hidden_layers=1, norm=nn.BatchNorm1d, activation="relu"
        )
        self.edge_embedding = FeedForward(
            310, config.hidden_size, config.hidden_size, num_hidden_layers=1, norm=nn.BatchNorm1d, activation="relu"
        )
        self.virtual_embedding = nn.Embedding(
            30,
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
            len(TOKENS),
        )
        self.autoregressive_decoder = OInvFoldAutoregressiveDecoder(
            config.hidden_size,
            len(TOKENS),
            num_blocks=4,
            num_attention_heads=8,
            dropout=config.dropout,
            max_positions=8192,
        )
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[list(_output_classes(config.modality))] = True
        self.register_buffer("trained_classes", trained, persistent=False)
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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Encode one polymer graph, applying the released corruption in training mode.

        Args:
            input: Processed polymer graph.
            mask_probability: Fraction of node and edge rows replaced in each
                encoder block while training.
            generator: Random generator used for graph corruption.
        """

        return encode_oinvfold_graph(
            input.graph,
            self.node_embedding,
            self.edge_embedding,
            self.virtual_embedding,
            self.encoder,
            mask_probability=mask_probability,
            generator=generator,
        )

    def forward(
        self,
        input: OInvFoldPolymerInput,
        *,
        mask_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> OInvFoldOutput:
        node = self.encode(
            input,
            mask_probability=mask_probability,
            generator=generator,
        )
        logits = self.decoder(node).logits
        return OInvFoldOutput(logits=logits.masked_fill(~self.trained_classes, -torch.inf), batch_size=[])

    def autoregressive_probabilities(
        self,
        input: OInvFoldPolymerInput,
        *,
        mask_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> OInvFoldOutput:
        r"""
        Predict polymer identities with the released teacher-forced causal decoder.

        Args:
            input: Processed polymer graph with canonical sequence tokens (teacher classes).
            mask_probability: Fraction of node and edge rows replaced in each
                encoder block while training.
            generator: Random generator used for graph corruption.
        """

        sequence_tokens = input.sequence_tokens
        node = self.encode(
            input,
            mask_probability=mask_probability,
            generator=generator,
        )
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
        return OInvFoldOutput(logits=logits.masked_fill(~self.trained_classes, -torch.inf), batch_size=[])


class OInvFoldLigandModel(Model):
    r"""
    OInvFold atom-element assignment for a coordinate-defined ligand graph.

    Examples:
        >>> import torch
        >>> from ofoldx.models.oinvfold import (
        ...     OInvFoldLigandConfig,
        ...     OInvFoldLigandModel,
        ...     OInvFoldLigandProcessor,
        ... )
        >>> coordinates = torch.tensor(
        ...         [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.2, 1.3, 0.0]],
        ...     dtype=torch.float32,
        ... )
        >>> input = OInvFoldLigandProcessor()(
        ...     coordinates,
        ...     slot_ids=("atom_0", "atom_1", "atom_2"),
        ... )
        >>> config = OInvFoldLigandConfig(
        ...     {"hidden_size": 8, "num_encoder_blocks": 1, "num_virtual_atoms": 2,
        ...      "num_attention_heads": 1, "num_geometric_layers": 1,
        ...      "num_edge_layers": 1}
        ... )
        >>> model = OInvFoldLigandModel(config)
        >>> with torch.inference_mode():
        ...     output = model(input, mask_probability=0.0)
        >>> output.logits.shape
        torch.Size([3, 37])
        >>> output.log_probabilities.shape == output.logits.shape
        True
    """

    config_class = OInvFoldLigandConfig
    config: OInvFoldLigandConfig

    def __init__(self, config: OInvFoldLigandConfig | None = None) -> None:
        config = OInvFoldLigandConfig() if config is None else config
        super().__init__(config)
        self.node_embedding = FeedForward(
            114, config.hidden_size, config.hidden_size, num_hidden_layers=1, norm=nn.BatchNorm1d, activation="relu"
        )
        self.edge_embedding = FeedForward(
            272, config.hidden_size, config.hidden_size, num_hidden_layers=1, norm=nn.BatchNorm1d, activation="relu"
        )
        self.virtual_embedding = nn.Embedding(
            30,
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
            len(OINVFOLD_LIGAND_ELEMENTS),
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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Encode one ligand graph, applying the released corruption in training mode.

        Args:
            input: Processed ligand graph.
            mask_probability: Fraction of node and edge rows replaced in each
                encoder block while training.
            generator: Random generator used for graph corruption.
        """

        return encode_oinvfold_graph(
            input.graph,
            self.node_embedding,
            self.edge_embedding,
            self.virtual_embedding,
            self.encoder,
            mask_probability=mask_probability,
            generator=generator,
        )

    def forward(
        self,
        input: OInvFoldLigandInput,
        *,
        mask_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> OInvFoldOutput:
        node = self.encode(
            input,
            mask_probability=mask_probability,
            generator=generator,
        )
        output = self.decoder(node)
        if input.slot_logit_bias is None:
            return output
        logits = output.logits + input.slot_logit_bias.to(output.logits.dtype)
        return OInvFoldOutput(logits=logits, batch_size=[])


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
        self.block_gate = nn.Linear(hidden_size, 1)
        self.gradient_checkpointing = False

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

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
        num_graphs: int,
        mask_probability: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not 0.0 <= mask_probability <= 1.0:
            fail("value_invalid", RequestRef("mask_probability", str(mask_probability)))
        outputs = []
        for module in self.blocks:
            block = cast(OInvFoldEncoderBlock, module)
            node, edge = block.corrupt(node, edge, mask_probability, generator)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node, edge = self._gradient_checkpointing_func(
                    block.__call__,
                    node,
                    edge,
                    edge_rotations,
                    edge_translations,
                    edge_rbf,
                    edge_index,
                    batch_index,
                    num_graphs,
                )
            else:
                node, edge = block(
                    node,
                    edge,
                    edge_rotations,
                    edge_translations,
                    edge_rbf,
                    edge_index,
                    batch_index,
                    num_graphs,
                )
            outputs.append(node[:num_real_nodes])
        stacked = torch.stack(outputs, dim=1)
        weights = torch.sigmoid(self.block_gate(stacked))
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

    def corrupt(
        self,
        node: Tensor,
        edge: Tensor,
        mask_probability: float,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Apply this block's training-time node and edge corruption."""

        if not self.training:
            return node, edge
        node = node.clone()
        edge = edge.clone()
        node_indices = torch.randperm(
            node.shape[0],
            device=node.device,
            generator=generator,
        )[: int(node.shape[0] * mask_probability)]
        edge_indices = torch.randperm(
            edge.shape[0],
            device=edge.device,
            generator=generator,
        )[: int(edge.shape[0] * mask_probability)]
        node[node_indices] = self.mask_token.weight[0]
        edge[edge_indices] = self.mask_token.weight[1]
        return node, edge

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_rotations: Tensor,
        edge_translations: Tensor,
        edge_rbf: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        edge = self.geometric_edge_update(
            node,
            edge,
            edge_rotations,
            edge_translations,
            edge_rbf,
            edge_index,
        )
        node = self.attention(node, edge, edge_index)
        node = self.node_update(node, batch_index, num_graphs)
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
        self.geometry_feed_forward = FeedForward(
            4 * num_virtual_atoms * 3 + 9 + 16 + num_virtual_atoms,
            hidden_size,
            hidden_size,
            dropout=dropout,
            num_hidden_layers=max(num_layers - 1, 1),
            norm=nn.BatchNorm1d,
            activation="relu",
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
        transformed_vectors = apply_rigid_transform(source_vectors, edge_rotations, edge_translations)
        rotated_source = rotate_vectors(virtual_node[source], edge_rotations.mT)
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
        update = self.geometry_feed_forward(geometric)
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
        self.logit_feed_forward = FeedForward(
            3 * hidden_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers=2,
            activation="relu",
        )
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_gate = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        head_size = self.hidden_size // self.num_attention_heads
        attention_logits = self.logit_feed_forward(torch.cat((node[source], edge, node[destination]), dim=-1)).view(
            edge.shape[0],
            self.num_attention_heads,
            1,
        )
        attention = grouped_softmax(
            attention_logits / math.sqrt(head_size),
            source,
            node.shape[0],
        )
        values = F.gelu(self.value_projection(edge)).view(
            edge.shape[0],
            self.num_attention_heads,
            head_size,
        )
        aggregate = indexed_sum(attention * values, source, node.shape[0]).reshape(
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
        self.feed_forward = FeedForward(hidden_size, 4 * hidden_size, activation="relu")
        self.transition_output_norm = nn.BatchNorm1d(hidden_size)
        self.graph_gate_feed_forward = FeedForward(hidden_size, hidden_size, num_hidden_layers=2, activation="relu")

    def forward(self, node: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
        update = self.transition_input_norm(node)
        update = self.feed_forward(update)
        node = node + self.transition_output_norm(update)
        graph_sum = indexed_sum(node, batch_index, num_graphs)
        graph_count = indexed_sum(
            node.new_ones((node.shape[0], 1)),
            batch_index,
            num_graphs,
        )
        graph_mean = graph_sum / graph_count
        graph_gate = self.graph_gate_feed_forward(graph_mean)
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
        self.feed_forward = FeedForward(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            num_hidden_layers=max(num_layers - 1, 1),
            dropout=dropout,
            activation="gelu",
        )
        self.output_norm = nn.BatchNorm1d(hidden_size)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        update = self.feed_forward(torch.cat((node[source], edge, node[destination]), dim=-1))
        return self.output_norm(edge + update)


class OInvFoldPolymerDecoder(nn.Module):
    r"""
    Released residual transition decoder for the unified polymer vocabulary.
    """

    def __init__(
        self,
        hidden_size: int,
        decoder_hidden_size: int,
        vocab_size: int,
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
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, node: Tensor) -> OInvFoldOutput:
        update = self.transition_norm(node)
        update = F.gelu(self.transition_input_projection(update))
        update = self.transition_hidden_dropout(update)
        update = self.transition_output_projection(update)
        update = self.transition_output_dropout(update)
        logits = self.output_projection(node + update)
        return OInvFoldOutput(logits=logits, batch_size=[])


class OInvFoldAutoregressiveDecoder(nn.Module):
    r"""
    Teacher-forced causal decoder conditioned on OInvFold polymer states.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        num_blocks: int,
        num_attention_heads: int,
        dropout: float,
        max_positions: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.bos_embedding = nn.Parameter(torch.randn(hidden_size))
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.structure_projection = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList(
            OInvFoldAutoregressiveDecoderBlock(
                hidden_size,
                num_attention_heads,
                dropout,
            )
            for _ in range(num_blocks)
        )
        self.output_bias = nn.Parameter(torch.empty(vocab_size))
        self.gradient_checkpointing = False
        nn.init.uniform_(
            self.output_bias,
            -1 / math.sqrt(hidden_size),
            1 / math.sqrt(hidden_size),
        )

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(
        self,
        structure: Tensor,
        target_tokens: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch_size, num_residues, _ = structure.shape
        previous_tokens = torch.roll(target_tokens, shifts=1, dims=1)
        token = self.token_embedding(previous_tokens)
        token = torch.cat(
            (
                self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1),
                token[:, 1:],
            ),
            dim=1,
        )
        positions = torch.arange(num_residues, device=structure.device)
        states = token + self.position_embedding(positions) + self.structure_projection(structure)
        attention_mask = torch.triu(
            torch.ones(
                (num_residues, num_residues),
                dtype=torch.bool,
                device=structure.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                states = self._gradient_checkpointing_func(
                    block.__call__,
                    states,
                    attention_mask,
                    ~valid_mask,
                )
            else:
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
        num_attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
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
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, node: Tensor) -> OInvFoldOutput:
        logits = self.output_projection(node)
        return OInvFoldOutput(logits=logits, batch_size=[])


# ===========================================================================
# OInvFold typed graph, inputs, and outputs.
# ===========================================================================


@tensorclass
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


@tensorclass
class OInvFoldPolymerInput:
    r"""
    Lowered polymer graph aligned to the prepared rows.

    ``component_index``, ``design_mask``, ``chain_index`` and ``sequence_tokens`` (canonical classes) align to the real
    graph nodes.
    """

    graph: OInvFoldGraph
    component_index: Tensor
    design_mask: Tensor
    chain_index: Tensor
    sequence_tokens: Tensor


@tensorclass
class OInvFoldLigandInput:
    r"""
    Lowered atom graph and ordered atom-slot decode alignment.
    """

    graph: OInvFoldGraph
    context: Structure | None
    slot_ids: tuple[str, ...]
    coordinates: Tensor
    slot_logit_bias: Tensor | None = None


class OInvFoldOutput(TensorCollection):
    r"""
    Raw OInvFold categorical predictions.

    Attributes:
        logits: Unnormalized values with shape ``(num_nodes, vocab_size)``.
    """

    logits: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Log-softmax values aligned with ``logits``.
        """

        return F.log_softmax(self.logits, dim=-1)
