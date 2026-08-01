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
gRNAde geometric-vector encoder and autoregressive RNA decoder.

The equations reproduce geometric-rna-design v1.0.0 revision
``2453b18778a1981ac25f314790e5395f1e9ab218`` without its PyG carrier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.functional import indexed_sum
from ...modules.gvp import Gvp, GvpDropout, GvpLayerNorm, GvpValue
from ..processing import project_model_features
from .config import GrnadeConfig

#: The classes the released checkpoints predict, in their output order (canonical spelling); every other canonical
#: class has logit ``-inf``, so identical inputs give the upstream distribution.
GRNADE_VOCABULARY: tuple[str, ...] = ("RNA_A", "RNA_G", "RNA_C", "RNA_U")
#: The released decoder input rows: the predicted classes and the unknown residue (``_``, here ``RNA_X``).
GRNADE_INPUT_VOCABULARY: tuple[str, ...] = (*GRNADE_VOCABULARY, "RNA_X")
#: The predicted classes as indices of the canonical axis.
GRNADE_CLASSES: tuple[int, ...] = tuple(TOKEN_INDEX[token] for token in GRNADE_VOCABULARY)
_NODE_INPUT_SIZE = (15, 4)
_EDGE_VECTOR_SIZE = 3
#: Width of the decoder's token feature on each edge (upstream sets it to its four-class output size).
_TOKEN_FEATURE_SIZE = len(GRNADE_VOCABULARY)


def _invalid(argument: str, value: object) -> None:
    fail("value_invalid", RequestRef(argument, str(value)))


def normalize_grnade_input(input: GrnadeInput, *, dtype: torch.dtype | None = None) -> GrnadeInput:
    """Select available conformations before graph embedding and dtype lowering."""
    result = cast(GrnadeInput, cast(TensorClass, input).clone(recurse=False))
    observed = input.conformation_mask
    edge_observed = observed[input.edge_index[0]] & observed[input.edge_index[1]]
    result.node_scalar_features = project_model_features(
        input.node_scalar_features, observed[..., None], dtype=dtype, feature="RNA node scalar features"
    )
    result.node_vector_features = project_model_features(
        input.node_vector_features, observed[..., None, None], dtype=dtype, feature="RNA node vector features"
    )
    result.edge_scalar_features = project_model_features(
        input.edge_scalar_features, edge_observed[..., None], dtype=dtype, feature="RNA edge scalar features"
    )
    result.edge_vector_features = project_model_features(
        input.edge_vector_features, edge_observed[..., None, None], dtype=dtype, feature="RNA edge vector features"
    )
    return result


class GrnadeModel(Model):
    r"""
    Design RNA identities with the released gRNAde GVP graph network.

    Args:
        config: Model architecture configuration.

    Notes:
        Outputs use the 72 canonical classes. Logits for classes outside the checkpoint's predicted RNA
        vocabulary are ``-inf``. The paired Pipeline reconstructs molecular targets and binds total/mean log
        probability, perplexity, native recovery, and each candidate's design record.
    """

    config_class = GrnadeConfig
    config: GrnadeConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: GrnadeConfig | None = None) -> None:
        config = GrnadeConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        node_input_size = _NODE_INPUT_SIZE
        node_hidden_size = (config.node_hidden_scalar_size, config.node_hidden_vector_size)
        edge_input_size = (config.edge_scalar_size, _EDGE_VECTOR_SIZE)
        edge_hidden_size = (config.edge_hidden_scalar_size, config.edge_hidden_vector_size)

        self.node_embedding = nn.Sequential(
            GvpLayerNorm(node_input_size),
            Gvp(
                node_input_size,
                node_hidden_size,
                scalar_activation=None,
                vector_activation=None,
            ),
        )
        self.edge_embedding = nn.Sequential(
            GvpLayerNorm(edge_input_size),
            Gvp(
                edge_input_size,
                edge_hidden_size,
                scalar_activation=None,
                vector_activation=None,
            ),
        )
        self.encoder_blocks = nn.ModuleList(
            GrnadeEncoderBlock(
                node_hidden_size,
                edge_hidden_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        )
        self.token_embedding = nn.Embedding(len(TOKENS), _TOKEN_FEATURE_SIZE)
        decoder_edge_hidden_size = (edge_hidden_size[0] + _TOKEN_FEATURE_SIZE, edge_hidden_size[1])
        self.decoder_blocks = nn.ModuleList(
            GrnadeDecoderBlock(
                node_hidden_size,
                decoder_edge_hidden_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        )
        self.output_projection = Gvp(
            node_hidden_size,
            (len(TOKENS), 0),
            scalar_activation=None,
            vector_activation=None,
        )
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[list(GRNADE_CLASSES)] = True
        self.register_buffer("trained_classes", trained, persistent=False)
        self.post_init()

    def _logits(self, node: GvpValue) -> Tensor:
        r"""Logits over the canonical classes; classes the checkpoint does not predict are ``-inf``."""
        return cast(Tensor, self.output_projection(node)).masked_fill(~self.trained_classes, -torch.inf)

    def forward(self, input: GrnadeInput) -> GrnadeOutput:
        r"""
        Predict each native identity under released teacher forcing.
        """

        node, edge = self._encode(input)
        encoder_node = node
        source, destination = input.edge_index
        source_token = self.token_embedding(input.sequence_tokens)[source]
        source_token = source_token * (source < destination).unsqueeze(-1)
        edge = (torch.cat((edge[0], source_token), dim=-1), edge[1])
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            if checkpointing:
                node = cast(
                    GvpValue,
                    self._gradient_checkpointing_func(
                        block.__call__,
                        node,
                        input.edge_index,
                        edge,
                        encoder_node,
                    ),
                )
            else:
                node = block(
                    node,
                    input.edge_index,
                    edge,
                    encoder_node=encoder_node,
                )
        return GrnadeOutput(logits=self._logits(node), batch_size=[])

    @torch.inference_mode()
    def sample(
        self,
        input: GrnadeInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        residue_logit_bias: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeSamplingOutput:
        r"""
        Sample complete RNA sequences from 5' to 3'.

        Fixed residues of a predicted class retain their native identity. The emitted preference
        logits keep the released ``+100`` native-identity logit bias;
        ``residue_logit_bias`` adds caller-owned per-position preferences over the 72
        canonical classes with shape ``(num_residues, 72)``.
        Only ``design_mask`` rows are realized by the decoding Pipeline.
        """

        if num_samples <= 0:
            _invalid("num_samples", num_samples)
        if temperature <= 0:
            _invalid("temperature", temperature)

        node, edge = self._encode(input)
        num_nodes = node[0].shape[0]
        device = node[0].device
        node = (
            node[0].repeat(num_samples, 1),
            node[1].repeat(num_samples, 1, 1),
        )
        edge = (
            edge[0].repeat(num_samples, 1),
            edge[1].repeat(num_samples, 1, 1),
        )

        edge_index = input.edge_index.unsqueeze(0).expand(num_samples, -1, -1)
        offsets = num_nodes * torch.arange(num_samples, device=device).view(-1, 1, 1)
        edge_index = (edge_index + offsets).permute(1, 0, 2).reshape(2, -1)

        sequence_tokens = torch.zeros(num_samples * num_nodes, device=device, dtype=torch.long)
        token_features = torch.zeros(
            num_samples * num_nodes,
            _TOKEN_FEATURE_SIZE,
            device=device,
            dtype=node[0].dtype,
        )
        logits = torch.zeros(num_samples * num_nodes, len(TOKENS), device=device, dtype=node[0].dtype)
        node_caches = [(node[0].clone(), node[1].clone()) for _ in self.decoder_blocks]

        fixed_known = (~input.design_mask) & self.trained_classes[input.sequence_tokens]
        fixed_bias = F.one_hot(input.sequence_tokens, num_classes=len(TOKENS)).to(dtype=node[0].dtype)
        fixed_bias = fixed_bias * fixed_known.unsqueeze(-1) * 100.0

        for residue_index in range(num_nodes):
            source, destination = edge_index
            source_token = token_features[source]
            source_token = source_token * (source < destination).unsqueeze(-1)
            decoder_edge = (torch.cat((edge[0], source_token), dim=-1), edge[1])
            edge_mask = destination.remainder(num_nodes) == residue_index
            incoming_edge_index = edge_index[:, edge_mask]
            incoming_edge = _tuple_index(decoder_edge, edge_mask)
            node_mask = torch.zeros(num_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[residue_index::num_nodes] = True

            output = _tuple_index(node, node_mask)
            for block_index, block in enumerate(self.decoder_blocks):
                input_at_position = _tuple_index(node_caches[block_index], node_mask)
                output = block(
                    node_caches[block_index],
                    incoming_edge_index,
                    incoming_edge,
                    encoder_node=node,
                    node_mask=node_mask,
                )
                output = _tuple_index(output, node_mask)
                node_caches[block_index][0][node_mask] = input_at_position[0]
                node_caches[block_index][1][node_mask] = input_at_position[1]
                if block_index + 1 < len(self.decoder_blocks):
                    node_caches[block_index + 1][0][residue_index::num_nodes] = output[0]
                    node_caches[block_index + 1][1][residue_index::num_nodes] = output[1]

            residue_logits = self._logits(output) + fixed_bias[residue_index]
            if residue_logit_bias is not None:
                residue_logits = residue_logits + residue_logit_bias[residue_index]
            probabilities = torch.softmax(residue_logits / temperature, dim=-1)
            draw_device = device if generator is None else generator.device
            sampled = (
                torch.multinomial(
                    probabilities.to(device=draw_device),
                    1,
                    generator=generator,
                )
                .squeeze(-1)
                .to(device=device)
            )
            sampled = torch.where(fixed_known[residue_index], input.sequence_tokens[residue_index], sampled)
            sequence_tokens[residue_index::num_nodes] = sampled
            token_features[residue_index::num_nodes] = self.token_embedding(sampled)
            logits[residue_index::num_nodes] = residue_logits

        sequence_tokens = sequence_tokens.view(num_samples, num_nodes)
        logits = logits.view(num_samples, num_nodes, len(TOKENS))
        return GrnadeSamplingOutput(
            sequence_tokens=sequence_tokens,
            logits=logits,
            batch_size=[num_samples],
        )

    def _encode(self, input: GrnadeInput) -> tuple[GvpValue, GvpValue]:
        input = normalize_grnade_input(input, dtype=next(self.parameters()).dtype)
        node = cast(
            GvpValue,
            self.node_embedding(
                (
                    input.node_scalar_features,
                    input.node_vector_features,
                )
            ),
        )
        edge = cast(
            GvpValue,
            self.edge_embedding(
                (
                    input.edge_scalar_features,
                    input.edge_vector_features,
                )
            ),
        )
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.encoder_blocks:
            if checkpointing:
                node = cast(
                    GvpValue,
                    self._gradient_checkpointing_func(
                        block.__call__,
                        node,
                        input.edge_index,
                        edge,
                    ),
                )
            else:
                node = block(node, input.edge_index, edge)
        return _pool_conformations(
            node,
            edge,
            input.conformation_mask,
            input.edge_index,
        )


class GrnadeEncoderBlock(nn.Module):
    r"""
    One multi-conformation gRNAde encoder block.
    """

    def __init__(
        self,
        node_size: tuple[int, int],
        edge_size: tuple[int, int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.message_norm = GvpLayerNorm(node_size)
        self.message_passing = GvpMessagePassing(
            node_size,
            edge_size,
            aggregation="mean",
        )
        self.message_dropout = GvpDropout(dropout)
        self.transition = GrnadeTransition(node_size)
        self.transition_dropout = GvpDropout(dropout)

    def forward(
        self,
        node: GvpValue,
        edge_index: Tensor,
        edge: GvpValue,
    ) -> GvpValue:
        normalized = cast(GvpValue, self.message_norm(node))
        message = self.message_passing(normalized, edge_index, edge)
        node = _tuple_sum(node, cast(GvpValue, self.message_dropout(message)))
        transition = self.transition(node)
        return _tuple_sum(node, cast(GvpValue, self.transition_dropout(transition)))


class GrnadeDecoderBlock(nn.Module):
    r"""
    One masked-autoregressive gRNAde decoder block.
    """

    def __init__(
        self,
        node_size: tuple[int, int],
        edge_size: tuple[int, int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.message_passing = GvpMessagePassing(
            node_size,
            edge_size,
            aggregation="add",
        )
        self.message_dropout = GvpDropout(dropout)
        self.transition = GrnadeTransition(node_size)
        self.transition_dropout = GvpDropout(dropout)

    def forward(
        self,
        node: GvpValue,
        edge_index: Tensor,
        edge: GvpValue,
        encoder_node: GvpValue,
        node_mask: Tensor | None = None,
    ) -> GvpValue:
        source, destination = edge_index
        forward = source < destination
        message = _tuple_sum(
            self.message_passing(
                node,
                edge_index[:, forward],
                _tuple_index(edge, forward),
            ),
            self.message_passing(
                encoder_node,
                edge_index[:, ~forward],
                _tuple_index(edge, ~forward),
            ),
        )
        counts = indexed_sum(
            torch.ones_like(destination, dtype=node[0].dtype),
            destination,
            node[0].shape[0],
        )
        counts = counts.clamp_min(1).unsqueeze(-1)
        message = (message[0] / counts, message[1] / counts.unsqueeze(-1))

        if node_mask is None:
            selected_node = node
            selected_message = message
        else:
            selected_node = _tuple_index(node, node_mask)
            selected_message = _tuple_index(message, node_mask)

        selected_node = _tuple_sum(
            selected_node,
            cast(GvpValue, self.message_dropout(selected_message)),
        )
        transition = self.transition(selected_node)
        selected_node = _tuple_sum(
            selected_node,
            cast(GvpValue, self.transition_dropout(transition)),
        )
        if node_mask is None:
            return selected_node
        node[0][node_mask] = selected_node[0]
        node[1][node_mask] = selected_node[1]
        return node


class GrnadeTransition(nn.Module):
    r"""Normalize scalar/vector node features, then compute the two-GVP feed-forward update."""

    def __init__(self, node_size: tuple[int, int]) -> None:
        super().__init__()
        self.input_norm = GvpLayerNorm(node_size)
        hidden_size = (4 * node_size[0], 2 * node_size[1])
        self.feed_forward = nn.Sequential(
            Gvp(node_size, hidden_size, vector_activation=None),
            Gvp(hidden_size, node_size, scalar_activation=None, vector_activation=None),
        )

    def forward(self, node: GvpValue) -> GvpValue:
        return cast(GvpValue, self.feed_forward(self.input_norm(node)))


class GvpMessagePassing(nn.Module):
    r"""
    GVP message function followed by destination aggregation.
    """

    def __init__(
        self,
        node_size: tuple[int, int],
        edge_size: tuple[int, int],
        *,
        aggregation: str,
    ) -> None:
        super().__init__()
        scalar_size, vector_size = node_size
        input_size = (
            2 * scalar_size + edge_size[0],
            2 * vector_size + edge_size[1],
        )
        self.message = nn.Sequential(
            Gvp(input_size, node_size, vector_activation=None),
            Gvp(node_size, node_size, vector_activation=None),
            Gvp(
                node_size,
                node_size,
                scalar_activation=None,
                vector_activation=None,
            ),
        )
        self.aggregation = aggregation

    def forward(
        self,
        node: GvpValue,
        edge_index: Tensor,
        edge: GvpValue,
    ) -> GvpValue:
        source, destination = edge_index
        message = cast(
            GvpValue,
            self.message(
                _tuple_cat(
                    _tuple_index(node, source),
                    edge,
                    _tuple_index(node, destination),
                )
            ),
        )
        scalar = _aggregate(message[0], destination, node[0].shape[0], self.aggregation)
        vector = _aggregate(message[1], destination, node[0].shape[0], self.aggregation)
        return scalar, vector


def _pool_conformations(
    node: GvpValue,
    edge: GvpValue,
    conformation_mask: Tensor,
    edge_index: Tensor,
) -> tuple[GvpValue, GvpValue]:
    node_mask = conformation_mask[..., None]
    edge_mask = (conformation_mask[edge_index[0]] & conformation_mask[edge_index[1]])[..., None]
    node_count = node_mask.sum(dim=1).clamp_min(1)
    edge_count = edge_mask.sum(dim=1).clamp_min(1)
    return (
        (
            torch.where(node_mask, node[0], 0).sum(dim=1) / node_count,
            torch.where(node_mask[..., None], node[1], 0).sum(dim=1) / node_count[..., None],
        ),
        (
            torch.where(edge_mask, edge[0], 0).sum(dim=1) / edge_count,
            torch.where(edge_mask[..., None], edge[1], 0).sum(dim=1) / edge_count[..., None],
        ),
    )


def _aggregate(
    values: Tensor,
    destination: Tensor,
    num_nodes: int,
    aggregation: str,
) -> Tensor:
    output = indexed_sum(values, destination, num_nodes)
    if aggregation == "add":
        return output
    counts = indexed_sum(torch.ones_like(destination, dtype=values.dtype), destination, num_nodes)
    return output / counts.clamp_min(1).view(num_nodes, *((1,) * (values.ndim - 1)))


def _tuple_sum(first: GvpValue, second: GvpValue) -> GvpValue:
    return first[0] + second[0], first[1] + second[1]


def _tuple_cat(*values: GvpValue) -> GvpValue:
    return (
        torch.cat(tuple(value[0] for value in values), dim=-1),
        torch.cat(tuple(value[1] for value in values), dim=-2),
    )


def _tuple_index(value: GvpValue, index: Tensor) -> GvpValue:
    return value[0][index], value[1][index]


# ===========================================================================
# gRNAde typed inputs and outputs.
# ===========================================================================


@tensorclass
class GrnadeInput:
    r"""
    Typed tensor graph consumed by gRNAde.

    Attributes:
        node_scalar_features: Scalar node features with shape
            ``(num_residues, num_conformations, 15)``.
        node_vector_features: Vector node features with shape
            ``(num_residues, num_conformations, 4, 3)``.
        edge_scalar_features: Scalar edge features with shape
            ``(num_edges, num_conformations, edge_scalar_size)``. The configured
            edge feature size is 132 for v1 and 131 for v0.3.2.
        edge_vector_features: Vector edge features with shape
            ``(num_edges, num_conformations, 3, 3)``.
        edge_index: Directed source and destination indices with shape ``(2, num_edges)``.
        sequence_tokens: Native canonical classes with shape ``(num_residues,)``.
        conformation_mask: Valid-conformation mask with shape
            ``(num_residues, num_conformations)``.
        design_mask: Residues whose identities may be replaced.
        component_index: Canonical component indices aligned with all node tensors.
    """

    node_scalar_features: Tensor
    node_vector_features: Tensor
    edge_scalar_features: Tensor
    edge_vector_features: Tensor
    edge_index: Tensor
    sequence_tokens: Tensor
    conformation_mask: Tensor
    design_mask: Tensor
    component_index: Tensor


class GrnadeOutput(TensorCollection):
    r"""
    Teacher-forced RNA identity predictions.

    Attributes:
        logits: Unnormalized logits over the canonical classes with shape
            ``(num_residues, 72)``.
    """

    logits: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Log-softmax values aligned with ``logits``.
        """

        return self.logits.log_softmax(dim=-1)


class GrnadeSamplingOutput(TensorCollection):
    r"""
    Autoregressively sampled RNA identities.

    Attributes:
        sequence_tokens: Sampled token IDs with shape
            ``(num_samples, num_residues)``.
        logits: Realized autoregressive logits with shape
            ``(num_samples, num_residues, 72)``.
    """

    sequence_tokens: Tensor
    logits: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Log-softmax values aligned with ``logits``.
        """

        return self.logits.log_softmax(dim=-1)
