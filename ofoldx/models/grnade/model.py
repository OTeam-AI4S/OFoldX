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
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.structure import Structure
from ...model import Model, _TensorCollection
from .config import GrnadeConfig

#: Canonical RNA vocabulary in the released gRNAde checkpoint order.
GRNADE_COMPONENT_IDS = ("A", "G", "C", "U")
GRNADE_TOKEN_BY_COMPONENT = {component_id: index for index, component_id in enumerate(GRNADE_COMPONENT_IDS)}
GRNADE_UNKNOWN_TOKEN = len(GRNADE_COMPONENT_IDS)
_NODE_INPUT_SIZE = (15, 4)
_EDGE_VECTOR_SIZE = 3
_VOCABULARY_SIZE = len(GRNADE_COMPONENT_IDS)

GvpValue = tuple[Tensor, Tensor]
Activation = Callable[[Tensor], Tensor] | None


class GrnadeModel(Model):
    r"""
    Design RNA identities with the released gRNAde GVP graph network.

    Args:
        config: Model architecture configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import AtomSite, MolecularSystem, PolymerSequence, Structure
        >>> from ofoldx.models.grnade import GrnadeConfig, GrnadeModel, GrnadeProcessor
        >>> sequence = PolymerSequence.from_one_letter("AG", polymer_type="rna")
        >>> system = MolecularSystem.from_entities({"rna": sequence}, {"A": "rna"})
        >>> sites = tuple(
        ...     AtomSite("A", residue, atom)
        ...     for residue in range(2)
        ...     for atom in ("P", "C4'", "N9")
        ... )
        >>> atom_index = torch.tensor([system.atom_index(site) for site in sites])
        >>> coordinates = torch.zeros(system.atom_symbol.numel(), 3)
        >>> coordinates[atom_index] = torch.tensor(
        ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0],
        ...      [2, 0.5, 0.2], [3, 0.5, 0.2], [3, 1.5, 0.2]]
        ... )
        >>> coordinate_mask = torch.zeros(system.atom_symbol.numel(), dtype=torch.bool)
        >>> coordinate_mask[atom_index] = True
        >>> structure = Structure(
        ...     system=system,
        ...     coordinates=coordinates,
        ...     coordinate_mask=coordinate_mask,
        ...     occupancy=torch.zeros(system.atom_symbol.numel()),
        ...     occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     b_factor=torch.zeros(system.atom_symbol.numel()),
        ...     b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     source_symbols=(),
        ...     residue_id=torch.full((system.component_symbol.numel(),), -1),
        ...     insertion_code=torch.full((system.component_symbol.numel(),), -1),
        ...     standard_component_id=torch.full((system.component_symbol.numel(),), -1),
        ...     batch_size=[],
        ... )
        >>> graph = GrnadeProcessor()(structure)
        >>> config = GrnadeConfig(
        ...     node_hidden_scalar_size=8, node_hidden_vector_size=2,
        ...     edge_hidden_scalar_size=8, edge_hidden_vector_size=2,
        ...     num_blocks=1, dropout=0.0,
        ... )
        >>> model = GrnadeModel(config).eval()
        >>> model(graph).logits.shape
        torch.Size([2, 4])
    """

    config_class = GrnadeConfig
    config: GrnadeConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: GrnadeConfig | None = None) -> None:
        config = GrnadeConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        node_input = _NODE_INPUT_SIZE
        node_hidden = (config.node_hidden_scalar_size, config.node_hidden_vector_size)
        edge_input = (config.edge_scalar_size, _EDGE_VECTOR_SIZE)
        edge_hidden = (config.edge_hidden_scalar_size, config.edge_hidden_vector_size)

        self.node_embedding = nn.Sequential(
            GvpLayerNorm(node_input),
            Gvp(
                node_input,
                node_hidden,
                scalar_activation=None,
                vector_activation=None,
            ),
        )
        self.edge_embedding = nn.Sequential(
            GvpLayerNorm(edge_input),
            Gvp(
                edge_input,
                edge_hidden,
                scalar_activation=None,
                vector_activation=None,
            ),
        )
        self.encoder_blocks = nn.ModuleList(
            GrnadeEncoderBlock(
                node_hidden,
                edge_hidden,
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        )
        self.token_embedding = nn.Embedding(config.num_token_embeddings, _VOCABULARY_SIZE)
        decoder_edge_hidden = (edge_hidden[0] + _VOCABULARY_SIZE, edge_hidden[1])
        self.decoder_blocks = nn.ModuleList(
            GrnadeDecoderBlock(
                node_hidden,
                decoder_edge_hidden,
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        )
        self.output_projection = Gvp(
            node_hidden,
            (_VOCABULARY_SIZE, 0),
            scalar_activation=None,
            vector_activation=None,
        )
        self.post_init()

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
        logits = cast(Tensor, self.output_projection(node))
        return GrnadeOutput(logits=logits, batch_size=[])

    @torch.inference_mode()
    def sample(
        self,
        input: GrnadeInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        logit_bias: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeSamplingOutput:
        r"""
        Sample complete RNA sequences from 5' to 3'.

        Fixed residues receive the released ``+100`` native-identity logit bias;
        ``logit_bias`` adds caller-owned per-position nucleotide preferences.
        Only ``design_mask`` rows are realized by the decoding Pipeline.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

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
            _VOCABULARY_SIZE,
            device=device,
            dtype=node[0].dtype,
        )
        logits = torch.zeros_like(token_features)
        node_caches = [(node[0].clone(), node[1].clone()) for _ in self.decoder_blocks]

        fixed_known = (~input.design_mask) & (input.sequence_tokens < _VOCABULARY_SIZE)
        fixed_bias = F.one_hot(
            input.sequence_tokens.clamp(max=_VOCABULARY_SIZE - 1),
            num_classes=_VOCABULARY_SIZE,
        ).to(dtype=node[0].dtype)
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

            output: GvpValue | None = None
            for block_index, block in enumerate(self.decoder_blocks):
                output = block(
                    node_caches[block_index],
                    incoming_edge_index,
                    incoming_edge,
                    encoder_node=node_caches[0],
                    node_mask=node_mask,
                )
                output = _tuple_index(output, node_mask)
                if block_index + 1 < len(self.decoder_blocks):
                    node_caches[block_index + 1][0][residue_index::num_nodes] = output[0]
                    node_caches[block_index + 1][1][residue_index::num_nodes] = output[1]

            assert output is not None
            residue_logits = cast(Tensor, self.output_projection(output)) + fixed_bias[residue_index]
            if logit_bias is not None:
                residue_logits = residue_logits + logit_bias[residue_index]
            probabilities = torch.softmax(residue_logits / temperature, dim=-1)
            sampled = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            ).squeeze(-1)
            sequence_tokens[residue_index::num_nodes] = sampled
            token_features[residue_index::num_nodes] = self.token_embedding(sampled)
            logits[residue_index::num_nodes] = residue_logits

        sequence_tokens = sequence_tokens.view(num_samples, num_nodes)
        logits = logits.view(num_samples, num_nodes, _VOCABULARY_SIZE)
        return GrnadeSamplingOutput(
            sequence_tokens=sequence_tokens,
            logits=logits,
            batch_size=[num_samples],
        )

    def _encode(self, input: GrnadeInput) -> tuple[GvpValue, GvpValue]:
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
        self.norms = nn.ModuleList(GvpLayerNorm(node_size) for _ in range(2))
        self.message_passing = GvpMessagePassing(
            node_size,
            edge_size,
            aggregation="mean",
        )
        self.dropouts = nn.ModuleList(GvpDropout(dropout) for _ in range(2))
        hidden_size = (4 * node_size[0], 2 * node_size[1])
        self.transition = nn.Sequential(
            Gvp(node_size, hidden_size, vector_activation=None),
            Gvp(
                hidden_size,
                node_size,
                scalar_activation=None,
                vector_activation=None,
            ),
        )

    def forward(
        self,
        node: GvpValue,
        edge_index: Tensor,
        edge: GvpValue,
    ) -> GvpValue:
        normalized = cast(GvpValue, self.norms[0](node))
        message = self.message_passing(normalized, edge_index, edge)
        node = _tuple_sum(node, cast(GvpValue, self.dropouts[0](message)))
        transition = cast(GvpValue, self.transition(self.norms[1](node)))
        return _tuple_sum(node, cast(GvpValue, self.dropouts[1](transition)))


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
        self.dropouts = nn.ModuleList(GvpDropout(dropout) for _ in range(2))
        self.norms = nn.ModuleList(GvpLayerNorm(node_size) for _ in range(2))
        hidden_size = (4 * node_size[0], 2 * node_size[1])
        self.transition = nn.Sequential(
            Gvp(node_size, hidden_size, vector_activation=None),
            Gvp(
                hidden_size,
                node_size,
                scalar_activation=None,
                vector_activation=None,
            ),
        )

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
        counts = torch.zeros(
            node[0].shape[0],
            device=destination.device,
            dtype=node[0].dtype,
        )
        counts.index_add_(0, destination, torch.ones_like(destination, dtype=node[0].dtype))
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
            cast(GvpValue, self.dropouts[0](selected_message)),
        )
        transition = cast(GvpValue, self.transition(self.norms[1](selected_node)))
        selected_node = _tuple_sum(
            selected_node,
            cast(GvpValue, self.dropouts[1](transition)),
        )
        if node_mask is None:
            return selected_node
        node[0][node_mask] = selected_node[0]
        node[1][node_mask] = selected_node[1]
        return node


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


class Gvp(nn.Module):
    r"""
    Geometric vector perceptron over scalar and vector channels.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        *,
        scalar_activation: Activation = F.silu,
        vector_activation: Activation = None,
    ) -> None:
        super().__init__()
        scalar_input, vector_input = input_size
        scalar_output, vector_output = output_size
        self.vector_input = vector_input
        self.vector_output = vector_output
        if vector_input:
            hidden_vectors = max(vector_input, vector_output)
            self.hidden_vector_projection = nn.Linear(
                vector_input,
                hidden_vectors,
                bias=False,
            )
            self.scalar_projection = nn.Linear(
                hidden_vectors + scalar_input,
                scalar_output,
            )
            if vector_output:
                self.output_vector_projection = nn.Linear(
                    hidden_vectors,
                    vector_output,
                    bias=False,
                )
                self.vector_gate_projection = nn.Linear(
                    scalar_output,
                    vector_output,
                )
        else:
            self.scalar_projection = nn.Linear(scalar_input, scalar_output)
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        vector: Tensor | None = None
        if self.vector_input:
            scalar, vector = cast(GvpValue, input)
            vector = vector.transpose(-1, -2)
            hidden_vector = self.hidden_vector_projection(vector)
            vector_norm = _norm_no_nan(hidden_vector, dim=-2)
            scalar = self.scalar_projection(torch.cat((scalar, vector_norm), dim=-1))
            if self.vector_output:
                vector = self.output_vector_projection(hidden_vector).transpose(-1, -2)
                gate_input = scalar if self.vector_activation is None else self.vector_activation(scalar)
                gate = torch.sigmoid(self.vector_gate_projection(gate_input))
                vector = vector * gate.unsqueeze(-1)
        else:
            scalar = self.scalar_projection(cast(Tensor, input))
            if self.vector_output:
                vector = torch.zeros(
                    *scalar.shape[:-1],
                    self.vector_output,
                    3,
                    device=scalar.device,
                    dtype=scalar.dtype,
                )
        if self.scalar_activation is not None:
            scalar = self.scalar_activation(scalar)
        if self.vector_output:
            return scalar, cast(Tensor, vector)
        return scalar


class GvpDropout(nn.Module):
    r"""
    Scalar dropout plus whole-vector-channel dropout.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.scalar_dropout = nn.Dropout(probability)
        self.probability = probability

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        if isinstance(input, Tensor):
            return self.scalar_dropout(input)
        scalar, vector = input
        if not self.training:
            return scalar, vector
        mask = torch.bernoulli(
            torch.full(
                vector.shape[:-1],
                1 - self.probability,
                device=vector.device,
                dtype=vector.dtype,
            )
        ).unsqueeze(-1)
        return self.scalar_dropout(scalar), mask * vector / (1 - self.probability)


class GvpLayerNorm(nn.Module):
    r"""
    Scalar layer normalization and vector-channel RMS normalization.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.vector_size = size[1]
        self.scalar_norm = nn.LayerNorm(size[0])

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        if not self.vector_size:
            return self.scalar_norm(cast(Tensor, input))
        scalar, vector = cast(GvpValue, input)
        vector_norm = _norm_no_nan(
            vector,
            dim=-1,
            keepdim=True,
            square=True,
        )
        vector_norm = torch.sqrt(vector_norm.mean(dim=-2, keepdim=True))
        return self.scalar_norm(scalar), vector / vector_norm


def _pool_conformations(
    node: GvpValue,
    edge: GvpValue,
    conformation_mask: Tensor,
    edge_index: Tensor,
) -> tuple[GvpValue, GvpValue]:
    if conformation_mask.shape[1] == 1:
        return (
            (node[0][:, 0], node[1][:, 0]),
            (edge[0][:, 0], edge[1][:, 0]),
        )

    # Clamp so a residue that is masked in every conformation pools to zero rather than dividing
    # 0/0 into NaN; the masked numerators are already zero on those rows.
    count = conformation_mask.sum(dim=1, keepdim=True).clamp_min(1)
    scalar_mask = conformation_mask.unsqueeze(-1)
    vector_mask = scalar_mask.unsqueeze(-1)
    source = edge_index[0]
    return (
        (
            (node[0] * scalar_mask).sum(dim=1) / count,
            (node[1] * vector_mask).sum(dim=1) / count.unsqueeze(-1),
        ),
        (
            (edge[0] * scalar_mask[source]).sum(dim=1) / count[source],
            (edge[1] * vector_mask[source]).sum(dim=1) / count[source].unsqueeze(-1),
        ),
    )


def _aggregate(
    values: Tensor,
    destination: Tensor,
    num_nodes: int,
    aggregation: str,
) -> Tensor:
    output = torch.zeros(
        num_nodes,
        *values.shape[1:],
        device=values.device,
        dtype=values.dtype,
    )
    output.index_add_(0, destination, values)
    if aggregation == "add":
        return output
    counts = torch.zeros(
        num_nodes,
        device=values.device,
        dtype=values.dtype,
    )
    counts.index_add_(0, destination, torch.ones_like(destination, dtype=values.dtype))
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


def _norm_no_nan(
    value: Tensor,
    *,
    dim: int,
    keepdim: bool = False,
    square: bool = False,
) -> Tensor:
    norm_square = value.square().sum(dim=dim, keepdim=keepdim).clamp_min(1e-8)
    return norm_square if square else torch.sqrt(norm_square)


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
            ``(num_edges, num_conformations, 132)``.
        edge_vector_features: Vector node features with shape
            ``(num_edges, num_conformations, 3, 3)``.
        edge_index: Directed source and destination indices with shape ``(2, num_edges)``.
        sequence_tokens: Native RNA tokens with shape ``(num_residues,)``.
        conformation_mask: Valid-conformation mask with shape
            ``(num_residues, num_conformations)``.
        design_mask: Residues whose identities may be replaced.
        structure: Canonical source structure.
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
    structure: Structure
    component_index: Tensor


class GrnadeOutput(_TensorCollection):
    r"""
    Teacher-forced RNA identity predictions.

    Attributes:
        logits: Unnormalized canonical-RNA logits with shape
            ``(num_residues, 4)``.
    """

    logits: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Log-softmax values aligned with ``logits``.
        """

        return self.logits.log_softmax(dim=-1)


class GrnadeSamplingOutput(_TensorCollection):
    r"""
    Autoregressively sampled RNA identities.

    Attributes:
        sequence_tokens: Sampled token IDs with shape
            ``(num_samples, num_residues)``.
        logits: Realized autoregressive logits with shape
            ``(num_samples, num_residues, 4)``.
    """

    sequence_tokens: Tensor
    logits: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Log-softmax values aligned with ``logits``.
        """

        return self.logits.log_softmax(dim=-1)
