r"""
NA-MPNN inverse-folding model for protein, DNA, RNA, and mixed-polymer structures.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...modules.message_passing import (
    ProteinDecoderLayer,
    ProteinEncoderLayer,
    RelativePositionEncoding,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)
from ...modules.representation import Single
from ...results_registry import register_sequence_decoder
from ..contract import BaseInput, SequenceDesignCandidate, register_pytree_dataclass
from ..mixin import ModelMixin
from ..registry import DESIGN_MODELS, MODELS
from .config import NAMPNNConfig

NAMPNN_ATOM_NAMES = (
    "N",
    "CA",
    "C",
    "O",
    "OP1",
    "OP2",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
)
NAMPNN_ATOM_TO_ID = {name: index for index, name in enumerate(NAMPNN_ATOM_NAMES)}
NAMPNN_POLYMER_TYPES = ("PP", "DNA", "RNA", "UNK", "MAS", "PAD")
NAMPNN_POLYMER_TYPE_TO_ID = {name: index for index, name in enumerate(NAMPNN_POLYMER_TYPES)}
NAMPNN_RESTYPES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "DA",
    "DC",
    "DG",
    "DT",
    "DX",
    "A",
    "C",
    "G",
    "U",
    "RX",
    "MAS",
    "PAD",
)
NAMPNN_RESTYPE_TO_ID = {name: index for index, name in enumerate(NAMPNN_RESTYPES)}
NAMPNN_TOKEN_TO_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
    "DA": "a",
    "DC": "c",
    "DG": "g",
    "DT": "t",
    "DX": "x",
    "A": "b",
    "C": "d",
    "G": "h",
    "U": "u",
    "RX": "y",
    "MAS": "-",
    "PAD": "+",
}
NAMPNN_LETTER_TO_TOKEN = {value: key for key, value in NAMPNN_TOKEN_TO_LETTER.items()}
NAMPNN_INVALID_SAMPLE_TOKEN_IDS = (
    NAMPNN_RESTYPE_TO_ID["UNK"],
    NAMPNN_RESTYPE_TO_ID["DX"],
    NAMPNN_RESTYPE_TO_ID["RX"],
    NAMPNN_RESTYPE_TO_ID["MAS"],
    NAMPNN_RESTYPE_TO_ID["PAD"],
)


@MODELS.register("nampnn")
@DESIGN_MODELS.register("nampnn")
class NAMPNNDesigner(ModelMixin):
    r"""
    Message-passing model that designs protein, DNA, and RNA sequences from fixed polymer backbones.

    Args:
        config: Architecture configuration. Defaults to
            [`NAMPNNConfig`][ofoldx.models.nampnn.config.NAMPNNConfig].

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/nampnn-design")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/nampnn-design")  # doctest: +SKIP
        >>> output = model.design(processor("complex.pdb"), num_samples=8)  # doctest: +SKIP
    """

    config_class = NAMPNNConfig
    input_class: ClassVar[type[BaseInput]]  # bound to NAMPNNInput once that dataclass is defined below.

    def __init__(self, config: NAMPNNConfig | None = None) -> None:
        config = NAMPNNConfig() if config is None else config
        super().__init__(config)
        self.backbone = NAMPNNBackbone(config)
        self.post_init()

    def _init_module_weights(self, module: nn.Module) -> None:
        return None

    def reset_parameters(self) -> None:
        r"""
        Re-initialize parameters with the released NA-MPNN Xavier-uniform scheme.

        Examples:
            >>> model = NAMPNNDesigner()  # doctest: +SKIP
            >>> model.reset_parameters()  # doctest: +SKIP
        """

        for parameter in self.backbone.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        batch: NAMPNNInput,
        *,
        use_sequence: bool = True,
    ) -> SequenceDesignCandidate:
        r"""
        Teacher-forced training forward: one graph encode plus a single-order autoregressive decode.

        This is the TRAINING path and mirrors the upstream NA-MPNN ``ProteinMPNN.forward`` (encode, then
        teacher-forced decode over one noised decoding order, returning per-position ``log_probs`` for the
        masked cross-entropy loss). It is a lean, ``.item()``-free tensor path built to trace under
        ``torch.compile(model, fullgraph=True)`` for both forward and backward; the batched inference verbs
        [`score`][ofoldx.models.nampnn.model.NAMPNNDesigner.score] and
        [`design`][ofoldx.models.nampnn.model.NAMPNNDesigner.design] cover the distinct inference path.

        Args:
            batch: Typed NA-MPNN input carrying atom coords and sequence tokens.
            use_sequence: Whether the decoder conditions on the input sequence identities (teacher forcing).

        Returns:
            Design output with per-position logits, log-probabilities, and the masked sequence log-likelihood.

        Examples:
            >>> output = model(batch)  # doctest: +SKIP
        """

        features = self._training_feature_dict(batch)
        representations = self._representations(features)
        logits = self.backbone.decode_teacher_forced(
            representations.single.tensor,
            representations.edges,
            representations.neighbor_idx,
            features["S"],
            features["mask"],
            features["chain_mask"],
            features["randn"],
            use_sequence=use_sequence,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        return SequenceDesignCandidate(
            sequence_logits=logits,
            sequence_tokens=features["S"],
            scores={
                "log_probs": log_probs,
                "sequence_log_likelihood": _sequence_log_likelihood(features["S"], log_probs, features["mask"]),
            },
            metadata={
                "architecture": "nampnn",
                "backend": "native_nampnn",
                "num_edges": self.config.k_neighbors,
                "use_sequence": use_sequence,
                "mode": "training",
            },
        )

    def _representations(self, features: Mapping[str, Any]) -> NAMPNNRepresentations:
        r"""
        Run the graph encoder and wrap node states in the typed [`Single`][ofoldx.modules.representation.Single]
        carrier at the representations boundary.

        The k-nearest-neighbor edge features have shape `(B, L, K, hidden_size)` -- a sparse per-edge tensor,
        not a dense square `(i, j)` pair -- so they ride alongside as a raw tensor rather than in the
        [`Pair`][ofoldx.modules.representation.Pair] carrier, whose contract requires equal `i`/`j` axes.
        """

        node_states, edge_states, neighbor_idx = self.backbone.encode(features)
        return NAMPNNRepresentations(single=Single(node_states), edges=edge_states, neighbor_idx=neighbor_idx)

    def _training_feature_dict(self, batch: NAMPNNInput) -> dict[str, Tensor]:
        r"""
        Build the tensor-only feature mapping consumed by the teacher-forced training forward.

        Unlike [`_feature_dict`][ofoldx.models.nampnn.model.NAMPNNDesigner._feature_dict] (which carries the
        sampler controls, symmetry lists, RNG, and shape validation that would break a compiled graph), this
        keeps the training pass a pure tensor path: every value is a tensor derived straight from ``batch``.
        """

        dtype = batch.atom_coords.dtype
        mask = batch.token_mask.to(dtype=dtype)
        chain_mask = batch.design_mask.to(dtype=dtype) if batch.design_mask is not None else torch.ones_like(mask)
        noise = batch.decoding_order_noise
        randn = torch.randn_like(mask) if noise is None else noise.to(dtype=dtype)
        return {
            "X": batch.atom_coords,
            "X_m": batch.atom_mask,
            "S": batch.sequence_tokens.long(),
            "mask": mask,
            "R_idx": batch.residue_index.long(),
            "chain_labels": batch.chain_index.long(),
            "chain_mask": chain_mask,
            "protein_mask": batch.protein_mask.to(dtype=dtype),
            "dna_mask": batch.dna_mask.to(dtype=dtype),
            "rna_mask": batch.rna_mask.to(dtype=dtype),
            "R_polymer_type": batch.polymer_type_ids.long(),
            "randn": randn,
        }

    def score(
        self,
        batch: NAMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Score sequence tokens under the NA-MPNN autoregressive decoder.

        Args:
            batch: Typed NA-MPNN input.
            use_sequence: Whether to condition on the input sequence.
            num_samples: Number of decode orders to score; ``None`` uses the batch default.

        Returns:
            Design output whose ``scores`` include ``log_probs`` and ``sequence_log_likelihood``.

        Examples:
            >>> scored = model.score(batch)  # doctest: +SKIP
            >>> scored.scores["log_probs"].shape  # doctest: +SKIP
            torch.Size([1, 12, 33])
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.score(features, use_sequence=use_sequence)
        log_probs = scored["log_probs"]
        return SequenceDesignCandidate(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={
                "log_probs": log_probs,
                "sequence_log_likelihood": _sequence_log_likelihood(scored["S"], log_probs, features["mask"]),
            },
            metadata={
                "architecture": "nampnn",
                "backend": "native_nampnn",
                "num_edges": self.config.k_neighbors,
                "use_sequence": use_sequence,
            },
        )

    @torch.no_grad()
    def design(
        self,
        batch: NAMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Generate designed polymer sequences under ``torch.no_grad``.

        Args:
            batch: Typed NA-MPNN input for one backbone.
            temperature: Sampling temperature; ``None`` uses the batch default.
            num_samples: Number of independent samples; ``None`` uses the batch default.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Design output with sampled tokens and sampling probabilities.

        Examples:
            >>> output = model.design(batch, temperature=0.1, num_samples=8)  # doctest: +SKIP
        """

        return self._sample(batch, temperature=temperature, num_samples=num_samples, generator=generator)

    def _sample(
        self,
        batch: NAMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Autoregressively sample protein, DNA, and RNA sequence tokens (the inference sampler behind
        [`design`][ofoldx.models.nampnn.model.NAMPNNDesigner.design]).

        The public inference entry point is ``design``; this runs the N-step Python decoding loop it wraps.

        Args:
            batch: Typed NA-MPNN input for one backbone.
            temperature: Sampling temperature; ``None`` uses the batch default.
            num_samples: Number of independent samples; ``None`` uses the batch default.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Design output with sampled tokens, logits, log-probabilities, and sampling probabilities.
        """

        features = self._feature_dict(
            batch,
            temperature=temperature,
            num_samples=num_samples,
            generator=generator,
            for_sampling=True,
        )
        sampled = self.backbone.sample(features)
        return SequenceDesignCandidate(
            sequence_logits=sampled["logits"],
            sequence_tokens=sampled["S"],
            scores={
                "log_probs": sampled["log_probs"],
                "sampling_probs": sampled["sampling_probs"],
                "sequence_log_likelihood": _sequence_log_likelihood(
                    sampled["S"], sampled["log_probs"], features["mask"] * features["chain_mask"]
                ),
            },
            metadata={
                "architecture": "nampnn",
                "backend": "native_nampnn",
                "num_edges": self.config.k_neighbors,
                "temperature": features["temperature"],
            },
        )

    def unconditional_probs(
        self,
        batch: NAMPNNInput,
        *,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Predict per-position log-probabilities from backbone context alone.

        Args:
            batch: Typed NA-MPNN input.
            num_samples: Number of repeated decode passes; ``None`` uses the batch default.

        Returns:
            Design output whose ``scores`` include unconditional ``log_probs``.

        Examples:
            >>> output = model.unconditional_probs(batch)  # doctest: +SKIP
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        output = self.backbone.unconditional_probs(features)
        return SequenceDesignCandidate(
            sequence_logits=output["logits"],
            sequence_tokens=output["S"],
            scores={"log_probs": output["log_probs"]},
            metadata={"architecture": "nampnn", "backend": "native_nampnn", "score_mode": "unconditional"},
        )

    def _feature_dict(
        self,
        batch: NAMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        for_sampling: bool,
    ) -> dict[str, Any]:
        r"""
        Convert a typed input dataclass into the tensor mapping consumed by the backbone.
        """

        _validate_nampnn_input(batch, require_sampling=for_sampling, vocab_size=self.config.vocab_size)
        sample_count = int(num_samples if num_samples is not None else (batch.num_samples or 1))
        if sample_count <= 0:
            raise ValueError("num_samples must be positive")
        device = batch.sequence_tokens.device
        length = batch.sequence_tokens.shape[1]
        randn = batch.decoding_order_noise
        if randn is None:
            randn = torch.randn(sample_count, length, device=device, generator=generator)
        else:
            randn = randn.to(device=device)
            sample_count = int(randn.shape[0])
        default_bias = torch.zeros(
            *batch.sequence_tokens.shape,
            self.config.vocab_size,
            dtype=batch.atom_coords.dtype,
            device=device,
        )
        bias = default_bias if batch.amino_acid_bias is None else batch.amino_acid_bias.to(device=device)
        features = {
            "X": batch.atom_coords,
            "X_m": batch.atom_mask,
            "S": batch.sequence_tokens.long(),
            "mask": batch.token_mask.to(device=device, dtype=batch.atom_coords.dtype),
            "R_idx": batch.residue_index.long(),
            "chain_labels": batch.chain_index.long(),
            "chain_mask": _optional_mask(batch.design_mask, batch.token_mask),
            "protein_mask": batch.protein_mask.to(device=device, dtype=batch.atom_coords.dtype),
            "dna_mask": batch.dna_mask.to(device=device, dtype=batch.atom_coords.dtype),
            "rna_mask": batch.rna_mask.to(device=device, dtype=batch.atom_coords.dtype),
            "R_polymer_type": batch.polymer_type_ids.long(),
            "batch_size": sample_count,
            "num_samples": sample_count,
            "randn": randn,
            "temperature": temperature if temperature is not None else batch.temperature,
            "bias": bias,
            "symmetry_residues": batch.symmetry_residue_groups or [[]],
            "symmetry_weights": batch.symmetry_weights or [[]],
            "fixed_position_mask": batch.fixed_position_mask,
            "generator": generator,
        }
        if batch.pair_amino_acid_bias is not None:
            features["pair_bias"] = batch.pair_amino_acid_bias.to(device=device)
        return features


class NAMPNNBackbone(nn.Module):
    r"""
    NA-MPNN graph encoder and autoregressive decoder.

    Args:
        config: Architecture configuration providing widths, layer counts, and atom/polymer vocab sizes.
    """

    def __init__(self, config: NAMPNNConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.feature_encoder = NAMPNNFeatureEncoder(config)
        self.node_embedding = nn.Linear(config.node_feature_size, config.hidden_size)
        self.edge_embedding = nn.Linear(config.edge_feature_size, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_layers = nn.ModuleList(
            [
                ProteinEncoderLayer(config.hidden_size, config.hidden_size * 2, dropout=config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                ProteinDecoderLayer(config.hidden_size, config.hidden_size * 3, dropout=config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def encode(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Encode polymer geometry into node and edge states.

        Args:
            features: Feature mapping with atom coords, masks, residue indices, and polymer ids.

        Returns:
            A ``(node_states, edge_states, neighbor_idx)`` tuple.
        """

        node_features, edge_features, neighbor_idx = self.feature_encoder(features)
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edge_features)
        mask = features["mask"]
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_idx).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(node_states, edge_states, neighbor_idx, mask, attend_mask)
        return node_states, edge_states, neighbor_idx

    def decode_teacher_forced(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        randn: Tensor,
        *,
        use_sequence: bool,
    ) -> Tensor:
        r"""
        Teacher-forced autoregressive decode over a single noised decoding order (training path).

        Mirrors the decode half of the upstream NA-MPNN ``ProteinMPNN.forward``: it conditions each position on
        the true neighbor tokens under a backward autoregressive mask and returns the vocabulary logits. Pure
        tensor ops with no ``.item()`` so it traces under ``torch.compile(fullgraph=True)``.

        Args:
            node_states: Encoded node states `[B, L, hidden_size]`.
            edge_states: Encoded edge states `[B, L, K, hidden_size]`.
            neighbor_idx: Neighbor index `[B, L, K]`.
            sequence: True residue tokens `[B, L]` used for teacher forcing.
            mask: Valid-token mask `[B, L]`.
            chain_mask: Designable-position mask `[B, L]`.
            randn: Decoding-order noise `[B, L]`.
            use_sequence: Whether to condition on the true neighbor tokens.

        Returns:
            Per-position vocabulary logits `[B, L, vocab_size]`.
        """

        decode_mask = mask * chain_mask
        decode_order = torch.argsort((decode_mask + 0.0001) * randn.abs())
        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)

        token_states = self.token_embedding(sequence)
        edge_token_states = cat_neighbors_nodes(token_states, edge_states, neighbor_idx)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        if use_sequence:
            for layer in self.decoder_layers:
                decoder_context = cat_neighbors_nodes(node_states, edge_token_states, neighbor_idx)
                node_states = layer(node_states, backward_mask * decoder_context + forward_context, mask)
        else:
            for layer in self.decoder_layers:
                node_states = layer(node_states, forward_context, mask)
        return self.output_projection(node_states)

    def score(self, features: Mapping[str, Any], *, use_sequence: bool) -> dict[str, Tensor]:
        r"""
        Tensor-level scoring pass.

        Args:
            features: Feature mapping built by
                [`NAMPNNDesigner._feature_dict`][ofoldx.models.nampnn.model.NAMPNNDesigner].
            use_sequence: Whether the decoder conditions on the input sequence.

        Returns:
            Mapping with sequence tokens, logits, log-probabilities, and decode order.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"]
        randn = features["randn"]
        num_samples = int(features["num_samples"])

        node_states, edge_states, neighbor_idx = self.encode(features)
        decode_order = torch.argsort((chain_mask + 0.0001) * randn.abs())
        neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
        mask = mask.repeat(num_samples, 1)
        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        sequence = sequence.repeat(num_samples, 1)
        node_states = node_states.repeat(num_samples, 1, 1)
        edge_states = edge_states.repeat(num_samples, 1, 1, 1)

        token_states = self.token_embedding(sequence)
        edge_token_states = cat_neighbors_nodes(token_states, edge_states, neighbor_idx)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        if use_sequence:
            for layer in self.decoder_layers:
                decoder_context = cat_neighbors_nodes(node_states, edge_token_states, neighbor_idx)
                node_states = layer(node_states, backward_mask * decoder_context + forward_context, mask)
        else:
            for layer in self.decoder_layers:
                node_states = layer(node_states, forward_context, mask)

        logits = self.output_projection(node_states)
        return {
            "S": sequence,
            "logits": logits,
            "log_probs": F.log_softmax(logits, dim=-1),
            "decoding_order": decode_order,
        }

    def sample(self, features: Mapping[str, Any]) -> dict[str, Tensor]:
        r"""
        Tensor-level autoregressive sampler.

        Args:
            features: Feature mapping with sampling controls and conditioning masks.

        Returns:
            Mapping with sampled tokens, logits, log-probabilities, sampling probabilities, and decode order.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"] * _fixed_position_mask(features, mask)
        bias = features["bias"]
        randn = features["randn"]
        num_samples = int(features["num_samples"])
        symmetry_groups = features["symmetry_residues"]

        node_states, edge_states, neighbor_idx = self.encode(features)
        decode_order = torch.argsort((chain_mask + 0.0001) * randn.abs())
        if _has_no_symmetry(symmetry_groups):
            return self._sample_independent(
                sequence,
                mask,
                chain_mask,
                bias,
                features.get("pair_bias"),
                features["temperature"],
                decode_order,
                node_states,
                edge_states,
                neighbor_idx,
                num_samples,
                features.get("generator"),
            )
        return self._sample_symmetric(
            sequence,
            mask,
            chain_mask,
            bias,
            features.get("pair_bias"),
            features["temperature"],
            decode_order,
            symmetry_groups,
            features["symmetry_weights"],
            node_states,
            edge_states,
            neighbor_idx,
            num_samples,
            features.get("generator"),
        )

    def _decode_state(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        pair_bias: Tensor | None,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
    ) -> dict[str, Any]:
        r"""
        Initialize tensors shared by the sampling loops.
        """

        _, length = sequence.shape
        device = sequence.device
        return {
            "sequence_true": sequence.repeat(num_samples, 1),
            "mask": mask.repeat(num_samples, 1),
            "chain_mask": chain_mask.repeat(num_samples, 1),
            "bias": bias.repeat(num_samples, 1, 1),
            "pair_bias": None if pair_bias is None else pair_bias.repeat(num_samples, 1, 1, 1, 1),
            "node_states": node_states.repeat(num_samples, 1, 1),
            "edge_states": edge_states.repeat(num_samples, 1, 1, 1),
            "neighbor_idx": neighbor_idx.repeat(num_samples, 1, 1),
            "sampled": (self.vocab_size - 1) * torch.ones(num_samples, length, dtype=torch.long, device=device),
            "token_states": node_states.new_zeros(num_samples, length, node_states.shape[-1]),
            "layer_states": [
                node_states.repeat(num_samples, 1, 1),
                *[node_states.new_zeros(num_samples, length, node_states.shape[-1]) for _ in self.decoder_layers],
            ],
            "probabilities": node_states.new_zeros(num_samples, length, self.vocab_size),
            "log_probs_out": node_states.new_zeros(num_samples, length, self.vocab_size),
            "logits_out": node_states.new_zeros(num_samples, length, self.vocab_size),
        }

    def _sample_independent(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        pair_bias: Tensor | None,
        temperature: float | Tensor,
        decode_order: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Independent-position autoregressive decoding loop.
        """

        _, length = sequence.shape
        state = self._decode_state(
            sequence, mask, chain_mask, bias, pair_bias, node_states, edge_states, neighbor_idx, num_samples
        )
        neighbor_idx = state["neighbor_idx"]
        mask = state["mask"]
        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(state["token_states"]), state["edge_states"], neighbor_idx)
        encoder_context = cat_neighbors_nodes(state["node_states"], encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        for step in range(length):
            residue = decode_order[:, step]
            logits = self._decode_residue(state, residue, backward_mask, forward_context)
            bias_t = _gather_position(state["bias"], residue)
            pair_bias_t = _pair_bias_at(state["pair_bias"], residue, state["sampled"], self.vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = _sampling_distribution(logits, bias_t, pair_bias_t, temperature)
            sampled_t = torch.multinomial(probs, 1, generator=generator)[:, 0]
            chain_mask_t = torch.gather(state["chain_mask"], 1, residue[:, None])[:, 0]
            true_t = torch.gather(state["sequence_true"], 1, residue[:, None])[:, 0]
            sampled_t = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
            self._scatter_decode_result(state, residue, sampled_t, probs, logits, log_probs, chain_mask_t)

        return {
            "S": state["sampled"],
            "sampling_probs": state["probabilities"],
            "logits": state["logits_out"],
            "log_probs": state["log_probs_out"],
            "decoding_order": decode_order,
        }

    def _sample_symmetric(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        pair_bias: Tensor | None,
        temperature: float | Tensor,
        decode_order: Tensor,
        symmetry_groups: list[list[int]],
        symmetry_weights_groups: list[list[float]],
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Symmetry-aware autoregressive decoding loop.
        """

        _, length = sequence.shape
        device = sequence.device
        ordered_groups = _decode_groups(decode_order[0], symmetry_groups)
        flat_order = torch.tensor(list(itertools.chain(*ordered_groups)), device=device)[None, :]
        decode_order = flat_order.repeat(num_samples, 1)
        symmetry_weights = torch.ones(length, dtype=node_states.dtype, device=device)
        for group_index, residue_group in enumerate(symmetry_groups):
            for member_index, residue in enumerate(residue_group):
                symmetry_weights[residue] = float(symmetry_weights_groups[group_index][member_index])

        state = self._decode_state(
            sequence, mask, chain_mask, bias, pair_bias, node_states, edge_states, neighbor_idx, num_samples
        )
        neighbor_idx = state["neighbor_idx"]
        mask = state["mask"]
        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(state["token_states"]), state["edge_states"], neighbor_idx)
        encoder_context = cat_neighbors_nodes(state["node_states"], encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        for residue_group in ordered_groups:
            group_logits = 0.0
            group_bias = None
            group_pair_bias = None
            for residue_int in residue_group:
                residue = torch.full((num_samples,), residue_int, dtype=torch.long, device=device)
                logits = self._decode_residue(state, residue, backward_mask, forward_context)
                log_probs = F.log_softmax(logits, dim=-1)
                chain_mask_t = state["chain_mask"][:, residue_int]
                state["log_probs_out"][:, residue_int] = chain_mask_t[:, None] * log_probs
                group_logits = group_logits + symmetry_weights[residue_int] * logits
                group_bias = state["bias"][:, residue_int]
                group_pair_bias = _pair_bias_at(state["pair_bias"], residue, state["sampled"], self.vocab_size)

            assert group_bias is not None
            probs = _sampling_distribution(group_logits, group_bias, group_pair_bias, temperature)
            sampled_t = torch.multinomial(probs, 1, generator=generator)[:, 0]
            for residue_int in residue_group:
                residue = torch.full((num_samples,), residue_int, dtype=torch.long, device=device)
                chain_mask_t = state["chain_mask"][:, residue_int]
                true_t = state["sequence_true"][:, residue_int]
                token = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
                self._scatter_decode_result(
                    state,
                    residue,
                    token,
                    probs,
                    group_logits,
                    state["log_probs_out"][:, residue_int],
                    chain_mask_t,
                )

        return {
            "S": state["sampled"],
            "sampling_probs": state["probabilities"],
            "logits": state["logits_out"],
            "log_probs": state["log_probs_out"],
            "decoding_order": decode_order,
        }

    def _decode_residue(
        self,
        state: Mapping[str, Any],
        residue: Tensor,
        backward_mask: Tensor,
        forward_context: Tensor,
    ) -> Tensor:
        r"""
        Decode one residue position from the current autoregressive state.
        """

        neighbor_idx = state["neighbor_idx"]
        edge_states = state["edge_states"]
        token_states = state["token_states"]
        layer_states = state["layer_states"]
        edge_t = torch.gather(
            edge_states,
            1,
            residue[:, None, None, None].repeat(1, 1, edge_states.shape[-2], edge_states.shape[-1]),
        )
        neighbor_idx_t = torch.gather(neighbor_idx, 1, residue[:, None, None].repeat(1, 1, neighbor_idx.shape[-1]))
        edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
        forward_t = torch.gather(
            forward_context,
            1,
            residue[:, None, None, None].repeat(1, 1, forward_context.shape[-2], forward_context.shape[-1]),
        )
        backward_t = torch.gather(
            backward_mask,
            1,
            residue[:, None, None, None].repeat(1, 1, backward_mask.shape[-2], backward_mask.shape[-1]),
        )
        mask_t = torch.gather(state["mask"], 1, residue[:, None])[:, 0]
        for layer_index, layer in enumerate(self.decoder_layers):
            decoder_context = cat_neighbors_nodes(layer_states[layer_index], edge_token_t, neighbor_idx_t)
            node_t = torch.gather(
                layer_states[layer_index],
                1,
                residue[:, None, None].repeat(1, 1, layer_states[layer_index].shape[-1]),
            )
            updated = layer(node_t, backward_t * decoder_context + forward_t, mask_t)
            layer_states[layer_index + 1].scatter_(
                1,
                residue[:, None, None].repeat(1, 1, updated.shape[-1]),
                updated,
            )
        final_t = torch.gather(
            layer_states[-1],
            1,
            residue[:, None, None].repeat(1, 1, layer_states[-1].shape[-1]),
        )[:, 0]
        return self.output_projection(final_t)

    def _scatter_decode_result(
        self,
        state: Mapping[str, Any],
        residue: Tensor,
        token: Tensor,
        probs: Tensor,
        logits: Tensor,
        log_probs: Tensor,
        chain_mask_t: Tensor,
    ) -> None:
        r"""
        Scatter one decoded position back into the running sampling state.
        """

        token_states = state["token_states"]
        state["probabilities"].scatter_(
            1,
            residue[:, None, None].repeat(1, 1, self.vocab_size),
            chain_mask_t[:, None, None] * probs[:, None, :],
        )
        state["logits_out"].scatter_(
            1,
            residue[:, None, None].repeat(1, 1, self.vocab_size),
            chain_mask_t[:, None, None] * logits[:, None, :],
        )
        state["log_probs_out"].scatter_(
            1,
            residue[:, None, None].repeat(1, 1, self.vocab_size),
            chain_mask_t[:, None, None] * log_probs[:, None, :],
        )
        state["sampled"].scatter_(1, residue[:, None], token[:, None])
        token_states.scatter_(
            1,
            residue[:, None, None].repeat(1, 1, token_states.shape[-1]),
            self.token_embedding(token)[:, None, :],
        )

    def unconditional_probs(self, features: Mapping[str, Any]) -> dict[str, Tensor]:
        r"""
        Tensor-level unconditional probability pass.

        Args:
            features: Feature mapping built by the model wrapper.

        Returns:
            Mapping with sequence tokens, logits, and log-probabilities.
        """

        sequence = features["S"]
        mask = features["mask"]
        num_samples = int(features["num_samples"])
        node_states, edge_states, neighbor_idx = self.encode(features)
        neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
        mask = mask.repeat(num_samples, 1)
        order_mask = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[1], dtype=mask.dtype, device=mask.device)
        attend = torch.gather(order_mask, 2, neighbor_idx).unsqueeze(-1)
        node_states = node_states.repeat(num_samples, 1, 1)
        edge_states = edge_states.repeat(num_samples, 1, 1, 1)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = mask.view(mask.shape[0], mask.shape[1], 1, 1) * (1.0 - attend) * encoder_context
        for layer in self.decoder_layers:
            node_states = layer(node_states, forward_context, mask)
        logits = self.output_projection(node_states)
        return {
            "S": sequence.repeat(num_samples, 1),
            "logits": logits,
            "log_probs": F.log_softmax(logits, dim=-1),
        }


class NAMPNNFeatureEncoder(nn.Module):
    r"""
    Geometric feature encoder for unified protein/nucleic-acid polymer graphs.

    Args:
        config: NA-MPNN architecture configuration.
    """

    def __init__(self, config: NAMPNNConfig) -> None:
        super().__init__()
        if config.na_ref_atom not in NAMPNN_ATOM_TO_ID:
            raise ValueError(f"unknown NA reference atom {config.na_ref_atom!r}")
        self.k_neighbors = config.k_neighbors
        self.num_rbf = config.num_rbf
        self.num_polymer_types = config.num_polymer_types
        self.include_predicted_na_n = bool(config.include_predicted_na_n)
        self.protein_coord_noise_scale = _noise_scale(config.protein_coord_noise_scale, config.coord_noise_scale)
        self.dna_coord_noise_scale = _noise_scale(config.dna_coord_noise_scale, config.coord_noise_scale)
        self.rna_coord_noise_scale = _noise_scale(config.rna_coord_noise_scale, config.coord_noise_scale)
        self.n_idx = NAMPNN_ATOM_TO_ID["N"]
        self.ca_idx = NAMPNN_ATOM_TO_ID["CA"]
        self.c_idx = NAMPNN_ATOM_TO_ID["C"]
        self.o4prime_idx = NAMPNN_ATOM_TO_ID["O4'"]
        self.c1prime_idx = NAMPNN_ATOM_TO_ID["C1'"]
        self.c2prime_idx = NAMPNN_ATOM_TO_ID["C2'"]
        self.na_ref_atom_idx = NAMPNN_ATOM_TO_ID[config.na_ref_atom]
        total_atoms = config.num_atom_types + 1 + int(self.include_predicted_na_n)
        self.relative_position = RelativePositionEncoding(config.num_positional_embeddings)
        self.node_projection = nn.Linear(config.num_polymer_types, config.node_feature_size, bias=False)
        self.node_norm = nn.LayerNorm(config.node_feature_size)
        self.edge_projection = nn.Linear(
            config.num_positional_embeddings + config.num_rbf * total_atoms * total_atoms,
            config.edge_feature_size,
            bias=False,
        )
        self.edge_norm = nn.LayerNorm(config.edge_feature_size)

    def forward(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Build node features, edge features, and nearest-neighbor indices.

        Args:
            features: Feature mapping with atom coords, atom masks, token masks, and polymer ids.

        Returns:
            ``(node_features, edge_features, neighbor_idx)``.
        """

        dtype = self.edge_projection.weight.dtype
        coords = features["X"].to(dtype=dtype)
        atom_mask = features["X_m"].to(dtype=dtype)
        token_mask = features["mask"].to(dtype=dtype)
        protein_mask = features["protein_mask"].to(dtype=dtype)
        dna_mask = features["dna_mask"].to(dtype=dtype)
        rna_mask = features["rna_mask"].to(dtype=dtype)
        if self.training and (
            self.protein_coord_noise_scale > 0 or self.dna_coord_noise_scale > 0 or self.rna_coord_noise_scale > 0
        ):
            scale = (
                protein_mask * self.protein_coord_noise_scale
                + dna_mask * self.dna_coord_noise_scale
                + rna_mask * self.rna_coord_noise_scale
            )
            coords = coords + atom_mask[:, :, :, None] * scale[:, :, None, None] * torch.randn_like(coords)

        ca = coords[:, :, self.ca_idx]
        n_atom = coords[:, :, self.n_idx]
        c_atom = coords[:, :, self.c_idx]
        cb = _virtual_cb(n_atom, ca, c_atom)
        na_ref = coords[:, :, self.na_ref_atom_idx]
        augmented_coords = [coords, cb[:, :, None, :]]
        augmented_mask = [atom_mask, protein_mask[:, :, None]]
        if self.include_predicted_na_n:
            o4prime = coords[:, :, self.o4prime_idx]
            c1prime = coords[:, :, self.c1prime_idx]
            c2prime = coords[:, :, self.c2prime_idx]
            na_n = _virtual_na_n(o4prime, c1prime, c2prime)
            augmented_coords.append(na_n[:, :, None, :])
            augmented_mask.append((dna_mask + rna_mask).clamp(max=1.0)[:, :, None])
        all_coords = torch.cat(augmented_coords, dim=-2)
        all_mask = torch.cat(augmented_mask, dim=-1)

        _, neighbor_idx = _knn(ca + na_ref, token_mask, self.k_neighbors)
        pair_rbf = _all_pair_rbf(all_coords, all_mask, neighbor_idx, self.num_rbf)
        residue_offset = features["R_idx"][:, :, None] - features["R_idx"][:, None, :]
        residue_offset = gather_edges(residue_offset[:, :, :, None], neighbor_idx)[:, :, :, 0]
        same_chain = (features["chain_labels"][:, :, None] == features["chain_labels"][:, None, :]).long()
        same_chain = gather_edges(same_chain[:, :, :, None], neighbor_idx)[:, :, :, 0]
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_features = self.edge_norm(self.edge_projection(torch.cat((position_features, pair_rbf), dim=-1)))

        polymer_one_hot = F.one_hot(features["R_polymer_type"], num_classes=self.num_polymer_types).to(dtype=dtype)
        node_features = self.node_norm(self.node_projection(polymer_one_hot))
        return node_features, edge_features, neighbor_idx


@register_pytree_dataclass
@dataclass(slots=True)
class NAMPNNRepresentations:
    r"""
    Family-owned encoder outputs at the representations boundary.

    Node states are wrapped in the typed [`Single`][ofoldx.modules.representation.Single] carrier (the genuine
    per-token representation), the way [`Coords`][ofoldx.modules.representation.Coords] wraps predicted
    coordinates on structure-prediction outputs. The k-nearest-neighbor edge features are a sparse
    `(B, L, K, hidden_size)` per-edge tensor -- not a dense square `(i, j)` pair -- so they ride alongside as a
    raw tensor rather than in the [`Pair`][ofoldx.modules.representation.Pair] carrier (which requires equal
    `i`/`j` axes). Registered as a pytree node so the carrier boundary introduces no ``torch.compile`` graph
    break.

    Attributes:
        single: Encoded node states `[B, L, hidden_size]` wrapped in the ``Single`` carrier.
        edges: Encoded neighbor edge states `[B, L, K, hidden_size]`.
        neighbor_idx: Neighbor index `[B, L, K]`.
    """

    single: Single
    edges: Tensor
    neighbor_idx: Tensor


@register_pytree_dataclass
@dataclass(slots=True, kw_only=True)
class NAMPNNInput(BaseInput):
    r"""
    Batched atom/sequence tensors and sampler controls consumed by NA-MPNN.

    Attributes:
        atom_coords: Atom coords in NA-MPNN atom order with shape `(B, L, 16, 3)`.
        atom_mask: Atom-validity mask with shape `(B, L, 16)`.
        sequence_tokens: Residue tokens in the NA-MPNN alphabet with shape `(B, L)`.
        token_mask: Valid-token mask with shape `(B, L)`.
        residue_index: Residue numbers used for relative-position features with shape `(B, L)`.
        chain_index: Integer chain ids with shape `(B, L)`.
        polymer_type_ids: Polymer-type ids with shape `(B, L)`.
        protein_mask: Protein-token mask with shape `(B, L)`.
        dna_mask: DNA-token mask with shape `(B, L)`.
        rna_mask: RNA-token mask with shape `(B, L)`.
        design_mask: Optional mask whose ones mark positions to design.
        amino_acid_bias: Optional per-position token logit bias with shape `(B, L, 33)`.
        pair_amino_acid_bias: Optional nearest-neighbor token-pair bias with shape `(B, L, 33, L, 33)`.
        decoding_order_noise: Optional decode-order noise with shape `(num_samples, L)`.
        temperature: Sampling temperature.
        num_samples: Number of autoregressive samples.
        symmetry_residue_groups: Residue groups decoded with tied identities.
        symmetry_weights: Logit weights aligned with ``symmetry_residue_groups``.
        fixed_position_mask: Optional hard-fixed-position mask with shape `(B, L)`.
        rna_token_conversion_mask: Optional mask identifying RNA residues represented by shared DNA token ids.
    """

    atom_coords: Tensor
    atom_mask: Tensor
    sequence_tokens: Tensor
    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    polymer_type_ids: Tensor
    protein_mask: Tensor
    dna_mask: Tensor
    rna_mask: Tensor
    design_mask: Tensor | None = None
    amino_acid_bias: Tensor | None = None
    pair_amino_acid_bias: Tensor | None = None
    decoding_order_noise: Tensor | None = None
    temperature: float | Tensor = 0.1
    num_samples: int | None = None
    symmetry_residue_groups: list[list[int]] | None = None
    symmetry_weights: list[list[float]] | None = None
    fixed_position_mask: Tensor | None = None
    rna_token_conversion_mask: Tensor | None = None


#: Bind the keyword-tensor construction path (``model(**tensors)``) to this family's typed input.
NAMPNNDesigner.input_class = NAMPNNInput


def _validate_nampnn_input(batch: NAMPNNInput, *, require_sampling: bool, vocab_size: int) -> None:
    r"""
    Validate the shape contract of a typed NA-MPNN input.
    """

    if batch.atom_coords.ndim != 4 or batch.atom_coords.shape[2:] != (len(NAMPNN_ATOM_NAMES), 3):
        raise ValueError("atom_coords must have shape (B, L, 16, 3)")
    if batch.atom_mask.shape != batch.atom_coords.shape[:3]:
        raise ValueError("atom_mask must have shape (B, L, 16)")
    if batch.sequence_tokens.shape != batch.token_mask.shape:
        raise ValueError("sequence_tokens and token_mask must have matching shape (B, L)")
    if batch.atom_coords.shape[:2] != batch.sequence_tokens.shape:
        raise ValueError("atom_coords and sequence_tokens/token_mask must share batch and length axes")
    for name in ("residue_index", "chain_index", "polymer_type_ids", "protein_mask", "dna_mask", "rna_mask"):
        if getattr(batch, name).shape != batch.sequence_tokens.shape:
            raise ValueError(f"{name} must have shape (B, L)")
    if batch.design_mask is not None and batch.design_mask.shape != batch.sequence_tokens.shape:
        raise ValueError("design_mask must have shape (B, L)")
    if batch.fixed_position_mask is not None and batch.fixed_position_mask.shape != batch.sequence_tokens.shape:
        raise ValueError("fixed_position_mask must have shape (B, L)")
    if batch.amino_acid_bias is not None and batch.amino_acid_bias.shape != (*batch.sequence_tokens.shape, vocab_size):
        raise ValueError(f"amino_acid_bias must have shape (B, L, {vocab_size})")
    expected_pair_shape = (*batch.sequence_tokens.shape, vocab_size, batch.sequence_tokens.shape[1], vocab_size)
    if batch.pair_amino_acid_bias is not None and batch.pair_amino_acid_bias.shape != expected_pair_shape:
        raise ValueError(f"pair_amino_acid_bias must have shape {expected_pair_shape}")
    if batch.decoding_order_noise is not None and (
        batch.decoding_order_noise.ndim != 2 or batch.decoding_order_noise.shape[1] != batch.sequence_tokens.shape[1]
    ):
        raise ValueError("decoding_order_noise must have shape (num_samples, L)")
    if require_sampling and batch.sequence_tokens.shape[0] != 1:
        raise ValueError("NA-MPNN sampling follows the official single-backbone batch contract")


def _optional_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    r"""
    Return ``mask`` as ``reference``'s dtype/device, or an all-ones mask when absent.
    """

    if mask is None:
        return torch.ones_like(reference)
    return mask.to(device=reference.device, dtype=reference.dtype)


def _noise_scale(polymer_scale: float | None, default: float) -> float:
    r"""
    Resolve a polymer-specific noise scale.
    """

    return float(default if polymer_scale is None else polymer_scale)


def _virtual_cb(n_atom: Tensor, ca: Tensor, c_atom: Tensor) -> Tensor:
    r"""
    Reconstruct the idealized protein Cβ coordinate from N, Cα, and C.
    """

    b = ca - n_atom
    c = c_atom - ca
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca


def _virtual_na_n(o4prime: Tensor, c1prime: Tensor, c2prime: Tensor) -> Tensor:
    r"""
    Reconstruct the idealized nucleic-acid base nitrogen coordinate.
    """

    b = c1prime - o4prime
    c = c2prime - c1prime
    a = torch.cross(b, c, dim=-1)
    return -0.56967352 * a + 0.51055973 * b - 0.53122153 * c + c1prime


def _knn(points: Tensor, mask: Tensor, k_neighbors: int, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    r"""
    Find nearest valid graph neighbors for every token.
    """

    pair_mask = mask[:, None, :] * mask[:, :, None]
    distances = torch.sqrt(torch.sum((points[:, :, None] - points[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    max_distance = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * max_distance
    return torch.topk(adjusted, min(k_neighbors, points.shape[1]), dim=-1, largest=False)


def _rbf(distances: Tensor, num_bins: int) -> Tensor:
    r"""
    Expand distances into Gaussian radial-basis features.
    """

    centers = torch.linspace(2.0, 22.0, num_bins, device=distances.device, dtype=distances.dtype)
    centers = centers.view(*((1,) * distances.ndim), num_bins)
    width = (22.0 - 2.0) / num_bins
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def _all_pair_rbf(coords: Tensor, atom_mask: Tensor, neighbor_idx: Tensor, num_bins: int) -> Tensor:
    r"""
    RBF-encode all ordered atom-pair distances for each neighbor edge.
    """

    flat = coords.reshape(coords.shape[0], coords.shape[1], -1)
    neighbor_flat = gather_nodes(flat, neighbor_idx)
    neighbor_coords = neighbor_flat.reshape(*neighbor_flat.shape[:-1], coords.shape[-2], coords.shape[-1])
    distances = torch.sqrt(torch.sum((coords[:, :, None, :, None, :] - neighbor_coords[:, :, :, None]) ** 2, -1) + 1e-6)
    rbf = _rbf(distances, num_bins)
    neighbor_mask = gather_nodes(atom_mask, neighbor_idx)
    rbf = rbf * atom_mask[:, :, None, :, None, None] * neighbor_mask[:, :, :, None, :, None]
    return rbf.reshape(coords.shape[0], coords.shape[1], neighbor_idx.shape[-1], -1)


def _autoregressive_masks(decode_order: Tensor, neighbor_idx: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Build backward and forward autoregressive neighbor masks.
    """

    length = decode_order.shape[1]
    device = decode_order.device
    reverse = F.one_hot(decode_order, num_classes=length).float()
    backward_order = torch.einsum(
        "ij,biq,bjp->bqp",
        1 - torch.triu(torch.ones(length, length, device=device)),
        reverse,
        reverse,
    )
    attend = torch.gather(backward_order, 2, neighbor_idx).unsqueeze(-1)
    mask = mask.view(mask.shape[0], length, 1, 1)
    return mask * attend, mask * (1.0 - attend)


def _sequence_log_likelihood(sequence: Tensor, log_probs: Tensor, mask: Tensor) -> Tensor:
    r"""
    Compute masked-mean per-sequence token log-likelihood.
    """

    token_log_probs = torch.gather(log_probs, -1, sequence[..., None]).squeeze(-1)
    mask = mask.to(dtype=token_log_probs.dtype)
    if mask.shape[0] != token_log_probs.shape[0]:
        mask = mask.repeat(token_log_probs.shape[0] // mask.shape[0], 1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _fixed_position_mask(features: Mapping[str, Any], mask: Tensor) -> Tensor:
    r"""
    Return the hard-fixed-position mask, defaulting to ones.
    """

    fixed = features.get("fixed_position_mask")
    if fixed is None:
        return torch.ones_like(mask)
    return fixed.to(device=mask.device, dtype=mask.dtype)


def _gather_position(tensor: Tensor, position: Tensor) -> Tensor:
    r"""
    Select per-batch rows from a `(B, L, C)` tensor.
    """

    index = position[:, None, None].expand(-1, 1, tensor.shape[-1])
    return torch.gather(tensor, 1, index)[:, 0, :]


def _pair_bias_at(pair_bias: Tensor | None, residue: Tensor, sampled: Tensor, vocab_size: int) -> Tensor | None:
    r"""
    Gather pairwise token bias induced by already-sampled positions.
    """

    if pair_bias is None:
        return None
    selected = torch.gather(
        pair_bias,
        1,
        residue[:, None, None, None, None].repeat(1, 1, vocab_size, sampled.shape[1], vocab_size),
    )[:, 0]
    selected = torch.gather(selected, -1, sampled[:, None, :, None].repeat(1, vocab_size, 1, 1))[:, :, :, 0]
    return selected.sum(dim=-1)


def _sampling_distribution(
    logits: Tensor,
    bias: Tensor,
    pair_bias: Tensor | None,
    temperature: float | Tensor,
) -> Tensor:
    r"""
    Convert logits to a valid NA-MPNN sampling distribution.
    """

    total = logits + bias
    if pair_bias is not None:
        total = total + pair_bias
    probs = F.softmax(total / temperature, dim=-1)
    probs = probs.clone()
    probs[:, list(NAMPNN_INVALID_SAMPLE_TOKEN_IDS)] = 0.0
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)


def _has_no_symmetry(groups: list[list[int]]) -> bool:
    r"""
    Return whether ``groups`` is the empty-symmetry sentinel.
    """

    return len(groups) == 1 and len(groups[0]) == 0


def _decode_groups(decode_order: Tensor, symmetry_groups: list[list[int]]) -> list[list[int]]:
    r"""
    Expand a per-position decode order into symmetry groups.
    """

    ordered: list[list[int]] = []
    seen: set[int] = set()
    for residue in decode_order.detach().cpu().tolist():
        residue = int(residue)
        if residue in seen:
            continue
        group = next((candidate for candidate in symmetry_groups if residue in candidate), None)
        selected = list(group) if group is not None else [residue]
        ordered.append(selected)
        seen.update(selected)
    return ordered


def _decode_nampnn_sequence(tokens):
    r"""Decode NA-MPNN token ids into a residue-letter string (protein/DNA; RNA renders as its DNA letter)."""
    letters = []
    for token in tokens:
        index = int(token)
        restype = NAMPNN_RESTYPES[index] if 0 <= index < len(NAMPNN_RESTYPES) else "UNK"
        letters.append(NAMPNN_TOKEN_TO_LETTER.get(restype, "X"))
    return "".join(letters)


register_sequence_decoder("nampnn", _decode_nampnn_sequence)
