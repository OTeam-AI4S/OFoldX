r"""
Message-passing graph model that scores and samples protein sequences over backbone and ligand context.

The OFoldX-native reimplementation of LigandMPNN: a message-passing graph model that scores or
autoregressively samples protein sequences conditioned on backbone geometry and nearby ligand/context atoms.
The public surface is [`LigandMPNNDesignGenerator`]
[ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator] and its typed input
[`LigandMPNNInput`][ofoldx.models.ligandmpnn.model.LigandMPNNInput]; the remaining classes are the graph
encoder, autoregressive decoder, and the tensor helpers they share with the side-chain
[`Packer`][ofoldx.models.ligandmpnn.packer.Packer].
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contract import BaseInput, DesignGeneratorOutput
from ..mixin import ModelMixin
from ..registry import DESIGN_GENERATOR_MODELS, MODELS
from .chemical import periodic_group_period, side_chain_atom_types
from .config import LigandMPNNConfig


@MODELS.register("ligandmpnn")
@DESIGN_GENERATOR_MODELS.register("ligandmpnn")
class LigandMPNNDesignGenerator(ModelMixin):
    r"""
    Message-passing model that designs and scores sequences for a protein-ligand complex
    (LigandMPNN inverse folding).

    Runs the LigandMPNN encoder/decoder over a [`LigandMPNNInput`]
    [ofoldx.models.ligandmpnn.model.LigandMPNNInput] batch through three protocols, each returning a
    [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput]: [`score`]
    [ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator.score] (log-likelihood of a given
    sequence), [`single_residue_score`]
    [ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator.single_residue_score] (one masked position
    at a time), and [`sample`][ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator.sample]
    (autoregressive sequence generation).

    Args:
        config: Model configuration; a default
            [`LigandMPNNConfig`][ofoldx.models.ligandmpnn.config.LigandMPNNConfig] is used when ``None``.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("ofoldx/ligandmpnn")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("ofoldx/ligandmpnn")  # doctest: +SKIP
        >>> model.sample(processor("complex.pdb"))  # doctest: +SKIP
    """

    config_class = LigandMPNNConfig

    def __init__(self, config: LigandMPNNConfig | None = None) -> None:
        config = LigandMPNNConfig() if config is None else config
        super().__init__(config)
        self.backbone = LigandMPNNBackbone(config)
        self.post_init()

    def _init_module_weights(self, module: nn.Module) -> None:
        return None

    def reset_parameters(self) -> None:
        r"""
        Re-initialize backbone weights to match the released LigandMPNN scheme.

        Applies Xavier-uniform initialization to every weight matrix (parameters with more than one
        dimension), leaving biases and normalization scales untouched. Called for freshly-created models so
        their initialization matches the reference implementation rather than PyTorch defaults.
        """

        for parameter in self.backbone.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        batch: LigandMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Score the input sequence against the backbone (delegates to [`score`][.score]).

        Provided so the model is callable as a standard module; scoring is the default forward behavior.

        Args:
            batch: Typed LigandMPNN input carrying backbone, sequence, and ligand-context tensors.
            use_sequence: Whether the decoder may condition on the ground-truth sequence context.
            num_samples: Number of decode orders to evaluate; defaults to the value on ``batch``.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] with per-token
            logits, the scored tokens, and per-sequence log-likelihood scores.

        Examples:
            >>> model(batch)  # doctest: +SKIP
        """

        return self.score(batch, use_sequence=use_sequence, num_samples=num_samples)

    def score(
        self,
        batch: LigandMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Score a given sequence with the full-context autoregressive protocol.

        Runs one forward pass of the encoder and decoder over the whole sequence, returning the per-token
        logits and the masked-average sequence log-likelihood. When ``use_sequence`` is ``False`` the decoder
        sees only the encoder context (no teacher-forced tokens), giving a structure-only score.

        Args:
            batch: Typed LigandMPNN input carrying backbone, sequence, and ligand-context tensors.
            use_sequence: Whether the decoder may condition on the ground-truth sequence context.
            num_samples: Number of decode orders to evaluate; defaults to the value on ``batch``.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] whose ``scores``
            hold per-token ``log_probs`` and a ``sequence_log_likelihood``, with LigandMPNN metadata.

        Examples:
            >>> model.score(batch, use_sequence=False)  # doctest: +SKIP
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.score(features, use_sequence=use_sequence)
        log_probs = scored["log_probs"]
        return DesignGeneratorOutput(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={
                "log_probs": log_probs,
                "sequence_log_likelihood": _sequence_log_likelihood(scored["S"], log_probs, features["mask"]),
            },
            metadata={
                "architecture": "ligandmpnn",
                "backend": "native_ligandmpnn",
                "num_edges": self.config.k_neighbors,
                "atom_context_num": self.config.atom_context_num,
                "use_sequence": use_sequence,
            },
        )

    def single_residue_score(
        self,
        batch: LigandMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Score each residue independently with the official single-position protocol.

        Unlike [`score`][ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator.score], which decodes
        the whole sequence in one order, this masks one position at a time (revealing all others when
        ``use_sequence`` is set) and reads off that position's log-probabilities. It reproduces LigandMPNN's
        per-residue "conditional probabilities" mode and is proportionally more expensive (one pass per
        residue).

        Args:
            batch: Typed LigandMPNN input carrying backbone, sequence, and ligand-context tensors.
            use_sequence: Whether the surrounding ground-truth sequence is visible when scoring a position.
            num_samples: Number of decode-order replicas to evaluate; defaults to the value on ``batch``.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] with per-residue
            logits, ``log_probs``, and the aggregate ``sequence_log_likelihood``.

        Examples:
            >>> model.single_residue_score(batch)  # doctest: +SKIP
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.single_residue_score(features, use_sequence=use_sequence)
        log_probs = scored["log_probs"]
        return DesignGeneratorOutput(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={
                "log_probs": log_probs,
                "sequence_log_likelihood": _sequence_log_likelihood(scored["S"], log_probs, features["mask"]),
            },
            metadata={
                "architecture": "ligandmpnn",
                "backend": "native_ligandmpnn",
                "num_edges": self.config.k_neighbors,
                "atom_context_num": self.config.atom_context_num,
                "score_mode": "single_residue",
                "use_sequence": use_sequence,
            },
        )

    @torch.no_grad()
    def generate(
        self,
        batch: LigandMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Sample sequences without tracking gradients (delegates to [`sample`][.sample]).

        The public generation entry point: identical to [`sample`]
        [ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator.sample] but wrapped in
        ``torch.no_grad`` for inference.

        Args:
            batch: Typed LigandMPNN input carrying backbone, sequence, and ligand-context tensors.
            temperature: Softmax temperature for sampling; ``None`` uses the value on ``batch``.
            num_samples: Number of independent sequences to sample; defaults to the value on ``batch``.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] with sampled tokens,
            logits, and per-token log-probabilities and sampling probabilities.

        Examples:
            >>> model.generate(batch, temperature=0.1)  # doctest: +SKIP
        """

        return self.sample(batch, temperature=temperature, num_samples=num_samples, generator=generator)

    def sample(
        self,
        batch: LigandMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Autoregressively sample new sequences for the input backbone.

        Decodes residues in a noise-perturbed order, sampling each token from the temperature-scaled,
        bias-adjusted logits while honoring the design mask (fixed positions keep their input token) and any
        symmetry residue groups (tied positions share one draw). Follows the official single-backbone batch
        contract: the input must carry exactly one backbone, replicated ``num_samples`` times internally.

        Args:
            batch: Typed LigandMPNN input carrying backbone, sequence, and ligand-context tensors.
            temperature: Softmax temperature for sampling; ``None`` uses the value on ``batch``.
            num_samples: Number of independent sequences to sample; defaults to the value on ``batch``.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A [`DesignGeneratorOutput`][ofoldx.models.contract.DesignGeneratorOutput] whose
            ``sequence_tokens`` are the sampled sequences and whose ``scores`` hold ``log_probs`` and
            ``sampling_probs``.

        Examples:
            >>> model.sample(batch, num_samples=8)  # doctest: +SKIP
        """

        features = self._feature_dict(
            batch,
            num_samples=num_samples,
            temperature=temperature,
            generator=generator,
            for_sampling=True,
        )
        sampled = self.backbone.sample(features)
        return DesignGeneratorOutput(
            sequence_logits=sampled["logits"],
            sequence_tokens=sampled["S"],
            scores={
                "log_probs": sampled["log_probs"],
                "sampling_probs": sampled["sampling_probs"],
            },
            metadata={
                "architecture": "ligandmpnn",
                "backend": "native_ligandmpnn",
                "num_edges": self.config.k_neighbors,
                "atom_context_num": self.config.atom_context_num,
            },
        )

    def encode(self, batch: LigandMPNNInput | Mapping[str, Any]):  # type: ignore[override]
        r"""
        Run only the encoder, returning the graph node/edge states.

        Exposes the encoder output (before autoregressive decoding) for downstream consumers such as the
        side-chain Packer or embedding extraction. Accepts either a typed input, which is featurized first,
        or a raw pre-built feature dict.

        Args:
            batch: A [`LigandMPNNInput`][ofoldx.models.ligandmpnn.model.LigandMPNNInput] or an
                already-featurized mapping of encoder inputs.

        Returns:
            Tuple ``(node_states, edge_states, neighbor_idx)`` from the backbone encoder.

        Examples:
            >>> model.encode(batch)  # doctest: +SKIP
        """

        features = self._feature_dict(batch, for_sampling=False) if isinstance(batch, LigandMPNNInput) else batch
        return self.backbone.encode(features)

    def _feature_dict(
        self,
        batch: LigandMPNNInput,
        *,
        num_samples: int | None = None,
        temperature: float | Tensor | None = None,
        generator: torch.Generator | None = None,
        for_sampling: bool,
    ) -> dict[str, Any]:
        r"""
        Assemble the backbone feature dict from a typed input.

        Validates the input, fills in optional fields (design mask, amino-acid bias), and resolves the
        decode-order noise: an explicit ``decoding_order_noise`` is reshaped to the requested sample count,
        otherwise fresh Gaussian noise is drawn for sampling or zeros for deterministic scoring.

        Args:
            batch: Typed LigandMPNN input to featurize.
            num_samples: Requested number of samples / decode orders; ``None`` falls back to the input.
            temperature: Sampling temperature override; ``None`` keeps the input value.
            generator: Optional RNG used when drawing decode-order noise.
            for_sampling: Whether the features are for autoregressive sampling (vs. scoring).

        Returns:
            Feature mapping consumed by the backbone encoder/decoder.

        Raises:
            ValueError: If any input tensor violates the LigandMPNN shape contract, or the supplied
                ``decoding_order_noise`` does not match the requested sample count.
        """

        _validate_ligandmpnn_input(batch, require_sampling=for_sampling)
        batch_count, residue_count = batch.sequence_tokens.shape
        device = batch.sequence_tokens.device
        num_samples_requested = num_samples is not None or batch.num_samples is not None
        num_samples = int(num_samples if num_samples is not None else (batch.num_samples or 1))
        chain_mask = batch.design_mask if batch.design_mask is not None else torch.ones_like(batch.token_mask)
        bias = batch.amino_acid_bias
        if bias is None:
            bias = torch.zeros(batch_count, residue_count, self.config.vocab_size, device=device)
        if batch.decoding_order_noise is not None:
            randn = batch.decoding_order_noise
            if not num_samples_requested:
                num_samples = int(randn.shape[0] if batch_count == 1 else max(randn.shape[0] // batch_count, 1))
            if randn.shape[0] == num_samples:
                if batch_count > 1 and not for_sampling:
                    randn = randn[:, None, :].expand(num_samples, batch_count, residue_count)
                    randn = randn.reshape(num_samples * batch_count, residue_count)
            elif not (not for_sampling and randn.shape[0] == num_samples * batch_count):
                expected = (
                    num_samples if batch_count == 1 or for_sampling else f"{num_samples} or {num_samples * batch_count}"
                )
                raise ValueError(f"decoding_order_noise first dimension must match num_samples ({expected})")
        elif for_sampling:
            randn = torch.randn(num_samples, residue_count, device=device, generator=generator)
        else:
            randn = torch.zeros(num_samples * batch_count, residue_count, device=device)
        features: dict[str, Any] = {
            "X": batch.backbone_coords,
            "S": batch.sequence_tokens,
            "mask": batch.token_mask,
            "residue_index": batch.residue_index,
            "chain_labels": batch.chain_index,
            "Y": batch.ligand_atom_coords,
            "Y_t": batch.ligand_atom_types,
            "Y_m": batch.ligand_atom_mask,
            "chain_mask": chain_mask,
            "bias": bias,
            "randn": randn,
            "generator": generator,
            "temperature": batch.temperature if temperature is None else temperature,
            "batch_size": num_samples,
            "symmetry_residues": batch.symmetry_residue_groups if batch.symmetry_residue_groups is not None else [[]],
            "symmetry_weights": batch.symmetry_weights if batch.symmetry_weights is not None else [[]],
        }
        if batch.all_atom_coords is not None:
            features["xyz_37"] = batch.all_atom_coords
        if batch.all_atom_mask is not None:
            features["xyz_37_m"] = batch.all_atom_mask
        return features


class LigandMPNNBackbone(nn.Module):
    r"""
    LigandMPNN graph encoder and autoregressive sequence decoder.

    Holds the full message-passing stack: a [`LigandBackboneFeatures`]
    [ofoldx.models.ligandmpnn.model.LigandBackboneFeatures] encoder that fuses backbone geometry with
    ligand-context atoms, and encoder/decoder layer stacks operating on the k-nearest-neighbor residue graph.
    The three public methods ([`score`][ofoldx.models.ligandmpnn.model.LigandMPNNBackbone.score],
    [`single_residue_score`][ofoldx.models.ligandmpnn.model.LigandMPNNBackbone.single_residue_score],
    [`sample`][ofoldx.models.ligandmpnn.model.LigandMPNNBackbone.sample]) operate on the feature dict
    built by the enclosing generator and return raw token tensors, not typed outputs.

    Args:
        config: Model configuration supplying the widths, depths, and graph/context sizes.
    """

    def __init__(self, config: LigandMPNNConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.feature_encoder = LigandBackboneFeatures(
            node_size=config.node_size,
            edge_size=config.edge_size,
            num_positional_embeddings=16,
            num_rbf=16,
            k_neighbors=config.k_neighbors,
            coord_noise_scale=config.coord_noise_scale,
            atom_context_num=config.atom_context_num,
            use_side_chain_context=config.use_side_chain_context,
        )
        self.edge_embedding = nn.Linear(config.edge_size, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.atom_context_embedding = nn.Linear(config.node_size, config.hidden_size)
        self.residue_context_input = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_node_embedding = nn.Linear(config.node_size, config.hidden_size)
        self.atom_edge_embedding = nn.Linear(config.node_size, config.hidden_size)
        self.context_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(config.hidden_size)
        self.context_layers = nn.ModuleList(
            [LigandDecoderLayer(config.hidden_size, config.hidden_size * 2, dropout=config.dropout) for _ in range(2)]
        )
        self.atom_context_layers = nn.ModuleList(
            [LigandAtomLayer(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(2)]
        )
        self.encoder_layers = nn.ModuleList(
            [
                LigandEncoderLayer(config.hidden_size, config.hidden_size * 2, dropout=config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                LigandDecoderLayer(config.hidden_size, config.hidden_size * 3, dropout=config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def encode(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Encode backbone geometry and ligand context into residue node/edge states.

        Builds the k-nearest-neighbor graph, runs the encoder layers, then folds in the ligand-context
        atom graph so each residue node carries information about nearby ligand atoms.

        Args:
            features: Feature mapping from the generator (backbone coords, masks, ligand context, ...).

        Returns:
            Tuple ``(node_states, edge_states, neighbor_idx)``: per-residue node states `[B, L, H]`, per-edge
            states `[B, L, K, H]`, and neighbor indices `[B, L, K]`.
        """

        atom_features, edges, neighbor_idx, atom_nodes, atom_edges, atom_mask = self.feature_encoder(features)
        mask = features["mask"]
        node_states = edges.new_zeros(edges.shape[0], edges.shape[1], self.hidden_size)
        edge_states = self.edge_embedding(edges)
        atom_context = self.atom_context_embedding(atom_features)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_idx).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(node_states, edge_states, neighbor_idx, mask, attend_mask)
        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for layer_idx, layer in enumerate(self.context_layers):
            atom_nodes = self.atom_context_layers[layer_idx](atom_nodes, atom_edges, atom_mask, atom_pair_mask)
            residue_context = layer(residue_context, torch.cat((atom_context, atom_nodes), dim=-1), mask, atom_mask)
        node_states = node_states + self.context_norm(self.dropout(self.context_output(residue_context)))
        return node_states, edge_states, neighbor_idx

    def score(self, features: Mapping[str, Any], *, use_sequence: bool) -> dict[str, Tensor]:
        r"""
        Score a sequence in one decode pass over the whole graph.

        Encodes the backbone once, derives a decode order from the chain mask and noise, builds the causal
        backward/forward attention masks, and runs the decoder. With ``use_sequence`` the decoder mixes
        teacher-forced token embeddings into the backward context; without it, only the encoder forward
        context is used. Symmetry groups, when present, collapse tied residues into a single decode step.

        Args:
            features: Feature mapping from the generator.
            use_sequence: Whether the decoder conditions on the ground-truth sequence context.

        Returns:
            Mapping with ``S`` (the scored tokens, replicated per sample), ``logits`` `[B*, L, vocab]`, and
            ``log_probs``.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"]
        randn = features["randn"]
        num_samples = int(features["batch_size"])
        batch, length = sequence.shape
        device = sequence.device

        node_states, edge_states, neighbor_idx = self.encode(features)
        chain_mask_order = chain_mask if randn.shape[0] == chain_mask.shape[0] else chain_mask.repeat(num_samples, 1)
        decode_order = torch.argsort((chain_mask_order + 0.0001) * randn.abs())
        symmetry_groups = features["symmetry_residues"]

        if _has_no_symmetry(symmetry_groups):
            neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
            mask_decode = mask.repeat(num_samples, 1)
            backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask_decode)
        else:
            ordered_groups = _decode_groups(decode_order[0], symmetry_groups)
            decode_order = torch.tensor(list(itertools.chain(*ordered_groups)), device=device)[None, :].repeat(batch, 1)
            backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
            neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
            backward_mask = backward_mask.repeat(num_samples, 1, 1, 1)
            forward_mask = forward_mask.repeat(num_samples, 1, 1, 1)
            decode_order = decode_order.repeat(num_samples, 1)

        sequence = sequence.repeat(num_samples, 1)
        node_states = node_states.repeat(num_samples, 1, 1)
        edge_states = edge_states.repeat(num_samples, 1, 1, 1)
        mask = mask.repeat(num_samples, 1)

        token_states = self.token_embedding(sequence)
        edge_token_states = cat_neighbors_nodes(token_states, edge_states, neighbor_idx)
        empty_token_states = torch.zeros_like(token_states)
        encoder_edges = cat_neighbors_nodes(empty_token_states, edge_states, neighbor_idx)
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
        }

    def single_residue_score(self, features: Mapping[str, Any], *, use_sequence: bool) -> dict[str, Tensor]:
        r"""
        Score each residue independently, one masked position per decode pass.

        Loops over positions: for each, builds a decode order that isolates that residue (masked when
        ``use_sequence``, or the only visible one otherwise), runs the decoder, and records that position's
        logits and log-probabilities. Reproduces LigandMPNN's per-residue conditional-probability mode.

        Args:
            features: Feature mapping from the generator.
            use_sequence: Whether the surrounding ground-truth sequence is visible when scoring a position.

        Returns:
            Mapping with ``S`` (replicated tokens), ``logits`` `[B*, L, 21]`, and ``log_probs`` `[B*, L, 21]`.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = features["chain_mask"]
        randn = features["randn"]
        num_samples = int(features["batch_size"])
        batch, length = sequence.shape
        device = sequence.device
        node_states_ref, edge_states_ref, neighbor_idx_ref = self.encode(features)
        total_decodes = batch * num_samples
        log_probs_out = torch.zeros(total_decodes, length, 21, device=device)
        logits_out = torch.zeros(total_decodes, length, 21, device=device)

        for residue_position in range(length):
            neighbor_idx = neighbor_idx_ref.repeat(num_samples, 1, 1)
            if use_sequence:
                order_mask = torch.ones(chain_mask.shape[1], device=device)
                order_mask[residue_position] = 0.0
            else:
                order_mask = torch.zeros(chain_mask.shape[1], device=device)
                order_mask[residue_position] = 1.0
            decode_order = torch.argsort((order_mask + 0.0001) * randn.abs())
            mask_decode = mask.repeat(num_samples, 1)
            backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask_decode)

            sequence_decode = sequence.repeat(num_samples, 1)
            node_states = node_states_ref.repeat(num_samples, 1, 1)
            edge_states = edge_states_ref.repeat(num_samples, 1, 1, 1)
            token_states = self.token_embedding(sequence_decode)
            edge_token_states = cat_neighbors_nodes(token_states, edge_states, neighbor_idx)
            encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
            encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
            forward_context = forward_mask * encoder_context
            for layer in self.decoder_layers:
                decoder_context = cat_neighbors_nodes(node_states, edge_token_states, neighbor_idx)
                node_states = layer(node_states, backward_mask * decoder_context + forward_context, mask_decode)
            logits = self.output_projection(node_states)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_out[:, residue_position, :] = log_probs[:, residue_position, :]
            logits_out[:, residue_position, :] = logits[:, residue_position, :]

        return {
            "S": sequence.repeat(num_samples, 1),
            "logits": logits_out,
            "log_probs": log_probs_out,
        }

    def sample(self, features: Mapping[str, Any]) -> dict[str, Tensor]:
        r"""
        Autoregressively sample tokens along a noise-perturbed decode order.

        Encodes the backbone, derives the decode order, then dispatches to the independent-position sampler
        or, when symmetry groups are present, the symmetric sampler that ties grouped residues to a single
        weighted draw.

        Args:
            features: Feature mapping from the generator.

        Returns:
            Mapping with ``S`` (sampled tokens), ``sampling_probs``, ``logits``, and ``log_probs``.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"]
        bias = features["bias"]
        randn = features["randn"]
        temperature = features["temperature"]
        symmetry_groups = features["symmetry_residues"]
        symmetry_weights_groups = features["symmetry_weights"]
        generator = features.get("generator")
        num_samples = int(features["batch_size"])
        batch, length = sequence.shape
        device = sequence.device

        node_states, edge_states, neighbor_idx = self.encode(features)
        decode_order = torch.argsort((chain_mask + 0.0001) * randn.abs())
        if _has_no_symmetry(symmetry_groups):
            return self._sample_independent(
                sequence,
                mask,
                chain_mask,
                bias,
                temperature,
                decode_order,
                node_states,
                edge_states,
                neighbor_idx,
                num_samples,
                generator,
            )

        ordered_groups = _decode_groups(decode_order[0], symmetry_groups)
        symmetry_weights = torch.ones(length, device=device)
        for group_idx, residue_group in enumerate(symmetry_groups):
            for residue_position, residue in enumerate(residue_group):
                symmetry_weights[residue] = symmetry_weights_groups[group_idx][residue_position]
        decode_order = torch.tensor(list(itertools.chain(*ordered_groups)), device=device)[None, :].repeat(batch, 1)
        return self._sample_symmetric(
            sequence,
            mask,
            chain_mask,
            bias,
            temperature,
            decode_order,
            ordered_groups,
            symmetry_weights,
            node_states,
            edge_states,
            neighbor_idx,
            num_samples,
            generator,
        )

    def _init_decode_state(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
    ) -> dict[str, Any]:
        r"""
        Build the order-independent decode setup shared by the sampling paths.
        """

        _, length = sequence.shape
        device = sequence.device
        neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
        node_states = node_states.repeat(num_samples, 1, 1)
        return {
            "sequence_true": sequence.repeat(num_samples, 1),
            "node_states": node_states,
            "edge_states": edge_states.repeat(num_samples, 1, 1, 1),
            "neighbor_idx": neighbor_idx,
            "chain_mask": chain_mask.repeat(num_samples, 1),
            "mask": mask.repeat(num_samples, 1),
            "bias": bias.repeat(num_samples, 1, 1),
            "sampled": 20 * torch.ones(num_samples, length, dtype=torch.long, device=device),
            "token_states": torch.zeros_like(node_states),
            "layer_states": [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_layers],
            "probabilities": torch.zeros(num_samples, length, 20, device=device),
            "log_probs_out": torch.zeros(num_samples, length, 21, device=device),
            "logits_out": torch.zeros(num_samples, length, 21, device=device),
        }

    def _sample_independent(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        temperature: float | Tensor,
        decode_order: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Sample tokens position-by-position with no symmetry constraints.

        Walks the decode order one residue at a time, gathering that residue's cached decoder state, sampling
        a token from the temperature-scaled bias-adjusted probabilities, and scattering the new token
        embedding back so later positions can attend to it. Fixed positions (chain mask ``0``) keep their
        input token.

        Args:
            sequence: Input sequence tokens `[1, L]`.
            mask: Valid-residue mask `[1, L]`.
            chain_mask: Design mask marking positions to sample `[1, L]`.
            bias: Per-position amino-acid logit bias `[1, L, vocab]`.
            temperature: Softmax sampling temperature.
            decode_order: Residue decode order `[num_samples, L]`.
            node_states: Encoder node states `[1, L, H]`.
            edge_states: Encoder edge states `[1, L, K, H]`.
            neighbor_idx: Neighbor indices `[1, L, K]`.
            num_samples: Number of independent sequences to sample.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Mapping with ``S`` (sampled tokens), ``sampling_probs``, ``logits``, and ``log_probs``.
        """

        state = self._init_decode_state(
            sequence, mask, chain_mask, bias, node_states, edge_states, neighbor_idx, num_samples
        )
        sequence_true = state["sequence_true"]
        node_states = state["node_states"]
        edge_states = state["edge_states"]
        neighbor_idx = state["neighbor_idx"]
        chain_mask = state["chain_mask"]
        mask = state["mask"]
        bias = state["bias"]
        sampled = state["sampled"]
        token_states = state["token_states"]
        layer_states = state["layer_states"]
        probabilities = state["probabilities"]
        log_probs_out = state["log_probs_out"]
        logits_out = state["logits_out"]
        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        length = sequence.shape[1]
        for step in range(length):
            residue = decode_order[:, step]
            chain_mask_t = torch.gather(chain_mask, 1, residue[:, None])[:, 0]
            mask_t = torch.gather(mask, 1, residue[:, None])[:, 0]
            bias_t = torch.gather(bias, 1, residue[:, None, None].repeat(1, 1, bias.shape[-1]))[:, 0, :]
            neighbor_idx_t = torch.gather(neighbor_idx, 1, residue[:, None, None].repeat(1, 1, neighbor_idx.shape[-1]))
            edge_t = torch.gather(
                edge_states,
                1,
                residue[:, None, None, None].repeat(1, 1, edge_states.shape[-2], edge_states.shape[-1]),
            )
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
            for layer_idx, layer in enumerate(self.decoder_layers):
                decoder_context = cat_neighbors_nodes(layer_states[layer_idx], edge_token_t, neighbor_idx_t)
                node_t = torch.gather(
                    layer_states[layer_idx],
                    1,
                    residue[:, None, None].repeat(1, 1, layer_states[layer_idx].shape[-1]),
                )
                updated = layer(node_t, backward_t * decoder_context + forward_t, mask_t)
                layer_states[layer_idx + 1].scatter_(
                    1,
                    residue[:, None, None].repeat(1, 1, node_states.shape[-1]),
                    updated,
                )

            final_t = torch.gather(
                layer_states[-1],
                1,
                residue[:, None, None].repeat(1, 1, layer_states[-1].shape[-1]),
            )[:, 0]
            logits = self.output_projection(final_t)
            log_probs = F.log_softmax(logits, dim=-1)
            sample_probs = F.softmax((logits + bias_t) / temperature, dim=-1)
            sample_probs = sample_probs[:, :20] / sample_probs[:, :20].sum(dim=-1, keepdim=True)
            sampled_t = torch.multinomial(sample_probs, 1, generator=generator)[:, 0]
            probabilities.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, 20),
                chain_mask_t[:, None, None] * sample_probs[:, None, :],
            )
            log_probs_out.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, 21),
                chain_mask_t[:, None, None] * log_probs[:, None, :],
            )
            logits_out.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, 21),
                chain_mask_t[:, None, None] * logits[:, None, :],
            )
            true_t = torch.gather(sequence_true, 1, residue[:, None])[:, 0]
            sampled_t = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
            token_states.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, token_states.shape[-1]),
                self.token_embedding(sampled_t)[:, None, :],
            )
            sampled.scatter_(1, residue[:, None], sampled_t[:, None])

        return {
            "S": sampled,
            "sampling_probs": probabilities,
            "logits": logits_out,
            "log_probs": log_probs_out,
        }

    def _sample_symmetric(
        self,
        sequence: Tensor,
        mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        temperature: float | Tensor,
        decode_order: Tensor,
        ordered_groups: list[list[int]],
        symmetry_weights: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_idx: Tensor,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Sample tokens with tied symmetry groups drawn as one weighted decision.

        Like [`_sample_independent`][ofoldx.models.ligandmpnn.model.LigandMPNNBackbone._sample_independent]
        but decodes residues group-by-group: within a group the per-residue logits are combined with the
        symmetry weights, a single token is sampled, and every group member is assigned that token so tied
        positions stay identical.

        Args:
            sequence: Input sequence tokens `[1, L]`.
            mask: Valid-residue mask `[1, L]`.
            chain_mask: Design mask marking positions to sample `[1, L]`.
            bias: Per-position amino-acid logit bias `[1, L, vocab]`.
            temperature: Softmax sampling temperature.
            decode_order: Residue decode order `[1, L]` derived from the ordered groups.
            ordered_groups: Symmetry groups in decode order, each a list of residue indices.
            symmetry_weights: Per-residue logit weights `[L]` combining tied positions.
            node_states: Encoder node states `[1, L, H]`.
            edge_states: Encoder edge states `[1, L, K, H]`.
            neighbor_idx: Neighbor indices `[1, L, K]`.
            num_samples: Number of independent sequences to sample.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Mapping with ``S`` (sampled tokens), ``sampling_probs``, ``logits``, and ``log_probs``.
        """

        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        backward_mask = backward_mask.repeat(num_samples, 1, 1, 1)
        forward_mask = forward_mask.repeat(num_samples, 1, 1, 1)
        state = self._init_decode_state(
            sequence, mask, chain_mask, bias, node_states, edge_states, neighbor_idx, num_samples
        )
        node_states = state["node_states"]
        edge_states = state["edge_states"]
        neighbor_idx = state["neighbor_idx"]
        chain_mask = state["chain_mask"]
        mask = state["mask"]
        bias = state["bias"]
        sequence_true = state["sequence_true"]
        sampled = state["sampled"]
        token_states = state["token_states"]
        layer_states = state["layer_states"]
        probabilities = state["probabilities"]
        log_probs_out = state["log_probs_out"]
        logits_out = state["logits_out"]
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        for residue_group in ordered_groups:
            total_logits = 0.0
            bias_t = bias[:, residue_group[0]]
            for residue in residue_group:
                chain_mask_t = chain_mask[:, residue]
                mask_t = mask[:, residue]
                edge_t = edge_states[:, residue : residue + 1]
                neighbor_idx_t = neighbor_idx[:, residue : residue + 1]
                edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
                forward_t = forward_context[:, residue : residue + 1]
                for layer_idx, layer in enumerate(self.decoder_layers):
                    decoder_context = cat_neighbors_nodes(layer_states[layer_idx], edge_token_t, neighbor_idx_t)
                    node_t = layer_states[layer_idx][:, residue : residue + 1]
                    context_t = backward_mask[:, residue : residue + 1] * decoder_context + forward_t
                    layer_states[layer_idx + 1][:, residue : residue + 1] = layer(
                        node_t,
                        context_t,
                        mask_t[:, None],
                    )
                logits = self.output_projection(layer_states[-1][:, residue])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_out[:, residue] = chain_mask_t[:, None] * log_probs
                logits_out[:, residue] = chain_mask_t[:, None] * logits
                total_logits = total_logits + symmetry_weights[residue] * logits

            sample_probs = F.softmax((total_logits + bias_t) / temperature, dim=-1)
            sample_probs = sample_probs[:, :20] / sample_probs[:, :20].sum(dim=-1, keepdim=True)
            sampled_t = torch.multinomial(sample_probs, 1, generator=generator)[:, 0]
            for residue in residue_group:
                chain_mask_t = chain_mask[:, residue]
                probabilities[:, residue] = chain_mask_t[:, None] * sample_probs
                true_t = sequence_true[:, residue]
                sampled_t = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
                token_states[:, residue] = self.token_embedding(sampled_t)
                sampled[:, residue] = sampled_t

        return {
            "S": sampled,
            "sampling_probs": probabilities,
            "logits": logits_out,
            "log_probs": log_probs_out,
        }


class LigandBackboneFeatures(nn.Module):
    r"""
    Input featurizer for the LigandMPNN backbone graph and ligand context.

    Turns raw backbone coordinates into the graph the encoder consumes: it derives the virtual CB atom,
    builds the k-nearest-neighbor residue graph, and computes pairwise RBF distance edges plus
    relative-position features. In parallel it featurizes the ligand/context atoms (per-atom RBF distances to
    backbone atoms, element one-hots, and local-frame angle features) and, when enabled, folds observed
    side-chain atoms into that context.

    Args:
        node_size: Width of the per-atom context node features.
        edge_size: Width of the per-residue-edge features.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
        k_neighbors: Number of nearest-neighbor residues in the graph.
        coord_noise_scale: Standard deviation of Gaussian coordinate noise; ``0`` disables it.
        atom_context_num: Number of ligand/context atoms kept per residue.
        use_side_chain_context: Whether to append observed side-chain atoms to the ligand context.
    """

    def __init__(
        self,
        *,
        node_size: int,
        edge_size: int,
        num_positional_embeddings: int,
        num_rbf: int,
        k_neighbors: int,
        coord_noise_scale: float,
        atom_context_num: int,
        use_side_chain_context: bool,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.coord_noise_scale = coord_noise_scale
        self.num_rbf = num_rbf
        self.atom_context_num = atom_context_num
        self.use_side_chain_context = use_side_chain_context
        self.relative_position = RelativePositionEncoding(num_positional_embeddings)
        self.edge_projection = nn.Linear(num_positional_embeddings + 25 * num_rbf, edge_size, bias=False)
        self.edge_norm = nn.LayerNorm(edge_size)
        self.atom_projection = nn.Linear(5 * num_rbf + 64 + 4, node_size)
        self.atom_norm = nn.LayerNorm(node_size)
        self.element_projection = nn.Linear(147, 64)
        self.context_node_projection = nn.Linear(147, node_size, bias=False)
        self.context_edge_projection = nn.Linear(num_rbf, node_size, bias=False)
        self.context_edge_norm = nn.LayerNorm(node_size)
        self.context_node_norm = nn.LayerNorm(node_size)
        group, period = periodic_group_period()
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)
        self.register_buffer("side_chain_atom_types", side_chain_atom_types(), persistent=False)

    def forward(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Featurize backbone geometry and ligand context into graph inputs.

        Args:
            features: Feature mapping with backbone coords ``X``, ligand context ``Y``/``Y_t``/``Y_m``, the
                residue/chain indices, and the validity ``mask``.

        Returns:
            Tuple ``(atom_features, edges, neighbor_idx, context_nodes, context_edges, context_mask)``:
            per-residue ligand-context node features, residue-graph edges, neighbor indices, and the
            context-atom node/edge features and mask.
        """

        coords = features["X"]
        context_coords = features["Y"]
        context_mask = features["Y_m"]
        context_types = features["Y_t"]
        if self.coord_noise_scale > 0:
            generator = features.get("generator")
            coords = coords + self.coord_noise_scale * _randn_like(coords, generator)
            context_coords = context_coords + self.coord_noise_scale * _randn_like(context_coords, generator)
        atoms = _backbone_atoms(coords)
        _, neighbor_idx = _knn(atoms["CA"], features["mask"], self.k_neighbors)
        pair_features = _backbone_pair_rbf(atoms, neighbor_idx, self.num_rbf)
        residue_offset = features["residue_index"][:, :, None] - features["residue_index"][:, None, :]
        residue_offset = gather_edges(residue_offset[:, :, :, None], neighbor_idx)[:, :, :, 0]
        same_chain = (features["chain_labels"][:, :, None] == features["chain_labels"][:, None, :]).long()
        same_chain = gather_edges(same_chain[:, :, :, None], neighbor_idx)[:, :, :, 0]
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_input = torch.cat((position_features, pair_features), dim=-1)
        edges = self.edge_norm(self.edge_projection(edge_input))

        if self.use_side_chain_context:
            context_coords, context_types, context_mask = self._append_side_chain_context(
                features,
                atoms["CB"],
                context_coords,
                context_types,
                context_mask,
                neighbor_idx,
            )

        context_types = context_types.long()
        element_features = self._element_features(context_types)
        element_projection = self.element_projection(element_features.to(dtype=coords.dtype))

        def atom_context_rbf(atom: Tensor) -> Tensor:
            r"""
            RBF-expand the distances from one backbone atom to every ligand-context atom.

            Args:
                atom: Coordinates of a single backbone atom per residue `[B, L, 3]`.

            Returns:
                Radial-basis distance features to each context atom `[B, L, A, num_rbf]`.
            """

            distance = torch.sqrt(torch.sum((atom[:, :, None] - context_coords) ** 2, dim=-1) + 1e-6)
            return _rbf(distance, self.num_rbf)

        atom_features = torch.cat(
            (
                atom_context_rbf(atoms["N"]),
                atom_context_rbf(atoms["CA"]),
                atom_context_rbf(atoms["C"]),
                atom_context_rbf(atoms["O"]),
                atom_context_rbf(atoms["CB"]),
                element_projection,
                _local_angle_features(atoms["N"], atoms["CA"], atoms["C"], context_coords),
            ),
            dim=-1,
        )
        atom_features = self.atom_norm(self.atom_projection(atom_features))
        context_distances = torch.sqrt(
            torch.sum((context_coords[:, :, :, None] - context_coords[:, :, None, :]) ** 2, dim=-1) + 1e-6
        )
        context_edges = self.context_edge_norm(self.context_edge_projection(_rbf(context_distances, self.num_rbf)))
        context_nodes = self.context_node_norm(self.context_node_projection(element_features.to(dtype=coords.dtype)))
        return atom_features, edges, neighbor_idx, context_nodes, context_edges, context_mask

    def _append_side_chain_context(
        self,
        features: Mapping[str, Tensor],
        cb: Tensor,
        context_coords: Tensor,
        context_types: Tensor,
        context_mask: Tensor,
        neighbor_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Append observed neighbor side-chain atoms to the ligand context, then re-select the nearest atoms.

        Args:
            features: Feature mapping; must include atom37 side-chain coords ``xyz_37`` and mask ``xyz_37_m``.
            cb: Virtual CB coordinates `[B, L, 3]` used as the distance reference.
            context_coords: Existing ligand-context coordinates `[B, L, A, 3]`.
            context_types: Existing context atomic numbers `[B, L, A]`.
            context_mask: Existing context validity mask `[B, L, A]`.
            neighbor_idx: Residue neighbor indices `[B, L, K]`.

        Returns:
            The context ``(coords, types, mask)`` after adding side-chain atoms and keeping the nearest
            ``atom_context_num`` per residue.

        Raises:
            ValueError: If ``xyz_37``/``xyz_37_m`` are absent from ``features``.
        """

        if "xyz_37" not in features or "xyz_37_m" not in features:
            raise ValueError("LigandMPNN side-chain context requires xyz_37 and xyz_37_m")
        xyz_37 = features["xyz_37"]
        xyz_37_m = features["xyz_37_m"] * (1 - features["chain_mask"][:, :, None])
        batch, length = context_mask.shape[:2]
        neighbor_subset = neighbor_idx[:, :, :16]
        side_chain_mask = gather_nodes(xyz_37_m[:, :, 5:], neighbor_subset).reshape(batch, length, -1)
        side_chain_coords = gather_nodes(xyz_37[:, :, 5:].reshape(batch, length, -1), neighbor_subset)
        side_chain_coords = side_chain_coords.reshape(batch, length, -1, 3)
        side_chain_types = self.side_chain_atom_types[None, None, None, :].expand(
            batch,
            length,
            neighbor_subset.shape[2],
            -1,
        )
        side_chain_types = side_chain_types.reshape(batch, length, -1)
        context_coords = torch.cat((side_chain_coords, context_coords), dim=2)
        context_mask = torch.cat((side_chain_mask, context_mask), dim=2)
        context_types = torch.cat((side_chain_types, context_types), dim=2)
        distances = torch.sum((cb[:, :, None] - context_coords) ** 2, dim=-1)
        valid = features["mask"][:, :, None] * context_mask
        adjusted = distances * valid + (1.0 - valid) * 10000.0
        atom_count = min(self.atom_context_num, adjusted.shape[-1])
        atom_idx = torch.topk(adjusted, atom_count, dim=-1, largest=False).indices
        context_coords = torch.gather(context_coords, 2, atom_idx[:, :, :, None].expand(-1, -1, -1, 3))
        context_types = torch.gather(context_types, 2, atom_idx)
        context_mask = torch.gather(context_mask, 2, atom_idx)
        return context_coords, context_types, context_mask

    def _element_features(self, atomic_numbers: Tensor) -> Tensor:
        r"""One-hot-encode each context atom by element, periodic group, and period.

        Args:
            atomic_numbers: Context atom atomic numbers `[B, L, A]`.

        Returns:
            Concatenated one-hot element features `[B, L, A, 147]` (120 element + 19 group + 8 period).

        Raises:
            ValueError: If any atomic number is outside ``[0, 119]``.
        """

        if atomic_numbers.min() < 0 or atomic_numbers.max() >= self.periodic_group.shape[0]:
            raise ValueError("ligand_atom_types atomic numbers must be in [0, 119)")
        group = self.periodic_group[atomic_numbers]
        period = self.periodic_period[atomic_numbers]
        return torch.cat(
            (
                F.one_hot(atomic_numbers, 120),
                F.one_hot(group, 19),
                F.one_hot(period, 8),
            ),
            dim=-1,
        )


class LigandEncoderLayer(nn.Module):
    r"""
    One encoder message-passing layer updating both node and edge states.

    Aggregates neighbor messages into each node (with a feed-forward residual), then recomputes edge
    messages from the updated nodes. Messages are summed over neighbors and divided by ``scale`` to keep the
    residual magnitude stable, matching the reference LigandMPNN encoder block.

    Args:
        hidden_size: Node/edge feature width.
        edge_input_size: Width of the concatenated edge context feeding the message MLPs.
        dropout: Dropout probability on the message and feed-forward residuals.
        scale: Divisor applied to the summed neighbor messages.
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
        Update node states from neighbor messages, then refresh edge states.

        Args:
            node_states: Per-residue node states `[B, L, H]`.
            edge_states: Per-edge states `[B, L, K, H]`.
            neighbor_idx: Neighbor indices `[B, L, K]`.
            node_mask: Optional per-node validity mask `[B, L]` zeroing invalid nodes.
            attend_mask: Optional per-edge attention mask `[B, L, K]` zeroing invalid messages.

        Returns:
            Tuple ``(node_states, edge_states)`` with the updated node and edge states.
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


class LigandDecoderLayer(nn.Module):
    r"""
    One node-only message-passing layer over a precomputed edge context.

    Aggregates messages from a supplied ``edge_context`` (already carrying neighbor node and edge state) into
    each node, followed by a feed-forward residual. Unlike the encoder layer it does not update edge states;
    it is reused for the autoregressive decoder and the ligand-context mixing stack.

    Args:
        hidden_size: Node feature width.
        edge_input_size: Width of the supplied edge context feeding the message MLP.
        dropout: Dropout probability on the message and feed-forward residuals.
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
        Update node states from messages over the supplied edge context.

        Args:
            node_states: Per-residue node states `[B, L, H]`.
            edge_context: Concatenated neighbor context `[B, L, K, edge_input_size]`.
            node_mask: Optional per-node validity mask zeroing invalid nodes.
            attend_mask: Optional per-edge attention mask zeroing invalid messages.

        Returns:
            The updated node states `[B, L, H]`.
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


class LigandAtomLayer(nn.Module):
    r"""
    Message-passing layer over the per-residue ligand-atom graph.

    Structurally the decoder layer applied one axis deeper: it mixes ligand/context atom nodes among
    themselves within each residue (an extra ``L`` axis) before those context nodes are folded into the
    residue graph. Nodes are updated from an all-pairs atom edge context with a feed-forward residual.

    Args:
        hidden_size: Atom-node feature width.
        edge_input_size: Width of the supplied atom-edge context feeding the message MLP.
        dropout: Dropout probability on the message and feed-forward residuals.
        scale: Divisor applied to the summed atom messages.
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
        Update ligand-atom node states from all-pairs atom messages.

        Args:
            node_states: Per-residue atom-node states `[B, L, A, H]`.
            edge_context: Concatenated atom-pair context `[B, L, A, A, edge_input_size]`.
            node_mask: Optional per-atom validity mask zeroing invalid atoms.
            attend_mask: Optional per-atom-pair mask zeroing invalid messages.

        Returns:
            The updated atom-node states `[B, L, A, H]`.
        """

        source_states = node_states.unsqueeze(-2).expand(-1, -1, -1, edge_context.shape[-2], -1)
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
    Three-layer GELU MLP that computes a message from concatenated node/edge features.

    Args:
        input_size: Width of the concatenated source-node, neighbor-node, and edge input.
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
        Compute the message for ``x``.

        Args:
            x: Concatenated message input `[..., input_size]`.

        Returns:
            The message `[..., hidden_size]`.
        """

        return self.output(self.activation(self.middle(self.activation(self.input(x)))))


class FeedForward(nn.Module):
    r"""
    Position-wise GELU feed-forward block used as the layer residual.

    Args:
        hidden_size: Input/output width.
        intermediate_size: Width of the expanded inner layer.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Apply the expand-activate-project feed-forward transform.

        Args:
            x: Input `[..., hidden_size]`.

        Returns:
            The transformed tensor `[..., hidden_size]`.
        """

        return self.output(self.activation(self.input(x)))


class RelativePositionEncoding(nn.Module):
    r"""
    Relative sequence-position edge embedding with a distinct cross-chain bucket.

    Clips the residue-index offset to ``[-max_relative_feature, max_relative_feature]`` into one-hot buckets;
    edges between different chains are routed to a single extra bucket so cross-chain contacts are encoded as
    "far / different chain" rather than by their (meaningless) index difference.

    Args:
        num_embeddings: Output embedding width.
        max_relative_feature: Maximum absolute residue offset kept before clipping.
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""
        Embed the per-edge relative position.

        Args:
            offset: Signed residue-index offsets `[B, L, K]`.
            same_chain: Binary same-chain indicator `[B, L, K]` (``1`` if the pair shares a chain).

        Returns:
            The relative-position edge embedding `[B, L, K, num_embeddings]`.
        """

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        return self.linear(F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.linear.weight.dtype))


def _randn_like(tensor: Tensor, generator: torch.Generator | None) -> Tensor:
    r"""Draw standard-normal noise shaped like ``tensor``, honoring an optional RNG.

    Args:
        tensor: Template tensor providing shape/dtype/device.
        generator: Optional RNG; when ``None`` uses the global default.

    Returns:
        Gaussian noise with the same shape, dtype, and device as ``tensor``.
    """

    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)


def _validate_ligandmpnn_input(batch: LigandMPNNInput, *, require_sampling: bool) -> None:
    r"""Assert that a typed input satisfies the LigandMPNN tensor shape contract.

    Args:
        batch: The input to validate.
        require_sampling: When ``True``, additionally enforce the single-backbone (``B == 1``) contract.

    Raises:
        ValueError: If any field has an inconsistent shape, or ``require_sampling`` and ``B != 1``.
    """

    if batch.backbone_coords.ndim != 4 or batch.backbone_coords.shape[2:] != (4, 3):
        raise ValueError(
            "backbone_coords must have shape (B, L, 4, 3) for N/CA/C/O atoms, "
            f"got {tuple(batch.backbone_coords.shape)}"
        )
    if batch.sequence_tokens.shape != batch.token_mask.shape:
        raise ValueError("sequence_tokens and token_mask must have matching shape (B, L)")
    if batch.backbone_coords.shape[:2] != batch.sequence_tokens.shape:
        raise ValueError("backbone_coords and sequence_tokens/token_mask must share batch and residue axes")
    if (
        batch.ligand_atom_coords.ndim != 4
        or batch.ligand_atom_coords.shape[:2] != batch.sequence_tokens.shape
        or batch.ligand_atom_coords.shape[-1] != 3
    ):
        raise ValueError("ligand_atom_coords must have shape (B, L, M, 3)")
    if (
        batch.ligand_atom_types.shape != batch.ligand_atom_mask.shape
        or batch.ligand_atom_types.shape != batch.ligand_atom_coords.shape[:3]
    ):
        raise ValueError("ligand_atom_types and ligand_atom_mask must have shape (B, L, M)")
    if (
        batch.residue_index.shape != batch.sequence_tokens.shape
        or batch.chain_index.shape != batch.sequence_tokens.shape
    ):
        raise ValueError("residue_index and chain_index must have shape (B, L)")
    if batch.design_mask is not None and batch.design_mask.shape != batch.sequence_tokens.shape:
        raise ValueError("design_mask must have shape (B, L)")
    if batch.amino_acid_bias is not None and batch.amino_acid_bias.shape != (*batch.sequence_tokens.shape, 21):
        raise ValueError("amino_acid_bias must have shape (B, L, 21)")
    if batch.decoding_order_noise is not None and (
        batch.decoding_order_noise.ndim != 2 or batch.decoding_order_noise.shape[1] != batch.sequence_tokens.shape[1]
    ):
        raise ValueError("decoding_order_noise must have shape (num_samples, L)")
    if require_sampling and batch.sequence_tokens.shape[0] != 1:
        raise ValueError("LigandMPNN sampling follows the official single-backbone batch contract")
    if batch.all_atom_coords is not None and (
        batch.all_atom_coords.ndim != 4 or batch.all_atom_coords.shape[:2] != batch.sequence_tokens.shape
    ):
        raise ValueError("all_atom_coords must have shape (B, L, A, 3)")
    if batch.all_atom_mask is not None and (
        batch.all_atom_mask.ndim != 3 or batch.all_atom_mask.shape[:2] != batch.sequence_tokens.shape
    ):
        raise ValueError("all_atom_mask must have shape (B, L, A)")


def _sequence_log_likelihood(sequence: Tensor, log_probs: Tensor, mask: Tensor) -> Tensor:
    r"""Mask-averaged per-token log-likelihood of ``sequence`` under ``log_probs``.

    Args:
        sequence: Token ids `[B*, L]`.
        log_probs: Per-token log-probabilities `[B*, L, vocab]`.
        mask: Valid-residue mask `[B, L]`; tiled to the sample count when needed.

    Returns:
        Per-sequence mean log-likelihood over valid residues `[B*]`.
    """

    token_log_probs = torch.gather(log_probs, -1, sequence[..., None]).squeeze(-1)
    mask = mask.to(dtype=token_log_probs.dtype)
    if mask.shape[0] != token_log_probs.shape[0]:
        mask = mask.repeat(token_log_probs.shape[0] // mask.shape[0], 1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _has_no_symmetry(groups: list[list[int]]) -> bool:
    r"""Return whether ``groups`` is the sentinel meaning "no symmetry constraints" (``[[]]``)."""

    return len(groups) == 1 and len(groups[0]) == 0


def _decode_groups(decode_order: Tensor, symmetry_groups: list[list[int]]) -> list[list[int]]:
    r"""Order the symmetry groups by the first time any member appears in the decode order.

    Args:
        decode_order: A 1-D residue decode order.
        symmetry_groups: Groups of tied residue indices.

    Returns:
        The symmetry groups (and singletons for ungrouped residues) in decode order, each residue once.
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


def _autoregressive_masks(decode_order: Tensor, neighbor_idx: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    r"""Build the causal backward/forward neighbor masks for a decode order.

    A neighbor is "backward" (already decoded) when it precedes the current residue in ``decode_order``, and
    "forward" otherwise. The two masks partition the valid neighbors so the decoder mixes teacher-forced
    tokens only from already-decoded positions.

    Args:
        decode_order: Residue decode order `[B, L]`.
        neighbor_idx: Neighbor indices `[B, L, K]`.
        mask: Valid-residue mask `[B, L]`.

    Returns:
        Tuple ``(backward_mask, forward_mask)``, each `[B, L, K, 1]`.
    """

    length = decode_order.shape[1]
    device = decode_order.device
    reverse_permutation = F.one_hot(decode_order, num_classes=length).float()
    backward_order = torch.einsum(
        "ij,biq,bjp->bqp",
        1 - torch.triu(torch.ones(length, length, device=device)),
        reverse_permutation,
        reverse_permutation,
    )
    attend = torch.gather(backward_order, 2, neighbor_idx).unsqueeze(-1)
    mask = mask.view(mask.shape[0], length, 1, 1)
    return mask * attend, mask * (1.0 - attend)


def _local_angle_features(n: Tensor, ca: Tensor, c: Tensor, points: Tensor) -> Tensor:
    r"""Direction features of ``points`` expressed in each residue's local backbone frame.

    Builds a per-residue orthonormal frame from N/CA/C, projects the context points into it, and returns
    scale-invariant planar/spatial direction cosines (the LigandMPNN local-angle context features).

    Args:
        n: Nitrogen coordinates `[B, L, 3]`.
        ca: Alpha-carbon coordinates `[B, L, 3]`.
        c: Carbon coordinates `[B, L, 3]`.
        points: Context atom coordinates `[B, L, A, 3]`.

    Returns:
        Direction features `[B, L, A, 4]`.
    """

    first_axis = F.normalize(n - ca, dim=-1)
    c_vector = c - ca
    projected = torch.einsum("bli,bli->bl", first_axis, c_vector)[..., None]
    second_axis = F.normalize(c_vector - first_axis * projected, dim=-1)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    frame = torch.stack((first_axis, second_axis, third_axis), dim=-1)
    local = torch.einsum("blqp,blyq->blyp", frame, points - ca[:, :, None])
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


def _backbone_atoms(coords: Tensor) -> dict[str, Tensor]:
    r"""Split N/CA/C/O coordinates and add the idealized virtual CB atom.

    Args:
        coords: Backbone coordinates `[B, L, 4, 3]` ordered N, CA, C, O.

    Returns:
        Mapping ``{"N", "CA", "C", "O", "CB"}`` of `[B, L, 3]` coordinate tensors.
    """

    n = coords[:, :, 0]
    ca = coords[:, :, 1]
    c = coords[:, :, 2]
    o = coords[:, :, 3]
    b = ca - n
    c_vec = c - ca
    a = torch.cross(b, c_vec, dim=-1)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca
    return {"N": n, "CA": ca, "C": c, "O": o, "CB": cb}


def _knn(points: Tensor, mask: Tensor, k_neighbors: int, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    r"""Find the ``k_neighbors`` nearest residues by CA distance, pushing invalid pairs to the far end.

    Args:
        points: Reference (CA) coordinates `[B, L, 3]`.
        mask: Valid-residue mask `[B, L]`.
        k_neighbors: Number of neighbors to keep (capped at ``L``).
        eps: Numerical floor under the distance square root.

    Returns:
        Tuple ``(distances, neighbor_idx)`` of the top-k nearest neighbors, each `[B, L, k]`.
    """

    pair_mask = mask[:, None, :] * mask[:, :, None]
    distances = torch.sqrt(torch.sum((points[:, :, None] - points[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    max_distance = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * max_distance
    return torch.topk(adjusted, min(k_neighbors, points.shape[1]), dim=-1, largest=False)


def _rbf(distances: Tensor, num_bins: int) -> Tensor:
    r"""Expand distances into Gaussian radial basis features over the fixed range ``[2, 22]`` angstrom.

    Args:
        distances: Distance tensor `[...]`.
        num_bins: Number of evenly-spaced RBF centers.

    Returns:
        RBF features `[..., num_bins]`.
    """

    centers = torch.linspace(2.0, 22.0, num_bins, device=distances.device, dtype=distances.dtype)
    centers = centers.view(*((1,) * distances.ndim), num_bins)
    width = (22.0 - 2.0) / num_bins
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def gather_edges(edges: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Select each residue's neighbor edge features from a dense edge tensor.

    Args:
        edges: Dense per-pair edge features `[B, L, L, C]`.
        neighbor_idx: Neighbor indices `[B, L, K]`.

    Returns:
        The gathered neighbor edges `[B, L, K, C]`.

    Examples:
        >>> import torch
        >>> edges = torch.zeros(1, 6, 6, 3)
        >>> idx = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]])
        >>> gather_edges(edges, idx).shape
        torch.Size([1, 6, 2, 3])
    """

    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.shape[-1])
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Gather neighbor node features indexed by a residue-neighbor map.

    Args:
        nodes: Per-residue node features `[B, L, C]`.
        neighbor_idx: Neighbor indices `[B, L, K]`.

    Returns:
        The gathered neighbor node features `[B, L, K, C]`.

    Examples:
        >>> import torch
        >>> nodes = torch.arange(12).float().reshape(1, 6, 2)
        >>> idx = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]])
        >>> out = gather_nodes(nodes, idx)
        >>> out.shape
        torch.Size([1, 6, 2, 2])
        >>> out[0, 0].tolist()
        [[0.0, 1.0], [2.0, 3.0]]
    """

    flat_idx = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    flat_idx = flat_idx.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
    gathered = torch.gather(nodes, 1, flat_idx)
    return gathered.view(*neighbor_idx.shape, nodes.shape[-1])


def cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Concatenate gathered neighbor node states onto an existing edge/neighbor context.

    Args:
        node_states: Per-residue node states `[B, L, C]` to gather by neighbor.
        neighbor_states: Existing neighbor context `[B, L, K, D]` to append to.
        neighbor_idx: Neighbor indices `[B, L, K]`.

    Returns:
        The concatenated context `[B, L, K, D + C]`.

    Examples:
        >>> import torch
        >>> nodes = torch.arange(12).float().reshape(1, 6, 2)
        >>> ctx = torch.zeros(1, 6, 2, 5)
        >>> idx = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]])
        >>> cat_neighbors_nodes(nodes, ctx, idx).shape
        torch.Size([1, 6, 2, 7])
    """

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_idx)), dim=-1)


def _gather_pair_rbf(source: Tensor, target: Tensor, neighbor_idx: Tensor, num_bins: int) -> Tensor:
    r"""RBF-expand the neighbor distances between two backbone-atom sets.

    Args:
        source: Source atom coordinates `[B, L, 3]`.
        target: Target atom coordinates `[B, L, 3]`.
        neighbor_idx: Neighbor indices `[B, L, K]`.
        num_bins: Number of RBF centers.

    Returns:
        Per-edge RBF distance features `[B, L, K, num_bins]`.
    """

    distances = torch.sqrt(torch.sum((source[:, :, None] - target[:, None, :]) ** 2, dim=-1) + 1e-6)
    neighbor_distances = gather_edges(distances[:, :, :, None], neighbor_idx)[:, :, :, 0]
    return _rbf(neighbor_distances, num_bins)


def _backbone_pair_rbf(atoms: Mapping[str, Tensor], neighbor_idx: Tensor, num_bins: int) -> Tensor:
    r"""Concatenate RBF distance features for all 25 ordered backbone-atom pairs per edge.

    Args:
        atoms: Mapping of backbone atom coordinates from
            [`_backbone_atoms`][ofoldx.models.ligandmpnn.model._backbone_atoms].
        neighbor_idx: Neighbor indices `[B, L, K]`.
        num_bins: Number of RBF centers per pair.

    Returns:
        Concatenated pair RBF edge features `[B, L, K, 25 * num_bins]`.
    """

    atom_pairs = (
        ("CA", "CA"),
        ("N", "N"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "CB"),
        ("CA", "N"),
        ("CA", "C"),
        ("CA", "O"),
        ("CA", "CB"),
        ("N", "C"),
        ("N", "O"),
        ("N", "CB"),
        ("CB", "C"),
        ("CB", "O"),
        ("O", "C"),
        ("N", "CA"),
        ("C", "CA"),
        ("O", "CA"),
        ("CB", "CA"),
        ("C", "N"),
        ("O", "N"),
        ("CB", "N"),
        ("C", "CB"),
        ("O", "CB"),
        ("C", "O"),
    )
    return torch.cat([_gather_pair_rbf(atoms[a], atoms[b], neighbor_idx, num_bins) for a, b in atom_pairs], dim=-1)


@dataclass(slots=True, kw_only=True)
class LigandMPNNInput(BaseInput):
    r"""
    Typed tensor input for the native LigandMPNN design generator.

    Bundles the backbone geometry, sequence, and ligand context that
    [`LigandMPNNDesignGenerator`][ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator] scores or
    samples over. All tensors share the batch ``B`` and residue ``L`` axes; ``A`` is the per-residue
    ligand/context atom budget. Optional fields tailor generation (design mask, amino-acid bias, decode-order
    noise, symmetry ties) and default to no constraint when omitted.

    Attributes:
        backbone_coords: Backbone N/CA/C/O coords with shape ``(B, L, 4, 3)``.
        sequence_tokens: Residue tokens in the LigandMPNN alphabet with shape ``(B, L)``.
        token_mask: Valid-residue mask with shape ``(B, L)``.
        residue_index: Residue numbers used for relative-position features with shape ``(B, L)``.
        chain_index: Integer chain ids with shape ``(B, L)``.
        ligand_atom_coords: Ligand/context atom coords with shape ``(B, L, A, 3)``.
        ligand_atom_types: Ligand/context atom atomic numbers with shape ``(B, L, A)``.
        ligand_atom_mask: Valid ligand/context atom mask with shape ``(B, L, A)``.
        design_mask: Optional mask where one marks positions to design and zero marks fixed positions.
        amino_acid_bias: Optional per-position amino-acid logit bias with shape ``(B, L, 21)``.
        decoding_order_noise: Optional decoding-order noise with shape ``(num_samples, L)``.
        temperature: Sampling temperature used by ``sample`` and ``generate``.
        num_samples: Number of autoregressive samples or decode orders.
        symmetry_residue_groups: Optional groups of residue indices tied to share one sampled token.
        symmetry_weights: Optional per-residue logit weights aligned with ``symmetry_residue_groups``.
        all_atom_coords: Optional atom37 coordinates ``(B, L, 37, 3)`` supplying side-chain context.
        all_atom_mask: Optional atom37 existence mask ``(B, L, 37)`` for the side-chain context.
    """

    backbone_coords: Tensor
    sequence_tokens: Tensor
    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    ligand_atom_coords: Tensor
    ligand_atom_types: Tensor
    ligand_atom_mask: Tensor
    design_mask: Tensor | None = None
    amino_acid_bias: Tensor | None = None
    decoding_order_noise: Tensor | None = None
    temperature: float | Tensor = 0.1
    num_samples: int | None = None
    symmetry_residue_groups: list[list[int]] | None = None
    symmetry_weights: list[list[float]] | None = None
    all_atom_coords: Tensor | None = None
    all_atom_mask: Tensor | None = None
