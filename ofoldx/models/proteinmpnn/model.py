r"""
ProteinMPNN inverse-folding model for fixed-backbone sequence design.
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
from ..contract import BaseInput, SequenceDesignCandidate, register_pytree_dataclass
from ..mixin import ModelMixin
from ..registry import DESIGN_MODELS, MODELS
from .config import ProteinMPNNConfig


@register_pytree_dataclass
@dataclass(slots=True)
class ProteinMPNNRepresentations:
    r"""
    Encoder output at the ProteinMPNN message-passing boundary.

    Wraps the encoded node states in the typed [`Single`][ofoldx.modules.representation.Single] carrier -- the
    ProteinMPNN analogue of a trunk single representation -- so the encode boundary is axis-typed the same way
    [`StructurePredictionOutput.coords`][ofoldx.models.contract.StructurePredictionOutput] wraps predicted
    coordinates. The decoder consumes raw tensors, so callers unwrap ``single.tensor`` before the autoregressive
    decode step.

    ProteinMPNN has no dense ``token_i x token_j`` pair representation: its edges live on a sparse
    ``k``-nearest-neighbor graph with shape ``(B, L, K, hidden)`` (``K != L``), which does not satisfy the
    [`Pair`][ofoldx.modules.representation.Pair] carrier's equal ``i``/``j`` axis contract. The graph edges and
    neighbor index therefore ride along as raw tensors rather than a mislabelled ``Pair``.

    The dataclass is registered as a ``torch.utils._pytree`` node (via the ``Single`` carrier and
    [`register_pytree_dataclass`][ofoldx.models.contract.register_pytree_dataclass]), so ``torch.compile`` traces
    straight through it with no graph break.

    Attributes:
        single: Encoded node states ``(B, L, hidden)`` as a ``Single`` carrier.
        edges: Encoded edge states over the neighbor graph ``(B, L, K, hidden)``.
        neighbor_idx: Neighbor index ``(B, L, K)``.
    """

    single: Single
    edges: Tensor
    neighbor_idx: Tensor


@MODELS.register("proteinmpnn")
@DESIGN_MODELS.register("proteinmpnn")
class ProteinMPNNDesigner(ModelMixin):
    r"""
    Message-passing model that designs and scores sequences for a fixed protein backbone
    (ProteinMPNN inverse folding).

    Runs the ProteinMPNN encoder/decoder over the backbone coordinates in a
    [`ProteinMPNNInput`][ofoldx.models.proteinmpnn.model.ProteinMPNNInput] and returns a
    [`SequenceDesignCandidate`][ofoldx.models.contract.SequenceDesignCandidate].

    ``forward`` is the **training** path: it runs the teacher-forced encoder/decoder (upstream
    ``ProteinMPNN.forward``) and returns per-residue ``log_probs`` a sequence-recovery loss is taken over, with
    gradients flowing through the whole message-passing stack. ``design`` is the **inference** capability verb: it
    autoregressively samples sequences under ``torch.no_grad`` (dispatching to the tied-position decoder when the
    input carries ``tied_positions``). The scoring verbs (``score``, ``single_residue_score``,
    ``conditional_probs``, ``unconditional_probs``) evaluate a given sequence; numerics match the released
    reference so converted checkpoints reproduce upstream results.

    Args:
        config: Architecture configuration. Defaults to a fresh
            [`ProteinMPNNConfig`][ofoldx.models.proteinmpnn.config.ProteinMPNNConfig] when omitted.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("proteinmpnn")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("proteinmpnn")  # doctest: +SKIP
        >>> batch = processor("complex.pdb")  # doctest: +SKIP
        >>> output = model.design(batch, temperature=0.1)  # doctest: +SKIP
    """

    config_class = ProteinMPNNConfig
    #: Typed input built by the keyword-tensor call path ``model(**inputs)``; assigned at module load once
    #: ``ProteinMPNNInput`` is defined (see the bottom of this module).
    input_class: ClassVar[type[BaseInput]]

    def __init__(self, config: ProteinMPNNConfig | None = None) -> None:
        config = ProteinMPNNConfig() if config is None else config
        super().__init__(config)
        self.backbone = ProteinMPNNBackbone(config)
        self.post_init()

    def _init_module_weights(self, module: nn.Module) -> None:
        return None

    def reset_parameters(self) -> None:
        r"""
        Re-initialize backbone weights to match the released ProteinMPNN scheme.

        Applies Xavier-uniform init to every parameter with more than one dimension (weight matrices), leaving
        biases and 1-D parameters untouched, exactly as the reference does for freshly-created models.

        Examples:
            >>> model = ProteinMPNNDesigner()  # doctest: +SKIP
            >>> model.reset_parameters()  # doctest: +SKIP
        """

        for parameter in self.backbone.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        batch: ProteinMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Training forward: teacher-forced sequence-recovery logits over the backbone (upstream ``ProteinMPNN.forward``).

        Runs the encoder once and the autoregressive decoder over a noised decode order with the true sequence
        teacher-forced, returning per-residue ``log_probs`` a cross-entropy sequence-recovery loss is taken over.
        Gradients flow through the entire message-passing stack, so this is the model's training entry point;
        [`design`][ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner.design] is the distinct ``torch.no_grad``
        inference path. The computation is identical to
        [`score`][ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner.score] (teacher-forced logits are the same
        tensors used for scoring); ``forward`` simply leaves autograd enabled.

        Args:
            batch: Typed ProteinMPNN input carrying backbone coords and the true sequence to recover.
            use_sequence: Whether to teacher-force on the input sequence identities; ``False`` trains from backbone
                context alone.
            num_samples: Number of decode orders to average over; ``None`` uses the batch default.

        Returns:
            A design output whose ``scores`` hold per-residue ``log_probs`` `[B, L, 21]` and the per-sequence
            ``sequence_log_likelihood`` (negate and mean-reduce for the training loss).

        Examples:
            >>> output = model(batch)  # doctest: +SKIP
            >>> loss = -output.scores["sequence_log_likelihood"].mean()  # doctest: +SKIP
            >>> loss.backward()  # doctest: +SKIP
        """

        return self.score(batch, use_sequence=use_sequence, num_samples=num_samples)

    def score(
        self,
        batch: ProteinMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Score a sequence against the backbone with the standard one-pass decoder.

        Runs the encoder once and the autoregressive decoder over a noised decode order, returning per-residue
        log-probabilities and the masked mean sequence log-likelihood.

        Args:
            batch: Typed ProteinMPNN input carrying backbone coords and the sequence to score.
            use_sequence: Whether to condition on the input sequence identities; ``False`` scores from backbone
                context alone.
            num_samples: Number of decode orders to average over; ``None`` uses the batch default.

        Returns:
            A design output with sequence logits/tokens and ``scores`` holding ``log_probs`` `[B, L, 21]` and the
            per-sequence ``sequence_log_likelihood``.

        Examples:
            >>> output = model.score(batch)  # doctest: +SKIP
            >>> output.scores["log_probs"].shape  # doctest: +SKIP
            torch.Size([1, 9, 21])
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
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
                "use_sequence": use_sequence,
            },
        )

    def single_residue_score(
        self,
        batch: ProteinMPNNInput,
        *,
        use_sequence: bool = True,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Score each residue with the official single-position decoding protocol.

        Re-runs the decoder once per residue with that residue placed last (or first, when ``use_sequence`` is
        ``False``) in the decode order, giving each position's log-probabilities its own maximally-conditioned
        context. Slower than ``score`` but matches the reference single-residue scoring path.

        Args:
            batch: Typed ProteinMPNN input carrying backbone coords and the sequence to score.
            use_sequence: Whether to condition on the input sequence identities at the other positions.
            num_samples: Number of decode orders to average over; ``None`` uses the batch default.

        Returns:
            A design output whose ``scores`` hold per-residue ``log_probs`` and the sequence log-likelihood, each
            position scored under its own single-residue decode order.

        Examples:
            >>> output = model.single_residue_score(batch)  # doctest: +SKIP
            >>> output.scores["log_probs"].shape  # doctest: +SKIP
            torch.Size([1, 9, 21])
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.single_residue_score(features, use_sequence=use_sequence)
        log_probs = scored["log_probs"]
        return SequenceDesignCandidate(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={
                "log_probs": log_probs,
                "sequence_log_likelihood": _sequence_log_likelihood(scored["S"], log_probs, features["mask"]),
            },
            metadata={
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
                "score_mode": "single_residue",
                "use_sequence": use_sequence,
            },
        )

    @torch.no_grad()
    def design(
        self,
        batch: ProteinMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Design sequences for the backbone by autoregressive sampling under ``torch.no_grad`` (inference verb).

        The house design capability verb and the model's distinct inference path (``forward`` is the training
        path). Draws each position from ``softmax((logits + bias) / temperature)`` in the noised decode order,
        honoring the full conditioning surface (fixed positions, per-residue bias, PSSM, omit-AA); native symmetric
        residue groups are decoded jointly when present. When the input carries ``tied_positions`` the tied-position
        decoder is used so each tied group shares one sampled amino acid (upstream ``tied_sample``).

        Args:
            batch: Typed ProteinMPNN input for a single backbone (batch axis size one).
            temperature: Sampling temperature; ``None`` uses the batch temperature.
            num_samples: Number of independent sequences to sample; ``None`` uses the batch default.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A design output with sampled tokens/logits and ``scores`` holding ``log_probs`` and ``sampling_probs``.

        Examples:
            >>> output = model.design(batch, temperature=0.1, num_samples=8)  # doctest: +SKIP
            >>> output.sequence_tokens.shape  # doctest: +SKIP
            torch.Size([8, 9])
        """

        if batch.tied_positions:
            return self._tied_sample(batch, temperature=temperature, num_samples=num_samples, generator=generator)
        return self._sample(batch, temperature=temperature, num_samples=num_samples, generator=generator)

    def _sample(
        self,
        batch: ProteinMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Independent-position autoregressive sampler backing ``design`` (upstream ``ProteinMPNN.sample``).

        Draws each position from ``softmax((logits + bias) / temperature)`` in the noised decode order, honoring
        the full conditioning surface (fixed positions, per-residue bias, PSSM, omit-AA). Symmetric residue groups
        are decoded jointly when present. Private: callers reach it through
        [`design`][ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner.design].

        Args:
            batch: Typed ProteinMPNN input for a single backbone (batch axis size one).
            temperature: Sampling temperature; ``None`` uses the batch temperature.
            num_samples: Number of independent sequences to sample; ``None`` uses the batch default.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A design output with sampled tokens/logits and ``scores`` holding ``log_probs`` `[num_samples, L, 21]`
            and per-position ``sampling_probs``.
        """

        features = self._feature_dict(
            batch,
            num_samples=num_samples,
            temperature=temperature,
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
            },
            metadata={
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
            },
        )

    def _tied_sample(
        self,
        batch: ProteinMPNNInput,
        *,
        temperature: float | Tensor | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Tied-position autoregressive sampler backing ``design`` (upstream ``ProteinMPNN.tied_sample``).

        Positions in a tied group share one sampled amino acid, drawn from the ``tied_beta``-weighted mean of the
        member logits with the conditioning surface applied once per group. Tied groups come from
        ``batch.tied_positions``. Private: callers reach it through
        [`design`][ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner.design].

        Args:
            batch: Typed ProteinMPNN input carrying ``tied_positions`` (and optional ``tied_weights``).
            temperature: Sampling temperature; ``None`` uses the batch temperature.
            num_samples: Number of independent sequences to sample; ``None`` uses the batch default.
            generator: Optional RNG for reproducible sampling.

        Returns:
            A design output with sampled tokens/logits and ``scores`` holding ``log_probs`` and ``sampling_probs``,
            with tied group members sharing identical values.
        """

        features = self._feature_dict(
            batch,
            num_samples=num_samples,
            temperature=temperature,
            generator=generator,
            for_sampling=True,
        )
        sampled = self.backbone.tied_sample(features)
        return SequenceDesignCandidate(
            sequence_logits=sampled["logits"],
            sequence_tokens=sampled["S"],
            scores={
                "log_probs": sampled["log_probs"],
                "sampling_probs": sampled["sampling_probs"],
            },
            metadata={
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
                "sample_mode": "tied",
            },
        )

    def conditional_probs(
        self,
        batch: ProteinMPNNInput,
        *,
        backbone_only: bool = False,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        Per-residue conditional amino-acid log-probabilities given the rest of the context.

        For each designable position the decoder is run so its logits see every other residue; ``backbone_only``
        additionally hides that position's own identity, leaving pure backbone context. Matches upstream
        ``ProteinMPNN.conditional_probs``.

        Args:
            batch: Typed ProteinMPNN input carrying backbone coords and the conditioning sequence.
            backbone_only: If ``True``, also hide each target position's own identity (backbone-only conditioning).
            num_samples: Number of decode orders; ``None`` uses the batch default.

        Returns:
            A design output whose ``scores`` hold per-residue conditional ``log_probs`` `[B, L, 21]`.

        Examples:
            >>> output = model.conditional_probs(batch)  # doctest: +SKIP
            >>> output.scores["log_probs"].shape  # doctest: +SKIP
            torch.Size([1, 9, 21])
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.conditional_probs(features, backbone_only=backbone_only)
        return SequenceDesignCandidate(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={"log_probs": scored["log_probs"]},
            metadata={
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
                "probs_mode": "conditional_backbone" if backbone_only else "conditional",
            },
        )

    def unconditional_probs(
        self,
        batch: ProteinMPNNInput,
        *,
        num_samples: int | None = None,
    ) -> SequenceDesignCandidate:
        r"""
        One-pass unconditional amino-acid log-probabilities from backbone context alone.

        Runs the decoder a single time with an empty backward (autoregressive) mask, so every position sees only
        the encoded backbone context. Matches upstream ``ProteinMPNN.unconditional_probs``.

        Args:
            batch: Typed ProteinMPNN input carrying backbone coords.
            num_samples: Number of decode orders; ``None`` uses the batch default.

        Returns:
            A design output whose ``scores`` hold per-residue unconditional ``log_probs`` `[B, L, 21]`.

        Examples:
            >>> output = model.unconditional_probs(batch)  # doctest: +SKIP
            >>> output.scores["log_probs"].shape  # doctest: +SKIP
            torch.Size([1, 9, 21])
        """

        features = self._feature_dict(batch, num_samples=num_samples, for_sampling=False)
        scored = self.backbone.unconditional_probs(features)
        return SequenceDesignCandidate(
            sequence_logits=scored["logits"],
            sequence_tokens=scored["S"],
            scores={"log_probs": scored["log_probs"]},
            metadata={
                "architecture": "proteinmpnn",
                "backend": "native_proteinmpnn",
                "num_edges": self.config.k_neighbors,
                "probs_mode": "unconditional",
            },
        )

    def encode(self, batch: ProteinMPNNInput | Mapping[str, Any]):  # type: ignore[override]
        r"""
        Run the geometric encoder and return node/edge states plus the neighbor graph.

        Accepts either a typed input (converted to the internal feature dict first) or an already-built feature
        mapping, so the encoder can be reused across the scoring and sampling protocols.

        Args:
            batch: A typed ProteinMPNN input or a pre-built feature mapping.

        Returns:
            A ``(node_states, edge_states, neighbor_idx)`` tuple from the backbone encoder.

        Examples:
            >>> node_states, edge_states, neighbor_idx = model.encode(batch)  # doctest: +SKIP
        """

        features = self._feature_dict(batch, for_sampling=False) if isinstance(batch, ProteinMPNNInput) else batch
        return self.backbone.encode(features)

    def _feature_dict(
        self,
        batch: ProteinMPNNInput,
        *,
        num_samples: int | None = None,
        temperature: float | Tensor | None = None,
        generator: torch.Generator | None = None,
        for_sampling: bool,
    ) -> dict[str, Any]:
        r"""
        Validate a typed input and lower it into the upstream-style feature dict the backbone consumes.

        Resolves the effective sample count, materializes a default design mask and amino-acid bias when absent,
        and produces the per-sample decode-order noise (``randn``): random for sampling, zeros for scoring, or the
        caller-supplied ``decoding_order_noise`` reshaped to the resolved sample count.

        Args:
            batch: Typed ProteinMPNN input to lower.
            num_samples: Explicit sample count; ``None`` falls back to the batch default.
            temperature: Sampling temperature override; ``None`` keeps the batch temperature.
            generator: Optional RNG used when sampling decode-order noise.
            for_sampling: Whether the features feed a sampling path (random noise) or a scoring path (zero noise).

        Returns:
            A feature mapping keyed by the upstream names (``X``, ``S``, ``mask``, ``randn``, the PSSM/omit/tied
            conditioning surface, ...) that the backbone methods consume.

        Raises:
            ValueError: If ``decoding_order_noise``'s leading dimension is incompatible with the resolved sample
                count, or if the input fails
                [`_validate_proteinmpnn_input`][ofoldx.models.proteinmpnn.model._validate_proteinmpnn_input].
        """

        _validate_proteinmpnn_input(batch, require_sampling=for_sampling)
        batch_count, residue_count = batch.sequence_tokens.shape
        device = batch.sequence_tokens.device
        num_samples_resolved = int(num_samples if num_samples is not None else (batch.num_samples or 1))
        chain_mask = batch.design_mask if batch.design_mask is not None else torch.ones_like(batch.token_mask)
        bias = batch.amino_acid_bias
        if bias is None:
            bias = torch.zeros(batch_count, residue_count, self.config.vocab_size, device=device)
        if batch.decoding_order_noise is not None:
            randn = batch.decoding_order_noise
            if num_samples is None and batch.num_samples is None:
                num_samples_resolved = int(
                    randn.shape[0] if batch_count == 1 else max(randn.shape[0] // batch_count, 1)
                )
            if randn.shape[0] == num_samples_resolved:
                if batch_count > 1 and not for_sampling:
                    randn = randn[:, None, :].expand(num_samples_resolved, batch_count, residue_count)
                    randn = randn.reshape(num_samples_resolved * batch_count, residue_count)
            elif not (not for_sampling and randn.shape[0] == num_samples_resolved * batch_count):
                expected = (
                    num_samples_resolved
                    if batch_count == 1 or for_sampling
                    else f"{num_samples_resolved} or {num_samples_resolved * batch_count}"
                )
                raise ValueError(f"decoding_order_noise first dimension must match num_samples ({expected})")
        elif for_sampling:
            randn = torch.randn(num_samples_resolved, residue_count, device=device, generator=generator)
        else:
            randn = torch.zeros(num_samples_resolved * batch_count, residue_count, device=device)
        return {
            "X": batch.backbone_coords,
            "S": batch.sequence_tokens,
            "mask": batch.token_mask,
            "residue_index": batch.residue_index,
            "chain_index": batch.chain_index,
            "membrane_labels": batch.membrane_labels,
            "chain_mask": chain_mask,
            "bias": bias,
            "randn": randn,
            "generator": generator,
            "temperature": batch.temperature if temperature is None else temperature,
            "num_samples": num_samples_resolved,
            "symmetry_residues": batch.symmetry_residue_groups if batch.symmetry_residue_groups is not None else [[]],
            "symmetry_weights": batch.symmetry_weights if batch.symmetry_weights is not None else [[]],
            "tied_positions": batch.tied_positions if batch.tied_positions is not None else [],
            "tied_weights": batch.tied_weights,
            "fixed_position_mask": batch.fixed_position_mask,
            "omit_amino_acid_mask": batch.omit_amino_acid_mask,
            "pssm_coefficient": batch.pssm_coefficient,
            "pssm_bias": batch.pssm_bias,
            "pssm_log_odds_mask": batch.pssm_log_odds_mask,
            "pssm_multiplier": batch.pssm_multiplier,
            "pssm_log_odds_flag": batch.pssm_log_odds_flag,
            "pssm_bias_flag": batch.pssm_bias_flag,
            "bias_by_residue": batch.bias_by_residue,
        }


class ProteinMPNNBackbone(nn.Module):
    r"""
    Backbone-only ProteinMPNN message passing and autoregressive decoder.

    Holds the geometric feature encoder, the node/edge embeddings, the encoder and decoder layer stacks, and the
    output projection. Exposes the tensor-level implementation of every protocol
    ([`ProteinMPNNDesigner`][ofoldx.models.proteinmpnn.model.ProteinMPNNDesigner] wraps these in
    typed inputs/outputs); each method takes the feature dict produced by ``_feature_dict`` and returns a mapping of
    result tensors.

    Args:
        config: Architecture configuration providing the widths, layer counts, and neighbor count.
    """

    def __init__(self, config: ProteinMPNNConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_encoder = ProteinBackboneFeatures(
            edge_feature_size=config.edge_feature_size,
            node_feature_size=config.node_feature_size,
            num_positional_embeddings=16,
            num_rbf=16,
            k_neighbors=config.k_neighbors,
            coord_noise_scale=config.coord_noise_scale,
        )
        if config.node_feature_size > 0:
            self.node_embedding = nn.Linear(config.edge_feature_size, config.hidden_size)
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
        Encode the backbone into node and edge states over the k-nearest-neighbor graph.

        Builds geometric edge features and the neighbor index, embeds the edges, zero-initializes the node states,
        and runs the encoder layer stack under the residue-pair attend mask.

        Args:
            features: Feature mapping (as built by ``_feature_dict``) with backbone coords, masks, and indices.

        Returns:
            A ``(node_states, edge_states, neighbor_idx)`` tuple with shapes `[B, L, hidden_size]`,
            `[B, L, K, hidden_size]`, and `[B, L, K]`.
        """

        edges, neighbor_idx = self.feature_encoder(features)
        mask = features["mask"]
        node_features = self.feature_encoder.node_features(features)
        if node_features is None:
            node_states = edges.new_zeros(edges.shape[0], edges.shape[1], self.hidden_size)
        else:
            node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edges)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_idx).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(node_states, edge_states, neighbor_idx, mask, attend_mask)
        return node_states, edge_states, neighbor_idx

    @staticmethod
    def _representations(node_states: Tensor, edge_states: Tensor, neighbor_idx: Tensor) -> ProteinMPNNRepresentations:
        r"""
        Wrap the raw encoder tuple in the typed
        [`ProteinMPNNRepresentations`][ofoldx.models.proteinmpnn.model.ProteinMPNNRepresentations] boundary.

        Puts the node states behind the [`Single`][ofoldx.modules.representation.Single] carrier while the sparse
        neighbor-graph edges ride along as raw tensors (ProteinMPNN has no square ``token_i x token_j`` pair). The
        carrier is pytree-registered, so this introduces no ``torch.compile`` graph break.
        """

        return ProteinMPNNRepresentations(single=Single(node_states), edges=edge_states, neighbor_idx=neighbor_idx)

    def score(self, features: Mapping[str, Any], *, use_sequence: bool) -> dict[str, Tensor]:
        r"""
        Tensor-level standard scoring pass (upstream ``ProteinMPNN.forward``).

        Encodes the backbone, builds the autoregressive forward/backward masks from a noised decode order (or the
        symmetric group order), then runs the decoder either sequence-conditioned or backbone-only.

        Args:
            features: Feature mapping built by ``_feature_dict``.
            use_sequence: Whether the decoder conditions on the input sequence identities.

        Returns:
            A mapping with ``"S"`` (tiled sequence), ``"logits"``, and ``"log_probs"``, each of shape
            `[B * num_samples, L, 21]`.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"]
        randn = features["randn"]
        num_samples = int(features["num_samples"])
        batch, length = sequence.shape
        device = sequence.device

        # Typed carrier at the encode boundary: node states are the ProteinMPNN single representation. The decoder
        # consumes raw tensors, so unwrap ``.single.tensor`` before the autoregressive decode below.
        representations = self._representations(*self.encode(features))
        node_states = representations.single.tensor
        edge_states = representations.edges
        neighbor_idx = representations.neighbor_idx
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
        Tensor-level single-position scoring pass.

        Loops over every residue, rebuilding the decode order so the target position is decoded last (or first when
        ``use_sequence`` is ``False``), and keeps only that position's logits/log-probs from each pass.

        Args:
            features: Feature mapping built by ``_feature_dict``.
            use_sequence: Whether the decoder conditions on the input sequence identities at other positions.

        Returns:
            A mapping with ``"S"``, ``"logits"``, and ``"log_probs"`` of shape `[B * num_samples, L, 21]`, each
            position filled from its own single-residue decode order.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = features["chain_mask"]
        randn = features["randn"]
        num_samples = int(features["num_samples"])
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
        Tensor-level autoregressive sampler (upstream ``ProteinMPNN.sample``).

        Encodes the backbone, forms the decode order from the fixed-position-aware chain mask, and dispatches to the
        independent or symmetric decoding routine depending on whether symmetry groups are present.

        Args:
            features: Feature mapping built by ``_feature_dict`` (with sampling ``randn`` and conditioning fields).

        Returns:
            A mapping with sampled ``"S"``, ``"sampling_probs"``, ``"logits"``, and ``"log_probs"``.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"] * _fixed_position_mask(features, mask)
        bias = features["bias"]
        randn = features["randn"]
        temperature = features["temperature"]
        symmetry_groups = features["symmetry_residues"]
        symmetry_weights_groups = features["symmetry_weights"]
        generator = features.get("generator")
        num_samples = int(features["num_samples"])
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
                _conditioning_bundle(features).repeat(num_samples),
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
        Order-independent decode setup shared by the independent and symmetric samplers.
        """

        _, length = sequence.shape
        device = sequence.device
        node_states = node_states.repeat(num_samples, 1, 1)
        edge_states = edge_states.repeat(num_samples, 1, 1, 1)
        neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
        token_states = torch.zeros_like(node_states)
        return {
            "sequence_true": sequence.repeat(num_samples, 1),
            "node_states": node_states,
            "edge_states": edge_states,
            "neighbor_idx": neighbor_idx,
            "chain_mask": chain_mask.repeat(num_samples, 1),
            "mask": mask.repeat(num_samples, 1),
            "bias": bias.repeat(num_samples, 1, 1),
            "sampled": 20 * torch.ones(num_samples, length, dtype=torch.long, device=device),
            "token_states": token_states,
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
        conditioning: _Conditioning | None = None,
    ) -> dict[str, Tensor]:
        r"""
        Independent-position autoregressive decode loop (no symmetry ties).

        Walks the decode order one position at a time, gathering that position's edges/context, running the decoder
        stack, applying the conditioning surface to the logits, and sampling a token that is scattered back into the
        running state for later positions.

        Args:
            sequence: True sequence tokens `[B, L]` (used where the chain mask holds a position fixed).
            mask: Valid-residue mask `[B, L]`.
            chain_mask: Designable-position mask `[B, L]`.
            bias: Per-position amino-acid logit bias `[B, L, 21]`.
            temperature: Sampling temperature.
            decode_order: Per-batch decode order `[B, L]`.
            node_states: Encoder node states.
            edge_states: Encoder edge states.
            neighbor_idx: Neighbor index `[B, L, K]`.
            num_samples: Number of independent samples to draw.
            generator: Optional RNG for reproducibility.
            conditioning: Per-position conditioners; an empty bundle is used when omitted.

        Returns:
            A mapping with sampled ``"S"``, ``"sampling_probs"``, ``"logits"``, and ``"log_probs"``.
        """

        _, length = sequence.shape
        conditioning = conditioning if conditioning is not None else _Conditioning.empty()
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
            sample_probs = conditioning.apply(logits, bias_t, temperature, residue)
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
        Symmetric autoregressive decode loop where grouped residues share one sampled token.

        Decodes group by group: every member is run through the decoder, their logits are ``symmetry_weights``-summed
        into one group distribution, a single token is sampled, and it is written to all members before moving on.

        Args:
            sequence: True sequence tokens `[B, L]`.
            mask: Valid-residue mask `[B, L]`.
            chain_mask: Designable-position mask `[B, L]`.
            bias: Per-position amino-acid logit bias `[B, L, 21]`.
            temperature: Sampling temperature.
            decode_order: Per-batch decode order `[B, L]` following the grouped order.
            ordered_groups: Symmetry residue groups in decode order.
            symmetry_weights: Per-position logit weights `[L]` used to combine member logits.
            node_states: Encoder node states.
            edge_states: Encoder edge states.
            neighbor_idx: Neighbor index `[B, L, K]`.
            num_samples: Number of independent samples to draw.
            generator: Optional RNG for reproducibility.

        Returns:
            A mapping with sampled ``"S"``, ``"sampling_probs"``, ``"logits"``, and ``"log_probs"``, with grouped
            positions sharing identical sampled values.
        """

        backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
        backward_mask = backward_mask.repeat(num_samples, 1, 1, 1)
        forward_mask = forward_mask.repeat(num_samples, 1, 1, 1)
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

    def tied_sample(self, features: Mapping[str, Any]) -> dict[str, Tensor]:
        r"""
        Sample tied-position sequences where grouped residues share one identity (upstream ``ProteinMPNN.tied_sample``).

        Positions in a tied group share one sampled amino acid. Per-group logits are the ``tied_beta``-weighted
        mean over member logits, and the conditioning surface (bias/PSSM/omit) is applied once per group.

        Args:
            features: Feature mapping produced by ``_feature_dict``. Beyond the encoder inputs it carries the
                sampler controls: ``S`` (`[B, L]` seed tokens), ``mask``/``chain_mask`` (`[B, L]`),
                ``temperature``, ``randn`` decode-order noise, ``num_samples``, the ``tied_positions`` groups and
                optional ``tied_weights`` (`[L]` per-position ``tied_beta``), an optional ``generator``, and the
                per-position conditioning fields (bias/PSSM/omit) consumed by ``_conditioning_bundle``.

        Returns:
            A mapping with ``S`` sampled tokens `[B * num_samples, L]`, ``sampling_probs`` and ``log_probs``
            `[B * num_samples, L, 21]`, per-position ``logits`` `[B * num_samples, L, 21]`, and the
            ``decoding_order`` used `[B * num_samples, L]`.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"] * _fixed_position_mask(features, mask)
        temperature = features["temperature"]
        randn = features["randn"]
        generator = features.get("generator")
        num_samples = int(features["num_samples"])
        conditioning = _conditioning_bundle(features).repeat(num_samples)
        tied_positions = features["tied_positions"] or []
        device = sequence.device
        length = sequence.shape[1]

        tied_weights = features.get("tied_weights")
        tied_weights = (
            torch.ones(length, device=device)
            if tied_weights is None
            else tied_weights.to(device=device, dtype=torch.float32)
        )

        node_states, edge_states, neighbor_idx = self.encode(features)
        decode_order = torch.argsort((chain_mask + 0.0001) * randn.abs())
        tied_groups = _tied_groups(decode_order[0], tied_positions)
        ordered = torch.tensor(list(itertools.chain(*tied_groups)), device=device)[None, :].repeat(num_samples, 1)

        node_states = node_states.repeat(num_samples, 1, 1)
        edge_states = edge_states.repeat(num_samples, 1, 1, 1)
        neighbor_idx = neighbor_idx.repeat(num_samples, 1, 1)
        sequence = sequence.repeat(num_samples, 1)
        chain_mask = chain_mask.repeat(num_samples, 1)
        mask = mask.repeat(num_samples, 1)
        batch = num_samples
        backward_mask, forward_mask = _autoregressive_masks(ordered, neighbor_idx, mask)

        token_states = torch.zeros_like(node_states)
        sampled = sequence.clone()
        probabilities = torch.zeros(batch, length, 21, device=device)
        log_probs_out = torch.zeros(batch, length, 21, device=device)
        logits_out = torch.zeros(batch, length, 21, device=device)
        layer_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_layers]
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)
        forward_context = forward_mask * encoder_context

        for residue_group in tied_groups:
            group_logits: Tensor | float = 0.0
            done = False
            for residue in residue_group:
                if bool((mask[:, residue] == 0).all()):
                    fixed_t = sequence[:, residue]
                    for member in residue_group:
                        token_states[:, member] = self.token_embedding(fixed_t)
                        sampled[:, member] = fixed_t
                    done = True
                    break
                edge_t = edge_states[:, residue : residue + 1]
                neighbor_idx_t = neighbor_idx[:, residue : residue + 1]
                edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
                forward_t = forward_context[:, residue : residue + 1]
                for layer_idx, layer in enumerate(self.decoder_layers):
                    decoder_context = cat_neighbors_nodes(layer_states[layer_idx], edge_token_t, neighbor_idx_t)
                    node_t = layer_states[layer_idx][:, residue : residue + 1]
                    context_t = backward_mask[:, residue : residue + 1] * decoder_context + forward_t
                    layer_states[layer_idx + 1][:, residue : residue + 1] = layer(
                        node_t, context_t, mask[:, residue][:, None]
                    )
                member_logits = self.output_projection(layer_states[-1][:, residue]) / temperature / len(residue_group)
                group_logits = group_logits + tied_weights[residue] * member_logits
            if done:
                continue

            last = residue_group[-1]
            sample_probs = conditioning.apply_logits(group_logits, temperature, last)
            sample_probs = sample_probs[:, :20] / sample_probs[:, :20].sum(dim=-1, keepdim=True)
            sampled_t = torch.multinomial(sample_probs, 1, generator=generator)[:, 0]
            chain_mask_t = chain_mask[:, last]
            sampled_t = (sampled_t * chain_mask_t + sequence[:, last] * (1.0 - chain_mask_t)).long()
            padded = F.pad(sample_probs, (0, 21 - sample_probs.shape[-1]))
            log_padded = torch.log(padded.clamp_min(1e-12))
            for member in residue_group:
                token_states[:, member] = self.token_embedding(sampled_t)
                sampled[:, member] = sampled_t
                probabilities[:, member] = padded
                log_probs_out[:, member] = log_padded
                logits_out[:, member] = group_logits

        return {
            "S": sampled,
            "sampling_probs": probabilities,
            "logits": logits_out,
            "log_probs": log_probs_out,
            "decoding_order": ordered,
        }

    def conditional_probs(self, features: Mapping[str, Any], *, backbone_only: bool = False) -> dict[str, Tensor]:
        r"""
        Score each residue conditioned on all others via one-position decode orders (upstream ``conditional_probs``).

        For each designable position the decoder is run with a one-position decode order so the logits at that
        position are conditioned on every other residue (``backbone_only`` additionally hides that position's own
        identity, leaving only backbone context).

        Args:
            features: Feature mapping produced by ``_feature_dict``, providing the encoder inputs plus ``S``
                (`[B, L]` reference tokens), ``mask`` and ``chain_mask`` (`[B, L]`); only positions where
                ``chain_mask`` is one are scored.
            backbone_only: When set, hide each scored position's own identity so its logits depend on backbone
                context alone; otherwise condition on the full context sequence.

        Returns:
            A mapping with ``S`` (the reference tokens, unchanged), and per-position ``logits`` and ``log_probs``
            of shape `[B, L, 21]`.
        """

        sequence = features["S"]
        mask = features["mask"]
        chain_mask = mask * features["chain_mask"]
        device = sequence.device
        batch, length = sequence.shape

        node_states_enc, edge_states, neighbor_idx = self.encode(features)
        token_states = self.token_embedding(sequence)
        edge_token_states = cat_neighbors_nodes(token_states, edge_states, neighbor_idx)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states_enc, encoder_edges, neighbor_idx)

        loop_positions = torch.nonzero(chain_mask[0] == 1, as_tuple=False).squeeze(-1).tolist()
        logits_out = torch.zeros(batch, length, 21, device=device)
        log_probs_out = torch.zeros(batch, length, 21, device=device)
        # order_mask alone partitions the decode order (target vs context); a constant, strictly-positive
        # tie-break makes it deterministic regardless of the caller's decode-order noise.
        tie_break = torch.ones(1, length, device=device)

        for position in loop_positions:
            node_states = node_states_enc.clone()
            if backbone_only:
                order_mask = torch.ones(length, device=device)
                order_mask[position] = 0.0
            else:
                order_mask = torch.zeros(length, device=device)
                order_mask[position] = 1.0
            decode_order = torch.argsort((order_mask[None, :] + 0.0001) * tie_break)
            backward_mask, forward_mask = _autoregressive_masks(decode_order, neighbor_idx, mask)
            forward_context = forward_mask * encoder_context
            for layer in self.decoder_layers:
                decoder_context = cat_neighbors_nodes(node_states, edge_token_states, neighbor_idx)
                node_states = layer(node_states, backward_mask * decoder_context + forward_context, mask)
            logits = self.output_projection(node_states)
            logits_out[:, position, :] = logits[:, position, :]
            log_probs_out[:, position, :] = F.log_softmax(logits, dim=-1)[:, position, :]

        return {"S": sequence, "logits": logits_out, "log_probs": log_probs_out}

    def unconditional_probs(self, features: Mapping[str, Any]) -> dict[str, Tensor]:
        r"""
        Score every residue from backbone context alone in a single decoder pass (upstream ``unconditional_probs``).

        With an all-forward autoregressive mask no position sees any sampled identity, so every residue's logits
        depend only on the encoded backbone and can be computed in one pass rather than position by position.

        Args:
            features: Feature mapping produced by ``_feature_dict``, providing the encoder inputs plus ``S``
                (`[B, L]` reference tokens, returned unchanged) and ``mask`` (`[B, L]`).

        Returns:
            A mapping with ``S`` (the reference tokens, unchanged), and per-position ``logits`` and ``log_probs``
            of shape `[B, L, 21]`.
        """

        sequence = features["S"]
        mask = features["mask"]
        node_states, edge_states, neighbor_idx = self.encode(features)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_idx)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_idx)

        order_mask_backward = torch.zeros(
            sequence.shape[0], sequence.shape[1], sequence.shape[1], device=sequence.device
        )
        attend = torch.gather(order_mask_backward, 2, neighbor_idx).unsqueeze(-1)
        mask_1d = mask.view(mask.shape[0], mask.shape[1], 1, 1)
        forward_context = mask_1d * (1.0 - attend) * encoder_context
        for layer in self.decoder_layers:
            node_states = layer(node_states, forward_context, mask)

        logits = self.output_projection(node_states)
        return {"S": sequence, "logits": logits, "log_probs": F.log_softmax(logits, dim=-1)}


class ProteinBackboneFeatures(nn.Module):
    r"""
    Geometric edge featurizer for protein backbones (upstream ``ProteinFeatures``).

    Builds the k-nearest-neighbor graph from Cα coordinates and encodes each edge from radial-basis distances over
    the 25 backbone atom-pair combinations (including a virtual Cβ) plus a relative-position encoding, then projects
    and normalizes to the edge feature width.

    Args:
        edge_feature_size: Output width of the edge features.
        num_positional_embeddings: Width of the relative-position encoding.
        num_rbf: Number of radial-basis bins per atom-pair distance.
        k_neighbors: Number of nearest neighbors per residue.
        coord_noise_scale: Standard deviation of Gaussian noise added to coords before featurization.
    """

    def __init__(
        self,
        *,
        edge_feature_size: int,
        node_feature_size: int,
        num_positional_embeddings: int,
        num_rbf: int,
        k_neighbors: int,
        coord_noise_scale: float,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.coord_noise_scale = coord_noise_scale
        self.num_rbf = num_rbf
        self.node_feature_size = node_feature_size
        self.relative_position = RelativePositionEncoding(num_positional_embeddings)
        if node_feature_size > 0:
            self.node_projection = nn.Linear(node_feature_size, edge_feature_size, bias=False)
            self.node_norm = nn.LayerNorm(edge_feature_size)
        self.edge_projection = nn.Linear(num_positional_embeddings + 25 * num_rbf, edge_feature_size, bias=False)
        self.edge_norm = nn.LayerNorm(edge_feature_size)

    def node_features(self, features: Mapping[str, Any]) -> Tensor | None:
        r"""
        Project optional membrane labels into initial node features.

        Converted membrane ProteinMPNN checkpoints use a three-channel per-residue label embedding. When labels are
        omitted, zero labels are used so those artifacts remain callable with the standard ProteinMPNN input surface.
        """

        if self.node_feature_size <= 0:
            return None
        labels = features.get("membrane_labels")
        if labels is None:
            labels = features.get("node_features")
        mask = features["mask"]
        if labels is None:
            labels = mask.new_zeros((*mask.shape, self.node_feature_size))
        elif not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=mask.device)
        expected_shape = (*mask.shape, self.node_feature_size)
        if tuple(labels.shape) != expected_shape:
            raise ValueError(f"membrane_labels must have shape {expected_shape}, got {tuple(labels.shape)}")
        labels = labels.to(device=mask.device, dtype=self.edge_projection.weight.dtype)
        return self.node_norm(self.node_projection(labels))

    def forward(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        r"""
        Compute normalized edge features and the neighbor index for a batch of backbones.

        Args:
            features: Feature mapping with backbone coords ``X``, ``mask``, ``residue_index``, and ``chain_index``.

        Returns:
            A ``(edge_features, neighbor_idx)`` tuple with shapes `[B, L, K, edge_feature_size]` and `[B, L, K]`.
        """

        coords = features["X"]
        if self.coord_noise_scale > 0:
            coords = coords + self.coord_noise_scale * _randn_like(coords, features.get("generator"))
        atoms = _backbone_atoms(coords)
        _, neighbor_idx = _knn(atoms["CA"], features["mask"], self.k_neighbors)
        pair_features = _backbone_pair_rbf(atoms, neighbor_idx, self.num_rbf)
        residue_offset = features["residue_index"][:, :, None] - features["residue_index"][:, None, :]
        residue_offset = gather_edges(residue_offset[:, :, :, None], neighbor_idx)[:, :, :, 0]
        same_chain = (features["chain_index"][:, :, None] == features["chain_index"][:, None, :]).long()
        same_chain = gather_edges(same_chain[:, :, :, None], neighbor_idx)[:, :, :, 0]
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_input = torch.cat((position_features, pair_features), dim=-1)
        return self.edge_norm(self.edge_projection(edge_input)), neighbor_idx


def _randn_like(tensor: Tensor, generator: torch.Generator | None) -> Tensor:
    r"""
    Draw standard-normal noise shaped like ``tensor``, using ``generator`` when given.

    Args:
        tensor: Tensor whose shape, dtype, and device the noise matches.
        generator: Optional RNG for reproducible noise.

    Returns:
        Gaussian noise with the same shape/dtype/device as ``tensor``.
    """

    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)


def _validate_proteinmpnn_input(batch: ProteinMPNNInput, *, require_sampling: bool) -> None:
    r"""
    Validate the shape contract of a typed ProteinMPNN input.

    Checks that backbone coords are `[B, L, 4, 3]`, that the sequence/mask/index tensors share `[B, L]`, and that
    the optional conditioning tensors have their expected shapes. Sampling additionally requires a single-backbone
    batch, matching the official contract.

    Args:
        batch: Typed input to validate.
        require_sampling: Whether to also enforce the single-backbone (batch size one) sampling contract.

    Raises:
        ValueError: If any tensor violates its expected shape (or the sampling batch contract).
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
        batch.residue_index.shape != batch.sequence_tokens.shape
        or batch.chain_index.shape != batch.sequence_tokens.shape
    ):
        raise ValueError("residue_index and chain_index must have shape (B, L)")
    if batch.membrane_labels is not None and (
        batch.membrane_labels.ndim != 3 or batch.membrane_labels.shape[:2] != batch.sequence_tokens.shape
    ):
        raise ValueError("membrane_labels must have shape (B, L, C)")
    if batch.design_mask is not None and batch.design_mask.shape != batch.sequence_tokens.shape:
        raise ValueError("design_mask must have shape (B, L)")
    if batch.amino_acid_bias is not None and batch.amino_acid_bias.shape != (*batch.sequence_tokens.shape, 21):
        raise ValueError("amino_acid_bias must have shape (B, L, 21)")
    if batch.decoding_order_noise is not None and (
        batch.decoding_order_noise.ndim != 2 or batch.decoding_order_noise.shape[1] != batch.sequence_tokens.shape[1]
    ):
        raise ValueError("decoding_order_noise must have shape (num_samples, L)")
    if require_sampling and batch.sequence_tokens.shape[0] != 1:
        raise ValueError("ProteinMPNN sampling follows the official single-backbone batch contract")


def _sequence_log_likelihood(sequence: Tensor, log_probs: Tensor, mask: Tensor) -> Tensor:
    r"""
    Compute the masked-mean per-sequence token log-likelihood.

    Gathers the log-probability of each true token and averages over valid positions, tiling ``mask`` when it has
    fewer rows than the (sample-expanded) sequence.

    Args:
        sequence: True token ids `[..., L]`.
        log_probs: Per-position log-probabilities `[..., L, 21]`.
        mask: Valid-residue mask `[..., L]`.

    Returns:
        The masked-mean log-likelihood per sequence `[...]`.
    """

    token_log_probs = torch.gather(log_probs, -1, sequence[..., None]).squeeze(-1)
    mask = mask.to(dtype=token_log_probs.dtype)
    if mask.shape[0] != token_log_probs.shape[0]:
        mask = mask.repeat(token_log_probs.shape[0] // mask.shape[0], 1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _has_no_symmetry(groups: list[list[int]]) -> bool:
    r"""
    Return whether ``groups`` is the sentinel empty-symmetry value (a single empty group).

    Args:
        groups: Symmetry residue groups, or the ``[[]]`` sentinel meaning "no symmetry".

    Returns:
        ``True`` when ``groups`` is the empty-symmetry sentinel.
    """

    return len(groups) == 1 and len(groups[0]) == 0


def _decode_groups(decode_order: Tensor, symmetry_groups: list[list[int]]) -> list[list[int]]:
    r"""
    Expand a per-position decode order into symmetry groups in first-seen order.

    Each position pulls in the whole symmetry group it belongs to (or forms a singleton) the first time it is
    reached in ``decode_order``; already-placed positions are skipped.

    Args:
        decode_order: A 1-D per-position decode order.
        symmetry_groups: Residue groups whose members decode together.

    Returns:
        The groups ordered by their first-seen member in ``decode_order``.
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


def _tied_groups(decode_order: Tensor, tied_positions: list[list[int]]) -> list[list[int]]:
    r"""
    Expand a per-position decode order into tied groups, mirroring upstream ``tied_sample``.

    Each position keeps the first tied group it appears in; untied positions form singleton groups. Group
    order follows the argsorted decode order of the first-seen member.
    """

    ordered: list[list[int]] = []
    flat: set[int] = set()
    for residue in decode_order.detach().cpu().tolist():
        residue = int(residue)
        if residue in flat:
            continue
        group = next((list(candidate) for candidate in tied_positions if residue in candidate), None)
        selected = group if group is not None else [residue]
        ordered.append(selected)
        flat.update(selected)
    return ordered


def _fixed_position_mask(features: Mapping[str, Any], mask: Tensor) -> Tensor:
    r"""
    Return the upstream ``chain_M_pos`` factor (ones when absent) so fixed positions stay hard-fixed.
    """

    fixed = features.get("fixed_position_mask")
    if fixed is None:
        return torch.ones_like(mask)
    return fixed.to(device=mask.device, dtype=mask.dtype)


def _conditioning_bundle(features: Mapping[str, Any]) -> _Conditioning:
    r"""
    Assemble the per-position sampling conditioners from a feature mapping.

    Args:
        features: Feature mapping holding the optional omit-AA / PSSM / per-residue-bias fields.

    Returns:
        A [`_Conditioning`][ofoldx.models.proteinmpnn.model._Conditioning] bundling those fields.
    """

    return _Conditioning(
        omit_amino_acid_mask=features.get("omit_amino_acid_mask"),
        pssm_coefficient=features.get("pssm_coefficient"),
        pssm_bias=features.get("pssm_bias"),
        pssm_log_odds_mask=features.get("pssm_log_odds_mask"),
        pssm_multiplier=float(features.get("pssm_multiplier", 1.0)),
        pssm_log_odds_flag=bool(features.get("pssm_log_odds_flag", False)),
        pssm_bias_flag=bool(features.get("pssm_bias_flag", False)),
        bias_by_residue=features.get("bias_by_residue"),
    )


@dataclass(slots=True)
class _Conditioning:
    r"""
    Per-position sampling conditioners (omit-AA / PSSM / per-residue bias) applied during decoding.

    Reproduces the upstream ``sample``/``tied_sample`` probability post-processing. When every field is unset,
    `apply` collapses to ``softmax((logits + amino_acid_bias) / temperature)`` so untouched calls stay
    numerically identical to the pre-existing sampler.
    """

    omit_amino_acid_mask: Tensor | None = None
    pssm_coefficient: Tensor | None = None
    pssm_bias: Tensor | None = None
    pssm_log_odds_mask: Tensor | None = None
    pssm_multiplier: float = 1.0
    pssm_log_odds_flag: bool = False
    pssm_bias_flag: bool = False
    bias_by_residue: Tensor | None = None

    @classmethod
    def empty(cls) -> _Conditioning:
        r"""
        Return a conditioner with every field unset (a no-op ``apply``).

        Returns:
            A [`_Conditioning`][ofoldx.models.proteinmpnn.model._Conditioning] whose ``apply`` reduces to plain
            biased softmax.
        """

        return cls()

    def repeat(self, num_samples: int) -> _Conditioning:
        r"""
        Tile every per-position tensor ``num_samples`` times along the batch axis (independent sampler).

        Args:
            num_samples: Number of independent samples to broadcast the conditioners across; ``1`` returns ``self``
                unchanged.

        Returns:
            A [`_Conditioning`][ofoldx.models.proteinmpnn.model._Conditioning] whose tensor fields are repeated
            ``num_samples`` times along the batch axis (scalar/flag fields unchanged).
        """

        if num_samples == 1:
            return self

        def tile(tensor: Tensor | None) -> Tensor | None:
            r"""
            Repeat one optional conditioner tensor ``num_samples`` times along the batch axis.

            Args:
                tensor: Optional per-position tensor to tile.

            Returns:
                The tiled tensor, or ``None`` when the input is ``None``.
            """

            if tensor is None:
                return None
            return tensor.repeat(num_samples, *([1] * (tensor.dim() - 1)))

        return _Conditioning(
            omit_amino_acid_mask=tile(self.omit_amino_acid_mask),
            pssm_coefficient=tile(self.pssm_coefficient),
            pssm_bias=tile(self.pssm_bias),
            pssm_log_odds_mask=tile(self.pssm_log_odds_mask),
            pssm_multiplier=self.pssm_multiplier,
            pssm_log_odds_flag=self.pssm_log_odds_flag,
            pssm_bias_flag=self.pssm_bias_flag,
            bias_by_residue=tile(self.bias_by_residue),
        )

    def apply(self, logits: Tensor, amino_acid_bias: Tensor, temperature: float | Tensor, position: Tensor) -> Tensor:
        r"""
        Build the sampling distribution from raw logits plus the per-position ``amino_acid_bias`` term.

        Adds the caller's bias (and any per-residue bias), applies the temperature softmax, then runs the shared
        PSSM/omit-AA post-processing. Used by the untied ``sample`` path.

        Args:
            logits: Raw decoder logits `[B, 21]` at the decoded position.
            amino_acid_bias: Per-position amino-acid logit bias `[B, 21]` supplied by the caller.
            temperature: Sampling temperature dividing the combined logits (scalar or `[B, 1]`).
            position: Scalar/int or per-batch index selecting the conditioner rows.

        Returns:
            The post-processed sampling probabilities `[B, 21]`.
        """

        combined = logits + amino_acid_bias
        if self.bias_by_residue is not None:
            combined = combined + _gather_position(self.bias_by_residue, position)
        probs = F.softmax(combined / temperature, dim=-1)
        return self._post(probs, position)

    def apply_logits(self, scaled_logits: Tensor, temperature: float | Tensor, position: Tensor | int) -> Tensor:
        r"""
        Build the sampling distribution from already temperature-scaled logits (tied path).

        The tied sampler pre-divides its group logits by temperature, so this variant skips the temperature softmax
        scaling, folding in any per-residue bias (divided by temperature) before the softmax and post-processing.

        Args:
            scaled_logits: Group logits `[B, 21]` already divided by temperature.
            temperature: Sampling temperature used to scale the per-residue bias term (scalar or `[B, 1]`).
            position: Scalar/int or per-batch index selecting the conditioner rows.

        Returns:
            The post-processed sampling probabilities `[B, 21]`.
        """

        combined = scaled_logits
        if self.bias_by_residue is not None:
            combined = combined + _gather_position(self.bias_by_residue, position) / temperature
        probs = F.softmax(combined, dim=-1)
        return self._post(probs, position)

    def _post(self, probs: Tensor, position: Tensor | int) -> Tensor:
        r"""
        Apply the PSSM and omit-AA post-processing to a sampling distribution at ``position``.

        In upstream order: PSSM probability mixing, PSSM log-odds gating, then forbidden-amino-acid masking, each
        renormalizing as needed. Fields that are unset are skipped.

        Args:
            probs: Sampling probabilities `[B, 21]` at the decoded position.
            position: Scalar/int or per-batch index selecting the conditioner rows.

        Returns:
            The post-processed probabilities `[B, 21]`.
        """

        if self.pssm_bias_flag and self.pssm_coefficient is not None and self.pssm_bias is not None:
            coefficient = _gather_position(self.pssm_coefficient.unsqueeze(-1), position)
            bias = _gather_position(self.pssm_bias, position)
            probs = (1 - self.pssm_multiplier * coefficient) * probs + self.pssm_multiplier * coefficient * bias
        if self.pssm_log_odds_flag and self.pssm_log_odds_mask is not None:
            keep = _gather_position(self.pssm_log_odds_mask, position)
            masked = probs * keep + probs * 0.001
            probs = masked / masked.sum(dim=-1, keepdim=True)
        if self.omit_amino_acid_mask is not None:
            forbid = _gather_position(self.omit_amino_acid_mask, position)
            masked = probs * (1.0 - forbid)
            probs = masked / masked.sum(dim=-1, keepdim=True)
        return probs


def _gather_position(tensor: Tensor, position: Tensor | int) -> Tensor:
    r"""
    Select the per-batch row(s) at ``position`` from a ``(B, L, C)`` tensor, returning ``(B, C)``.

    ``position`` is either a scalar/int (symmetric and tied paths) or a per-batch index tensor (independent path).
    """

    if isinstance(position, int) or position.dim() == 0:
        return tensor[:, int(position), :]
    index = position[:, None, None].expand(-1, 1, tensor.shape[-1])
    return torch.gather(tensor, 1, index)[:, 0, :]


def _autoregressive_masks(decode_order: Tensor, neighbor_idx: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Build the per-neighbor backward (already-decoded) and forward (not-yet-decoded) masks.

    From the decode-order permutation, marks for each residue which neighbors precede it in decoding, then splits
    the valid-residue mask into a backward part (decoder context) and a forward part (encoder-only context).

    Args:
        decode_order: Per-batch decode order `[B, L]`.
        neighbor_idx: Neighbor index `[B, L, K]`.
        mask: Valid-residue mask `[B, L]`.

    Returns:
        A ``(backward_mask, forward_mask)`` tuple, each `[B, L, K, 1]`, that partition neighbor messages.
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


def _backbone_atoms(coords: Tensor) -> dict[str, Tensor]:
    r"""
    Split N/CA/C/O backbone coords into named atoms and add the idealized virtual Cβ.

    The Cβ position is reconstructed from N, Cα, and C with the upstream fixed geometric coefficients.

    Args:
        coords: Backbone coordinates `[B, L, 4, 3]` ordered N, CA, C, O.

    Returns:
        A mapping ``{"N", "CA", "C", "O", "CB"}`` of `[B, L, 3]` coordinate tensors.
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
    r"""
    Find the ``k_neighbors`` nearest valid points per residue by Euclidean distance.

    Distances to masked-out points are pushed to the per-row maximum so padded residues are never selected as
    neighbors.

    Args:
        points: Reference points `[B, L, 3]` (Cα coordinates).
        mask: Valid-residue mask `[B, L]`.
        k_neighbors: Number of neighbors to return (capped at ``L``).
        eps: Small constant added under the square root for numerical stability.

    Returns:
        A ``(distances, neighbor_idx)`` tuple of `[B, L, K]` tensors from ``torch.topk`` (smallest first).
    """

    pair_mask = mask[:, None, :] * mask[:, :, None]
    distances = torch.sqrt(torch.sum((points[:, :, None] - points[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    max_distance = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * max_distance
    return torch.topk(adjusted, min(k_neighbors, points.shape[1]), dim=-1, largest=False)


def _rbf(distances: Tensor, num_bins: int) -> Tensor:
    r"""
    Expand distances into Gaussian radial-basis features over ``num_bins`` centers in [2, 22] angstroms.

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


def _gather_pair_rbf(source: Tensor, target: Tensor, neighbor_idx: Tensor, num_bins: int) -> Tensor:
    r"""
    RBF-encode the distance from each residue's ``source`` atom to its neighbors' ``target`` atom.

    Args:
        source: Source atom coordinates `[B, L, 3]`.
        target: Target atom coordinates `[B, L, 3]`.
        neighbor_idx: Neighbor index `[B, L, K]`.
        num_bins: Number of RBF centers.

    Returns:
        Per-neighbor RBF features `[B, L, K, num_bins]`.
    """

    distances = torch.sqrt(torch.sum((source[:, :, None] - target[:, None, :]) ** 2, dim=-1) + 1e-6)
    neighbor_distances = gather_edges(distances[:, :, :, None], neighbor_idx)[:, :, :, 0]
    return _rbf(neighbor_distances, num_bins)


def _backbone_pair_rbf(atoms: Mapping[str, Tensor], neighbor_idx: Tensor, num_bins: int) -> Tensor:
    r"""
    Concatenate RBF distance features over the 25 ordered backbone atom-pair combinations.

    Args:
        atoms: Named backbone atom coordinates from
            [`_backbone_atoms`][ofoldx.models.proteinmpnn.model._backbone_atoms].
        neighbor_idx: Neighbor index `[B, L, K]`.
        num_bins: Number of RBF centers per atom pair.

    Returns:
        Stacked pairwise RBF features `[B, L, K, 25 * num_bins]`.
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


@register_pytree_dataclass
@dataclass(slots=True, kw_only=True)
class ProteinMPNNInput(BaseInput):
    r"""
    Batched backbone/sequence tensors and sampler controls consumed by the native ProteinMPNN protocols.

    Attributes:
        backbone_coords: Backbone N/CA/C/O coords with shape ``(B, L, 4, 3)``.
        sequence_tokens: Residue tokens in the ProteinMPNN alphabet with shape ``(B, L)``.
        token_mask: Valid-residue mask with shape ``(B, L)``.
        residue_index: Residue numbers used for relative-position features with shape ``(B, L)``.
        chain_index: Integer chain ids with shape ``(B, L)``.
        membrane_labels: Optional per-residue membrane labels with shape ``(B, L, node_feature_size)`` for converted
            membrane ProteinMPNN checkpoints. Omit it to use zero membrane labels.
        design_mask: Optional mask where one marks positions to design and zero marks fixed positions.
        amino_acid_bias: Optional per-position amino-acid logit bias with shape ``(B, L, 21)``.
        decoding_order_noise: Optional decoding-order noise with shape ``(num_samples, L)``.
        temperature: Sampling temperature used by ``sample`` and ``generate``.
        num_samples: Number of autoregressive samples or decode orders.
        symmetry_residue_groups: Residue groups whose sampled identity is tied during native symmetric decoding.
        symmetry_weights: Per-position logit weights aligned with ``symmetry_residue_groups``.
        tied_positions: Official tied-decoding groups (upstream ``tied_positions_dict``) sampled by ``tied_sample``.
        tied_weights: Per-position ``tied_beta`` logit weights used by ``tied_sample`` with shape ``(L,)``.
        fixed_position_mask: Upstream ``chain_M_pos``; one for designable positions and zero for hard-fixed ones.
        omit_amino_acid_mask: Per-position amino-acids to forbid with shape ``(B, L, 21)`` (upstream ``omit_AA_mask``).
        pssm_coefficient: Per-position PSSM mixing coefficient with shape ``(B, L)`` (upstream ``pssm_coef``).
        pssm_bias: Per-position PSSM probability bias with shape ``(B, L, 21)`` (upstream ``pssm_bias``).
        pssm_log_odds_mask: Per-position PSSM log-odds keep mask with shape ``(B, L, 21)``.
        pssm_multiplier: Global PSSM interpolation strength (upstream ``pssm_multi``).
        pssm_log_odds_flag: Enable PSSM log-odds gating during sampling.
        pssm_bias_flag: Enable PSSM probability biasing during sampling.
        bias_by_residue: Per-position amino-acid logit bias with shape ``(B, L, 21)`` (upstream ``bias_by_res``).
    """

    backbone_coords: Tensor
    sequence_tokens: Tensor
    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    membrane_labels: Tensor | None = None
    design_mask: Tensor | None = None
    amino_acid_bias: Tensor | None = None
    decoding_order_noise: Tensor | None = None
    temperature: float | Tensor = 0.1
    num_samples: int | None = None
    symmetry_residue_groups: list[list[int]] | None = None
    symmetry_weights: list[list[float]] | None = None
    tied_positions: list[list[int]] | None = None
    tied_weights: Tensor | None = None
    fixed_position_mask: Tensor | None = None
    omit_amino_acid_mask: Tensor | None = None
    pssm_coefficient: Tensor | None = None
    pssm_bias: Tensor | None = None
    pssm_log_odds_mask: Tensor | None = None
    pssm_multiplier: float = 1.0
    pssm_log_odds_flag: bool = False
    pssm_bias_flag: bool = False
    bias_by_residue: Tensor | None = None


# ProteinMPNNInput is defined after ProteinMPNNDesigner, so bind the keyword-tensor call path's input class here
# (see ModelMixin.__call__, which builds ``self.input_class(**kwargs)`` for ``model(**inputs)``).
ProteinMPNNDesigner.input_class = ProteinMPNNInput
