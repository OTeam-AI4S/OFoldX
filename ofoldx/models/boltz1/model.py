r"""
Boltz-1 model.

This module implements the native Boltz-1 structure-prediction path: token/atom input embedding, recycling
trunk, optional MSA update, distogram head, EDM coordinate sampling, and the released Boltz-1 confidence/ranking
head family.

The family is registered as ``ofoldx.boltz1``: alongside the differentiable ``forward`` training graph it
exposes the public ``fold`` task, which runs the released reverse-diffusion path and decodes a self-contained
canonical ``FoldResult`` from the raw ``Boltz1Output`` produced by its ``Boltz1Processor`` lowering.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from typing_extensions import Self

from ...data.molecular import AtomSite, ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.atom.encoders import PackedAtomAttention
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.transfer import TokenPairToLocalAtomPairBroadcast, TokenToAtomBroadcast, dense_mean_to_tokens
from ...modules.atom.window import gather_key_windows
from ...modules.attention import GatedMultiheadAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.diffusion.conditioning import condition_pair, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import DistogramHead
from ...modules.head.tm import tm_weight_from_pae
from ...modules.transformer import ConditionedTransformer
from ...modules.transition import Transition
from ...modules.trunk.msa import MsaModule, MsaPairWeightedAveraging, OuterProductMean
from ...modules.trunk.pairformer import PairformerStack
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import Boltz1Config


class Boltz1Model(nn.Module):
    r"""
    Predict all-atom structure from tokenized complex features via the recycling trunk and EDM diffusion sampler.

    Runs the full Boltz-1 path: token/atom input embedding, a recycling representation trunk (Pairformer plus
    an optional MSA module), a distogram head, EDM reverse-diffusion coordinate sampling, and the released
    Boltz-1 confidence/ranking head. Optional Boltz-1x inference-time steering (Feynman-Kac particles and
    physical-potential guidance) is available at sampling time.

    Args:
        config: The Boltz-1 configuration selecting representation widths, block counts, recycling, the
            sampler schedule, and which optional heads (MSA, confidence) to build.

    """

    config_class = Boltz1Config

    def __init__(self, config: Boltz1Config | None = None) -> None:
        config = Boltz1Config() if config is None else config
        super().__init__()
        self.config = config
        self._recycling_steps = config.recycling_steps

        single_size, pair_size = config.single_size, config.pair_size
        embedder = config.input_embedder
        input_size = single_size + 2 * embedder.vocab_size + 1 + embedder.pocket_feature_size
        trunk = config.trunk
        diffusion = config.diffusion

        self.input_embedder = Boltz1InputEmbedder(
            input_size=input_size,
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=embedder.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=embedder.num_atom_encoder_attention_heads,
            no_atom_encoder=embedder.no_atom_encoder,
            operator_backend="auto",
        )

        self.single_init_projection = nn.Linear(input_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)

        self.single_recycle_norm = LayerNorm(single_size)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        self.msa_module: MsaModule | None = None
        if not config.no_msa:
            self.msa_module = MsaModule(
                single_size=input_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                num_blocks=trunk.num_msa_blocks,
                vocab_size=embedder.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                operator_backend="auto",
            )
        self.pairformer_stack = PairformerStack(
            num_blocks=trunk.num_pairformer_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
            num_triangle_attention_heads=trunk.num_triangle_attention_heads,
            num_pair_attention_heads=trunk.num_pair_attention_heads,
            dropout=trunk.dropout,
            operator_backend="auto",
        )
        self.distogram_head = DistogramHead(
            pair_size=pair_size,
            num_bins=config.num_distogram_bins,
            symmetrize_before_projection=True,
        )

        self.diffusion_conditioning = DiffusionConditioning(
            single_size=single_size,
            pair_size=pair_size,
            atom_hidden_size=diffusion.atom_hidden_size,
            atom_pair_hidden_size=diffusion.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=diffusion.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=diffusion.num_atom_attention_heads,
            num_atom_decoder_blocks=diffusion.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=diffusion.num_atom_attention_heads,
            num_token_transformer_blocks=diffusion.num_token_transformer_blocks,
            num_token_transformer_heads=diffusion.num_token_transformer_heads,
            num_conditioning_transitions=diffusion.num_conditioning_transitions,
        )
        score_model = Boltz1DiffusionScoreModel(
            input_size=input_size,
            single_size=single_size,
            atom_hidden_size=diffusion.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=diffusion.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=diffusion.num_atom_attention_heads,
            num_token_transformer_blocks=diffusion.num_token_transformer_blocks,
            num_token_transformer_heads=diffusion.num_token_transformer_heads,
            num_atom_decoder_blocks=diffusion.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=diffusion.num_atom_attention_heads,
            num_conditioning_transitions=diffusion.num_conditioning_transitions,
            operator_backend="auto",
        )
        sampler = config.sampler
        self.structure_module = Boltz1AtomDiffusion(
            score_model,
            sigma_data=diffusion.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            min_churn_noise_level=sampler.min_churn_noise_level,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=diffusion.alignment_reverse_diff,
            accumulate_token_repr=config.confidence.enabled and config.confidence.use_single_diffusion,
            token_size=single_size,
            fourier_embedding_size=config.fourier_embedding_size,
        )

        confidence = config.confidence
        self.confidence_head: Boltz1ConfidenceHead | None = None
        if confidence.enabled:
            self.confidence_head = Boltz1ConfidenceHead(
                input_size=input_size,
                single_size=single_size,
                pair_size=pair_size,
                atom_hidden_size=config.atom_hidden_size,
                atom_pair_hidden_size=config.atom_pair_hidden_size,
                atom_feature_size=config.atom_feature_size,
                num_query_atoms=diffusion.num_query_atoms,
                num_key_atoms=diffusion.num_key_atoms,
                num_atom_encoder_blocks=embedder.num_atom_encoder_blocks,
                num_atom_encoder_attention_heads=embedder.num_atom_encoder_attention_heads,
                num_blocks=confidence.num_pairformer_blocks,
                num_pair_attention_heads=confidence.num_pair_attention_heads,
                num_distogram_bins=confidence.num_distogram_bins,
                max_distance=confidence.max_distance,
                num_plddt_bins=confidence.num_plddt_bins,
                num_pae_bins=confidence.num_pae_bins,
                num_pde_bins=confidence.num_pde_bins,
                compute_pae=confidence.compute_pae,
                imitate_trunk=confidence.imitate_trunk,
                add_single_to_pair_product=confidence.add_single_to_pair_product,
                add_single_input_to_single=confidence.add_single_input_to_single,
                add_pair_input_to_pair=confidence.add_pair_input_to_pair,
                use_single_diffusion=confidence.use_single_diffusion,
                ranking_plddt_weight=confidence.ranking_plddt_weight,
                ranking_ptm_weight=confidence.ranking_ptm_weight,
                no_atom_encoder=embedder.no_atom_encoder,
                no_msa=config.no_msa,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                vocab_size=embedder.vocab_size,
                msa_blocks=trunk.num_msa_blocks,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                relative_position_size=config.relative_position_size,
                operator_backend="auto",
            )
        self.set_backend("auto")

    def set_backend(self, backend: str) -> Self:
        r"""Select the operator implementation used by this model."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend
        attention_backend = "eager" if backend == "torch" else backend
        operator_backend = "eager" if backend == "torch" else backend
        for module in self.modules():
            if isinstance(module, GatedMultiheadAttention):
                module.torch_backend = "eager"
                module.operator_backend = attention_backend
            elif isinstance(
                module,
                (
                    AdaptiveLayerNorm,
                    Transition,
                    TriangleMultiplication,
                    MsaPairWeightedAveraging,
                    OuterProductMean,
                    TokenToAtomBroadcast,
                    TokenPairToLocalAtomPairBroadcast,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def forward(
        self,
        batch: Boltz1Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Boltz1Output:
        r"""
        Run the Boltz-1 training path: one single-step diffusion denoise of the noised ground truth.

        ``forward`` is the structure-training graph in both ``train()`` and ``eval()`` modes; those modes only
        control ordinary stochastic layers such as dropout. It runs the recycling trunk (with gradient), then
        denoises the noised, augmented ground-truth coordinates in a single EDM step, returning the tensors the
        EDM-weighted coordinate loss scores (``denoised_coords`` against ``augmented_target`` at
        ``noise_level``) alongside the trunk distogram logits. The full multi-step reverse-diffusion inference
        path is exposed separately by the public ``fold`` task.

        Args:
            batch: The Boltz-1 input batch; training requires ``batch.target_positions``.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
            num_samples: Number of independently noised training samples to draw per input.
            generator: Optional RNG for reproducible augmentation and noise.

        Returns:
            A [`Boltz1Output`][ofoldx.models.boltz1.model.Boltz1Output] carrying ``denoised_coords``,
            ``augmented_target``, and ``noise_level`` for the diffusion loss
            alongside the distogram logits.

        """

        representations = self._trunk(batch, recycling_steps=recycling_steps)
        conditioning = self._diffusion_conditioning(representations, batch)
        denoised_coords, augmented_target, noise_level = self.structure_module.denoise_training(
            batch.target_positions,
            batch.atom_mask,
            target_mask=batch.target_mask,
            multiplicity=num_samples,
            generator=generator,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=batch.atom_to_token,
            token_mask=batch.token_mask,
        )
        return Boltz1Output(
            pred_coords=denoised_coords,
            distogram_logits=representations.distogram_logits,
            denoised_coords=denoised_coords,
            augmented_target=augmented_target,
            noise_level=noise_level,
        )

    @torch.inference_mode()
    def fold(
        self,
        batch: Boltz1Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> FoldResult:
        r"""
        Run released Boltz-1 folding and decode a self-contained canonical ``FoldResult``.

        Runs the recycling trunk, draws ``num_samples`` reverse-diffusion samples, scores each with the
        confidence/ranking head, and wraps the raw
        [`Boltz1Output`][ofoldx.models.boltz1.model.Boltz1Output] into a
        [`FoldResult`][ofoldx.results.FoldResult]: one generation-ordered
        [`FoldPrediction`][ofoldx.results.FoldPrediction] per sample, carrying the sampled ``Structure``,
        per-token ``metrics`` (``plddt``, ``resolved_probability``), token-pair ``pair_metrics``
        (``pae``, ``pde``), and per-sample ``scores`` (``plddt``, ``ptm``, ``iptm``, ``confidence_score``, ...).

        Args:
            batch: One lowered Boltz-1 input carrying the canonical
                [`Boltz1Layout`][ofoldx.models.boltz1.model.Boltz1Layout] produced by ``Boltz1Processor``.
            num_samples: Number of reverse-diffusion samples to draw.
            generator: Optional RNG for reproducible sampling noise; resolved once at the Pipeline boundary.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            use_potentials: Whether to enable Boltz-1x physical-potential steering.
            steering_args: Optional overrides merged onto the configured steering flags.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] with one prediction per requested sample.

        """

        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if batch.token_mask.shape[0] != 1:
            raise ValueError("Boltz1Model.fold accepts one molecular system per call")
        if batch.layout is None:
            raise ValueError("Boltz1Model.fold requires the canonical layout produced by its Processor")

        output = self._predict(
            batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
        return _decode_boltz1_fold_result(output, batch)

    def embeddings(
        self,
        batch: Boltz1Input,
        *,
        recycling_steps: int | None = None,
    ) -> Boltz1Embeddings:
        r"""
        Surface the released ``--write_embeddings`` trunk representations.

        Runs the same recycling trunk as ``fold`` and returns the raw trunk single ``[B, N, 384]`` and pair
        ``[B, N, N, 128]`` tensors on the Model's computation device and dtype. Unlike ``fold`` this method is
        not wrapped in inference mode: it retains autograd when called under grad. The raw representations carry
        vocabulary/layout axes excluded from canonical ``Metric``/``PairMetric``, so they are deliberately not
        folded into ``FoldResult``; this typed expert method is the only surface on which they are reachable.

        Args:
            batch: One lowered Boltz-1 input batch.
            recycling_steps: Total number of trunk passes; defaults to the configured value.

        Returns:
            A [`Boltz1Embeddings`][ofoldx.models.boltz1.model.Boltz1Embeddings] with the trunk ``single``
            and ``pair`` representations.

        """

        representations = self._trunk(batch, recycling_steps=recycling_steps)
        return Boltz1Embeddings(single=representations.single, pair=representations.pair)

    def _predict(
        self,
        batch: Boltz1Input,
        *,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> Boltz1Output:
        r"""
        Run the trunk and sampler and return the full Boltz-1 output.

        Runs the recycling trunk once, draws ``num_samples`` reverse-diffusion samples, and (when the
        confidence head is present) scores them into confidence logits and ranking metrics. Sample-axis
        tensors are unflattened to a leading ``(batch, num_samples, ...)`` shape. This private tensor path
        does not declare a public fold task.

        Args:
            batch: The Boltz-1 input batch.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            num_samples: Number of reverse-diffusion samples to draw per input.
            generator: Optional RNG for reproducible sampling noise.
            use_potentials: Whether to enable Boltz-1x physical-potential steering.
            steering_args: Optional overrides merged onto the configured steering flags.

        Returns:
            A [`Boltz1Output`][ofoldx.models.boltz1.model.Boltz1Output] with predicted coords, distogram
            logits, optional confidence logits, and optional derived ranking scores.

        """

        representations = self._trunk(batch, recycling_steps=recycling_steps)
        diffusion_coords, diffusion_token_repr = self._sample_diffusion(
            representations,
            batch,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
        batch_size = representations.single.shape[0]
        confidence_logits: dict[str, Tensor | None] = {
            "plddt_logits": None,
            "pae_logits": None,
            "pde_logits": None,
            "resolved_logits": None,
        }
        confidence_scores: dict[str, Tensor] | None = None
        confidence_head = self.confidence_head
        if confidence_head is not None:
            confidence = confidence_head(
                representations.single_inputs,
                representations.single,
                representations.pair,
                diffusion_coords,
                batch=batch,
                pred_distogram_logits=representations.distogram_logits,
                single_diffusion=diffusion_token_repr,
                multiplicity=num_samples,
            )
            confidence_logits = {
                name: confidence.get(name) for name in ("plddt_logits", "pae_logits", "pde_logits", "resolved_logits")
            }
            confidence_scores = {
                name: _unflatten_sample_axis(value, batch_size, num_samples)
                for name, value in confidence.items()
                if name not in confidence_logits
            }
        return Boltz1Output(
            pred_coords=_unflatten_sample_axis(diffusion_coords, batch_size, num_samples),
            distogram_logits=representations.distogram_logits,
            plddt_logits=_unflatten_optional_sample_axis(
                cast(Tensor | None, confidence_logits["plddt_logits"]), batch_size, num_samples
            ),
            pae_logits=_unflatten_optional_sample_axis(
                cast(Tensor | None, confidence_logits["pae_logits"]), batch_size, num_samples
            ),
            pde_logits=_unflatten_optional_sample_axis(
                cast(Tensor | None, confidence_logits["pde_logits"]), batch_size, num_samples
            ),
            resolved_logits=_unflatten_optional_sample_axis(
                cast(Tensor | None, confidence_logits["resolved_logits"]), batch_size, num_samples
            ),
            confidence_scores=confidence_scores,
        )

    def _trunk(self, batch: Boltz1Input, *, recycling_steps: int | None = None) -> _Boltz1Representations:
        r"""
        Iterate the recycling loop: recycle-project the running single/pair, fold in the MSA module (when
        present), and run the Pairformer, for ``recycling_steps + 1`` cycles.

        Args:
            batch: The Boltz-1 input batch.
            recycling_steps: Number of extra recycles beyond the first pass; defaults to the configured value.

        Returns:
            The final trunk [`_Boltz1Representations`][ofoldx.models.boltz1.model._Boltz1Representations].

        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if type(recycling_steps) is not int or recycling_steps < 1:
            raise ValueError("recycling_steps must be a positive integer")
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(batch)
        token_mask = batch.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if batch.msa is not None:
                    pair = pair + cast(MsaModule, self.msa_module)(
                        pair,
                        single_inputs,
                        msa=batch.msa,
                        has_deletion=cast(Tensor, batch.has_deletion),
                        deletion_value=cast(Tensor, batch.deletion_value),
                        is_paired=cast(Tensor, batch.msa_paired),
                        msa_mask=cast(Tensor, batch.msa_mask),
                        token_mask=token_mask,
                    )
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        return _Boltz1Representations(
            single_inputs=single_inputs,
            single=cast(Tensor, single),
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
        )

    def _initial_representations(self, batch: Boltz1Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Embed inputs and build the pre-recycle single/pair initializations.

        Runs the input embedder, projects single-inputs into the single and (outer-sum) pair tracks, and adds
        the relative-position encoding and token-bond bias to the pair init.

        Args:
            batch: The Boltz-1 input batch.

        Returns:
            Tuple ``(single_inputs, single_init, pair_init, relative_position_encoding)``.
        """

        single_inputs = self.input_embedder(
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
            residue_type=batch.residue_type,
            msa_profile=batch.msa_profile,
            deletion_mean=batch.deletion_mean,
            pocket_feature=batch.pocket_feature,
        )

        single_init = self.single_init_projection(single_inputs)
        pair_init = self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair_init = pair_init + self.pair_right_projection(single_inputs).unsqueeze(-3)

        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            cyclic_period=batch.cyclic_period,
            chain_gate="not_same_chain",
            dtype=single_init.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(batch.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        return single_inputs, single_init, pair_init, relative_position_encoding

    def _sample_diffusion(
        self,
        representations: _Boltz1Representations,
        batch: Boltz1Input,
        *,
        num_sampling_steps: int | None,
        num_samples: int,
        generator: torch.Generator | None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        r"""
        Precompute the diffusion conditioning and run the EDM sampler.

        Args:
            representations: The trunk representations from
                [`_trunk`][ofoldx.models.boltz1.model.Boltz1Model._trunk].
            batch: The Boltz-1 input batch.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            num_samples: Number of samples (sampler ``multiplicity``).
            generator: Optional RNG for reproducible sampling noise.
            use_potentials: Whether to enable Boltz-1x physical-potential steering.
            steering_args: Optional overrides merged onto the configured steering flags.

        Returns:
            Sampled coords and, when confidence single-diffusion is enabled, the accumulated token representation.
        """

        conditioning = self._diffusion_conditioning(representations, batch)
        steering = {} if steering_args is None else dict(steering_args)
        if use_potentials:
            steering["fk_steering"] = True
            steering["physical_guidance_update"] = True
        steering.setdefault("physical_guidance_update", False)
        steering.setdefault("fk_steering", False)
        steering.setdefault("num_particles", 3)
        steering.setdefault("fk_lambda", 4.0)
        steering.setdefault("fk_resampling_interval", 3)
        steering.setdefault("num_gd_steps", 16)
        return_token_repr = self.confidence_head is not None and self.config.confidence.use_single_diffusion
        sample = self.structure_module._sample(
            batch.atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=num_samples,
            generator=generator,
            return_token_repr=return_token_repr,
            steering_args=steering,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=batch.atom_to_token,
            token_mask=batch.token_mask,
            feats=_boltz1_steering_features(batch, atom_features=batch.atom_features),
        )
        if return_token_repr:
            return cast(tuple[Tensor, Tensor], sample)
        return cast(Tensor, sample), None

    def _diffusion_conditioning(self, representations: _Boltz1Representations, batch: Boltz1Input) -> dict[str, Tensor]:
        r"""
        Precompute the per-structure diffusion conditioning reused by every denoiser call.

        The conditioning depends only on the trunk representations and reference geometry, so it is computed
        once and shared by the training single-step denoise and every reverse-diffusion sampling step.

        Args:
            representations: The trunk representations from
                [`_trunk`][ofoldx.models.boltz1.model.Boltz1Model._trunk].
            batch: The Boltz-1 input batch.

        Returns:
            The precomputed diffusion-conditioning dict (atom states and per-block attention biases).
        """

        return self.diffusion_conditioning(
            representations.single,
            representations.pair,
            representations.relative_position_encoding,
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
        )


class Boltz1InputEmbedder(nn.Module):
    r"""
    Build the Boltz-1 token input embedding from atom-attention outputs and token-level features.

    Runs the atom encoder and atom-attention encoder to pool a per-token representation, then concatenates it
    with the residue-type one-hot, MSA profile, deletion mean, and pocket one-hot to form the wide
    ``single_inputs`` vector. When ``no_atom_encoder`` is set, the atom path is skipped and its slice is zeroed.

    Args:
        input_size: Expected width of the concatenated ``single_inputs`` output.
        single_size: Width of the token-from-atom slice.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        atom_feature_size: Expected width of the raw per-atom feature vector.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        no_atom_encoder: When ``True``, skip the atom encoder and emit a zeroed token-from-atom slice.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 389,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        no_atom_encoder: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.single_size = single_size
        self.atom_encoder: AtomEncoder | None = None
        self.atom_pair_bias_projection: PerLayerPairBiasProjection | None = None
        self.atom_attention_encoder: AtomAttentionEncoder | None = None
        if not no_atom_encoder:
            self.atom_encoder = AtomEncoder(
                atom_feature_size=atom_feature_size,
                atom_hidden_size=atom_hidden_size,
                atom_pair_hidden_size=atom_pair_hidden_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
            )
            self.atom_pair_bias_projection = PerLayerPairBiasProjection(
                pair_size=atom_pair_hidden_size,
                num_layers=num_atom_encoder_blocks,
                num_attention_heads=num_atom_encoder_attention_heads,
            )
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_hidden_size=atom_hidden_size,
                single_size=single_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                num_atom_encoder_blocks=num_atom_encoder_blocks,
                num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
                operator_backend=operator_backend,
            )

    def forward(
        self,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        residue_type: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        pocket_feature: Tensor | None = None,
    ) -> Tensor:
        r"""
        Assemble the wide ``single_inputs`` token embedding for one batch.

        Args:
            atom_features: `[B, M, atom_feature_size]` raw per-atom features.
            reference_positions: `[B, M, 3]` reference (template) atom positions.
            atom_mask: `[B, M]` atom validity mask.
            reference_space_id: `[B, M]` per-atom reference-space identifier for window gating.
            atom_to_token: `[B, M]` atom-to-token index map.
            residue_type: `[B, N, vocab_size]` residue-type one-hot.
            msa_profile: `[B, N, vocab_size]` MSA profile feature.
            deletion_mean: `[B, N]` mean-deletion feature.
            pocket_feature: Optional `[B, N, pocket_feature_size]` pocket one-hot; defaults to all-``UNSPECIFIED``.

        Returns:
            The `[B, N, input_size]` concatenated ``single_inputs`` token embedding.

        """

        atom_encoder = self.atom_encoder
        atom_pair_bias_projection = self.atom_pair_bias_projection
        atom_attention_encoder = self.atom_attention_encoder
        if atom_encoder is None:
            token_from_atom = torch.zeros(
                (*residue_type.shape[:-1], self.single_size),
                device=residue_type.device,
                dtype=residue_type.dtype,
            )
        else:
            atom_query, atom_conditioning, atom_pair = atom_encoder(
                atom_features,
                reference_positions,
                atom_mask,
                reference_space_id,
            )
            token_from_atom = cast(AtomAttentionEncoder, atom_attention_encoder)(
                atom_query,
                atom_conditioning,
                cast(PerLayerPairBiasProjection, atom_pair_bias_projection)(atom_pair),
                atom_mask,
                atom_to_token,
                residue_type.shape[-2],
            )
        pocket_size = self.input_size - self.single_size - 2 * residue_type.shape[-1] - 1
        if pocket_feature is None:
            pocket_feature = residue_type.new_zeros(*residue_type.shape[:-1], pocket_size)
            pocket_feature[..., 0] = 1
        single = torch.cat(
            [
                token_from_atom,
                residue_type.to(token_from_atom.dtype),
                msa_profile.to(token_from_atom.dtype),
                deletion_mean.unsqueeze(-1).to(token_from_atom.dtype),
                pocket_feature.to(token_from_atom.dtype),
            ],
            dim=-1,
        )
        return single


class AtomEncoder(nn.Module):
    r"""
    Build atom single states and windowed atom-pair states from reference geometry.

    Projects raw atom features to a single state and encodes windowed atom-pair states from reference
    inter-atomic distance vectors, inverse distances, and same-reference-space validity. When
    ``structure_prediction`` is set, it additionally injects trunk single/pair representations gathered per
    atom so the diffusion atom encoder is conditioned on the folded trunk.

    Args:
        atom_feature_size: Width of the raw per-atom feature vector.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        single_size: Width of the trunk single representation (used when ``structure_prediction``).
        pair_size: Width of the trunk pair representation (used when ``structure_prediction``).
        num_query_atoms: Query-window size for the windowed atom-pair encoding.
        num_key_atoms: Key-window size for the windowed atom-pair encoding.
        structure_prediction: When ``True``, add the trunk single/pair conditioning projections.
    """

    def __init__(
        self,
        atom_feature_size: int = 388,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        single_size: int = 384,
        pair_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        structure_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.single_trunk_projection: nn.Sequential | None = None
        self.pair_trunk_projection: nn.Sequential | None = None
        if structure_prediction:
            self.single_trunk_projection = nn.Sequential(
                LayerNorm(single_size), nn.Linear(single_size, atom_hidden_size, bias=False)
            )
            self.pair_trunk_projection = nn.Sequential(
                LayerNorm(pair_size),
                nn.Linear(pair_size, atom_pair_hidden_size, bias=False),
            )
        self.left_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.right_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.feed_forward = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
        )

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        *,
        atom_to_token: Tensor | None = None,
        single_trunk: Tensor | None = None,
        pair_trunk: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Encode atom single states and windowed atom-pair states from reference geometry.

        Args:
            atom_features: `[B, M, atom_feature_size]` raw per-atom features.
            reference_positions: `[B, M, 3]` reference atom positions.
            atom_mask: `[B, M]` atom validity mask.
            reference_space_id: `[B, M]` per-atom reference-space identifier for window gating.
            atom_to_token: `[B, M]` atom-to-token index map; required when ``structure_prediction``.
            single_trunk: `[B, N, single_size]` trunk single representation; used when ``structure_prediction``.
            pair_trunk: `[B, N, N, pair_size]` trunk pair representation; used when ``structure_prediction``.

        Returns:
            Tuple ``(atom_query, atom_conditioning, atom_pair)``: the atom query state, the atom conditioning
            state, and the windowed atom-pair states `[B, W, Q, K, atom_pair_hidden_size]`.
        """

        atom_query = self.atom_feature_projection(atom_features)
        atom_single = atom_query

        num_query_atoms, num_key_atoms = self.num_query_atoms, self.num_key_atoms
        batch, length = atom_single.shape[:2]
        padding = (-length) % num_query_atoms
        num_windows = (length + padding) // num_query_atoms
        to_keys = partial(
            gather_key_windows,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )

        def pad_atoms(value: Tensor) -> Tensor:
            r"""
            Zero-pad ``value``'s atom axis (dim 1) up to a whole number of query windows; no-op when aligned.

            Args:
                value: Tensor whose dim-1 atom axis is right-padded with zeros to a multiple of the window size.

            Returns:
                ``value`` padded on its atom axis, or the original tensor when it is already window-aligned.
            """

            if not padding:
                return value
            return torch.nn.functional.pad(value, [0, 0] * (value.dim() - 2) + [0, padding])

        positions = pad_atoms(reference_positions)
        positions_queries = positions.view(batch, num_windows, num_query_atoms, 1, 3)
        positions_keys = to_keys(positions).view(batch, num_windows, 1, num_key_atoms, 3)
        distance_vectors = positions_keys - positions_queries

        mask = pad_atoms(atom_mask).bool()
        mask_queries = mask.view(batch, num_windows, num_query_atoms, 1)
        mask_keys = to_keys(mask.unsqueeze(-1).float()).view(batch, num_windows, 1, num_key_atoms).bool()
        space_id = pad_atoms(reference_space_id)
        uid_queries = space_id.view(batch, num_windows, num_query_atoms, 1)
        uid_keys = to_keys(space_id.unsqueeze(-1).float()).view(batch, num_windows, 1, num_key_atoms).long()
        valid = (mask_queries & mask_keys & (uid_queries == uid_keys)).to(distance_vectors.dtype).unsqueeze(-1)

        pair_bias = None
        single_trunk_projection = self.single_trunk_projection
        if single_trunk_projection is not None:
            token_index = cast(Tensor, atom_to_token)
            single_proj = single_trunk_projection(cast(Tensor, single_trunk))
            atom_single = atom_single + single_proj.gather(
                1, token_index.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1])
            )
            pair_from_trunk = cast(nn.Sequential, self.pair_trunk_projection)(cast(Tensor, pair_trunk))
            token_windows = pad_atoms(token_index)
            query_token = token_windows.view(batch, num_windows, num_query_atoms)
            key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
            batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
            pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
            pair_bias = pair_gathered * mask_keys.unsqueeze(-1).to(distance_vectors.dtype)

        single_windows = pad_atoms(atom_single)
        atom_pair = encode_atom_pair(
            self,
            distance_vectors,
            single_windows.view(batch, num_windows, num_query_atoms, 1, -1),
            to_keys(single_windows).view(batch, num_windows, 1, num_key_atoms, -1),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=pair_bias,
            prefix_ndim=1,
        )
        return atom_query, atom_single, atom_pair


class AtomAttentionEncoder(nn.Module):
    r"""
    Run windowed atom attention and mean-pool the result to per-token representations.

    When ``structure_prediction`` is set, the encoder additionally folds in a projection of the noisy atom
    positions and returns the atom/conditioning skip states used by the diffusion decoder.

    Args:
        atom_hidden_size: Width of the atom single representation.
        single_size: Width of the pooled per-token output (doubled when ``structure_prediction``).
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_encoder_blocks: Number of atom-attention blocks.
        num_atom_encoder_attention_heads: Head count for the atom attention.
        structure_prediction: When ``True``, add the noisy-position projection and return skip states.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        structure_prediction: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.noisy_position_projection: nn.Linear | None = None
        if structure_prediction:
            self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        token_out = 2 * single_size if structure_prediction else single_size
        self.token_projection = nn.Sequential(nn.Linear(atom_hidden_size, token_out, bias=False), nn.ReLU())

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        noisy_positions: Tensor | None = None,
        multiplicity: int = 1,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""
        Attend over windowed atoms and mean-pool to per-token representations.

        Args:
            atom_single: `[B, M, atom_hidden_size]` atom single states.
            atom_conditioning: `[B, M, atom_hidden_size]` atom conditioning states.
            atom_pair_bias: Packed per-block windowed atom-pair attention bias.
            atom_mask: `[B, M]` atom validity mask.
            atom_to_token: `[B, M]` atom-to-token index map.
            num_tokens: Number of tokens to scatter into.
            noisy_positions: `[B*mult, M, 3]` noisy atom positions; required when ``structure_prediction``.
            multiplicity: Sample multiplicity to repeat-interleave the atom inputs by.

        Returns:
            The `[B, num_tokens, single_size]` pooled token representation. When ``structure_prediction`` is
            set, a tuple ``(token_repr, atom_skip, conditioning_skip)`` for the diffusion decoder.
        """

        query = atom_single
        conditioning = atom_conditioning
        mask = atom_mask
        token_index = atom_to_token
        noisy_position_projection = self.noisy_position_projection
        if noisy_position_projection is not None:
            query = query.repeat_interleave(multiplicity, 0) + noisy_position_projection(cast(Tensor, noisy_positions))
            conditioning = conditioning.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            token_index = token_index.repeat_interleave(multiplicity, 0)

        atom_query = self.atom_transformer(
            query,
            conditioning,
            atom_pair_bias,
            atom_mask=mask.bool(),
            attention_layout="local",
        )
        token_from_atom = self.token_projection(atom_query)
        token_repr = dense_mean_to_tokens(token_from_atom, token_index, num_tokens, atom_mask=mask, count_eps=1e-6)
        if noisy_position_projection is not None:
            return token_repr, atom_query, conditioning
        return token_repr


class AtomAttentionDecoder(nn.Module):
    r"""
    Broadcast token activations back to atoms and predict a per-atom position update.

    Args:
        atom_hidden_size: Width of the atom single representation.
        single_size: Width of the incoming token representation (token projection reads ``2 * single_size``).
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks.
        num_atom_decoder_attention_heads: Head count for the atom attention.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2 * single_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        self.position_update = nn.Sequential(LayerNorm(atom_hidden_size), nn.Linear(atom_hidden_size, 3, bias=False))

    def forward(
        self,
        token_repr: Tensor,
        atom_skip: Tensor,
        conditioning_skip: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        r"""
        Broadcast token activations to atoms and predict the per-atom coordinate update.

        Args:
            token_repr: `[B*mult, N, 2*single_size]` token representation to broadcast to atoms.
            atom_skip: `[B*mult, M, atom_hidden_size]` atom skip state from the encoder.
            conditioning_skip: `[B*mult, M, atom_hidden_size]` conditioning skip state from the encoder.
            atom_pair_bias: Packed per-block windowed atom-pair attention bias.
            atom_mask: `[B, M]` atom validity mask.
            atom_to_token: `[B, M]` atom-to-token index map.
            multiplicity: Sample multiplicity to repeat-interleave the atom inputs by.

        Returns:
            The `[B*mult, M, 3]` per-atom coordinate update.
        """

        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        token_proj = self.token_projection(token_repr)
        atom_query = atom_skip + token_proj.gather(1, token_index.unsqueeze(-1).expand(-1, -1, token_proj.shape[-1]))
        mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask.bool(),
            attention_layout="local",
        )
        return self.position_update(atom_query)


class DiffusionConditioning(nn.Module):
    r"""
    Precompute the diffusion conditioning shared across denoiser calls (Boltz DiffusionConditioning).

    The conditioning depends only on the trunk outputs, so it is computed once per structure and reused across
    every reverse-diffusion step and every sample: the conditioned pair, the atom single/conditioning states,
    and the per-block attention biases for the atom encoder, atom decoder, and token transformer.

    Args:
        single_size: Width of the trunk single representation.
        pair_size: Width of the trunk pair representation.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        atom_feature_size: Width of the raw per-atom feature vector.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_encoder_blocks: Number of atom-encoder blocks (one bias projection each).
        num_atom_encoder_attention_heads: Head count for the atom-encoder attention bias.
        num_atom_decoder_blocks: Number of atom-decoder blocks (one bias projection each).
        num_atom_decoder_attention_heads: Head count for the atom-decoder attention bias.
        num_token_transformer_blocks: Number of token-transformer blocks (one bias projection each).
        num_token_transformer_heads: Head count for the token-transformer attention bias.
        num_conditioning_transitions: Number of transition blocks in the pairwise conditioner.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 16,
        num_conditioning_transitions: int = 2,
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_conditioning_transitions=num_conditioning_transitions,
        )
        self.atom_encoder = AtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            structure_prediction=True,
        )
        self.atom_enc_proj_pair = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size),
                nn.Linear(atom_pair_hidden_size, num_atom_encoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_encoder_blocks)
        )
        self.atom_dec_proj_pair = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size),
                nn.Linear(atom_pair_hidden_size, num_atom_decoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_decoder_blocks)
        )
        self.token_trans_proj_pair = nn.ModuleList(
            nn.Sequential(LayerNorm(pair_size), nn.Linear(pair_size, num_token_transformer_heads, bias=False))
            for _ in range(num_token_transformer_blocks)
        )

    def forward(
        self,
        single_trunk: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
    ) -> dict[str, Tensor]:
        r"""
        Precompute the per-call-invariant diffusion conditioning from the trunk representations.

        The trunk pair ``(B, N, N, pair_size)`` is conditioned with the relative-position encoding, then the
        atom encoder builds atom single/pair states injected with trunk single/pair; the atom-pair and token-pair
        states are projected to per-block attention biases (concatenated over blocks along the last axis).

        Args:
            single_trunk: `[B, N, single_size]` trunk single representation.
            pair_trunk: `[B, N, N, pair_size]` trunk pair representation.
            relative_position_encoding: `[B, N, N, pair_size]` relative-position encoding to condition the pair.
            atom_features: `[B, M, atom_feature_size]` raw per-atom features.
            reference_positions: `[B, M, 3]` reference atom positions.
            atom_mask: `[B, M]` atom validity mask.
            reference_space_id: `[B, M]` per-atom reference-space identifier for window gating.
            atom_to_token: `[B, M]` atom-to-token index map.

        Returns a dict with:
            atom_query: Atom single states ``(B, M, atom_hidden_size)``.
            atom_conditioning: Atom conditioning states ``(B, M, atom_hidden_size)``.
            atom_enc_bias: Encoder atom-pair attention bias
                ``(B, W, Q, K, num_atom_encoder_blocks * num_atom_encoder_attention_heads)``.
            atom_dec_bias: Decoder atom-pair attention bias
                ``(B, W, Q, K, num_atom_decoder_blocks * num_atom_decoder_attention_heads)``.
            token_trans_bias: Token-transformer pair bias ``(B, N, N, num_token_transformer_blocks * heads)``.
        """

        pair = self.pairwise_conditioner(pair_trunk, relative_position_encoding)
        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features,
            reference_positions,
            atom_mask,
            reference_space_id,
            atom_to_token=atom_to_token,
            single_trunk=single_trunk,
            pair_trunk=pair,
        )
        atom_enc_bias = torch.cat([layer(atom_pair) for layer in self.atom_enc_proj_pair], dim=-1)
        atom_dec_bias = torch.cat([layer(atom_pair) for layer in self.atom_dec_proj_pair], dim=-1)
        token_trans_bias = torch.cat([layer(pair) for layer in self.token_trans_proj_pair], dim=-1)
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }


class PairwiseConditioning(nn.Module):
    r"""
    Condition the trunk pair representation with the relative-position encoding (Algorithm 21).

    Args:
        pair_size: Width of the trunk pair representation.
        relative_position_size: Width of the relative-position encoding concatenated onto the pair.
        num_conditioning_transitions: Number of transition blocks refining the conditioned pair.
        transition_expansion: Inner-width expansion factor of each transition block.
    """

    def __init__(
        self,
        pair_size: int = 128,
        relative_position_size: int = 128,
        num_conditioning_transitions: int = 2,
        transition_expansion: int = 2,
    ) -> None:
        super().__init__()
        self.init_projection = nn.Sequential(
            LayerNorm(pair_size + relative_position_size),
            nn.Linear(pair_size + relative_position_size, pair_size, bias=False),
        )
        self.transitions = nn.ModuleList(
            Transition(pair_size, expansion=transition_expansion) for _ in range(num_conditioning_transitions)
        )

    def forward(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        r"""
        Concatenate ``pair_trunk`` and the relative-position encoding, project back to ``pair_size``, then refine
        with the transition stack. Both inputs and the output are ``(B, N, N, pair_size)``.

        Args:
            pair_trunk: `[B, N, N, pair_size]` trunk pair representation to condition.
            relative_position_encoding: `[B, N, N, relative_position_size]` relative-position encoding concatenated
                onto the pair before projection.

        Returns:
            The `[B, N, N, pair_size]` conditioned pair representation.
        """

        return condition_pair(
            pair_trunk,
            relative_position_encoding,
            norm=self.init_projection[0],
            projection=self.init_projection[1],
            transitions=self.transitions,
        )


class AtomDiffusion(nn.Module):
    r"""
    EDM (Karras) reverse-diffusion sampler wrapped around a Boltz diffusion score model.

    Args:
        score_model: The diffusion score network that predicts denoised coordinates.
        sigma_data: EDM data-scale hyperparameter for the preconditioner and schedule.
        min_noise_level: Lowest non-zero noise level of the schedule, in ``sigma_data`` units.
        max_noise_level: Highest noise level (start of the schedule), in ``sigma_data`` units.
        rho: Karras schedule power controlling the sigma spacing.
        churn_rate: Stochastic-churn factor applied above ``min_churn_noise_level``.
        min_churn_noise_level: Noise level below which churn is disabled.
        noise_scale: Multiplier on the churn-injected noise.
        step_scale: Multiplier on the Euler update step.
        num_sampling_steps: Default number of reverse-diffusion steps.
        alignment_reverse_diff: Whether to weighted-rigid-align each step to the denoised coords.
    """

    def __init__(
        self,
        score_model: Boltz1DiffusionScoreModel,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
    ) -> None:
        super().__init__()
        self.score_model = score_model
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho = rho
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Draw ``batch_size`` training noise levels from the EDM log-normal distribution.

        Mirrors the upstream Boltz-1 ``noise_distribution``: ``sigma_data * exp(P_mean + P_std * N(0, 1))``.

        Args:
            batch_size: Number of noise levels to sample.
            device: Device for the sampled tensor.
            dtype: Floating dtype for the sampled tensor.
            generator: Optional RNG for reproducible noise-level sampling.

        Returns:
            A ``(batch_size,)`` tensor of positive noise levels.
        """

        normal = torch.empty(batch_size, device=device, dtype=dtype).normal_(generator=generator)
        return self.sigma_data * (self.noise_log_mean + self.noise_log_std * normal).exp()

    def loss_weight(self, sigma: Tensor) -> Tensor:
        r"""
        Return the EDM per-sample coordinate-loss weight ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``.

        Args:
            sigma: Noise levels the training samples were noised at.

        Returns:
            The per-sample loss weight, matching ``sigma``'s shape.
        """

        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def sample_schedule(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        r"""
        Build the EDM (Karras rho-power) noise schedule scaled by ``sigma_data``.

        The schedule decreases from ``max_noise_level`` to ``min_noise_level`` with a trailing zero.

        Args:
            num_sampling_steps: Number of steps; defaults to ``self.num_sampling_steps``.
            device: Device to allocate the schedule on.

        Returns:
            The ``(steps + 1,)`` sigma schedule (the final entry is ``0``).
        """

        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        # `steps` is the number of sampling steps (transitions); the trailing pad adds the final sigma=0.
        # Clamp the divisor so a single step mirrors the AF3 scheduler (ramp [0] -> a lone max_noise_level, then 0)
        # instead of dividing by zero; for steps >= 2 the divisor is steps - 1 exactly, so the schedule is unchanged.
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)


class PerLayerPairBiasProjection(nn.Module):
    r"""
    Project a raw pair representation to one packed per-layer/head attention bias tensor.

    Args:
        pair_size: Width of the input pair representation.
        num_layers: Number of per-layer bias projections to pack.
        num_attention_heads: Head count each layer projects the pair to.
    """

    def __init__(self, *, pair_size: int, num_layers: int, num_attention_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(LayerNorm(pair_size), nn.Linear(pair_size, num_attention_heads, bias=False))
            for _ in range(num_layers)
        )

    def forward(self, pair: Tensor) -> Tensor:
        r"""
        Project and concatenate the per-layer pair biases along the last axis.

        Args:
            pair: `[..., pair_size]` pair representation.

        Returns:
            The `[..., num_layers * num_attention_heads]` packed per-layer/head attention bias.
        """

        return torch.cat([layer(pair) for layer in self.layers], dim=-1)


class Boltz1DiffusionScoreModel(nn.Module):
    r"""
    Boltz-1 EDM score network with the wider Boltz-1 ``s_inputs`` conditioning width.

    Given noisy atom positions and a noise level, it noise-conditions the single track, runs the atom-attention
    encoder, the conditioned token transformer, and the atom-attention decoder to predict a per-atom
    coordinate update. The precomputed diffusion conditioning is threaded through so no trunk work repeats.

    Args:
        input_size: Width of the ``single_inputs`` conditioning vector.
        single_size: Width of the trunk single representation (token hidden is ``2 * single_size``).
        atom_hidden_size: Width of the atom single representation.
        fourier_embedding_size: Width of the frozen Fourier noise embedding.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_encoder_blocks: Number of atom-encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-encoder attention.
        num_token_transformer_blocks: Number of conditioned token-transformer blocks.
        num_token_transformer_heads: Head count for the token transformer.
        num_atom_decoder_blocks: Number of atom-decoder blocks.
        num_atom_decoder_attention_heads: Head count for the atom-decoder attention.
        num_conditioning_transitions: Number of transition blocks in the single conditioner.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 8,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = Boltz1SingleConditioning(
            input_size=input_size,
            single_size=single_size,
            fourier_embedding_size=fourier_embedding_size,
            num_conditioning_transitions=num_conditioning_transitions,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            structure_prediction=True,
            operator_backend=operator_backend,
        )
        self.single_to_token = nn.Sequential(LayerNorm(hidden), nn.Linear(hidden, hidden, bias=False))
        self.token_transformer = ConditionedTransformer(
            num_blocks=num_token_transformer_blocks,
            num_attention_heads=num_token_transformer_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            operator_backend=operator_backend,
        )
        self.token_norm = LayerNorm(hidden)
        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_decoder_blocks=num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=num_atom_decoder_attention_heads,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: dict[str, Tensor],
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        multiplicity: int = 1,
        return_token_repr: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Predict the per-atom coordinate update for one reverse-diffusion step.

        Args:
            single_inputs: `[B, N, input_size]` single-input conditioning vector.
            single_trunk: `[B, N, single_size]` trunk single representation.
            noisy_positions: `[B*mult, M, 3]` noisy atom positions.
            noise_level: `[B*mult]` per-sample noise level.
            conditioning: The precomputed diffusion-conditioning dict (atom states and per-block biases).
            atom_mask: `[B, M]` atom validity mask.
            atom_to_token: `[B, M]` atom-to-token index map.
            token_mask: `[B, N]` token validity mask.
            multiplicity: Sample multiplicity to repeat-interleave the trunk inputs by.
            return_token_repr: When ``True``, also return the final token representation.

        Returns:
            The `[B*mult, M, 3]` coordinate update, or a ``(update, token_repr)`` tuple when
            ``return_token_repr`` is set.
        """

        single, _ = self.single_conditioner(
            noise_level,
            single_trunk.repeat_interleave(multiplicity, 0),
            single_inputs.repeat_interleave(multiplicity, 0),
        )
        token_repr, atom_skip, conditioning_skip = self.atom_attention_encoder(
            conditioning["atom_query"],
            conditioning["atom_conditioning"],
            conditioning["atom_enc_bias"].repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            token_mask.shape[-1],
            noisy_positions=noisy_positions,
            multiplicity=multiplicity,
        )
        token_repr = token_repr + self.single_to_token(single)
        token_repr = self.token_transformer(
            token_repr,
            single,
            conditioning["token_trans_bias"].repeat_interleave(multiplicity, 0),
            token_mask.repeat_interleave(multiplicity, 0).float(),
        )
        token_repr = self.token_norm(token_repr)
        coordinate_update = self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            conditioning["atom_dec_bias"].repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            multiplicity=multiplicity,
        )
        if return_token_repr:
            return coordinate_update, token_repr
        return coordinate_update


class Boltz1SingleConditioning(nn.Module):
    r"""
    Noise-condition the concatenated trunk-single and single-input representation for Boltz-1.

    Args:
        input_size: Width of the ``single_inputs`` representation.
        single_size: Width of the trunk single representation (hidden width is ``2 * single_size``).
        fourier_embedding_size: Width of the frozen Fourier noise embedding.
        num_conditioning_transitions: Number of transition blocks refining the conditioned single.
        transition_expansion: Inner-width expansion factor of each transition block.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_conditioning_transitions: int = 2,
        transition_expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(single_size + input_size)
        self.single_projection = nn.Linear(single_size + input_size, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=transition_expansion) for _ in range(num_conditioning_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Noise-condition the concatenated single representation.

        Args:
            noise_level: `[B]` per-sample noise level fed to the Fourier embedding.
            single_trunk: `[B, N, single_size]` trunk single representation.
            single_inputs: `[B, N, input_size]` single-input representation.

        Returns:
            A ``(conditioned_single, normalized_single)`` tuple, each `[B, N, 2*single_size]`.
        """

        return condition_single(
            single_trunk,
            single_inputs,
            noise_level,
            norm=self.single_norm,
            projection=self.single_projection,
            noise_fourier=self.noise_fourier,
            noise_norm=self.fourier_norm,
            noise_projection=self.fourier_projection,
            transitions=self.transitions,
            sample_axis=False,
        )


class Boltz1OutTokenFeatureUpdate(nn.Module):
    r"""
    Accumulate the diffusion-token representation across reverse-diffusion steps for confidence conditioning.

    This is the official Boltz-1 ``out_token_feat_update``: at each step it noise-conditions the incoming
    token representation and adds a gated transition-block update into the running accumulator.

    Args:
        token_size: Width of the trunk token representation (hidden width is ``2 * token_size``).
        fourier_embedding_size: Width of the frozen Fourier noise embedding.
    """

    def __init__(self, *, token_size: int = 384, fourier_embedding_size: int = 256) -> None:
        super().__init__()
        hidden = 2 * token_size
        self.next_norm = LayerNorm(hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size)
        self.transition_block = Boltz1ConditionedTransitionBlock(hidden, hidden + fourier_embedding_size)

    def forward(self, *, noise_level: Tensor, accumulated: Tensor, next_repr: Tensor) -> Tensor:
        r"""
        Fold the current-step token representation into the running accumulator.

        Args:
            noise_level: `[B]` per-sample noise level for the Fourier conditioning.
            accumulated: `[B, N, 2*token_size]` running token accumulator.
            next_repr: `[B, N, 2*token_size]` token representation from the current step.

        Returns:
            The updated `[B, N, 2*token_size]` accumulator.
        """

        next_repr = self.next_norm(next_repr)
        fourier = self.fourier_norm(self.noise_fourier(noise_level))
        fourier = fourier.unsqueeze(1).expand(-1, next_repr.shape[1], -1)
        conditioning = torch.cat([accumulated, fourier], dim=-1)
        return accumulated + self.transition_block(next_repr, conditioning)


class Boltz1ConditionedTransitionBlock(nn.Module):
    r"""
    Conditioned SwiGLU transition block used by the Boltz-1 ``out_token_feat_update`` accumulator.

    Applies AdaLN conditioning, a SwiGLU expansion, and an output projection gated by a sigmoid of the
    conditioning.

    Args:
        hidden_size: Width of the hidden representation.
        conditioning_size: Width of the conditioning vector driving AdaLN and the output gate.
        expansion: Inner-width expansion factor of the SwiGLU transition.
    """

    def __init__(self, hidden_size: int, conditioning_size: int, expansion: int = 2) -> None:
        super().__init__()
        self.adaptive_norm = Boltz1AdaptiveLayerNorm(hidden_size, conditioning_size)
        expanded_size = hidden_size * expansion
        self.gate_value_projection = nn.Linear(hidden_size, 2 * expanded_size, bias=False)
        self.expand_projection = nn.Linear(hidden_size, expanded_size, bias=False)
        self.output_projection = nn.Linear(expanded_size, hidden_size, bias=False)
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Apply the conditioned SwiGLU transition.

        Args:
            hidden: `[..., hidden_size]` hidden representation.
            conditioning: `[..., conditioning_size]` conditioning vector.

        Returns:
            The `[..., hidden_size]` gated transition output.
        """

        hidden = self.adaptive_norm(hidden, conditioning)
        gate, value = self.gate_value_projection(hidden).chunk(2, dim=-1)
        update = self.silu(gate) * value * self.expand_projection(hidden)
        return self.sigmoid(self.output_gate_projection(conditioning)) * self.output_projection(update)


class Boltz1AdaptiveLayerNorm(nn.Module):
    r"""
    Lightweight adaptive layer norm (AdaLN) matching the official Boltz conditioned transition.

    Normalizes the hidden state (affine-free), then applies a sigmoid gate and shift both derived from the
    normalized conditioning vector.

    Args:
        hidden_size: Width of the hidden representation.
        conditioning_size: Width of the conditioning vector driving the gate and shift.
    """

    def __init__(self, hidden_size: int, conditioning_size: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, bias=False)
        self.conditioning_norm = nn.LayerNorm(conditioning_size, bias=False)
        self.gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.shift_projection = nn.Linear(conditioning_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Adaptively normalize ``hidden`` under the ``conditioning`` gate and shift.

        Args:
            hidden: `[..., hidden_size]` hidden representation.
            conditioning: `[..., conditioning_size]` conditioning vector.

        Returns:
            The `[..., hidden_size]` conditioned representation.
        """

        hidden = self.hidden_norm(hidden)
        conditioning = self.conditioning_norm(conditioning)
        return self.sigmoid(self.gate_projection(conditioning)) * hidden + self.shift_projection(conditioning)


class Boltz1AtomDiffusion(AtomDiffusion):
    r"""
    Boltz-1 EDM sampler that can accumulate and surface the final diffusion token representation.

    Extends [`AtomDiffusion`][ofoldx.models.boltz1.model.AtomDiffusion] with the optional
    ``out_token_feat_update`` accumulator and the Boltz-1x steering path (Feynman-Kac particle resampling and
    physical-potential guidance).

    Args:
        score_model: The diffusion score network that predicts denoised coordinates.
        sigma_data: EDM data-scale hyperparameter.
        min_noise_level: Lowest non-zero noise level in the sampling schedule.
        max_noise_level: Highest noise level in the sampling schedule.
        rho: Karras schedule power.
        churn_rate: Stochastic churn factor.
        min_churn_noise_level: Noise level below which churn is disabled.
        noise_scale: Multiplier on churn noise.
        step_scale: Multiplier on the Euler update.
        num_sampling_steps: Default number of reverse-diffusion steps.
        alignment_reverse_diff: Whether to rigid-align each reverse-diffusion step.
        noise_log_mean: Mean of the EDM training log-noise distribution.
        noise_log_std: Standard deviation of the EDM training log-noise distribution.
        accumulate_token_repr: When ``True``, build and run the token-representation accumulator.
        token_size: Width of the trunk token representation for the accumulator.
        fourier_embedding_size: Width of the frozen Fourier noise embedding for the accumulator.
    """

    def __init__(
        self,
        score_model: Boltz1DiffusionScoreModel,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        *,
        accumulate_token_repr: bool = False,
        token_size: int = 384,
        fourier_embedding_size: int = 256,
    ) -> None:
        super().__init__(
            score_model=score_model,
            sigma_data=sigma_data,
            min_noise_level=min_noise_level,
            max_noise_level=max_noise_level,
            rho=rho,
            churn_rate=churn_rate,
            min_churn_noise_level=min_churn_noise_level,
            noise_scale=noise_scale,
            step_scale=step_scale,
            num_sampling_steps=num_sampling_steps,
            alignment_reverse_diff=alignment_reverse_diff,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
        )
        self.out_token_feat_update: Boltz1OutTokenFeatureUpdate | None = None
        if accumulate_token_repr:
            self.out_token_feat_update = Boltz1OutTokenFeatureUpdate(
                token_size=token_size, fourier_embedding_size=fourier_embedding_size
            )

    def preconditioned_network_forward(  # type: ignore[override]
        self,
        noised_coords: Tensor,
        sigma: Tensor | float,
        *,
        multiplicity: int = 1,
        return_token_repr: bool = False,
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Apply the EDM preconditioning and run the score model to produce denoised coordinates.

        Args:
            noised_coords: `[B, M, 3]` noised atom coordinates.
            sigma: The current noise level, as a scalar or `[B]` tensor.
            multiplicity: Sample multiplicity threaded into the score model.
            return_token_repr: When ``True``, also return the diffusion token representation.
            **conditioning: The precomputed conditioning forwarded to the score model.

        Returns:
            The `[B, M, 3]` denoised coordinates, or a ``(denoised, token_repr)`` tuple when
            ``return_token_repr`` is set.

        """

        batch = noised_coords.shape[0]
        sigma = torch.as_tensor(sigma, device=noised_coords.device, dtype=noised_coords.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.score_model(
            noisy_positions=self.preconditioner.c_in(padded) * noised_coords,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            return_token_repr=return_token_repr,
            **conditioning,
        )
        if return_token_repr:
            update, token_repr = cast(tuple[Tensor, Tensor], update)
        else:
            update = cast(Tensor, update)
        denoised = self.preconditioner.c_skip(padded) * noised_coords + self.preconditioner.c_out(padded) * update
        if return_token_repr:
            return denoised, token_repr
        return denoised

    def denoise_training(
        self,
        target_coords: Tensor | None,
        atom_mask: Tensor,
        *,
        target_mask: Tensor | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Run one EDM training denoise step on the noised, augmented ground-truth coordinates.

        Mirrors the upstream Boltz-1 ``AtomDiffusion`` training forward: the (resolved-atom-centered, randomly
        rotated/translated) ground truth is repeat-interleaved by ``multiplicity``, noised at an independent
        per-sample log-normal noise level, and denoised in a single preconditioned score-model pass. The
        denoiser runs with gradient enabled so the EDM-weighted coordinate loss can backpropagate.

        Args:
            target_coords: `[B, M, 3]` ground-truth atom coordinates.
            atom_mask: `[B, M]` atom validity mask forwarded to the score model.
            target_mask: Optional `[B, M]` resolved-atom mask used to center the augmentation; defaults to
                ``atom_mask``.
            multiplicity: Number of independent noised samples to draw per input.
            generator: Optional RNG for reproducible augmentation and noise.
            **conditioning: The precomputed conditioning (``single_inputs``, ``single_trunk``,
                ``conditioning``, ``atom_to_token``, ``token_mask``) forwarded to the score model.

        Returns:
            ``(denoised_coords, augmented_target, noise_level)`` with a leading ``B*multiplicity`` axis, for the
            EDM-weighted coordinate loss.

        """

        single_trunk = cast(Tensor, conditioning["single_trunk"])
        dtype, device = single_trunk.dtype, single_trunk.device
        target = cast(Tensor, target_coords).to(dtype=dtype, device=device).repeat_interleave(multiplicity, 0)
        resolved = (target_mask if target_mask is not None else atom_mask).to(dtype=dtype, device=device)
        resolved = resolved.repeat_interleave(multiplicity, 0)
        augmented_target = self._augment_target(target, resolved, generator=generator)
        batch_size = augmented_target.shape[0]
        noise_level = self.noise_distribution(batch_size, device=device, dtype=dtype, generator=generator)
        noise = torch.empty_like(augmented_target).normal_(generator=generator)
        noised_coords = augmented_target + noise_level[:, None, None] * noise
        denoised_coords = self.preconditioned_network_forward(
            noised_coords,
            noise_level,
            multiplicity=multiplicity,
            atom_mask=atom_mask,
            return_token_repr=False,
            **conditioning,
        )
        return cast(Tensor, denoised_coords), augmented_target, noise_level

    def _augment_target(self, coords: Tensor, mask: Tensor, *, generator: torch.Generator | None) -> Tensor:
        r"""
        Center on the masked atoms, then apply one random rigid augmentation (rotation plus translation).

        Args:
            coords: `[B, M, 3]` coordinates to augment.
            mask: `[B, M]` validity mask; the centroid is computed over masked atoms only.
            generator: Optional RNG for reproducible rotation/translation.

        Returns:
            The `[B, M, 3]` augmented coordinates.
        """

        weight = mask[..., None]
        centroid = (coords * weight).sum(-2, keepdim=True) / weight.sum(-2, keepdim=True).clamp_min(1.0)
        coords = coords - centroid
        rotation = uniform_random_rotation(
            coords.shape[0], device=coords.device, dtype=coords.dtype, generator=generator
        )
        translation = torch.empty(coords.shape[0], 1, 3, device=coords.device, dtype=coords.dtype).normal_(
            generator=generator
        )
        return torch.einsum("bmd,bds->bms", coords, rotation) + translation

    @torch.no_grad()
    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        return_token_repr: bool = False,
        steering_args: Mapping[str, object],
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Run the full reverse-diffusion loop and return the sampled atom coordinates.

        Steps the EDM schedule with stochastic churn and per-step random rigid augmentation, optionally
        applying Boltz-1x Feynman-Kac particle resampling and physical-potential guidance, and (when enabled)
        accumulating the final token representation used by the Boltz-1 confidence head.

        Args:
            atom_mask: `[B, M]` atom validity mask.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            multiplicity: Number of samples requested per input.
            generator: Optional RNG for reproducible sampling noise.
            return_token_repr: When ``True``, also return the accumulated token representation.
            steering_args: Boltz-1x steering flags (FK steering / physical guidance); normalized internally.
            **conditioning: The precomputed conditioning (plus optional ``feats``) forwarded to the score model.

        Returns:
            The `[B*mult, M, 3]` sampled coordinates, or a ``(coords, token_repr)`` tuple when
            ``return_token_repr`` is set.

        """

        device = atom_mask.device
        feats = cast(dict[str, Tensor] | None, conditioning.pop("feats", None))
        steering = dict(steering_args)
        potentials = (
            _boltz1_potentials(steering)
            if bool(steering["fk_steering"] or steering["physical_guidance_update"])
            else ()
        )
        num_particles = int(cast(float | int | str, steering["num_particles"]))
        requested_multiplicity = multiplicity
        effective_multiplicity = multiplicity * num_particles if bool(steering["fk_steering"]) else multiplicity
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(effective_multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)

        def randn() -> Tensor:
            r"""Draw standard-normal noise of the expanded coordinate ``shape`` on the sampler device/generator."""

            return torch.randn(shape, device=device, generator=generator)

        coords = sigmas[0] * randn()
        if bool(steering["fk_steering"]):
            num_groups = atom_mask.shape[0] * requested_multiplicity
            energy_traj = torch.empty((expanded_atom_mask.shape[0], 0), device=device)
            resample_weights = torch.ones(num_groups, num_particles, device=device)
        else:
            energy_traj = torch.empty((0, 0), device=device)
            resample_weights = torch.empty((0, 0), device=device)
        if bool(steering["physical_guidance_update"]):
            scaled_guidance_update = torch.zeros(shape, dtype=torch.float32, device=device)
        else:
            scaled_guidance_update = None
        denoised = None
        token_repr = None
        token_a = None
        # This loop stays separate from the shared CoordinateEdmSampler by design (Frankenstein principle):
        # Boltz-1 is a strict superset of the plain EDM step -- it interleaves Boltz-1x Feynman-Kac particle
        # steering (resampling, guidance updates, out_token_feat accumulation) into the reverse diffusion, which
        # the shared sampler deliberately does not model. Genuine structural divergence, so it is left as-is.
        # Keep the schedule scalars as tensors. The Python loop fixes the number of denoising calls, while the
        # per-step tensor math remains free of host synchronizations on the ordinary (unsteered) sampling path.
        for index in range(len(sigmas) - 1):
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0], device=device, dtype=coords.dtype, generator=generator
            )
            translation = torch.randn(expanded_atom_mask.shape[0], 1, 3, device=device, generator=generator)
            coords = torch.einsum("bmd,bds->bms", coords - coords.mean(-2, keepdim=True), rotation) + translation
            if denoised is not None:
                denoised = (
                    torch.einsum("bmd,bds->bms", denoised - denoised.mean(-2, keepdim=True), rotation) + translation
                )
            if scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum("bmd,bds->bms", scaled_guidance_update, rotation)

            sigma_hat = sigma_current * (1 + gamma)
            steering_t = 1.0 - (index / max(len(sigmas) - 1, 1))
            # Derive the churn ``noise_var`` and the reverse-step scalars in float64 to match upstream Boltz-1,
            # which computes ``t_hat``/``noise_var``/``sigma_t - t_hat`` from Python-float (float64) schedule
            # scalars via ``.item()``; ``sigma_hat**2 - sigma_current**2`` and the fp32 Euler step otherwise drift
            # from upstream. The float32 ``sigma_hat`` is still what the score network sees (its ``c_noise`` is
            # float32 upstream too). These stay 0-dim tensors (no host ``.item()``), keeping the loop
            # graph-break-free. Mirrors the boltz2 sibling's sampler fix.
            sigma_current_hp = sigma_current.double()
            sigma_hat_hp = sigma_current_hp * (1 + gamma.double())
            sigma_next_hp = sigma_next.double()
            noise_var = self.noise_scale**2 * (sigma_hat_hp**2 - sigma_current_hp**2)
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)).to(coords.dtype) * randn()
            denoised_out = self.preconditioned_network_forward(
                noisy,
                sigma_hat,
                multiplicity=effective_multiplicity,
                atom_mask=base_atom_mask,
                return_token_repr=return_token_repr,
                **conditioning,
            )
            if return_token_repr:
                denoised, token_a = cast(tuple[Tensor, Tensor], denoised_out)
            else:
                denoised = cast(Tensor, denoised_out)

            should_resample = bool(steering["fk_steering"]) and (
                (index % int(cast(float | int | str, steering["fk_resampling_interval"])) == 0 and noise_var > 0)
                or index == len(sigmas) - 2
            )
            if should_resample:
                energy = torch.zeros(denoised.shape[0], device=device, dtype=denoised.dtype)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if float(parameters["resampling_weight"]) > 0:
                        energy = energy + float(parameters["resampling_weight"]) * potential.compute(
                            denoised,
                            cast(dict[str, Tensor], feats),
                            parameters,
                        )
                energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)
                log_g = -energy if energy_traj.shape[1] == 1 else energy_traj[:, -2] - energy_traj[:, -1]
                if scaled_guidance_update is not None and noise_var > 0:
                    ll_difference = ((noisy - coords) ** 2 - (noisy - coords + scaled_guidance_update) ** 2).sum(
                        dim=(-1, -2)
                    ) / (2 * noise_var)
                else:
                    ll_difference = torch.zeros_like(energy)
                resample_weights = torch.softmax(
                    (ll_difference + float(cast(float | int | str, steering["fk_lambda"])) * log_g).reshape(
                        -1, num_particles
                    ),
                    dim=1,
                )

            if bool(steering["physical_guidance_update"]) and index < len(sigmas) - 2:
                guidance_update = torch.zeros_like(denoised)
                for guidance_step in range(int(cast(float | int | str, steering["num_gd_steps"]))):
                    energy_gradient = torch.zeros_like(denoised)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        interval = max(int(parameters["guidance_interval"]), 1)
                        if float(parameters["guidance_weight"]) > 0 and guidance_step % interval == 0:
                            energy_gradient = energy_gradient + float(
                                parameters["guidance_weight"]
                            ) * potential.compute_gradient(
                                denoised + guidance_update, cast(dict[str, Tensor], feats), parameters
                            )
                    guidance_update = guidance_update - energy_gradient
                denoised = denoised + guidance_update
                scaled_guidance_update = guidance_update * -self.step_scale * (sigma_next - sigma_hat) / sigma_hat

            if should_resample:
                sample_count = num_particles if index < len(sigmas) - 2 else 1
                resample_indices = (
                    torch.multinomial(resample_weights, sample_count, replacement=True, generator=generator)
                    + num_particles * torch.arange(resample_weights.shape[0], device=device).unsqueeze(-1)
                ).flatten()
                coords = coords[resample_indices]
                noisy = noisy[resample_indices]
                expanded_atom_mask = expanded_atom_mask[resample_indices]
                denoised = denoised[resample_indices]
                energy_traj = energy_traj[resample_indices]
                if scaled_guidance_update is not None:
                    scaled_guidance_update = scaled_guidance_update[resample_indices]
                if token_repr is not None:
                    token_repr = token_repr[resample_indices]
                if token_a is not None:
                    token_a = token_a[resample_indices]
                if index == len(sigmas) - 2:
                    effective_multiplicity = requested_multiplicity
            out_token_feat_update = self.out_token_feat_update
            if out_token_feat_update is not None:
                if token_repr is None:
                    token_repr = torch.zeros_like(cast(Tensor, token_a))
                token_a = cast(Tensor, token_a)
                sigma = torch.full((token_a.shape[0],), sigma_hat, device=token_a.device, dtype=token_a.dtype)
                token_repr = out_token_feat_update(
                    noise_level=self.preconditioner.c_noise(sigma),
                    accumulated=token_repr,
                    next_repr=token_a,
                )
            elif token_a is not None:
                token_repr = token_a
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy.float(), target=denoised.float(), weight=expanded_atom_mask.float()
                ).to(denoised)
            derivative = (noisy - denoised) / sigma_hat_hp
            coords = (noisy + self.step_scale * (sigma_next_hp - sigma_hat_hp) * derivative).to(coords.dtype)
        return (coords, token_repr) if return_token_repr else coords


class Boltz1ConfidenceHead(nn.Module):
    r"""
    Score sampled coordinates into Boltz-1 confidence logits and ranking metrics.

    In the released ``imitate_trunk`` configuration this module re-embeds the inputs and runs its own trunk
    (input embedder, optional MSA module, recycling projections, Pairformer) before injecting the predicted
    inter-atomic distances and the accumulated diffusion single representation, then projecting to
    pLDDT/PAE/PDE/resolved logits. The lighter non-``imitate_trunk`` path instead reuses the trunk outputs
    directly. Derived complex/interface scores and pTM/ipTM ranking metrics are computed from the logits.

    Args:
        input_size: Width of the ``single_inputs`` representation.
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        atom_feature_size: Width of the raw per-atom feature vector.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        num_blocks: Number of Pairformer blocks in the confidence module.
        num_pair_attention_heads: Head count for the confidence pair-biased attention.
        num_distogram_bins: Number of bins for the distance embedding.
        max_distance: Upper distance bound (Angstrom) for the distance binning.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
        compute_pae: Whether to emit PAE logits and pTM/ipTM metrics.
        imitate_trunk: Whether to re-run a full trunk-like stack instead of reusing trunk outputs.
        add_single_to_pair_product: Whether to add the single-input outer product into the pair track.
        add_single_input_to_single: Whether to inject the single-input representation into the single track.
        add_pair_input_to_pair: Whether to inject relative-position and bond features into the pair track.
        use_single_diffusion: Whether to condition on the accumulated diffusion single representation.
        ranking_plddt_weight: Weight on complex pLDDT when forming the combined ranking score.
        ranking_ptm_weight: Weight on pTM/ipTM when forming the combined ranking score.
        no_atom_encoder: When ``True``, skip the atom encoder in the re-embedded input embedder.
        no_msa: When ``True``, build the confidence module without the MSA module.
        msa_size: Number of MSA sequences retained per token.
        msa_feature_size: Width of the raw per-MSA feature vector.
        vocab_size: Residue-type vocabulary size for the MSA module.
        msa_blocks: Number of MSA-module blocks.
        use_paired_feature: Whether the MSA carries the cross-chain pairing feature.
        outer_product_hidden_size: Hidden width of the MSA outer-product-mean projection.
        msa_dropout: Dropout applied inside the MSA module.
        pair_dropout: Dropout applied to the Pairformer pair track.
        num_msa_attention_heads: Head count for the MSA pair-weighted-averaging.
        msa_head_dim: Per-head dimension inside the MSA attention.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Head count for the triangle attention.
        relative_position_size: Width of the relative-position feature.
        operator_backend: Selector for the fused-operator backend.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 389,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_blocks: int = 48,
        num_pair_attention_heads: int = 16,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        compute_pae: bool = True,
        imitate_trunk: bool = True,
        add_single_to_pair_product: bool = True,
        add_single_input_to_single: bool = True,
        add_pair_input_to_pair: bool = True,
        use_single_diffusion: bool = True,
        ranking_plddt_weight: float = 4.0,
        ranking_ptm_weight: float = 1.0,
        no_atom_encoder: bool = False,
        no_msa: bool = False,
        msa_size: int = 64,
        msa_feature_size: int = 36,
        vocab_size: int = 33,
        msa_blocks: int = 4,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        relative_position_size: int = 139,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.single_size = single_size
        self.pair_size = pair_size
        self.ranking_plddt_weight = ranking_plddt_weight
        self.ranking_ptm_weight = ranking_ptm_weight
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)

        self.diffusion_single_norm: nn.Module | None = None
        self.diffusion_single_projection: nn.Linear | None = None
        self.single_to_pair_product_left: nn.Linear | None = None
        self.single_to_pair_product_right: nn.Linear | None = None
        self.single_to_pair_product_out: nn.Linear | None = None
        self.single_init_projection: nn.Linear | None = None
        self.pair_left_projection: nn.Linear | None = None
        self.pair_right_projection: nn.Linear | None = None
        self.input_embedder: Boltz1InputEmbedder | None = None
        self.relative_position_embedder: RelativePositionEmbedder | None = None
        self.token_bond_projection: nn.Linear | None = None
        self.single_recycle_norm: nn.Module | None = None
        self.pair_recycle_norm: nn.Module | None = None
        self.single_recycle_projection: nn.Linear | None = None
        self.pair_recycle_projection: nn.Linear | None = None
        self.msa_module: MsaModule | None = None
        self.final_single_norm: nn.Module | None = None
        self.final_pair_norm: nn.Module | None = None
        self.input_single_norm: nn.Module | None = None
        self.single_norm: nn.Module | None = None
        self.pair_norm: nn.Module | None = None
        self.input_single_projection: nn.Linear | None = None

        if use_single_diffusion:
            self.diffusion_single_norm = LayerNorm(2 * single_size)
            self.diffusion_single_projection = nn.Linear(2 * single_size, single_size, bias=False)

        self.single_to_pair_row = nn.Linear(input_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(input_size, pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)

        if imitate_trunk:
            self.single_init_projection = nn.Linear(input_size, single_size, bias=False)
            self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
            self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
            self.input_embedder = Boltz1InputEmbedder(
                input_size=input_size,
                single_size=single_size,
                atom_hidden_size=atom_hidden_size,
                atom_pair_hidden_size=atom_pair_hidden_size,
                atom_feature_size=atom_feature_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                num_atom_encoder_blocks=num_atom_encoder_blocks,
                num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
                no_atom_encoder=no_atom_encoder,
                operator_backend=operator_backend,
            )
            self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
            self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
            self.single_recycle_norm = LayerNorm(single_size)
            self.pair_recycle_norm = LayerNorm(pair_size)
            self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
            self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
            nn.init.zeros_(self.single_recycle_projection.weight)
            nn.init.zeros_(self.pair_recycle_projection.weight)
            if not no_msa:
                self.msa_module = MsaModule(
                    single_size=input_size,
                    pair_size=pair_size,
                    msa_size=msa_size,
                    msa_feature_size=msa_feature_size,
                    num_blocks=msa_blocks,
                    vocab_size=vocab_size,
                    use_paired_feature=use_paired_feature,
                    outer_product_hidden_size=outer_product_hidden_size,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_dim=msa_head_dim,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    operator_backend=operator_backend,
                )
            self.pairformer_stack = PairformerStack(
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                operator_backend=operator_backend,
            )
            self.final_single_norm = LayerNorm(single_size)
            self.final_pair_norm = LayerNorm(pair_size)
        else:
            self.input_single_norm = LayerNorm(input_size)
            self.single_norm = LayerNorm(single_size)
            self.pair_norm = LayerNorm(pair_size)
            if add_single_input_to_single:
                self.input_single_projection = nn.Linear(input_size, single_size, bias=False)
            if add_pair_input_to_pair:
                self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
                self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
            self.pairformer_stack = PairformerStack(
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                operator_backend=operator_backend,
            )

        self.confidence_heads = Boltz1ConfidenceHeads(
            single_size=single_size,
            pair_size=pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            compute_pae=compute_pae,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        pred_coords: Tensor,
        *,
        batch: Boltz1Input,
        pred_distogram_logits: Tensor,
        single_diffusion: Tensor | None = None,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Score sampled coords and return confidence logits plus derived ranking metrics.

        Args:
            single_inputs: `[B, N, input_size]` single-input representation.
            single: `[B, N, single_size]` trunk single representation.
            pair: `[B, N, N, pair_size]` trunk pair representation.
            pred_coords: `[B*mult, M, 3]` sampled atom coordinates to score.
            batch: The Boltz-1 input batch.
            pred_distogram_logits: `[B, N, N, num_bins]` trunk distogram logits used for contact probabilities.
            single_diffusion: Accumulated diffusion single representation; required when ``use_single_diffusion``.
            multiplicity: Number of samples per input.

        Returns:
            A dict of confidence logits (``plddt_logits``, ``pde_logits``, ``resolved_logits``, and
            ``pae_logits`` when enabled) merged with the derived complex/interface and pTM/ipTM ranking scores.

        """

        if self.input_embedder is not None:
            single_inputs = self._imitate_trunk_inputs(batch)
            single_init, pair_init = self._confidence_initial_representations(batch, single_inputs, single.dtype)
            single = single_init + cast(nn.Linear, self.single_recycle_projection)(
                cast(nn.Module, self.single_recycle_norm)(single)
            )
            pair = pair_init + cast(nn.Linear, self.pair_recycle_projection)(
                cast(nn.Module, self.pair_recycle_norm)(pair)
            )
            # The imitate-trunk MSA module, Pairformer, and final norms run once at the end -- after the
            # single-input pair projections and the per-sample predicted-distance embedding -- mirroring the
            # released Boltz-1 ConfidenceModule (they must not run here on the pre-distogram pair).
        else:
            normalized_inputs = cast(nn.Module, self.input_single_norm)(single_inputs)
            single = cast(nn.Module, self.single_norm)(single)
            input_single_projection = self.input_single_projection
            if input_single_projection is not None:
                single = single + input_single_projection(normalized_inputs)
            pair = cast(nn.Module, self.pair_norm)(pair)
            relative_position_embedder = self.relative_position_embedder
            if relative_position_embedder is not None:
                relative_features = generate_relative_position_features(
                    residue_index=batch.residue_index,
                    chain_index=batch.chain_index,
                    entity_index=batch.entity_index,
                    symmetry_index=batch.symmetry_index,
                    token_index=batch.token_index,
                    cyclic_period=batch.cyclic_period,
                    chain_gate="not_same_chain",
                    dtype=pair.dtype,
                )
                pair = pair + relative_position_embedder(relative_features)
                pair = pair + cast(nn.Linear, self.token_bond_projection)(
                    batch.token_bonds.unsqueeze(-1).to(pair.dtype)
                )
            single_inputs = normalized_inputs

        single = single.repeat_interleave(multiplicity, 0)
        single_inputs = single_inputs.repeat_interleave(multiplicity, 0)
        diffusion_single_projection = self.diffusion_single_projection
        if diffusion_single_projection is not None:
            single = single + diffusion_single_projection(
                cast(nn.Module, self.diffusion_single_norm)(cast(Tensor, single_diffusion))
            )

        pair = pair.repeat_interleave(multiplicity, 0)
        pair = pair + self.single_to_pair_row(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        single_to_pair_product_out = self.single_to_pair_product_out
        if single_to_pair_product_out is not None:
            pair = pair + single_to_pair_product_out(
                cast(nn.Linear, self.single_to_pair_product_left)(single_inputs).unsqueeze(-2)
                * cast(nn.Linear, self.single_to_pair_product_right)(single_inputs).unsqueeze(-3)
            )

        rep_atom_index = batch.token_to_rep_atom.repeat_interleave(multiplicity, 0)
        representative = pred_coords.gather(1, rep_atom_index.unsqueeze(-1).expand(-1, -1, pred_coords.shape[-1]))
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = batch.token_mask.repeat_interleave(multiplicity, 0)
        confidence_pair_mask = pair_mask_from_token_mask(token_mask)
        if self.input_embedder is not None:
            if batch.msa is not None:
                pair = pair + cast(MsaModule, self.msa_module)(
                    pair,
                    single_inputs,
                    msa=batch.msa,
                    has_deletion=cast(Tensor, batch.has_deletion),
                    deletion_value=cast(Tensor, batch.deletion_value),
                    is_paired=cast(Tensor, batch.msa_paired),
                    msa_mask=cast(Tensor, batch.msa_mask),
                    token_mask=batch.token_mask,
                )
            single, pair = self.pairformer_stack(single, pair, pair_mask=confidence_pair_mask)
            single = cast(nn.Module, self.final_single_norm)(cast(Tensor, single))
            pair = cast(nn.Module, self.final_pair_norm)(pair)
        else:
            single, pair = self.pairformer_stack(single, pair, pair_mask=confidence_pair_mask)
        logits = self.confidence_heads(cast(Tensor, single), pair)
        scores = _boltz1_confidence_scores(
            logits,
            distances=distances,
            batch=batch,
            pred_distogram_logits=pred_distogram_logits,
            pred_coords=pred_coords,
            multiplicity=multiplicity,
            ranking_plddt_weight=self.ranking_plddt_weight,
            ranking_ptm_weight=self.ranking_ptm_weight,
        )
        logits.update(scores)
        return logits

    def _imitate_trunk_inputs(self, batch: Boltz1Input) -> Tensor:
        r"""
        Re-run the confidence module's own input embedder to rebuild the ``single_inputs`` representation.

        Args:
            batch: The Boltz-1 input batch.

        Returns:
            The `[B, N, input_size]` re-embedded single-input representation.
        """

        input_embedder = cast(Boltz1InputEmbedder, self.input_embedder)
        return input_embedder(
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
            residue_type=batch.residue_type,
            msa_profile=batch.msa_profile,
            deletion_mean=batch.deletion_mean,
            pocket_feature=batch.pocket_feature,
        )

    def _confidence_initial_representations(
        self, batch: Boltz1Input, single_inputs: Tensor, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        r"""
        Build the pre-recycle single/pair initializations for the ``imitate_trunk`` confidence path.

        Args:
            batch: The Boltz-1 input batch.
            single_inputs: `[B, N, input_size]` re-embedded single-input representation.
            dtype: Dtype to build the relative-position features in.

        Returns:
            Tuple ``(single_init, pair_init)`` mirroring the main trunk initialization.
        """

        single_init = cast(nn.Linear, self.single_init_projection)(single_inputs)
        pair_init = cast(nn.Linear, self.pair_left_projection)(single_inputs).unsqueeze(-2)
        pair_init = pair_init + cast(nn.Linear, self.pair_right_projection)(single_inputs).unsqueeze(-3)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            cyclic_period=batch.cyclic_period,
            chain_gate="not_same_chain",
            dtype=dtype,
        )
        pair_init = pair_init + cast(RelativePositionEmbedder, self.relative_position_embedder)(relative_features)
        pair_init = pair_init + cast(nn.Linear, self.token_bond_projection)(
            batch.token_bonds.unsqueeze(-1).to(pair_init.dtype)
        )
        return single_init, pair_init


class Boltz1ConfidenceHeads(nn.Module):
    r"""
    Linear projections from the confidence single/pair representations to the Boltz-1 confidence logits.

    Args:
        single_size: Width of the confidence single representation.
        pair_size: Width of the confidence pair representation.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
        compute_pae: Whether to build and emit the PAE projection.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        compute_pae: bool = True,
    ) -> None:
        super().__init__()
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.plddt_projection = nn.Linear(single_size, num_plddt_bins, bias=False)
        self.resolved_projection = nn.Linear(single_size, 2, bias=False)
        self.pae_projection: nn.Linear | None = None
        if compute_pae:
            self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)

    def forward(self, single: Tensor, pair: Tensor) -> dict[str, Tensor]:
        r"""
        Project the confidence representations to logits.

        Args:
            single: `[B, N, single_size]` confidence single representation.
            pair: `[B, N, N, pair_size]` confidence pair representation.

        Returns:
            A dict with ``pde_logits``, ``plddt_logits``, ``resolved_logits``, and (when ``compute_pae``)
            ``pae_logits``.
        """

        logits = {
            "pde_logits": self.pde_projection(pair + pair.transpose(-3, -2)),
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }
        if self.pae_projection is not None:
            logits["pae_logits"] = self.pae_projection(pair)
        return logits


def _boltz1_steering_features(batch: Boltz1Input, *, atom_features: Tensor) -> dict[str, Tensor]:
    r"""
    Assemble the Boltz-1x steering feature dict consumed by the physical potentials.

    Args:
        batch: The Boltz-1 input batch (source of the constraint index/mask tensors).
        atom_features: `[B, M, atom_feature_size]` atom features; used to slice the reference-element one-hot.

    Returns:
        Mapping from steering feature name to a device-placed tensor (empty tensors when a feature is absent).
    """

    device = atom_features.device
    return {
        "atom_to_token": torch.nn.functional.one_hot(
            batch.atom_to_token.to(torch.long), num_classes=batch.token_mask.shape[-1]
        ).to(device=device, dtype=torch.float32),
        "asym_id": batch.chain_index.to(device=device, dtype=torch.long),
        "atom_pad_mask": batch.atom_mask.to(device=device),
        "ref_element": atom_features[..., 5:133].to(device=device),
        "rdkit_bounds_index": _feature_or_empty(
            batch.rdkit_bounds_index, (batch.atom_mask.shape[0], 2, 0), torch.long, device
        ),
        "rdkit_bounds_bond_mask": _feature_or_empty(
            batch.rdkit_bounds_bond_mask, (batch.atom_mask.shape[0], 0), torch.bool, device
        ),
        "rdkit_bounds_angle_mask": _feature_or_empty(
            batch.rdkit_bounds_angle_mask, (batch.atom_mask.shape[0], 0), torch.bool, device
        ),
        "rdkit_upper_bounds": _feature_or_empty(
            batch.rdkit_upper_bounds, (batch.atom_mask.shape[0], 0), atom_features.dtype, device
        ),
        "rdkit_lower_bounds": _feature_or_empty(
            batch.rdkit_lower_bounds, (batch.atom_mask.shape[0], 0), atom_features.dtype, device
        ),
        "chiral_atom_index": _feature_or_empty(
            batch.chiral_atom_index, (batch.atom_mask.shape[0], 4, 0), torch.long, device
        ),
        "chiral_atom_orientations": _feature_or_empty(
            batch.chiral_atom_orientations, (batch.atom_mask.shape[0], 0), torch.bool, device
        ),
        "stereo_bond_index": _feature_or_empty(
            batch.stereo_bond_index, (batch.atom_mask.shape[0], 4, 0), torch.long, device
        ),
        "stereo_bond_orientations": _feature_or_empty(
            batch.stereo_bond_orientations, (batch.atom_mask.shape[0], 0), torch.bool, device
        ),
        "planar_bond_index": _feature_or_empty(
            batch.planar_bond_index, (batch.atom_mask.shape[0], 6, 0), torch.long, device
        ),
        "connected_chain_index": _feature_or_empty(
            batch.connected_chain_index, (batch.atom_mask.shape[0], 2, 0), torch.long, device
        ),
        "connected_atom_index": _feature_or_empty(
            batch.connected_atom_index, (batch.atom_mask.shape[0], 2, 0), torch.long, device
        ),
        "symmetric_chain_index": _feature_or_empty(
            batch.symmetric_chain_index, (batch.atom_mask.shape[0], 2, 0), torch.long, device
        ),
        "contact_pair_index": _feature_or_empty(
            batch.contact_pair_index, (batch.atom_mask.shape[0], 2, 0), torch.long, device
        ),
        "contact_union_index": _feature_or_empty(
            batch.contact_union_index, (batch.atom_mask.shape[0], 0), torch.long, device
        ),
        "contact_negation_mask": _feature_or_empty(
            batch.contact_negation_mask, (batch.atom_mask.shape[0], 0), torch.bool, device
        ),
        "contact_thresholds": _feature_or_empty(
            batch.contact_thresholds, (batch.atom_mask.shape[0], 0), atom_features.dtype, device
        ),
    }


def _feature_or_empty(
    value: Tensor | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    r"""
    Return ``value`` cast to the target device/dtype, or an empty tensor of ``shape`` when it is ``None``.

    Args:
        value: The optional feature tensor.
        shape: Shape of the empty fallback tensor.
        dtype: Target dtype for the value or the fallback.
        device: Target device for the value or the fallback.

    Returns:
        The device/dtype-placed value, or an empty tensor of ``shape``.
    """

    if value is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return value.to(device=device, dtype=dtype)


class ParameterSchedule:
    r"""
    Abstract base for a steering-parameter schedule evaluated over normalized sampling time.

    Subclasses implement [`compute`][ofoldx.models.boltz1.model.ParameterSchedule.compute] to map a time
    in ``[0, 1]`` to a scalar parameter value.
    """

    def compute(self, time: float) -> float:
        r"""
        Evaluate the schedule at a normalized sampling time.

        Args:
            time: Normalized sampling time in ``[0, 1]``.

        Returns:
            The scheduled scalar parameter value.

        Raises:
            NotImplementedError: Always; subclasses override this.
        """

        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    r"""
    Interpolate between ``start`` and ``end`` with an exponential curvature ``alpha`` (linear when ``alpha=0``).

    Args:
        start: Value at ``time == 0``.
        end: Value at ``time == 1``.
        alpha: Exponential curvature; ``0`` gives linear interpolation.
    """

    def __init__(self, *, start: float, end: float, alpha: float) -> None:
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, time: float) -> float:
        r"""
        Evaluate the exponential interpolation at ``time``.

        Args:
            time: Normalized sampling time in ``[0, 1]``.

        Returns:
            The interpolated scalar value.

        Examples:
            >>> schedule = ExponentialInterpolation(start=1.0, end=5.0, alpha=0.0)
            >>> round(schedule.compute(0.5), 4)
            3.0
        """

        if self.alpha == 0:
            return self.start + (self.end - self.start) * time
        return self.start + (self.end - self.start) * (math.exp(self.alpha * time) - 1) / (math.exp(self.alpha) - 1)


class PiecewiseStepFunction(ParameterSchedule):
    r"""
    Step schedule that returns the value for the first ``thresholds`` bracket containing ``time``.

    Args:
        thresholds: Ascending time thresholds; there is one more ``values`` entry than thresholds.
        values: Per-bracket values, indexed by the number of thresholds ``time`` exceeds.
    """

    def __init__(self, *, thresholds: list[float], values: list[float]) -> None:
        self.thresholds = thresholds
        self.values = values

    def compute(self, time: float) -> float:
        r"""
        Return the step value for the bracket containing ``time``.

        Args:
            time: Normalized sampling time in ``[0, 1]``.

        Returns:
            The value for the bracket ``time`` falls into.

        Examples:
            >>> schedule = PiecewiseStepFunction(thresholds=[0.5], values=[0.1, 0.9])
            >>> schedule.compute(0.2)
            0.1
            >>> schedule.compute(0.8)
            0.9
        """

        index = 0
        while index < len(self.thresholds) and time > self.thresholds[index]:
            index += 1
        return self.values[index]


class Boltz1Potential:
    r"""
    Base class for a Boltz-1x physical potential over predicted coordinates.

    A potential exposes a per-time parameter schedule plus three subclass hooks: ``compute_args`` selects the
    relevant index/bound tensors from the feature dict, ``compute_variable`` turns coordinates into the
    geometric quantity (distance, dihedral, ...), and ``compute_function`` maps that quantity to an energy
    (and derivative). The base implements the energy and coordinate-gradient reductions over those hooks.

    Args:
        parameters: Mapping from parameter name to a constant or a
            [`ParameterSchedule`][ofoldx.models.boltz1.model.ParameterSchedule].
    """

    def __init__(self, parameters: Mapping[str, float | int | ParameterSchedule]) -> None:
        self.parameters = parameters

    def compute_parameters(self, time: float) -> dict[str, float | int]:
        r"""
        Resolve the schedule-valued parameters to scalars at a given sampling time.

        Args:
            time: Normalized sampling time in ``[0, 1]``.

        Returns:
            Mapping from parameter name to its scalar value at ``time``.
        """

        return {
            name: value.compute(time) if isinstance(value, ParameterSchedule) else value
            for name, value in self.parameters.items()
        }

    def compute(self, coords: Tensor, feats: dict[str, Tensor], parameters: Mapping[str, float | int]) -> Tensor:
        r"""
        Evaluate the total potential energy of ``coords`` under the resolved ``parameters``.

        Args:
            coords: `[..., M, 3]` predicted coordinates.
            feats: The Boltz-1x steering feature dict.
            parameters: Resolved scalar parameters for this step.

        Returns:
            The `[...]` summed potential energy over the selected constraints.
        """

        index, args, com_args = self.compute_args(feats, parameters)
        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
        local_coords = coords
        if com_args is not None:
            com_index, atom_pad_mask = com_args
            if not torch.any(atom_pad_mask):
                return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = local_coords[..., atom_pad_mask, :]
            local_coords = torch.zeros(
                (*unpad_coords.shape[:-2], int(unpad_com_index.max().item()) + 1, 3),
                dtype=coords.dtype,
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        value = self.compute_variable(local_coords, index, compute_gradient=False)
        energy = cast(Tensor, self.compute_function(value, *args, compute_derivative=False))
        return energy.sum(dim=-1)

    def compute_gradient(
        self,
        coords: Tensor,
        feats: dict[str, Tensor],
        parameters: Mapping[str, float | int],
    ) -> Tensor:
        r"""
        Evaluate the per-atom gradient of the potential energy with respect to ``coords``.

        Args:
            coords: `[..., M, 3]` predicted coordinates.
            feats: The Boltz-1x steering feature dict.
            parameters: Resolved scalar parameters for this step.

        Returns:
            The `[..., M, 3]` energy gradient, scattered back to per-atom coordinates.
        """

        index, args, com_args = self.compute_args(feats, parameters)
        if index.shape[1] == 0:
            return torch.zeros_like(coords)
        local_coords = coords
        com_index = None
        if com_args is not None:
            com_index, atom_pad_mask = com_args
            if not torch.any(atom_pad_mask):
                return torch.zeros_like(coords)
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = local_coords[..., atom_pad_mask, :]
            local_coords = torch.zeros(
                (*unpad_coords.shape[:-2], int(unpad_com_index.max().item()) + 1, 3),
                dtype=coords.dtype,
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        value, grad_value = self.compute_variable(local_coords, index, compute_gradient=True)
        _, d_energy = self.compute_function(value, *args, compute_derivative=True)
        prod = d_energy.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(start_dim=-3, end_dim=-2)
        if prod.dim() > 3:
            prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
        grad_atom = torch.zeros_like(local_coords).scatter_reduce(
            -2,
            index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*local_coords.shape[:-2], -1, 3)),
            prod,
            "sum",
        )
        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        return grad_atom

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], tuple[Tensor, Tensor] | None]:
        r"""
        Select the constraint index, energy-function args, and optional center-of-mass args from the features.

        Args:
            feats: The Boltz-1x steering feature dict.
            parameters: Resolved scalar parameters for this step.

        Returns:
            Tuple ``(index, function_args, com_args)`` consumed by the energy and gradient reductions.

        Raises:
            NotImplementedError: Always; subclasses override this.
        """

        raise NotImplementedError

    def compute_function(
        self, value: Tensor, *args: object, compute_derivative: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Map the geometric quantity to an energy (and optionally its derivative).

        Args:
            value: The geometric quantity (e.g. distances or dihedral angles).
            *args: The energy-function arguments from ``compute_args``.
            compute_derivative: When ``True``, also return the derivative with respect to ``value``.

        Returns:
            The energy, or an ``(energy, derivative)`` tuple when ``compute_derivative`` is set.

        Raises:
            NotImplementedError: Always; subclasses override this.
        """

        raise NotImplementedError

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Compute the geometric quantity (and optionally its coordinate gradient) from coordinates.

        Args:
            coords: `[..., M, 3]` (or center-of-mass) coordinates.
            index: The constraint index tensor selecting the participating atoms.
            compute_gradient: When ``True``, also return the per-atom gradient of the quantity.

        Returns:
            The geometric quantity, or a ``(quantity, gradient)`` tuple when ``compute_gradient`` is set.

        Raises:
            NotImplementedError: Always; subclasses override this.
        """

        raise NotImplementedError


class FlatBottomPotential(Boltz1Potential):
    r"""
    Flat-bottom energy: zero inside ``[lower_bounds, upper_bounds]`` and linear with slope ``k`` outside.

    Supplies the [`Boltz1Potential`][ofoldx.models.boltz1.model.Boltz1Potential] ``compute_function`` hook
    with a piecewise-linear well, so concrete potentials only provide the geometric quantity and its bounds. The
    energy is ``k * (lower_bounds - value)`` below the lower bound, ``k * (value - upper_bounds)`` above the upper
    bound, and zero in between; the derivative is the matching constant slope.
    """

    def compute_function(  # type: ignore[override]
        self,
        value: Tensor,
        k: Tensor,
        lower_bounds: Tensor | None,
        upper_bounds: Tensor | None,
        *,
        compute_derivative: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Evaluate the flat-bottom energy (and optionally its derivative) at ``value``.

        Args:
            value: The geometric quantity to penalize.
            k: Per-constraint linear penalty slope outside the bounds.
            lower_bounds: Lower bound (treated as ``-inf`` when ``None``).
            upper_bounds: Upper bound (treated as ``+inf`` when ``None``).
            compute_derivative: When ``True``, also return the derivative with respect to ``value``.

        Returns:
            The energy, or an ``(energy, derivative)`` tuple when ``compute_derivative`` is set.
        """

        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.to(device=value.device, dtype=value.dtype)
        upper_bounds = upper_bounds.to(device=value.device, dtype=value.dtype)
        k = k.to(device=value.device, dtype=value.dtype)
        negative = value < lower_bounds
        positive = value > upper_bounds
        energy = torch.zeros_like(value)
        energy[negative] = (k * (lower_bounds - value))[negative]
        energy[positive] = (k * (value - upper_bounds))[positive]
        if not compute_derivative:
            return energy
        derivative = torch.zeros_like(value)
        derivative[negative] = -k.expand_as(value)[negative]
        derivative[positive] = k.expand_as(value)[positive]
        return energy, derivative


class DistancePotential(Boltz1Potential):
    r"""
    Geometric quantity mixin computing pairwise atom distances from a 2-row index.

    Supplies the [`Boltz1Potential`][ofoldx.models.boltz1.model.Boltz1Potential] ``compute_variable`` hook: it
    turns a ``[2, P]`` atom-pair index into the ``P`` Euclidean distances (and their per-endpoint coordinate
    gradient), leaving the energy shape to whichever ``compute_function`` mixin it is combined with.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Compute pairwise distances (and optionally their coordinate gradient) for the indexed atom pairs.

        Args:
            coords: `[..., M, 3]` coordinates.
            index: `[2, P]` index selecting the two atoms of each pair.
            compute_gradient: When ``True``, also return the per-endpoint gradient.

        Returns:
            The `[..., P]` distances, or a ``(distances, gradient)`` tuple when ``compute_gradient`` is set.
        """

        delta = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        distance = torch.linalg.norm(delta, dim=-1).clamp_min(1e-8)
        if not compute_gradient:
            return distance
        direction = delta / distance.unsqueeze(-1)
        return distance, torch.stack((direction, -direction), dim=1)


class DihedralPotential(Boltz1Potential):
    r"""
    Geometric quantity mixin computing signed dihedral angles from a 4-row index.

    Supplies the [`Boltz1Potential`][ofoldx.models.boltz1.model.Boltz1Potential] ``compute_variable`` hook: it
    turns a ``[4, P]`` atom-quad index into the ``P`` signed dihedral angles (and their per-atom coordinate
    gradient), for use with any energy-function mixin bounding the angle.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Compute signed dihedral angles (and optionally their coordinate gradient) for the indexed atom quads.

        Args:
            coords: `[..., M, 3]` coordinates.
            index: `[4, P]` index selecting the four atoms of each dihedral.
            compute_gradient: When ``True``, also return the per-atom gradient.

        Returns:
            The `[..., P]` signed dihedral angles, or an ``(angles, gradient)`` tuple when
            ``compute_gradient`` is set.
        """

        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)
        r_kj_norm = torch.linalg.norm(r_kj, dim=-1).clamp_min(1e-8)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1).clamp_min(1e-8)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1).clamp_min(1e-8)
        sign_phi = torch.sign(r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)).squeeze(-1, -2)
        cosine = (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2) / (n_ijk_norm * n_jkl_norm)
        phi = sign_phi * torch.arccos(torch.clamp(cosine, -1 + 1e-8, 1 - 1e-8))
        if not compute_gradient:
            return phi
        a = ((r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)).unsqueeze(-1)
        b = ((r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)).unsqueeze(-1)
        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        return phi, torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)


class AbsDihedralPotential(DihedralPotential):
    r"""
    Dihedral variant that returns the absolute angle, with the gradient sign flipped where the angle is negative.

    Used by potentials that care about the magnitude of a dihedral rather than its sign (planarity, E/Z stereo):
    it folds the [`DihedralPotential`][ofoldx.models.boltz1.model.DihedralPotential] output through ``abs`` and
    negates the gradient wherever the underlying signed angle was negative, so the chain rule stays consistent.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Compute absolute dihedral angles (and optionally their sign-adjusted gradient).

        Args:
            coords: `[..., M, 3]` coordinates.
            index: `[4, P]` index selecting the four atoms of each dihedral.
            compute_gradient: When ``True``, also return the sign-adjusted per-atom gradient.

        Returns:
            The `[..., P]` absolute dihedral angles, or an ``(angles, gradient)`` tuple when
            ``compute_gradient`` is set.
        """

        if not compute_gradient:
            return torch.abs(cast(Tensor, super().compute_variable(coords, index, compute_gradient=False)))
        phi, grad = cast(tuple[Tensor, Tensor], super().compute_variable(coords, index, compute_gradient=True))
        grad = torch.where((phi < 0)[..., None, :, None], -grad, grad)
        return torch.abs(phi), grad


class PoseBustersPotential(DistancePotential, FlatBottomPotential):
    r"""
    PoseBusters bond/angle/clash distance potential over the RDKit bounds, with per-category buffers.

    Combines the distance geometry of [`DistancePotential`][ofoldx.models.boltz1.model.DistancePotential] with
    the [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` reads the
    RDKit bounds pairs and widens each pair's lower/upper bound by a per-category buffer (bond, angle, or clash) so
    predicted geometry that violates the RDKit-derived bounds is penalized.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Select the RDKit bounds pairs and widen their bounds per bond/angle/clash buffer.

        Args:
            feats: The Boltz-1x steering feature dict (RDKit bounds index/masks and bounds).
            parameters: Resolved scalar parameters (``bond_buffer``, ``angle_buffer``, ``clash_buffer``).

        Returns:
            Tuple ``(pair_index, (k, lower_bounds, upper_bounds), None)``.
        """

        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]
        if pair_index.shape[1] == 0:
            return pair_index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None
        bond_buffer = float(parameters["bond_buffer"])
        angle_buffer = float(parameters["angle_buffer"])
        clash_buffer = float(parameters["clash_buffer"])
        lower_bounds[bond_mask & ~angle_mask] *= 1.0 - bond_buffer
        upper_bounds[bond_mask & ~angle_mask] *= 1.0 + bond_buffer
        lower_bounds[~bond_mask & angle_mask] *= 1.0 - angle_buffer
        upper_bounds[~bond_mask & angle_mask] *= 1.0 + angle_buffer
        lower_bounds[bond_mask & angle_mask] *= 1.0 - min(bond_buffer, angle_buffer)
        upper_bounds[bond_mask & angle_mask] *= 1.0 + min(bond_buffer, angle_buffer)
        lower_bounds[~bond_mask & ~angle_mask] *= 1.0 - clash_buffer
        upper_bounds[~bond_mask & ~angle_mask] = float("inf")
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class ConnectionsPotential(DistancePotential, FlatBottomPotential):
    r"""
    Distance potential pulling cross-chain connected atom pairs within a fixed upper-bound buffer.

    Combines the distance geometry of [`DistancePotential`][ofoldx.models.boltz1.model.DistancePotential] with
    the [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` selects
    the explicitly connected atom pairs and sets only an upper bound, so the energy grows once a bonded connection
    is stretched beyond the buffer distance, keeping covalently linked chains together.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Select the connected atom pairs and set their upper-bound buffer.

        Args:
            feats: The Boltz-1x steering feature dict (connected atom index).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(pair_index, (k, None, upper_bounds), None)``.
        """

        pair_index = feats["connected_atom_index"][0]
        upper_bounds = torch.full((pair_index.shape[1],), float(parameters["buffer"]), device=pair_index.device)
        return pair_index, (torch.ones_like(upper_bounds), None, upper_bounds), None


class VDWOverlapPotential(DistancePotential, FlatBottomPotential):
    r"""
    Van-der-Waals clash potential penalizing non-bonded, cross-chain atom pairs closer than their summed radii.

    Combines the distance geometry of [`DistancePotential`][ofoldx.models.boltz1.model.DistancePotential] with
    the [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` enumerates
    non-bonded, cross-chain, multi-atom pairs and sets each pair's lower bound to the buffered sum of the two atoms'
    Van-der-Waals radii, so overlapping (clashing) atoms are pushed apart.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Enumerate non-bonded atom pairs and set their lower bound to the buffered summed VdW radii.

        Args:
            feats: The Boltz-1x steering feature dict (atom-to-token, chain ids, pad mask, reference element).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(pair_index, (k, lower_bounds, None), None)``; ``pair_index`` is empty when no atoms remain.
        """

        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()).squeeze(-1).long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        if not torch.any(atom_pad_mask):
            pair_index = torch.empty(2, 0, dtype=torch.long, device=atom_chain_id.device)
            empty = torch.empty(0, device=atom_chain_id.device)
            return pair_index, (empty, empty, None), None
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask], minlength=int(atom_chain_id.max().item()) + 1)
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]
        atom_vdw_radii = (feats["ref_element"].float() @ _vdw_radii(feats["ref_element"].device).unsqueeze(-1)).squeeze(
            -1
        )[0]
        pair_index = torch.triu_indices(atom_chain_id.shape[0], atom_chain_id.shape[0], 1, device=atom_chain_id.device)
        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]
        num_chains = int(atom_chain_id.max().item()) + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(num_chains, device=atom_chain_id.device, dtype=torch.bool)
        if connected_chain_index.shape[1] > 0:
            connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = True
            connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = True
        connected_chain_mask = connected_chain_matrix[atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]]
        pair_index = pair_index[:, pair_pad_mask & pair_ion_mask & ~connected_chain_mask]
        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (1.0 - float(parameters["buffer"]))
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), None


class SymmetricChainCOMPotential(DistancePotential, FlatBottomPotential):
    r"""
    Distance potential separating the centers of mass of same-entity symmetric chain pairs.

    Combines the distance geometry of [`DistancePotential`][ofoldx.models.boltz1.model.DistancePotential] with
    the [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well, but reduces coordinates to
    per-chain centers of mass first (via the ``com_args`` returned by ``compute_args``). A lower-bound buffer on the
    center-of-mass distance keeps symmetric copies of the same entity from collapsing onto each other.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], tuple[Tensor, Tensor] | None]:
        r"""
        Select the symmetric-chain pairs and emit the per-chain center-of-mass reduction args.

        Args:
            feats: The Boltz-1x steering feature dict (atom-to-token, chain ids, pad mask, symmetric pairs).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(pair_index, (k, lower_bounds, None), (atom_chain_id, atom_pad_mask))`` so the base reduces
            over chain centers of mass.
        """

        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()).squeeze(-1).long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        pair_index = feats["symmetric_chain_index"][0]
        if pair_index.shape[1] == 0 or not torch.any(atom_pad_mask):
            lower_bounds = torch.empty(0, dtype=torch.float32, device=atom_chain_id.device)
            return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), None
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask], minlength=int(atom_chain_id.max().item()) + 1)
        single_ion_mask = chain_sizes > 1
        pair_index = pair_index[:, single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]]
        lower_bounds = torch.full((pair_index.shape[1],), float(parameters["buffer"]), device=pair_index.device)
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), (atom_chain_id, atom_pad_mask)


class StereoBondPotential(AbsDihedralPotential, FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing E/Z stereo-bond orientations toward 0 or pi.

    Combines the absolute-angle geometry of
    [`AbsDihedralPotential`][ofoldx.models.boltz1.model.AbsDihedralPotential] with the
    [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` reads each
    stereo bond's target orientation and sets the absolute-angle bounds toward ``pi`` (E) or ``0`` (Z), penalizing
    predicted double-bond geometry that has the wrong E/Z configuration.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Select the stereo-bond dihedrals and set orientation-dependent absolute-angle bounds.

        Args:
            feats: The Boltz-1x steering feature dict (stereo-bond index and orientations).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(index, (k, lower_bounds, upper_bounds), None)``.
        """

        index = feats["stereo_bond_index"][0]
        orientation = feats["stereo_bond_orientations"][0].bool()
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
        lower_bounds[orientation] = torch.pi - float(parameters["buffer"])
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = float(parameters["buffer"])
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class ChiralAtomPotential(DihedralPotential, FlatBottomPotential):
    r"""
    Signed-dihedral potential enforcing chiral-center handedness by sign of the improper dihedral.

    Combines the signed-angle geometry of
    [`DihedralPotential`][ofoldx.models.boltz1.model.DihedralPotential] with the
    [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` reads each
    chiral center's target orientation and sets a one-sided signed-angle bound, so the improper dihedral is driven
    to the correct sign (R vs S handedness).
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Select the chiral-atom dihedrals and set orientation-dependent signed-angle bounds.

        Args:
            feats: The Boltz-1x steering feature dict (chiral-atom index and orientations).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(index, (k, lower_bounds, upper_bounds), None)``.
        """

        index = feats["chiral_atom_index"][0]
        orientation = feats["chiral_atom_orientations"][0].bool()
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
        lower_bounds[orientation] = float(parameters["buffer"])
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = -float(parameters["buffer"])
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class PlanarBondPotential(AbsDihedralPotential, FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing planarity around double bonds via improper dihedrals near zero.

    Combines the absolute-angle geometry of
    [`AbsDihedralPotential`][ofoldx.models.boltz1.model.AbsDihedralPotential] with the
    [`FlatBottomPotential`][ofoldx.models.boltz1.model.FlatBottomPotential] well: ``compute_args`` builds the
    improper dihedrals around each planar double bond and sets an upper bound near zero, penalizing out-of-plane
    geometry at conjugated/double-bond centers.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, tuple[object, ...], None]:
        r"""
        Build the improper dihedrals around each planar double bond and set their upper-bound buffer.

        Args:
            feats: The Boltz-1x steering feature dict (planar-bond index).
            parameters: Resolved scalar parameters (``buffer``).

        Returns:
            Tuple ``(improper_index, (k, None, upper_bounds), None)``.
        """

        double_bond_index = feats["planar_bond_index"][0].T
        improper_map = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 3]], device=double_bond_index.device).T
        improper_index = double_bond_index[:, improper_map].swapaxes(0, 1).flatten(start_dim=1)
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            float(parameters["buffer"]),
            device=improper_index.device,
        )
        return improper_index, (torch.ones_like(upper_bounds), None, upper_bounds), None


def _boltz1_potentials(steering_args: Mapping[str, object]) -> tuple[Boltz1Potential, ...]:
    r"""
    Instantiate the official Boltz-1x physical-potential set with their scheduled guidance/resampling weights.

    Args:
        steering_args: The normalized steering flags; ``physical_guidance_update`` toggles the guidance weights on.

    Returns:
        The tuple of configured Boltz-1x potentials.
    """

    guidance_weight = 1.0 if bool(steering_args["physical_guidance_update"]) else 0.0
    return (
        SymmetricChainCOMPotential(
            {
                "guidance_interval": 4,
                "guidance_weight": 0.5 * guidance_weight,
                "resampling_weight": 0.5,
                "buffer": ExponentialInterpolation(start=1.0, end=5.0, alpha=-2.0),
            }
        ),
        VDWOverlapPotential(
            {
                "guidance_interval": 5,
                "guidance_weight": (
                    PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]) if guidance_weight else 0.0
                ),
                "resampling_weight": PiecewiseStepFunction(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        ConnectionsPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.15 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 2.0,
            }
        ),
        PoseBustersPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 0.1,
                "bond_buffer": 0.20,
                "angle_buffer": 0.20,
                "clash_buffer": 0.15,
            }
        ),
        ChiralAtomPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.10 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        StereoBondPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        PlanarBondPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.26180,
            }
        ),
    )


_VDW_RADII = (
    1.2,
    1.4,
    2.2,
    1.9,
    1.8,
    1.7,
    1.6,
    1.55,
    1.5,
    1.54,
    2.4,
    2.2,
    2.1,
    2.1,
    1.95,
    1.8,
    1.8,
    1.88,
    2.8,
    2.4,
    2.3,
    2.15,
    2.05,
    2.05,
    2.05,
    2.05,
    2.0,
    2.0,
    2.0,
    2.1,
    2.1,
    2.1,
    2.05,
    1.9,
    1.9,
    2.02,
    2.9,
    2.55,
    2.4,
    2.3,
    2.15,
    2.1,
    2.05,
    2.05,
    2.0,
    2.05,
    2.1,
    2.2,
    2.2,
    2.25,
    2.2,
    2.1,
    2.1,
    2.16,
    3.0,
    2.7,
    2.5,
    2.48,
    2.47,
    2.45,
    2.43,
    2.42,
    2.4,
    2.38,
    2.37,
    2.35,
    2.33,
    2.32,
    2.3,
    2.28,
    2.27,
    2.25,
    2.2,
    2.1,
    2.05,
    2.0,
    2.0,
    2.05,
    2.1,
    2.05,
    2.2,
    2.3,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.4,
    2.0,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
)


def _vdw_radii(device: torch.device) -> Tensor:
    r"""
    Build the per-element (atomic-number-indexed) van-der-Waals radius lookup table.

    Args:
        device: Device to allocate the table on.

    Returns:
        A `[128]` float tensor indexed by atomic number (index 0 and unused entries are zero).
    """

    radii = torch.zeros(128, dtype=torch.float32, device=device)
    radii[1:119] = torch.tensor(_VDW_RADII, dtype=torch.float32, device=device)
    return radii


def _boltz1_confidence_scores(
    logits: dict[str, Tensor],
    *,
    distances: Tensor,
    batch: Boltz1Input,
    pred_distogram_logits: Tensor,
    pred_coords: Tensor,
    multiplicity: int,
    ranking_plddt_weight: float,
    ranking_ptm_weight: float,
) -> dict[str, Tensor]:
    r"""
    Derive Boltz-1 ranking metrics (complex/interface pLDDT, PDE, pTM/ipTM, confidence score) from the logits.

    Args:
        logits: The confidence logits dict from the confidence heads.
        distances: `[B*mult, N, N]` representative-atom distances used for interface contact masks.
        batch: The Boltz-1 input batch.
        pred_distogram_logits: `[B, N, N, num_bins]` trunk distogram logits for contact probabilities.
        pred_coords: `[B*mult, M, 3]` sampled coordinates used for pTM frames.
        multiplicity: Number of samples per input.
        ranking_plddt_weight: Weight on complex pLDDT in the combined confidence score.
        ranking_ptm_weight: Weight on pTM/ipTM in the combined confidence score.

    Returns:
        Mapping from score name to tensor (per-token metrics and per-sample complex/interface aggregates).
    """

    plddt = _expected_metric(logits["plddt_logits"])
    token_mask = batch.token_mask.repeat_interleave(multiplicity, 0).to(plddt)
    complex_plddt = (plddt * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1)

    chain_type = _chain_type_for_confidence(batch, multiplicity, plddt.device)
    is_ligand_token = (chain_type == 3).to(plddt)
    is_contact = (distances < 8).to(plddt)
    chain_index = batch.chain_index.repeat_interleave(multiplicity, 0)
    is_different_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(plddt)
    token_interface_mask = torch.max(
        is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
        dim=-1,
    ).values
    iplddt_weight = is_ligand_token * 2.0 + token_interface_mask
    complex_iplddt = (plddt * token_mask * iplddt_weight).sum(dim=-1) / (
        (token_mask * iplddt_weight).sum(dim=-1).clamp_min(torch.finfo(plddt.dtype).tiny)
    )

    pde = _expected_metric(logits["pde_logits"], end=32.0)
    pred_distogram_prob = torch.softmax(pred_distogram_logits, dim=-1).repeat_interleave(multiplicity, 0)
    contact_bins = min(20, pred_distogram_prob.shape[-1])
    prob_contact = pred_distogram_prob[..., :contact_bins].sum(dim=-1)
    eye = torch.eye(token_mask.shape[-1], dtype=token_mask.dtype, device=token_mask.device).unsqueeze(0)
    token_pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * (1 - eye) * prob_contact
    complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(dim=(1, 2)).clamp_min(
        torch.finfo(pde.dtype).tiny
    )
    token_interface_pair_mask = token_pair_mask * is_different_chain
    complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / token_interface_pair_mask.sum(
        dim=(1, 2)
    ).clamp_min(torch.finfo(pde.dtype).tiny)

    scores = {
        "plddt": plddt,
        "pde": pde,
        "complex_plddt": complex_plddt,
        "complex_iplddt": complex_iplddt,
        "complex_pde": complex_pde,
        "complex_ipde": complex_ipde,
    }
    pae_logits = logits.get("pae_logits")
    if pae_logits is not None:
        pae = _expected_metric(pae_logits, end=32.0)
        ptm_scores = _compute_boltz1_ptm_scores(
            pae_logits,
            pred_coords=pred_coords,
            batch=batch,
            multiplicity=multiplicity,
        )
        scores["pae"] = pae
        scores.update(ptm_scores)
        ptm_for_ranking = torch.where(
            torch.isclose(ptm_scores["iptm"], torch.zeros_like(ptm_scores["iptm"])),
            ptm_scores["ptm"],
            ptm_scores["iptm"],
        )
        scores["confidence_score"] = (ranking_plddt_weight * complex_plddt + ranking_ptm_weight * ptm_for_ranking) / (
            ranking_plddt_weight + ranking_ptm_weight
        )
    else:
        scores["confidence_score"] = complex_plddt
    return scores


def _expected_metric(logits: Tensor, *, end: float = 1.0) -> Tensor:
    r"""
    Compute the softmax-expected value of a binned metric over uniformly spaced bin centers.

    Args:
        logits: `[..., num_bins]` metric logits.
        end: Upper edge of the binning range; bin centers span ``(0, end)``.

    Returns:
        The `[...]` expected metric value.
    """

    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width,
        end=end,
        step=bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (torch.softmax(logits, dim=-1) * bounds.view(*((1,) * (logits.dim() - 1)), -1)).sum(dim=-1)


def _compute_boltz1_ptm_scores(
    pae_logits: Tensor,
    *,
    pred_coords: Tensor,
    batch: Boltz1Input,
    multiplicity: int,
) -> dict[str, Tensor]:
    r"""
    Compute pTM, ipTM, and the ligand/protein interface ipTM variants from PAE logits.

    Args:
        pae_logits: `[B*mult, N, N, num_bins]` PAE logits.
        pred_coords: `[B*mult, M, 3]` sampled coordinates used to build valid frames.
        batch: The Boltz-1 input batch.
        multiplicity: Number of samples per input.

    Returns:
        Mapping with ``ptm``, ``iptm``, ``ligand_iptm``, ``protein_iptm``, and the ``[B*mult, C, C]``
        per-chain-pair ``chain_pair_iptm`` matrix.
    """

    token_mask = batch.token_mask.repeat_interleave(multiplicity, 0).to(pae_logits)
    frame_mask = _predicted_frame_mask(pred_coords, batch=batch, multiplicity=multiplicity).to(pae_logits)
    chain_index = batch.chain_index.repeat_interleave(multiplicity, 0)
    pair_mask_ptm = frame_mask[:, :, None] * token_mask[:, None, :] * token_mask[:, :, None]
    pair_mask_iptm = pair_mask_ptm * (chain_index[:, None, :] != chain_index[:, :, None]).to(pae_logits)

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = torch.arange(
        start=0.5 * bin_width,
        end=32.0,
        step=bin_width,
        device=pae_logits.device,
        dtype=pae_logits.dtype,
    )
    num_residues = token_mask.sum(dim=-1, keepdim=True)
    tm_expected = tm_weight_from_pae(pae_logits, num_residues.unsqueeze(-1), bin_centers=pae_value)
    tiny = torch.finfo(tm_expected.dtype).tiny
    ptm = torch.max(
        (tm_expected * pair_mask_ptm).sum(dim=-1) / pair_mask_ptm.sum(dim=-1).clamp_min(tiny),
        dim=1,
    ).values
    iptm = torch.max(
        (tm_expected * pair_mask_iptm).sum(dim=-1) / pair_mask_iptm.sum(dim=-1).clamp_min(tiny),
        dim=1,
    ).values

    chain_type = _chain_type_for_confidence(batch, multiplicity, pae_logits.device)
    is_ligand = (chain_type == 3).to(pae_logits)
    is_protein = (chain_type == 0).to(pae_logits)
    ligand_mask = pair_mask_iptm * (
        is_ligand[:, :, None] * is_protein[:, None, :] + is_protein[:, :, None] * is_ligand[:, None, :]
    )
    protein_mask = pair_mask_iptm * (is_protein[:, :, None] * is_protein[:, None, :])
    ligand_iptm = torch.max(
        (tm_expected * ligand_mask).sum(dim=-1) / ligand_mask.sum(dim=-1).clamp_min(tiny),
        dim=1,
    ).values
    protein_iptm = torch.max(
        (tm_expected * protein_mask).sum(dim=-1) / protein_mask.sum(dim=-1).clamp_min(tiny),
        dim=1,
    ).values

    # Restore the released per-chain-pair interface pTM matrix (upstream ``chain_pair_iptm`` /
    # ``pair_chains_iptm``). Entry ``[i, j]`` is the max-over-frame interface pTM of query chain ``i``
    # against key chain ``j``; the matrix is not symmetrized, preserving the released query/key roles.
    chain_ids = torch.unique(chain_index).tolist()
    num_chains = len(chain_ids)
    chain_pair_iptm = tm_expected.new_zeros(tm_expected.shape[0], num_chains, num_chains)
    for row, query in enumerate(chain_ids):
        for column, key in enumerate(chain_ids):
            chain_pair_mask = (
                pair_mask_ptm
                * (chain_index[:, None, :] == key).to(pae_logits)
                * (chain_index[:, :, None] == query).to(pae_logits)
            )
            chain_pair_iptm[:, row, column] = torch.max(
                (tm_expected * chain_pair_mask).sum(dim=-1) / chain_pair_mask.sum(dim=-1).clamp_min(tiny),
                dim=1,
            ).values

    return {
        "ptm": ptm,
        "iptm": iptm,
        "ligand_iptm": ligand_iptm,
        "protein_iptm": protein_iptm,
        "chain_pair_iptm": chain_pair_iptm,
    }


def _predicted_frame_mask(pred_coords: Tensor, *, batch: Boltz1Input, multiplicity: int) -> Tensor:
    r"""
    Mask tokens whose predicted three-atom frame is valid (present and non-collinear).

    Args:
        pred_coords: `[B*mult, M, 3]` sampled coordinates.
        batch: The Boltz-1 input batch (source of frame atom indices and frame mask).
        multiplicity: Number of samples per input.

    Returns:
        A `[B*mult, N]` mask of tokens with a usable frame (falls back to the token mask when frames are absent).
    """

    if batch.frame_atom_index is None or batch.frame_mask is None:
        return batch.token_mask.repeat_interleave(multiplicity, 0)
    frame_index = batch.frame_atom_index.repeat_interleave(multiplicity, 0).to(
        device=pred_coords.device, dtype=torch.long
    )
    frame_mask = batch.frame_mask.repeat_interleave(multiplicity, 0).to(device=pred_coords.device)
    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    frames = pred_coords.gather(1, flat_index.unsqueeze(-1).expand(-1, -1, 3))
    frames = frames.reshape(*frame_index.shape, 3)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    cross_norm = torch.linalg.norm(torch.cross(first, second, dim=-1), dim=-1)
    non_collinear = (first_norm > 1e-6) & (second_norm > 1e-6) & (cross_norm > 1e-6)
    return frame_mask & non_collinear & batch.token_mask.repeat_interleave(multiplicity, 0).to(frame_mask.device)


def _chain_type_for_confidence(batch: Boltz1Input, multiplicity: int, device: torch.device) -> Tensor:
    r"""
    Resolve the per-token chain/molecule type used to classify ligand/protein tokens for confidence.

    Args:
        batch: The Boltz-1 input batch (prefers ``chain_type``, falls back to ``molecule_type``).
        multiplicity: Number of samples per input.
        device: Target device.

    Returns:
        A `[B*mult, N]` long tensor of chain types (all zeros when neither field is present).
    """

    chain_type = batch.chain_type if batch.chain_type is not None else batch.molecule_type
    if chain_type is None:
        return torch.zeros_like(batch.token_mask, dtype=torch.long).repeat_interleave(multiplicity, 0).to(device)
    return chain_type.repeat_interleave(multiplicity, 0).to(device=device, dtype=torch.long)


def _decode_boltz1_fold_result(output: Boltz1Output, batch: Boltz1Input) -> FoldResult:
    r"""
    Decode one raw [`Boltz1Output`][ofoldx.models.boltz1.model.Boltz1Output] into a canonical ``FoldResult``.

    Maps model atom order back onto the system's complete heavy-atom sites to build one ``Structure`` per
    sample, and lifts the confidence head's per-token metrics, token-pair PAE/PDE, and per-sample ranking
    scores onto the canonical value carriers aligned to the layout's token axis.

    Args:
        output: The raw folding output (sampled coords, confidence logits, derived ranking scores).
        batch: The Boltz-1 input batch whose ``layout`` names the aligned molecular identities.

    Returns:
        A [`FoldResult`][ofoldx.results.FoldResult] with one prediction per drawn sample.
    """

    layout = cast(Boltz1Layout, batch.layout)
    token_count = len(layout.token_sites)
    atom_count = len(layout.atom_sites)

    complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)

    coordinates = output.pred_coords[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
    num_samples = coordinates.shape[0]
    token_axis = layout.token_sites

    scores = output.confidence_scores or {}

    def per_sample(name: str) -> np.ndarray | None:
        value = scores.get(name)
        return None if value is None else value[0].to(device="cpu", dtype=torch.float32).numpy()

    plddt = per_sample("plddt")
    pae = per_sample("pae")
    pde = per_sample("pde")
    complex_plddt = per_sample("complex_plddt")
    complex_iplddt = per_sample("complex_iplddt")
    complex_pde = per_sample("complex_pde")
    complex_ipde = per_sample("complex_ipde")
    confidence_score = per_sample("confidence_score")
    ptm = per_sample("ptm")
    iptm = per_sample("iptm")
    ligand_iptm = per_sample("ligand_iptm")
    protein_iptm = per_sample("protein_iptm")
    chain_pair_iptm = per_sample("chain_pair_iptm")
    chain_axis = tuple(layout.system.instances.keys())

    resolved_probability: np.ndarray | None = None
    if output.resolved_logits is not None:
        resolved_probability = output.resolved_logits[0, :, :token_count].float().softmax(dim=-1)[..., 1].cpu().numpy()

    predictions: list[FoldPrediction] = []
    for sample_index in range(num_samples):
        complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        complete_coordinates[model_to_complete] = coordinates[sample_index]
        coordinate_mask[model_to_complete] = True
        structure = Structure(
            system=layout.system,
            sites=complete_sites,
            coordinates=complete_coordinates,
            coordinate_mask=coordinate_mask,
        )

        sample_scores: dict[str, Score] = {}
        if complex_plddt is not None:
            # Published pLDDT is on the cross-family 0-100 scale; the released expected pLDDT is 0-1.
            sample_scores["plddt"] = Score(float(complex_plddt[sample_index]) * 100.0, preferred="higher")
        if complex_iplddt is not None:
            sample_scores["iplddt"] = Score(float(complex_iplddt[sample_index]) * 100.0, preferred="higher")
        if complex_pde is not None:
            sample_scores["pde"] = Score(float(complex_pde[sample_index]), unit="angstrom", preferred="lower")
        if complex_ipde is not None:
            sample_scores["ipde"] = Score(float(complex_ipde[sample_index]), unit="angstrom", preferred="lower")
        if ptm is not None:
            sample_scores["ptm"] = Score(float(ptm[sample_index]), preferred="higher")
        if iptm is not None:
            sample_scores["iptm"] = Score(float(iptm[sample_index]), preferred="higher")
        if ligand_iptm is not None:
            sample_scores["ligand_iptm"] = Score(float(ligand_iptm[sample_index]), preferred="higher")
        if protein_iptm is not None:
            sample_scores["protein_iptm"] = Score(float(protein_iptm[sample_index]), preferred="higher")
        if confidence_score is not None:
            sample_scores["confidence_score"] = Score(float(confidence_score[sample_index]), preferred="higher")

        sample_metrics: dict[str, Metric] = {}
        if plddt is not None:
            sample_metrics["plddt"] = Metric(values=plddt[sample_index] * 100.0, axis=token_axis)
        if resolved_probability is not None:
            sample_metrics["resolved_probability"] = Metric(values=resolved_probability[sample_index], axis=token_axis)

        sample_pair_metrics: dict[str, PairMetric] = {}
        if pae is not None:
            sample_pair_metrics["pae"] = PairMetric(
                values=pae[sample_index], row_axis=token_axis, column_axis=token_axis, unit="angstrom"
            )
        if pde is not None:
            sample_pair_metrics["pde"] = PairMetric(
                values=pde[sample_index], row_axis=token_axis, column_axis=token_axis, unit="angstrom"
            )
        if chain_pair_iptm is not None:
            sample_pair_metrics["chain_pair_iptm"] = PairMetric(
                values=chain_pair_iptm[sample_index], row_axis=chain_axis, column_axis=chain_axis
            )

        predictions.append(
            FoldPrediction(
                structure=structure,
                scores=sample_scores,
                metrics=sample_metrics,
                pair_metrics=sample_pair_metrics,
            )
        )

    return FoldResult(predictions=tuple(predictions))


def _unflatten_optional_sample_axis(tensor: Tensor | None, batch_size: int, multiplicity: int) -> Tensor | None:
    r"""
    Unflatten the leading sample axis of an optional tensor, passing ``None`` through unchanged.

    Args:
        tensor: The tensor to reshape, or ``None``.
        batch_size: The batch size.
        multiplicity: The number of samples per input.
    Returns:
        The reshaped tensor, or ``None``.
    """

    if tensor is None:
        return None
    return _unflatten_sample_axis(tensor, batch_size, multiplicity)


def _unflatten_sample_axis(tensor: Tensor, batch_size: int, multiplicity: int) -> Tensor:
    r"""
    Split a flattened ``batch_size * multiplicity`` leading axis into ``(batch_size, multiplicity)``.

    Args:
        tensor: The tensor whose leading axis to split.
        batch_size: The batch size.
        multiplicity: The number of samples per input.
    Returns:
        The tensor reshaped to ``(batch_size, multiplicity, ...)``.
    """

    return tensor.reshape(batch_size, multiplicity, *tensor.shape[1:])


@dataclass(kw_only=True)
class _Boltz1Representations:
    r"""
    Boltz-1 trunk representations returned by the recycling trunk.

    Attributes:
        single_inputs: `[B, N, input_size]` single-input representation.
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
        distogram_logits: `[B, N, N, num_distogram_bins]` distogram-head logits.
        relative_position_encoding: `[B, N, N, pair_size]` relative-position encoding for diffusion conditioning.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


# ---------------------------------------------------------------------------
# Tensor boundary of the released Boltz-1 model graph.
#
# Formerly ``ofoldx/models/boltz1/contract.py``; folded in verbatim. Anchors the
# Boltz-1 tensor boundary in one place, matching the AlphaFold 2 and AlphaFold 3
# ``contract.py`` verticals: the pure-tensor input, the raw output, the canonical
# identity layout, the trunk-embedding value, and the family-owned pocket record.
# ---------------------------------------------------------------------------

# Boltz-1 pocket one-hot classes (config ``pocket_feature_size == 4``); mirrors upstream
# ``const.py`` ``pocket_contact_info`` labels: UNSPECIFIED, UNSELECTED, POCKET, BINDER.
_POCKET_UNSPECIFIED = 0
_POCKET_UNSELECTED = 1
_POCKET_POCKET = 2
_POCKET_BINDER = 3
_POCKET_SIZE = 4


@dataclass(frozen=True)
class Boltz1Pocket:
    r"""One released Boltz-1 unary pocket conditioning declaration.

    The released encoding admits a single binder chain and an explicitly declared
    contact-residue list; the 6 A cutoff is the training-time contact definition and
    is not recomputed at fold time. Training-only sampling controls are not
    representable.

    Attributes:
        binder: The canonical ``instance_id`` of the single binder chain.
        pocket: Canonical [`ComponentReference`][ofoldx.data.molecular.ComponentReference]
            values naming the pocket contact residues.
    """

    binder: str
    pocket: tuple[ComponentReference, ...] = ()


@dataclass(frozen=True)
class Boltz1Layout:
    r"""Canonical molecular identities aligned to one Boltz-1 tensor layout.

    Attributes:
        system: The canonical molecular system the tensor batch was lowered from.
        atom_sites: Model atom order aligned to the atom axis (``M`` heavy atoms, sans padding).
        token_sites: Model token order aligned to the token axis (``N`` tokens); polymer tokens carry
            ``(instance_id, monomer_index)`` and ligand-atom tokens carry the single ``AtomSite``.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]


@dataclass(kw_only=True)
class Boltz1Input:
    r"""
    Input tensors consumed by [`Boltz1Model`][ofoldx.models.boltz1.model.Boltz1Model].

    The contract is Boltz-1 specific: residue-token geometry, optional MSA features, molecule/frame fields used
    by confidence ranking, and optional token-level ``pocket_feature`` one-hot with classes ``UNSPECIFIED``,
    ``UNSELECTED``, ``POCKET``, and ``BINDER``. When omitted, the model uses all ``UNSPECIFIED`` pocket
    conditioning. All tensors carry a leading batch axis; ``N`` is the token count and ``M`` the atom count.

    Attributes:
        token_mask: `[B, N]` per-token validity mask.
        residue_index: `[B, N]` residue-position index.
        chain_index: `[B, N]` per-token chain index.
        entity_index: `[B, N]` per-token entity index.
        symmetry_index: `[B, N]` per-token symmetry-copy index.
        token_index: `[B, N]` per-token within-chain index.
        atom_features: `[B, M, atom_feature_size]` raw per-atom features.
        atom_to_token: `[B, M]` atom-to-token index map.
        atom_mask: `[B, M]` per-atom validity mask.
        reference_positions: `[B, M, 3]` reference (template) atom positions.
        reference_space_id: `[B, M]` per-atom reference-space identifier for window gating.
        representative_atom_mask: `[B, M]` mask of representative atoms per token.
        atom_token_index: `[B, M]` token index of each atom.
        token_to_rep_atom: `[B, N]` representative-atom index per token.
        residue_type: `[B, N, vocab_size]` residue-type one-hot.
        msa_profile: `[B, N, vocab_size]` MSA profile feature.
        deletion_mean: `[B, N]` mean-deletion feature.
        token_bonds: `[B, N, N]` inter-token bond indicator.
        molecule_type: Optional `[B, N]` per-token molecule type.
        chain_type: Optional `[B, N]` per-token chain type (preferred over ``molecule_type`` for confidence).
        target_positions: Optional `[B, M, 3]` ground-truth atom positions.
        target_mask: Optional `[B, M]` ground-truth validity mask.
        frame_atom_index: Optional `[B, N, 3]` three-atom frame indices per token.
        frame_mask: Optional `[B, N]` frame validity mask.
        cyclic_period: Optional `[B, N]` cyclic-period feature for relative positions.
        msa: Optional `[B, S, N]` MSA token grid.
        has_deletion: Optional `[B, S, N]` per-MSA deletion-present flag.
        deletion_value: Optional `[B, S, N]` per-MSA deletion count feature.
        msa_paired: Optional `[B, S, N]` cross-chain pairing flag.
        msa_mask: Optional `[B, S, N]` MSA validity mask.
        pocket_feature: Optional `[B, N, pocket_feature_size]` pocket one-hot conditioning.
        rdkit_bounds_index: Optional Boltz-1x RDKit bounds atom-pair index.
        rdkit_bounds_bond_mask: Optional Boltz-1x mask marking bond constraints.
        rdkit_bounds_angle_mask: Optional Boltz-1x mask marking angle constraints.
        rdkit_upper_bounds: Optional Boltz-1x RDKit upper distance bounds.
        rdkit_lower_bounds: Optional Boltz-1x RDKit lower distance bounds.
        chiral_atom_index: Optional Boltz-1x chiral-center atom-quad index.
        chiral_reference_mask: Optional Boltz-1x chiral reference mask.
        chiral_atom_orientations: Optional Boltz-1x chiral handedness flags.
        stereo_bond_index: Optional Boltz-1x stereo-bond atom-quad index.
        stereo_reference_mask: Optional Boltz-1x stereo reference mask.
        stereo_bond_orientations: Optional Boltz-1x E/Z orientation flags.
        planar_bond_index: Optional Boltz-1x planar double-bond atom index.
        connected_chain_index: Optional Boltz-1x cross-chain connected chain pairs.
        connected_atom_index: Optional Boltz-1x cross-chain connected atom pairs.
        symmetric_chain_index: Optional Boltz-1x same-entity symmetric chain pairs.
        contact_pair_index: Optional Boltz-1x contact atom-pair index.
        contact_union_index: Optional Boltz-1x contact union index.
        contact_negation_mask: Optional Boltz-1x contact negation mask.
        contact_thresholds: Optional Boltz-1x contact distance thresholds.
        layout: Optional canonical [`Boltz1Layout`][ofoldx.models.boltz1.model.Boltz1Layout] aligning the
            tensor axes to molecular identities; populated by ``Boltz1Processor`` and required by ``fold``.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    representative_atom_mask: Tensor
    atom_token_index: Tensor
    token_to_rep_atom: Tensor
    residue_type: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    molecule_type: Tensor | None = None
    chain_type: Tensor | None = None
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    pocket_feature: Tensor | None = None
    rdkit_bounds_index: Tensor | None = None
    rdkit_bounds_bond_mask: Tensor | None = None
    rdkit_bounds_angle_mask: Tensor | None = None
    rdkit_upper_bounds: Tensor | None = None
    rdkit_lower_bounds: Tensor | None = None
    chiral_atom_index: Tensor | None = None
    chiral_reference_mask: Tensor | None = None
    chiral_atom_orientations: Tensor | None = None
    stereo_bond_index: Tensor | None = None
    stereo_reference_mask: Tensor | None = None
    stereo_bond_orientations: Tensor | None = None
    planar_bond_index: Tensor | None = None
    connected_chain_index: Tensor | None = None
    connected_atom_index: Tensor | None = None
    symmetric_chain_index: Tensor | None = None
    contact_pair_index: Tensor | None = None
    contact_union_index: Tensor | None = None
    contact_negation_mask: Tensor | None = None
    contact_thresholds: Tensor | None = None
    layout: Boltz1Layout | None = None


@dataclass(kw_only=True)
class Boltz1Output:
    r"""
    Bundle the sampled coordinates, distogram logits, and confidence head outputs of one Boltz-1 forward pass.

    Attributes:
        pred_coords: `[B, num_samples, M, 3]` sampled atom coordinates.
        distogram_logits: `[B, N, N, num_distogram_bins]` trunk distogram logits.
        plddt_logits: Optional `[B, num_samples, N, num_plddt_bins]` pLDDT logits.
        pae_logits: Optional `[B, num_samples, N, N, num_pae_bins]` PAE logits.
        pde_logits: Optional `[B, num_samples, N, N, num_pde_bins]` PDE logits.
        resolved_logits: Optional `[B, num_samples, N, 2]` resolved-atom logits.
        confidence_scores: Optional mapping of derived per-sample confidence/ranking scores.
    """

    pred_coords: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor | None = None
    pae_logits: Tensor | None = None
    pde_logits: Tensor | None = None
    resolved_logits: Tensor | None = None
    confidence_scores: dict[str, Tensor] | None = None
    denoised_coords: Tensor | None = None
    augmented_target: Tensor | None = None
    noise_level: Tensor | None = None


@dataclass(kw_only=True)
class Boltz1Embeddings:
    r"""Raw Boltz-1 trunk representations surfaced by ``Boltz1Model.embeddings``.

    These carry vocabulary/layout axes excluded from canonical ``Metric``/``PairMetric``,
    so they are surfaced by a typed expert method rather than folded into ``FoldResult``.

    Attributes:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor
