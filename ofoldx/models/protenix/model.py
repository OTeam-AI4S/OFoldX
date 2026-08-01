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
Protenix default, structure-conditioned, and constraint predictors.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Callable, NamedTuple, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from multimolecule.graph import scatter_mean
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn
from transformers import EsmConfig, EsmModel

from ...data.molecular import AtomSite, ComponentRef
from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...metrics import (
    categorical_expectation,
    categorical_mean,
    categorical_probability_below,
    mean_partner_tm_scores,
    plddt_from_logits,
    predicted_tm_score,
    resolved_probability_from_logits,
    tm_bin_weights,
)
from ...model import Model, TensorCollection
from ...modules.atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowing,
)
from ...modules.attention import MultiheadAttention, MultiheadAttentionProjection, PerBlockPairBiasProjection
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoisingOutput,
    CoordinateEdmSampler,
    EdmScheduler,
    HeunCoordinateEdmSampler,
    denoise_coordinates,
)
from ...modules.diffusion.sampling import draw_normal, draw_normal_like
from ...modules.embedding import (
    FactorizedPairLinear,
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.feed_forward import FeedForward
from ...modules.functional import pair_mask_from_token_mask, pairwise_distance, weighted_mean
from ...modules.head import ConfidenceHead, ConfidencePairEmbedder, DistogramHead, PairConfidenceProjector
from ...modules.msa import MsaEncoder
from ...modules.pair import TrianglePairUpdateStack
from ...modules.pairformer import PairformerStack
from ...modules.potential import (
    indexed_distance,
    indexed_distance_with_gradient,
    indexed_signed_dihedral,
    indexed_signed_dihedral_with_gradient,
)
from ...modules.transformer import NormalizedKeyAtomCrossAttentionTransformerEncoderBlock
from ...modules.transition import Transition
from ..processing import (
    gather_model_coordinates,
    project_model_categories,
    project_model_coordinates,
    project_model_counts,
    project_model_features,
)
from ..vocabulary import FOLDING_SOURCE_TOKEN_INDEX, folding_source_map, map_token_feature_planes
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixStructureConditionConfig,
    ProtenixTemplateModelConfig,
)

#: The upstream residue alphabet (``STD_RESIDUES_WITH_GAP``: 20 amino acids, UNK, RNA, DNA, gap) in canonical
#: spelling; ``OTHER`` reads its ``X`` row, as upstream gives every per-atom ligand token UNK.
PROTENIX_SOURCE_TOKENS = (*"ARNDCQEGHILKMFPSTWYVX", *(f"RNA_{x}" for x in "AGCUN"), *(f"DNA_{x}" for x in "AGCTN"), "-")
# Atom-derived token state + residue class + MSA profile (both on the 72 canonical classes) + deletion mean.
_TOKEN_FEATURE_SIZE = 384 + 2 * len(TOKENS) + 1
_PROTENIX_SOURCE_TOKEN_FEATURE_SIZE = 384 + 2 * len(PROTENIX_SOURCE_TOKENS) + 1


def protenix_source_map() -> Tensor:
    r"""``[32, 72]``: per canonical class (column), its coefficients on the upstream alphabet (``folding_source_map``
    read in ``PROTENIX_SOURCE_TOKENS`` order)."""
    return folding_source_map()[[FOLDING_SOURCE_TOKEN_INDEX[token] for token in PROTENIX_SOURCE_TOKENS]]


class ProtenixModel(Model):
    r"""
    Predict biomolecular complex structures with Protenix.

    ``forward`` accepts [`ProtenixInput`][] and [`ProtenixStructureTarget`][]
    for stochastic training denoising and can omit auxiliary heads for a
    diffusion-only [`CoordinateDenoisingOutput`][]. ``denoise`` accepts explicit
    noisy coordinates, while ``sample`` runs reverse diffusion.

    Training with ragged inputs and nonzero shared dropout does not support
    full-graph or regional compilation of the dropout path.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = ProtenixConfig
    config: ProtenixConfig

    #: Whether this architecture has a template graph. The base and structure-condition architectures do
    #: not, and their paired Processor produces no template features to feed one; ``ProtenixTemplateModel``
    #: overrides this. Enabling ``config.template`` on an architecture that cannot feed it is rejected below.
    _supports_template = False
    #: Whether this architecture has the four-branch constraint graph. Only ``ProtenixConstraintModel`` does;
    #: the others' paired Processor exposes no constraint inputs, so the graph would sit silently unreachable.
    _supports_constraints = False

    def __init__(self, config: ProtenixConfig | None = None) -> None:
        config = ProtenixConfig() if config is None else config
        super().__init__(config)
        if config.template is not None and not self._supports_template:
            fail("value_invalid", RequestRef("config.template", "set"), invariant="architecture_has_no_template_graph")
        if config.constraints and not self._supports_constraints:
            fail("value_invalid", RequestRef("config.constraints", "True"), invariant="architecture_has_no_constraints")
        self._num_recycles = config.num_recycles
        self.register_buffer("token_feature_map", protenix_source_map(), persistent=False)
        self.input_embeddings = ProtenixInputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        esm = config.esm
        if esm is None:
            self.esm_encoder = None
            self.esm_projection = None
        else:
            self.esm_encoder = EsmModel(
                EsmConfig(
                    vocab_size=esm.vocab_size,
                    hidden_size=esm.hidden_size,
                    num_hidden_layers=esm.num_hidden_layers,
                    num_attention_heads=esm.num_attention_heads,
                    intermediate_size=esm.intermediate_size,
                    max_position_embeddings=esm.max_position_embeddings,
                    rope_theta=esm.rope_theta,
                    layer_norm_eps=esm.layer_norm_eps,
                    hidden_dropout_prob=esm.hidden_dropout_prob,
                    attention_probs_dropout_prob=esm.attention_probs_dropout_prob,
                    pad_token_id=esm.pad_token_id,
                    mask_token_id=esm.mask_token_id,
                    bos_token_id=esm.bos_token_id,
                    eos_token_id=esm.eos_token_id,
                    position_embedding_type=esm.position_embedding_type,
                    emb_layer_norm_before=esm.emb_layer_norm_before,
                    token_dropout=esm.token_dropout,
                ),
                add_pooling_layer=False,
            )
            # Protenix consumes only the final residue representation. The
            # separately released contact regression is not part of this graph.
            self.esm_encoder.contact_head = None
            self.esm_encoder.requires_grad_(False)
            self.esm_projection = nn.Linear(esm.hidden_size, _TOKEN_FEATURE_SIZE, bias=False)
        self.single_init_projection = nn.Linear(_TOKEN_FEATURE_SIZE, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.constraint_embeddings = ProtenixConstraintEmbeddings(config.pair_size) if config.constraints else None
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        template = config.template
        self.template_embedder = (
            None
            if template is None
            else ProtenixTemplateEmbedder(
                pair_size=config.pair_size,
                hidden_size=template.hidden_size,
                num_blocks=template.num_blocks,
                vocab_size=template.vocab_size,
                num_distogram_bins=template.num_distogram_bins,
                triangle_multiplication_hidden_size=template.triangle_multiplication_hidden_size,
                triangle_attention_head_size=template.triangle_attention_head_size,
                num_triangle_attention_heads=template.num_triangle_attention_heads,
                expansion=template.expansion,
                dropout=template.dropout,
            )
        )
        self.msa_trunk = MsaEncoder(
            msa_size=config.trunk.msa_size,
            pair_size=config.pair_size,
            msa_vocab_size=len(TOKENS),
            token_feature_size=_TOKEN_FEATURE_SIZE,
            num_stages=config.trunk.num_msa_stages,
            outer_product_hidden_size=config.trunk.msa_outer_product_hidden_size,
            num_msa_attention_heads=config.trunk.num_msa_attention_heads,
            msa_attention_head_size=config.trunk.msa_attention_head_size,
            triangle_multiplication_hidden_size=config.trunk.msa_triangle_multiplication_hidden_size,
            triangle_attention_head_size=config.trunk.msa_triangle_attention_head_size,
            num_triangle_attention_heads=config.trunk.num_msa_triangle_attention_heads,
            expansion=config.trunk.msa_expansion,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
        )
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pairformer_stack = PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            triangle_multiplication_hidden_size=config.trunk.pairformer_triangle_multiplication_hidden_size,
            triangle_attention_head_size=config.trunk.pairformer_triangle_attention_head_size,
            num_triangle_attention_heads=config.trunk.num_pairformer_triangle_attention_heads,
            num_pair_attention_heads=config.trunk.num_pair_attention_heads,
            expansion=config.trunk.pairformer_expansion,
            dropout=config.trunk.pairformer_dropout,
            checkpoint_group_size=4,
        )
        self.coordinate_denoiser = CoordinateDenoiser(
            num_encoder_blocks=config.denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=config.denoiser.num_token_encoder_attention_heads,
            sigma_data=config.denoiser.sigma_data,
            atom_hidden_size=config.denoiser.atom_hidden_size,
            atom_pair_hidden_size=config.denoiser.atom_pair_hidden_size,
            token_hidden_size=config.denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=_PROTENIX_SOURCE_TOKEN_FEATURE_SIZE,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        sampler_type = {"euler": CoordinateEdmSampler, "heun": HeunCoordinateEdmSampler}[config.sampler.solver]
        self.coordinate_sampler = sampler_type(
            scheduler=EdmScheduler(
                max_noise_level=config.sampler.max_noise_level,
                min_noise_level=config.sampler.min_noise_level,
                rho=config.sampler.rho,
                sigma_data=config.denoiser.sigma_data,
            ),
            churn_rate=config.sampler.churn_rate,
            min_churn_noise_level=config.sampler.min_churn_noise_level,
            noise_scale=config.sampler.noise_scale,
            step_scale=config.sampler.step_scale,
        )
        self._num_sampling_steps = config.sampler.num_sampling_steps
        self.confidence_head = ProtenixConfidenceHead(
            token_feature_size=_TOKEN_FEATURE_SIZE,
            single_size=config.single_size,
            pair_size=config.pair_size,
            distance_bin_start=config.confidence.distance_bin_start,
            distance_bin_end=config.confidence.distance_bin_end,
            distance_bin_step=config.confidence.distance_bin_step,
            num_plddt_bins=config.confidence.num_plddt_bins,
            num_resolved_bins=config.confidence.num_resolved_bins,
            max_atoms_per_token=config.confidence.max_atoms_per_token,
            num_pae_bins=config.confidence.num_pae_bins,
            num_pde_bins=config.confidence.num_pde_bins,
            single_max_abs=config.confidence.single_max_abs,
        )
        self.confidence_pairformer_stack = PairformerStack(
            num_blocks=config.confidence.num_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_head_size=config.confidence.triangle_attention_head_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            expansion=config.confidence.expansion,
            dropout=config.confidence.pairformer_dropout,
        )
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Zero-initialize recycling, confidence, and constraint projections.

        Returns:
            ``None``.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        if self.esm_projection is not None:
            nn.init.zeros_(self.esm_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)
        if self.constraint_embeddings is not None:
            self.constraint_embeddings.reset_parameters()

    def forward(
        self,
        input: ProtenixInput,
        target: ProtenixStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        compute_auxiliary_heads: bool = True,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        msa_indices: tuple[Tensor, ...] | None = None,
        use_conditioning: bool = True,
        generator: torch.Generator | None = None,
    ) -> ProtenixOutput | CoordinateDenoisingOutput:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            target: Heavy-atom truth-pool coordinates and their current atom-axis
                mapping.
            num_samples: Number ``S`` of independently corrupted copies per input.
            sample_chunk_size: Maximum number of coordinate samples denoised
                together. ``None`` processes all samples together.
            compute_auxiliary_heads: Whether to compute the distogram and
                confidence heads. Disable for a diffusion-only objective.
            synchronize_noise_levels: Whether samples from one example share sigma.
            coordinate_augmentation: Whether to apply random rigid augmentation.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_indices: Explicit MSA row indices for each recycling iteration.
                ``None`` uses every prepared MSA row.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.
            generator: Optional generator for rigid augmentation, noise levels,
                and Gaussian coordinate noise.

        Returns:
            The complete [`ProtenixOutput`][] by default. When
                ``compute_auxiliary_heads=False``, only the denoised coordinates,
                augmented targets, and noise levels required by a diffusion objective
                are returned.

        Raises:
            ValueError: If ``num_samples``, ``sample_chunk_size``, or
                ``num_recycles`` is not positive, or if ``msa_indices`` does
                not contain one tensor per recycling iteration.
        """
        input = normalize_protenix_input(
            input,
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            max_atoms_per_token=self.confidence_head.atom_projector.plddt_weight.shape[0],
        )

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        if msa_indices is not None and len(msa_indices) != num_recycles:
            fail("value_invalid", RequestRef("msa_indices", str(len(msa_indices))), num_recycles=num_recycles)
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            reference_positions=input.reference_positions,
            msa_indices=msa_indices,
        )
        target_coordinates, target_mask = _gather_protenix_structure_target(target, input)
        target_coordinates = project_model_coordinates(
            target_coordinates, target_mask, dtype=representations.single.dtype, feature="Protenix training target"
        ).to(device=representations.single.device)
        target_mask = target_mask.to(device=target_coordinates.device)
        augmented_target_coordinates = self.coordinate_sampler.augment_target(
            target_coordinates,
            num_samples=num_samples,
            atom_mask=target_mask,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise_template = target_mask.any(dim=-1).to(dtype=augmented_target_coordinates.dtype).unsqueeze(-1)
        if not synchronize_noise_levels:
            noise_template = noise_template.expand(-1, num_samples)
        normal = draw_normal_like(noise_template, generator)
        noise_level = (
            self.coordinate_denoiser.preconditioner.sigma_data * (noise_log_mean + noise_log_std * normal).exp()
        )
        if synchronize_noise_levels:
            noise_level = noise_level.expand(-1, num_samples)
        coordinate_noise = draw_normal_like(augmented_target_coordinates, generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * coordinate_noise
        denoise_context = self._denoise_context(
            input,
            representations,
            reference_positions=input.reference_positions,
            use_conditioning=use_conditioning,
        )
        denoise = self.coordinate_denoiser.prepare(denoise_context)
        coordinates = denoise_coordinates(
            denoise,
            context=denoise_context,
            noisy_coords=noisy_coordinates,
            noise_level=noise_level,
            sample_chunk_size=sample_chunk_size,
        )
        if not compute_auxiliary_heads:
            return CoordinateDenoisingOutput(
                coordinates=coordinates,
                augmented_target_coordinates=augmented_target_coordinates,
                noise_level=noise_level,
                batch_size=[augmented_target_coordinates.shape[0]],
            )
        raw_output = self._assemble_output(input, representations, coordinates)
        return ProtenixOutput(
            coordinates=raw_output.coordinates,
            distogram_logits=raw_output.distogram_logits,
            plddt_logits=raw_output.plddt_logits,
            pae_logits=raw_output.pae_logits,
            pde_logits=raw_output.pde_logits,
            resolved_logits=raw_output.resolved_logits,
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=noise_level,
            batch_size=[augmented_target_coordinates.shape[0]],
        )

    def denoise(
        self,
        input: ProtenixInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        num_recycles: int | None = None,
        use_conditioning: bool = True,
    ) -> ProtenixSamplingOutput:
        r"""
        Denoise atom coordinates at explicitly supplied EDM noise levels.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            noisy_coordinates: Noisy coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: EDM noise standard deviations with shape ``(B, S)`` in
                angstroms.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.

        Returns:
            Raw denoised coordinates, distogram, and confidence logits.

        Raises:
            ValueError: If ``num_recycles`` is not positive.
        """
        input = normalize_protenix_input(
            input,
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            max_atoms_per_token=self.confidence_head.atom_projector.plddt_weight.shape[0],
        )
        noisy_coordinates = project_model_coordinates(
            noisy_coordinates,
            input.atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(noisy_coordinates[..., 0], dtype=torch.bool),
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            feature="Protenix noisy coordinates",
        )

        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            reference_positions=input.reference_positions,
        )
        coordinates = self.coordinate_denoiser(
            self._denoise_context(
                input,
                representations,
                reference_positions=input.reference_positions,
                use_conditioning=use_conditioning,
            ).to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return self._assemble_output(input, representations, coordinates)

    @torch.no_grad()
    def sample(
        self,
        input: ProtenixInput,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> ProtenixSamplingOutput:
        r"""
        Run reverse diffusion and return raw coordinates and prediction heads.

        Args:
            input: Processor output for one or more molecular assemblies.
            num_samples: Number of coordinate samples. ``None`` uses five.
            generator: Optional generator for MSA subsampling, initial coordinates,
                stochastic churn, and per-step rigid augmentations.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the configured value.
            step_scale: Per-call solver-step multiplier. ``None`` uses the configured value.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Raw sampled coordinates, distogram, and confidence logits.

        Raises:
            ValueError: If a sampling or guidance iteration count is invalid,
                concrete guidance contains no potential term, or guidance is
                paired with input that has no prepared TFG geometry.
        """
        input = normalize_protenix_input(
            input,
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            max_atoms_per_token=self.confidence_head.atom_projector.plddt_weight.shape[0],
        )

        num_samples = 5 if num_samples is None else operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            fail("value_invalid", RequestRef("num_sampling_steps", str(num_sampling_steps)))
        tfg_potential = None
        if tfg_guidance is not None:
            if input.tfg_features is None:
                fail("value_invalid", RequestRef("tfg_guidance", "set"), invariant="input_without_tfg_features")
            tfg_potential = ProtenixTfgPotential(input.tfg_features, tfg_guidance, backend=self.backend)

        reference_positions = input.reference_positions
        msa_indices = tuple(
            self._sample_msa_indices(input.msa_tokens.shape[-2], input.msa_tokens.device, generator=generator)
            for _ in range(num_recycles)
        )
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            reference_positions=reference_positions,
            msa_indices=msa_indices,
        )
        coordinates = self._predict_coords(
            input,
            representations=representations,
            reference_positions=reference_positions,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            generator=generator,
            tfg_potential=tfg_potential,
        )
        return self._assemble_output(input, representations, coordinates)

    def _assemble_output(
        self,
        input: ProtenixInput,
        representations: _ProtenixTrunkOutput,
        coordinates: Tensor,
    ) -> ProtenixSamplingOutput:
        r"""
        Compute the distogram and confidence predictions.
        """

        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            representations.single,
            pair=representations.pair,
            token_embedding=representations.token_embedding,
            pred_coords=coordinates,
            token_representative_atom_index=input.token_representative_atom_index,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
            pairformer_stack=self.confidence_pairformer_stack,
        )
        return ProtenixSamplingOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(representations.pair),
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            batch_size=[coordinates.shape[0]],
        )

    def _denoise_context(
        self,
        input: ProtenixInput,
        representations: _ProtenixTrunkOutput,
        *,
        reference_positions: Tensor,
        use_conditioning: bool = True,
    ) -> CoordinateDenoiseContext:
        r"""
        Assemble conditioning shared by forward and the sampler.
        """

        conditioning_single = representations.single if use_conditioning else 0 * representations.single
        conditioning_pair = representations.pair if use_conditioning else 0 * representations.pair
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            # the denoiser's single conditioner normalises the whole vector: it reads the canonical planes upstream-wide
            token_embedding=map_token_feature_planes(representations.token_embedding, self.token_feature_map, 384),
            single=representations.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=representations.relative_features,
            local_geometry=self.coordinate_denoiser.windowing(
                reference_positions,
                input.reference_space_id,
                input.atom_mask,
                input.reference_mask,
            ),
        )

    def _run_trunk(
        self,
        input: ProtenixInput,
        *,
        num_recycles: int,
        reference_positions: Tensor,
        msa_indices: tuple[Tensor, ...] | None = None,
        pair_recycling_dropout: float = 0.0,
    ) -> _ProtenixTrunkOutput:
        r"""
        Embed the inputs and run the recycling trunk.
        """

        token_embedding = self.input_embeddings(
            input.atom_features,
            token_features=input.token_features,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_to_token=input.atom_to_token,
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embeddings.windowing(
                reference_positions,
                input.reference_space_id,
                input.atom_mask,
                input.reference_mask,
            ),
        )
        if self.esm_projection is not None:
            token_embedding = token_embedding + self.esm_projection(self._esm_token_embedding(input))
        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            dtype=single_init.dtype,
        )
        pair_init = self._initial_pair(single_init, input, relative_features)
        pair_mask = pair_mask_from_token_mask(input.token_mask, backend=self.backend)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            num_recycles,
            msa_indices=msa_indices,
            pair_recycling_dropout=pair_recycling_dropout,
        )
        return _ProtenixTrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
        )

    def _esm_token_embedding(self, input: ProtenixInput) -> Tensor:
        r"""
        Run the frozen ESM2 child and gather one residue representation per Protenix token.
        """

        esm_encoder = cast(EsmModel, self.esm_encoder)
        batch_size, num_sequences, sequence_length = input.esm_tokens.shape
        if num_sequences == 0:
            shape = (*input.token_features.shape[:-1], esm_encoder.config.hidden_size)
            return input.token_features.new_zeros(shape)

        sequence_embedding = esm_encoder(
            input.esm_tokens.reshape(batch_size * num_sequences, sequence_length),
            attention_mask=input.esm_mask.reshape(batch_size * num_sequences, sequence_length),
            return_dict=True,
        ).last_hidden_state
        sequence_embedding = sequence_embedding.reshape(
            batch_size,
            num_sequences * sequence_length,
            sequence_embedding.shape[-1],
        )
        token_embedding = sequence_embedding.gather(
            -2,
            input.token_to_esm_residue.unsqueeze(-1).expand(
                *input.token_to_esm_residue.shape,
                sequence_embedding.shape[-1],
            ),
        )
        return token_embedding * input.token_has_esm.unsqueeze(-1).to(dtype=token_embedding.dtype)

    def _run_recycling(
        self,
        input: ProtenixInput,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        num_recycles: int,
        *,
        msa_indices: tuple[Tensor, ...] | None,
        pair_recycling_dropout: float,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run the recycling loop.
        """

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(num_recycles - 1):
            with torch.no_grad():
                single, pair = self._run_recycling_step(
                    input,
                    token_embedding,
                    single_init,
                    pair_init,
                    single,
                    pair,
                    pair_mask,
                    indices=None if msa_indices is None else msa_indices[cycle],
                    pair_recycling_dropout=pair_recycling_dropout,
                )
        single, pair = self._run_recycling_step(
            input,
            token_embedding,
            single_init,
            pair_init,
            single,
            pair,
            pair_mask,
            indices=None if msa_indices is None else msa_indices[-1],
            pair_recycling_dropout=pair_recycling_dropout,
        )
        return single, pair

    def _run_recycling_step(
        self,
        input: ProtenixInput,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        single: Tensor,
        pair: Tensor,
        pair_mask: Tensor,
        *,
        indices: Tensor | None,
        pair_recycling_dropout: float,
    ) -> tuple[Tensor, Tensor]:
        pair = self._recycled_pair(input, pair_init, pair, pair_recycling_dropout)
        if self.template_embedder is not None:
            pair = pair + self.template_embedder(
                pair,
                cast(ProtenixTemplateFeatures, input.template_features),
                same_chain_mask=input.chain_index[..., :, None] == input.chain_index[..., None, :],
                pair_mask=pair_mask,
            )
        msa_tokens, msa_continuous_features = self._msa_features(input, indices)
        msa_args = (
            msa_tokens,
            msa_continuous_features,
            pair,
            token_embedding,
            pair_mask,
            input.token_mask.bool().unsqueeze(-2) & torch.ones_like(msa_tokens, dtype=torch.bool),
        )
        if self.msa_trunk.gradient_checkpointing and self.msa_trunk.training and torch.is_grad_enabled():
            pair = self.msa_trunk._gradient_checkpointing_func(self._run_msa_pair, *msa_args)
        else:
            pair = self._run_msa_pair(*msa_args)
        del msa_args
        single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
        return self.pairformer_stack(single, pair, pair_mask=pair_mask)

    def _run_msa_pair(
        self,
        msa_tokens: Tensor,
        msa_continuous_features: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pair_mask: Tensor | None,
        msa_mask: Tensor,
    ) -> Tensor:
        return self.msa_trunk(
            msa_tokens,
            msa_continuous_features,
            pair,
            token_embedding,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
        )[1]

    def _recycled_pair(
        self,
        input: ProtenixInput,
        pair_init: Tensor,
        pair: Tensor,
        pair_recycling_dropout: float,
    ) -> Tensor:
        del input
        pair_update = self.pair_recycle_projection(self.pair_recycle_norm(pair))
        if pair_recycling_dropout:
            pair_update = F.dropout(pair_update, p=pair_recycling_dropout, training=True)
        return pair_init + pair_update

    def _msa_features(self, input: ProtenixInput, indices: Tensor | None) -> tuple[Tensor, Tensor]:
        r"""
        Select compact categorical and continuous MSA features.
        """

        if indices is None:
            msa_tokens = input.msa_tokens
            deletion_counts = input.msa_deletion_counts
        else:
            msa_tokens = torch.index_select(input.msa_tokens, -2, indices)
            deletion_counts = torch.index_select(input.msa_deletion_counts, -2, indices)
        deletion_counts = deletion_counts.to(dtype=torch.float64)
        has_deletion = deletion_counts.clamp(0, 1)
        deletion_value = torch.atan(deletion_counts / 3.0) * (2.0 / math.pi)
        continuous_features = torch.stack(
            (has_deletion, deletion_value),
            dim=-1,
        )
        return msa_tokens, continuous_features.to(dtype=self.msa_trunk.msa_projection.weight.dtype)

    def _initial_pair(
        self,
        single: Tensor,
        input: ProtenixInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        r"""
        Build the initial pair representation and optional constraint bias.
        """

        left = self.pair_left_projection(single).unsqueeze(-2)
        right = self.pair_right_projection(single).unsqueeze(-3)
        token_bonds = input.token_bonds.to(device=single.device, dtype=single.dtype)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1))
        if self.constraint_embeddings is not None:
            pair = pair + self.constraint_embeddings(
                input.pocket,
                input.token_contact,
                input.atom_contact,
                input.substructure,
            )
        pair_mask = pair_mask_from_token_mask(input.token_mask, backend=self.backend)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _predict_coords(
        self,
        input: ProtenixInput,
        *,
        representations: _ProtenixTrunkOutput,
        reference_positions: Tensor,
        num_samples: int,
        num_sampling_steps: int,
        churn_rate: float | None,
        step_scale: float | None,
        generator: torch.Generator | None,
        tfg_potential: ProtenixTfgPotential | None,
    ) -> Tensor:
        r"""
        Run sequential EDM sampling.
        """

        context = self._denoise_context(
            input,
            representations,
            reference_positions=reference_positions,
        )
        if tfg_potential is not None:
            return self._predict_coords_with_tfg(
                context=context,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                churn_rate=churn_rate,
                step_scale=step_scale,
                generator=generator,
                potential=tfg_potential,
            )
        return self.coordinate_sampler(
            self.coordinate_denoiser.prepare(context),
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            generator=generator,
        )

    def _predict_coords_with_tfg(
        self,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        churn_rate: float | None,
        step_scale: float | None,
        generator: torch.Generator | None,
        potential: ProtenixTfgPotential,
    ) -> Tensor:
        r"""
        Sample with Protenix's released training-free chemical guidance.

        TFG is a sampling operation over the ordinary denoiser graph.
        """

        guidance = potential.guidance
        churn_rate = self.coordinate_sampler.churn_rate if churn_rate is None else churn_rate
        step_scale = self.coordinate_sampler.step_scale if step_scale is None else step_scale
        noise_schedule = self.coordinate_sampler.scheduler(
            num_sampling_steps=num_sampling_steps,
            device=context.device,
            dtype=context.dtype,
        )
        denoise = self.coordinate_denoiser.prepare(context)

        reference = context.reference_positions
        noise_template = reference.unsqueeze(-3).expand(
            *context.batch_shape,
            num_samples,
            reference.shape[-2],
            3,
        )
        coordinates = noise_schedule[0] * draw_normal_like(noise_template, generator)
        for step_index, (sigma_previous, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            coordinates = self.coordinate_sampler.augment_target(
                coordinates,
                num_samples=1,
                generator=generator,
            ).squeeze(dim=-3)
            churn = torch.where(
                sigma_next > self.coordinate_sampler.min_churn_noise_level,
                sigma_next.new_tensor(churn_rate),
                sigma_next.new_zeros(()),
            )
            sigma_hat = sigma_previous * (churn + 1)
            delta_noise = torch.sqrt(sigma_hat**2 - sigma_previous**2)
            noisy_coordinates = coordinates + (
                self.coordinate_sampler.noise_scale * delta_noise * draw_normal_like(coordinates, generator)
            )
            noise_level = (
                sigma_hat.reshape((1,) * (len(context.batch_shape) + 1))
                .expand(
                    *context.batch_shape,
                    num_samples,
                )
                .to(dtype=coordinates.dtype)
            )
            normalized_time = 1.0 - step_index / max(1, num_sampling_steps)
            monte_carlo_noise = _protenix_tfg_monte_carlo_noise(
                guidance,
                noisy_coordinates,
                generator=generator,
            )
            working_coordinates = noisy_coordinates
            next_coordinates = working_coordinates
            for outer_index in range(guidance.num_outer_steps):
                if guidance.rho == 0.0:
                    guided_coordinates = working_coordinates
                else:
                    with torch.enable_grad():
                        differentiable_coordinates = working_coordinates.detach().requires_grad_(True)
                        denoised_for_guidance = denoise(
                            context.to_denoising_input(
                                noisy_coords=differentiable_coordinates,
                                noise_level=noise_level,
                            )
                        )
                        log_probability = potential.log_probability(
                            denoised_for_guidance,
                            monte_carlo_noise,
                            time=normalized_time,
                            step_index=step_index,
                        )
                        noisy_gradient = torch.autograd.grad(
                            log_probability.sum(),
                            differentiable_coordinates,
                            retain_graph=False,
                            create_graph=False,
                        )[0]
                    guided_coordinates = working_coordinates + guidance.rho * noisy_gradient.detach()

                denoised_coordinates = denoise(
                    context.to_denoising_input(
                        noisy_coords=guided_coordinates,
                        noise_level=noise_level,
                    )
                )
                denoised_coordinates = denoised_coordinates + potential.project(
                    denoised_coordinates,
                    time=normalized_time,
                    step_index=step_index,
                )
                for _ in range(guidance.num_inner_steps):
                    if guidance.mu == 0.0:
                        break
                    _, clean_gradient = potential.log_probability_and_gradient(
                        denoised_coordinates,
                        monte_carlo_noise,
                        time=normalized_time,
                        step_index=step_index,
                    )
                    denoised_coordinates = denoised_coordinates + guidance.mu * clean_gradient

                # The released TFG branch is one Euler update; it does not run
                # the ordinary sampler's optional second (Heun) denoising call.
                derivative = (guided_coordinates - denoised_coordinates) / noise_level[..., None, None]
                next_coordinates = guided_coordinates + (
                    step_scale * (sigma_next - noise_level)[..., None, None] * derivative
                )
                if outer_index + 1 < guidance.num_outer_steps:
                    outer_noise = draw_normal_like(next_coordinates, generator)
                    outer_noise_scale = torch.sqrt(noise_level**2 - sigma_next**2)
                    working_coordinates = next_coordinates + outer_noise_scale[..., None, None] * outer_noise
            coordinates = next_coordinates
        return coordinates

    def _sample_msa_indices(
        self,
        num_rows: int,
        device: torch.device,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Draw one recycle MSA subset without privileging the query row.
        """

        random_device = device if generator is None else generator.device
        num_selected_rows = int(
            torch.randint(
                1,
                num_rows + 1,
                (),
                device=random_device,
                generator=generator,
            )
        )
        num_selected_rows = min(num_selected_rows, 16384)
        return torch.randperm(
            num_rows,
            device=random_device,
            generator=generator,
        )[
            :num_selected_rows
        ].to(device)


class ProtenixStructureConditionModel(ProtenixModel):
    r"""
    Predict structures with the released Protenix mini structure condition.

    The structure update is a real part of this graph and is applied before the
    MSA path on every recycling pass.

    Args:
        config: Structure-conditioned model architecture and inference
            configuration.
    """

    config_class = ProtenixStructureConditionConfig
    config: ProtenixStructureConditionConfig

    def __init__(self, config: ProtenixStructureConditionConfig | None = None) -> None:
        resolved = ProtenixStructureConditionConfig() if config is None else config
        super().__init__(resolved)
        self.noisy_structure_embedder = ProtenixNoisyStructureEmbedder(resolved.pair_size)
        nn.init.zeros_(self.noisy_structure_embedder.transition.feed_forward.output_projection.weight)

    def _recycled_pair(
        self,
        input: ProtenixInput,
        pair_init: Tensor,
        pair: Tensor,
        pair_recycling_dropout: float,
    ) -> Tensor:
        r"""
        Add the fixed noisy-structure features to each recycled pair state.
        """

        pair = super()._recycled_pair(input, pair_init, pair, pair_recycling_dropout)
        return pair + self.noisy_structure_embedder(
            pair,
            input.structure_rep_positions,
            input.structure_rep_mask,
        )


class ProtenixConstraintModel(ProtenixModel):
    r"""
    Predict constrained biomolecular complex structures with Protenix.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = ProtenixConstraintConfig
    config: ProtenixConstraintConfig
    _supports_constraints = True

    def __init__(self, config: ProtenixConstraintConfig | None = None) -> None:
        resolved = ProtenixConstraintConfig() if config is None else config
        if not resolved.constraints:
            fail("value_invalid", RequestRef("config.constraints", "False"))
        super().__init__(resolved)


class ProtenixTemplateModel(ProtenixModel):
    r"""
    Base Protenix graph with the template embedder enabled -- the ``protenix_template`` architecture.

    Identical to [`ProtenixModel`][] except its config defaults ``template`` on, so a fresh build always
    constructs the template embedder that its ``ProtenixTemplateProcessor`` feeds, rather than silently
    dropping template features.

    Examples:
        >>> import torch
        >>> from ofoldx.models.protenix import ProtenixTemplateModel, ProtenixTemplateModelConfig
        >>> config = ProtenixTemplateModelConfig({
        ...     "trunk": {"num_msa_stages": 1, "num_pairformer_blocks": 1},
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...     },
        ...     "confidence": {"num_blocks": 1},
        ...     "template": {"num_blocks": 1},
        ... })
        >>> with torch.device("meta"):
        ...     model = ProtenixTemplateModel(config)
        >>> model.template_embedder is not None
        True
    """

    config_class = ProtenixTemplateModelConfig
    config: ProtenixTemplateModelConfig
    _supports_template = True

    def __init__(self, config: ProtenixTemplateModelConfig | None = None) -> None:
        resolved = ProtenixTemplateModelConfig() if config is None else config
        if resolved.template is None:
            fail("value_invalid", RequestRef("config.template", "None"))
        super().__init__(resolved)


class ProtenixNoisyStructureEmbedder(nn.Module):
    r"""
    Embed representative-atom distances and the current Protenix pair state.

    Args:
        pair_size: Width of the trunk pair representation. The released graph
            splits it into two equal projected halves.
        min_distance: First strict-open distance-bin lower edge in angstroms.
        max_distance: Last finite distance-bin lower edge in angstroms.
        num_bins: Number of strict-open distance bins.
    """

    def __init__(
        self,
        pair_size: int = 128,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        num_bins: int = 39,
    ) -> None:
        super().__init__()
        if pair_size % 2:
            fail("value_invalid", RequestRef("pair_size", str(pair_size)), invariant="even")
        half_size = pair_size // 2
        self.pair_norm = LayerNorm(pair_size)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_bins = num_bins
        self.pair_projection = nn.Linear(pair_size, half_size, bias=False)
        self.structure_projection = nn.Linear(num_bins + 1, half_size, bias=False)
        self.transition = Transition(pair_size, expansion=2)

    def forward(
        self,
        pair: Tensor,
        structure_rep_positions: Tensor,
        structure_rep_mask: Tensor,
        *,
        add_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Return one structure-conditioned pair update.

        Args:
            pair: Current pair state with shape ``(..., N, N, C_pair)``.
            structure_rep_positions: Representative coordinates with shape
                ``(..., N, 3)`` in angstroms.
            structure_rep_mask: Valid-coordinate mask with shape ``(..., N)``.
            add_noise: Standard deviation of optional Gaussian distance noise
                in angstroms.
            generator: Random generator used for the optional distance noise.

        Returns:
            Pair update with the same shape as ``pair``. Examples without any
                valid structure coordinate receive exactly zero.
        """

        structure_rep_positions = project_model_coordinates(
            structure_rep_positions,
            structure_rep_mask.to(torch.bool),
            dtype=pair.dtype,
            feature="Protenix noisy structure conditioning",
        )
        distance = pairwise_distance(structure_rep_positions).unsqueeze(-1)
        if add_noise > 0.0:
            distance = (
                distance
                + draw_normal(
                    distance.shape,
                    device=distance.device,
                    dtype=distance.dtype,
                    generator=generator,
                )
                * add_noise
            )
        valid_pair = (structure_rep_mask.bool().unsqueeze(-1) & structure_rep_mask.bool().unsqueeze(-2)).unsqueeze(-1)
        lower_bins = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins,
            device=distance.device,
            dtype=distance.dtype,
        )
        upper_bins = torch.cat((lower_bins[1:], lower_bins.new_tensor([torch.inf])))
        distance_bins = ((distance > lower_bins) & (distance < upper_bins)).to(dtype=pair.dtype)
        structure_features = torch.cat(
            (
                distance_bins * valid_pair.to(dtype=pair.dtype),
                valid_pair.to(dtype=pair.dtype),
            ),
            dim=-1,
        )
        hidden = torch.cat(
            (
                self.pair_projection(self.pair_norm(pair)),
                self.structure_projection(structure_features),
            ),
            dim=-1,
        )
        update = self.transition(hidden)
        has_valid_input = valid_pair.any(dim=(-3, -2, -1), keepdim=True)
        return update * has_valid_input.to(dtype=update.dtype)


class ProtenixInputEmbeddings(nn.Module):
    r"""
    Build token input embeddings from packed atom and token features.

    The centred [`windowing`][ofoldx.modules.atom.window.LocalAtomWindowing] builds the local atom geometry
    the embeddings consume. The reference atoms are conditioned, their local pairs encoded and projected to
    every block's attention bias, and the attended atoms mean-pooled to tokens.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 385,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.windowing = LocalAtomWindowing(32, 128)
        self.atom_conditioner = AtomConditioner(atom_feature_size, 128)
        self.atom_pair_encoder = AtomPairEncoder(128, 16)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=16,
            num_blocks=3,
            num_attention_heads=4,
            bias=False,
        )
        self.atom_encoder = AtomToTokenEncoder(
            128,
            single_size,
            3,
            4,
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        token_features: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        local_geometry: LocalAtomGeometry,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        atom_dtype = self.atom_conditioner.atom_feature_projection.weight.dtype
        reference_positions = project_model_coordinates(
            reference_positions,
            reference_mask.to(torch.bool) & atom_mask.to(torch.bool),
            dtype=atom_dtype,
            feature="Protenix embedding references",
        )
        atom_features = project_model_features(
            atom_features,
            atom_mask.to(torch.bool)[..., None],
            dtype=atom_dtype,
            feature="Protenix embedding atom features",
        )
        reference_charges = project_model_features(
            reference_charges,
            atom_mask.to(torch.bool)[..., None],
            dtype=atom_dtype,
            feature="Protenix embedding charges",
        )
        token_features = project_model_features(
            token_features,
            (
                torch.ones_like(token_features[..., 0], dtype=torch.bool)
                if token_mask is None
                else token_mask.to(torch.bool)
            )[..., None],
            dtype=atom_dtype,
            feature="Protenix embedding token features",
        )
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, local_geometry)
        token_single = self.atom_encoder(
            atom_conditioning,
            self.atom_pair_bias_projection(atom_pair),
            atom_mask,
            atom_to_token,
            token_features,
            local_geometry.window_info,
        )

        token_tail = token_features[..., self.single_size :].to(token_single)
        token_embedding = torch.cat([token_single, token_tail], dim=-1)

        if token_mask is not None:
            token_embedding = token_embedding * token_mask.unsqueeze(-1).to(dtype=token_embedding.dtype)
        return token_embedding


class ProtenixTemplateEmbedder(nn.Module):
    r"""
    Embed the fixed four Protenix template slots into a pair update.

    Args:
        pair_size: Width of the trunk pair representation.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks.
        vocab_size: Number of template residue types.
        num_distogram_bins: Number of template distance bins.
        triangle_multiplication_hidden_size: Hidden width of template triangle multiplication.
        triangle_attention_head_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Template transition expansion factor.
        dropout: Dropout probability inside each template block.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = len(TOKENS),
        num_distogram_bins: int = 39,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pair_norm = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = FactorizedPairLinear(
            pair_size=num_distogram_bins + 5,
            token_size=vocab_size,
            output_size=hidden_size,
            pair_prefix_size=num_distogram_bins + 1,
            bias=False,
        )
        self.pair_update_stack = TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
        )
        self.hidden_norm = LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(
        self,
        pair: Tensor,
        templates: ProtenixTemplateFeatures,
        *,
        same_chain_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        r"""
        Return the released fixed-four normalized template pair update.

        Args:
            pair: Trunk pair representation with shape ``[..., N, N, C_pair]``.
            templates: Four padded template feature slots.
            same_chain_mask: Same-chain pair mask with shape ``[..., N, N]``.
            pair_mask: Valid-token pair mask with shape ``[..., N, N]``.

        Returns:
            Template update with the same shape as ``pair``.
        """
        templates = _project_protenix_templates(templates, dtype=pair.dtype, pair_mask=pair_mask)

        projected_pair = self.pair_projection(self.pair_norm(pair))
        update = torch.zeros_like(projected_pair)
        # The released multichain mask applies only to geometry; residue one-hot and pair projections remain intact.
        geometry_mask = (same_chain_mask & pair_mask.bool()).to(dtype=pair.dtype)
        for template_index in range(4):
            geometry, residue_one_hot = self._template_features(
                templates,
                template_index=template_index,
                geometry_mask=geometry_mask,
            )
            template_pair = projected_pair + self.template_projection(geometry, residue_one_hot)
            template_pair = self.pair_update_stack(template_pair, pair_mask=pair_mask)
            update = update + self.hidden_norm(template_pair)
        return self.output_projection(self.activation(update / 4.0))

    def _template_features(
        self,
        templates: ProtenixTemplateFeatures,
        *,
        template_index: int,
        geometry_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residue_type_ids = templates.residue_type_ids[..., template_index, :]
        residue_one_hot = F.one_hot(residue_type_ids, num_classes=self.vocab_size).to(dtype=templates.distogram.dtype)
        geometry = torch.cat(
            (
                templates.distogram[..., template_index, :, :, :],
                templates.pseudo_beta_mask[..., template_index, :, :].unsqueeze(-1),
                templates.unit_vector[..., template_index, :, :, :],
                templates.backbone_frame_mask[..., template_index, :, :].unsqueeze(-1),
            ),
            dim=-1,
        )
        return geometry * geometry_mask.unsqueeze(-1), residue_one_hot


class ProtenixConstraintEmbeddings(nn.Module):
    r"""
    Project and combine pocket, token-contact, atom-contact, and substructure constraints.
    """

    def __init__(self, pair_size: int = 128, *, backend: str = "auto") -> None:
        super().__init__()
        self.pocket_projection = nn.Linear(1, pair_size, bias=False)
        self.token_contact_projection = nn.Linear(2, pair_size, bias=False)
        self.atom_contact_projection = nn.Linear(2, pair_size, bias=False)
        self.substructure_embeddings = ProtenixConstraintSubstructureEmbeddings(pair_size, backend=backend)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        projection = self.substructure_embeddings.block.attention.projection
        qkv_projections = (
            projection.query_projection,
            projection.key_projection,
            projection.value_projection,
        )
        for module in self.modules():
            # The release's packed MHA QKV parameter is not an nn.Linear, so
            # its nonzero initializer survives the surrounding zero projections.
            if isinstance(module, nn.Linear) and module not in qkv_projections:
                nn.init.zeros_(module.weight)

    def forward(
        self,
        pocket: Tensor,
        token_contact: Tensor,
        atom_contact: Tensor,
        substructure: Tensor,
    ) -> Tensor:
        return (
            self.pocket_projection(pocket)
            + self.token_contact_projection(token_contact)
            + self.atom_contact_projection(atom_contact)
            + self.substructure_embeddings(substructure)
        )


class ProtenixConstraintSubstructureEmbeddings(nn.Module):
    r"""
    Embed the flattened token-pair substructure grid with a Transformer.
    """

    def __init__(self, pair_size: int = 128, *, backend: str = "auto") -> None:
        super().__init__()
        hidden_size = 128
        self.input_projection = nn.Linear(4, hidden_size, bias=False)
        self.block = ProtenixConstraintSubstructureBlock(
            hidden_size=hidden_size,
            num_attention_heads=4,
            transition_size=hidden_size * 4,
            dropout=0.1,
            backend=backend,
        )
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, substructure: Tensor) -> Tensor:
        if isinstance(substructure, NestedTensor):
            pair_counts = substructure.element_sizes()[:, :2].prod(dim=-1)
            flattened = substructure.packed_with_lengths(substructure.concat, pair_counts)
        else:
            flattened = substructure.flatten(-3, -2)
        hidden = self.block(self.input_projection(flattened))
        output = self.output_projection(hidden)
        if isinstance(substructure, NestedTensor):
            return substructure.packed_with_static_tail(cast(NestedTensor, output).concat)
        return output.unflatten(-2, substructure.shape[-3:-1])


class ProtenixConstraintSubstructureBlock(nn.Module):
    r"""
    Post-norm self-attention and feed-forward residual block.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        transition_size: int,
        dropout: float,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.attention = MultiheadAttention(
            MultiheadAttentionProjection(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                query_bias=True,
                value_bias=True,
                output_bias=True,
            ),
            backend=backend,
            dropout=dropout,
        )
        # Match the variance of one Xavier-initialized packed (3C, C) QKV map.
        qkv_bound = math.sqrt(6.0 / (4 * hidden_size))
        for projection in (
            self.attention.projection.query_projection,
            self.attention.projection.key_projection,
            self.attention.projection.value_projection,
        ):
            nn.init.uniform_(projection.weight, -qkv_bound, qkv_bound)
        for projection in (
            self.attention.projection.query_projection,
            self.attention.projection.value_projection,
            self.attention.projection.output_projection,
        ):
            nn.init.zeros_(projection.bias)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size, backend=backend)
        self.feed_forward = FeedForward(
            hidden_size,
            transition_size,
            dropout=dropout,
            activation="relu",
        )
        self.feed_forward_output_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = LayerNorm(hidden_size, backend=backend)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.attention_norm(hidden + self.attention_output_dropout(self.attention(hidden)))
        return self.feed_forward_norm(hidden + self.feed_forward_output_dropout(self.feed_forward(hidden)))


@dataclass(frozen=True)
class ProtenixExponentialInterpolation:
    r"""
    Exponentially interpolate one scalar TFG parameter over normalized diffusion time.

    Protenix uses time one for the first, noisiest transition and approaches
    time zero at the end of sampling.

    Attributes:
        start: Value at normalized time zero.
        end: Value at normalized time one.
        alpha: Exponential warping strength. Zero selects linear interpolation.
    """

    start: float
    end: float
    alpha: float = 0.0

    def __call__(self, time: float) -> float:
        if self.alpha == 0.0:
            return float(self.start + (self.end - self.start) * time)
        numerator = math.exp(self.alpha * time) - 1.0
        denominator = math.exp(self.alpha) - 1.0
        return float(self.start + (self.end - self.start) * numerator / denominator)


@dataclass(frozen=True)
class ProtenixVinaStericTerm:
    r"""
    Configure the interchain Vina steric TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
        buffer: Fractional active-pair threshold below the van-der-Waals
            equilibrium distance.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.1
    buffer: float | Callable[[float], float] = 0.225


@dataclass(frozen=True)
class ProtenixExperimentalTorsionTerm:
    r"""
    Configure the experimental-torsion TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.0015


@dataclass(frozen=True)
class ProtenixInterchainBondTerm:
    r"""
    Configure the covalent interchain-bond TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
        buffer: Flat-bottom upper bond distance in angstroms, or its schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.15
    buffer: float | Callable[[float], float] = 2.0


@dataclass(frozen=True)
class ProtenixPairwiseDistanceTerm:
    r"""
    Configure the component-local bond, angle, and clash TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
        enable_projection: Whether this term participates in projection.
        bond_buffer: Fractional slack for direct-bond bounds, or its schedule.
        angle_buffer: Fractional slack for angle-derived bounds, or its schedule.
        clash_buffer: Fractional slack for non-bonded lower bounds, or its schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.5
    enable_projection: bool = True
    bond_buffer: float | Callable[[float], float] = 0.0
    angle_buffer: float | Callable[[float], float] = 0.0
    clash_buffer: float | Callable[[float], float] = 0.0


@dataclass(frozen=True)
class ProtenixChiralAtomTerm:
    r"""
    Configure the tetrahedral-chirality TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule. Projection is
            independent of this weight.
        enable_projection: Whether this term participates in projection.
        buffer: Minimum signed chiral angle in radians, or its schedule.
        scale_x: Preserve affected-chain radii of gyration after projection.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.0
    enable_projection: bool = True
    buffer: float | Callable[[float], float] = 0.6155
    scale_x: bool = True


@dataclass(frozen=True)
class ProtenixStereoBondTerm:
    r"""
    Configure the stereogenic double-bond TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
        buffer: Angular tolerance in radians, or its schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.25
    buffer: float | Callable[[float], float] = 0.52360


@dataclass(frozen=True)
class ProtenixPlanarImproperTerm:
    r"""
    Configure the planar-improper TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.12


@dataclass(frozen=True)
class ProtenixLinearBondTerm:
    r"""
    Configure the linear-bond TFG term.

    Attributes:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
        buffer: Angular tolerance in radians, or its schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.25
    buffer: float | Callable[[float], float] = 0.08726646259


@dataclass(frozen=True)
class ProtenixTfgGuidance:
    r"""
    Configure Protenix training-free chemical guidance.

    ``ProtenixTfgGuidance()`` is the released enabled configuration. Set an
    individual term to ``None`` to remove it.

    Attributes:
        rho: Strength of the gradient propagated through the denoiser to noisy
            coordinates.
        mu: Direct clean-coordinate log-probability ascent step size.
        monte_carlo_std: Standard deviation of clean-coordinate perturbations
            used for Monte Carlo smoothing.
        num_monte_carlo_samples: Number of Monte Carlo perturbations when the
            standard deviation is nonzero.
        num_outer_steps: TFG predictor-corrector iterations per diffusion transition.
        num_inner_steps: Direct clean-coordinate refinement iterations per outer step.
        num_projection_outer_steps: Passes over the chiral-then-pairwise projection order.
        num_projection_inner_steps: Projection updates for each enabled projection term.
        vina_steric: Interchain Vina steric term, or ``None``.
        experimental_torsion: Experimental-torsion term, or ``None``.
        interchain_bond: Covalent interchain-bond term, or ``None``.
        pairwise_distance: Component-local distance term, or ``None``.
        chiral_atom: Tetrahedral-chirality term, or ``None``.
        stereo_bond: Stereogenic double-bond term, or ``None``.
        planar_improper: Planar-improper term, or ``None``.
        linear_bond: Linear-bond term, or ``None``.
    """

    rho: float = 0.0
    mu: float = 0.1
    monte_carlo_std: float = 0.0
    num_monte_carlo_samples: int = 1
    num_outer_steps: int = 1
    num_inner_steps: int = 20
    num_projection_outer_steps: int = 2
    num_projection_inner_steps: int = 10
    vina_steric: ProtenixVinaStericTerm | None = ProtenixVinaStericTerm()
    experimental_torsion: ProtenixExperimentalTorsionTerm | None = ProtenixExperimentalTorsionTerm()
    interchain_bond: ProtenixInterchainBondTerm | None = ProtenixInterchainBondTerm()
    pairwise_distance: ProtenixPairwiseDistanceTerm | None = ProtenixPairwiseDistanceTerm()
    chiral_atom: ProtenixChiralAtomTerm | None = ProtenixChiralAtomTerm()
    stereo_bond: ProtenixStereoBondTerm | None = ProtenixStereoBondTerm()
    planar_improper: ProtenixPlanarImproperTerm | None = ProtenixPlanarImproperTerm()
    linear_bond: ProtenixLinearBondTerm | None = ProtenixLinearBondTerm()


class ProtenixTfgPotential(nn.Module):
    r"""
    Compose Protenix's released training-free chemical potential.

    Static molecular geometry is prepared by a Protenix processor and supplied
    through [`ProtenixTfgFeatures`][]. The module owns only tensor equations and
    can therefore follow ordinary module device and dtype placement.

    Args:
        features: Static molecular geometry aligned to the model atom axis.
        guidance: Enabled terms, schedules, and predictor-corrector controls.
    """

    def __init__(
        self,
        features: ProtenixTfgFeatures,
        guidance: ProtenixTfgGuidance,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        _validate_protenix_tfg_guidance(guidance)
        self.guidance = guidance
        self.vina_steric = (
            None
            if guidance.vina_steric is None
            else ProtenixVinaStericPotential(
                features.atom_chain,
                features.atom_vdw_radius,
                features.interchain_bond_index,
            )
        )
        self.experimental_torsion = (
            None
            if guidance.experimental_torsion is None
            else ProtenixExperimentalTorsionPotential(
                features.experimental_torsion_index,
                features.experimental_torsion_force_constant,
                features.experimental_torsion_sign,
                backend=backend,
            )
        )
        self.interchain_bond = (
            None
            if guidance.interchain_bond is None
            else ProtenixInterchainBondPotential(features.interchain_bond_index)
        )
        self.pairwise_distance = (
            None
            if guidance.pairwise_distance is None
            else ProtenixPairwiseDistancePotential(
                features.pairwise_distance_index,
                features.pairwise_distance_lower_bound,
                features.pairwise_distance_upper_bound,
                features.pairwise_distance_is_bond,
                features.pairwise_distance_is_angle,
                features.atom_vdw_radius,
            )
        )
        self.chiral_atom = (
            None
            if guidance.chiral_atom is None
            else ProtenixChiralAtomPotential(
                features.chiral_index, features.chiral_orientation, features.atom_chain, backend=backend
            )
        )
        self.stereo_bond = (
            None
            if guidance.stereo_bond is None
            else ProtenixStereoBondPotential(
                features.stereo_bond_index, features.stereo_bond_orientation, backend=backend
            )
        )
        self.planar_improper = (
            None
            if guidance.planar_improper is None
            else ProtenixPlanarImproperPotential(features.planar_improper_index)
        )
        self.linear_bond = (
            None if guidance.linear_bond is None else ProtenixLinearBondPotential(features.linear_bond_index)
        )

    def forward(
        self,
        coordinates: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        energy = _protenix_zero_energy(coordinates)

        vina = self.guidance.vina_steric
        if vina is not None and self.vina_steric is not None and step_index % vina.interval == 0:
            weight = _protenix_tfg_scalar(vina.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.vina_steric(
                    coordinates,
                    buffer=_protenix_tfg_scalar(vina.buffer, time),
                )

        torsion = self.guidance.experimental_torsion
        if torsion is not None and self.experimental_torsion is not None and step_index % torsion.interval == 0:
            weight = _protenix_tfg_scalar(torsion.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.experimental_torsion(coordinates)

        interchain = self.guidance.interchain_bond
        if interchain is not None and self.interchain_bond is not None and step_index % interchain.interval == 0:
            weight = _protenix_tfg_scalar(interchain.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.interchain_bond(
                    coordinates,
                    buffer=_protenix_tfg_scalar(interchain.buffer, time),
                )

        pairwise = self.guidance.pairwise_distance
        if pairwise is not None and self.pairwise_distance is not None and step_index % pairwise.interval == 0:
            weight = _protenix_tfg_scalar(pairwise.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.pairwise_distance(
                    coordinates,
                    bond_buffer=_protenix_tfg_scalar(pairwise.bond_buffer, time),
                    angle_buffer=_protenix_tfg_scalar(pairwise.angle_buffer, time),
                    clash_buffer=_protenix_tfg_scalar(pairwise.clash_buffer, time),
                )

        chiral = self.guidance.chiral_atom
        if chiral is not None and self.chiral_atom is not None and step_index % chiral.interval == 0:
            weight = _protenix_tfg_scalar(chiral.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.chiral_atom(
                    coordinates,
                    buffer=_protenix_tfg_scalar(chiral.buffer, time),
                )

        stereo = self.guidance.stereo_bond
        if stereo is not None and self.stereo_bond is not None and step_index % stereo.interval == 0:
            weight = _protenix_tfg_scalar(stereo.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.stereo_bond(
                    coordinates,
                    buffer=_protenix_tfg_scalar(stereo.buffer, time),
                )

        planar = self.guidance.planar_improper
        if planar is not None and self.planar_improper is not None and step_index % planar.interval == 0:
            weight = _protenix_tfg_scalar(planar.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.planar_improper(coordinates)

        linear = self.guidance.linear_bond
        if linear is not None and self.linear_bond is not None and step_index % linear.interval == 0:
            weight = _protenix_tfg_scalar(linear.weight, time)
            if weight != 0.0:
                energy = energy + weight * self.linear_bond(
                    coordinates,
                    buffer=_protenix_tfg_scalar(linear.buffer, time),
                )
        return energy

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        energy = _protenix_zero_energy(coordinates)
        gradient = torch.zeros_like(coordinates)

        vina = self.guidance.vina_steric
        if vina is not None and self.vina_steric is not None and step_index % vina.interval == 0:
            weight = _protenix_tfg_scalar(vina.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.vina_steric.energy_and_gradient(
                    coordinates,
                    buffer=_protenix_tfg_scalar(vina.buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        torsion = self.guidance.experimental_torsion
        if torsion is not None and self.experimental_torsion is not None and step_index % torsion.interval == 0:
            weight = _protenix_tfg_scalar(torsion.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.experimental_torsion.energy_and_gradient(coordinates)
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        interchain = self.guidance.interchain_bond
        if interchain is not None and self.interchain_bond is not None and step_index % interchain.interval == 0:
            weight = _protenix_tfg_scalar(interchain.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.interchain_bond.energy_and_gradient(
                    coordinates,
                    buffer=_protenix_tfg_scalar(interchain.buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        pairwise = self.guidance.pairwise_distance
        if pairwise is not None and self.pairwise_distance is not None and step_index % pairwise.interval == 0:
            weight = _protenix_tfg_scalar(pairwise.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.pairwise_distance.energy_and_gradient(
                    coordinates,
                    bond_buffer=_protenix_tfg_scalar(pairwise.bond_buffer, time),
                    angle_buffer=_protenix_tfg_scalar(pairwise.angle_buffer, time),
                    clash_buffer=_protenix_tfg_scalar(pairwise.clash_buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        chiral = self.guidance.chiral_atom
        if chiral is not None and self.chiral_atom is not None and step_index % chiral.interval == 0:
            weight = _protenix_tfg_scalar(chiral.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.chiral_atom.energy_and_gradient(
                    coordinates,
                    buffer=_protenix_tfg_scalar(chiral.buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        stereo = self.guidance.stereo_bond
        if stereo is not None and self.stereo_bond is not None and step_index % stereo.interval == 0:
            weight = _protenix_tfg_scalar(stereo.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.stereo_bond.energy_and_gradient(
                    coordinates,
                    buffer=_protenix_tfg_scalar(stereo.buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        planar = self.guidance.planar_improper
        if planar is not None and self.planar_improper is not None and step_index % planar.interval == 0:
            weight = _protenix_tfg_scalar(planar.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.planar_improper.energy_and_gradient(coordinates)
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient

        linear = self.guidance.linear_bond
        if linear is not None and self.linear_bond is not None and step_index % linear.interval == 0:
            weight = _protenix_tfg_scalar(linear.weight, time)
            if weight != 0.0:
                term_energy, term_gradient = self.linear_bond.energy_and_gradient(
                    coordinates,
                    buffer=_protenix_tfg_scalar(linear.buffer, time),
                )
                energy = energy + weight * term_energy
                gradient = gradient + weight * term_gradient
        return energy, gradient

    def log_probability(
        self,
        coordinates: Tensor,
        monte_carlo_noise: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        num_monte_carlo_samples = monte_carlo_noise.shape[0]
        perturbed = (coordinates.unsqueeze(0) + monte_carlo_noise).reshape(
            -1,
            *coordinates.shape[-2:],
        )
        log_probability = -self(
            perturbed,
            time=time,
            step_index=step_index,
        ).reshape(num_monte_carlo_samples, *coordinates.shape[:-2])
        if num_monte_carlo_samples == 1:
            return log_probability.squeeze(0)
        return torch.logsumexp(log_probability, dim=0) - math.log(num_monte_carlo_samples)

    def log_probability_and_gradient(
        self,
        coordinates: Tensor,
        monte_carlo_noise: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        num_monte_carlo_samples = monte_carlo_noise.shape[0]
        perturbed = (coordinates.unsqueeze(0) + monte_carlo_noise).reshape(
            -1,
            *coordinates.shape[-2:],
        )
        energy, energy_gradient = self.energy_and_gradient(
            perturbed,
            time=time,
            step_index=step_index,
        )
        energy = energy.reshape(num_monte_carlo_samples, *coordinates.shape[:-2])
        energy_gradient = energy_gradient.reshape(num_monte_carlo_samples, *coordinates.shape)
        log_probability = -energy
        if num_monte_carlo_samples == 1:
            return log_probability.squeeze(0), -energy_gradient.squeeze(0)
        average = torch.logsumexp(log_probability, dim=0) - math.log(num_monte_carlo_samples)
        importance = torch.softmax(log_probability, dim=0)
        gradient = (importance.unsqueeze(-1).unsqueeze(-1) * -energy_gradient).sum(dim=0)
        return average, gradient

    def project(
        self,
        coordinates: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        delta = torch.zeros_like(coordinates)
        chiral = self.guidance.chiral_atom
        pairwise = self.guidance.pairwise_distance
        for _ in range(self.guidance.num_projection_outer_steps):
            if (
                chiral is not None
                and self.chiral_atom is not None
                and chiral.enable_projection
                and step_index % chiral.interval == 0
            ):
                buffer = _protenix_tfg_scalar(chiral.buffer, time)
                for _ in range(self.guidance.num_projection_inner_steps):
                    delta = delta + self.chiral_atom.project(
                        coordinates + delta,
                        buffer=buffer,
                        scale_x=chiral.scale_x,
                    )
            if (
                pairwise is not None
                and self.pairwise_distance is not None
                and pairwise.enable_projection
                and step_index % pairwise.interval == 0
            ):
                bond_buffer = _protenix_tfg_scalar(pairwise.bond_buffer, time)
                angle_buffer = _protenix_tfg_scalar(pairwise.angle_buffer, time)
                clash_buffer = _protenix_tfg_scalar(pairwise.clash_buffer, time)
                for _ in range(self.guidance.num_projection_inner_steps):
                    delta = delta + self.pairwise_distance.project(
                        coordinates + delta,
                        bond_buffer=bond_buffer,
                        angle_buffer=angle_buffer,
                        clash_buffer=clash_buffer,
                    )
        return delta


class ProtenixVinaStericPotential(nn.Module):
    r"""
    Penalize steric overlap between non-covalently connected Protenix chains.

    Args:
        atom_chain: Chain index for every atom.
        atom_vdw_radius: Van-der-Waals radius for every atom.
        interchain_bond_index: Covalent interchain atom pairs with shape ``(2, M)``.
    """

    def __init__(
        self,
        atom_chain: Tensor,
        atom_vdw_radius: Tensor,
        interchain_bond_index: Tensor,
    ) -> None:
        super().__init__()
        pair_index, equilibrium_distance = _protenix_vina_pairs(
            atom_chain,
            atom_vdw_radius,
            interchain_bond_index,
        )
        self.pair_index: Tensor
        self.equilibrium_distance: Tensor
        self.register_buffer("pair_index", pair_index)
        self.register_buffer("equilibrium_distance", equilibrium_distance)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.225) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        energy, *_ = _protenix_vina_energy_terms(coordinates, self.pair_index, self.equilibrium_distance, buffer=buffer)
        return energy

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.225,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        return _protenix_vina_energy_and_gradient(
            coordinates,
            self.pair_index,
            self.equilibrium_distance,
            buffer=buffer,
        )


class ProtenixExperimentalTorsionPotential(nn.Module):
    r"""
    Apply Protenix's RDKit experimental-torsion cosine series.

    Args:
        atom_index: Torsion atom quadruples with shape ``(4, M)``.
        force_constant: Six Fourier force constants per torsion.
        sign: Six Fourier signs per torsion.
    """

    def __init__(
        self,
        atom_index: Tensor,
        force_constant: Tensor,
        sign: Tensor,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.atom_index: Tensor
        self.force_constant: Tensor
        self.sign: Tensor
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("force_constant", force_constant)
        self.register_buffer("sign", sign)

    def forward(self, coordinates: Tensor) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        angle = indexed_signed_dihedral(coordinates, self.atom_index, backend=self.backend)
        order = torch.arange(1, 7, device=coordinates.device, dtype=coordinates.dtype)
        energy = self.force_constant.to(device=coordinates.device) * (
            1.0 + self.sign.to(device=coordinates.device) * torch.cos(angle.unsqueeze(-1) * order)
        )
        return energy.sum(dim=(-1, -2))

    def energy_and_gradient(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = indexed_signed_dihedral_with_gradient(
            coordinates, self.atom_index, backend=self.backend
        )
        order = torch.arange(1, 7, device=coordinates.device, dtype=coordinates.dtype)
        force_constant = self.force_constant.to(device=coordinates.device)
        sign = self.sign.to(device=coordinates.device)
        energy = force_constant * (1.0 + sign * torch.cos(angle.unsqueeze(-1) * order))
        angle_derivative = (-force_constant * sign * order * torch.sin(angle.unsqueeze(-1) * order)).sum(dim=-1)
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            angle_gradient,
            angle_derivative,
        )
        return energy.sum(dim=(-1, -2)), gradient


class ProtenixInterchainBondPotential(nn.Module):
    r"""
    Apply the released upper-bound energy to covalent interchain bonds.

    Args:
        atom_index: Interchain atom pairs with shape ``(2, M)``.
    """

    def __init__(self, atom_index: Tensor) -> None:
        super().__init__()
        self.atom_index: Tensor
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor, *, buffer: float = 2.0) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        distances = indexed_distance(coordinates, self.atom_index, minimum=1e-8)
        return torch.clamp_min(distances - buffer, 0.0).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 2.0,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        distances, distance_gradient = indexed_distance_with_gradient(coordinates, self.atom_index, minimum=1e-8)
        active = distances > buffer
        energy = torch.where(active, distances - buffer, 0.0)
        derivative = active.to(dtype=coordinates.dtype)
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            distance_gradient,
            derivative,
        )
        return energy.sum(dim=-1), gradient


class ProtenixPairwiseDistancePotential(nn.Module):
    r"""
    Apply component-local bond, angle, and clash distance bounds.

    Args:
        atom_index: Atom pairs with shape ``(2, M)``.
        lower_bound: RDKit lower distance bounds.
        upper_bound: RDKit upper distance bounds.
        is_bond: Whether each pair is a direct bond.
        is_angle: Whether each pair spans a bond angle.
        atom_vdw_radius: Van-der-Waals radius for every model atom.
    """

    def __init__(
        self,
        atom_index: Tensor,
        lower_bound: Tensor,
        upper_bound: Tensor,
        is_bond: Tensor,
        is_angle: Tensor,
        atom_vdw_radius: Tensor,
    ) -> None:
        super().__init__()
        if atom_index.numel() == 0:
            vdw_limit = lower_bound
        else:
            vdw_limit = 0.35 + 0.5 * atom_vdw_radius[atom_index].sum(dim=0)
        self.atom_index: Tensor
        self.is_bond: Tensor
        self.is_angle: Tensor
        self.lower_bound: Tensor
        self.upper_bound: Tensor
        self.vdw_limit: Tensor
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("is_bond", is_bond)
        self.register_buffer("is_angle", is_angle)
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)
        self.register_buffer("vdw_limit", vdw_limit)

    def forward(
        self,
        coordinates: Tensor,
        *,
        bond_buffer: float = 0.05,
        angle_buffer: float = 0.05,
        clash_buffer: float = 0.05,
    ) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        lower_bound, upper_bound = self._distance_bounds(
            bond_buffer=bond_buffer,
            angle_buffer=angle_buffer,
            clash_buffer=clash_buffer,
        )
        distances = indexed_distance(coordinates, self.atom_index, minimum=1e-8)
        below = torch.clamp_min(lower_bound.to(device=distances.device) - distances, 0.0)
        above = torch.clamp_min(distances - upper_bound.to(device=distances.device), 0.0)
        return (0.5 * (below**2 + above**2)).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        bond_buffer: float = 0.05,
        angle_buffer: float = 0.05,
        clash_buffer: float = 0.05,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        lower_bound, upper_bound = self._distance_bounds(
            bond_buffer=bond_buffer,
            angle_buffer=angle_buffer,
            clash_buffer=clash_buffer,
        )
        distances, distance_gradient = indexed_distance_with_gradient(coordinates, self.atom_index, minimum=1e-8)
        below = lower_bound.to(device=distances.device) - distances
        above = distances - upper_bound.to(device=distances.device)
        active_below = below > 0
        active_above = above > 0
        energy = 0.5 * (torch.where(active_below, below**2, 0.0) + torch.where(active_above, above**2, 0.0))
        derivative = torch.where(active_above, above, 0.0) - torch.where(active_below, below, 0.0)
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            distance_gradient,
            derivative,
        )
        return energy.sum(dim=-1), gradient

    def project(
        self,
        coordinates: Tensor,
        *,
        bond_buffer: float = 0.05,
        angle_buffer: float = 0.05,
        clash_buffer: float = 0.05,
    ) -> Tensor:
        r"""
        Project bond and angle distances into their released feasible intervals.
        """
        coordinates = _protenix_coordinates(coordinates)

        lower_bound, upper_bound = self._distance_bounds(
            bond_buffer=bond_buffer,
            angle_buffer=angle_buffer,
            clash_buffer=clash_buffer,
        )
        delta_angle = self._project_mask(
            coordinates,
            self.is_angle,
            lower_bound,
            upper_bound,
        )
        delta_bond = self._project_mask(
            coordinates + delta_angle,
            self.is_bond,
            lower_bound,
            upper_bound,
        )
        return delta_angle + delta_bond

    def _distance_bounds(
        self,
        *,
        bond_buffer: float,
        angle_buffer: float,
        clash_buffer: float,
    ) -> tuple[Tensor, Tensor]:
        if self.atom_index.numel() == 0:
            return self.lower_bound, self.upper_bound
        both = self.is_bond & self.is_angle
        shared_buffer = min(bond_buffer, angle_buffer)
        lower_buffer = torch.where(
            both,
            self.lower_bound.new_tensor(shared_buffer),
            torch.where(
                self.is_bond,
                self.lower_bound.new_tensor(bond_buffer),
                torch.where(
                    self.is_angle,
                    self.lower_bound.new_tensor(angle_buffer),
                    self.lower_bound.new_tensor(clash_buffer),
                ),
            ),
        )
        upper_buffer = torch.where(
            both,
            self.upper_bound.new_tensor(shared_buffer),
            torch.where(
                self.is_bond,
                self.upper_bound.new_tensor(bond_buffer),
                self.upper_bound.new_tensor(angle_buffer),
            ),
        )
        lower_bound = self.lower_bound * (1.0 - lower_buffer)
        upper_bound = self.upper_bound * (1.0 + upper_buffer)
        vdw_limit = self.vdw_limit.to(device=lower_bound.device)
        lower_bound = torch.where(
            self.is_bond,
            lower_bound,
            torch.maximum(lower_bound, vdw_limit),
        )
        upper_bound = torch.where(
            self.is_bond,
            torch.minimum(upper_bound, vdw_limit),
            torch.where(
                self.is_angle,
                upper_bound,
                torch.full_like(upper_bound, float("inf")),
            ),
        )
        return lower_bound, upper_bound

    def _project_mask(
        self,
        coordinates: Tensor,
        selected: Tensor,
        lower_bound: Tensor,
        upper_bound: Tensor,
    ) -> Tensor:
        atom_index = self.atom_index[:, selected]
        if atom_index.numel() == 0:
            return torch.zeros_like(coordinates)
        lower = lower_bound[selected].to(device=coordinates.device)
        upper = upper_bound[selected].to(device=coordinates.device)
        distances, distance_gradient = indexed_distance_with_gradient(coordinates, atom_index, minimum=1e-8)
        below = distances < lower
        above = distances > upper
        active = below | above
        if not bool(active.any()):
            return torch.zeros_like(coordinates)
        violation = distances - torch.where(below, lower, upper)
        return _protenix_constraint_projection(
            coordinates,
            atom_index,
            violation,
            distance_gradient,
            active,
        )


class ProtenixChiralAtomPotential(nn.Module):
    r"""
    Preserve tetrahedral stereochemistry during Protenix sampling.

    Args:
        atom_index: Chiral dihedral quadruples with shape ``(4, M)``.
        orientation: Reference signs for the chiral dihedrals.
        atom_chain: Chain index for every model atom.
    """

    def __init__(
        self,
        atom_index: Tensor,
        orientation: Tensor,
        atom_chain: Tensor,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.atom_index: Tensor
        self.orientation: Tensor
        self.atom_chain: Tensor
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("orientation", orientation)
        self.register_buffer("atom_chain", atom_chain)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.34906) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        signed_angle = indexed_signed_dihedral(
            coordinates, self.atom_index, backend=self.backend
        ) * self.orientation.to(coordinates)
        return torch.clamp_min(buffer - signed_angle, 0.0).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.34906,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = indexed_signed_dihedral_with_gradient(
            coordinates, self.atom_index, backend=self.backend
        )
        orientation = self.orientation.to(coordinates)
        violation = buffer - angle * orientation
        active = violation > 0
        derivative = torch.where(active, -orientation, 0.0)
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            angle_gradient,
            derivative,
        )
        return torch.where(active, violation, 0.0).sum(dim=-1), gradient

    def project(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.34906,
        scale_x: bool = True,
    ) -> Tensor:
        r"""
        Project violated signed dihedrals and preserve affected-chain radii of gyration.
        """
        coordinates = _protenix_coordinates(coordinates)

        if self.atom_index.numel() == 0:
            return torch.zeros_like(coordinates)
        angle, angle_gradient = indexed_signed_dihedral_with_gradient(
            coordinates, self.atom_index, backend=self.backend
        )
        orientation = self.orientation.to(coordinates)
        violation = angle * orientation - buffer
        angle_gradient = angle_gradient * orientation.reshape(
            *([1] * len(coordinates.shape[:-2])),
            1,
            -1,
            1,
        )
        active = violation < 0
        if not bool(active.any()):
            return torch.zeros_like(coordinates)
        delta = _protenix_constraint_projection(
            coordinates,
            self.atom_index,
            violation,
            angle_gradient,
            active,
        )
        if not scale_x:
            return delta
        return _protenix_preserve_chain_radius(
            coordinates,
            delta,
            atom_index=self.atom_index,
            atom_chain=self.atom_chain,
        )


class ProtenixStereoBondPotential(nn.Module):
    r"""
    Preserve declared E/Z double-bond geometry.

    Args:
        atom_index: Stereo-bond dihedral quadruples with shape ``(4, M)``.
        orientation: ``1`` for the reference trans-like orientation and ``0`` otherwise.
    """

    def __init__(
        self,
        atom_index: Tensor,
        orientation: Tensor,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.atom_index: Tensor
        self.orientation: Tensor
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("orientation", orientation)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.52360) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        angle = indexed_signed_dihedral(coordinates, self.atom_index, backend=self.backend).abs()
        orientation = self.orientation.to(coordinates) > 0.5
        violation = torch.where(
            orientation,
            torch.clamp_min(math.pi - buffer - angle, 0.0),
            torch.clamp_min(angle - buffer, 0.0),
        )
        return violation.sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.52360,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        signed_angle, signed_gradient = indexed_signed_dihedral_with_gradient(
            coordinates, self.atom_index, backend=self.backend
        )
        sign = 1 - 2 * (signed_angle < 0).to(signed_angle)
        angle = signed_angle.abs()
        angle_gradient = signed_gradient * sign.unsqueeze(-2).unsqueeze(-1)
        trans = self.orientation.to(coordinates) > 0.5
        below = trans & (angle < math.pi - buffer)
        above = (~trans) & (angle > buffer)
        violation = torch.where(
            below,
            math.pi - buffer - angle,
            torch.where(above, angle - buffer, 0.0),
        )
        derivative = above.to(angle) - below.to(angle)
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            angle_gradient,
            derivative,
        )
        return violation.sum(dim=-1), gradient


class ProtenixPlanarImproperPotential(nn.Module):
    r"""
    Penalize non-planar three-neighbor sp2 centers.

    Args:
        atom_index: Improper-dihedral atom quadruples with shape ``(4, M)``.
    """

    def __init__(self, atom_index: Tensor) -> None:
        super().__init__()
        self.atom_index: Tensor
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        return _protenix_planar_improper_energy(coordinates, self.atom_index).sum(dim=-1)

    def energy_and_gradient(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        energy, value_gradient = _protenix_planar_improper_energy_and_gradient(
            coordinates,
            self.atom_index,
        )
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            value_gradient,
            torch.ones_like(energy),
        )
        return energy.sum(dim=-1), gradient


class ProtenixLinearBondPotential(nn.Module):
    r"""
    Keep substituents around non-aromatic triple bonds linear.

    Args:
        atom_index: Angle triples with shape ``(3, M)``.
    """

    def __init__(self, atom_index: Tensor) -> None:
        super().__init__()
        self.atom_index: Tensor
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.08726646259) -> Tensor:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        angle = _protenix_angle(coordinates, self.atom_index)
        return torch.clamp_min(math.pi - buffer - angle, 0.0).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.08726646259,
    ) -> tuple[Tensor, Tensor]:
        coordinates = _protenix_coordinates(coordinates)
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = _protenix_angle_and_gradient(coordinates, self.atom_index)
        violation = math.pi - buffer - angle
        active = violation > 0
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            angle_gradient,
            -active.to(angle),
        )
        return torch.where(active, violation, 0.0).sum(dim=-1), gradient


def _protenix_bin_centers(
    tensor: Tensor,
    minimum: float,
    maximum: float,
    num_bins: int,
) -> Tensor:
    r"""
    Return the evenly spaced confidence-bin centers.
    """

    width = (maximum - minimum) / num_bins
    return (
        torch.linspace(
            minimum,
            maximum - width,
            num_bins,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        + 0.5 * width
    )


def protenix_contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""
    Decode the probability of distance below 8 Å.
    """

    centers = _protenix_bin_centers(
        distogram_logits,
        2.3125,
        21.6875,
        distogram_logits.shape[-1],
    )
    return categorical_probability_below(distogram_logits, centers, 8.0)


def protenix_confidence_summary(
    output: ProtenixSamplingOutput,
    input: ProtenixInput,
    *,
    contact_probability: Tensor | None = None,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""
    Decode Protenix confidence predictions for one molecular assembly.

    Args:
        output: Raw predictions with an explicit sample axis.
        input: The corresponding processor output, with dense batch axis ``B = 1``.
        contact_probability: Optional contact probabilities already aligned to
            the output token axes, with shape ``(B, N, N)`` or
            ``(B, S, N, N)``. ``None`` decodes the shared distogram.

    Returns:
        Named tensors for pLDDT, PAE, PDE, resolved and contact probabilities,
            global and chain confidence summaries, clashes, and ranking.
            Sample-dependent values retain the leading sample axis.
            Ranking is ``0.8 * ipTM + 0.2 * pTM - 100 * has_clash``; no
            disorder estimator is provided.

    Raises:
        ValueError: If either dense batch axis is not ``B = 1``, or an explicit
            contact probability has incompatible axes.
    """
    output = ProtenixSamplingOutput(**output)
    output.coordinates = project_model_coordinates(
        output.coordinates,
        input.atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(output.coordinates[..., 0], dtype=torch.bool),
        feature="Protenix confidence coordinates",
    )

    if input.token_mask.shape[0] != 1 or output.coordinates.shape[0] != 1:
        fail("value_invalid", RequestRef("input", str(input.token_mask.shape[0])), invariant="batch_size_one")

    atom_selection = input.atom_mask[0].bool().nonzero(as_tuple=False).flatten()
    token_selection = input.token_mask[0].bool().nonzero(as_tuple=False).flatten()
    atom_output = atom_selection.to(output.plddt_logits.device)
    token_output = token_selection.to(output.pae_logits.device)
    plddt_logits = output.plddt_logits[0][:, atom_output]
    pae_logits = output.pae_logits[0][:, token_output][:, :, token_output]
    pde_tokens = token_selection.to(output.pde_logits.device)
    pde_logits = output.pde_logits[0][:, pde_tokens][:, :, pde_tokens]
    resolved_logits = output.resolved_logits[0][:, atom_selection.to(output.resolved_logits.device)]
    distogram_tokens = token_selection.to(output.distogram_logits.device)
    distogram_logits = output.distogram_logits[0][distogram_tokens][:, distogram_tokens]

    atom_plddt = plddt_from_logits(plddt_logits)
    pae_probability = pae_logits.softmax(dim=-1)
    pae = categorical_expectation(pae_probability, _protenix_bin_centers(pae_logits, 0.0, 32.0, pae_logits.shape[-1]))
    pde = categorical_mean(pde_logits, end=32.0)
    resolved_probability = resolved_probability_from_logits(resolved_logits)
    if contact_probability is None:
        contact_probability = protenix_contact_probability(distogram_logits)
    elif contact_probability.ndim == 3 and contact_probability.shape[0] == 1:
        selected = token_selection.to(contact_probability.device)
        contact_probability = contact_probability[0][selected][:, selected]
    elif (
        contact_probability.ndim == 4
        and contact_probability.shape[0] == 1
        and contact_probability.shape[1] == output.coordinates.shape[1]
    ):
        selected = token_selection.to(contact_probability.device)
        contact_probability = contact_probability[0][:, selected][:, :, selected]
    else:
        fail("value_invalid", RequestRef("contact_probability", str(tuple(contact_probability.shape))))
    contact_probability = contact_probability.to(device=pde.device)
    token_has_frame = input.frame_mask[0][token_selection].to(torch.bool)
    token_chain = input.chain_index[0][token_selection].to(device=pae_logits.device, dtype=torch.long)
    compact_token_index = torch.full_like(input.token_mask[0], -1, dtype=torch.long)
    compact_token_index[token_selection] = torch.arange(token_selection.numel(), device=token_selection.device)
    atom_to_token = compact_token_index[input.atom_to_token[0][atom_selection].long()]
    if bool((atom_to_token < 0).any()):
        fail("value_invalid", RequestRef("input", "atom_to_token"), invariant="valid_atom_in_valid_token")
    atom_to_token = atom_to_token.to(device=pae_logits.device, dtype=torch.long)
    atom_is_polymer = input.atom_is_polymer[0][atom_selection].to(device=pae_logits.device, dtype=torch.bool)
    ptm, iptm, chain_ptm, chain_mean_partner_iptm, chain_pair_complex_iptm, chain_pair_mean_partner_iptm = (
        _protenix_tm_scores(
            pae_probability,
            token_has_frame=token_has_frame,
            token_chain=token_chain,
            atom_to_token=atom_to_token,
            atom_is_polymer=atom_is_polymer,
            backend=backend,
        )
    )
    gpde = weighted_mean(pde.flatten(-2), contact_probability.flatten(-2), backend=backend)
    chain_gpde, chain_pair_gpde = _protenix_chain_gpde(
        pde, contact_probability=contact_probability, token_chain=token_chain, backend=backend
    )
    chain_plddt, chain_pair_plddt = _protenix_chain_plddt(
        atom_plddt,
        token_chain=token_chain,
        atom_to_token=atom_to_token,
    )
    has_clash = _protenix_has_clash(
        output.coordinates[0][:, atom_selection.to(output.coordinates.device)],
        atom_to_token=atom_to_token,
        token_chain=token_chain,
        atom_is_polymer=atom_is_polymer,
    )
    ranking_score = 0.8 * iptm + 0.2 * ptm - 100.0 * has_clash.to(ptm.dtype)
    return {
        "plddt": atom_plddt,
        "pae": pae,
        "pde": pde,
        "resolved_probability": resolved_probability,
        "contact_probability": contact_probability,
        "ptm": ptm,
        "iptm": iptm,
        "chain_ptm": chain_ptm,
        "chain_mean_partner_iptm": chain_mean_partner_iptm,
        "chain_pair_complex_iptm": chain_pair_complex_iptm,
        "chain_pair_mean_partner_iptm": chain_pair_mean_partner_iptm,
        "gpde": gpde,
        "chain_gpde": chain_gpde,
        "chain_pair_gpde": chain_pair_gpde,
        "chain_plddt": chain_plddt,
        "chain_pair_plddt": chain_pair_plddt,
        "has_clash": has_clash,
        "ranking_score": ranking_score,
    }


def _protenix_tm_scores(
    pae_probability: Tensor,
    *,
    token_has_frame: Tensor,
    token_chain: Tensor,
    atom_to_token: Tensor,
    atom_is_polymer: Tensor,
    backend: str = "auto",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Compute global and chain TM summaries.
    """

    token_has_frame = token_has_frame.to(device=pae_probability.device, dtype=torch.bool)
    token_chain = token_chain.to(device=pae_probability.device, dtype=torch.long)

    def calculate(probability: Tensor, has_frame: Tensor, chain: Tensor) -> tuple[Tensor, Tensor]:
        num_tokens = has_frame.shape[0]
        centers = _protenix_bin_centers(probability, 0.0, 32.0, probability.shape[-1])
        token_pair_tm = categorical_expectation(probability, tm_bin_weights(centers, num_tokens))
        ptm_score = predicted_tm_score(token_pair_tm, reference_weights=has_frame, backend=backend)
        different_chain = chain[None, :] != chain[:, None]
        # An explicit source frame retains its undefined mean without cross-chain targets.
        iptm_score = predicted_tm_score(
            token_pair_tm, pair_weights=different_chain, reference_weights=has_frame, backend=backend
        )
        return ptm_score, iptm_score

    ptm, iptm = calculate(pae_probability, token_has_frame, token_chain)
    chains, chain_inverse = torch.unique(token_chain, return_inverse=True)
    chain_masks = tuple(token_chain == chain for chain in chains)
    batch_shape = pae_probability.shape[:-3]
    chain_ptm = pae_probability.new_zeros(batch_shape + (len(chains),))
    chain_pair_complex_iptm = pae_probability.new_zeros(batch_shape + (len(chains), len(chains)))
    for first_index, first_mask in enumerate(chain_masks):
        first_probability = pae_probability[..., first_mask, :, :][..., :, first_mask, :]
        chain_ptm[..., first_index] = calculate(
            first_probability,
            token_has_frame[first_mask],
            token_chain[first_mask],
        )[0]
        for second_index in range(first_index + 1, len(chains)):
            pair_mask = first_mask | chain_masks[second_index]
            pair_probability = pae_probability[..., pair_mask, :, :][..., :, pair_mask, :]
            pair_iptm = calculate(
                pair_probability,
                token_has_frame[pair_mask],
                token_chain[pair_mask],
            )[1]
            chain_pair_complex_iptm[..., first_index, second_index] = pair_iptm
            chain_pair_complex_iptm[..., second_index, first_index] = pair_iptm

    atom_to_token = atom_to_token.to(device=pae_probability.device, dtype=torch.long)
    atom_is_polymer = atom_is_polymer.to(device=pae_probability.device, dtype=torch.bool)
    token_is_ligand = (
        torch.zeros_like(token_chain).scatter_add(
            0,
            atom_to_token,
            (~atom_is_polymer).long(),
        )
        > 0
    )
    chain_has_frame = torch.zeros_like(chains).scatter_add(0, chain_inverse, token_has_frame.long()) > 0
    chain_mean_partner_iptm = mean_partner_tm_scores(
        chain_pair_complex_iptm,
        pair_mask=chain_has_frame[:, None] | chain_has_frame[None, :],
        backend=backend,
    )

    # Retain the at-least-half criterion without rounding odd chain sizes down.
    chain_is_ligand = tuple(2 * token_is_ligand[mask].sum() >= mask.sum() for mask in chain_masks)
    chain_pair_mean_partner_iptm = torch.zeros_like(chain_pair_complex_iptm)
    for first_index in range(len(chains)):
        for second_index in range(len(chains)):
            if first_index == second_index:
                continue
            if chain_is_ligand[first_index]:
                value = chain_mean_partner_iptm[..., first_index]
            elif chain_is_ligand[second_index]:
                value = chain_mean_partner_iptm[..., second_index]
            else:
                value = 0.5 * (chain_mean_partner_iptm[..., first_index] + chain_mean_partner_iptm[..., second_index])
            chain_pair_mean_partner_iptm[..., first_index, second_index] = value
    return (
        ptm,
        iptm,
        chain_ptm,
        chain_mean_partner_iptm,
        chain_pair_complex_iptm,
        chain_pair_mean_partner_iptm,
    )


def _protenix_chain_gpde(
    token_pair_pde: Tensor,
    *,
    contact_probability: Tensor,
    token_chain: Tensor,
    backend: str = "auto",
) -> tuple[Tensor, Tensor]:
    r"""
    Compute chain and chain-pair gPDE summaries.
    """

    token_chain = token_chain.to(device=token_pair_pde.device, dtype=torch.long)
    contact_probability = contact_probability.to(device=token_pair_pde.device)
    chains = torch.unique(token_chain)
    mean_dtype = torch.promote_types(token_pair_pde.dtype, contact_probability.dtype)
    chain_gpde = torch.zeros(token_pair_pde.shape[:-2] + (len(chains),), device=token_pair_pde.device, dtype=mean_dtype)
    chain_pair_gpde = torch.zeros(
        token_pair_pde.shape[:-2] + (len(chains), len(chains)), device=token_pair_pde.device, dtype=mean_dtype
    )

    def calculate(row_mask: Tensor, column_mask: Tensor) -> Tensor:
        contact = contact_probability[..., row_mask, :][..., column_mask]
        pde = token_pair_pde[..., row_mask, :][..., column_mask]
        return weighted_mean(pde.flatten(-2), contact.flatten(-2), backend=backend)

    chain_masks = tuple(token_chain == chain for chain in chains)
    for first_index, first_mask in enumerate(chain_masks):
        chain_gpde[..., first_index] = calculate(first_mask, first_mask)
        for second_index in range(first_index + 1, len(chains)):
            value = calculate(first_mask, chain_masks[second_index])
            chain_pair_gpde[..., first_index, second_index] = value
            chain_pair_gpde[..., second_index, first_index] = value
    return chain_gpde, chain_pair_gpde


def _protenix_chain_plddt(
    atom_plddt: Tensor,
    *,
    token_chain: Tensor,
    atom_to_token: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute chain and chain-pair pLDDT summaries on the 0–100 scale.
    """

    token_chain = token_chain.to(device=atom_plddt.device, dtype=torch.long)
    atom_to_token = atom_to_token.to(device=atom_plddt.device, dtype=torch.long)
    chains = torch.unique(token_chain)
    atom_chain = token_chain[atom_to_token]
    chain_plddt = atom_plddt.new_zeros(atom_plddt.shape[:-1] + (len(chains),))
    chain_pair_plddt = atom_plddt.new_zeros(atom_plddt.shape[:-1] + (len(chains), len(chains)))
    for first_index, first_chain in enumerate(chains):
        first_mask = atom_chain == first_chain
        chain_plddt[..., first_index] = atom_plddt[..., first_mask].mean(dim=-1)
        for second_index, second_chain in enumerate(chains):
            if first_index == second_index:
                continue
            pair_mask = first_mask | (atom_chain == second_chain)
            chain_pair_plddt[..., first_index, second_index] = atom_plddt[..., pair_mask].mean(dim=-1)
    return chain_plddt, chain_pair_plddt


def _protenix_has_clash(
    coordinates: Tensor,
    *,
    atom_to_token: Tensor,
    token_chain: Tensor,
    atom_is_polymer: Tensor,
) -> Tensor:
    r"""
    Apply the AF3-style inter-polymer-chain clash rule.
    """

    atom_to_token = atom_to_token.to(device=coordinates.device, dtype=torch.long)
    token_chain = token_chain.to(device=coordinates.device, dtype=torch.long)
    atom_chain = token_chain[atom_to_token]
    atom_is_polymer = atom_is_polymer.to(device=coordinates.device, dtype=torch.bool)
    polymer_chains = torch.unique(atom_chain[atom_is_polymer])
    has_clash = torch.zeros(coordinates.shape[0], dtype=torch.bool, device=coordinates.device)
    for first_index, first_chain in enumerate(polymer_chains):
        first_mask = atom_is_polymer & (atom_chain == first_chain)
        for second_chain in polymer_chains[first_index + 1 :]:
            second_mask = atom_is_polymer & (atom_chain == second_chain)
            distances = pairwise_distance(coordinates[:, first_mask], coordinates[:, second_mask])
            clashes = (distances < 1.1).sum(dim=(-1, -2))
            shorter_chain = min(int(first_mask.sum()), int(second_mask.sum()))
            has_clash |= (clashes > 100) | (clashes / shorter_chain > 0.5)
    return has_clash


def _validate_protenix_tfg_guidance(guidance: ProtenixTfgGuidance) -> None:
    if guidance.num_monte_carlo_samples < 1:
        fail("value_invalid", RequestRef("num_monte_carlo_samples", str(guidance.num_monte_carlo_samples)))
    if guidance.num_outer_steps < 1:
        fail("value_invalid", RequestRef("num_outer_steps", str(guidance.num_outer_steps)))
    if guidance.num_inner_steps < 0:
        fail("value_invalid", RequestRef("num_inner_steps", str(guidance.num_inner_steps)))
    if guidance.num_projection_outer_steps < 0:
        fail("value_invalid", RequestRef("num_projection_outer_steps", str(guidance.num_projection_outer_steps)))
    if guidance.num_projection_inner_steps < 0:
        fail("value_invalid", RequestRef("num_projection_inner_steps", str(guidance.num_projection_inner_steps)))

    terms = (
        guidance.vina_steric,
        guidance.experimental_torsion,
        guidance.interchain_bond,
        guidance.pairwise_distance,
        guidance.chiral_atom,
        guidance.stereo_bond,
        guidance.planar_improper,
        guidance.linear_bond,
    )
    if all(term is None for term in terms):
        fail("value_invalid", RequestRef("tfg_guidance", "no term"))
    for term in terms:
        if term is not None and term.interval < 1:
            fail("value_invalid", RequestRef(f"{type(term).__name__}.interval", str(term.interval)))


def _protenix_tfg_scalar(
    value: float | Callable[[float], float],
    time: float,
) -> float:
    return float(value(time) if callable(value) else value)


def _protenix_tfg_monte_carlo_noise(
    guidance: ProtenixTfgGuidance,
    coordinates: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    if guidance.monte_carlo_std == 0.0:
        return coordinates.new_zeros((1, *coordinates.shape))
    noise = draw_normal(
        (guidance.num_monte_carlo_samples, *coordinates.shape),
        device=coordinates.device,
        dtype=coordinates.dtype,
        generator=generator,
    )
    return guidance.monte_carlo_std * noise


def _protenix_vina_pairs(
    atom_chain: Tensor,
    atom_vdw_radius: Tensor,
    interchain_bond_index: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the released interchain Vina candidate set once per sampling call.
    """

    chain_ids, counts = torch.unique(atom_chain, sorted=True, return_counts=True)
    valid_chains = {
        int(chain_id) for chain_id, count in zip(chain_ids.cpu().tolist(), counts.cpu().tolist()) if int(count) > 1
    }
    chain_by_atom = atom_chain.cpu().tolist()
    prohibited = {
        tuple(sorted((chain_by_atom[first], chain_by_atom[second])))
        for first, second in interchain_bond_index.T.cpu().tolist()
    }
    first_atoms: list[Tensor] = []
    second_atoms: list[Tensor] = []
    ordered_chains = sorted(valid_chains)
    for position, first_chain in enumerate(ordered_chains):
        first = torch.where(atom_chain == first_chain)[0]
        for second_chain in ordered_chains[position + 1 :]:
            if (first_chain, second_chain) in prohibited:
                continue
            second = torch.where(atom_chain == second_chain)[0]
            first_atoms.append(first.repeat_interleave(second.numel()))
            second_atoms.append(second.repeat(first.numel()))
    if not first_atoms:
        return (
            torch.empty((2, 0), device=atom_chain.device, dtype=torch.long),
            atom_vdw_radius.new_empty((0,)),
        )
    pair_index = torch.stack(
        (
            torch.cat(first_atoms),
            torch.cat(second_atoms),
        )
    )
    return pair_index, atom_vdw_radius[pair_index].sum(dim=0)


def _protenix_zero_energy(coordinates: Tensor) -> Tensor:
    return coordinates[..., :0, :].sum(dim=(-1, -2))


def _protenix_vina_energy_terms(
    coordinates: Tensor,
    atom_index: Tensor,
    equilibrium_distance: Tensor,
    *,
    buffer: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Evaluate complete Vina energy and retain its active-pair terms for analytic differentiation."""

    if atom_index.numel() == 0:
        empty_index = atom_index.new_empty((0,))
        empty = coordinates.new_empty((0,))
        return _protenix_zero_energy(coordinates), empty_index, empty_index, empty, empty, empty

    batch_shape = coordinates.shape[:-2]
    num_atoms = coordinates.shape[-2]
    flattened_batch = math.prod(batch_shape) if batch_shape else 1
    flattened_coordinates = coordinates.reshape(flattened_batch, num_atoms, 3)
    distances = indexed_distance(flattened_coordinates, atom_index, minimum=1e-8)
    equilibrium = equilibrium_distance.to(device=distances.device)
    active = distances < equilibrium.unsqueeze(0) * (1.0 - buffer)
    batch_index, pair_index = active.nonzero(as_tuple=True)
    if batch_index.numel() == 0:
        empty = coordinates.new_empty((0,))
        return _protenix_zero_energy(coordinates), batch_index, pair_index, empty, empty, empty

    active_equilibrium = equilibrium.index_select(0, pair_index)
    active_distance = distances[batch_index, pair_index]
    difference = active_distance - active_equilibrium
    first_gaussian = -0.0356 * torch.exp(-((difference / 0.5) ** 2))
    second_gaussian = -0.00516 * torch.exp(-(((difference - 3.0) / 2.0) ** 2))
    repulsion = 0.840 * torch.where(difference < 0.0, difference**2, 0.0)

    contributions = first_gaussian + second_gaussian + repulsion
    energy = contributions.new_zeros((flattened_batch,))
    energy.scatter_add_(0, batch_index, contributions)
    return energy.reshape(batch_shape), batch_index, pair_index, difference, first_gaussian, second_gaussian


def _protenix_vina_energy_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
    equilibrium_distance: Tensor,
    *,
    buffer: float,
) -> tuple[Tensor, Tensor]:
    energy, batch_index, pair_index, difference, first_gaussian, second_gaussian = _protenix_vina_energy_terms(
        coordinates, atom_index, equilibrium_distance, buffer=buffer
    )
    if batch_index.numel() == 0:
        return energy, torch.zeros_like(coordinates)

    num_atoms = coordinates.shape[-2]
    first_atom = atom_index[0].index_select(0, pair_index)
    second_atom = atom_index[1].index_select(0, pair_index)
    flat_first_atom = batch_index * num_atoms + first_atom
    flat_second_atom = batch_index * num_atoms + second_atom
    _, endpoint_gradient = indexed_distance_with_gradient(
        coordinates.reshape(-1, 3), torch.stack((flat_first_atom, flat_second_atom)), minimum=1e-8
    )
    distance_derivative = (
        -2.0 * first_gaussian * (difference / 0.5) * (1.0 / 0.5)
        - 0.5 * second_gaussian * (difference - 3.0)
        + 0.840 * torch.where(difference < 0.0, 2.0 * difference, 0.0)
    )
    first_gradient = distance_derivative.unsqueeze(-1) * endpoint_gradient[0]
    flattened_gradient = first_gradient.new_zeros(coordinates.shape).reshape(-1, 3)
    flattened_gradient.scatter_add_(
        0,
        flat_first_atom.unsqueeze(-1).expand(-1, 3),
        first_gradient,
    )
    flattened_gradient.scatter_add_(
        0,
        flat_second_atom.unsqueeze(-1).expand(-1, 3),
        -first_gradient,
    )
    gradient = flattened_gradient.reshape(*coordinates.shape)
    return energy, gradient


def _protenix_angle(coordinates: Tensor, atom_index: Tensor) -> Tensor:
    first = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[1])
    second = coordinates.index_select(-2, atom_index[2]) - coordinates.index_select(-2, atom_index[1])
    return torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1),
        torch.linalg.vecdot(first, second) + 1e-8,
    )


def _protenix_angle_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    first = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[1])
    second = coordinates.index_select(-2, atom_index[2]) - coordinates.index_select(-2, atom_index[1])
    cross = torch.linalg.cross(first, second, dim=-1)
    cross_norm = torch.linalg.vector_norm(cross, dim=-1)
    dot = torch.linalg.vecdot(first, second) + 1e-8
    angle = torch.atan2(cross_norm, dot)
    # Differentiate this stabilized atan2, including its radial epsilon term.
    # vector_norm uses the zero subgradient when the cross product is zero.
    safe_cross_norm = torch.where(cross_norm > 0, cross_norm, 1.0).unsqueeze(-1)
    denominator = (cross_norm.square() + dot.square()).unsqueeze(-1)
    first_gradient = (
        dot.unsqueeze(-1) * torch.linalg.cross(second, cross, dim=-1) / safe_cross_norm
        - cross_norm.unsqueeze(-1) * second
    ) / denominator
    third_gradient = (
        dot.unsqueeze(-1) * torch.linalg.cross(cross, first, dim=-1) / safe_cross_norm
        - cross_norm.unsqueeze(-1) * first
    ) / denominator
    second_gradient = -first_gradient - third_gradient
    return angle, torch.stack(
        (first_gradient, second_gradient, third_gradient),
        dim=-3,
    )


def _protenix_planar_improper_geometry(coordinates: Tensor, atom_index: Tensor) -> tuple[Tensor, ...]:
    center = coordinates.index_select(-2, atom_index[2])
    first_atom = coordinates.index_select(-2, atom_index[0])
    second_atom = coordinates.index_select(-2, atom_index[1])
    third_atom = coordinates.index_select(-2, atom_index[3])
    # Preserve native physical cutoffs without using overflowing lengths as divisors.
    valid_distance = (
        (torch.linalg.vector_norm(first_atom - center, dim=-1) > 1e-5)
        & (torch.linalg.vector_norm(second_atom - center, dim=-1) > 1e-5)
        & (torch.linalg.vector_norm(third_atom - center, dim=-1) > 1e-5)
    )
    first = torch.where(valid_distance.unsqueeze(-1), 0.5 * first_atom - 0.5 * center, 0.0)
    second = torch.where(valid_distance.unsqueeze(-1), 0.5 * second_atom - 0.5 * center, 0.0)
    third = torch.where(valid_distance.unsqueeze(-1), 0.5 * third_atom - 0.5 * center, 0.0)

    def normalize(vector: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scale = vector.abs().amax(dim=-1, keepdim=True).detach()
        scale = torch.where(scale > 0, scale, 1.0)
        scaled = vector / scale
        length = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        length = torch.where(length > 0, length, 1.0)
        return scaled / length, scale, length

    first, first_scale, first_length = normalize(first)
    second, second_scale, second_length = normalize(second)
    normal = torch.linalg.cross(-first, second, dim=-1)
    normal_length = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    valid = valid_distance & (normal_length.squeeze(-1) > 1e-5)
    normal_length = torch.where(valid.unsqueeze(-1), normal_length, 1.0)
    normal = torch.where(valid.unsqueeze(-1), normal, 0.0) / normal_length
    third, third_scale, third_length = normalize(torch.where(valid.unsqueeze(-1), third, 0.0))
    return (
        first,
        second,
        third,
        normal,
        first_scale,
        second_scale,
        third_scale,
        first_length,
        second_length,
        third_length,
        normal_length,
        valid,
    )


def _protenix_planar_improper_energy(coordinates: Tensor, atom_index: Tensor) -> Tensor:
    (
        _first,
        _second,
        third,
        normal,
        _first_scale,
        _second_scale,
        _third_scale,
        _first_length,
        _second_length,
        _third_length,
        _normal_length,
        valid,
    ) = _protenix_planar_improper_geometry(coordinates, atom_index)
    sine = torch.linalg.vector_norm(torch.linalg.cross(normal, third, dim=-1), dim=-1)
    return torch.where(valid, 1.0 - sine.clamp_min(1e-8), 0.0)


def _protenix_planar_improper_energy_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    (
        first,
        second,
        third,
        normal,
        first_scale,
        second_scale,
        third_scale,
        first_length,
        second_length,
        third_length,
        normal_length,
        valid,
    ) = _protenix_planar_improper_geometry(coordinates, atom_index)
    cross = torch.linalg.cross(normal, third, dim=-1)
    sine = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
    energy = torch.where(valid, 1.0 - sine.squeeze(-1).clamp_min(1e-8), 0.0)

    # A Half angular floor may round to zero; the norm's zero subgradient still applies.
    active = valid.unsqueeze(-1) & (sine >= 1e-8) & (sine > 0)
    cross_gradient = -torch.where(active, cross, 0.0) / torch.where(active, sine, 1.0)
    normal_gradient = torch.linalg.cross(third, cross_gradient, dim=-1)
    third_gradient = torch.linalg.cross(cross_gradient, normal, dim=-1)
    normal_gradient = normal_gradient - normal * (normal * normal_gradient).sum(dim=-1, keepdim=True)
    first_gradient = torch.linalg.cross(normal_gradient, second, dim=-1)
    second_gradient = torch.linalg.cross(first, normal_gradient, dim=-1)
    # Apply the coordinate-scale pullback before small length divisions, and defer
    # the plane inverse until after the complete endpoint tangent projection.
    first_gradient = (
        (first_gradient - first * (first * first_gradient).sum(dim=-1, keepdim=True))
        * 0.5
        / first_length
        / first_scale
        / normal_length
    )
    second_gradient = (
        (second_gradient - second * (second * second_gradient).sum(dim=-1, keepdim=True))
        * 0.5
        / second_length
        / second_scale
        / normal_length
    )
    third_gradient = (
        (third_gradient - third * (third * third_gradient).sum(dim=-1, keepdim=True)) * 0.5 / third_length / third_scale
    )
    center_gradient = -(first_gradient + second_gradient + third_gradient)
    gradient = torch.stack(
        (first_gradient, second_gradient, center_gradient, third_gradient),
        dim=-3,
    )
    return energy, gradient


def _protenix_aggregate_atom_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
    value_gradient: Tensor,
    energy_derivative: Tensor,
) -> Tensor:
    combined = energy_derivative.unsqueeze(-2).unsqueeze(-1) * value_gradient
    flattened_gradient = combined.flatten(start_dim=-3, end_dim=-2)
    flattened_index = atom_index.flatten()
    batch_shape = coordinates.shape[:-2]
    target_index = flattened_index.view(
        *([1] * len(batch_shape)),
        -1,
        1,
    ).expand(*batch_shape, -1, 3)
    gradient = combined.new_zeros(coordinates.shape)
    gradient.scatter_add_(
        dim=-2,
        index=target_index,
        src=flattened_gradient,
    )
    return gradient


def _protenix_constraint_projection(
    coordinates: Tensor,
    atom_index: Tensor,
    violation: Tensor,
    value_gradient: Tensor,
    active: Tensor,
) -> Tensor:
    r"""
    Solve the minimum-norm linearized correction for each coordinate sample.
    """

    batch_shape = coordinates.shape[:-2]
    num_atoms = coordinates.shape[-2]
    num_constraints = atom_index.shape[1]
    flattened_batch = math.prod(batch_shape) if batch_shape else 1
    dtype = torch.promote_types(value_gradient.dtype, violation.dtype)
    violation = violation.to(dtype=dtype).reshape(flattened_batch, num_constraints)
    active = active.reshape(flattened_batch, num_constraints)
    value_gradient = value_gradient.to(dtype=dtype).reshape(
        flattened_batch,
        atom_index.shape[0],
        num_constraints,
        3,
    )
    delta = value_gradient.new_zeros((flattened_batch, num_atoms, 3))
    for batch_index in range(flattened_batch):
        selected = active[batch_index]
        num_selected = int(selected.sum())
        if num_selected == 0:
            continue
        selected_index = atom_index[:, selected]
        selected_gradient = value_gradient[batch_index, :, selected, :]
        atom_gradient = value_gradient.new_zeros((num_selected, num_atoms, 3))
        rows = torch.arange(num_selected, device=coordinates.device)
        for endpoint in range(atom_index.shape[0]):
            atom_gradient[rows, selected_index[endpoint]] += selected_gradient[endpoint]
        jacobian = atom_gradient.reshape(num_selected, num_atoms * 3)
        right_hand_side = violation[batch_index, selected]
        gram = jacobian @ jacobian.T
        gram = gram + 1e-8 * torch.eye(
            num_selected,
            device=coordinates.device,
            dtype=jacobian.dtype,
        )
        multiplier = torch.linalg.solve(gram, right_hand_side)
        delta[batch_index] = (-(jacobian.T @ multiplier)).reshape(num_atoms, 3)
    return delta.reshape(*coordinates.shape)


def _protenix_preserve_chain_radius(
    coordinates: Tensor,
    delta: Tensor,
    *,
    atom_index: Tensor,
    atom_chain: Tensor,
) -> Tensor:
    r"""Rescale the proposed move of the constrained atoms so each chain keeps its centre's radius of gyration."""
    affected = torch.zeros(coordinates.shape[-2], dtype=torch.bool, device=coordinates.device)
    affected[atom_index.flatten()] = True
    selected = affected.nonzero().squeeze(-1)
    chains, group = torch.unique(atom_chain[selected], return_inverse=True)
    count = len(chains)
    original = coordinates[..., selected, :]
    proposed = original + delta[..., selected, :]
    original_center = scatter_mean(original, group, dim=-2, dim_size=count)[..., group, :]
    proposed_center = scatter_mean(proposed, group, dim=-2, dim_size=count)[..., group, :]
    original_radius = scatter_mean(((original - original_center) ** 2).sum(dim=-1), group, dim=-1, dim_size=count)
    proposed_radius = scatter_mean(((proposed - proposed_center) ** 2).sum(dim=-1), group, dim=-1, dim_size=count)
    scale = torch.sqrt(original_radius / proposed_radius.clamp_min(1e-12))[..., group, None]
    corrected = delta.clone()
    corrected[..., selected, :] = (proposed - proposed_center) * scale + original_center - original
    return corrected


def _gather_protenix_structure_target(
    target: ProtenixStructureTarget,
    input: ProtenixInput,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather canonical truth onto the current model atom axis.
    """

    return gather_model_coordinates(
        target.coordinates,
        target.coordinate_mask,
        input.atom_index,
        input.atom_mask,
    )


class ProtenixConfidencePairEmbedder(ConfidencePairEmbedder):
    r"""
    Confidence pair embedder of the released Protenix head.

    Adds a linear projection of the unbucketed representative-atom distance alongside the shared
    one-hot binned projection.
    """

    def __init__(
        self,
        token_feature_size: int,
        pair_size: int = 128,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__(
            token_feature_size=token_feature_size,
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.distance_projection = nn.Linear(1, pair_size, bias=False)

    def embed_distance(self, pair: Tensor, distance: Tensor) -> Tensor:
        r"""
        Add the one-hot binned distance projection, then the unbucketed distance projection.

        Args:
            pair: Pair state already carrying the left and right token projections.
            distance: Representative-atom distances in Angstrom, in ``float32``.

        Returns:
            The pair state with both distance contributions added.
        """

        pair = super().embed_distance(pair, distance)
        return pair + self.distance_projection(distance.to(pair.dtype).unsqueeze(-1))


class ProtenixPairConfidenceProjector(PairConfidenceProjector):
    r"""
    Pair confidence projector of the released Protenix head.

    Symmetrizes the pair representation before normalizing and projecting it into PDE logits,
    rather than projecting per ordered pair and symmetrizing the logits afterward.
    """

    def project_pde(self, pair: Tensor) -> Tensor:
        r"""
        Symmetrize the pair representation, then normalize and project it.

        Args:
            pair: Pair representation cast to the PDE projection dtype.

        Returns:
            PDE logits.
        """

        return self.pde_projection(self.pde_norm(pair + pair.transpose(-2, -3)))


class ProtenixConfidenceHead(ConfidenceHead):
    r"""
    Confidence head of the released Protenix model.

    Layer-normalizes the clamped single representation before the Pairformer stack, which adds a
    learned affine layer the shared head has no equivalent for, and supplies the Protenix pair
    embedder and pair projector.

    Args:
        token_feature_size: Width of the incoming per-token feature vector.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        single_max_abs: Optional clamp on the single representation before scoring.
        backend: Public backend request forwarded to the atom projector and the single norm.
    """

    def __init__(
        self,
        token_feature_size: int,
        single_size: int = 384,
        pair_size: int = 128,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            token_feature_size=token_feature_size,
            single_size=single_size,
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            single_max_abs=single_max_abs,
            backend=backend,
        )
        self.single_norm = LayerNorm(single_size, backend=backend)

    def build_pair_embedder(
        self,
        *,
        token_feature_size: int,
        pair_size: int,
        distance_bin_start: float,
        distance_bin_end: float,
        distance_bin_step: float,
    ) -> ConfidencePairEmbedder:
        r"""
        Returns:
            A [`ProtenixConfidencePairEmbedder`][ofoldx.models.protenix.model.ProtenixConfidencePairEmbedder].
        """

        return ProtenixConfidencePairEmbedder(
            token_feature_size=token_feature_size,
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )

    def build_pair_projector(
        self, *, pair_size: int, num_pae_bins: int, num_pde_bins: int, backend: str = "auto"
    ) -> PairConfidenceProjector:
        r"""
        Returns:
            A [`ProtenixPairConfidenceProjector`][ofoldx.models.protenix.model.ProtenixPairConfidenceProjector].
        """

        return ProtenixPairConfidenceProjector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            backend=backend,
        )

    def prepare_single(self, single: Tensor) -> Tensor:
        r"""
        Clamp the single representation, then layer-normalize it.

        Args:
            single: Trunk single representation.

        Returns:
            The clamped and normalized single representation.
        """

        return self.single_norm(super().prepare_single(single))


# ===========================================================================
# Typed tensor and semantic boundary of the Protenix graph.
# ===========================================================================


@tensorclass
class ProtenixTemplateFeatures:
    r"""
    Fixed-four template features consumed by `ProtenixTemplateEmbedder`.

    Attributes:
        distogram: Template distograms with shape ``(B, 4, N, N, 39)``.
        pseudo_beta_mask: Pseudo-beta pair masks with shape ``(B, 4, N, N)``.
        residue_type_ids: Template residue type IDs with shape ``(B, 4, N)``.
        unit_vector: Backbone-frame pair vectors with shape ``(B, 4, N, N, 3)``.
        backbone_frame_mask: Backbone-frame pair masks with shape ``(B, 4, N, N)``.
    """

    distogram: Tensor
    pseudo_beta_mask: Tensor
    residue_type_ids: Tensor
    unit_vector: Tensor
    backbone_frame_mask: Tensor


@tensorclass
class ProtenixTfgFeatures:
    r"""
    Static molecular geometry consumed by Protenix training-free guidance.

    Index tensors are unbatched and use the model atom axis.

    Attributes:
        atom_chain: Chain index for every model atom with shape ``(A,)``.
        atom_vdw_radius: Van-der-Waals radius for every model atom with shape ``(A,)``.
        interchain_bond_index: Covalent interchain atom pairs with shape ``(2, M_bond)``.
        pairwise_distance_index: Component-local atom pairs with shape ``(2, M_pair)``.
        pairwise_distance_lower_bound: RDKit lower bounds with shape ``(M_pair,)``.
        pairwise_distance_upper_bound: RDKit upper bounds with shape ``(M_pair,)``.
        pairwise_distance_is_bond: Direct-bond mask with shape ``(M_pair,)``.
        pairwise_distance_is_angle: Bond-angle endpoint mask with shape ``(M_pair,)``.
        experimental_torsion_index: Torsion atom quadruples with shape ``(4, M_torsion)``.
        experimental_torsion_force_constant: Six Fourier force constants per torsion.
        experimental_torsion_sign: Six Fourier signs per torsion.
        linear_bond_index: Linear-angle atom triples with shape ``(3, M_linear)``.
        chiral_index: Chiral atom quadruples with shape ``(4, M_chiral)``.
        chiral_orientation: Reference chiral signs with shape ``(M_chiral,)``.
        stereo_bond_index: Stereo-bond quadruples with shape ``(4, M_stereo)``.
        stereo_bond_orientation: Reference stereo classes with shape ``(M_stereo,)``.
        planar_improper_index: Planar-improper quadruples with shape ``(4, M_planar)``.
    """

    atom_chain: Tensor
    atom_vdw_radius: Tensor
    interchain_bond_index: Tensor
    pairwise_distance_index: Tensor
    pairwise_distance_lower_bound: Tensor
    pairwise_distance_upper_bound: Tensor
    pairwise_distance_is_bond: Tensor
    pairwise_distance_is_angle: Tensor
    experimental_torsion_index: Tensor
    experimental_torsion_force_constant: Tensor
    experimental_torsion_sign: Tensor
    linear_bond_index: Tensor
    chiral_index: Tensor
    chiral_orientation: Tensor
    stereo_bond_index: Tensor
    stereo_bond_orientation: Tensor
    planar_improper_index: Tensor


class _ProtenixTrunkOutput(NamedTuple):
    r"""
    Named representations returned by the Protenix recycling trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@dataclass(frozen=True)
class ProtenixAtomContact:
    r"""
    Set an upper distance bound between two atoms.

    Attributes:
        atom_a: First atom occurrence.
        atom_b: Second atom occurrence.
        max_distance: Maximum distance in angstroms.
    """

    atom_a: AtomSite
    atom_b: AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if self.atom_a == self.atom_b:
            fail("value_invalid", RequestRef("atom_b", str(self.atom_b)), invariant="distinct_atoms")
        if self.max_distance <= 0:  # [0, 0] is indistinguishable from no contact
            fail("value_invalid", RequestRef("max_distance", str(self.max_distance)))


@dataclass(frozen=True)
class ProtenixTokenContact:
    r"""
    Set an upper distance bound between token-level endpoints.

    Attributes:
        endpoint_a: First residue or atom endpoint.
        endpoint_b: Second residue or atom endpoint.
        max_distance: Maximum token-center distance in angstroms.

    Two atom endpoints are the atom channel (``ProtenixAtomContact``).
    """

    endpoint_a: ComponentRef | AtomSite
    endpoint_b: ComponentRef | AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if isinstance(self.endpoint_a, AtomSite) and isinstance(self.endpoint_b, AtomSite):
            fail("value_invalid", RequestRef("endpoint_b", str(self.endpoint_b)), invariant="atom_channel")
        if self.max_distance <= 0:  # [0, 0] is indistinguishable from no contact
            fail("value_invalid", RequestRef("max_distance", str(self.max_distance)))


@dataclass(frozen=True)
class ProtenixPocket:
    r"""
    Condition one binder chain to contact a set of residues.

    Attributes:
        binder: Molecular instance ID of the binder chain.
        contact_residues: Target residue occurrences outside the binder chain.
        max_distance: Maximum binder-to-residue distance in angstroms.
    """

    binder: str
    contact_residues: tuple[ComponentRef, ...]
    max_distance: float

    def __post_init__(self) -> None:
        if not self.contact_residues:
            fail("value_invalid", RequestRef("contact_residues", "()"))
        if self.max_distance <= 0:
            fail("value_invalid", RequestRef("max_distance", str(self.max_distance)))
        for residue in self.contact_residues:
            if residue.instance == self.binder:
                fail("value_invalid", RequestRef("contact_residues", str(residue)), invariant="outside_binder")


@tensorclass
class ProtenixStructureTarget:
    r"""
    Canonical full-axis structure truth for Protenix training.

    Attributes:
        coordinates: Full canonical atom-axis coordinates with shape
            ``(B, T, 3)`` in angstroms.
        coordinate_mask: Resolved-coordinate mask with shape ``(B, T)``.
        resolution: Experimental resolution with shape ``(B,)`` in angstroms.
        resolution_mask: Whether each experimental resolution is known.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    resolution: Tensor
    resolution_mask: Tensor


@tensorclass
class ProtenixInput:
    r"""
    Tensor input of one prepared task (its atom axis is the slot axis of ``PreparedInput.context.slots``).

    ``B`` is the explicit tensor batch axis, ``N`` token count, ``A`` atom
    count, and ``M`` MSA depth. The tensorclass batch size is empty because
    ragged batches nest their axes rather than share a dense prefix.

    Attributes:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        atom_features: Reference atom features with shape ``(B, A, C_atom)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        atom_token_index: Atom indices within each token with shape ``(B, A)``.
        token_representative_atom_index: Representative atom index for each token with shape
            ``(B, N)``.
        frame_atom_index: Prepared reference-frame defining model atoms, with
            shape ``(B, N, 3)`` and ``-1`` for unavailable frame slots.
        frame_mask: Available, nondegenerate reference frames with shape ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, C_token)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        msa_tokens: MSA canonical classes with shape ``(B, M, N)``.
        msa_deletion_counts: Raw MSA deletion counts with shape ``(B, M, N)``.
        esm_tokens: ESM2 tokens for distinct encoded physical protein sequences,
            with shape ``(B, E, L)``.
        esm_mask: Valid ESM2 sequence-token mask with shape ``(B, E, L)``.
        token_to_esm_residue: Flattened ESM sequence/residue index for each
            model token with shape ``(B, N)``.
        token_has_esm: Protein-token mask with shape ``(B, N)``.
        pocket: Directed pocket-distance features with shape ``(B, N, N, 1)``
            in angstroms.
        token_contact: Token-contact distance features with shape
            ``(B, N, N, 2)`` in angstroms.
        atom_contact: Atom-contact distance features with shape
            ``(B, N, N, 2)`` in angstroms.
        substructure: Substructure-condition features with shape
            ``(B, N, N, 4)``.
        structure_rep_positions: Explicit representative-token structure
            coordinates with shape ``(B, N, 3)`` in angstroms.
        structure_rep_mask: Valid explicit structure-coordinate mask with shape
            ``(B, N)``.
        template_features: Optional fixed-four template features for base v1 and v2.
        tfg_features: Optional static geometry prepared for training-free guidance.
        atom_is_polymer: Whether each model atom belongs to a polymer instance, with shape ``(B, A)``.
        atom_index: Canonical atom index for each model atom (its slot's atom) with shape ``(B, A)``.
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
    atom_token_index: Tensor
    token_representative_atom_index: Tensor
    frame_atom_index: Tensor
    frame_mask: Tensor
    token_features: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    token_bonds: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    esm_tokens: Tensor
    esm_mask: Tensor
    token_to_esm_residue: Tensor
    token_has_esm: Tensor
    pocket: Tensor
    token_contact: Tensor
    atom_contact: Tensor
    substructure: Tensor
    structure_rep_positions: Tensor
    structure_rep_mask: Tensor
    template_features: ProtenixTemplateFeatures | None
    tfg_features: ProtenixTfgFeatures | None
    atom_is_polymer: Tensor
    atom_index: Tensor


def _project_protenix_templates(
    templates: ProtenixTemplateFeatures, *, dtype: torch.dtype | None = None, pair_mask: Tensor | None = None
) -> ProtenixTemplateFeatures:
    """Exclude unavailable derived template payload before numerical lowering."""
    template = cast(ProtenixTemplateFeatures, cast(TensorClass, templates).clone(recurse=False))
    pseudo_mask = template.pseudo_beta_mask.to(torch.bool)
    frame_mask = template.backbone_frame_mask.to(torch.bool)
    if pair_mask is not None:
        pseudo_mask = pseudo_mask & pair_mask.to(torch.bool).unsqueeze(-3)
        frame_mask = frame_mask & pair_mask.to(torch.bool).unsqueeze(-3)
    template.distogram = project_model_features(
        template.distogram,
        pseudo_mask[..., None],
        dtype=dtype,
        feature="Protenix template distogram",
    )
    template.unit_vector = project_model_features(
        template.unit_vector,
        frame_mask[..., None],
        dtype=dtype,
        feature="Protenix template vectors",
    )
    return template


def normalize_protenix_input(
    input: ProtenixInput, *, dtype: torch.dtype | None = None, max_atoms_per_token: int | None = None
) -> ProtenixInput:
    """Project finite features under their declared availability masks.

    This pure operation retains canonical chemistry, observation knownness and
    the processor's chosen atom/token/anchor correspondences.
    """
    result = cast(ProtenixInput, cast(TensorClass, input).clone(recurse=False))
    atom_mask = input.atom_mask.to(torch.bool)
    offsets = torch.where(atom_mask, input.atom_token_index, 0)
    nonnegative = (offsets >= 0).all()
    if torch.compiler.is_compiling():
        torch._assert_async(nonnegative, "Protenix atom offsets must be nonnegative")
    elif not bool(nonnegative):
        fail("value_invalid", RequestRef("atom_token_index", "negative"))
    if max_atoms_per_token is not None:
        representable = (offsets < max_atoms_per_token).all()
        if torch.compiler.is_compiling():
            torch._assert_async(representable, "Protenix atom offsets exceed confidence capacity")
        elif not bool(representable):
            fail("model_capacity", RequestRef("atom_token_index", str(int(offsets.max()))), limit=max_atoms_per_token)
    result.atom_token_index = offsets
    token_mask = input.token_mask.to(torch.bool)
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    result.reference_positions = project_model_coordinates(
        input.reference_positions,
        input.reference_mask.to(torch.bool) & atom_mask,
        dtype=dtype,
        feature="Protenix reference positions",
    )
    result.structure_rep_positions = project_model_coordinates(
        input.structure_rep_positions,
        input.structure_rep_mask.to(torch.bool) & token_mask,
        dtype=dtype,
        feature="Protenix structure conditioning",
    )
    for name, mask in (
        ("atom_features", atom_mask[..., None]),
        ("reference_charges", atom_mask[..., None]),
        ("token_features", token_mask[..., None]),
        ("token_bonds", pair_mask),
        ("pocket", pair_mask[..., None]),
        ("token_contact", pair_mask[..., None]),
        ("atom_contact", pair_mask[..., None]),
        ("substructure", pair_mask[..., None]),
    ):
        setattr(
            result, name, project_model_features(getattr(input, name), mask, dtype=dtype, feature=f"Protenix {name}")
        )
    if input.template_features is not None:
        result.template_features = _project_protenix_templates(
            input.template_features, dtype=dtype, pair_mask=pair_mask
        )
    msa_available = token_mask.unsqueeze(-2)
    result.msa_tokens = project_model_categories(
        input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="Protenix MSA tokens"
    )
    result.msa_deletion_counts = project_model_counts(
        input.msa_deletion_counts, msa_available, feature="Protenix MSA deletion counts"
    )
    return result


class ProtenixSamplingOutput(TensorCollection):
    r"""
    Predictions from one Protenix sampling-compatible denoising call.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        plddt_logits: Per-atom confidence logits with shape ``(B, S, A, D_plddt)``.
        pae_logits: Token-pair aligned-error logits with shape
            ``(B, S, N, N, D_pae)``.
        pde_logits: Token-pair distance-error logits with shape
            ``(B, S, N, N, D_pde)``.
        resolved_logits: Per-atom resolved-state logits with shape
            ``(B, S, A, D_resolved)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor

    @property
    def plddt(self) -> Tensor:
        r"""
        Return expected per-atom pLDDT.

        Returns:
            Confidence values with shape ``(B, S, A)`` on the ``0--100`` scale.
        """

        return plddt_from_logits(self.plddt_logits)


class ProtenixOutput(ProtenixSamplingOutput):
    r"""
    Predictions from one Protenix training denoising pass.

    Attributes:
        augmented_target_coordinates: Training targets after rigid augmentation,
            with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
    """

    augmented_target_coordinates: Tensor
    noise_level: Tensor


def _protenix_coordinates(coordinates: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Project a complete model-state coordinate field at a public boundary."""
    if coordinates.dim() == 0 or coordinates.shape[-1] != 3:
        fail("value_invalid", RequestRef("coordinates", str(tuple(coordinates.shape))))
    return project_model_coordinates(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        dtype=dtype,
        feature="Protenix model-state coordinates",
    )
