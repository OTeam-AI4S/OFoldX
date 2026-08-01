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
Boltz-2 model.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.molecular import AtomSite, ComponentRef
from ...data.tokenization import TOKENS
from ...errors import RequestRef, StructureRef, fail
from ...metrics import (
    categorical_expectation,
    categorical_mean,
    chain_pair_tm_scores,
    directional_ipsae,
    interface_confidence_weights,
    minimum_confidence_error,
    plddt_from_logits,
    predicted_tm_score,
    resolved_probability_from_logits,
    tm_weight_from_pae,
)
from ...model import Model, OptionalTensor, TensorCollection
from ...modules.atom.coordinates import gather_atoms_by_index
from ...modules.atom.encoders import (
    AtomPairEncoder,
    AtomToTokenEncoder,
    CoordinateAtomToTokenEncoder,
    TokenToAtomDecoder,
)
from ...modules.atom.window import LocalAtomWindowing
from ...modules.attention import PerBlockPairBiasProjection
from ...modules.backend import LayerNorm
from ...modules.batch import insert_sample_axis
from ...modules.denoiser import AtomDenoiserConditioning, AtomDenoiserContext
from ...modules.diffusion.conditioning import PairwiseConditioning, SingleConditioning
from ...modules.diffusion.edm import EdmPreconditioner, edm_loss_weight, lognormal_training_noise
from ...modules.diffusion.sampling import draw_normal, draw_normal_like
from ...modules.diffusion.schedule import karras_noise_schedule
from ...modules.embedding.contact import ContactConditioning
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.feed_forward import FeedForward
from ...modules.functional import pair_mask_from_token_mask, pairwise_distance, weighted_mean
from ...modules.geometry import rotate_vectors, uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import DistogramHead
from ...modules.msa import MsaPairBlock
from ...modules.pair import LineDropoutTrianglePairUpdateBlock, TrianglePairUpdateStack
from ...modules.pairformer import (
    LineDropoutPairformerBlock,
    PairformerStack,
    PostNormLineDropoutPairformerBlock,
)
from ...modules.potential import (
    chain_center_separation,
    indexed_distance,
    indexed_distance_with_gradient,
    indexed_signed_dihedral,
    indexed_signed_dihedral_with_gradient,
    linear_flat_bottom_derivative,
    linear_flat_bottom_energy,
)
from ...modules.transformer import (
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
    PostNormConditionedTransformerEncoderBlock,
)
from ...templates import Template
from ..processing import (
    gather_model_coordinates,
    project_model_categories,
    project_model_coordinates,
    project_model_features,
)
from .config import Boltz2AffinityConfig, Boltz2AffinityHeadConfig, Boltz2Config


class Boltz2Model(Model):
    r"""
    Predict biomolecular complex structures with Boltz 2.

    Structure training consumes separate [`Boltz2Input`][] and
    [`Boltz2StructureTarget`][] values. ``sample_with_confidence`` executes the inference graph and
    returns raw tensors; [`Boltz2Pipeline`][ofoldx.models.boltz2.Boltz2Pipeline]
    prepares inputs and decodes those tensors into public results.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = Boltz2Config
    config: Boltz2Config

    def __init__(self, config: Boltz2Config | None = None) -> None:
        config = Boltz2Config() if config is None else config
        super().__init__(config)
        self._num_recycles = config.num_recycles
        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config = config.input_embeddings
        trunk = config.trunk

        self.input_embeddings = Boltz2InputEmbeddings(
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
            num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
            vocab_size=input_embeddings_config.vocab_size,
            num_method_types=input_embeddings_config.num_method_types,
            num_molecule_types=input_embeddings_config.num_molecule_types,
            add_method_conditioning=input_embeddings_config.add_method_conditioning,
            add_modified_flag=input_embeddings_config.add_modified_flag,
            add_cyclic_flag=input_embeddings_config.add_cyclic_flag,
            add_mol_type_feat=input_embeddings_config.add_mol_type_feat,
            backend="auto",
        )

        # s/z initialisation. z_init is built from the single INPUT embedding (s_inputs).
        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, min_cutoff=config.min_conditioning_cutoff, max_cutoff=config.max_conditioning_cutoff
        )

        # Recycling (zero-init gates).
        self.single_recycle_norm = LayerNorm(single_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        self.template_embedder: Boltz2TemplateEmbedder | None = None
        if config.use_templates:
            self.template_embedder = Boltz2TemplateEmbedder(
                pair_size=pair_size,
                hidden_size=config.template.hidden_size,
                num_blocks=config.template.num_blocks,
                vocab_size=config.template.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_distance=config.template.min_distance,
                max_distance=config.template.max_distance,
                expansion=config.template.expansion,
                triangle_attention_head_size=config.template.triangle_attention_head_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                dropout=config.template.dropout,
                backend="auto",
            )
        self.msa_trunk = Boltz2MsaTrunk(
            single_size=single_size,
            pair_size=pair_size,
            msa_size=trunk.msa_size,
            num_blocks=trunk.num_msa_blocks,
            use_paired_feature=trunk.use_paired_feature,
            outer_product_hidden_size=trunk.outer_product_hidden_size,
            msa_dropout=trunk.msa_dropout,
            pair_dropout=trunk.msa_pair_dropout,
            num_msa_attention_heads=trunk.num_msa_attention_heads,
            msa_attention_head_size=trunk.msa_attention_head_size,
            triangle_attention_head_size=trunk.msa_triangle_attention_head_size,
            num_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
            backend="auto",
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=trunk.num_pairformer_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_head_size=trunk.triangle_attention_head_size,
            num_triangle_attention_heads=trunk.num_triangle_attention_heads,
            num_pair_attention_heads=trunk.num_pair_attention_heads,
            dropout=trunk.pair_dropout,
            block=PostNormLineDropoutPairformerBlock if trunk.post_layer_norm else LineDropoutPairformerBlock,
            backend="auto",
        )
        self.distogram_head = DistogramHead(pair_size=pair_size, num_bins=config.num_distogram_bins)

        denoiser_config = config.denoiser
        self.denoiser_conditioning = AtomDenoiserConditioning(
            atom_feature_bias=True,
            single_size=single_size,
            pair_size=pair_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            atom_pair_hidden_size=denoiser_config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
        )
        denoiser = Boltz2Denoiser(
            single_size=single_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            post_layer_norm=denoiser_config.post_layer_norm,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
            backend="auto",
        )
        sampler = config.sampler
        self.coordinate_diffusion = Boltz2CoordinateDiffusion(
            denoiser,
            sigma_data=denoiser_config.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=denoiser_config.alignment_reverse_diff,
        )

        confidence = config.confidence
        self.confidence_head = Boltz2ConfidenceHead(
            single_size=single_size,
            pair_size=pair_size,
            num_blocks=confidence.num_pairformer_blocks,
            num_pair_attention_heads=confidence.num_pair_attention_heads,
            triangle_attention_head_size=confidence.triangle_attention_head_size,
            num_triangle_attention_heads=confidence.num_triangle_attention_heads,
            num_distogram_bins=confidence.num_distogram_bins,
            max_distance=confidence.max_distance,
            relative_position_size=config.relative_position_size,
            num_bond_types=config.num_bond_types,
            min_conditioning_cutoff=config.min_conditioning_cutoff,
            max_conditioning_cutoff=config.max_conditioning_cutoff,
            num_plddt_bins=confidence.num_plddt_bins,
            num_pae_bins=confidence.num_pae_bins,
            num_pde_bins=confidence.num_pde_bins,
            pair_dropout=confidence.pair_dropout,
            post_layer_norm=confidence.post_layer_norm,
            normalize_single_input=confidence.normalize_single_input,
            backend="auto",
        )
        self.b_factor_head: nn.Linear | None = None
        if config.predict_b_factor:
            self.b_factor_head = nn.Linear(single_size, config.num_distogram_bins)
        self.post_init()

    def forward(
        self,
        input: Boltz2Input,
        target: Boltz2StructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = True,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Output:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model-visible structure and evidence features.
            target: Complete pre-crop structure truth. Its conformer axis must
                have length one or ``num_samples``.
            num_samples: Number ``S`` of independently corrupted copies per input.
            sample_chunk_size: Maximum samples per input evaluated by one
                denoiser call. ``None`` evaluates all samples together.
            synchronize_noise_levels: Whether samples from one example share sigma.
            coordinate_augmentation: Whether to apply random rigid augmentation.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            generator: Optional generator for MSA row sampling, log-normal noise
                levels, rigid augmentation, and Gaussian coordinate noise.

        Returns:
            Denoised and centered target coordinates with shape ``(B, S, A, 3)``,
                noise levels with shape ``(B, S)`` in angstroms, and
                distogram logits with shape ``(B, N, N, D)``.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                fail("value_invalid", RequestRef("sample_chunk_size", str(sample_chunk_size)))
            sample_chunk_size = min(sample_chunk_size, num_samples)
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        target_positions, target_mask = _gather_boltz2_structure_target(
            target,
            input,
            num_samples=num_samples,
        )
        conditioning = self._build_denoiser_conditioning(representations, input)
        diffusion_output = self.coordinate_diffusion.denoise_training(
            target_positions,
            input.atom_mask,
            coordinate_mask=target_mask,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            generator=generator,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        denoised_coordinates, augmented_target_coordinates, noise_level = diffusion_output
        batch_size = representations.single.shape[0]
        b_factor_logits = self.b_factor_head(representations.single) if self.b_factor_head is not None else None
        return Boltz2Output(
            coordinates=denoised_coordinates,
            distogram_logits=representations.distogram_logits,
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=noise_level,
            **({"b_factor_logits": b_factor_logits} if b_factor_logits is not None else {}),
            batch_size=[batch_size],
        )

    def confidence(
        self,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2ConfidenceOutput:
        r"""
        Return raw differentiable confidence predictions for coordinate samples.

        Args:
            input: Model-visible structure and evidence features.
            coordinates: Atom coordinates with shape ``(B, S, A, 3)``.
            num_recycles: Number of trunk recycling iterations.
            msa_max_depth: Optional MSA rows retained per MSA-module invocation.
            generator: Optional generator for MSA row sampling.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)
        coordinates = project_model_coordinates(
            coordinates,
            input.atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(coordinates[..., 0], dtype=torch.bool),
            dtype=self.single_init_projection.weight.dtype,
            feature="Boltz2 candidate coordinates",
        )

        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        return self._confidence_from_representations(
            representations,
            input,
            coordinates,
        )

    def representations(
        self,
        input: Boltz2Input,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Representations:
        r"""
        Return the final single and pair trunk representations.

        Args:
            input: Model features with token axis length ``N``.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            generator: Optional generator advanced by MSA row sampling.

        Returns:
            Single representations with shape ``(B, N, C_single)`` and pair
                representations with shape ``(B, N, N, C_pair)``. Gradients are
                retained when autograd is enabled.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)

        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        return Boltz2Representations(
            single=representations.single,
            pair=representations.pair,
            batch_size=[representations.single.shape[0]],
        )

    def sample(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        generator: torch.Generator | None = None,
    ) -> Boltz2SamplingOutput:
        r"""
        Sample atom coordinates with explicit ``(B, S, A, 3)`` axes.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        coordinates = self._sample_coords(
            representations,
            input,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            max_parallel_samples=max_parallel_samples,
            feynman_kac=feynman_kac or use_potentials,
            physical_guidance=physical_guidance or use_potentials,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            generator=generator,
        )
        return Boltz2SamplingOutput(
            coordinates=coordinates,
            batch_size=[coordinates.shape[0]],
        )

    def sample_with_confidence(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> Boltz2ConfidenceOutput:
        r"""
        Run the trunk, coordinate sampler, and confidence head.

        Args:
            input: Typed model input.
            num_samples: Number of coordinate samples.
            generator: Optional generator for MSA row sampling, initial
                coordinates, churn noise, and per-step rigid augmentations.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            max_parallel_samples: Maximum batch-by-sample candidates evaluated by one denoiser call.
            use_potentials: Enable the released Feynman--Kac and physical-guidance
                preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            contact_guidance: Whether to apply forced contact/template gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle
                resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            Raw sampled coordinates, distogram logits, confidence logits, and
                tensor-valued derived confidence predictions.

        Raises:
            InvalidRequestError: ``value_invalid`` if ``num_samples`` is not positive.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        sample_coordinates = self._sample_coords(
            representations,
            input,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            max_parallel_samples=max_parallel_samples,
            feynman_kac=feynman_kac or use_potentials,
            physical_guidance=physical_guidance or use_potentials,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            generator=generator,
        )
        return self._confidence_from_representations(
            representations,
            input,
            sample_coordinates,
        )

    def _confidence_from_representations(
        self,
        representations: _Boltz2TrunkOutput,
        input: Boltz2Input,
        coordinates: Tensor,
    ) -> Boltz2ConfidenceOutput:
        batch_size = coordinates.shape[0]
        confidence = self.confidence_head(
            representations.single_inputs,
            representations.single,
            representations.pair,
            coordinates,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            token_bonds=input.token_bonds,
            type_bonds=input.type_bonds,
            contact_conditioning=input.contact_conditioning,
            contact_threshold=input.contact_threshold,
            token_representative_atom_index=input.token_representative_atom_index,
            token_mask=input.token_mask,
            cyclic_period=input.cyclic_period,
        )
        plddt_logits = confidence["plddt_logits"]
        pae_logits = confidence["pae_logits"]
        pde_logits = confidence["pde_logits"]
        resolved_logits = confidence["resolved_logits"]
        values = _boltz2_confidence_values(
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            coordinates=coordinates,
            distogram_logits=representations.distogram_logits,
            input=input,
            backend=self.backend,
        )
        b_factor_logits = self.b_factor_head(representations.single) if self.b_factor_head is not None else None
        return Boltz2ConfidenceOutput(
            coordinates=coordinates,
            distogram_logits=representations.distogram_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            **({"b_factor_logits": b_factor_logits} if b_factor_logits is not None else {}),
            **values,
            batch_size=[batch_size],
        )

    def _run_trunk(
        self,
        input: Boltz2Input,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _Boltz2TrunkOutput:
        r"""
        Run the recycled folding trunk (template + MSA + Pairformer) and emit the trunk representations.

        Args:
            input: Typed model input.
            num_recycles: Total number of trunk passes; defaults to the configured value. The trunk runs
                ``num_recycles`` times.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            generator: Optional generator advanced by each MSA row sample.

        Returns:
            Trunk representations and distogram logits.

        """

        num_recycles = operator.index(self._num_recycles if num_recycles is None else num_recycles)
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(input)
        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        # Boltz runs num_recycles trunk passes.
        for cycle in range(num_recycles):
            grad_enabled = torch.is_grad_enabled() and cycle == num_recycles - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if input.template_residue_type_features is not None:
                    pair = pair + cast(Boltz2TemplateEmbedder, self.template_embedder)(
                        pair,
                        residue_type_features=input.template_residue_type_features,
                        frame_rotations=cast(Tensor, input.template_frame_rotations),
                        frame_translations=cast(Tensor, input.template_frame_translations),
                        frame_mask=cast(Tensor, input.template_frame_mask),
                        pseudo_beta_positions=cast(
                            Tensor,
                            input.template_pseudo_beta_positions,
                        ),
                        alpha_carbon_positions=cast(
                            Tensor,
                            input.template_alpha_carbon_positions,
                        ),
                        pseudo_beta_mask=cast(Tensor, input.template_pseudo_beta_mask),
                        visibility_index=cast(Tensor, input.template_visibility_index),
                        template_mask=cast(Tensor, input.template_mask),
                        pair_mask=pair_mask,
                    )
                if input.msa_tokens is not None:
                    # Boltz adds the MSA module's (input-inclusive) refined pair as a residual.
                    pair = pair + self.msa_trunk(
                        pair,
                        single_inputs,
                        msa_tokens=input.msa_tokens,
                        has_deletion=cast(Tensor, input.has_deletion),
                        deletion_value=cast(Tensor, input.deletion_value),
                        is_paired=cast(Tensor, input.msa_paired),
                        msa_mask=cast(Tensor, input.msa_mask),
                        token_mask=token_mask,
                        msa_max_depth=msa_max_depth,
                        generator=generator,
                    )
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        return _Boltz2TrunkOutput(
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
        )

    def _initial_representations(self, input: Boltz2Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build the recycling-invariant input representations from the batch.

        Args:
            input: Typed model input.

        Returns:
            Tuple ``(single_inputs, single_init, pair_init, relative_position_encoding)`` where ``single_inputs``
            and ``single_init`` are ``(B, N, single_size)``, and ``pair_init`` and
            ``relative_position_encoding`` are ``(B, N, N, pair_size)``.
        """

        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )

        single_init = self.single_init_projection(single_inputs)
        pair_init = self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair_init = pair_init + self.pair_right_projection(single_inputs).unsqueeze(-3)

        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            cyclic_period=input.cyclic_period,
            dtype=single_init.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(input.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            input.contact_conditioning.to(pair_init.dtype), input.contact_threshold.to(pair_init.dtype)
        )
        return single_inputs, single_init, pair_init, relative_position_encoding

    def _sample_coords(
        self,
        representations: _Boltz2TrunkOutput,
        input: Boltz2Input,
        *,
        num_sampling_steps: int | None,
        step_scale: float | None,
        num_samples: int,
        max_parallel_samples: int | None,
        feynman_kac: bool,
        physical_guidance: bool,
        contact_guidance: bool,
        num_particles: int,
        feynman_kac_lambda: float,
        feynman_kac_resampling_interval: int,
        num_guidance_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Diffusion-sample atom coords from the trunk representations.

        Args:
            representations: Trunk representations from
                [`_run_trunk`][ofoldx.models.boltz2.model.Boltz2Model._run_trunk].
            input: Typed model input.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            num_samples: Number of independent samples to draw per batch element.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Sampled atom coordinates with explicit ``(B, num_samples, M, 3)`` axes.
        """

        conditioning = self._build_denoiser_conditioning(representations, input)
        return self.coordinate_diffusion._sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            max_parallel_samples=max_parallel_samples,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            generator=generator,
            input=input,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )

    def _build_denoiser_conditioning(
        self, representations: _Boltz2TrunkOutput, input: Boltz2Input
    ) -> AtomDenoiserContext:
        r"""
        Precompute the per-call-invariant diffusion conditioning from the trunk representations.

        Shared by the training denoising step ([`forward`][ofoldx.models.boltz2.model.Boltz2Model.forward]) and
        the reverse-diffusion sampler ([`_sample_coords`][ofoldx.models.boltz2.model.Boltz2Model._sample_coords]).

        Args:
            representations: Trunk representations from
                [`_run_trunk`][ofoldx.models.boltz2.model.Boltz2Model._run_trunk].
            input: Typed model input.

        Returns:
            The atom states, per-block attention biases, and local window layout of
            [`AtomDenoiserConditioning`][ofoldx.modules.denoiser.AtomDenoiserConditioning].
        """

        return self.denoiser_conditioning(
            representations.single,
            representations.pair,
            representations.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )


class Boltz2AffinityModel(Boltz2Model):
    r"""
    Predict protein-ligand binding affinity with Boltz 2.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = Boltz2AffinityConfig
    config: Boltz2AffinityConfig

    def __init__(self, config: Boltz2AffinityConfig | None = None) -> None:
        config = Boltz2AffinityConfig() if config is None else config
        super().__init__(config)
        self.affinity_heads = nn.ModuleList(self._build_affinity_head(head) for head in config.affinity_heads)
        self.post_init(backend=self.backend)

    def affinity(
        self,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityOutput:
        r"""
        Return raw differentiable affinity predictions for coordinate samples.

        Args:
            input: Affinity Processor output.
            coordinates: Candidate coordinates with shape ``(B, S, A, 3)``.
            num_recycles: Number of trunk recycling iterations.
            msa_max_depth: Optional MSA rows retained per MSA-module invocation.
            molecular_weight_correction: Whether to calibrate the ensemble value.
            generator: Optional generator for MSA row sampling.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)
        coordinates = project_model_coordinates(
            coordinates,
            input.atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(coordinates[..., 0], dtype=torch.bool),
            dtype=self.single_init_projection.weight.dtype,
            feature="Boltz2 candidate coordinates",
        )

        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        values = self._score_affinity(
            representations,
            input,
            coordinates,
            molecular_weight_correction=molecular_weight_correction,
        )
        return _affinity_output(values)

    @torch.inference_mode()
    def score(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 5,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
    ) -> Boltz2AffinityScoreOutput:
        r"""
        Score protein-ligand affinity for one prepared complex.

        Candidate structures are sampled and ranked by iPTM before the affinity
        heads score the highest-ranked sample.

        Args:
            input: Affinity Processor output for one complex and one selected
                ligand, with batch size ``B = 1``.
            num_samples: Number of candidate coordinate samples.
            generator: Optional generator for MSA row sampling and coordinate
                diffusion.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            molecular_weight_correction: Whether to apply molecular-weight
                calibration to the ensemble affinity value.

        Returns:
            Raw structure-confidence samples, selected sample index, and affinity
                tensors. Pipeline decoding assigns public units and symbolic scope.

        Raises:
            InvalidRequestError: ``value_invalid`` if a count is not positive, the input is a batch, or no affinity
                ligand is identified.
        """
        input = normalize_boltz2_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples <= 0:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if num_recycles is not None:
            num_recycles = operator.index(num_recycles)
            if num_recycles < 1:
                fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        if num_sampling_steps is not None:
            num_sampling_steps = operator.index(num_sampling_steps)
            if num_sampling_steps < 2:
                fail("value_invalid", RequestRef("num_sampling_steps", str(num_sampling_steps)))
        if input.token_mask.shape[0] != 1:
            fail("value_invalid", RequestRef("input", f"batch of {input.token_mask.shape[0]}"))
        if input.affinity_token_mask is None:
            fail("value_invalid", RequestRef("affinity_token_mask", "None"))

        return self._score(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )

    def _build_affinity_head(self, affinity: Boltz2AffinityHeadConfig) -> Boltz2AffinityHead:
        r"""
        Instantiate one affinity head from an affinity-head sub-config.

        Args:
            affinity: An affinity-head sub-config (block count, distogram bins, triangle widths, etc.).

        Returns:
            A configured [`Boltz2AffinityHead`][ofoldx.models.boltz2.model.Boltz2AffinityHead].
        """

        return Boltz2AffinityHead(
            single_size=self.config.single_size,
            pair_size=self.config.pair_size,
            num_blocks=affinity.num_pair_update_blocks,
            num_distogram_bins=affinity.num_distogram_bins,
            max_distance=affinity.max_distance,
            num_conditioning_transitions=affinity.num_conditioning_transitions,
            expansion=affinity.expansion,
            triangle_attention_head_size=affinity.triangle_attention_head_size,
            num_triangle_attention_heads=affinity.num_triangle_attention_heads,
            dropout=affinity.dropout,
            backend="auto",
        )

    @torch.inference_mode()
    def _score(
        self,
        input: Boltz2Input,
        *,
        coordinates: Tensor | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityScoreOutput:
        r"""
        Raw affinity protocol, optionally using caller-supplied candidate coords.

        Args:
            input: Typed Boltz-2 input. ``affinity_token_mask`` may select the ligand; otherwise ligand tokens are
                inferred from ``molecule_type == 3``.
            coordinates: Optional atom coordinates to score with shape ``(B, S, M, 3)``.
            num_recycles: Optional trunk recycling passes. Defaults to the config value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            num_samples: Number of coordinate sets to sample when ``coordinates`` is absent.
            num_sampling_steps: Optional EDM sampling steps for internally sampled coords.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            molecular_weight_correction: Whether to apply the ligand-molecular-weight calibration.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            All sampled structures and the affinity outputs for the best-iPTM sample.

        """

        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        batch_size = representations.single.shape[0]
        if coordinates is None:
            coordinates = self._sample_coords(
                representations,
                input,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                num_samples=num_samples,
                max_parallel_samples=1,
                feynman_kac=False,
                physical_guidance=False,
                contact_guidance=False,
                num_particles=3,
                feynman_kac_lambda=4.0,
                feynman_kac_resampling_interval=3,
                num_guidance_steps=20,
                generator=generator,
            )
        structure_prediction = self._confidence_from_representations(
            representations,
            input,
            coordinates,
        )
        iptm = structure_prediction.iptm
        selected_sample_index = iptm.argmax(dim=1)
        selected_coords = coordinates[torch.arange(batch_size, device=coordinates.device), selected_sample_index]
        affinity = _affinity_output(
            self._score_affinity(
                representations,
                input,
                selected_coords.unsqueeze(1),
                molecular_weight_correction=molecular_weight_correction,
            )
        )
        return Boltz2AffinityScoreOutput(
            structure=structure_prediction,
            affinity=affinity,
            selected_sample_index=selected_sample_index,
            batch_size=[batch_size],
        )

    def _score_affinity(
        self,
        representations: _Boltz2TrunkOutput,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        affinity_token_mask: Tensor | None = None,
        affinity_mw: Tensor | None = None,
        molecular_weight_correction: bool = False,
    ) -> dict[str, Tensor]:
        r"""
        Score coords using precomputed Boltz-2 trunk representations.

        Rebuilds the affinity-specific input single, pre-masks the trunk pair to
        receptor/ligand and ligand/ligand pairs, evaluates both heads,
        and optionally applies the molecular-weight calibration.

        Args:
            representations: Trunk representations from
                [`_run_trunk`][ofoldx.models.boltz2.model.Boltz2Model._run_trunk].
            input: Typed Boltz-2 input.
            coordinates: Atom coordinates to score with shape ``(B, S, M, 3)``.
            affinity_token_mask: Optional explicit ligand token mask ``(B, N)``.
            affinity_mw: Optional per-batch molecular weight used by the molecular-weight correction.
            molecular_weight_correction: Whether to apply the ligand-molecular-weight calibration.

        Returns:
            Dict of affinity score tensors keyed by name, each with an explicit sample axis.
        """

        batch_size, num_samples = coordinates.shape[:2]
        coords = coordinates.flatten(0, 1)
        if affinity_token_mask is None:
            affinity_token_mask = input.affinity_token_mask
        if affinity_token_mask is None:
            fail("value_invalid", RequestRef("affinity_token_mask", "None"))
        if input.affinity_msa_profile is None or input.affinity_deletion_mean is None:
            fail("value_invalid", RequestRef("affinity_msa_profile", "None"))
        ligand_mask = affinity_token_mask.bool() & input.token_mask.bool()
        receptor_mask = input.molecule_type == 0
        receptor_mask = receptor_mask & input.token_mask.bool()
        affinity_pair_mask = (
            ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
            | receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
            | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        )
        pair = representations.pair * affinity_pair_mask.unsqueeze(-1).to(representations.pair)
        affinity_single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.affinity_msa_profile,
            deletion_mean=input.affinity_deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        head_kwargs = {
            "single_inputs": affinity_single_inputs,
            "pair": pair,
            "coordinates": coords,
            "token_representative_atom_index": input.token_representative_atom_index,
            "token_mask": input.token_mask,
            "ligand_mask": ligand_mask,
            "receptor_mask": receptor_mask,
            "num_samples": num_samples,
        }
        flat_scores1, flat_scores2 = (head(**head_kwargs) for head in self.affinity_heads)
        scores1 = {
            name: _unflatten_sample_axis(tensor, batch_size, num_samples) for name, tensor in flat_scores1.items()
        }
        scores2 = {
            name: _unflatten_sample_axis(tensor, batch_size, num_samples) for name, tensor in flat_scores2.items()
        }
        probability1 = torch.sigmoid(scores1["affinity_logits_binary"])
        probability2 = torch.sigmoid(scores2["affinity_logits_binary"])
        scores = {
            "affinity_pred_value": (scores1["affinity_pred_value"] + scores2["affinity_pred_value"]) / 2,
            "affinity_probability_binary": (probability1 + probability2) / 2,
            "affinity_pred_value1": scores1["affinity_pred_value"],
            "affinity_logits_binary1": scores1["affinity_logits_binary"],
            "affinity_probability_binary1": probability1,
            "affinity_pred_value2": scores2["affinity_pred_value"],
            "affinity_logits_binary2": scores2["affinity_logits_binary"],
            "affinity_probability_binary2": probability2,
        }
        if molecular_weight_correction:
            affinity_mw = input.affinity_mw if affinity_mw is None else affinity_mw
            if affinity_mw is None:  # the ligand's graph is undetermined, so its weight is not a fact
                fail("rdkit_incomplete", StructureRef(item="affinity ligand molecular weight"))
            affinity_pred_value = scores["affinity_pred_value"]
            mw = affinity_mw.reshape(-1)
            scores["affinity_pred_value"] = (
                1.03525938 * affinity_pred_value - 0.59992683 * mw.pow(0.3)[:, None, None] + 2.83288489
            )
        return scores


class Boltz2InputEmbeddings(nn.Module):
    r"""
    Compose atom-derived token states with token-level encodings and conditioning embeddings.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 388,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        vocab_size: int = len(TOKENS),
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_mol_type_feat: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.atom_encoder = AtomToTokenEncoder(
            atom_hidden_size,
            single_size,
            num_atom_encoder_blocks,
            num_atom_encoder_attention_heads,
            backend,
            block=ConditionedTransformerEncoderBlock,
        )

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(len(TOKENS) + 1, single_size, bias=False)
        self.method_embedding: nn.Embedding | None = None
        self.modified_embedding: nn.Embedding | None = None
        self.cyclic_projection: nn.Linear | None = None
        self.molecule_type_embedding: nn.Embedding | None = None
        if add_method_conditioning:
            self.method_embedding = nn.Embedding(num_method_types, single_size)
        if add_modified_flag:
            self.modified_embedding = nn.Embedding(2, single_size)
        if add_cyclic_flag:
            self.cyclic_projection = nn.Linear(1, single_size, bias=False)
        if add_mol_type_feat:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        for module in (self.method_embedding, self.modified_embedding, self.molecule_type_embedding):
            if module is not None:
                nn.init.zeros_(module.weight)
        if self.cyclic_projection is not None:
            nn.init.zeros_(self.cyclic_projection.weight)

    def forward(
        self,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        residue_type_features: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        method_feature: Tensor | None = None,
        modified: Tensor | None = None,
        cyclic_period: Tensor | None = None,
        molecule_type: Tensor | None = None,
    ) -> Tensor:
        feature_dtype = self.atom_feature_projection.weight.dtype
        reference_positions = project_model_coordinates(
            reference_positions,
            atom_mask.to(torch.bool),
            dtype=feature_dtype,
            feature="Boltz2 embedding references",
        )
        atom_features = project_model_features(
            atom_features,
            atom_mask.to(torch.bool)[..., None],
            dtype=feature_dtype,
            feature="Boltz2 embedding atom features",
        )
        residue_type_features = project_model_features(
            residue_type_features,
            torch.ones_like(residue_type_features, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz2 complete embedding residue features",
        )
        msa_profile = project_model_features(
            msa_profile,
            torch.ones_like(msa_profile, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz2 complete embedding MSA profile",
        )
        deletion_mean = project_model_features(
            deletion_mean,
            torch.ones_like(deletion_mean, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz2 complete embedding deletion features",
        )
        geometry = self.windowing(reference_positions, reference_space_id, atom_mask)
        atom_single = self.atom_feature_projection(atom_features)
        token_from_atom = self.atom_encoder(
            atom_single,
            self.atom_pair_bias_projection(self.atom_pair_encoder(atom_single, geometry)),
            atom_mask,
            atom_to_token,
            residue_type_features,
            geometry.window_info,
        )

        single = token_from_atom
        single = single + self.residue_type_projection(residue_type_features.to(single.dtype))
        msa_features = torch.cat([msa_profile, deletion_mean.unsqueeze(-1)], dim=-1)
        single = single + self.msa_profile_projection(msa_features.to(single.dtype))

        if self.method_embedding is not None:
            single = single + self.method_embedding(cast(Tensor, method_feature))
        if self.modified_embedding is not None:
            single = single + self.modified_embedding(cast(Tensor, modified))
        if self.cyclic_projection is not None:
            single = single + self.cyclic_projection(
                cast(Tensor, cyclic_period).clamp(max=1.0).unsqueeze(-1).to(single.dtype)
            )
        if self.molecule_type_embedding is not None:
            single = single + self.molecule_type_embedding(cast(Tensor, molecule_type))
        return single


class Boltz2TemplateEmbedder(nn.Module):
    r"""
    Boltz-2 template embedder with on-the-fly distogram / unit-vector featurization.

    Directions are normalized over their three Cartesian components with a
    ``1e-6`` norm floor. Coincident positions contribute a zero direction.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = len(TOKENS),
        num_bins: int = 38,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        expansion: int = 4,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.template_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=hidden_size,
            bias=False,
            backend=backend,
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.pair_update_stack = TrianglePairUpdateStack(
            block=LineDropoutTrianglePairUpdateBlock,
            num_blocks=num_blocks,
            pair_size=hidden_size,
            triangle_multiplication_hidden_size=hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def _featurize(
        self,
        residue_type_features: Tensor,
        frame_rotations: Tensor,
        frame_translations: Tensor,
        frame_mask: Tensor,
        pseudo_beta_positions: Tensor,
        alpha_carbon_positions: Tensor,
        pseudo_beta_mask: Tensor,
        visibility_index: Tensor,
    ) -> Tensor:
        pair_pseudo_beta_mask = (pseudo_beta_mask[..., :, None] * pseudo_beta_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility_index[..., :, None] == visibility_index[..., None, :]).to(
            residue_type_features.dtype
        )

        pseudo_beta_distances = pairwise_distance(pseudo_beta_positions)
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=pseudo_beta_distances.device,
        )
        distogram = (pseudo_beta_distances[..., None] > boundaries).sum(dim=-1).long()
        distogram = F.one_hot(distogram, num_classes=self.num_bins).to(residue_type_features.dtype)

        translation = frame_translations.unsqueeze(-2)
        targets = alpha_carbon_positions.unsqueeze(-3)
        vector = rotate_vectors(targets - translation, frame_rotations).transpose(-3, -2)
        unit_vector = F.normalize(vector, dim=-1, eps=1e-6)

        geometry = torch.cat(
            [distogram, pair_pseudo_beta_mask, unit_vector, pair_frame_mask],
            dim=-1,
        )
        geometry = geometry * same_visibility.unsqueeze(-1)
        return geometry

    def forward(
        self,
        pair: Tensor,
        *,
        residue_type_features: Tensor,
        frame_rotations: Tensor,
        frame_translations: Tensor,
        frame_mask: Tensor,
        pseudo_beta_positions: Tensor,
        alpha_carbon_positions: Tensor,
        pseudo_beta_mask: Tensor,
        visibility_index: Tensor,
        template_mask: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        present = template_mask.any(dim=-1).to(pair.dtype)
        # as upstream, every templated token keeps its rotation (identity without a frame), origin and CA
        templated = template_mask.to(torch.bool)
        pseudo_valid = pseudo_beta_mask.to(torch.bool) & templated
        frame_rotations = project_model_features(
            frame_rotations, templated[..., None, None], dtype=pair.dtype, feature="Boltz2 template rotations"
        )
        frame_translations = project_model_coordinates(
            frame_translations, templated, dtype=pair.dtype, feature="Boltz2 template frame origins"
        )
        alpha_carbon_positions = project_model_coordinates(
            alpha_carbon_positions, templated, dtype=pair.dtype, feature="Boltz2 template alpha carbons"
        )
        pseudo_beta_positions = project_model_coordinates(
            pseudo_beta_positions, pseudo_valid, dtype=pair.dtype, feature="Boltz2 template pseudo-beta positions"
        )
        template_count = present.sum(dim=-1).clamp(min=1)

        geometry = self._featurize(
            residue_type_features,
            frame_rotations,
            frame_translations,
            frame_mask,
            pseudo_beta_positions,
            alpha_carbon_positions,
            pseudo_beta_mask,
            visibility_index,
        )
        projected = self.template_projection(geometry, residue_type_features.to(pair.dtype))
        template_pair = self.pair_projection(self.pair_norm(pair)).unsqueeze(-4) + projected
        template_pair_mask = None if pair_mask is None else pair_mask.unsqueeze(-3)
        stacked = self.pair_update_stack(template_pair, pair_mask=template_pair_mask)
        template_pair = self.hidden_norm(template_pair + stacked)

        weight = present[..., None, None, None]
        update = (template_pair * weight).sum(dim=1) / template_count[:, None, None, None].to(pair.dtype)
        return self.output_projection(self.activation(update))


class Boltz2MsaTrunk(nn.Module):
    r"""
    Build Boltz-2 MSA features and refine the pair representation.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 4,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.use_paired_feature = use_paired_feature
        self.msa_projection = nn.Linear(
            len(TOKENS) + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            MsaPairBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_attention_head_size=msa_attention_head_size,
                expansion=expansion,
                backend=backend,
                pair_block=partial(
                    LineDropoutTrianglePairUpdateBlock,
                    triangle_attention_head_size=triangle_attention_head_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                ),
            )
            for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        *,
        msa_tokens: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        msa_mask = msa_mask.bool() & token_mask.bool().unsqueeze(-2)
        msa_tokens = project_model_categories(
            msa_tokens, msa_mask, num_classes=len(TOKENS), feature="Boltz2 MSA tokens"
        )
        has_deletion = project_model_features(
            torch.where(msa_mask, has_deletion, 0).to(torch.float64),
            msa_mask,
            dtype=pair.dtype,
            feature="Boltz2 deletion presence",
        )
        deletion_value = project_model_features(
            deletion_value, msa_mask, dtype=pair.dtype, feature="Boltz2 deletion feature"
        )
        if self.use_paired_feature:
            is_paired = project_model_features(
                torch.where(msa_mask, is_paired, 0).to(torch.float64),
                msa_mask,
                dtype=pair.dtype,
                feature="Boltz2 MSA pairing",
            )
        if msa_max_depth is not None:
            msa_max_depth = operator.index(msa_max_depth)
            if msa_max_depth < 1:
                fail("value_invalid", RequestRef("msa_max_depth", str(msa_max_depth)))
            random_device = msa_tokens.device if generator is None else generator.device
            indices = torch.randperm(
                msa_tokens.shape[1],
                device=random_device,
                generator=generator,
            )[
                :msa_max_depth
            ].to(msa_tokens.device)
            msa_tokens = msa_tokens[:, indices]
            has_deletion = has_deletion[:, indices]
            deletion_value = deletion_value[:, indices]
            is_paired = is_paired[:, indices]
            msa_mask = msa_mask[:, indices]
        features = [
            F.one_hot(msa_tokens, num_classes=len(TOKENS)).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                msa, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    msa,
                    pair,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )
            else:
                msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return pair


class Boltz2Denoiser(nn.Module):
    r"""
    EDM score network: ``(noisy positions, noise level, trunk, conditioning) -> position update``.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        post_layer_norm: bool = False,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = SingleConditioning(
            hidden,
            hidden,
            fourier_embedding_size=fourier_embedding_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            atom_hidden_size,
            hidden,
            num_atom_encoder_blocks,
            num_atom_encoder_attention_heads,
            backend,
            block=(
                PostNormConditionedTransformerEncoderBlock if post_layer_norm else ConditionedTransformerEncoderBlock
            ),
        )
        self.single_to_token_norm = LayerNorm(hidden, backend=backend)
        self.single_to_token_projection = nn.Linear(hidden, hidden, bias=False)
        self.token_encoder = ConditionedTransformerEncoder(
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            backend=backend,
            block=ConditionedTransformerEncoderBlock,
        )
        self.token_norm = LayerNorm(hidden, backend=backend)
        self.atom_decoder = TokenToAtomDecoder(
            hidden,
            atom_hidden_size,
            num_atom_decoder_blocks,
            num_atom_decoder_attention_heads,
            backend,
            block=ConditionedTransformerEncoderBlock,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: AtomDenoiserContext,
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        noisy_positions = project_model_coordinates(
            noisy_positions,
            atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(noisy_positions[..., 0], dtype=torch.bool),
            dtype=self.atom_encoder.noisy_position_projection.weight.dtype,
            feature="Boltz2 noisy positions",
        )
        single = self.single_conditioner(noise_level, single_trunk, single_inputs)
        token_repr, atom_query = self.atom_encoder(
            noisy_positions,
            conditioning.atom_query,
            conditioning.atom_conditioning,
            conditioning.atom_encoder_bias,
            atom_mask,
            atom_to_token,
            single_trunk,
            conditioning.window_info,
        )
        token_repr = token_repr + self.single_to_token_projection(self.single_to_token_norm(single))

        token_repr = self.token_encoder(
            token_repr,
            single,
            conditioning.token_encoder_bias,
            token_mask,
        )
        token_repr = self.token_norm(token_repr)

        return self.atom_decoder(
            token_repr,
            atom_query,
            conditioning.atom_conditioning,
            conditioning.atom_decoder_bias,
            atom_mask,
            atom_to_token,
            conditioning.window_info,
        )


class Boltz2CoordinateDiffusion(nn.Module):
    r"""
    EDM sampler around a [`Boltz2Denoiser`][ofoldx.models.boltz2.model.Boltz2Denoiser].
    """

    def __init__(
        self,
        denoiser: Boltz2Denoiser,
        sigma_data: float = 16.0,
        min_noise_level: float = 1e-4,
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
        self.backend = "auto"
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.denoiser = denoiser
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho = rho
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        # EDM log-normal training noise distribution (Karras et al.); used by the training denoise step only.
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        log_mean: float | None = None,
        log_std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        log_mean = self.noise_log_mean if log_mean is None else log_mean
        log_std = self.noise_log_std if log_std is None else log_std
        return lognormal_training_noise(
            batch_size,
            sigma_data=self.sigma_data,
            log_mean=log_mean,
            log_std=log_std,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return edm_loss_weight(sigma, self.sigma_data)

    def denoise_training(
        self,
        target_coordinates: Tensor,
        atom_mask: Tensor,
        *,
        coordinate_mask: Tensor | None = None,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = True,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sample_chunk_size = num_samples if sample_chunk_size is None else min(sample_chunk_size, num_samples)
        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        batch_size = atom_mask.shape[0]
        observed = (
            atom_mask.to(torch.bool)
            if coordinate_mask is None
            else coordinate_mask.to(torch.bool) & atom_mask.to(torch.bool)
        )
        target = project_model_coordinates(
            target_coordinates,
            insert_sample_axis(observed, num_samples, dim=-2),
            dtype=dtype,
            feature="Boltz2 training target",
        ).to(device=device)
        if target.shape[-3] != num_samples:
            fail("value_invalid", RequestRef("target", str(tuple(target.shape))))
        # Target placement uses observed points; model-valid prediction slots
        # remain available to the denoiser independently of truth availability.
        augmentation_mask = observed.to(device=device)
        noise_batch_size = batch_size if synchronize_noise_levels else batch_size * num_samples
        noise_level = self.noise_distribution(
            noise_batch_size,
            device=device,
            dtype=dtype,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.unsqueeze(-1).expand(-1, num_samples)
        else:
            noise_level = noise_level.reshape(batch_size, num_samples)
        augmented_target_coordinates = self._augment_target(
            target,
            augmentation_mask,
            coordinate_augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise = draw_normal_like(augmented_target_coordinates, generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * noise
        denoised_chunks: list[Tensor] = []
        for sample_start in range(0, num_samples, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, num_samples)
            num_chunk_samples = sample_end - sample_start
            denoised_chunks.append(
                self.preconditioned_network_forward(
                    torch.narrow(noisy_coordinates, -3, sample_start, num_chunk_samples),
                    torch.narrow(noise_level, -1, sample_start, num_chunk_samples),
                    atom_mask=atom_mask,
                    **conditioning,
                )
            )
        denoised = denoised_chunks[0] if len(denoised_chunks) == 1 else torch.cat(denoised_chunks, dim=-3)
        return denoised, augmented_target_coordinates, noise_level

    def _augment_target(
        self,
        coordinates: Tensor,
        mask: Tensor,
        *,
        coordinate_augmentation: bool,
        translation_scale: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        observed = mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(coordinates[..., 0], dtype=torch.bool)
        coordinates = project_model_coordinates(coordinates, observed, feature="Boltz2 augmentation target")
        centroid = weighted_mean(coordinates.transpose(-1, -2), observed.unsqueeze(-2)).unsqueeze(-2)
        centroid = torch.where(observed.any(-1, keepdim=True).unsqueeze(-1), centroid, 0)
        coordinates = project_model_coordinates(coordinates - centroid, observed, feature="Boltz2 centered target")
        if not coordinate_augmentation:
            return coordinates
        random_device = coordinates.device if generator is None else generator.device
        sample_shape = tuple(coordinates.shape[:-2])
        num_augmentations = math.prod(sample_shape)
        rotation = (
            uniform_random_rotation(
                num_augmentations,
                device=random_device,
                dtype=coordinates.dtype,
                generator=generator,
            )
            .to(coordinates.device)
            .reshape(*sample_shape, 3, 3)
        )
        translation = draw_normal(
            (num_augmentations, 1, 3),
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        ).reshape(*sample_shape, 1, 3)
        return project_model_coordinates(
            rotate_vectors(coordinates, rotation) + translation_scale * translation,
            observed,
            feature="Boltz2 augmented target",
        )

    def preconditioned_network_forward(
        self, noisy_coordinates: Tensor, sigma: Tensor | float, **conditioning: object
    ) -> Tensor:
        atom_mask = cast(Tensor, conditioning["atom_mask"]).to(torch.bool)
        noisy_coordinates = project_model_coordinates(
            noisy_coordinates,
            atom_mask.unsqueeze(-2) & torch.ones_like(noisy_coordinates[..., 0], dtype=torch.bool),
            dtype=cast(Tensor, conditioning["single_trunk"]).dtype,
            feature="Boltz2 noisy coordinates",
        )
        sample_shape = tuple(noisy_coordinates.shape[:-2])
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device).expand(*sample_shape)
        update = self.denoiser(
            noisy_positions=self.preconditioner.scale_noisy_coords(noisy_coordinates, sigma),
            noise_level=self.preconditioner.c_noise(sigma),
            **conditioning,
        )
        return self.preconditioner.combine_update(noisy_coordinates, sigma, update)

    def sample_schedule(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return karras_noise_schedule(
            operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps),
            max_noise_level=self.max_noise_level,
            min_noise_level=self.min_noise_level,
            rho=self.rho,
            sigma_data=self.sigma_data,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        num_samples: int = 1,
        max_parallel_samples: int | None = None,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> Tensor:
        input = cast(Boltz2Input, conditioning["input"])
        template_contact = input.template_force is not None and bool(input.template_force.any())
        contact_guidance = contact_guidance and (input.contact_thresholds.numel() > 0 or template_contact)
        if not (feynman_kac or physical_guidance or contact_guidance):
            conditioning.pop("input")
            return self._sample_ordinary(
                atom_mask,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                num_samples=num_samples,
                max_parallel_samples=max_parallel_samples,
                generator=generator,
                **conditioning,
            )
        if isinstance(atom_mask, NestedTensor):
            fail("model_capacity", RequestRef("atom_mask", "packed sampling with potentials"))
        coordinates = self._sample_with_potentials(
            atom_mask,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            max_parallel_samples=max_parallel_samples,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            generator=generator,
            **conditioning,
        )
        return coordinates.reshape(atom_mask.shape[0], num_samples, *coordinates.shape[1:])

    def _sample_ordinary(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None,
        step_scale: float | None,
        num_samples: int,
        max_parallel_samples: int | None,
        generator: torch.Generator | None,
        **conditioning: object,
    ) -> Tensor:
        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        batch_size = atom_mask.shape[0]
        if max_parallel_samples is None:
            sample_chunk_size = num_samples
        else:
            max_parallel_samples = operator.index(max_parallel_samples)
            if max_parallel_samples < batch_size:
                fail("value_invalid", RequestRef("max_parallel_samples", str(max_parallel_samples)))
            sample_chunk_size = min(num_samples, max_parallel_samples // batch_size)

        device = atom_mask.device
        dtype = cast(Tensor, conditioning["single_trunk"]).dtype
        step_scale = self.step_scale if step_scale is None else step_scale
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_hats = sigmas[:-1] * (1 + gammas[1:])
        noise_variances = self.noise_scale**2 * (sigma_hats**2 - sigmas[:-1] ** 2)

        coordinate_template = insert_sample_axis(atom_mask.to(dtype).unsqueeze(-1), num_samples, dim=-3)
        coordinate_template = coordinate_template.expand(*([-1] * (coordinate_template.dim() - 1)), 3)

        coords = sigmas[0] * draw_normal_like(coordinate_template, generator)
        sample_atom_mask = insert_sample_axis(atom_mask, num_samples, dim=-2)
        for index in range(len(sigmas) - 1):
            sample_shape = tuple(coords.shape[:-2])
            num_augmentations = math.prod(sample_shape)
            random_device = device if generator is None else generator.device
            rotation = (
                uniform_random_rotation(
                    num_augmentations,
                    device=random_device,
                    dtype=coords.dtype,
                    generator=generator,
                )
                .to(device)
                .reshape(*sample_shape, 3, 3)
            )
            translation = draw_normal(
                (num_augmentations, 1, 3),
                device=device,
                dtype=dtype,
                generator=generator,
            ).reshape(*sample_shape, 1, 3)
            coords = rotate_vectors(coords - coords.mean(-2, keepdim=True), rotation) + translation

            sigma_hat = sigma_hats[index]
            noise_var = noise_variances[index]
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)) * draw_normal_like(
                coordinate_template, generator
            )
            noise_level = (
                torch.ones(
                    (coords.shape[0], num_samples),
                    device=device,
                    dtype=dtype,
                )
                * sigma_hat
            )
            chunks = []
            for sample_start in range(0, num_samples, sample_chunk_size):
                num_chunk_samples = min(sample_chunk_size, num_samples - sample_start)
                chunks.append(
                    self.preconditioned_network_forward(
                        torch.narrow(noisy, -3, sample_start, num_chunk_samples),
                        torch.narrow(noise_level, -1, sample_start, num_chunk_samples),
                        atom_mask=atom_mask,
                        **conditioning,
                    )
                )
            denoised = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-3)

            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy, target=denoised, weight=sample_atom_mask.to(noisy), backend=self.backend
                )
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale * (sigmas[index + 1] - sigma_hat) * derivative
        return coords

    @torch.no_grad()
    def _sample_with_potentials(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        num_samples: int = 1,
        max_parallel_samples: int | None = None,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        generator: torch.Generator | None = None,
        **conditioning: object,
    ) -> Tensor:
        device = atom_mask.device
        dtype = cast(Tensor, conditioning["single_trunk"]).dtype
        input = cast(Boltz2Input, conditioning.pop("input"))
        step_scale = self.step_scale if step_scale is None else step_scale
        batch_size = atom_mask.shape[0]
        num_samples = operator.index(num_samples)
        num_requested_samples = num_samples
        if feynman_kac:
            num_particles = operator.index(num_particles)
            if num_particles < 1:
                fail("value_invalid", RequestRef("num_particles", str(num_particles)))
            feynman_kac_resampling_interval = operator.index(feynman_kac_resampling_interval)
            if feynman_kac_resampling_interval < 1:
                fail(
                    "value_invalid", RequestRef("feynman_kac_resampling_interval", str(feynman_kac_resampling_interval))
                )
        num_effective_samples = num_samples * num_particles if feynman_kac else num_samples
        if max_parallel_samples is None:
            max_parallel_samples = num_effective_samples
        else:
            max_parallel_samples = operator.index(max_parallel_samples)
            if max_parallel_samples < 1:
                fail("value_invalid", RequestRef("max_parallel_samples", str(max_parallel_samples)))
        if physical_guidance or contact_guidance:
            num_guidance_steps = operator.index(num_guidance_steps)
            if num_guidance_steps < 0:
                fail("value_invalid", RequestRef("num_guidance_steps", str(num_guidance_steps)))
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(num_effective_samples, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_hats = sigmas[:-1] * (1 + gammas[1:])
        noise_variances = self.noise_scale**2 * (sigma_hats**2 - sigmas[:-1] ** 2)
        noise_active = tuple((noise_variances > 0).tolist()) if feynman_kac else ()

        coords = sigmas[0] * draw_normal(shape, device=device, dtype=dtype, generator=generator)
        if feynman_kac:
            energy_trajectory = torch.empty(
                batch_size * num_effective_samples,
                0,
                device=device,
                dtype=dtype,
            )
            resample_weights = torch.ones(
                batch_size * num_requested_samples,
                num_particles,
                device=device,
                dtype=dtype,
            )
        else:
            energy_trajectory = torch.empty(0, 0, device=device, dtype=dtype)
            resample_weights = torch.empty(0, 0, device=device, dtype=dtype)
        potential_rows = (
            _prepare_boltz2_potential_rows(
                input,
                device=device,
                dtype=dtype,
                physical=feynman_kac or physical_guidance,
                contact=feynman_kac or contact_guidance,
            )
            if feynman_kac or physical_guidance or contact_guidance
            else ()
        )
        # `contact_guidance` defaults True, so the inner guidance loop would otherwise run its full
        # `num_guidance_steps` sweep of gradient evaluations every step even when the input carries no
        # constraint to guide toward. Physical potentials are coordinate-only and always contribute, while
        # the contact and forced-template fields produce exactly zero gradient when absent.
        guidance_active = physical_guidance or (
            contact_guidance
            and any(
                row.contact_pair_index.shape[1]
                or (row.template_reference is not None and row.template_reference.shape[0])
                for row in potential_rows
            )
        )
        scaled_guidance_update = torch.zeros(shape, device=device, dtype=dtype) if guidance_active else None
        denoised = None
        for index in range(len(sigmas) - 1):
            sigma_next = sigmas[index + 1]
            random_device = device if generator is None else generator.device
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0],
                device=random_device,
                dtype=coords.dtype,
                generator=generator,
            ).to(device)
            translation = draw_normal(
                (expanded_atom_mask.shape[0], 1, 3),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            coords = torch.einsum("bmd,bds->bms", coords - coords.mean(-2, keepdim=True), rotation) + translation
            if denoised is not None:
                denoised = (
                    torch.einsum("bmd,bds->bms", denoised - denoised.mean(-2, keepdim=True), rotation) + translation
                )
            if scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms",
                    scaled_guidance_update,
                    rotation,
                )

            sigma_hat = sigma_hats[index]
            noise_var = noise_variances[index]
            epsilon = torch.sqrt(torch.clamp(noise_var, min=0.0)) * draw_normal(
                shape, device=device, dtype=dtype, generator=generator
            )
            noisy = coords + epsilon
            noisy_by_sample = noisy.reshape(
                batch_size,
                num_effective_samples,
                *noisy.shape[1:],
            )
            denoised_by_sample = torch.zeros_like(noisy_by_sample)
            denoiser_conditioning = cast(AtomDenoiserContext, conditioning["conditioning"])
            for batch_start in range(
                0,
                batch_size,
                max_parallel_samples,
            ):
                batch_stop = min(
                    batch_start + max_parallel_samples,
                    batch_size,
                )
                batch_count = batch_stop - batch_start
                sample_chunk_size = max(
                    1,
                    max_parallel_samples // batch_count,
                )
                batch_conditioning = {
                    "single_inputs": cast(
                        Tensor,
                        conditioning["single_inputs"],
                    )[batch_start:batch_stop],
                    "single_trunk": cast(
                        Tensor,
                        conditioning["single_trunk"],
                    )[batch_start:batch_stop],
                    # The dense window layout is batch-independent, so only the batched tensors are sliced.
                    "conditioning": denoiser_conditioning._replace(
                        atom_query=denoiser_conditioning.atom_query[batch_start:batch_stop],
                        atom_conditioning=denoiser_conditioning.atom_conditioning[batch_start:batch_stop],
                        atom_encoder_bias=denoiser_conditioning.atom_encoder_bias[batch_start:batch_stop],
                        atom_decoder_bias=denoiser_conditioning.atom_decoder_bias[batch_start:batch_stop],
                        token_encoder_bias=denoiser_conditioning.token_encoder_bias[batch_start:batch_stop],
                    ),
                    "atom_to_token": cast(
                        Tensor,
                        conditioning["atom_to_token"],
                    )[batch_start:batch_stop],
                    "token_mask": cast(
                        Tensor,
                        conditioning["token_mask"],
                    )[batch_start:batch_stop],
                }
                for sample_start in range(
                    0,
                    num_effective_samples,
                    sample_chunk_size,
                ):
                    sample_stop = min(
                        sample_start + sample_chunk_size,
                        num_effective_samples,
                    )
                    sample_count = sample_stop - sample_start
                    denoised_by_sample[
                        batch_start:batch_stop,
                        sample_start:sample_stop,
                    ] = self.preconditioned_network_forward(
                        noisy_by_sample[
                            batch_start:batch_stop,
                            sample_start:sample_stop,
                        ],
                        sigma_hat,
                        atom_mask=base_atom_mask[batch_start:batch_stop],
                        **batch_conditioning,
                    )
            denoised = denoised_by_sample.flatten(0, 1)

            steering_t = 1.0 - index / (len(sigmas) - 1)
            should_resample = feynman_kac and (
                (index % feynman_kac_resampling_interval == 0 and noise_active[index]) or index == len(sigmas) - 2
            )
            if should_resample:
                active_physical = {0, 2, 3, 4, 5, 6}
                if steering_t <= 0.6:
                    active_physical.add(1)
                terms = _boltz2_potential_terms(
                    denoised,
                    potential_rows,
                    num_samples=num_effective_samples,
                    t=steering_t,
                    physical_guidance=physical_guidance,
                    contact_guidance=contact_guidance,
                    active_physical=frozenset(active_physical),
                    backend=self.backend,
                )
                energy = torch.zeros(
                    denoised.shape[0],
                    device=device,
                    dtype=dtype,
                )
                for term_energy, resampling_weight, _guidance_weight, _interval in terms:
                    if resampling_weight > 0:
                        energy = energy + resampling_weight * term_energy
                energy_trajectory = torch.cat(
                    (energy_trajectory, energy.unsqueeze(1)),
                    dim=1,
                )
                log_g = (
                    -energy if energy_trajectory.shape[1] == 1 else energy_trajectory[:, -2] - energy_trajectory[:, -1]
                )
                if scaled_guidance_update is not None and noise_active[index]:
                    likelihood_difference = (epsilon.square() - (epsilon + scaled_guidance_update).square()).sum(
                        dim=(-1, -2)
                    ) / (2 * noise_var)
                else:
                    likelihood_difference = torch.zeros_like(energy)
                resample_weights = torch.softmax(
                    (likelihood_difference + feynman_kac_lambda * log_g).reshape(-1, num_particles),
                    dim=1,
                )

            if guidance_active and index < len(sigmas) - 2:
                guidance_update = torch.zeros_like(denoised)
                for guidance_step in range(num_guidance_steps):
                    guidance_update = guidance_update - _boltz2_potential_gradient(
                        denoised + guidance_update,
                        potential_rows,
                        num_samples=num_effective_samples,
                        t=steering_t,
                        guidance_step=guidance_step,
                        physical_guidance=physical_guidance,
                        contact_guidance=contact_guidance,
                        backend=self.backend,
                    )
                denoised = denoised + guidance_update
                scaled_guidance_update = -guidance_update * step_scale * (sigma_next - sigma_hat) / sigma_hat

            if should_resample:
                sample_count = num_particles if index < len(sigmas) - 2 else 1
                offsets = num_particles * torch.arange(
                    resample_weights.shape[0],
                    device=device,
                ).unsqueeze(-1)
                random_device = resample_weights.device if generator is None else generator.device
                resample_index = (
                    torch.multinomial(
                        resample_weights.to(random_device),
                        sample_count,
                        replacement=True,
                        generator=generator,
                    ).to(device)
                    + offsets
                ).flatten()
                coords = coords[resample_index]
                noisy = noisy[resample_index]
                denoised = denoised[resample_index]
                expanded_atom_mask = expanded_atom_mask[resample_index]
                energy_trajectory = energy_trajectory[resample_index]
                if scaled_guidance_update is not None:
                    scaled_guidance_update = scaled_guidance_update[resample_index]
                if index == len(sigmas) - 2:
                    num_effective_samples = num_requested_samples

            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy, target=denoised, weight=expanded_atom_mask.to(noisy), backend=self.backend
                )
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale * (sigma_next - sigma_hat) * derivative
        return coords


class Boltz2AffinityHead(nn.Module):
    r"""
    Predict scalar binding affinity and a binder-class from trunk features and predicted atom geometry.
    """

    boundaries: Tensor

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)
        nn.init.zeros_(self.distance_embedding.weight)
        # The boundaries follow from the configuration, so they are computed rather than stored.
        # A model built on the meta device still needs real boundaries, since loading never fills them.
        buffer_device = self.distance_embedding.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "boundaries",
            torch.linspace(2.0, max_distance, num_distogram_bins - 1, device=buffer_device),
            persistent=False,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            expansion=expansion,
            backend=backend,
        )
        self.pair_update_stack = TrianglePairUpdateStack(
            block=LineDropoutTrianglePairUpdateBlock,
            num_blocks=num_blocks,
            pair_size=pair_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            dropout=dropout,
            backend=backend,
        )
        self.output_head = Boltz2AffinityOutputHead(pair_size=pair_size, single_size=single_size)

    def forward(
        self,
        *,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        token_representative_atom_index: Tensor,
        token_mask: Tensor,
        ligand_mask: Tensor,
        receptor_mask: Tensor,
        num_samples: int = 1,
    ) -> dict[str, Tensor]:
        coordinates = _boltz2_coordinates(coordinates)
        pair = self.pair_projection(self.pair_norm(pair))
        pair = pair + self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        pair = pair.repeat_interleave(num_samples, 0)

        rep_atom_index = token_representative_atom_index.repeat_interleave(num_samples, 0).long()
        representative = coordinates.gather(
            1,
            rep_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]),
        )
        distances = pairwise_distance(representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.pairwise_conditioner(pair, self.distance_embedding(distogram))

        token_mask = token_mask.repeat_interleave(num_samples, 0).bool()
        ligand_mask = ligand_mask.repeat_interleave(num_samples, 0).bool() & token_mask
        receptor_mask = receptor_mask.repeat_interleave(num_samples, 0).bool() & token_mask
        affinity_pair_mask = (
            ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
            | receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
            | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        )
        pair = self.pair_update_stack(pair, pair_mask=affinity_pair_mask)
        return self.output_head(pair, affinity_pair_mask)


class Boltz2AffinityOutputHead(nn.Module):
    r"""
    Project pooled affinity pair features to value and binder-class logits.
    """

    def __init__(self, pair_size: int = 128, single_size: int = 384) -> None:
        super().__init__()
        self.pair_feature_feed_forward = FeedForward(pair_size, pair_size, single_size, activation="relu")
        self.affinity_value_feed_forward = FeedForward(
            single_size, single_size, 1, num_hidden_layers=2, activation="relu"
        )
        self.affinity_score_feed_forward = FeedForward(
            single_size, single_size, 1, num_hidden_layers=2, activation="relu"
        )
        self.binder_logit_projection = nn.Linear(1, 1)

    def forward(self, pair: Tensor, affinity_pair_mask: Tensor) -> dict[str, Tensor]:
        num_tokens = affinity_pair_mask.shape[-1]
        non_diagonal = ~torch.eye(num_tokens, device=affinity_pair_mask.device, dtype=torch.bool).unsqueeze(0)
        pool_mask = (affinity_pair_mask.bool() & non_diagonal).to(dtype=pair.dtype).unsqueeze(-1)
        pooled = torch.sum(pair * pool_mask, dim=(-3, -2)) / (torch.sum(pool_mask, dim=(-3, -2)) + 1e-7)
        pooled = F.relu(self.pair_feature_feed_forward(pooled))
        affinity_pred_value = self.affinity_value_feed_forward(pooled).reshape(-1, 1)
        affinity_pred_score = self.affinity_score_feed_forward(pooled).reshape(-1, 1)
        affinity_logits_binary = self.binder_logit_projection(affinity_pred_score).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }


class Boltz2ConfidenceHead(nn.Module):
    r"""
    Score predicted structures with per-token pLDDT / resolved / PAE / PDE confidence logits.
    """

    boundaries: Tensor

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        relative_position_size: int = 139,
        num_bond_types: int = 7,
        min_conditioning_cutoff: float = 4.0,
        max_conditioning_cutoff: float = 20.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        pair_dropout: float = 0.25,
        post_layer_norm: bool = False,
        normalize_single_input: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.input_single_norm = LayerNorm(single_size, backend=backend)
        self.single_norm: LayerNorm | None = None
        if normalize_single_input:
            self.single_norm = LayerNorm(single_size, backend=backend)
        self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)

        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, min_cutoff=min_conditioning_cutoff, max_cutoff=max_conditioning_cutoff
        )

        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)
        # The boundaries follow from the configuration, so they are computed rather than stored.
        # A model built on the meta device still needs real boundaries, since loading never fills them.
        buffer_device = self.distance_embedding.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "boundaries",
            torch.linspace(2.0, max_distance, num_distogram_bins - 1, device=buffer_device),
            persistent=False,
        )

        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            num_pair_attention_heads=num_pair_attention_heads,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            dropout=pair_dropout,
            block=PostNormLineDropoutPairformerBlock if post_layer_norm else LineDropoutPairformerBlock,
            backend=backend,
        )
        self.output_head = Boltz2ConfidenceOutputHead(
            single_size,
            pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        residue_index: Tensor,
        chain_index: Tensor,
        entity_index: Tensor,
        symmetry_index: Tensor,
        token_index: Tensor,
        token_bonds: Tensor,
        type_bonds: Tensor,
        contact_conditioning: Tensor,
        contact_threshold: Tensor,
        token_representative_atom_index: Tensor,
        token_mask: Tensor,
        cyclic_period: Tensor | None = None,
    ) -> dict[str, Tensor]:
        coordinates = _boltz2_coordinates(coordinates)
        single_inputs = self.input_single_norm(single_inputs)
        if self.single_norm is not None:
            single = self.single_norm(single)
        single = single + self.input_single_projection(single_inputs)
        pair = self.pair_norm(pair)

        relative_features = generate_relative_position_features(
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            symmetry_index=symmetry_index,
            token_index=token_index,
            cyclic_period=cyclic_period,
            dtype=single.dtype,
        )
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1).to(pair.dtype))
        pair = pair + self.token_bond_type_embedding(type_bonds.long())
        pair = pair + self.contact_conditioning(contact_conditioning.to(pair.dtype), contact_threshold.to(pair.dtype))

        pair = (
            pair
            + self.pair_left_projection(single_inputs).unsqueeze(-2)
            + self.pair_right_projection(single_inputs).unsqueeze(-3)
        )
        pair = pair + self.single_to_pair_product_out(
            self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
            * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        )
        num_samples = coordinates.shape[-3]
        single = insert_sample_axis(single, num_samples, dim=-3)
        pair = insert_sample_axis(pair, num_samples, dim=-4)
        representative = gather_atoms_by_index(
            coordinates,
            insert_sample_axis(token_representative_atom_index, num_samples, dim=-2),
        )
        distances = pairwise_distance(representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = insert_sample_axis(token_mask, num_samples, dim=-2)
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        chain_index = insert_sample_axis(chain_index, num_samples, dim=-2)
        return self.output_head(single, pair, chain_index)


class Boltz2ConfidenceOutputHead(nn.Module):
    r"""
    Project the confidence single/pair representations to pLDDT / resolved / PAE / PDE logits.
    """

    def __init__(
        self,
        single_size: int,
        pair_size: int,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.pae_intra_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pae_inter_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_intra_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.pde_inter_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.plddt_projection = nn.Linear(single_size, num_plddt_bins, bias=False)
        self.resolved_projection = nn.Linear(single_size, 2, bias=False)

    def forward(self, single: Tensor, pair: Tensor, chain_index: Tensor) -> dict[str, Tensor]:
        same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).unsqueeze(-1)
        pae_logits = torch.where(same_chain, self.pae_intra_projection(pair), self.pae_inter_projection(pair))
        symmetric = pair + pair.transpose(-3, -2)
        pde_logits = torch.where(same_chain, self.pde_intra_projection(symmetric), self.pde_inter_projection(symmetric))
        return {
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }


def _boltz2_confidence_values(
    *,
    plddt_logits: Tensor,
    pae_logits: Tensor,
    pde_logits: Tensor,
    resolved_logits: Tensor,
    coordinates: Tensor,
    distogram_logits: Tensor,
    input: Boltz2Input,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""
    Derive the released Boltz-2 confidence values while preserving ``(B, S)``.
    """

    batch_size, num_samples = coordinates.shape[:2]
    plddt = plddt_from_logits(plddt_logits)
    pae = categorical_mean(pae_logits, end=32.0)
    pde = categorical_mean(pde_logits, end=32.0)

    token_mask = input.token_mask[:, None].to(device=coordinates.device, dtype=coordinates.dtype)
    chain_index = input.chain_index[:, None].to(device=coordinates.device)
    molecule_type = input.molecule_type[:, None].to(device=coordinates.device)
    representative_index = input.token_representative_atom_index[:, None, :, None].expand(
        -1,
        num_samples,
        -1,
        3,
    )
    representative = coordinates.gather(2, representative_index)
    distances = pairwise_distance(representative)
    different_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(coordinates.dtype)
    is_ligand = (molecule_type == 3).to(coordinates.dtype)
    interface_weight = interface_confidence_weights(
        distances,
        chain_index,
        is_ligand,
        token_mask,
        ligand_weight=20.0,
        interface_weight=10.0,
        background_weight=1.0,
    )
    mean_plddt = weighted_mean(plddt, token_mask, backend=backend)
    complex_iplddt = weighted_mean(plddt, interface_weight, backend=backend)

    distogram_probability = distogram_logits.softmax(dim=-1)
    contact_bin_count = min(20, distogram_probability.shape[-1])
    predicted_contact = distogram_probability[..., :contact_bin_count].sum(dim=-1)[:, None]
    token_pair_mask = token_mask.bool().unsqueeze(-1) & token_mask.bool().unsqueeze(-2)
    diagonal = torch.eye(token_pair_mask.shape[-1], device=token_pair_mask.device, dtype=torch.bool)
    token_pair_mask = token_pair_mask & ~diagonal
    contact_pair_weight = predicted_contact.masked_fill(~token_pair_mask, 0.0)
    complex_pde = weighted_mean(pde.flatten(-2), contact_pair_weight.flatten(-2), backend=backend)
    interface_pair_weight = contact_pair_weight.masked_fill(~different_chain.bool(), 0.0)
    complex_ipde = weighted_mean(pde.flatten(-2), interface_pair_weight.flatten(-2), backend=backend)

    design_mask = input.design_mask[:, None].to(device=coordinates.device, dtype=torch.bool)
    chain_design_mask = input.chain_design_mask[:, None].to(device=coordinates.device, dtype=torch.bool)
    target_mask = ~chain_design_mask
    chain_design_target_mask = token_pair_mask.bool() & (
        (chain_design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2))
    )
    min_interaction_pae = minimum_confidence_error(pae, chain_design_target_mask)
    design_target_mask = token_pair_mask.bool() & (
        (design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & design_mask.unsqueeze(-2))
    )
    interaction_pae = weighted_mean(pae.flatten(-2), design_target_mask.flatten(-2), backend=backend)
    min_design_to_target_pae = minimum_confidence_error(pae, design_target_mask)
    has_design = design_mask.any(dim=-1)
    min_design_to_target_pae = torch.where(
        has_design,
        min_design_to_target_pae,
        min_design_to_target_pae.new_full(min_design_to_target_pae.shape, torch.nan),
    )

    ptm_scores = _boltz2_confidence_scope_scores(
        pae_logits.flatten(0, 1),
        coordinates=coordinates.flatten(0, 1),
        input=input,
        num_samples=num_samples,
        backend=backend,
    )
    return {
        "plddt": plddt,
        "pae": pae,
        "pde": pde,
        "resolved_probability": resolved_probability_from_logits(resolved_logits),
        "mean_plddt": mean_plddt,
        "complex_iplddt": complex_iplddt,
        "complex_pde": complex_pde,
        "complex_ipde": complex_ipde,
        "ptm": ptm_scores["ptm"].reshape(batch_size, num_samples),
        "iptm": ptm_scores["iptm"].reshape(batch_size, num_samples),
        "ligand_iptm": ptm_scores["ligand_iptm"].reshape(batch_size, num_samples),
        "protein_iptm": ptm_scores["protein_iptm"].reshape(batch_size, num_samples),
        "design_to_target_iptm": ptm_scores["design_to_target_iptm"].reshape(batch_size, num_samples),
        "design_residue_iptm": ptm_scores["design_residue_iptm"].reshape(batch_size, num_samples),
        "design_iptm": ptm_scores["design_iptm"].reshape(batch_size, num_samples),
        "design_iiptm": ptm_scores["design_iiptm"].reshape(batch_size, num_samples),
        "target_ptm": ptm_scores["target_ptm"].reshape(batch_size, num_samples),
        "design_ptm": ptm_scores["design_ptm"].reshape(batch_size, num_samples),
        "interaction_pae": interaction_pae,
        "min_interaction_pae": min_interaction_pae,
        "min_design_to_target_pae": min_design_to_target_pae,
        "design_ipsae_min": ptm_scores["design_ipsae_min"].reshape(batch_size, num_samples),
        "design_to_target_ipsae": ptm_scores["design_to_target_ipsae"].reshape(batch_size, num_samples),
        "target_to_design_ipsae": ptm_scores["target_to_design_ipsae"].reshape(batch_size, num_samples),
        "chain_pair_iptm": ptm_scores["chain_pair_iptm"].reshape(
            batch_size,
            num_samples,
            *ptm_scores["chain_pair_iptm"].shape[-2:],
        ),
    }


def _boltz2_confidence_scope_scores(
    pae_logits: Tensor,
    *,
    coordinates: Tensor,
    input: Boltz2Input,
    num_samples: int,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""
    Prepare frame and design scopes and assemble their shared TM/iPSAE scores.
    """

    token_mask = input.token_mask.repeat_interleave(num_samples, 0).to(pae_logits)
    frame_mask = _boltz2_predicted_frame_mask(
        coordinates,
        input=input,
        num_samples=num_samples,
    ).to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(num_samples, 0)
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
    num_tokens = token_mask.sum(dim=-1, keepdim=True)
    tm_expected = tm_weight_from_pae(pae_logits, num_tokens.unsqueeze(-1), bin_centers=pae_value)

    molecule_type = input.molecule_type.repeat_interleave(num_samples, 0)
    is_ligand = (molecule_type == 3).to(pae_logits)
    is_protein = (molecule_type == 0).to(pae_logits)
    ligand_mask = pair_mask_iptm * (
        is_ligand[:, :, None] * is_protein[:, None, :] + is_protein[:, :, None] * is_ligand[:, None, :]
    )
    protein_mask = pair_mask_iptm * (is_protein[:, :, None] * is_protein[:, None, :])

    design_mask = input.design_mask.bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    chain_design_mask = torch.where(has_design, input.chain_design_mask.bool(), design_mask)
    target_mask = ~chain_design_mask
    design_mask = design_mask.repeat_interleave(num_samples, 0)
    chain_design_mask = chain_design_mask.repeat_interleave(num_samples, 0)
    target_mask = target_mask.repeat_interleave(num_samples, 0)
    design_iptm_mask = pair_mask_ptm * (
        (chain_design_mask[:, :, None] & target_mask[:, None, :])
        | (target_mask[:, :, None] & chain_design_mask[:, None, :])
    ).to(pae_logits)
    design_to_target_iptm_mask = pair_mask_ptm * (
        (design_mask[:, :, None] & target_mask[:, None, :]) | (target_mask[:, :, None] & design_mask[:, None, :])
    ).to(pae_logits)
    design_residue_iptm_mask = (
        pair_mask_ptm
        * (chain_index[:, :, None] != chain_index[:, None, :]).to(pae_logits)
        * (design_mask[:, :, None].to(pae_logits) + design_mask[:, None, :].to(pae_logits)).clamp(max=1)
    )
    design_ptm_mask = pair_mask_ptm * (chain_design_mask[:, :, None] & chain_design_mask[:, None, :]).to(pae_logits)
    target_ptm_mask = pair_mask_ptm * (target_mask[:, :, None] & target_mask[:, None, :]).to(pae_logits)

    num_chains = int(input.chain_index.max().item()) + 1
    chain_pair_iptm = chain_pair_tm_scores(
        tm_expected, chain_index, range(num_chains), pair_weights=pair_mask_ptm, backend=backend
    )

    interaction_mask = _boltz2_design_interaction_mask(
        coordinates,
        input=input,
        num_samples=num_samples,
        chain_design_mask=chain_design_mask,
    )
    design_iiptm_mask = pair_mask_ptm * (
        (interaction_mask[:, :, None] & target_mask[:, None, :])
        | (target_mask[:, :, None] & interaction_mask[:, None, :])
    ).to(pae_logits)

    expected_pae = categorical_expectation(pae_logits.softmax(dim=-1), pae_value)
    design_to_target_ipsae = directional_ipsae(
        expected_pae,
        source_mask=chain_design_mask,
        target_mask=target_mask,
        frame_mask=frame_mask,
        token_mask=token_mask,
        backend=backend,
    )
    target_to_design_ipsae = directional_ipsae(
        expected_pae,
        source_mask=target_mask,
        target_mask=chain_design_mask,
        frame_mask=frame_mask,
        token_mask=token_mask,
        backend=backend,
    )

    return {
        "ptm": predicted_tm_score(tm_expected, pair_weights=pair_mask_ptm, backend=backend),
        "iptm": predicted_tm_score(tm_expected, pair_weights=pair_mask_iptm, backend=backend),
        "ligand_iptm": predicted_tm_score(tm_expected, pair_weights=ligand_mask, backend=backend),
        "protein_iptm": predicted_tm_score(tm_expected, pair_weights=protein_mask, backend=backend),
        "design_to_target_iptm": predicted_tm_score(
            tm_expected, pair_weights=design_to_target_iptm_mask, backend=backend
        ),
        "design_residue_iptm": predicted_tm_score(tm_expected, pair_weights=design_residue_iptm_mask, backend=backend),
        "design_iptm": predicted_tm_score(tm_expected, pair_weights=design_iptm_mask, backend=backend),
        "design_iiptm": predicted_tm_score(tm_expected, pair_weights=design_iiptm_mask, backend=backend),
        "target_ptm": predicted_tm_score(tm_expected, pair_weights=target_ptm_mask, backend=backend),
        "design_ptm": predicted_tm_score(tm_expected, pair_weights=design_ptm_mask, backend=backend),
        "design_ipsae_min": torch.minimum(design_to_target_ipsae, target_to_design_ipsae),
        "design_to_target_ipsae": design_to_target_ipsae,
        "target_to_design_ipsae": target_to_design_ipsae,
        "chain_pair_iptm": chain_pair_iptm,
    }


def _boltz2_design_interaction_mask(
    coordinates: Tensor,
    *,
    input: Boltz2Input,
    num_samples: int,
    chain_design_mask: Tensor,
) -> Tensor:
    r"""
    Select design tokens with at least one atom within 8 Å of any target atom.
    """

    atom_to_token = input.atom_to_token.repeat_interleave(num_samples, 0).long()
    atom_mask = input.atom_mask.repeat_interleave(num_samples, 0).bool()
    atom_chain_design_mask = chain_design_mask.gather(1, atom_to_token) & atom_mask
    interaction_mask = chain_design_mask.new_zeros(chain_design_mask.shape)
    for sample_index in range(coordinates.shape[0]):
        design_atoms = atom_chain_design_mask[sample_index]
        target_atoms = atom_mask[sample_index] & ~design_atoms
        if design_atoms.any() and target_atoms.any():
            interacting_atoms = (
                pairwise_distance(
                    coordinates[sample_index, design_atoms],
                    coordinates[sample_index, target_atoms],
                ).amin(dim=-1)
                < 8
            )
            interacting_tokens = atom_to_token[sample_index, design_atoms][interacting_atoms]
            interaction_mask[sample_index].scatter_(
                0,
                interacting_tokens,
                torch.ones_like(interacting_tokens, dtype=torch.bool),
            )
    return interaction_mask


def _boltz2_predicted_frame_mask(
    coordinates: Tensor,
    *,
    input: Boltz2Input,
    num_samples: int,
) -> Tensor:
    r"""
    Return the exact predicted-frame validity mask used by Boltz-2 pTM.
    """

    if input.frame_atom_index is None:
        fail("value_invalid", RequestRef("frame_atom_index", "None"))
    frame_index = (
        input.frame_atom_index.repeat_interleave(num_samples, 0)
        .to(
            device=coordinates.device,
            dtype=torch.long,
        )
        .clone()
    )
    molecule_type = input.molecule_type.to(device=coordinates.device)
    chain_index = input.chain_index.to(device=coordinates.device)
    token_mask = input.token_mask.to(device=coordinates.device, dtype=torch.bool)
    atom_mask = input.atom_mask.to(device=coordinates.device, dtype=torch.bool)
    atom_to_token = input.atom_to_token.to(device=coordinates.device, dtype=torch.long)
    atom_chain = chain_index.gather(1, atom_to_token)

    for batch_index in range(input.token_mask.shape[0]):
        ligand_chains = torch.unique(
            chain_index[batch_index][token_mask[batch_index] & (molecule_type[batch_index] == 3)]
        )
        for chain in ligand_chains:
            chain_tokens = torch.nonzero(
                token_mask[batch_index] & (molecule_type[batch_index] == 3) & (chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            chain_atoms = torch.nonzero(
                atom_mask[batch_index] & (atom_chain[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if len(chain_atoms) < 3:
                continue
            for sample_index in range(num_samples):
                row = batch_index * num_samples + sample_index
                distances = pairwise_distance(coordinates[row, chain_atoms])
                nearest = distances.argsort(dim=-1)[:, :3]
                nearest_atoms = chain_atoms[nearest]
                frame_index[row, chain_tokens] = nearest_atoms[:, (1, 0, 2)]

    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    frames = coordinates.gather(1, flat_index.unsqueeze(-1).expand(-1, -1, 3))
    frames = frames.reshape(*frame_index.shape, 3)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    first_direction = first / (first_norm.unsqueeze(-1) + 1e-6)
    second_direction = second / (second_norm.unsqueeze(-1) + 1e-6)
    cosine = (first_direction * second_direction).sum(dim=-1)
    valid_geometry = (cosine.abs() < 0.9063) & (first_norm > 1e-2) & (second_norm > 1e-2)
    return valid_geometry & token_mask.repeat_interleave(num_samples, 0)


class _Boltz2PotentialRow(NamedTuple):
    atom_mask: Tensor
    atom_chain: Tensor
    num_chains: int
    symmetric_chain_index: Tensor
    vdw_pair_index: Tensor
    vdw_lower_bound: Tensor
    connected_atom_index: Tensor
    geometry_pair_index: Tensor
    geometry_lower_bound: Tensor
    geometry_upper_bound: Tensor
    chiral_atom_index: Tensor
    chiral_lower_bound: Tensor
    chiral_upper_bound: Tensor
    stereo_bond_index: Tensor
    stereo_lower_bound: Tensor
    stereo_upper_bound: Tensor
    planar_bond_index: Tensor
    contact_pair_index: Tensor
    contact_threshold: Tensor
    contact_union_masks: tuple[Tensor, ...]
    template_reference: Tensor | None
    template_reference_mask: Tensor | None
    template_frame_index: Tensor | None
    template_threshold: Tensor | None
    token_representative_atom_index: Tensor
    atom_to_token: Tensor


def _prepare_boltz2_potential_rows(
    input: Boltz2Input,
    *,
    device: torch.device,
    dtype: torch.dtype,
    physical: bool,
    contact: bool,
) -> tuple[_Boltz2PotentialRow, ...]:
    r"""
    Lower coordinate-invariant Boltz-2 steering inputs once per sampler call.
    """

    rows = []
    for batch_index in range(input.atom_mask.shape[0]):
        atom_mask = input.atom_mask[batch_index].to(device=device, dtype=torch.bool)
        atom_to_token = input.atom_to_token[batch_index].to(device=device, dtype=torch.long)
        chain_index = input.chain_index[batch_index].to(device=device, dtype=torch.long)
        atom_chain = chain_index.gather(0, atom_to_token)

        if physical:
            num_chains = int(chain_index.max().item()) + 1
            chain_size = torch.bincount(
                atom_chain[atom_mask],
                minlength=num_chains,
            )
            symmetric_chain_index = input.symmetric_chain_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            active_chains = torch.unique(atom_chain[atom_mask])
            symmetric_chain_index = symmetric_chain_index[
                :, torch.isin(symmetric_chain_index, active_chains).all(dim=0)
            ]
            symmetric_chain_index = symmetric_chain_index[:, (chain_size[symmetric_chain_index] > 1).all(dim=0)]

            vdw_pair_index = torch.triu_indices(
                atom_mask.shape[0],
                atom_mask.shape[0],
                1,
                device=device,
            )
            vdw_pair_mask = atom_mask[vdw_pair_index].all(dim=0)
            vdw_ion_mask = (chain_size > 1)[atom_chain[vdw_pair_index]].all(dim=0)
            connected_chain_index = input.connected_chain_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            connected_chain = torch.eye(
                num_chains,
                device=device,
                dtype=torch.bool,
            )
            if connected_chain_index.shape[1]:
                connected_chain[
                    connected_chain_index[0],
                    connected_chain_index[1],
                ] = True
                connected_chain[
                    connected_chain_index[1],
                    connected_chain_index[0],
                ] = True
            vdw_connected_mask = connected_chain[
                atom_chain[vdw_pair_index[0]],
                atom_chain[vdw_pair_index[1]],
            ]
            vdw_pair_index = vdw_pair_index[
                :,
                vdw_pair_mask & vdw_ion_mask & ~vdw_connected_mask,
            ]
            element_features = input.atom_features[batch_index, :, 4:132]
            vdw_lower_bound = _boltz2_pair_vdw_radii(element_features, vdw_pair_index, dtype=dtype).sum(dim=0) * 0.775

            connected_atom_index = input.connected_atom_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )

            geometry_pair_index = input.geometry_pair_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            geometry_available = atom_mask[geometry_pair_index].all(dim=0)
            geometry_pair_index = geometry_pair_index[:, geometry_available]
            geometry_lower_bound = (
                input.geometry_lower_bounds[batch_index][
                    geometry_available.to(device=input.geometry_lower_bounds.device)
                ]
                .to(
                    device=device,
                    dtype=dtype,
                )
                .clone()
            )
            geometry_upper_bound = (
                input.geometry_upper_bounds[batch_index][
                    geometry_available.to(device=input.geometry_upper_bounds.device)
                ]
                .to(
                    device=device,
                    dtype=dtype,
                )
                .clone()
            )
            geometry_bond_mask = input.geometry_bond_mask[batch_index].to(
                device=device,
                dtype=torch.bool,
            )[geometry_available]
            geometry_angle_mask = input.geometry_angle_mask[batch_index].to(
                device=device,
                dtype=torch.bool,
            )[geometry_available]
            geometry_lower_bound[geometry_bond_mask & ~geometry_angle_mask] *= 0.875
            geometry_upper_bound[geometry_bond_mask & ~geometry_angle_mask] *= 1.125
            geometry_lower_bound[~geometry_bond_mask & geometry_angle_mask] *= 0.875
            geometry_upper_bound[~geometry_bond_mask & geometry_angle_mask] *= 1.125
            geometry_lower_bound[geometry_bond_mask & geometry_angle_mask] *= 0.875
            geometry_upper_bound[geometry_bond_mask & geometry_angle_mask] *= 1.125
            geometry_lower_bound[~geometry_bond_mask & ~geometry_angle_mask] *= 0.9
            geometry_upper_bound[~geometry_bond_mask & ~geometry_angle_mask] = float("inf")
            geometry_cutoff = 0.35 + _boltz2_pair_vdw_radii(element_features, geometry_pair_index, dtype=dtype).mean(
                dim=0
            )
            geometry_lower_bound[~geometry_bond_mask] = torch.maximum(
                geometry_lower_bound[~geometry_bond_mask],
                geometry_cutoff[~geometry_bond_mask],
            )
            geometry_upper_bound[geometry_bond_mask] = torch.minimum(
                geometry_upper_bound[geometry_bond_mask],
                geometry_cutoff[geometry_bond_mask],
            )

            chiral_atom_index = input.chiral_atom_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            chiral_atom_positive = input.chiral_atom_positive[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            chiral_lower_bound = torch.where(
                chiral_atom_positive,
                geometry_lower_bound.new_tensor(0.52360),
                geometry_lower_bound.new_tensor(float("-inf")),
            )
            chiral_upper_bound = torch.where(
                chiral_atom_positive,
                geometry_lower_bound.new_tensor(float("inf")),
                geometry_lower_bound.new_tensor(-0.52360),
            )

            stereo_bond_index = input.stereo_bond_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            stereo_bond_trans = input.stereo_bond_trans[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            stereo_lower_bound = torch.where(
                stereo_bond_trans,
                geometry_lower_bound.new_tensor(torch.pi - 0.52360),
                geometry_lower_bound.new_tensor(float("-inf")),
            )
            stereo_upper_bound = torch.where(
                stereo_bond_trans,
                geometry_lower_bound.new_tensor(float("inf")),
                geometry_lower_bound.new_tensor(0.52360),
            )

            planar_bond_index = input.planar_bond_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            planar_bond_index = torch.cat(
                (
                    planar_bond_index[[1, 2, 3, 0]],
                    planar_bond_index[[4, 5, 0, 3]],
                ),
                dim=1,
            )
        else:
            num_chains = 0
            symmetric_chain_index = torch.empty(2, 0, device=device, dtype=torch.long)
            vdw_pair_index = torch.empty(2, 0, device=device, dtype=torch.long)
            vdw_lower_bound = torch.empty(0, device=device, dtype=dtype)
            connected_atom_index = torch.empty(2, 0, device=device, dtype=torch.long)
            geometry_pair_index = torch.empty(2, 0, device=device, dtype=torch.long)
            geometry_lower_bound = torch.empty(0, device=device, dtype=dtype)
            geometry_upper_bound = torch.empty(0, device=device, dtype=dtype)
            chiral_atom_index = torch.empty(4, 0, device=device, dtype=torch.long)
            chiral_lower_bound = torch.empty(0, device=device, dtype=dtype)
            chiral_upper_bound = torch.empty(0, device=device, dtype=dtype)
            stereo_bond_index = torch.empty(4, 0, device=device, dtype=torch.long)
            stereo_lower_bound = torch.empty(0, device=device, dtype=dtype)
            stereo_upper_bound = torch.empty(0, device=device, dtype=dtype)
            planar_bond_index = torch.empty(4, 0, device=device, dtype=torch.long)

        if contact:
            contact_pair_index = input.contact_pair_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            contact_threshold = input.contact_thresholds[batch_index].to(
                device=device,
                dtype=dtype,
            )
            contact_union_index = input.contact_union_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            contact_union_masks = tuple(contact_union_index == union for union in contact_union_index.unique())
            if input.template_force is None:
                template_reference = None
                template_reference_mask = None
                template_frame_index = None
                template_threshold = None
            else:
                template_force = input.template_force[batch_index].to(
                    device=device,
                    dtype=torch.bool,
                )
                template_reference = cast(
                    Tensor,
                    input.template_pseudo_beta_positions,
                )[
                    batch_index, template_force
                ].to(device=device, dtype=dtype)
                template_reference_mask = cast(
                    Tensor,
                    input.template_pseudo_beta_mask,
                )[
                    batch_index, template_force
                ].to(device=device, dtype=dtype)
                template_reference_mask = template_reference_mask * input.token_mask[batch_index].to(
                    device=device, dtype=dtype
                )
                template_frame_index = cast(Tensor, input.template_visibility_index)[batch_index, template_force].to(
                    device=device, dtype=torch.long
                )
                template_threshold = cast(
                    Tensor,
                    input.template_force_threshold,
                )[
                    batch_index, template_force
                ].to(device=device, dtype=dtype)
            token_representative_atom_index = input.token_representative_atom_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
        else:
            contact_pair_index = torch.empty(2, 0, device=device, dtype=torch.long)
            contact_threshold = torch.empty(0, device=device, dtype=dtype)
            contact_union_masks = ()
            template_reference = None
            template_reference_mask = None
            template_frame_index = None
            template_threshold = None
            token_representative_atom_index = torch.empty(0, device=device, dtype=torch.long)

        rows.append(
            _Boltz2PotentialRow(
                atom_mask=atom_mask,
                atom_chain=atom_chain,
                num_chains=num_chains,
                symmetric_chain_index=symmetric_chain_index,
                vdw_pair_index=vdw_pair_index,
                vdw_lower_bound=vdw_lower_bound,
                connected_atom_index=connected_atom_index,
                geometry_pair_index=geometry_pair_index,
                geometry_lower_bound=geometry_lower_bound,
                geometry_upper_bound=geometry_upper_bound,
                chiral_atom_index=chiral_atom_index,
                chiral_lower_bound=chiral_lower_bound,
                chiral_upper_bound=chiral_upper_bound,
                stereo_bond_index=stereo_bond_index,
                stereo_lower_bound=stereo_lower_bound,
                stereo_upper_bound=stereo_upper_bound,
                planar_bond_index=planar_bond_index,
                contact_pair_index=contact_pair_index,
                contact_threshold=contact_threshold,
                contact_union_masks=contact_union_masks,
                template_reference=template_reference,
                template_reference_mask=template_reference_mask,
                template_frame_index=template_frame_index,
                template_threshold=template_threshold,
                token_representative_atom_index=token_representative_atom_index,
                atom_to_token=atom_to_token,
            )
        )
    return tuple(rows)


def _boltz2_potential_terms(
    coordinates: Tensor,
    rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    t: float,
    physical_guidance: bool,
    contact_guidance: bool,
    include_contact: bool = True,
    active_physical: frozenset[int] | None = None,
    backend: str = "auto",
) -> tuple[tuple[Tensor, float, float, int], ...]:
    r"""
    Evaluate the released Boltz-2 physical, contact, and template energies.
    """

    batch_size = len(rows)
    coordinates = coordinates.reshape(
        batch_size,
        num_samples,
        *coordinates.shape[1:],
    )
    term_rows = [
        _boltz2_batch_potential_terms(
            coordinates[batch_index],
            rows[batch_index],
            t=t,
            include_contact=include_contact,
            active_physical=active_physical,
            backend=backend,
        )
        for batch_index in range(batch_size)
    ]
    energies = tuple(torch.cat([row[index] for row in term_rows]) for index in range(9))
    vdw_guidance = 0.125 if t <= 0.4 and physical_guidance else 0.0
    vdw_resampling = 0.01 if t <= 0.6 else 0.0
    if t <= 0.25:
        contact_weight = 0.0
    elif t <= 0.75:
        contact_weight = 0.5
    else:
        contact_weight = 1.0
    return (
        (
            energies[0],
            0.5,
            0.5 if physical_guidance else 0.0,
            4,
        ),
        (energies[1], vdw_resampling, vdw_guidance, 5),
        (
            energies[2],
            1.0,
            0.15 if physical_guidance else 0.0,
            1,
        ),
        (
            energies[3],
            0.1,
            0.01 if physical_guidance else 0.0,
            1,
        ),
        (
            energies[4],
            1.0,
            0.1 if physical_guidance else 0.0,
            1,
        ),
        (
            energies[5],
            1.0,
            0.05 if physical_guidance else 0.0,
            1,
        ),
        (
            energies[6],
            1.0,
            0.05 if physical_guidance else 0.0,
            1,
        ),
        (
            energies[7],
            1.0,
            contact_weight if contact_guidance else 0.0,
            4,
        ),
        (
            energies[8],
            1.0,
            0.1 if contact_guidance else 0.0,
            2,
        ),
    )


def _boltz2_potential_gradient(
    coordinates: Tensor,
    rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    t: float,
    guidance_step: int,
    physical_guidance: bool,
    contact_guidance: bool,
    backend: str = "auto",
) -> Tensor:
    r"""
    Evaluate the released scheduled Boltz-2 guidance vector field.

    Chain-center and contact guidance differentiate their reported energies. Template token
    fields are copied to every atom in that token.
    """

    gradient = torch.zeros_like(coordinates)
    if physical_guidance:
        if guidance_step % 4 == 0:
            gradient = gradient + 0.5 * _boltz2_symmetric_chain_com_gradient(
                coordinates,
                rows,
                num_samples=num_samples,
                t=t,
            )
        gradient = gradient + _boltz2_local_physical_gradient(
            coordinates, rows, num_samples=num_samples, t=t, guidance_step=guidance_step, backend=backend
        )
    if contact_guidance:
        if guidance_step % 4 == 0:
            if t <= 0.25:
                contact_weight = 0.0
            elif t <= 0.75:
                contact_weight = 0.5
            else:
                contact_weight = 1.0
            if contact_weight:
                gradient = gradient + contact_weight * _boltz2_contact_gradient(
                    coordinates,
                    rows,
                    num_samples=num_samples,
                    t=t,
                )
        if guidance_step % 2 == 0:
            gradient = gradient + 0.1 * _boltz2_template_gradient(
                coordinates, rows, num_samples=num_samples, backend=backend
            )
    return gradient


def _boltz2_local_physical_gradient(
    coordinates: Tensor,
    potential_rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    t: float,
    guidance_step: int,
    backend: str = "auto",
) -> Tensor:
    batch_size = len(potential_rows)
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        num_samples,
        *coordinates.shape[1:],
    )
    gradients = []
    for batch_index, row in enumerate(potential_rows):
        row_coordinates = coordinates_by_batch[batch_index]
        row_gradient = torch.zeros_like(row_coordinates)

        if t <= 0.4 and guidance_step % 5 == 0 and row.vdw_pair_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                row.vdw_pair_index,
            )
            slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(0.125),
                row.vdw_lower_bound,
                distance.new_tensor(float("inf")),
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.vdw_pair_index,
                    distance_gradient,
                    slope,
                )
            )

        if row.connected_atom_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                row.connected_atom_index,
            )
            slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(0.15),
                distance.new_tensor(float("-inf")),
                distance.new_tensor(2.0),
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.connected_atom_index,
                    distance_gradient,
                    slope,
                )
            )

        if row.geometry_pair_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                row.geometry_pair_index,
            )
            slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(0.01),
                row.geometry_lower_bound,
                row.geometry_upper_bound,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.geometry_pair_index,
                    distance_gradient,
                    slope,
                )
            )

        if row.chiral_atom_index.shape[1]:
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates, row.chiral_atom_index, backend=backend
            )
            slope = linear_flat_bottom_derivative(
                dihedral,
                dihedral.new_tensor(0.1),
                row.chiral_lower_bound,
                row.chiral_upper_bound,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.chiral_atom_index,
                    dihedral_gradient,
                    slope,
                )
            )

        if row.stereo_bond_index.shape[1]:
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates, row.stereo_bond_index, backend=backend
            )
            absolute = dihedral.abs()
            absolute_gradient = torch.where(
                (dihedral < 0)[:, None, :, None],
                -dihedral_gradient,
                dihedral_gradient,
            )
            slope = linear_flat_bottom_derivative(
                absolute,
                absolute.new_tensor(0.05),
                row.stereo_lower_bound,
                row.stereo_upper_bound,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.stereo_bond_index,
                    absolute_gradient,
                    slope,
                )
            )

        if row.planar_bond_index.shape[1]:
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates, row.planar_bond_index, backend=backend
            )
            absolute = dihedral.abs()
            absolute_gradient = torch.where(
                (dihedral < 0)[:, None, :, None],
                -dihedral_gradient,
                dihedral_gradient,
            )
            slope = linear_flat_bottom_derivative(
                absolute,
                absolute.new_tensor(0.05),
                absolute.new_tensor(float("-inf")),
                absolute.new_tensor(0.26180),
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    row.planar_bond_index,
                    absolute_gradient,
                    slope,
                )
            )
        gradients.append(row_gradient)
    return torch.cat(gradients)


def _boltz2_scatter_potential_gradient(
    coordinates: Tensor,
    index: Tensor,
    variable_gradient: Tensor,
    energy_slope: Tensor,
) -> Tensor:
    contribution = energy_slope[:, None, :, None] * variable_gradient
    return torch.zeros_like(coordinates).scatter_add(
        1,
        index.flatten()[None, :, None].expand(
            coordinates.shape[0],
            -1,
            3,
        ),
        contribution.flatten(1, 2),
    )


def _boltz2_symmetric_chain_com_gradient(
    coordinates: Tensor,
    potential_rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    t: float,
) -> Tensor:
    batch_size = len(potential_rows)
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        num_samples,
        *coordinates.shape[1:],
    )
    gradients = []
    buffer = 1.0 + 4.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
    for batch_index, row in enumerate(potential_rows):
        row_coordinates = coordinates_by_batch[batch_index]
        _, row_gradient = chain_center_separation(
            row_coordinates,
            row.atom_chain,
            row.atom_mask,
            row.num_chains,
            row.symmetric_chain_index,
            coordinates.new_tensor(buffer),
            coordinates.new_tensor(1.0),
            compute_gradient=True,
        )
        gradients.append(row_gradient)
    return torch.cat(gradients)


def _boltz2_contact_gradient(
    coordinates: Tensor,
    potential_rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    t: float,
) -> Tensor:
    batch_size = len(potential_rows)
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        num_samples,
        *coordinates.shape[1:],
    )
    gradients = []
    union_lambda = 8.0 * (1.0 - math.expm1(-2.0 * t) / math.expm1(-2.0))
    for batch_index, row in enumerate(potential_rows):
        row_coordinates = coordinates_by_batch[batch_index]
        row_gradient = torch.zeros_like(row_coordinates)
        if row.contact_pair_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                row.contact_pair_index,
            )
            energy = linear_flat_bottom_energy(
                distance,
                distance.new_tensor(1.0),
                distance.new_tensor(float("-inf")),
                row.contact_threshold,
            )
            energy_slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(1.0),
                distance.new_tensor(float("-inf")),
                row.contact_threshold,
            )
            coefficient = torch.zeros_like(energy)
            for union_mask in row.contact_union_masks:
                union_energy = energy[:, union_mask]
                centered_energy = union_energy - union_energy.amin(dim=-1, keepdim=True)
                weights = (-union_lambda * centered_energy).softmax(dim=-1)
                centered_mean = (centered_energy * weights).sum(dim=-1, keepdim=True)
                coefficient[:, union_mask] = (
                    weights - union_lambda * weights * (centered_energy - centered_mean)
                ) * energy_slope[:, union_mask]
            row_gradient = _boltz2_scatter_potential_gradient(
                row_coordinates,
                row.contact_pair_index,
                distance_gradient,
                coefficient,
            )
        gradients.append(row_gradient)
    return torch.cat(gradients)


def _boltz2_template_gradient(
    coordinates: Tensor,
    potential_rows: tuple[_Boltz2PotentialRow, ...],
    *,
    num_samples: int,
    backend: str = "auto",
) -> Tensor:
    batch_size = len(potential_rows)
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        num_samples,
        *coordinates.shape[1:],
    )
    gradients = []
    for batch_index, row in enumerate(potential_rows):
        row_coordinates = coordinates_by_batch[batch_index]
        row_gradient = torch.zeros_like(row_coordinates)
        if row.template_reference is not None:
            representative = row_coordinates[:, row.token_representative_atom_index]
            token_gradient = torch.zeros_like(representative)
            for members, delta, threshold in _boltz2_template_frames(row, representative, backend=backend):
                distance = torch.linalg.vector_norm(delta, dim=-1)
                direction = F.normalize(delta, dim=-1, eps=1e-8)
                slope = linear_flat_bottom_derivative(
                    distance,
                    distance.new_tensor(1.0),
                    distance.new_tensor(float("-inf")),
                    threshold,
                )
                token_gradient[:, members] += slope.unsqueeze(-1) * direction
            row_gradient = token_gradient[:, row.atom_to_token]
        gradients.append(row_gradient)
    return torch.cat(gradients)


def _boltz2_template_frames(
    row: _Boltz2PotentialRow,
    representative: Tensor,
    *,
    backend: str,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Align only observations belonging to one established template placement."""
    assert row.template_reference is not None
    assert row.template_reference_mask is not None
    assert row.template_frame_index is not None
    assert row.template_threshold is not None
    for template, reference in enumerate(row.template_reference):
        frame_index = row.template_frame_index[template]
        observed = row.template_reference_mask[template].bool() & (frame_index >= 0)
        for frame in torch.unique(frame_index[observed]):
            members = observed & (frame_index == frame)
            target = representative[:, members]
            source = reference[members].expand(representative.shape[0], -1, -1)
            aligned = weighted_rigid_align(
                source=source,
                target=target,
                weight=torch.ones_like(target[..., 0]),
                backend=backend,
            ).detach()
            yield members, target - aligned, row.template_threshold[template]


def _boltz2_batch_potential_terms(
    coordinates: Tensor,
    row: _Boltz2PotentialRow,
    *,
    t: float,
    include_contact: bool,
    active_physical: frozenset[int] | None,
    backend: str = "auto",
) -> tuple[Tensor, ...]:
    r"""
    Evaluate each potential for one batch row and every sample in that row.
    """

    zero = coordinates.new_zeros(coordinates.shape[0])
    active_physical = frozenset(range(7)) if active_physical is None else active_physical

    if 0 in active_physical and row.symmetric_chain_index.shape[1]:
        symmetric_buffer = 1.0 + 4.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
        symmetric_energy = cast(
            Tensor,
            chain_center_separation(
                coordinates,
                row.atom_chain,
                row.atom_mask,
                row.num_chains,
                row.symmetric_chain_index,
                coordinates.new_tensor(symmetric_buffer),
                coordinates.new_tensor(1.0),
                compute_gradient=False,
            ),
        )
    else:
        symmetric_energy = zero

    if 1 in active_physical and row.vdw_pair_index.shape[1]:
        distance = indexed_distance(coordinates, row.vdw_pair_index)
        vdw_energy = linear_flat_bottom_energy(
            distance,
            distance.new_tensor(1.0),
            row.vdw_lower_bound,
            distance.new_tensor(float("inf")),
        ).sum(dim=-1)
    else:
        vdw_energy = zero

    if 2 in active_physical and row.connected_atom_index.shape[1]:
        distance = indexed_distance(coordinates, row.connected_atom_index)
        connection_energy = linear_flat_bottom_energy(
            distance,
            distance.new_tensor(1.0),
            distance.new_tensor(float("-inf")),
            distance.new_tensor(2.0),
        ).sum(dim=-1)
    else:
        connection_energy = zero

    if 3 in active_physical and row.geometry_pair_index.shape[1]:
        distance = indexed_distance(coordinates, row.geometry_pair_index)
        pose_energy = linear_flat_bottom_energy(
            distance,
            distance.new_tensor(1.0),
            row.geometry_lower_bound,
            row.geometry_upper_bound,
        ).sum(dim=-1)
    else:
        pose_energy = zero

    if 4 in active_physical and row.chiral_atom_index.shape[1]:
        angle = indexed_signed_dihedral(coordinates, row.chiral_atom_index, backend=backend)
        chiral_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            row.chiral_lower_bound,
            row.chiral_upper_bound,
        ).sum(dim=-1)
    else:
        chiral_energy = zero

    if 5 in active_physical and row.stereo_bond_index.shape[1]:
        angle = indexed_signed_dihedral(coordinates, row.stereo_bond_index, backend=backend).abs()
        stereo_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            row.stereo_lower_bound,
            row.stereo_upper_bound,
        ).sum(dim=-1)
    else:
        stereo_energy = zero

    if 6 in active_physical and row.planar_bond_index.shape[1]:
        angle = indexed_signed_dihedral(coordinates, row.planar_bond_index, backend=backend).abs()
        planar_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            angle.new_tensor(float("-inf")),
            angle.new_tensor(0.26180),
        ).sum(dim=-1)
    else:
        planar_energy = zero

    if include_contact and row.contact_pair_index.shape[1]:
        forced_distance = indexed_distance(coordinates, row.contact_pair_index)
        forced_energy = linear_flat_bottom_energy(
            forced_distance,
            forced_distance.new_tensor(1.0),
            forced_distance.new_tensor(float("-inf")),
            row.contact_threshold,
        )
        union_lambda = 8.0 * (1.0 - math.expm1(-2.0 * t) / math.expm1(-2.0))
        contact_energy = zero
        for union_mask in row.contact_union_masks:
            values = forced_energy[:, union_mask]
            minimum_energy = values.amin(dim=-1, keepdim=True)
            centered_energy = values - minimum_energy
            weights = (-union_lambda * centered_energy).softmax(dim=-1)
            contact_energy = contact_energy + minimum_energy.squeeze(-1) + (centered_energy * weights).sum(dim=-1)
    else:
        contact_energy = zero

    template_energy = zero
    if include_contact and row.template_reference is not None:
        representative = coordinates[:, row.token_representative_atom_index]
        for _members, delta, threshold in _boltz2_template_frames(row, representative, backend=backend):
            distance = torch.linalg.vector_norm(delta, dim=-1)
            template_energy = template_energy + linear_flat_bottom_energy(
                distance,
                distance.new_tensor(1.0),
                distance.new_tensor(float("-inf")),
                threshold,
            ).sum(dim=-1)

    return (
        symmetric_energy,
        vdw_energy,
        connection_energy,
        pose_energy,
        chiral_energy,
        stereo_energy,
        planar_energy,
        contact_energy,
        template_energy,
    )


def _boltz2_pair_vdw_radii(reference_elements: Tensor, pair_index: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Read supported radii only for participating, exactly Z-encoded pair endpoints."""
    selected = reference_elements[pair_index.to(device=reference_elements.device)]
    if not pair_index.numel():
        return torch.empty(pair_index.shape, device=pair_index.device, dtype=dtype)
    valid = ((selected == 0) | (selected == 1)).all(dim=-1) & (selected.sum(dim=-1) == 1)
    if not bool(valid.all()):
        fail("value_invalid", RequestRef("reference_elements", "not exact atomic-number one-hot rows"))
    atomic_numbers = selected.argmax(dim=-1)
    if bool((atomic_numbers == 0).any()):
        fail("value_invalid", RequestRef("reference_elements", "atomic number 0"))
    radii = _boltz2_vdw_radii(device=pair_index.device, dtype=dtype)[atomic_numbers.to(device=pair_index.device)]
    if bool((radii <= 0).any()):
        fail("model_capacity", RequestRef("reference_elements", "no van-der-Waals radius"))
    return radii


def _boltz2_vdw_radii(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    radii = torch.zeros(128, device=device, dtype=dtype)
    radii[1:119] = radii.new_tensor(_BOLTZ2_VDW_RADII)
    return radii


_BOLTZ2_VDW_RADII = (
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


def _gather_boltz2_structure_target(
    target: Boltz2StructureTarget,
    input: Boltz2Input,
    *,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather canonical Boltz-2 conformers onto the model atom axis.
    """

    positions = target.coordinates
    if positions.ndim != 4:
        fail("value_invalid", RequestRef("target.coordinates", str(tuple(positions.shape))))
    conformers = positions.shape[1]
    if conformers not in (1, num_samples):
        fail("value_invalid", RequestRef("target.coordinates", str(tuple(positions.shape))))
    if conformers == 1:
        positions = positions.expand(-1, num_samples, -1, -1)

    gathered, resolved = gather_model_coordinates(
        positions,
        insert_sample_axis(target.coordinate_mask, num_samples, dim=-2),
        insert_sample_axis(input.atom_index, num_samples, dim=-2),
        insert_sample_axis(input.atom_mask.to(torch.bool), num_samples, dim=-2),
    )
    return gathered, resolved[:, 0]


def _unflatten_sample_axis(tensor: Tensor, batch_size: int, num_samples: int) -> Tensor:
    r"""
    Split a flattened leading axis ``B*S`` back into an explicit ``(B, S, ...)`` sample axis.

    Args:
        tensor: Tensor whose leading dimension is ``batch_size * num_samples``.
        batch_size: Batch size ``B``.
        num_samples: Sample count ``S``.
    Returns:
        The tensor reshaped to ``(B, S, ...)``.
    """

    return tensor.reshape(batch_size, num_samples, *tensor.shape[1:])


# ===========================================================================
# Boltz-2 public values and tensor boundary
#
# Family-owned scientific evidence precedes the pure-tensor input, output, and
# canonical molecular identity.
# ===========================================================================


@dataclass(frozen=True)
class Boltz2Contact:
    r"""
    Learned Boltz-2 contact conditioning between two token-level endpoints.

    Polymer endpoints are residue occurrences and non-polymer endpoints are exact
    atom occurrences.

    Attributes:
        endpoint_a: First residue or non-polymer atom occurrence.
        endpoint_b: Second residue or non-polymer atom occurrence.
        max_distance: Contact threshold in angstroms.
        force: Whether the condition also contributes potential guidance.
    """

    endpoint_a: ComponentRef | AtomSite
    endpoint_b: ComponentRef | AtomSite
    max_distance: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class Boltz2Pocket:
    r"""
    Learned Boltz-2 pocket conditioning between a binder instance and contact sites.

    Attributes:
        binder: Molecular instance whose complete token set forms the binder side.
        contacts: Polymer residue or non-polymer atom occurrences on the pocket side.
        max_distance: Binder-to-pocket threshold in angstroms.
        force: Whether the pocket also contributes potential guidance.
    """

    binder: str
    contacts: tuple[ComponentRef | AtomSite, ...]
    max_distance: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class Boltz2Template:
    r"""
    Boltz-2 template evidence with optional potential-guidance force.

    Attributes:
        template: Canonical structural template evidence.
        force: Whether each established template placement independently guides coordinate sampling.
        threshold: Flat-bottom template deviation threshold in angstroms.
            ``None`` applies force without a finite cutoff.
    """

    template: Template
    force: bool = False
    threshold: float | None = None


@tensorclass
class Boltz2StructureTarget:
    r"""
    Complete pre-crop structure truth for Boltz-2.

    Attributes:
        coordinates: Complete-assembly conformers with shape
            ``(B, R, A_system, 3)``.
        coordinate_mask: Resolved complete-assembly mask with shape
            ``(B, A_system)``.
        chain_span: Half-open canonical chain spans with shape ``(B, C, 2)``.
        chain_equivalence_index: Actual physical-evidence equivalence ids with
            shape ``(B, C)``.
        symmetry_atom_index: Current-axis atoms in each local symmetry
            group with shape ``(B, G, W)`` and ``-1`` padding.
        symmetry_permutation: Complete local permutations with shape
            ``(B, G, P, W)`` and ``-1`` padding.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    chain_span: Tensor
    chain_equivalence_index: Tensor
    symmetry_atom_index: Tensor
    symmetry_permutation: Tensor


@tensorclass
class Boltz2Input:
    r"""
    Input tensors consumed by [`Boltz2Model`][].

    Attributes:
        token_mask: Boolean token validity mask with shape ``(B, N_token)``.
        residue_index: Chain-local residue index used by relative-position features.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by chains with the same sequence.
        symmetry_index: Symmetry-copy ids for relative-position features.
        token_index: Token ids in structure order.
        atom_features: Per-atom reference features.
        atom_to_token: Token index for each atom.
        atom_mask: Boolean atom validity mask.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id for each atom.
        token_representative_atom_index: Representative atom index for each token.
        residue_type_features: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level physical connectivity indicator.
        type_bonds: Token-level bond type ids.
        contact_conditioning: Pairwise contact-class one-hot features with shape
            ``(B, N, N, 5)``.
        contact_threshold: Pairwise contact thresholds with shape ``(B, N, N)``.
        contact_pair_index: Forced-contact atom pairs with shape ``(B, 2, P)``.
        contact_union_index: Union-group id for each forced atom pair.
        contact_thresholds: Forced-contact upper bounds in angstroms.
        geometry_pair_index: Atom pairs with molecular distance bounds.
        geometry_bond_mask: Mask of directly bonded geometry pairs.
        geometry_angle_mask: Mask of two-bond angle endpoint pairs.
        geometry_upper_bounds: Upper molecular distance bounds.
        geometry_lower_bounds: Lower molecular distance bounds.
        chiral_atom_index: Chiral-center improper-dihedral atom indices.
        chiral_atom_positive: Positive-dihedral flag for each chiral-center constraint.
        stereo_bond_index: Stereo-bond dihedral atom indices.
        stereo_bond_trans: Trans flag for each declared control-atom tuple.
        planar_bond_index: Planar double-bond atom indices.
        connected_chain_index: Chain pairs joined by known covalent assembly bonds.
        connected_atom_index: Atom pairs joined by known covalent assembly bonds.
        symmetric_chain_index: Same-entity chain-copy pairs.
        molecule_type: Molecule type ids in the model vocabulary.
        design_mask: Boolean mask selecting the exact canonical component
            occurrences designed by the calling protocol, with shape ``(B, N)``.
        chain_design_mask: Boolean mask selecting complete chains in the
            covalently connected design component, with shape ``(B, N)``.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        method_feature: Optional experimental-method ids.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        msa_tokens: Optional integer MSA token ids.
        has_deletion: Optional MSA deletion flags.
        deletion_value: Optional MSA deletion values.
        msa_paired: Optional paired-MSA flags.
        msa_mask: Optional MSA validity mask.
        template_residue_type_features: Optional template residue one-hot features with
            shape ``(B, T, N, vocab_size)``.
        template_frame_rotations: Optional template frame rotations with shape
            ``(B, T, N, 3, 3)``.
        template_frame_translations: Optional template frame origins with shape
            ``(B, T, N, 3)``.
        template_frame_mask: Optional template frame validity mask with shape
            ``(B, T, N)``.
        template_pseudo_beta_positions: Optional template pseudo-beta (CB, Gly
            CA) positions with shape ``(B, T, N, 3)``.
        template_alpha_carbon_positions: Optional template CA positions with
            shape ``(B, T, N, 3)``.
        template_pseudo_beta_mask: Optional template pseudo-beta validity mask
            with shape ``(B, T, N)``.
        template_mask: Optional alignment-coverage mask with shape ``(B, T, N)``.
        template_visibility_index: Template visibility groups (upstream
            ``visibility_ids``) with shape ``(B, T, N)`` and int64 dtype: the
            tokens of an instance holding templated tokens carry their placement
            frame (``>= 0``; reused source chains have independent target-copy
            frames), every other instance a negative id of its own.
        template_force: Optional mask of templates used for potential guidance.
        template_force_threshold: Optional per-template guidance thresholds.
        affinity_token_mask: Optional mask selecting the ligand tokens to score for affinity.
        affinity_mw: Optional molecular weight for the selected ligand, used only when affinity MW correction is on.
        affinity_msa_profile: Optional single-sequence profile used by the affinity heads.
        affinity_deletion_mean: Optional deletion mean used by the affinity heads.
        atom_index: Canonical atom index for each model atom (its slot's atom) with shape ``(B, A)`` and ``-1``
            padding.
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
    token_representative_atom_index: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    contact_conditioning: Tensor
    contact_threshold: Tensor
    contact_pair_index: Tensor
    contact_union_index: Tensor
    contact_thresholds: Tensor
    geometry_pair_index: Tensor
    geometry_bond_mask: Tensor
    geometry_angle_mask: Tensor
    geometry_upper_bounds: Tensor
    geometry_lower_bounds: Tensor
    chiral_atom_index: Tensor
    chiral_atom_positive: Tensor
    stereo_bond_index: Tensor
    stereo_bond_trans: Tensor
    planar_bond_index: Tensor
    connected_chain_index: Tensor
    connected_atom_index: Tensor
    symmetric_chain_index: Tensor
    molecule_type: Tensor
    design_mask: Tensor
    chain_design_mask: Tensor
    atom_index: Tensor
    frame_atom_index: Tensor | None = None
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    template_residue_type_features: Tensor | None = None
    template_frame_rotations: Tensor | None = None
    template_frame_translations: Tensor | None = None
    template_frame_mask: Tensor | None = None
    template_pseudo_beta_positions: Tensor | None = None
    template_alpha_carbon_positions: Tensor | None = None
    template_pseudo_beta_mask: Tensor | None = None
    template_mask: Tensor | None = None
    template_visibility_index: Tensor | None = None
    template_force: Tensor | None = None
    template_force_threshold: Tensor | None = None
    affinity_token_mask: Tensor | None = None
    affinity_mw: Tensor | None = None
    affinity_msa_profile: Tensor | None = None
    affinity_deletion_mean: Tensor | None = None


def _boltz2_binary_features(values: Tensor, observed: Tensor, *, feature: str) -> Tensor:
    """Lower exact integer one-hot/binary carriers before continuous arithmetic."""
    if values.is_floating_point():
        return values
    if values.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        fail("value_invalid", RequestRef(feature, str(values.dtype)))
    binary = ((values == 0) | (values == 1) | ~observed).all()
    if torch.compiler.is_compiling():
        torch._assert_async(binary, "Boltz2 integer relation features must be binary")
    elif not bool(binary):
        fail("value_invalid", RequestRef(feature, "not binary"))
    return torch.where(observed, values, 0).to(torch.float32)


def normalize_boltz2_input(input: Boltz2Input, *, dtype: torch.dtype | None = None) -> Boltz2Input:
    """Return finite family features, retaining all canonical/prepared bindings."""
    result = cast(Boltz2Input, cast(TensorClass, input).clone(recurse=False))
    atom_mask = input.atom_mask.to(torch.bool)
    token_mask = input.token_mask.to(torch.bool)
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    result.reference_positions = project_model_coordinates(
        input.reference_positions, atom_mask, dtype=dtype, feature="Boltz2 reference positions"
    )
    for name, mask in (
        ("atom_features", atom_mask[..., None]),
        ("residue_type_features", token_mask[..., None]),
        ("msa_profile", token_mask[..., None]),
        ("deletion_mean", token_mask),
        ("token_bonds", pair_mask),
        ("contact_conditioning", pair_mask[..., None]),
        ("contact_threshold", pair_mask),
        ("affinity_msa_profile", token_mask[..., None]),
        ("affinity_deletion_mean", token_mask),
    ):
        values = getattr(input, name)
        if values is not None:
            if name in {"contact_conditioning", "token_bonds", "residue_type_features"}:
                values = _boltz2_binary_features(values, mask, feature=f"Boltz2 {name}")
            setattr(result, name, project_model_features(values, mask, dtype=dtype, feature=f"Boltz2 {name}"))
    if input.msa_tokens is not None:
        msa_available = token_mask.unsqueeze(-2) & torch.ones_like(input.msa_tokens, dtype=torch.bool)
        if input.msa_mask is not None:
            msa_available = msa_available & input.msa_mask.bool()
        result.msa_mask = msa_available
        result.msa_tokens = project_model_categories(
            input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="Boltz2 MSA tokens"
        )
        for name in ("has_deletion", "deletion_value", "msa_paired"):
            values = getattr(input, name)
            if values is not None:
                if not values.is_floating_point():
                    values = torch.where(msa_available, values, 0).to(torch.float64)
                setattr(
                    result, name, project_model_features(values, msa_available, dtype=dtype, feature=f"Boltz2 {name}")
                )
    if input.template_mask is not None:
        # an untemplated token keeps its <pad> class (upstream's class 0) and joins its chain's geometry, as upstream
        template_mask = input.template_mask.to(torch.bool) & token_mask.unsqueeze(-2)
        pseudo_mask = cast(Tensor, input.template_pseudo_beta_mask).to(torch.bool) & template_mask
        for name, mask in (
            ("template_residue_type_features", token_mask.unsqueeze(-2)[..., None]),
            ("template_frame_rotations", template_mask[..., None, None]),
        ):
            values = cast(Tensor, getattr(input, name))
            if name == "template_residue_type_features":
                values = _boltz2_binary_features(values, mask, feature="Boltz2 template residue features")
            setattr(result, name, project_model_features(values, mask, dtype=dtype, feature=f"Boltz2 {name}"))
        for name, mask in (
            ("template_frame_translations", template_mask),
            ("template_alpha_carbon_positions", template_mask),
            ("template_pseudo_beta_positions", pseudo_mask),
        ):
            setattr(
                result,
                name,
                project_model_coordinates(
                    cast(Tensor, getattr(input, name)), mask, dtype=dtype, feature=f"Boltz2 {name}"
                ),
            )
    return result


class _Boltz2TrunkOutput(NamedTuple):
    r"""
    Trunk outputs consumed by the diffusion / confidence heads.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


class Boltz2Representations(TensorCollection):
    r"""
    Raw Boltz-2 trunk representations surfaced by ``Boltz2Model.representations``.

    Attributes:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor


class Boltz2Output(TensorCollection):
    r"""
    Predictions from one Boltz-2 training denoising pass.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        augmented_target_coordinates: Training target coordinates after rigid
            augmentation, with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
        b_factor_logits: Optional token-level B-factor logits with shape
            ``(B, N, D_b_factor)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor
    b_factor_logits: OptionalTensor


class Boltz2SamplingOutput(TensorCollection):
    r"""
    Predictions from one Boltz-2 reverse-diffusion sampling pass.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    coordinates: Tensor


class Boltz2AffinityOutput(TensorCollection):
    r"""
    Raw differentiable Boltz-2 affinity predictions with explicit sample axes.

    All fields have shape ``(B, S, 1)``. The ensemble probability and value are
    followed by the two head-specific values, logits, and probabilities.
    """

    affinity_pred_value: Tensor
    affinity_probability_binary: Tensor
    affinity_pred_value1: Tensor
    affinity_logits_binary1: Tensor
    affinity_probability_binary1: Tensor
    affinity_pred_value2: Tensor
    affinity_logits_binary2: Tensor
    affinity_probability_binary2: Tensor


class Boltz2ConfidenceOutput(TensorCollection):
    r"""
    Raw predictions from Boltz-2 coordinate sampling and confidence evaluation.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        plddt_logits: Per-token confidence logits with shape
            ``(B, S, N, D_plddt)``.
        pae_logits: Token-pair aligned-error logits with shape
            ``(B, S, N, N, D_pae)``.
        pde_logits: Token-pair distance-error logits with shape
            ``(B, S, N, N, D_pde)``.
        resolved_logits: Per-token logits in ``[unresolved, resolved]`` order
            with shape ``(B, S, N, 2)``.
        b_factor_logits: Optional per-token B-factor logits with shape
            ``(B, N, D_b_factor)``.
        plddt: Expected pLDDT values on the ``0--100`` scale with shape
            ``(B, S, N)``.
        pae: Expected PAE in angstroms with shape ``(B, S, N, N)``.
        pde: Expected PDE in angstroms with shape ``(B, S, N, N)``.
        resolved_probability: Probability of resolved class one with shape
            ``(B, S, N)``.
        mean_plddt: Mean complex pLDDT with shape ``(B, S)``.
        complex_iplddt: Interface-weighted pLDDT with shape ``(B, S)``.
        complex_pde: Distogram-contact-weighted PDE with shape ``(B, S)``.
        complex_ipde: Interface distogram-contact-weighted PDE with shape
            ``(B, S)``.
        ptm: Predicted TM score with shape ``(B, S)``.
        iptm: Interface predicted TM score with shape ``(B, S)``.
        ligand_iptm: Protein-ligand predicted TM score with shape ``(B, S)``.
        protein_iptm: Protein-protein predicted TM score with shape ``(B, S)``.
        chain_pair_iptm: Directed chain-pair predicted TM scores with shape
            ``(B, S, C, C)`` and ``[source/frame chain, target chain]`` axes. C is
            max canonical chain ID plus one; unused sparse IDs retain carrier rows.
        design_to_target_iptm: Exact designed-component to target predicted TM
            score with shape ``(B, S)``.
        design_residue_iptm: Predicted TM score between exact designed residues
            and tokens in other chains, with shape ``(B, S)``.
        design_iptm: Predicted TM score between the covalently propagated design
            component and target, with shape ``(B, S)``.
        design_iiptm: Predicted TM score between target and the spatially
            interacting part of the design component, with shape ``(B, S)``.
        target_ptm: Target-only predicted TM score with shape ``(B, S)``.
        design_ptm: Design-component-only predicted TM score with shape
            ``(B, S)``.
        interaction_pae: Mean expected PAE between exact designed components
            and target, with shape ``(B, S)``.
        min_interaction_pae: Minimum expected PAE between the covalently
            propagated design component and target, with shape ``(B, S)``.
        min_design_to_target_pae: Minimum expected PAE between exact designed
            components and target, with shape ``(B, S)``.
        design_ipsae_min: Minimum of the two directional design/target iPSAE
            scores with shape ``(B, S)``.
        design_to_target_ipsae: Design-to-target iPSAE with shape ``(B, S)``.
        target_to_design_ipsae: Target-to-design iPSAE with shape ``(B, S)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor
    b_factor_logits: OptionalTensor
    plddt: Tensor
    pae: Tensor
    pde: Tensor
    resolved_probability: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    ptm: Tensor
    iptm: Tensor
    ligand_iptm: Tensor
    protein_iptm: Tensor
    chain_pair_iptm: Tensor
    design_to_target_iptm: Tensor
    design_residue_iptm: Tensor
    design_iptm: Tensor
    design_iiptm: Tensor
    target_ptm: Tensor
    design_ptm: Tensor
    interaction_pae: Tensor
    min_interaction_pae: Tensor
    min_design_to_target_pae: Tensor
    design_ipsae_min: Tensor
    design_to_target_ipsae: Tensor
    target_to_design_ipsae: Tensor


class Boltz2AffinityScoreOutput(TensorCollection):
    r"""
    Raw output of the complete Boltz-2 affinity scoring protocol.

    Attributes:
        structure: Candidate structures and confidence predictions.
        affinity: Affinity values for the selected candidate.
        selected_sample_index: Highest-iPTM candidate index with shape ``(B,)``.
    """

    structure: Boltz2ConfidenceOutput
    affinity: Boltz2AffinityOutput
    selected_sample_index: Tensor


def _affinity_output(values: dict[str, Tensor]) -> Boltz2AffinityOutput:
    r"""Pack named affinity-head tensors into their public raw carrier."""

    return Boltz2AffinityOutput(
        affinity_pred_value=values["affinity_pred_value"],
        affinity_probability_binary=values["affinity_probability_binary"],
        affinity_pred_value1=values["affinity_pred_value1"],
        affinity_logits_binary1=values["affinity_logits_binary1"],
        affinity_probability_binary1=values["affinity_probability_binary1"],
        affinity_pred_value2=values["affinity_pred_value2"],
        affinity_logits_binary2=values["affinity_logits_binary2"],
        affinity_probability_binary2=values["affinity_probability_binary2"],
        batch_size=[values["affinity_pred_value"].shape[0]],
    )


def _boltz2_coordinates(coordinates: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Project a complete model-state coordinate field at a public boundary."""
    if coordinates.dim() == 0 or coordinates.shape[-1] != 3:
        fail("value_invalid", RequestRef("coordinates", str(tuple(coordinates.shape))))
    return project_model_coordinates(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        dtype=dtype,
        feature="Boltz2 model-state coordinates",
    )
