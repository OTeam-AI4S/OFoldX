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
Boltz-1 model.
"""

from __future__ import annotations

import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.molecular import ComponentRef
from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...metrics import (
    categorical_mean,
    chain_pair_tm_scores,
    interface_confidence_weights,
    plddt_from_logits,
    predicted_tm_score,
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
from ...modules.diffusion.conditioning import SingleConditioning
from ...modules.diffusion.edm import EdmPreconditioner, edm_loss_weight, lognormal_training_noise
from ...modules.diffusion.sampling import draw_normal, draw_normal_like
from ...modules.diffusion.schedule import karras_noise_schedule
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask, pairwise_distance, weighted_mean
from ...modules.geometry import rotate_vectors, uniform_random_rotation, weighted_rigid_align
from ...modules.head.confidence import TokenConfidenceOutputHead
from ...modules.head.distogram import DistogramHead
from ...modules.msa import MsaPairBlock
from ...modules.pair import LineDropoutTrianglePairUpdateBlock, TrianglePairUpdateStack
from ...modules.pairformer import LineDropoutPairformerBlock, PairformerStack
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
)
from ...modules.transition import TripleProductConditionedTransition
from ..processing import (
    gather_model_coordinates,
    project_model_categories,
    project_model_coordinates,
    project_model_features,
)
from ..vocabulary import folding_source_map, map_token_feature_planes
from .config import Boltz1Config

_Boltz1PotentialArguments: TypeAlias = tuple[Tensor, Tensor | None, Tensor | None]
_Boltz1PotentialLowering: TypeAlias = tuple[
    Tensor,
    tuple[Tensor, ...],
    tuple[Tensor, Tensor, int] | None,
]
_Boltz1ResolvedPotential: TypeAlias = tuple[
    Tensor,
    _Boltz1PotentialArguments,
    tuple[Tensor, Tensor, int] | None,
]


class Boltz1Model(Model):
    r"""
    Predict biomolecular complex structures with Boltz 1.

    Structure training consumes separate [`Boltz1Input`][] and
    [`Boltz1StructureTarget`][] values. ``sample_with_confidence`` executes the inference graph and
    returns raw tensors; [`Boltz1Pipeline`][ofoldx.models.boltz1.Boltz1Pipeline]
    prepares inputs and decodes those tensors into public results.

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = Boltz1Config
    config: Boltz1Config

    def __init__(self, config: Boltz1Config | None = None) -> None:
        config = Boltz1Config() if config is None else config
        super().__init__(config)
        self._num_recycles = config.num_recycles

        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config = config.input_embeddings
        input_size = (
            single_size
            + input_embeddings_config.vocab_size
            + len(TOKENS)
            + 1
            + input_embeddings_config.pocket_feature_size
        )
        trunk = config.trunk
        denoiser_config = config.denoiser

        self.input_embeddings = Boltz1InputEmbeddings(
            input_size=input_size,
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
            use_atom_encoder=input_embeddings_config.use_atom_encoder,
            backend="auto",
        )

        self.single_init_projection = nn.Linear(input_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)

        self.single_recycle_norm = LayerNorm(single_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        self.msa_trunk: Boltz1MsaTrunk | None = None
        if config.use_msa:
            self.msa_trunk = Boltz1MsaTrunk(
                single_size=input_size,
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
        self.pairformer_stack: PairformerStack | None = None
        self.pair_update_stack: TrianglePairUpdateStack | None = None
        if trunk.update_single:
            self.pairformer_stack = PairformerStack(
                block=LineDropoutPairformerBlock,
                num_blocks=trunk.num_pairformer_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_head_size=trunk.triangle_attention_head_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
        else:
            self.pair_update_stack = TrianglePairUpdateStack(
                block=LineDropoutTrianglePairUpdateBlock,
                num_blocks=trunk.num_pairformer_blocks,
                pair_size=pair_size,
                triangle_attention_head_size=trunk.triangle_attention_head_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
        self.distogram_head = DistogramHead(
            pair_size=pair_size,
            num_bins=config.num_distogram_bins,
        )

        self.denoiser_conditioning = AtomDenoiserConditioning(
            atom_feature_bias=False,
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
        denoiser = Boltz1Denoiser(
            input_size=input_size,
            single_size=single_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
            backend="auto",
        )
        sampler = config.sampler
        self.coordinate_diffusion = Boltz1CoordinateDiffusion(
            denoiser,
            sigma_data=denoiser_config.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            min_churn_noise_level=sampler.min_churn_noise_level,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=denoiser_config.alignment_reverse_diff,
            accumulate_token_repr=config.confidence.enabled and config.confidence.use_single_diffusion,
            token_size=single_size,
            fourier_embedding_size=config.fourier_embedding_size,
        )

        confidence = config.confidence
        self.confidence_head: Boltz1TrunkImitationConfidenceHead | Boltz1TrunkReuseConfidenceHead | None = None
        if confidence.enabled:
            if confidence.imitate_trunk:
                self.confidence_head = Boltz1TrunkImitationConfidenceHead(
                    input_size=input_size,
                    single_size=single_size,
                    pair_size=pair_size,
                    atom_hidden_size=config.atom_hidden_size,
                    atom_pair_hidden_size=config.atom_pair_hidden_size,
                    atom_feature_size=config.atom_feature_size,
                    num_query_atoms=denoiser_config.num_query_atoms,
                    num_key_atoms=denoiser_config.num_key_atoms,
                    num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
                    num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
                    num_blocks=trunk.num_pairformer_blocks,
                    num_pair_attention_heads=trunk.num_pair_attention_heads,
                    num_distogram_bins=confidence.num_distogram_bins,
                    max_distance=confidence.max_distance,
                    num_plddt_bins=confidence.num_plddt_bins,
                    num_pae_bins=confidence.num_pae_bins,
                    num_pde_bins=confidence.num_pde_bins,
                    compute_pae=confidence.compute_pae,
                    add_single_to_pair_product=confidence.add_single_to_pair_product,
                    use_single_diffusion=confidence.use_single_diffusion,
                    ranking_plddt_weight=confidence.ranking_plddt_weight,
                    ranking_ptm_weight=confidence.ranking_ptm_weight,
                    use_atom_encoder=input_embeddings_config.use_atom_encoder,
                    use_msa=config.use_msa,
                    msa_size=trunk.msa_size,
                    use_paired_feature=trunk.use_paired_feature,
                    num_msa_blocks=trunk.num_msa_blocks,
                    outer_product_hidden_size=trunk.outer_product_hidden_size,
                    msa_dropout=trunk.msa_dropout,
                    msa_pair_dropout=trunk.msa_pair_dropout,
                    pair_dropout=trunk.pair_dropout,
                    update_single=trunk.update_single,
                    num_msa_attention_heads=trunk.num_msa_attention_heads,
                    msa_attention_head_size=trunk.msa_attention_head_size,
                    msa_triangle_attention_head_size=trunk.msa_triangle_attention_head_size,
                    num_msa_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
                    triangle_attention_head_size=trunk.triangle_attention_head_size,
                    num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                    relative_position_size=config.relative_position_size,
                    backend="auto",
                )
            else:
                self.confidence_head = Boltz1TrunkReuseConfidenceHead(
                    input_size=input_size,
                    single_size=single_size,
                    pair_size=pair_size,
                    num_blocks=confidence.num_pairformer_blocks,
                    num_pair_attention_heads=confidence.num_pair_attention_heads,
                    triangle_attention_head_size=confidence.triangle_attention_head_size,
                    num_triangle_attention_heads=confidence.num_triangle_attention_heads,
                    num_distogram_bins=confidence.num_distogram_bins,
                    max_distance=confidence.max_distance,
                    num_plddt_bins=confidence.num_plddt_bins,
                    num_pae_bins=confidence.num_pae_bins,
                    num_pde_bins=confidence.num_pde_bins,
                    compute_pae=confidence.compute_pae,
                    add_single_to_pair_product=confidence.add_single_to_pair_product,
                    add_single_input_to_single=confidence.add_single_input_to_single,
                    add_pair_input_to_pair=confidence.add_pair_input_to_pair,
                    use_single_diffusion=confidence.use_single_diffusion,
                    ranking_plddt_weight=confidence.ranking_plddt_weight,
                    ranking_ptm_weight=confidence.ranking_ptm_weight,
                    pair_dropout=confidence.pair_dropout,
                    update_single=confidence.update_single,
                    relative_position_size=config.relative_position_size,
                    backend="auto",
                )
        self.post_init()

    def forward(
        self,
        input: Boltz1Input,
        target: Boltz1StructureTarget,
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
    ) -> Boltz1Output:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model-visible structure and evidence features.
            target: Full-axis canonical structure truth.
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
            generator: Optional generator for MSA row sampling, rigid
                augmentation, log-normal noise levels, and Gaussian coordinate
                noise.

        Returns:
            Denoised and augmented target coordinates with shape ``(B, S, A, 3)``,
                noise levels with shape ``(B, S)`` in angstroms, and distogram logits
                with shape ``(B, N, N, D)``.
        """
        input = normalize_boltz1_input(input, dtype=self.single_init_projection.weight.dtype)

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
        conditioning = self._build_denoiser_conditioning(representations, input)
        target_coordinates, target_mask = _gather_boltz1_structure_target(
            target,
            input,
        )
        denoised_coordinates, augmented_target_coordinates, noise_level = self.coordinate_diffusion.denoise_training(
            target_coordinates,
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
        batch_size = input.token_mask.shape[0]
        return Boltz1Output(
            coordinates=denoised_coordinates,
            distogram_logits=representations.distogram_logits,
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=noise_level,
            batch_size=[batch_size],
        )

    def confidence(
        self,
        input: Boltz1Input,
        *,
        num_samples: int = 1,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1ConfidenceOutput:
        r"""
        Sample coordinates and return confidence logits and values with explicit sample axes.

        The score-network and coordinate trajectory are detached. When configured,
        the diffusion token accumulator remains differentiable into the confidence
        head.
        """

        if self.confidence_head is None:
            raise AttributeError("this Boltz1Model has no confidence head")
        output = self.sample_with_confidence(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            generator=generator,
        )
        return cast(Boltz1ConfidenceOutput, output.confidence)

    def representations(
        self,
        input: Boltz1Input,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Representations:
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
        input = normalize_boltz1_input(input, dtype=self.single_init_projection.weight.dtype)

        representations = self._run_trunk(
            input,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        return Boltz1Representations(
            single=representations.single,
            pair=representations.pair,
            batch_size=[representations.single.shape[0]],
        )

    def sample(
        self,
        input: Boltz1Input,
        *,
        num_samples: int = 1,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        generator: torch.Generator | None = None,
    ) -> Boltz1SamplingOutput:
        r"""
        Sample atom coordinates with explicit ``(B, S, A, 3)`` axes.

        Args:
            input: Model-visible structure and evidence features.
            num_samples: Number of coordinate samples.
            num_recycles: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses
                the configured value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            use_potentials: Enable the released Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle
                resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.
            generator: Optional generator for MSA row sampling, initial
                coordinates, churn noise, rigid augmentations, and
                potential-guided resampling.

        Returns:
            Sampled atom coordinates with shape ``(B, S, A, 3)``.
        """
        input = normalize_boltz1_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        with torch.no_grad():
            representations = self._run_trunk(
                input,
                num_recycles=num_recycles,
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
        coordinates, _ = self._sample_diffusion(
            representations,
            input,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            generator=generator,
            return_token_repr=False,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
        )
        return Boltz1SamplingOutput(
            coordinates=coordinates,
            batch_size=[coordinates.shape[0]],
        )

    def sample_with_confidence(
        self,
        input: Boltz1Input,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> Boltz1SamplingWithConfidenceOutput:
        r"""
        Run the trunk, coordinate sampler, and confidence head.

        Runs the recycling trunk once, draws ``num_samples`` reverse-diffusion samples, and (when the
        confidence head is present) scores them into confidence logits and ranking metrics. Coordinate
        tensors retain an explicit ``(batch, num_samples, ...)`` shape throughout diffusion.

        Args:
            input: The Boltz-1 input batch.
            num_recycles: Total number of trunk passes; defaults to the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            num_samples: Number of reverse-diffusion samples to draw per input.
            generator: Optional RNG for reproducible sampling noise.
            use_potentials: Enable the released Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle
                resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            A [`Boltz1SamplingWithConfidenceOutput`][]
                with sampled coordinates, distogram logits, and optional typed
                confidence predictions.

        """
        input = normalize_boltz1_input(input, dtype=self.single_init_projection.weight.dtype)

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        with torch.no_grad():
            representations = self._run_trunk(
                input,
                num_recycles=num_recycles,
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
        diffusion_coords, diffusion_token_repr = self._sample_diffusion(
            representations,
            input,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            generator=generator,
            return_token_repr=self.confidence_head is not None and self.config.confidence.use_single_diffusion,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
        )
        batch_size = representations.single.shape[0]
        coordinates = diffusion_coords.detach()
        distogram_logits = representations.distogram_logits.detach()
        confidence_output: Boltz1ConfidenceOutput | None = None
        confidence_head = self.confidence_head
        if confidence_head is not None:
            confidence_logits, distances = confidence_head(
                representations.single_inputs.detach(),
                representations.single.detach(),
                representations.pair.detach(),
                diffusion_coords.detach(),
                input=input,
                single_diffusion=diffusion_token_repr,
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
            confidence_values = dict(confidence_logits)
            confidence_values.update(
                _boltz1_confidence_scores(
                    confidence_logits,
                    distances=distances,
                    input=input,
                    pred_distogram_logits=distogram_logits,
                    coordinates=diffusion_coords,
                    ranking_plddt_weight=confidence_head.ranking_plddt_weight,
                    ranking_ptm_weight=confidence_head.ranking_ptm_weight,
                    backend=self.backend,
                )
            )
            confidence_output = Boltz1ConfidenceOutput(
                coordinates=coordinates,
                distogram_logits=distogram_logits,
                plddt_logits=confidence_values["plddt_logits"],
                pde_logits=confidence_values["pde_logits"],
                resolved_logits=confidence_values["resolved_logits"],
                plddt=confidence_values["plddt"],
                pde=confidence_values["pde"],
                mean_plddt=confidence_values["mean_plddt"],
                complex_iplddt=confidence_values["complex_iplddt"],
                complex_pde=confidence_values["complex_pde"],
                complex_ipde=confidence_values["complex_ipde"],
                confidence_score=confidence_values["confidence_score"],
                **{
                    name: value
                    for name in (
                        "pae_logits",
                        "pae",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "chain_pair_iptm",
                    )
                    if (value := confidence_values.get(name)) is not None
                },
                batch_size=[batch_size],
            )
        return Boltz1SamplingWithConfidenceOutput(
            coordinates=coordinates,
            distogram_logits=distogram_logits,
            confidence=confidence_output,
        )

    def _run_trunk(
        self,
        input: Boltz1Input,
        *,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _Boltz1TrunkOutput:
        r"""
        Iterate the recycling loop: recycle-project the running single/pair, fold in the MSA module (when
        present), and run the Pairformer for ``num_recycles`` total passes.

        Args:
            input: The Boltz-1 input batch.
            num_recycles: Total number of trunk passes; defaults to the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            generator: Optional generator advanced by each MSA row sample.

        Returns:
            The final trunk [`_Boltz1TrunkOutput`][ofoldx.models.boltz1.model._Boltz1TrunkOutput].

        """

        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        num_recycles = operator.index(num_recycles)
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(input)
        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(num_recycles):
            grad_enabled = torch.is_grad_enabled() and cycle == num_recycles - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.msa_trunk is not None and input.msa_tokens is not None:
                    pair = pair + cast(Boltz1MsaTrunk, self.msa_trunk)(
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
                if self.pairformer_stack is not None:
                    single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
                else:
                    pair = cast(TrianglePairUpdateStack, self.pair_update_stack)(
                        pair,
                        pair_mask=pair_mask,
                    )

        return _Boltz1TrunkOutput(
            single_inputs=single_inputs,
            single=cast(Tensor, single),
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
        )

    def _initial_representations(self, input: Boltz1Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Embed inputs and build the pre-recycle single/pair initializations.

        Runs the input embedder, projects single-inputs into the single and (outer-sum) pair tracks, and adds
        the relative-position encoding and token-bond bias to the pair init.

        Args:
            input: The Boltz-1 input batch.

        Returns:
            Tuple ``(single_inputs, single_init, pair_init, relative_position_encoding)``.
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
            pocket_feature=input.pocket_feature,
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
            chain_gate="not_same_chain",
            dtype=single_init.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        return single_inputs, single_init, pair_init, relative_position_encoding

    def _sample_diffusion(
        self,
        representations: _Boltz1TrunkOutput,
        input: Boltz1Input,
        *,
        num_sampling_steps: int | None,
        step_scale: float | None,
        num_samples: int,
        generator: torch.Generator | None,
        return_token_repr: bool,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> tuple[Tensor, Tensor | None]:
        r"""
        Precompute the diffusion conditioning and run the EDM sampler.

        Args:
            representations: The trunk representations from
                [`_run_trunk`][ofoldx.models.boltz1.model.Boltz1Model._run_trunk].
            input: The Boltz-1 input batch.
            num_sampling_steps: Number of reverse-diffusion steps; defaults to the sampler config.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            num_samples: Number of samples (sampler ``num_samples``).
            generator: Optional RNG for reproducible sampling noise.
            return_token_repr: Whether to accumulate diffusion token features
                for the confidence head.
            use_potentials: Enable the released Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle
                resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            Sampled coords and, when confidence single-diffusion is enabled, the accumulated token representation.
        """

        with torch.no_grad():
            conditioning = self._build_denoiser_conditioning(
                representations,
                input,
            )
        sample = self.coordinate_diffusion._sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            generator=generator,
            return_token_repr=return_token_repr,
            feynman_kac=feynman_kac or use_potentials,
            physical_guidance=physical_guidance or use_potentials,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
            feats=(
                _boltz1_steering_features(input, atom_features=input.atom_features)
                if use_potentials or feynman_kac or physical_guidance
                else None
            ),
        )
        if return_token_repr:
            return cast(tuple[Tensor, Tensor], sample)
        return cast(Tensor, sample), None

    def _build_denoiser_conditioning(
        self,
        representations: _Boltz1TrunkOutput,
        input: Boltz1Input,
    ) -> AtomDenoiserContext:
        r"""
        Precompute the per-structure diffusion conditioning reused by every denoiser call.

        The conditioning depends only on the trunk representations and reference geometry, so it is computed
        once and shared by the training single-step denoise and every reverse-diffusion sampling step.

        Args:
            representations: The trunk representations from
                [`_run_trunk`][ofoldx.models.boltz1.model.Boltz1Model._run_trunk].
            input: The Boltz-1 input batch.

        Returns:
            The precomputed atom states, per-block attention biases, and local window layout.
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


class Boltz1InputEmbeddings(nn.Module):
    r"""
    Build the Boltz-1 token input embedding from atom-attention outputs and token-level features.
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
        use_atom_encoder: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.single_size = single_size
        self.windowing: LocalAtomWindowing | None = None
        self.atom_pair_encoder: AtomPairEncoder | None = None
        self.atom_feature_projection: nn.Linear | None = None
        self.atom_pair_bias_projection: PerBlockPairBiasProjection | None = None
        self.atom_encoder: AtomToTokenEncoder | None = None
        if use_atom_encoder:
            self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
            self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
            self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)
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
        pocket_feature: Tensor | None = None,
    ) -> Tensor:
        feature_dtype = None if self.atom_feature_projection is None else self.atom_feature_projection.weight.dtype
        reference_positions = project_model_coordinates(
            reference_positions,
            atom_mask.to(torch.bool),
            dtype=feature_dtype,
            feature="Boltz1 embedding references",
        )
        atom_features = project_model_features(
            atom_features,
            atom_mask.to(torch.bool)[..., None],
            dtype=feature_dtype,
            feature="Boltz1 embedding atom features",
        )
        residue_type_features = project_model_features(
            residue_type_features,
            torch.ones_like(residue_type_features, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz1 complete embedding residue features",
        )
        msa_profile = project_model_features(
            msa_profile,
            torch.ones_like(msa_profile, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz1 complete embedding MSA profile",
        )
        deletion_mean = project_model_features(
            deletion_mean,
            torch.ones_like(deletion_mean, dtype=torch.bool),
            dtype=feature_dtype,
            feature="Boltz1 complete embedding deletion features",
        )
        if pocket_feature is not None:
            pocket_feature = project_model_features(
                pocket_feature,
                torch.ones_like(pocket_feature, dtype=torch.bool),
                dtype=feature_dtype,
                feature="Boltz1 complete embedding pocket features",
            )
        atom_encoder = self.atom_encoder
        if atom_encoder is None:
            token_from_atom = torch.zeros_like(residue_type_features[..., :1]).expand(
                *([-1] * (residue_type_features.dim() - 1)),
                self.single_size,
            )
        else:
            geometry = cast(LocalAtomWindowing, self.windowing)(reference_positions, reference_space_id, atom_mask)
            atom_single = cast(nn.Linear, self.atom_feature_projection)(atom_features)
            atom_pair = cast(AtomPairEncoder, self.atom_pair_encoder)(atom_single, geometry)
            token_from_atom = atom_encoder(
                atom_single,
                cast(PerBlockPairBiasProjection, self.atom_pair_bias_projection)(atom_pair),
                atom_mask,
                atom_to_token,
                residue_type_features,
                geometry.window_info,
            )
        pocket_size = self.input_size - self.single_size - residue_type_features.shape[-1] - msa_profile.shape[-1] - 1
        if pocket_feature is None:
            pocket_reference = residue_type_features[..., :1]
            pocket_feature = torch.cat(
                (
                    torch.ones_like(pocket_reference),
                    torch.zeros_like(pocket_reference).expand(
                        *([-1] * (pocket_reference.dim() - 1)),
                        pocket_size - 1,
                    ),
                ),
                dim=-1,
            )
        single = torch.cat(
            [
                token_from_atom,
                residue_type_features.to(token_from_atom.dtype),
                msa_profile.to(token_from_atom.dtype),
                deletion_mean.unsqueeze(-1).to(token_from_atom.dtype),
                pocket_feature.to(token_from_atom.dtype),
            ],
            dim=-1,
        )
        return single


class Boltz1MsaTrunk(nn.Module):
    r"""
    Build Boltz-1 MSA features and refine the pair representation.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 4,
        use_paired_feature: bool = False,
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
            msa_tokens, msa_mask, num_classes=len(TOKENS), feature="Boltz1 MSA tokens"
        )
        has_deletion = project_model_features(
            torch.where(msa_mask, has_deletion, 0).to(torch.float64),
            msa_mask,
            dtype=pair.dtype,
            feature="Boltz1 deletion presence",
        )
        deletion_value = project_model_features(
            deletion_value, msa_mask, dtype=pair.dtype, feature="Boltz1 deletion feature"
        )
        if self.use_paired_feature:
            is_paired = project_model_features(
                torch.where(msa_mask, is_paired, 0).to(torch.float64),
                msa_mask,
                dtype=pair.dtype,
                feature="Boltz1 MSA pairing",
            )
        if msa_max_depth is not None:
            msa_max_depth = operator.index(msa_max_depth)
            if msa_max_depth < 1:
                fail("value_invalid", RequestRef("msa_max_depth", str(msa_max_depth)))
            random_device = msa_tokens.device if generator is None else generator.device
            indices = torch.randperm(
                msa_tokens.shape[-2],
                device=random_device,
                generator=generator,
            )[
                :msa_max_depth
            ].to(msa_tokens.device)
            msa_tokens = msa_tokens[..., indices, :]
            has_deletion = has_deletion[..., indices, :]
            deletion_value = deletion_value[..., indices, :]
            is_paired = is_paired[..., indices, :]
            msa_mask = msa_mask[..., indices, :]
        features = [
            F.one_hot(msa_tokens, num_classes=len(TOKENS)).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(-3)
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


class Boltz1Denoiser(nn.Module):
    r"""
    Boltz-1 EDM score network with the wider Boltz-1 ``s_inputs`` conditioning width.
    """

    token_feature_map: Tensor

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 8,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.encoded_size = single_size
        feature_map = folding_source_map()
        self.register_buffer("token_feature_map", feature_map, persistent=False)
        source_input_size = input_size - 2 * (len(TOKENS) - feature_map.shape[0])
        self.single_conditioner = SingleConditioning(
            single_size + source_input_size,
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
            block=ConditionedTransformerEncoderBlock,
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
        return_token_repr: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        noisy_positions = project_model_coordinates(
            noisy_positions,
            atom_mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(noisy_positions[..., 0], dtype=torch.bool),
            dtype=self.atom_encoder.noisy_position_projection.weight.dtype,
            feature="Boltz1 noisy positions",
        )
        source_inputs = map_token_feature_planes(single_inputs, self.token_feature_map, self.encoded_size)
        single = self.single_conditioner(noise_level, single_trunk, source_inputs)
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
        coordinate_update = self.atom_decoder(
            token_repr,
            atom_query,
            conditioning.atom_conditioning,
            conditioning.atom_decoder_bias,
            atom_mask,
            atom_to_token,
            conditioning.window_info,
        )
        if return_token_repr:
            return coordinate_update, token_repr
        return coordinate_update


class Boltz1CoordinateDiffusion(nn.Module):
    r"""
    Boltz-1 EDM sampler that can accumulate and surface the final diffusion token representation.
    """

    def __init__(
        self,
        denoiser: Boltz1Denoiser,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 8.0,
        churn_rate: float = 0.605,
        min_churn_noise_level: float = 1.107,
        noise_scale: float = 0.901,
        step_scale: float = 1.638,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        *,
        accumulate_token_repr: bool = False,
        token_size: int = 384,
        fourier_embedding_size: int = 256,
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
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std
        self.output_token_feature_update: Boltz1OutputTokenFeatureUpdate | None = None
        if accumulate_token_repr:
            self.output_token_feature_update = Boltz1OutputTokenFeatureUpdate(
                token_size=token_size, fourier_embedding_size=fourier_embedding_size
            )

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

    def preconditioned_network_forward(
        self,
        noisy_coordinates: Tensor,
        sigma: Tensor | float,
        *,
        return_token_repr: bool = False,
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        atom_mask = cast(Tensor, conditioning["atom_mask"]).to(torch.bool)
        noisy_coordinates = project_model_coordinates(
            noisy_coordinates,
            atom_mask.unsqueeze(-2) & torch.ones_like(noisy_coordinates[..., 0], dtype=torch.bool),
            dtype=cast(Tensor, conditioning["single_trunk"]).dtype,
            feature="Boltz1 noisy coordinates",
        )
        batch_size = noisy_coordinates.shape[0]
        num_samples = noisy_coordinates.shape[-3]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.expand(batch_size, num_samples)
        padded = sigma[..., None, None]
        update = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            return_token_repr=return_token_repr,
            **conditioning,
        )
        if return_token_repr:
            update, token_repr = cast(tuple[Tensor, Tensor], update)
        else:
            update = cast(Tensor, update)
        denoised = self.preconditioner.c_skip(padded) * noisy_coordinates + self.preconditioner.c_out(padded) * update
        if return_token_repr:
            return denoised, token_repr
        return denoised

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
        single_inputs: Tensor,
        single_trunk: Tensor,
        conditioning: AtomDenoiserContext,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sample_chunk_size = num_samples if sample_chunk_size is None else min(sample_chunk_size, num_samples)
        dtype, device = single_trunk.dtype, single_trunk.device
        observed = (
            atom_mask.to(torch.bool)
            if coordinate_mask is None
            else coordinate_mask.to(torch.bool) & atom_mask.to(torch.bool)
        )
        target = (
            project_model_coordinates(target_coordinates, observed, dtype=dtype, feature="Boltz1 training target")
            .to(device=device)
            .unsqueeze(-3)
        )
        target = target.expand(
            *target.shape[:-3],
            num_samples,
            *target.shape[-2:],
        )
        # Only observed target points define the target's centroid and placement.
        # Missing truth remains distinct from valid model slots to be predicted.
        augmentation_mask = observed.to(device=device)
        batch_size = target_coordinates.shape[0]
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
            noise_level = noise_level.unsqueeze(-1).expand(batch_size, num_samples)
        else:
            noise_level = noise_level.reshape(batch_size, num_samples)
        augmented_target_coordinates = self._augment_target(
            target,
            augmentation_mask,
            coordinate_augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        random_device = augmented_target_coordinates.device if generator is None else generator.device
        noise = torch.empty_like(augmented_target_coordinates, device=random_device).normal_(generator=generator)
        noise = noise.to(augmented_target_coordinates.device)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * noise
        denoised_chunks = []
        for sample_start in range(0, num_samples, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, num_samples)
            num_chunk_samples = sample_end - sample_start
            noisy_chunk = torch.narrow(noisy_coordinates, -3, sample_start, num_chunk_samples)
            noise_level_chunk = torch.narrow(noise_level, -1, sample_start, num_chunk_samples)
            denoised_chunk = self.preconditioned_network_forward(
                noisy_chunk,
                noise_level_chunk,
                atom_mask=atom_mask,
                return_token_repr=False,
                single_inputs=single_inputs,
                single_trunk=single_trunk,
                conditioning=conditioning,
                atom_to_token=atom_to_token,
                token_mask=token_mask,
            )
            denoised_chunks.append(cast(Tensor, denoised_chunk))
        denoised_coordinates = denoised_chunks[0] if len(denoised_chunks) == 1 else torch.cat(denoised_chunks, dim=-3)
        return cast(Tensor, denoised_coordinates), augmented_target_coordinates, noise_level

    def _augment_target(
        self,
        coords: Tensor,
        mask: Tensor,
        *,
        coordinate_augmentation: bool,
        translation_scale: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        observed = mask.to(torch.bool).unsqueeze(-2) & torch.ones_like(coords[..., 0], dtype=torch.bool)
        coords = project_model_coordinates(coords, observed, feature="Boltz1 augmentation target")
        centroid = weighted_mean(coords.transpose(-1, -2), observed.unsqueeze(-2)).unsqueeze(-2)
        centroid = torch.where(observed.any(-1, keepdim=True).unsqueeze(-1), centroid, 0)
        coords = project_model_coordinates(coords - centroid, observed, feature="Boltz1 centered target")
        if not coordinate_augmentation:
            return coords
        random_device = coords.device if generator is None else generator.device
        rotation = uniform_random_rotation(
            math.prod(coords.shape[:-2]),
            device=random_device,
            dtype=coords.dtype,
            generator=generator,
        ).to(coords.device)
        rotation = rotation.reshape(*coords.shape[:-2], 3, 3)
        translation = draw_normal(
            (*coords.shape[:-2], 1, 3),
            device=coords.device,
            dtype=coords.dtype,
            generator=generator,
        )
        return project_model_coordinates(
            rotate_vectors(coords, rotation) + translation_scale * translation,
            observed,
            feature="Boltz1 augmented target",
        )

    def _sample_ordinary(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None,
        step_scale: float,
        num_samples: int,
        generator: torch.Generator | None,
        return_token_repr: bool,
        conditioning: dict[str, object],
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Run ordinary Boltz-1 reverse diffusion with an explicit sample axis."""

        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        expanded_atom_mask = atom_mask.unsqueeze(-2)
        expanded_atom_mask = expanded_atom_mask.expand(
            *expanded_atom_mask.shape[:-2],
            num_samples,
            expanded_atom_mask.shape[-1],
        )
        coordinate_template = expanded_atom_mask.unsqueeze(-1).expand(
            *expanded_atom_mask.shape,
            3,
        )
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_hats = sigmas[:-1] * (1 + gammas[1:])
        noise_variances = self.noise_scale**2 * (sigma_hats**2 - sigmas[:-1] ** 2)

        coords = sigmas[0] * draw_normal_like(coordinate_template.to(dtype=dtype), generator)
        token_repr = None
        token_a = None
        for index in range(len(sigmas) - 1):
            sigma_hat = sigma_hats[index]
            random_device = device if generator is None else generator.device
            rotation = uniform_random_rotation(
                math.prod(coords.shape[:-2]),
                device=random_device,
                dtype=coords.dtype,
                generator=generator,
            ).to(device)
            rotation = rotation.reshape(*coords.shape[:-2], 3, 3)
            translation = draw_normal_like(torch.narrow(coords, -2, 0, 1), generator)
            coords = rotate_vectors(coords - coords.mean(-2, keepdim=True), rotation) + translation

            noise_var = noise_variances[index]
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)) * draw_normal_like(coords, generator)
            with torch.no_grad():
                denoised_out = self.preconditioned_network_forward(
                    noisy,
                    sigma_hat,
                    atom_mask=atom_mask,
                    return_token_repr=return_token_repr,
                    **conditioning,
                )
            if return_token_repr:
                denoised, token_a = cast(tuple[Tensor, Tensor], denoised_out)
                token_a = token_a.detach()
                output_token_feature_update = self.output_token_feature_update
                if output_token_feature_update is not None:
                    if token_repr is None:
                        token_repr = torch.zeros_like(token_a)
                    sample_sigma = sigma_hat.expand(token_a.shape[0], token_a.shape[-3]).to(
                        device=token_a.device,
                    )
                    token_repr = output_token_feature_update(
                        noise_level=self.preconditioner.c_noise(sample_sigma),
                        accumulated=token_repr,
                        next_repr=token_a,
                    )
                else:
                    token_repr = token_a
            else:
                denoised = cast(Tensor, denoised_out)
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy, target=denoised, weight=expanded_atom_mask, backend=self.backend
                )
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale * (sigmas[index + 1] - sigma_hat) * derivative
        return (coords, cast(Tensor, token_repr)) if return_token_repr else coords

    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        return_token_repr: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        step_scale = self.step_scale if step_scale is None else step_scale
        feats = cast(dict[str, Tensor] | None, conditioning.pop("feats", None))
        guidance_active = physical_guidance and num_guidance_steps > 0
        if not feynman_kac and not guidance_active:
            return self._sample_ordinary(
                atom_mask,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                num_samples=num_samples,
                generator=generator,
                return_token_repr=return_token_repr,
                conditioning=conditioning,
            )
        if isinstance(atom_mask, NestedTensor):
            raise NotImplementedError("packed Boltz-1 sampling does not yet support Feynman-Kac or potentials")
        potentials = (
            _boltz1_potentials(physical_guidance, backend=self.backend) if feynman_kac or guidance_active else ()
        )
        num_requested_samples = num_samples
        num_effective_samples = num_samples * num_particles if feynman_kac else num_samples
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(num_effective_samples, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_hats = sigmas[:-1] * (1 + gammas[1:])
        noise_variances = self.noise_scale**2 * (sigma_hats**2 - sigmas[:-1] ** 2)
        noise_active = tuple((noise_variances > 0).tolist()) if feynman_kac else ()
        potential_inputs: list[
            tuple[
                Boltz1Potential,
                tuple[_Boltz1PotentialLowering, ...],
                tuple[_Boltz1ResolvedPotential, ...] | None,
            ]
        ] = []
        potential_schedule: list[tuple[dict[str, float | int], ...]] = []
        steering_slices: tuple[slice, ...] = ()
        if potentials:
            steering_feats = cast(dict[str, Tensor], feats)
            for potential in potentials:
                lowerings = tuple(
                    potential._lower(steering_feats, batch_index) for batch_index in range(atom_mask.shape[0])
                )
                scientific_parameters_are_scheduled = any(
                    isinstance(value, Boltz1ParameterSchedule)
                    for name, value in potential.parameters.items()
                    if name not in {"guidance_interval", "guidance_weight", "resampling_weight"}
                )
                resolved = None
                if not scientific_parameters_are_scheduled:
                    parameters = potential.compute_parameters(1.0)
                    resolved = tuple(
                        potential._cast_arguments(
                            potential._resolve_parameters(lowering, parameters, dtype=dtype),
                            device=device,
                            dtype=dtype,
                        )
                        for lowering in lowerings
                    )
                potential_inputs.append((potential, lowerings, resolved))
            for index in range(len(sigmas) - 1):
                steering_t = 1.0 - (index / max(len(sigmas) - 1, 1))
                potential_schedule.append(
                    tuple(potential.compute_parameters(steering_t) for potential, _, _ in potential_inputs)
                )
            steering_slices = tuple(
                slice(
                    batch_index * num_effective_samples,
                    (batch_index + 1) * num_effective_samples,
                )
                for batch_index in range(atom_mask.shape[0])
            )

        coords = sigmas[0] * draw_normal(shape, device=device, dtype=dtype, generator=generator)
        if feynman_kac:
            num_groups = atom_mask.shape[0] * num_requested_samples
            energy_traj = torch.empty((expanded_atom_mask.shape[0], 0), device=device, dtype=dtype)
            resample_weights = torch.ones(num_groups, num_particles, device=device, dtype=dtype)
        else:
            energy_traj = torch.empty((0, 0), device=device, dtype=dtype)
            resample_weights = torch.empty((0, 0), device=device, dtype=dtype)
        if guidance_active:
            scaled_guidance_update = torch.zeros(shape, dtype=dtype, device=device)
        else:
            scaled_guidance_update = None
        denoised = None
        token_repr = None
        token_a = None
        # Potential guidance interleaves particle resampling, coordinate updates,
        # and token-feature accumulation with the reverse-diffusion steps.
        # Keep the schedule scalars as tensors. The Python loop fixes the number of denoising calls, while the
        # per-step tensor math remains free of host synchronizations on the ordinary (unsteered) sampling path.
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
                scaled_guidance_update = torch.einsum("bmd,bds->bms", scaled_guidance_update, rotation)

            sigma_hat = sigma_hats[index]
            noise_var = noise_variances[index]
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)) * draw_normal(
                shape, device=device, dtype=dtype, generator=generator
            )
            with torch.no_grad():
                noisy_explicit = noisy.reshape(
                    base_atom_mask.shape[0],
                    num_effective_samples,
                    *noisy.shape[1:],
                )
                denoised_out = self.preconditioned_network_forward(
                    noisy_explicit,
                    sigma_hat,
                    atom_mask=base_atom_mask,
                    return_token_repr=return_token_repr,
                    **conditioning,
                )
            if return_token_repr:
                denoised_explicit, token_a_explicit = cast(tuple[Tensor, Tensor], denoised_out)
                denoised = denoised_explicit.flatten(0, 1)
                token_a = token_a_explicit.flatten(0, 1).detach()
            else:
                denoised = cast(Tensor, denoised_out).flatten(0, 1)

            should_resample = feynman_kac and (
                (index % feynman_kac_resampling_interval == 0 and noise_active[index]) or index == len(sigmas) - 2
            )
            active_potentials: list[
                tuple[
                    Boltz1Potential,
                    tuple[_Boltz1ResolvedPotential, ...],
                    float,
                    float,
                    int,
                ]
            ] = []
            if should_resample or (guidance_active and index < len(sigmas) - 2):
                for (potential, lowerings, static_arguments), parameters in zip(
                    potential_inputs,
                    potential_schedule[index],
                    strict=True,
                ):
                    arguments = static_arguments
                    if arguments is None:
                        arguments = tuple(
                            potential._cast_arguments(
                                potential._resolve_parameters(lowering, parameters, dtype=dtype),
                                device=device,
                                dtype=dtype,
                            )
                            for lowering in lowerings
                        )
                    active_potentials.append(
                        (
                            potential,
                            arguments,
                            float(parameters["resampling_weight"]),
                            float(parameters["guidance_weight"]),
                            max(int(parameters["guidance_interval"]), 1),
                        )
                    )
            if should_resample:
                energy = torch.zeros(denoised.shape[0], device=device, dtype=denoised.dtype)
                for potential, arguments, resampling_weight, _, _ in active_potentials:
                    if resampling_weight > 0:
                        for row_arguments, row_slice in zip(arguments, steering_slices, strict=True):
                            energy[row_slice] = energy[row_slice] + resampling_weight * potential._compute(
                                denoised[row_slice],
                                row_arguments,
                            )
                energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)
                log_g = -energy if energy_traj.shape[1] == 1 else energy_traj[:, -2] - energy_traj[:, -1]
                if scaled_guidance_update is not None and noise_active[index]:
                    ll_difference = ((noisy - coords) ** 2 - (noisy - coords + scaled_guidance_update) ** 2).sum(
                        dim=(-1, -2)
                    ) / (2 * noise_var)
                else:
                    ll_difference = torch.zeros_like(energy)
                resample_weights = torch.softmax(
                    (ll_difference + feynman_kac_lambda * log_g).reshape(-1, num_particles),
                    dim=1,
                )

            if guidance_active and index < len(sigmas) - 2:
                guidance_update = torch.zeros_like(denoised)
                for guidance_step in range(num_guidance_steps):
                    energy_gradient = torch.zeros_like(denoised)
                    guided_coordinates = denoised + guidance_update
                    for potential, arguments, _, guidance_weight, interval in active_potentials:
                        if guidance_weight > 0 and guidance_step % interval == 0:
                            for row_arguments, row_slice in zip(arguments, steering_slices, strict=True):
                                energy_gradient[row_slice] = energy_gradient[
                                    row_slice
                                ] + guidance_weight * potential._compute_gradient(
                                    guided_coordinates[row_slice],
                                    row_arguments,
                                )
                    guidance_update = guidance_update - energy_gradient
                denoised = denoised + guidance_update
                scaled_guidance_update = guidance_update * -step_scale * (sigma_next - sigma_hat) / sigma_hat

            if should_resample:
                sample_count = num_particles if index < len(sigmas) - 2 else 1
                random_device = resample_weights.device if generator is None else generator.device
                resample_indices = (
                    torch.multinomial(
                        resample_weights.to(random_device),
                        sample_count,
                        replacement=True,
                        generator=generator,
                    ).to(device)
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
                    num_effective_samples = num_requested_samples
            if return_token_repr:
                output_token_feature_update = self.output_token_feature_update
                if output_token_feature_update is not None:
                    if token_repr is None:
                        token_repr = torch.zeros_like(cast(Tensor, token_a))
                    token_a = cast(Tensor, token_a)
                    sigma = sigma_hat.expand(token_a.shape[0]).to(device=token_a.device)
                    token_repr = output_token_feature_update(
                        noise_level=self.preconditioner.c_noise(sigma),
                        accumulated=token_repr,
                        next_repr=token_a,
                    )
                elif token_a is not None:
                    token_repr = token_a
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy, target=denoised, weight=expanded_atom_mask, backend=self.backend
                )
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale * (sigma_next - sigma_hat) * derivative
        coordinates = coords.reshape(base_atom_mask.shape[0], num_requested_samples, *coords.shape[1:])
        if not return_token_repr:
            return coordinates
        token_repr = cast(Tensor, token_repr)
        return coordinates, token_repr.reshape(
            base_atom_mask.shape[0],
            num_requested_samples,
            *token_repr.shape[1:],
        )


class Boltz1OutputTokenFeatureUpdate(nn.Module):
    r"""
    Accumulate the diffusion-token representation across reverse-diffusion steps for confidence conditioning.
    """

    def __init__(self, *, token_size: int = 384, fourier_embedding_size: int = 256) -> None:
        super().__init__()
        hidden = 2 * token_size
        self.next_norm = LayerNorm(hidden)
        self.noise_embedding = FrozenFourierEmbedding(fourier_embedding_size)
        self.noise_norm = LayerNorm(fourier_embedding_size)
        self.transition = TripleProductConditionedTransition(hidden, hidden + fourier_embedding_size)

    def forward(self, *, noise_level: Tensor, accumulated: Tensor, next_repr: Tensor) -> Tensor:
        next_repr = self.next_norm(next_repr)
        fourier = self.noise_norm(self.noise_embedding(noise_level))
        fourier = next_repr[..., :1] * 0 + fourier.unsqueeze(-2)
        conditioning = torch.cat([accumulated, fourier], dim=-1)
        return accumulated + self.transition(next_repr, conditioning)


class Boltz1ConfidenceConditioning(nn.Module):
    r"""
    Condition confidence single and pair representations on sampled coordinates.
    """

    diffusion_single_norm: nn.Module | None
    diffusion_single_projection: nn.Linear | None
    pair_left_projection: nn.Linear
    pair_right_projection: nn.Linear
    single_to_pair_product_left: nn.Linear | None
    single_to_pair_product_right: nn.Linear | None
    single_to_pair_product_out: nn.Linear | None
    boundaries: Tensor
    distance_embedding: nn.Embedding

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int,
        pair_size: int,
        num_distogram_bins: int,
        max_distance: float,
        add_single_to_pair_product: bool,
        use_single_diffusion: bool,
        backend: str,
    ) -> None:
        super().__init__()
        self.backend = backend
        if use_single_diffusion:
            self.diffusion_single_norm = LayerNorm(2 * single_size, backend=backend)
            self.diffusion_single_projection = nn.Linear(2 * single_size, single_size, bias=False)
        else:
            self.diffusion_single_norm = None
            self.diffusion_single_projection = None

        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        else:
            self.single_to_pair_product_left = None
            self.single_to_pair_product_right = None
            self.single_to_pair_product_out = None

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

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        token_representative_atom_index: Tensor,
        token_mask: Tensor,
        single_diffusion: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        coordinates = _boltz1_coordinates(coordinates)
        num_samples = coordinates.shape[-3]
        single = insert_sample_axis(single, num_samples, dim=-3)
        single_inputs = insert_sample_axis(single_inputs, num_samples, dim=-3)
        if self.diffusion_single_projection is not None and self.diffusion_single_norm is not None:
            single = single + self.diffusion_single_projection(
                self.diffusion_single_norm(cast(Tensor, single_diffusion))
            )

        pair = insert_sample_axis(pair, num_samples, dim=-4)
        pair = pair + self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single_inputs).unsqueeze(-3)
        if (
            self.single_to_pair_product_left is not None
            and self.single_to_pair_product_right is not None
            and self.single_to_pair_product_out is not None
        ):
            pair = pair + self.single_to_pair_product_out(
                self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
                * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
            )

        representative_coordinates = gather_atoms_by_index(
            coordinates,
            insert_sample_axis(token_representative_atom_index, num_samples, dim=-2),
        )
        distances = pairwise_distance(representative_coordinates)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        pair_mask = pair_mask_from_token_mask(insert_sample_axis(token_mask, num_samples, dim=-2), backend=self.backend)
        return single_inputs, single, pair, distances, pair_mask


class Boltz1ConfidenceHead(nn.Module, ABC):
    r"""Prepare, condition, refine and read out one actual Boltz-1 confidence head."""

    conditioning: Boltz1ConfidenceConditioning
    output_head: TokenConfidenceOutputHead

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        input: Boltz1Input,
        single_diffusion: Tensor | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        coordinates = _boltz1_coordinates(coordinates)
        single_inputs, single, pair = self._prepare_representations(
            single_inputs,
            single,
            pair,
            input=input,
        )
        single_inputs, single, pair, distances, pair_mask = self.conditioning(
            single_inputs,
            single,
            pair,
            coordinates,
            token_representative_atom_index=input.token_representative_atom_index,
            token_mask=input.token_mask,
            single_diffusion=single_diffusion,
        )
        single, pair = self._refine_representations(
            single,
            pair,
            single_inputs=single_inputs,
            input=input,
            pair_mask=pair_mask,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        return self.output_head(single, pair), distances

    @abstractmethod
    def _prepare_representations(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        *,
        input: Boltz1Input,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @abstractmethod
    def _refine_representations(
        self,
        single: Tensor,
        pair: Tensor,
        *,
        single_inputs: Tensor,
        input: Boltz1Input,
        pair_mask: Tensor,
        msa_max_depth: int | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]: ...


class Boltz1TrunkImitationConfidenceHead(Boltz1ConfidenceHead):
    r"""
    Rebuild and refine a confidence-specific copy of the Boltz-1 trunk representations.
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
        add_single_to_pair_product: bool = True,
        use_single_diffusion: bool = True,
        ranking_plddt_weight: float = 4.0,
        ranking_ptm_weight: float = 1.0,
        use_atom_encoder: bool = True,
        use_msa: bool = True,
        msa_size: int = 64,
        use_paired_feature: bool = False,
        num_msa_blocks: int = 4,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        msa_pair_dropout: float = 0.25,
        pair_dropout: float = 0.25,
        update_single: bool = True,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        msa_triangle_attention_head_size: int = 32,
        num_msa_triangle_attention_heads: int = 4,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        relative_position_size: int = 139,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.ranking_plddt_weight = ranking_plddt_weight
        self.ranking_ptm_weight = ranking_ptm_weight
        self.input_embeddings = Boltz1InputEmbeddings(
            input_size=input_size,
            single_size=single_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            atom_feature_size=atom_feature_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            use_atom_encoder=use_atom_encoder,
            backend=backend,
        )
        self.single_init_projection = nn.Linear(input_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.single_recycle_norm = LayerNorm(single_size, backend=backend)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(pair_size, backend=backend)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        self.conditioning = Boltz1ConfidenceConditioning(
            input_size=input_size,
            single_size=single_size,
            pair_size=pair_size,
            num_distogram_bins=num_distogram_bins,
            max_distance=max_distance,
            add_single_to_pair_product=add_single_to_pair_product,
            use_single_diffusion=use_single_diffusion,
            backend=backend,
        )

        self.msa_trunk: Boltz1MsaTrunk | None = None
        if use_msa:
            self.msa_trunk = Boltz1MsaTrunk(
                single_size=input_size,
                pair_size=pair_size,
                msa_size=msa_size,
                num_blocks=num_msa_blocks,
                use_paired_feature=use_paired_feature,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=msa_pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_attention_head_size=msa_attention_head_size,
                triangle_attention_head_size=msa_triangle_attention_head_size,
                num_triangle_attention_heads=num_msa_triangle_attention_heads,
                backend=backend,
            )
        self.pairformer_stack: PairformerStack | None = None
        self.pair_update_stack: TrianglePairUpdateStack | None = None
        if update_single:
            self.pairformer_stack = PairformerStack(
                block=LineDropoutPairformerBlock,
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        else:
            self.pair_update_stack = TrianglePairUpdateStack(
                block=LineDropoutTrianglePairUpdateBlock,
                num_blocks=num_blocks,
                pair_size=pair_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        self.final_single_norm = LayerNorm(single_size, backend=backend)
        self.final_pair_norm = LayerNorm(pair_size, backend=backend)
        self.output_head = TokenConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins if compute_pae else None,
            num_pde_bins=num_pde_bins,
        )

    def _prepare_representations(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        *,
        input: Boltz1Input,
    ) -> tuple[Tensor, Tensor, Tensor]:
        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            pocket_feature=input.pocket_feature,
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
            chain_gate="not_same_chain",
            dtype=single.dtype,
        )
        pair_init = pair_init + self.relative_position_embedder(relative_features)
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
        pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
        return single_inputs, single, pair

    def _refine_representations(
        self,
        single: Tensor,
        pair: Tensor,
        *,
        single_inputs: Tensor,
        input: Boltz1Input,
        pair_mask: Tensor,
        msa_max_depth: int | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        if self.msa_trunk is not None and input.msa_tokens is not None:
            num_samples = single.shape[-3]
            pair = pair + cast(Boltz1MsaTrunk, self.msa_trunk)(
                pair,
                single_inputs,
                msa_tokens=insert_sample_axis(input.msa_tokens, num_samples, dim=-3),
                has_deletion=insert_sample_axis(cast(Tensor, input.has_deletion), num_samples, dim=-3),
                deletion_value=insert_sample_axis(cast(Tensor, input.deletion_value), num_samples, dim=-3),
                is_paired=insert_sample_axis(cast(Tensor, input.msa_paired), num_samples, dim=-3),
                msa_mask=insert_sample_axis(cast(Tensor, input.msa_mask), num_samples, dim=-3),
                token_mask=insert_sample_axis(input.token_mask, num_samples, dim=-2),
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
        if self.pairformer_stack is not None:
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        else:
            pair = cast(TrianglePairUpdateStack, self.pair_update_stack)(
                pair,
                pair_mask=pair_mask,
            )
        return self.final_single_norm(single), self.final_pair_norm(pair)


class Boltz1TrunkReuseConfidenceHead(Boltz1ConfidenceHead):
    r"""
    Normalize and refine the trunk representations supplied by the Boltz-1 model.
    """

    token_feature_map: Tensor

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 48,
        num_pair_attention_heads: int = 16,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        compute_pae: bool = True,
        add_single_to_pair_product: bool = True,
        add_single_input_to_single: bool = True,
        add_pair_input_to_pair: bool = True,
        use_single_diffusion: bool = True,
        ranking_plddt_weight: float = 4.0,
        ranking_ptm_weight: float = 1.0,
        pair_dropout: float = 0.25,
        update_single: bool = True,
        relative_position_size: int = 139,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.ranking_plddt_weight = ranking_plddt_weight
        self.ranking_ptm_weight = ranking_ptm_weight
        self.encoded_size = single_size
        feature_map = folding_source_map()
        self.register_buffer("token_feature_map", feature_map, persistent=False)
        source_input_size = input_size - 2 * (len(TOKENS) - feature_map.shape[0])
        self.input_single_norm = LayerNorm(source_input_size, backend=backend)
        self.single_norm: LayerNorm | None = None
        if update_single:
            self.single_norm = LayerNorm(single_size, backend=backend)
        self.input_single_projection: nn.Linear | None = None
        if add_single_input_to_single:
            self.input_single_projection = nn.Linear(source_input_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.relative_position_embedder: RelativePositionEmbedder | None = None
        self.token_bond_projection: nn.Linear | None = None
        if add_pair_input_to_pair:
            self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
            self.token_bond_projection = nn.Linear(1, pair_size, bias=False)

        self.conditioning = Boltz1ConfidenceConditioning(
            input_size=source_input_size,
            single_size=single_size,
            pair_size=pair_size,
            num_distogram_bins=num_distogram_bins,
            max_distance=max_distance,
            add_single_to_pair_product=add_single_to_pair_product,
            use_single_diffusion=use_single_diffusion,
            backend=backend,
        )

        self.pairformer_stack: PairformerStack | None = None
        self.pair_update_stack: TrianglePairUpdateStack | None = None
        if update_single:
            self.pairformer_stack = PairformerStack(
                block=LineDropoutPairformerBlock,
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        else:
            self.pair_update_stack = TrianglePairUpdateStack(
                block=LineDropoutTrianglePairUpdateBlock,
                num_blocks=num_blocks,
                pair_size=pair_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        self.output_head = TokenConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins if compute_pae else None,
            num_pde_bins=num_pde_bins,
        )

    def _prepare_representations(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        *,
        input: Boltz1Input,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_inputs = self.input_single_norm(
            map_token_feature_planes(single_inputs, self.token_feature_map, self.encoded_size)
        )
        if self.single_norm is not None:
            single = self.single_norm(single)
        if self.input_single_projection is not None:
            single = single + self.input_single_projection(normalized_inputs)
        pair = self.pair_norm(pair)
        if self.relative_position_embedder is not None and self.token_bond_projection is not None:
            relative_features = generate_relative_position_features(
                residue_index=input.residue_index,
                chain_index=input.chain_index,
                entity_index=input.entity_index,
                symmetry_index=input.symmetry_index,
                token_index=input.token_index,
                cyclic_period=input.cyclic_period,
                chain_gate="not_same_chain",
                dtype=pair.dtype,
            )
            pair = pair + self.relative_position_embedder(relative_features)
            pair = pair + self.token_bond_projection(input.token_bonds.unsqueeze(-1).to(pair.dtype))
        return normalized_inputs, single, pair

    def _refine_representations(
        self,
        single: Tensor,
        pair: Tensor,
        *,
        single_inputs: Tensor,
        input: Boltz1Input,
        pair_mask: Tensor,
        msa_max_depth: int | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        if self.pairformer_stack is not None:
            return self.pairformer_stack(single, pair, pair_mask=pair_mask)
        pair = cast(TrianglePairUpdateStack, self.pair_update_stack)(
            pair,
            pair_mask=pair_mask,
        )
        return single, pair


def _boltz1_steering_features(input: Boltz1Input, *, atom_features: Tensor) -> dict[str, Tensor]:
    r"""
    Assemble the Boltz-1x steering feature dict consumed by the physical potentials.

    Args:
        input: The Boltz-1 input batch (source of the constraint index/mask tensors).
        atom_features: `[B, M, atom_feature_size]` atom features; used to slice the reference-element one-hot.

    Returns:
        Mapping from steering feature name to a device-placed tensor (empty tensors when a feature is absent).
    """

    device = atom_features.device
    return {
        "atom_to_token": input.atom_to_token.to(device=device),
        "chain_index": input.chain_index.to(device=device, dtype=torch.long),
        "atom_mask": input.atom_mask.to(device=device),
        "reference_elements": atom_features[..., 5:133].to(device=device),
        "geometry_pair_index": input.geometry_pair_index.to(device=device, dtype=torch.long),
        "geometry_bond_mask": input.geometry_bond_mask.to(device=device, dtype=torch.bool),
        "geometry_angle_mask": input.geometry_angle_mask.to(device=device, dtype=torch.bool),
        "geometry_upper_bounds": input.geometry_upper_bounds.to(device=device),
        "geometry_lower_bounds": input.geometry_lower_bounds.to(device=device),
        "chiral_atom_index": input.chiral_atom_index.to(device=device, dtype=torch.long),
        "chiral_atom_positive": input.chiral_atom_positive.to(device=device, dtype=torch.bool),
        "stereo_bond_index": input.stereo_bond_index.to(device=device, dtype=torch.long),
        "stereo_bond_trans": input.stereo_bond_trans.to(device=device, dtype=torch.bool),
        "planar_bond_index": input.planar_bond_index.to(device=device, dtype=torch.long),
        "connected_chain_index": input.connected_chain_index.to(device=device, dtype=torch.long),
        "connected_atom_index": input.connected_atom_index.to(device=device, dtype=torch.long),
        "symmetric_chain_index": input.symmetric_chain_index.to(device=device, dtype=torch.long),
    }


class Boltz1ParameterSchedule:
    r"""
    Abstract base for a steering-parameter schedule evaluated over normalized sampling time.
    """

    def compute(self, time: float) -> float:
        raise NotImplementedError


class Boltz1ExponentialInterpolationSchedule(Boltz1ParameterSchedule):
    r"""
    Interpolate between ``start`` and ``end`` with an exponential curvature ``alpha`` (linear when ``alpha=0``).
    """

    def __init__(self, *, start: float, end: float, alpha: float) -> None:
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, time: float) -> float:
        if self.alpha == 0:
            return self.start + (self.end - self.start) * time
        return self.start + (self.end - self.start) * (math.exp(self.alpha * time) - 1) / (math.exp(self.alpha) - 1)


class Boltz1PiecewiseStepSchedule(Boltz1ParameterSchedule):
    r"""
    Step schedule that returns the value for the first ``thresholds`` bracket containing ``time``.
    """

    def __init__(self, *, thresholds: list[float], values: list[float]) -> None:
        self.thresholds = thresholds
        self.values = values

    def compute(self, time: float) -> float:
        index = 0
        while index < len(self.thresholds) and time > self.thresholds[index]:
            index += 1
        return self.values[index]


class Boltz1Potential:
    r"""
    Base class for a Boltz-1x physical potential over predicted coordinates.
    """

    def __init__(
        self, parameters: Mapping[str, float | int | Boltz1ParameterSchedule], *, backend: str = "auto"
    ) -> None:
        self.backend = backend
        self.parameters = parameters

    def compute_parameters(self, time: float) -> dict[str, float | int]:
        return {
            name: value.compute(time) if isinstance(value, Boltz1ParameterSchedule) else value
            for name, value in self.parameters.items()
        }

    def compute(self, coords: Tensor, feats: dict[str, Tensor], parameters: Mapping[str, float | int]) -> Tensor:
        coords = _boltz1_coordinates(coords)
        return self._compute(coords, self.compute_args(feats, parameters, dtype=coords.dtype))

    def _compute(self, coords: Tensor, arguments: _Boltz1ResolvedPotential) -> Tensor:
        index, args, _ = arguments
        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
        value = cast(Tensor, self.compute_variable(coords, index, compute_gradient=False))
        energy = cast(Tensor, self.compute_function(value, *args, compute_derivative=False))
        return energy.sum(dim=-1)

    def compute_gradient(
        self,
        coords: Tensor,
        feats: dict[str, Tensor],
        parameters: Mapping[str, float | int],
    ) -> Tensor:
        coords = _boltz1_coordinates(coords)
        return self._compute_gradient(coords, self.compute_args(feats, parameters, dtype=coords.dtype))

    def _compute_gradient(self, coords: Tensor, arguments: _Boltz1ResolvedPotential) -> Tensor:
        index, args, _ = arguments
        if index.shape[1] == 0:
            return torch.zeros_like(coords)
        value, grad_value = self.compute_variable(coords, index, compute_gradient=True)
        _, d_energy = self.compute_function(value, *args, compute_derivative=True)
        prod = d_energy.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(start_dim=-3, end_dim=-2)
        return torch.zeros_like(coords).scatter_reduce(
            -2,
            index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coords.shape[:-2], -1, 3)),
            prod,
            "sum",
        )

    def compute_args(
        self,
        feats: dict[str, Tensor],
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        return self._resolve_parameters(
            self._lower(feats, 0),
            parameters,
            dtype=dtype,
        )

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        raise NotImplementedError

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        raise NotImplementedError

    def _cast_arguments(
        self,
        arguments: _Boltz1ResolvedPotential,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _Boltz1ResolvedPotential:
        index, (k, lower_bounds, upper_bounds), com_args = arguments
        return (
            index,
            (
                k.to(device=device, dtype=dtype),
                None if lower_bounds is None else lower_bounds.to(device=device, dtype=dtype),
                None if upper_bounds is None else upper_bounds.to(device=device, dtype=dtype),
            ),
            com_args,
        )

    def compute_function(
        self,
        value: Tensor,
        k: Tensor,
        lower_bounds: Tensor | None,
        upper_bounds: Tensor | None,
        *,
        compute_derivative: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError


class Boltz1FlatBottomPotential(Boltz1Potential):
    r"""
    Flat-bottom energy: zero inside ``[lower_bounds, upper_bounds]`` and linear with slope ``k`` outside.
    """

    def compute_function(
        self,
        value: Tensor,
        k: Tensor,
        lower_bounds: Tensor | None,
        upper_bounds: Tensor | None,
        *,
        compute_derivative: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.to(device=value.device, dtype=value.dtype)
        upper_bounds = upper_bounds.to(device=value.device, dtype=value.dtype)
        k = k.to(device=value.device, dtype=value.dtype)
        energy = linear_flat_bottom_energy(value, k, lower_bounds, upper_bounds)
        if not compute_derivative:
            return energy
        return energy, linear_flat_bottom_derivative(value, k, lower_bounds, upper_bounds)


class Boltz1DistancePotential(Boltz1Potential):
    r"""
    Geometric quantity mixin computing pairwise atom distances from a 2-row index.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        coords = _boltz1_coordinates(coords)
        if not compute_gradient:
            return indexed_distance(coords, index)
        return indexed_distance_with_gradient(coords, index)


class Boltz1DihedralPotential(Boltz1Potential):
    r"""
    Geometric quantity mixin computing signed dihedral angles from a 4-row index.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        coords = _boltz1_coordinates(coords)
        if not compute_gradient:
            return indexed_signed_dihedral(coords, index, backend=self.backend)
        return indexed_signed_dihedral_with_gradient(coords, index, backend=self.backend)


class Boltz1AbsoluteDihedralPotential(Boltz1DihedralPotential):
    r"""
    Dihedral variant that returns the absolute angle, with the gradient sign flipped where the angle is negative.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        coords = _boltz1_coordinates(coords)
        if not compute_gradient:
            return torch.abs(cast(Tensor, super().compute_variable(coords, index, compute_gradient=False)))
        phi, grad = cast(tuple[Tensor, Tensor], super().compute_variable(coords, index, compute_gradient=True))
        grad = torch.where((phi < 0)[..., None, :, None], -grad, grad)
        return torch.abs(phi), grad


class Boltz1PoseBustersPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    PoseBusters bond/angle/clash potential over molecular distance bounds.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        pair_index = feats["geometry_pair_index"][batch_index]
        available = feats["atom_mask"][batch_index].bool()[pair_index].all(dim=0)
        pair_index = pair_index[:, available]
        lower_bounds = feats["geometry_lower_bounds"][batch_index][available]
        upper_bounds = feats["geometry_upper_bounds"][batch_index][available]
        bond_mask = feats["geometry_bond_mask"][batch_index][available]
        angle_mask = feats["geometry_angle_mask"][batch_index][available]
        pair_radii = _boltz1_pair_vdw_radii(feats["reference_elements"][batch_index], pair_index)
        bond_cutoffs = 0.35 + pair_radii.mean(dim=0)
        return pair_index, (lower_bounds, upper_bounds, bond_mask, angle_mask, bond_cutoffs), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        pair_index, raw_arguments, com_args = lowering
        lower_bounds, upper_bounds, bond_mask, angle_mask, bond_cutoffs = raw_arguments
        if dtype is None:
            dtype = torch.promote_types(lower_bounds.dtype, torch.promote_types(upper_bounds.dtype, bond_cutoffs.dtype))
        lower_bounds = lower_bounds.to(dtype=dtype).clone()
        upper_bounds = upper_bounds.to(dtype=dtype).clone()
        bond_cutoffs = bond_cutoffs.to(dtype=dtype)
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
        lower_bounds[~bond_mask] = torch.max(
            lower_bounds[~bond_mask],
            bond_cutoffs[~bond_mask],
        )
        upper_bounds[bond_mask] = torch.min(
            upper_bounds[bond_mask],
            bond_cutoffs[bond_mask],
        )
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), com_args


class Boltz1ConnectionsPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Distance potential pulling cross-chain connected atom pairs within a fixed upper-bound buffer.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        pair_index = feats["connected_atom_index"][batch_index]
        return pair_index, (torch.ones(pair_index.shape[1], device=pair_index.device),), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        pair_index, (template,), com_args = lowering
        upper_bounds = template.to(dtype=dtype) * float(parameters["buffer"])
        return pair_index, (torch.ones_like(upper_bounds), None, upper_bounds), com_args


class Boltz1VdwOverlapPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Van-der-Waals clash potential penalizing non-bonded, cross-chain atom pairs closer than their summed radii.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        atom_chain_index = feats["chain_index"].gather(1, feats["atom_to_token"])[batch_index]
        atom_mask = feats["atom_mask"][batch_index].bool()
        num_chains = torch.bincount(atom_chain_index).shape[0]
        chain_sizes = torch.bincount(
            atom_chain_index[atom_mask],
            minlength=num_chains,
        )
        single_ion_mask = (chain_sizes > 1)[atom_chain_index]
        pair_index = torch.triu_indices(
            atom_chain_index.shape[0],
            atom_chain_index.shape[0],
            1,
            device=atom_chain_index.device,
        )
        pair_mask = atom_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]
        connected_chain_index = feats["connected_chain_index"][batch_index]
        connected_chain_matrix = torch.eye(
            num_chains,
            device=atom_chain_index.device,
            dtype=torch.bool,
        )
        if connected_chain_index.shape[1] > 0:
            connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = True
            connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = True
        connected_chain_mask = connected_chain_matrix[
            atom_chain_index[pair_index[0]],
            atom_chain_index[pair_index[1]],
        ]
        pair_index = pair_index[:, pair_mask & pair_ion_mask & ~connected_chain_mask]
        pair_radii = _boltz1_pair_vdw_radii(feats["reference_elements"][batch_index], pair_index)
        unbuffered_bounds = pair_radii.sum(dim=0)
        return pair_index, (unbuffered_bounds,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        pair_index, (unbuffered_bounds,), com_args = lowering
        lower_bounds = unbuffered_bounds.to(dtype=dtype) * (1.0 - float(parameters["buffer"]))
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), com_args


class Boltz1SymmetricChainComPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Distance potential separating arithmetic centers of same-entity symmetric chain pairs.
    """

    def _compute(self, coords: Tensor, arguments: _Boltz1ResolvedPotential) -> Tensor:
        pair_index, (slope, lower_bound, _), com_args = arguments
        atom_chain, atom_mask, num_chains = cast(tuple[Tensor, Tensor, int], com_args)
        return cast(
            Tensor,
            chain_center_separation(
                coords,
                atom_chain,
                atom_mask,
                num_chains,
                pair_index,
                cast(Tensor, lower_bound).to(coords),
                slope.to(coords),
                compute_gradient=False,
            ),
        )

    def _compute_gradient(self, coords: Tensor, arguments: _Boltz1ResolvedPotential) -> Tensor:
        pair_index, (slope, lower_bound, _), com_args = arguments
        atom_chain, atom_mask, num_chains = cast(tuple[Tensor, Tensor, int], com_args)
        _, gradient = chain_center_separation(
            coords,
            atom_chain,
            atom_mask,
            num_chains,
            pair_index,
            cast(Tensor, lower_bound).to(coords),
            slope.to(coords),
            compute_gradient=True,
        )
        return gradient

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        atom_chain_index = feats["chain_index"].gather(1, feats["atom_to_token"])[batch_index]
        atom_mask = feats["atom_mask"][batch_index].bool()
        pair_index = feats["symmetric_chain_index"][batch_index]
        active_chains = torch.unique(atom_chain_index[atom_mask])
        pair_index = pair_index[:, torch.isin(pair_index, active_chains).all(dim=0)]
        num_chains = torch.bincount(atom_chain_index).shape[0]
        chain_sizes = torch.bincount(
            atom_chain_index[atom_mask],
            minlength=num_chains,
        )
        single_ion_mask = chain_sizes > 1
        pair_index = pair_index[:, single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]]
        lower_bounds = torch.ones(pair_index.shape[1], device=pair_index.device)
        return pair_index, (lower_bounds,), (atom_chain_index, atom_mask, num_chains)

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        pair_index, (template,), com_args = lowering
        lower_bounds = template.to(dtype=dtype) * float(parameters["buffer"])
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), com_args


class Boltz1StereoBondPotential(Boltz1AbsoluteDihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing E/Z stereo-bond orientations toward 0 or pi.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        index = feats["stereo_bond_index"][batch_index]
        orientation = feats["stereo_bond_trans"][batch_index].bool()
        return index, (orientation,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        index, (orientation,), com_args = lowering
        buffer = float(parameters["buffer"])
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device, dtype=dtype)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device, dtype=dtype)
        lower_bounds[orientation] = torch.pi - buffer
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = buffer
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), com_args


class Boltz1ChiralAtomPotential(Boltz1DihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Signed-dihedral potential enforcing chiral-center handedness by sign of the improper dihedral.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        index = feats["chiral_atom_index"][batch_index]
        orientation = feats["chiral_atom_positive"][batch_index].bool()
        return index, (orientation,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        index, (orientation,), com_args = lowering
        buffer = float(parameters["buffer"])
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device, dtype=dtype)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device, dtype=dtype)
        lower_bounds[orientation] = buffer
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = -buffer
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), com_args


class Boltz1PlanarBondPotential(Boltz1AbsoluteDihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing planarity around double bonds via improper dihedrals near zero.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        double_bond_index = feats["planar_bond_index"][batch_index].T
        improper_map = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 3]], device=double_bond_index.device).T
        improper_index = double_bond_index[:, improper_map].swapaxes(0, 1).flatten(start_dim=1)
        return improper_index, (), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
        *,
        dtype: torch.dtype | None = None,
    ) -> _Boltz1ResolvedPotential:
        improper_index, _, com_args = lowering
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            float(parameters["buffer"]),
            device=improper_index.device,
            dtype=dtype,
        )
        return improper_index, (torch.ones_like(upper_bounds), None, upper_bounds), com_args


def _boltz1_potentials(physical_guidance: bool, *, backend: str = "auto") -> tuple[Boltz1Potential, ...]:
    r"""
    Instantiate the physical-potential set with its scheduled guidance and resampling weights.

    Args:
        physical_guidance: Whether to enable the guidance weights.

    Returns:
        The tuple of configured Boltz-1x potentials.
    """

    guidance_weight = 1.0 if physical_guidance else 0.0
    return (
        Boltz1SymmetricChainComPotential(
            {
                "guidance_interval": 4,
                "guidance_weight": 0.5 * guidance_weight,
                "resampling_weight": 0.5,
                "buffer": Boltz1ExponentialInterpolationSchedule(start=1.0, end=5.0, alpha=-2.0),
            },
            backend=backend,
        ),
        Boltz1VdwOverlapPotential(
            {
                "guidance_interval": 5,
                "guidance_weight": (
                    Boltz1PiecewiseStepSchedule(thresholds=[0.4], values=[0.125, 0.0]) if guidance_weight else 0.0
                ),
                "resampling_weight": Boltz1PiecewiseStepSchedule(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            },
            backend=backend,
        ),
        Boltz1ConnectionsPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.15 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 2.0,
            },
            backend=backend,
        ),
        Boltz1PoseBustersPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.01 * guidance_weight,
                "resampling_weight": 0.1,
                "bond_buffer": 0.125,
                "angle_buffer": 0.125,
                "clash_buffer": 0.10,
            },
            backend=backend,
        ),
        Boltz1ChiralAtomPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.10 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            },
            backend=backend,
        ),
        Boltz1StereoBondPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            },
            backend=backend,
        ),
        Boltz1PlanarBondPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.26180,
            },
            backend=backend,
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


def _boltz1_pair_vdw_radii(reference_elements: Tensor, pair_index: Tensor) -> Tensor:
    """Read supported radii only for participating, exactly Z-encoded pair endpoints."""
    selected = reference_elements[pair_index]
    if not pair_index.numel():
        return selected.new_empty(pair_index.shape)
    valid = ((selected == 0) | (selected == 1)).all(dim=-1) & (selected.sum(dim=-1) == 1)
    if not bool(valid.all()):
        fail("value_invalid", RequestRef("reference_elements", "not exact atomic-number one-hot rows"))
    atomic_numbers = selected.argmax(dim=-1)
    if bool((atomic_numbers == 0).any()):
        fail("value_invalid", RequestRef("reference_elements", "atomic number 0"))
    radii = _vdw_radii(selected.device, dtype=selected.dtype)[atomic_numbers]
    if bool((radii <= 0).any()):
        fail("model_capacity", RequestRef("reference_elements", "no van-der-Waals radius"))
    return radii


def _vdw_radii(device: torch.device, *, dtype: torch.dtype | None = None) -> Tensor:
    r"""
    Build the per-element (atomic-number-indexed) van-der-Waals radius lookup table.

    Args:
        device: Device to allocate the table on.

    Returns:
        A `[128]` float tensor indexed by atomic number (index 0 and unused entries are zero).
    """

    radii = torch.zeros(128, dtype=dtype, device=device)
    radii[1:119] = torch.tensor(_VDW_RADII, dtype=dtype, device=device)
    return radii


def _boltz1_confidence_scores(
    logits: dict[str, Tensor],
    *,
    distances: Tensor,
    input: Boltz1Input,
    pred_distogram_logits: Tensor,
    coordinates: Tensor,
    ranking_plddt_weight: float,
    ranking_ptm_weight: float,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""
    Derive Boltz-1 ranking metrics (complex/interface pLDDT, PDE, pTM/ipTM, confidence score) from the logits.

    Args:
        logits: The confidence logits dict from the confidence heads.
        distances: `[B, S, N, N]` representative-atom distances used for interface contact masks.
        input: The Boltz-1 input batch.
        pred_distogram_logits: `[B, N, N, num_bins]` trunk distogram logits for contact probabilities.
        coordinates: `[B, S, M, 3]` sampled coordinates used for pTM frames.
        ranking_plddt_weight: Weight on complex pLDDT in the combined confidence score.
        ranking_ptm_weight: Weight on pTM/ipTM in the combined confidence score.

    Returns:
        Mapping from score name to tensor (per-token metrics and per-sample complex/interface aggregates).
    """

    plddt = plddt_from_logits(logits["plddt_logits"])
    num_samples = coordinates.shape[-3]
    token_mask = insert_sample_axis(input.token_mask, num_samples, dim=-2).to(plddt)
    mean_plddt = weighted_mean(plddt, token_mask, backend=backend)

    molecule_type = _molecule_type_for_confidence(input, num_samples, plddt.device)
    is_ligand_token = (molecule_type == 3).to(plddt)
    chain_index = insert_sample_axis(input.chain_index, num_samples, dim=-2)
    is_different_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(plddt)
    iplddt_weight = interface_confidence_weights(distances, chain_index, is_ligand_token, token_mask)
    # A single-chain, ligand-free complex has an identically zero weight, so this is NaN by design: a monomer
    # has no interface and the interface pLDDT is undefined rather than zero.
    complex_iplddt = weighted_mean(plddt, iplddt_weight, backend=backend)

    pde = categorical_mean(logits["pde_logits"], end=32.0)
    pred_distogram_prob = torch.softmax(
        insert_sample_axis(pred_distogram_logits, num_samples, dim=-4),
        dim=-1,
    )
    contact_bins = min(20, pred_distogram_prob.shape[-1])
    prob_contact = pred_distogram_prob[..., :contact_bins].sum(dim=-1)
    token_index = insert_sample_axis(input.token_index, num_samples, dim=-2)
    same_token = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)) & (
        token_index.unsqueeze(-1) == token_index.unsqueeze(-2)
    )
    valid_pair = token_mask.bool().unsqueeze(-1) & token_mask.bool().unsqueeze(-2) & ~same_token
    token_pair_weight = prob_contact.masked_fill(~valid_pair, 0.0)
    complex_pde = weighted_mean(pde.flatten(-2), token_pair_weight.flatten(-2), backend=backend)
    interface_pair_weight = token_pair_weight.masked_fill(~is_different_chain.bool(), 0.0)
    complex_ipde = weighted_mean(pde.flatten(-2), interface_pair_weight.flatten(-2), backend=backend)

    scores = {
        "plddt": plddt,
        "pde": pde,
        "mean_plddt": mean_plddt,
        "complex_iplddt": complex_iplddt,
        "complex_pde": complex_pde,
        "complex_ipde": complex_ipde,
    }
    pae_logits = logits.get("pae_logits")
    if pae_logits is not None:
        pae = categorical_mean(pae_logits, end=32.0)
        frame_mask = _boltz1_coordinate_frames(coordinates, input=input)[1].to(pae_logits)
        pair_mask = frame_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * token_mask.unsqueeze(-1)
        interface_mask = pair_mask * is_different_chain.to(pae_logits)
        num_bins = pae_logits.shape[-1]
        bin_centers = (torch.arange(num_bins, device=pae_logits.device, dtype=pae_logits.dtype) + 0.5) * (
            32.0 / num_bins
        )
        num_residues = token_mask.sum(dim=-1, keepdim=True)
        tm_expected = tm_weight_from_pae(pae_logits, num_residues.unsqueeze(-1), bin_centers=bin_centers)
        is_ligand = (molecule_type == 3).to(pae_logits)
        is_protein = (molecule_type == 0).to(pae_logits)
        ligand_mask = interface_mask * (
            is_ligand.unsqueeze(-1) * is_protein.unsqueeze(-2) + is_protein.unsqueeze(-1) * is_ligand.unsqueeze(-2)
        )
        protein_mask = interface_mask * is_protein.unsqueeze(-1) * is_protein.unsqueeze(-2)
        chain_indices = torch.unique(input.chain_index).tolist()
        chain_pair_iptm = chain_pair_tm_scores(
            tm_expected, chain_index, chain_indices, pair_weights=pair_mask, backend=backend
        )
        ptm_scores = {
            "ptm": predicted_tm_score(tm_expected, pair_weights=pair_mask, backend=backend),
            "iptm": predicted_tm_score(tm_expected, pair_weights=interface_mask, backend=backend),
            "ligand_iptm": predicted_tm_score(tm_expected, pair_weights=ligand_mask, backend=backend),
            "protein_iptm": predicted_tm_score(tm_expected, pair_weights=protein_mask, backend=backend),
            "chain_pair_iptm": chain_pair_iptm,
        }
        scores["pae"] = pae
        scores.update(ptm_scores)
        ptm_for_ranking = torch.where(
            torch.isclose(ptm_scores["iptm"], torch.zeros_like(ptm_scores["iptm"])),
            ptm_scores["ptm"],
            ptm_scores["iptm"],
        )
        scores["confidence_score"] = (
            ranking_plddt_weight * (mean_plddt / 100.0) + ranking_ptm_weight * ptm_for_ranking
        ) / (ranking_plddt_weight + ranking_ptm_weight)
    else:
        scores["confidence_score"] = mean_plddt / 100.0
    return scores


def _boltz1_coordinate_frames(
    coordinates: Tensor,
    *,
    input: Boltz1Input,
    atom_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Construct canonical polymer and coordinate-dependent nonpolymer frames.

    Args:
        coordinates: `[B, S, M, 3]` coordinate realizations.
        input: The Boltz-1 input batch.
        atom_mask: Optional realization-specific resolved-atom mask.

    Returns:
        Frame indices `[B, S, N, 3]` and valid-frame mask `[B, S, N]`.
    """

    num_samples = coordinates.shape[-3]
    frame_index = (
        insert_sample_axis(input.frame_atom_index, num_samples, dim=-3)
        .to(
            device=coordinates.device,
            dtype=torch.long,
        )
        .clone()
    )
    chain_index = input.chain_index
    molecule_type = input.molecule_type
    atom_to_token = input.atom_to_token.long()
    atom_chain_index = torch.gather(chain_index, dim=-1, index=atom_to_token)
    structural_atom_mask = input.atom_mask.bool()
    if atom_mask is None:
        atom_mask = insert_sample_axis(structural_atom_mask, num_samples, dim=-2)
    else:
        atom_mask = atom_mask.to(device=coordinates.device, dtype=torch.bool)

    for batch_index in range(input.token_mask.shape[0]):
        ligand_chains = torch.unique(
            chain_index[batch_index][input.token_mask[batch_index].bool() & (molecule_type[batch_index] == 3)]
        ).tolist()
        for chain in ligand_chains:
            chain_tokens = torch.nonzero(
                input.token_mask[batch_index].bool()
                & (molecule_type[batch_index] == 3)
                & (chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            chain_atoms = torch.nonzero(
                structural_atom_mask[batch_index] & (atom_chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if len(chain_atoms) < 3:
                continue
            batch_coordinates = coordinates[batch_index][..., chain_atoms, :]
            distances = pairwise_distance(batch_coordinates)
            resolved = atom_mask[batch_index][..., chain_atoms]
            nearest = distances.masked_fill(
                ~(resolved.unsqueeze(-1) & resolved.unsqueeze(-2)),
                torch.inf,
            ).argsort(
                dim=-1
            )[..., :3]
            batch_frame_index = frame_index[batch_index]
            batch_frame_index[..., chain_tokens, :] = chain_atoms[nearest][..., (1, 0, 2)]

    flat_index = frame_index.flatten(-2, -1)
    frames = gather_atoms_by_index(coordinates, flat_index).unflatten(-2, (-1, 3))
    frame_resolved = torch.gather(atom_mask, dim=-1, index=flat_index).unflatten(-1, (-1, 3)).all(dim=-1)
    center_index = insert_sample_axis(input.token_center_atom_index, num_samples, dim=-2).to(
        device=coordinates.device,
        dtype=torch.long,
    )
    center_resolved = torch.gather(atom_mask, dim=-1, index=center_index)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    first_direction = first / (first_norm.unsqueeze(-1) + 1e-6)
    second_direction = second / (second_norm.unsqueeze(-1) + 1e-6)
    cosine = (first_direction * second_direction).sum(dim=-1)
    valid_geometry = (
        (cosine.abs() < 0.9063)
        & (first_norm > 1e-2)
        & (second_norm > 1e-2)
        & frame_resolved
        & center_resolved
        & insert_sample_axis(input.token_mask, num_samples, dim=-2).to(
            device=coordinates.device,
            dtype=torch.bool,
        )
    )
    return frame_index, valid_geometry


def _molecule_type_for_confidence(input: Boltz1Input, num_samples: int, device: torch.device) -> Tensor:
    r"""
    Add the confidence sample axis to the per-token molecule type classification.

    Args:
        input: The Boltz-1 input batch.
        num_samples: Number of samples per input.
        device: Target device.

    Returns:
        A `[B, S, N]` long tensor of molecule types.
    """

    return insert_sample_axis(input.molecule_type, num_samples, dim=-2).to(device=device, dtype=torch.long)


def _gather_boltz1_structure_target(
    target: Boltz1StructureTarget,
    input: Boltz1Input,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather canonical truth onto the model atom axis.
    """

    return gather_model_coordinates(
        target.coordinates, target.coordinate_mask, input.atom_index, input.atom_mask.to(torch.bool)
    )


# ===========================================================================
# Tensor boundary of the Boltz-1 model graph: typed input and output tensors,
# trunk representations, and the public pocket declaration.
# ===========================================================================


@dataclass(frozen=True)
class Boltz1Pocket:
    r"""
    One Boltz-1 unary pocket conditioning declaration.

    The encoding admits a single binder chain and an explicitly declared
    contact-residue list; the 6 A cutoff is the training-time contact definition and
    is not recomputed at fold time. Training-only sampling controls are not
    representable.

    Attributes:
        binder: The ``instance_id`` of the single binder chain.
        pocket: `ComponentRef` values naming the pocket contact residues.
    """

    binder: str
    pocket: tuple[ComponentRef, ...] = ()


@tensorclass
class Boltz1StructureTarget:
    r"""
    Full canonical atom-axis structure truth for Boltz-1 training.

    Local symmetry groups name rows on the model atom axis; all coordinate
    values live on the canonical system atom axis.

    Attributes:
        coordinates: Full canonical coordinates with shape
            ``(B, A_system, 3)``.
        coordinate_mask: Resolved canonical atom mask with shape
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
class Boltz1Input:
    r"""
    Input tensors consumed by [`Boltz1Model`][].

    The contract is Boltz-1 specific: residue-token geometry, optional MSA features, molecule/frame fields used
    by confidence ranking, and optional token-level ``pocket_feature`` one-hot with classes ``UNSPECIFIED``,
    ``UNSELECTED``, ``POCKET``, and ``BINDER``. When omitted, the model uses all ``UNSPECIFIED`` pocket
    conditioning. Tensor fields carry an explicit leading family batch axis;
    ``N`` is the token count and ``M`` the atom count. The tensorclass batch
    size is empty because the nested canonical system uses ragged pointer axes.

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
        atom_token_index: `[B, M]` token index of each atom.
        token_center_atom_index: `[B, N]` center-atom index per token.
        token_representative_atom_index: `[B, N]` representative-atom index per token.
        residue_type_features: `[B, N, vocab_size]` residue-type one-hot features.
        msa_profile: `[B, N, 72]` canonical MSA profile feature.
        deletion_mean: `[B, N]` mean-deletion feature.
        token_bonds: `[B, N, N]` inter-token bond indicator.
        molecule_type: `[B, N]` per-token molecule type.
        geometry_pair_index: `[B, 2, P]` atom pairs with molecular distance bounds.
        geometry_bond_mask: `[B, P]` mask of directly bonded geometry pairs.
        geometry_angle_mask: `[B, P]` mask of two-bond angle endpoint pairs.
        geometry_upper_bounds: `[B, P]` upper distance bounds in angstroms.
        geometry_lower_bounds: `[B, P]` lower distance bounds in angstroms.
        chiral_atom_index: `[B, 4, C]` chiral-center improper-dihedral atom indices.
        chiral_atom_positive: `[B, C]` positive-dihedral flag for each chiral-center constraint.
        stereo_bond_index: `[B, 4, S]` stereo-bond dihedral atom indices.
        stereo_bond_trans: `[B, S]` Trans flag for each declared control-atom tuple.
        planar_bond_index: `[B, 6, D]` planar double-bond atom indices.
        connected_chain_index: `[B, 2, K]` chain pairs joined by known covalent assembly bonds.
        connected_atom_index: `[B, 2, K]` atom pairs joined by known covalent assembly bonds.
        symmetric_chain_index: `[B, 2, Q]` same-entity chain-copy pairs.
        frame_atom_index: `[B, N, 3]` canonical polymer frame indices per token.
        cyclic_period: Optional `[B, N]` cyclic-period feature for relative positions.
        msa_tokens: Optional `[B, S, N]` MSA token grid.
        has_deletion: Optional `[B, S, N]` per-MSA deletion-present flag.
        deletion_value: Optional `[B, S, N]` per-MSA deletion count feature.
        msa_paired: Optional `[B, S, N]` cross-chain pairing flag.
        msa_mask: Optional `[B, S, N]` MSA validity mask.
        pocket_feature: Optional `[B, N, pocket_feature_size]` pocket one-hot conditioning.
        atom_index: Canonical atom index for each model atom (its slot's atom) with shape
            ``(B, M)`` and ``-1`` padding.
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
    token_center_atom_index: Tensor
    token_representative_atom_index: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    molecule_type: Tensor
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
    frame_atom_index: Tensor
    atom_index: Tensor
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    pocket_feature: Tensor | None = None


def normalize_boltz1_input(input: Boltz1Input, *, dtype: torch.dtype | None = None) -> Boltz1Input:
    """Return finite Boltz features; references are complete model conformers.

    Reference completeness is processor admission, not source observation
    completeness. Valid model slots and observed training targets remain distinct.
    """
    result = cast(Boltz1Input, cast(TensorClass, input).clone(recurse=False))
    atom_mask = input.atom_mask.to(torch.bool)
    token_mask = input.token_mask.to(torch.bool)
    result.reference_positions = project_model_coordinates(
        input.reference_positions, atom_mask, dtype=dtype, feature="Boltz1 reference positions"
    )
    for name, mask in (
        ("atom_features", atom_mask[..., None]),
        ("residue_type_features", token_mask[..., None]),
        ("msa_profile", token_mask[..., None]),
        ("deletion_mean", token_mask),
        ("token_bonds", token_mask[..., :, None] & token_mask[..., None, :]),
        ("pocket_feature", token_mask[..., None]),
    ):
        values = getattr(input, name)
        if values is not None:
            setattr(result, name, project_model_features(values, mask, dtype=dtype, feature=f"Boltz1 {name}"))
    if input.msa_tokens is not None:
        msa_available = token_mask.unsqueeze(-2) & torch.ones_like(input.msa_tokens, dtype=torch.bool)
        if input.msa_mask is not None:
            msa_available = msa_available & input.msa_mask.bool()
        result.msa_mask = msa_available
        result.msa_tokens = project_model_categories(
            input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="Boltz1 MSA tokens"
        )
        for name in ("has_deletion", "deletion_value", "msa_paired"):
            values = getattr(input, name)
            if values is not None:
                if not values.is_floating_point():
                    values = torch.where(msa_available, values, 0).to(torch.float64)
                setattr(
                    result, name, project_model_features(values, msa_available, dtype=dtype, feature=f"Boltz1 {name}")
                )
    return result


class _Boltz1TrunkOutput(NamedTuple):
    r"""
    Boltz-1 trunk representations returned by the recycling trunk.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


class Boltz1Output(TensorCollection):
    r"""
    Predictions from one Boltz-1 training denoising pass.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        augmented_target_coordinates: Training targets after rigid augmentation,
            with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor


class Boltz1SamplingOutput(TensorCollection):
    r"""
    Predictions from one Boltz-1 reverse-diffusion sampling pass.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    coordinates: Tensor


class Boltz1ConfidenceOutput(TensorCollection):
    r"""
    Raw Boltz-1 confidence predictions and their official derived values.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)``.
        distogram_logits: Detached trunk distogram logits with shape
            ``(B, N, N, D)``.
        plddt_logits: Per-token pLDDT logits with shape ``(B, S, N, D_p)``.
        pde_logits: PDE logits with shape ``(B, S, N, N, D_d)``.
        resolved_logits: Per-token logits in ``[unresolved, resolved]`` order
            with shape ``(B, S, N, 2)``.
        plddt: Expected per-token pLDDT values on the ``0--100`` scale with
            shape ``(B, S, N)``.
        pde: Expected PDE values with shape ``(B, S, N, N)``.
        mean_plddt: Mean complex pLDDT with shape ``(B, S)``.
        complex_iplddt: Interface-weighted pLDDT with shape ``(B, S)``.
        complex_pde: Distogram-contact-weighted PDE with shape ``(B, S)``.
        complex_ipde: Interface distogram-contact-weighted PDE with shape
            ``(B, S)``.
        confidence_score: Ranking confidence with shape ``(B, S)``.
        pae_logits: Optional PAE logits with shape ``(B, S, N, N, D_a)``.
        pae: Optional expected PAE values with shape ``(B, S, N, N)``.
        ptm: Optional predicted TM score with shape ``(B, S)``.
        iptm: Optional interface predicted TM score with shape ``(B, S)``.
        ligand_iptm: Optional protein-ligand predicted TM score with shape
            ``(B, S)``.
        protein_iptm: Optional protein-protein predicted TM score with shape
            ``(B, S)``.
        chain_pair_iptm: Optional ordered chain-pair predicted TM scores with
            shape ``(B, S, C, C)`` in sorted unique modeled canonical chain-ID order.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor
    plddt: Tensor
    pde: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    confidence_score: Tensor
    pae_logits: OptionalTensor
    pae: OptionalTensor
    ptm: OptionalTensor
    iptm: OptionalTensor
    ligand_iptm: OptionalTensor
    protein_iptm: OptionalTensor
    chain_pair_iptm: OptionalTensor


class Boltz1SamplingWithConfidenceOutput(NamedTuple):
    r"""
    Raw predictions from Boltz-1 coordinate sampling and confidence evaluation.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        confidence: Typed confidence predictions when the checkpoint contains a
            confidence head.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    confidence: Boltz1ConfidenceOutput | None


class Boltz1Representations(TensorCollection):
    r"""
    Raw Boltz-1 trunk representations surfaced by ``Boltz1Model.representations``.

    Attributes:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor


def _boltz1_coordinates(coordinates: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Project a complete model-state coordinate field at a public boundary."""
    if coordinates.dim() == 0 or coordinates.shape[-1] != 3:
        fail("value_invalid", RequestRef("coordinates", str(tuple(coordinates.shape))))
    return project_model_coordinates(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        dtype=dtype,
        feature="Boltz1 model-state coordinates",
    )
