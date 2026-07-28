r"""
Boltz-1 model.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass

from tensordict import tensorclass
from functools import partial
from typing import TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.molecular import AtomSite, ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    PackedAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedTokenToAtomDecoder,
)
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.window import gather_key_windows
from ...modules.attention import (
    PairBiasedSelfAttention,
    PerBlockPairBiasProjection,
    TriangleAttention,
    TriangleMultiplication,
)
from ...modules.backend import LayerNorm
from ...modules.diffusion.conditioning import PairwiseConditioning, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.dropout import StructuredDropout
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import PairSymmetrizedDistogramHead
from ...modules.head.tm import tm_weight_from_pae
from ...modules.potential import (
    indexed_distance,
    indexed_distance_with_gradient,
    indexed_signed_dihedral,
    indexed_signed_dihedral_with_gradient,
    linear_flat_bottom_derivative,
    linear_flat_bottom_energy,
)
from ...modules.transformer import ConditionedTransformerEncoder, TripleProductConditionedTransition
from ...modules.transition import Transition
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import Boltz1Config


class Boltz1Model(Model):
    r"""
    Predict biomolecular complex structures with Boltz 1.

    Structure training consumes separate [`Boltz1Input`][] and
    [`Boltz1StructureTarget`][] values. ``fold`` consumes inference Input only.

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import MolecularSystem, PolymerSequence
        >>> from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model, Boltz1Processor
        >>> sequence = PolymerSequence.from_one_letter("G")
        >>> system = MolecularSystem({"protein": sequence}, {"A": "protein"})
        >>> input = Boltz1Processor()(system)
        >>> config = Boltz1Config(
        ...     {
        ...         "single_size": 8,
        ...         "pair_size": 8,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "num_distogram_bins": 4,
        ...         "fourier_embedding_size": 8,
        ...         "recycling_steps": 1,
        ...         "input_embeddings": {
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_encoder_attention_heads": 1,
        ...         },
        ...         "trunk": {
        ...             "num_pairformer_blocks": 1,
        ...             "num_pair_attention_heads": 1,
        ...             "num_msa_blocks": 1,
        ...             "msa_size": 8,
        ...             "msa_head_dim": 4,
        ...             "num_msa_attention_heads": 1,
        ...             "outer_product_hidden_size": 4,
        ...             "msa_triangle_attention_hidden_size": 4,
        ...             "num_msa_triangle_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "pair_dropout": 0.0,
        ...             "msa_dropout": 0.0,
        ...             "msa_pair_dropout": 0.0,
        ...         },
        ...         "denoiser": {
        ...             "num_token_encoder_blocks": 1,
        ...             "num_token_encoder_attention_heads": 1,
        ...             "num_atom_encoder_blocks": 1,
        ...             "num_atom_decoder_blocks": 1,
        ...             "num_atom_attention_heads": 1,
        ...             "atom_hidden_size": 8,
        ...             "atom_pair_hidden_size": 4,
        ...             "num_query_atoms": 4,
        ...             "num_key_atoms": 4,
        ...             "num_conditioning_transitions": 1,
        ...         },
        ...         "confidence": {
        ...             "num_distogram_bins": 4, "num_plddt_bins": 4,
        ...             "num_pae_bins": 4, "num_pde_bins": 4,
        ...         },
        ...     }
        ... )
        >>> model = Boltz1Model(config).eval()
        >>> with torch.inference_mode():
        ...     coordinates = model.sample(
        ...         input, num_samples=1, num_sampling_steps=1, recycling_steps=1,
        ...     )
        >>> tuple(coordinates.shape)
        (1, 1, 32, 3)
    """

    config_class = Boltz1Config
    config: Boltz1Config

    def __init__(self, config: Boltz1Config | None = None) -> None:
        config = Boltz1Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps

        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config = config.input_embeddings
        input_size = (
            single_size + 2 * input_embeddings_config.vocab_size + 1 + input_embeddings_config.pocket_feature_size
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
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
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
                vocab_size=input_embeddings_config.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.msa_pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.msa_triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
                backend="auto",
            )
        self.pairformer_stack: Boltz1PairformerStack | None = None
        self.pair_update_stack: Boltz1TrianglePairUpdateStack | None = None
        if trunk.update_single:
            self.pairformer_stack = Boltz1PairformerStack(
                num_blocks=trunk.num_pairformer_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
        else:
            self.pair_update_stack = Boltz1TrianglePairUpdateStack(
                num_blocks=trunk.num_pairformer_blocks,
                pair_size=pair_size,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                dropout=trunk.pair_dropout,
                backend="auto",
            )
        self.distogram_head = PairSymmetrizedDistogramHead(
            pair_size=pair_size,
            num_bins=config.num_distogram_bins,
        )

        self.denoiser_conditioning = Boltz1DenoiserConditioning(
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
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
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
        self.confidence_head: _Boltz1ConfidenceHead | None = None
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
                    vocab_size=input_embeddings_config.vocab_size,
                    use_paired_feature=trunk.use_paired_feature,
                    num_msa_blocks=trunk.num_msa_blocks,
                    outer_product_hidden_size=trunk.outer_product_hidden_size,
                    msa_dropout=trunk.msa_dropout,
                    msa_pair_dropout=trunk.msa_pair_dropout,
                    pair_dropout=trunk.pair_dropout,
                    update_single=trunk.update_single,
                    num_msa_attention_heads=trunk.num_msa_attention_heads,
                    msa_head_dim=trunk.msa_head_dim,
                    msa_triangle_attention_hidden_size=trunk.msa_triangle_attention_hidden_size,
                    num_msa_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
                    triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
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
                    triangle_attention_hidden_size=confidence.triangle_attention_hidden_size,
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
        synchronize_noise_levels: bool = True,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Output:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model-visible structure and evidence features.
            target: Complete pre-crop structure truth.
            num_samples: Number ``S`` of independently corrupted copies per input.
            synchronize_noise_levels: Whether samples from one example share sigma.
            coordinate_augmentation: Whether to apply random rigid augmentation.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
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

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        conditioning = self._build_denoiser_conditioning(representations, input)
        target_coordinates, _ = _gather_boltz1_structure_target(
            target,
            input,
        )
        denoised_coordinates, augmented_target_coordinates, noise_level = self.coordinate_diffusion.denoise_training(
            target_coordinates,
            input.atom_mask,
            multiplicity=num_samples,
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
            coordinates=_unflatten_sample_axis(
                denoised_coordinates,
                batch_size,
                num_samples,
            ),
            distogram_logits=representations.distogram_logits,
            augmented_target_coordinates=_unflatten_sample_axis(
                augmented_target_coordinates,
                batch_size,
                num_samples,
            ),
            noise_level=noise_level.reshape(batch_size, num_samples),
            batch_size=[batch_size],
        )

    def confidence(
        self,
        input: Boltz1Input,
        *,
        num_samples: int = 1,
        recycling_steps: int | None = None,
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
        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        output = self._predict(
            input,
            recycling_steps=recycling_steps,
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
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Representations:
        r"""
        Return the final single and pair trunk representations.

        Args:
            input: Model features with token axis length ``N``.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            generator: Optional generator advanced by MSA row sampling.

        Returns:
            Single representations with shape ``(B, N, C_single)`` and pair
            representations with shape ``(B, N, N, C_pair)``. Gradients are
            retained when autograd is enabled.
        """

        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
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
        recycling_steps: int | None = None,
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
    ) -> Tensor:
        r"""
        Sample raw atom coordinates with explicit ``(B, S, A, 3)`` axes.

        Args:
            input: Model-visible structure and evidence features.
            num_samples: Number of coordinate samples.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
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

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        with torch.no_grad():
            representations = self._run_trunk(
                input,
                recycling_steps=recycling_steps,
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
        return _unflatten_sample_axis(
            coordinates,
            input.token_mask.shape[0],
            num_samples,
        )

    @torch.inference_mode()
    def fold(
        self,
        input: Boltz1Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
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
    ) -> FoldResult:
        r"""
        Predict structures for one molecular system with Boltz 1.

        Args:
            input: Processor output for one system, with batch size ``B = 1`` and
                molecular identities aligned to its atom and token axes.
            num_samples: Number of coordinate samples.
            generator: Optional generator for MSA row sampling, initial
                coordinates, churn noise, rigid augmentations, and
                potential-guided resampling.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
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

        Returns:
            One generation-ordered prediction per sample. Structure, PAE, and PDE
            values are in angstroms; pLDDT is on the ``0--100`` scale.

        Raises:
            ValueError: If ``num_samples`` is not positive, the batch contains
                more than one system, or molecular identities are absent.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.token_mask.shape[0] != 1:
            raise ValueError("Boltz1Model.fold accepts one molecular system per call")
        if input.layout is None:
            raise ValueError("Boltz1Model.fold requires the canonical layout produced by its Processor")

        output = self._predict(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            num_samples=num_samples,
            generator=generator,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
        )
        return _decode_boltz1_fold_result(output, input)

    def _predict(
        self,
        input: Boltz1Input,
        *,
        recycling_steps: int | None = None,
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
    ) -> _Boltz1FoldOutput:
        r"""
        Run the trunk and sampler and return the full Boltz-1 output.

        Runs the recycling trunk once, draws ``num_samples`` reverse-diffusion samples, and (when the
        confidence head is present) scores them into confidence logits and ranking metrics. Sample-axis
        tensors are unflattened to a leading ``(batch, num_samples, ...)`` shape. This private tensor path
        does not declare a public fold task.

        Args:
            input: The Boltz-1 input batch.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
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
            A [`_Boltz1FoldOutput`][ofoldx.models.boltz1.model._Boltz1FoldOutput]
            with sampled coordinates, distogram logits, and optional typed
            confidence predictions.

        """

        with torch.no_grad():
            representations = self._run_trunk(
                input,
                recycling_steps=recycling_steps,
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
        coordinates = _unflatten_sample_axis(
            diffusion_coords.detach(),
            batch_size,
            num_samples,
        )
        distogram_logits = representations.distogram_logits.detach()
        confidence_output: Boltz1ConfidenceOutput | None = None
        confidence_head = self.confidence_head
        if confidence_head is not None:
            confidence_values = confidence_head(
                representations.single_inputs.detach(),
                representations.single.detach(),
                representations.pair.detach(),
                diffusion_coords.detach(),
                input=input,
                pred_distogram_logits=distogram_logits,
                single_diffusion=diffusion_token_repr,
                multiplicity=num_samples,
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
            confidence_values = {
                name: _unflatten_sample_axis(value, batch_size, num_samples)
                for name, value in confidence_values.items()
            }
            confidence_output = Boltz1ConfidenceOutput(
                coordinates=coordinates,
                distogram_logits=distogram_logits,
                plddt_logits=confidence_values["plddt_logits"],
                pde_logits=confidence_values["pde_logits"],
                resolved_logits=confidence_values["resolved_logits"],
                plddt=confidence_values["plddt"],
                pde=confidence_values["pde"],
                complex_plddt=confidence_values["complex_plddt"],
                complex_iplddt=confidence_values["complex_iplddt"],
                complex_pde=confidence_values["complex_pde"],
                complex_ipde=confidence_values["complex_ipde"],
                confidence_score=confidence_values["confidence_score"],
                pae_logits=confidence_values.get("pae_logits"),
                pae=confidence_values.get("pae"),
                ptm=confidence_values.get("ptm"),
                iptm=confidence_values.get("iptm"),
                ligand_iptm=confidence_values.get("ligand_iptm"),
                protein_iptm=confidence_values.get("protein_iptm"),
                chain_pair_iptm=confidence_values.get("chain_pair_iptm"),
                batch_size=[batch_size],
            )
        return _Boltz1FoldOutput(
            coordinates=coordinates,
            distogram_logits=distogram_logits,
            confidence=confidence_output,
            batch_size=[batch_size],
        )

    def _run_trunk(
        self,
        input: Boltz1Input,
        *,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _Boltz1TrunkOutput:
        r"""
        Iterate the recycling loop: recycle-project the running single/pair, fold in the MSA module (when
        present), and run the Pairformer for ``recycling_steps`` total passes.

        Args:
            input: The Boltz-1 input batch.
            recycling_steps: Total number of trunk passes; defaults to the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            generator: Optional generator advanced by each MSA row sample.

        Returns:
            The final trunk [`_Boltz1TrunkOutput`][ofoldx.models.boltz1.model._Boltz1TrunkOutput].

        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        recycling_steps = operator.index(recycling_steps)
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(input)
        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if input.msa_tokens is not None:
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
                    pair = cast(Boltz1TrianglePairUpdateStack, self.pair_update_stack)(
                        pair,
                        pair_mask=pair_mask,
                    )

        return _Boltz1TrunkOutput(
            single_inputs=single_inputs,
            single=cast(Tensor, single),
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
            batch_size=[single_inputs.shape[0]],
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
            num_samples: Number of samples (sampler ``multiplicity``).
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
            multiplicity=num_samples,
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
            feats=_boltz1_steering_features(input, atom_features=input.atom_features),
        )
        if return_token_repr:
            return cast(tuple[Tensor, Tensor], sample)
        return cast(Tensor, sample), None

    def _build_denoiser_conditioning(
        self,
        representations: _Boltz1TrunkOutput,
        input: Boltz1Input,
    ) -> dict[str, Tensor]:
        r"""
        Precompute the per-structure diffusion conditioning reused by every denoiser call.

        The conditioning depends only on the trunk representations and reference geometry, so it is computed
        once and shared by the training single-step denoise and every reverse-diffusion sampling step.

        Args:
            representations: The trunk representations from
                [`_run_trunk`][ofoldx.models.boltz1.model.Boltz1Model._run_trunk].
            input: The Boltz-1 input batch.

        Returns:
            The precomputed diffusion-conditioning dict (atom states and per-block attention biases).
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
        self.atom_encoder: Boltz1AtomEncoder | None = None
        self.atom_pair_bias_projection: PerBlockPairBiasProjection | None = None
        self.atom_attention_encoder: PackedAtomToTokenEncoder | None = None
        if use_atom_encoder:
            self.atom_encoder = Boltz1AtomEncoder(
                atom_feature_size=atom_feature_size,
                atom_hidden_size=atom_hidden_size,
                atom_pair_hidden_size=atom_pair_hidden_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
            )
            self.atom_pair_bias_projection = PerBlockPairBiasProjection(
                pair_size=atom_pair_hidden_size,
                num_blocks=num_atom_encoder_blocks,
                num_attention_heads=num_atom_encoder_attention_heads,
                backend=backend,
            )
            self.atom_attention_encoder = PackedAtomToTokenEncoder(
                atom_hidden_size=atom_hidden_size,
                token_hidden_size=single_size,
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                num_blocks=num_atom_encoder_blocks,
                num_attention_heads=num_atom_encoder_attention_heads,
                backend=backend,
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
        atom_encoder = self.atom_encoder
        atom_pair_bias_projection = self.atom_pair_bias_projection
        atom_attention_encoder = self.atom_attention_encoder
        if atom_encoder is None:
            token_from_atom = torch.zeros(
                (*residue_type_features.shape[:-1], self.single_size),
                device=residue_type_features.device,
                dtype=residue_type_features.dtype,
            )
        else:
            atom_query, atom_conditioning, atom_pair = atom_encoder(
                atom_features,
                reference_positions,
                atom_mask,
                reference_space_id,
            )
            token_from_atom = cast(PackedAtomToTokenEncoder, atom_attention_encoder)(
                atom_query,
                atom_conditioning,
                cast(PerBlockPairBiasProjection, atom_pair_bias_projection)(atom_pair),
                atom_mask,
                atom_to_token,
                residue_type_features.shape[-2],
            )
        pocket_size = self.input_size - self.single_size - 2 * residue_type_features.shape[-1] - 1
        if pocket_feature is None:
            pocket_feature = residue_type_features.new_zeros(*residue_type_features.shape[:-1], pocket_size)
            pocket_feature[..., 0] = 1
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
        vocab_size: int = 33,
        use_paired_feature: bool = False,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.use_paired_feature = use_paired_feature
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.msa_projection = nn.Linear(
            vocab_size + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            Boltz1MsaBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_head_dim=msa_head_dim,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

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
        if msa_max_depth is not None:
            msa_max_depth = operator.index(msa_max_depth)
            if msa_max_depth < 1:
                raise ValueError("msa_max_depth must be positive")
            indices = torch.randperm(
                msa_tokens.shape[1],
                device=msa_tokens.device,
                generator=generator,
            )[:msa_max_depth]
            msa_tokens = msa_tokens[:, indices]
            has_deletion = has_deletion[:, indices]
            deletion_value = deletion_value[:, indices]
            is_paired = is_paired[:, indices]
            msa_mask = msa_mask[:, indices]
        features = [
            F.one_hot(msa_tokens, num_classes=self.vocab_size).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask)
        for block in self.blocks:
            pair, msa = block(pair, msa, pair_mask, msa_mask)
        return pair


class Boltz1MsaBlock(nn.Module):
    r"""
    One Boltz-1 MSA block with scalar dropout per MSA row.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_dim=msa_head_dim,
            backend=backend,
        )
        self.msa_transition = Transition(msa_size, expansion=expansion, backend=backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = Boltz1TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )
        self.msa_dropout = StructuredDropout(msa_dropout, broadcast_dims=(-2, -1))

    def forward(
        self,
        pair: Tensor,
        msa: Tensor,
        pair_mask: Tensor,
        msa_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        msa = msa + self.msa_dropout(self.pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        msa = msa + self.msa_transition(msa)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return pair, msa


class Boltz1PairformerStack(nn.Module):
    r"""
    Homogeneous stack of Boltz-1 Pairformer blocks.
    """

    def __init__(
        self,
        num_blocks: int = 48,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            Boltz1PairformerBlock(
                single_size=single_size,
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                expansion=expansion,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class Boltz1PairformerBlock(nn.Module):
    r"""
    One Boltz-1 Pairformer block.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_update = Boltz1TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.pair_attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pair = self.pair_update(pair, pair_mask=pair_mask)
        single = single + self.pair_attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return single, pair


class Boltz1TrianglePairUpdateStack(nn.Module):
    r"""
    Homogeneous stack of Boltz-1 pair-only triangle update blocks.
    """

    def __init__(
        self,
        num_blocks: int = 48,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            Boltz1TrianglePairUpdateBlock(
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            pair = block(pair, pair_mask=pair_mask)
        return pair


class Boltz1TrianglePairUpdateBlock(nn.Module):
    r"""
    Boltz-1 pair update with one dropout scalar per pair row or column.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            backend=backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.columnwise_dropout = StructuredDropout(dropout, broadcast_dims=(-3, -1))
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask=pair_mask))
        end_pair_mask = None if pair_mask is None else pair_mask.transpose(-1, -2)
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        ).transpose(-3, -2)
        pair = pair + self.columnwise_dropout(ending_update)
        return pair + self.transition(pair)


class Boltz1AtomEncoder(nn.Module):
    r"""
    Build atom single states and windowed atom-pair states from reference geometry.
    """

    def __init__(
        self,
        atom_feature_size: int = 388,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.left_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.right_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_input_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_hidden_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_output_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )

    def feed_forward(self, pair: Tensor) -> Tensor:
        pair = self.feed_forward_input_projection(F.relu(pair))
        pair = self.feed_forward_hidden_projection(F.relu(pair))
        return self.feed_forward_output_projection(F.relu(pair))

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        uid_keys = to_keys(space_id.unsqueeze(-1)).view(batch, num_windows, 1, num_key_atoms)
        valid = (mask_queries & mask_keys & (uid_queries == uid_keys)).to(distance_vectors.dtype).unsqueeze(-1)

        single_windows = F.relu(pad_atoms(atom_single))
        atom_pair = encode_atom_pair(
            self,
            distance_vectors,
            single_windows.view(batch, num_windows, num_query_atoms, 1, -1),
            to_keys(single_windows).view(batch, num_windows, 1, num_key_atoms, -1),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=None,
            prefix_ndim=1,
        )
        return atom_query, atom_single, atom_pair


class Boltz1DenoiserConditioning(nn.Module):
    r"""
    Precompute the diffusion conditioning shared across denoiser calls (Boltz DiffusionConditioning).
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
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_encoder = Boltz1DenoiserAtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.atom_decoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
        )
        self.token_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=pair_size,
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            backend=backend,
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
        atom_enc_bias = self.atom_encoder_pair_bias_projection(atom_pair)
        atom_dec_bias = self.atom_decoder_pair_bias_projection(atom_pair)
        token_encoder_bias = self.token_encoder_pair_bias_projection(pair)
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_encoder_bias": token_encoder_bias,
        }


class Boltz1Denoiser(nn.Module):
    r"""
    Boltz-1 EDM score network with the wider Boltz-1 ``s_inputs`` conditioning width.
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
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 8,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = Boltz1SingleConditioning(
            input_size=input_size,
            single_size=single_size,
            fourier_embedding_size=fourier_embedding_size,
            num_conditioning_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_attention_encoder = PackedDenoisingAtomToTokenEncoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.single_to_token_norm = LayerNorm(hidden, backend=backend)
        self.single_to_token_projection = nn.Linear(hidden, hidden, bias=False)
        self.token_encoder = ConditionedTransformerEncoder(
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            backend=backend,
        )
        self.token_norm = LayerNorm(hidden, backend=backend)
        self.atom_attention_decoder = PackedTokenToAtomDecoder(
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
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
        token_repr = token_repr + self.single_to_token_projection(self.single_to_token_norm(single))
        token_repr = self.token_encoder(
            token_repr,
            single,
            conditioning["token_encoder_bias"].repeat_interleave(multiplicity, 0),
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
        self.denoiser = denoiser
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
        normal = torch.empty(batch_size, device=device, dtype=dtype).normal_(generator=generator)
        return self.sigma_data * (log_mean + log_std * normal).exp()

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def sample_schedule(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        steps = operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps)
        if steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        inv_rho = 1 / self.rho
        # `steps` is the number of sampling steps (transitions); the trailing pad adds the final sigma=0.
        # Clamp the divisor so a single step mirrors the AF3 scheduler (ramp [0] -> a lone max_noise_level, then 0)
        # instead of dividing by zero; for steps >= 2 the divisor is steps - 1 exactly, so the schedule is unchanged.
        ramp = torch.arange(steps, device=device, dtype=dtype) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def preconditioned_network_forward(
        self,
        noisy_coordinates: Tensor,
        sigma: Tensor | float,
        *,
        multiplicity: int = 1,
        return_token_repr: bool = False,
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch = noisy_coordinates.shape[0]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device, dtype=noisy_coordinates.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
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
        multiplicity: int = 1,
        synchronize_noise_levels: bool = True,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        generator: torch.Generator | None = None,
        single_inputs: Tensor,
        single_trunk: Tensor,
        conditioning: dict[str, Tensor],
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype, device = single_trunk.dtype, single_trunk.device
        target = target_coordinates.to(dtype=dtype, device=device).repeat_interleave(
            multiplicity,
            0,
        )
        # Centre and rigid-augment over ALL padded atoms (atom_pad_mask), matching upstream
        # ``center_random_augmentation``. The resolved-atom mask is a loss/alignment concern applied by the
        # criterion, not part of the corruption equation.
        augmentation_mask = atom_mask.to(dtype=dtype, device=device).repeat_interleave(multiplicity, 0)
        sample_batch_size = target.shape[0]
        noise_batch_size = target_coordinates.shape[0] if synchronize_noise_levels else sample_batch_size
        noise_level = self.noise_distribution(
            noise_batch_size,
            device=device,
            dtype=dtype,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.repeat_interleave(multiplicity)
        augmented_target_coordinates = self._augment_target(
            target,
            augmentation_mask,
            coordinate_augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise = torch.empty_like(augmented_target_coordinates).normal_(generator=generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[:, None, None] * noise
        denoised_coordinates = self.preconditioned_network_forward(
            noisy_coordinates,
            noise_level,
            multiplicity=multiplicity,
            atom_mask=atom_mask,
            return_token_repr=False,
            single_inputs=single_inputs,
            single_trunk=single_trunk,
            conditioning=conditioning,
            atom_to_token=atom_to_token,
            token_mask=token_mask,
        )
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
        weight = mask[..., None]
        centroid = (coords * weight).sum(-2, keepdim=True) / weight.sum(-2, keepdim=True).clamp_min(1.0)
        coords = coords - centroid
        if not coordinate_augmentation:
            return coords
        rotation = uniform_random_rotation(
            coords.shape[0], device=coords.device, dtype=coords.dtype, generator=generator
        )
        translation = torch.empty(coords.shape[0], 1, 3, device=coords.device, dtype=coords.dtype).normal_(
            generator=generator
        )
        return torch.einsum("bmd,bds->bms", coords, rotation) + translation_scale * translation

    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        multiplicity: int = 1,
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
        potentials = _boltz1_potentials(physical_guidance) if feynman_kac or physical_guidance else ()
        requested_multiplicity = multiplicity
        effective_multiplicity = multiplicity * num_particles if feynman_kac else multiplicity
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(effective_multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)

        def randn() -> Tensor:
            r"""
            Draw standard-normal noise of the expanded coordinate ``shape`` on the sampler device/generator.
            """

            return torch.randn(shape, device=device, dtype=dtype, generator=generator)

        coords = sigmas[0] * randn()
        if feynman_kac:
            num_groups = atom_mask.shape[0] * requested_multiplicity
            energy_traj = torch.empty((expanded_atom_mask.shape[0], 0), device=device, dtype=dtype)
            resample_weights = torch.ones(num_groups, num_particles, device=device, dtype=dtype)
        else:
            energy_traj = torch.empty((0, 0), device=device, dtype=dtype)
            resample_weights = torch.empty((0, 0), device=device, dtype=dtype)
        if physical_guidance:
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
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0], device=device, dtype=coords.dtype, generator=generator
            )
            translation = torch.randn(
                expanded_atom_mask.shape[0],
                1,
                3,
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

            sigma_hat = sigma_current * (1 + gamma)
            steering_t = 1.0 - (index / max(len(sigmas) - 1, 1))
            # Compute schedule-derived churn and reverse-step scalars in float64 to prevent cumulative rounding
            # drift. The score network still receives float32 ``sigma_hat``; scalar tensors stay on-device so the
            # sampling loop remains graph-break-free.
            sigma_current_hp = sigma_current.double()
            sigma_hat_hp = sigma_current_hp * (1 + gamma.double())
            sigma_next_hp = sigma_next.double()
            noise_var = self.noise_scale**2 * (sigma_hat_hp**2 - sigma_current_hp**2)
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)).to(coords.dtype) * randn()
            with torch.no_grad():
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
                token_a = token_a.detach()
            else:
                denoised = cast(Tensor, denoised_out)

            should_resample = feynman_kac and (
                (index % feynman_kac_resampling_interval == 0 and noise_var > 0) or index == len(sigmas) - 2
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
                    (ll_difference + feynman_kac_lambda * log_g).reshape(-1, num_particles),
                    dim=1,
                )

            if physical_guidance and index < len(sigmas) - 2:
                guidance_update = torch.zeros_like(denoised)
                for guidance_step in range(num_guidance_steps):
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
                scaled_guidance_update = guidance_update * -step_scale * (sigma_next - sigma_hat) / sigma_hat

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
            if return_token_repr:
                output_token_feature_update = self.output_token_feature_update
                if output_token_feature_update is not None:
                    if token_repr is None:
                        token_repr = torch.zeros_like(cast(Tensor, token_a))
                    token_a = cast(Tensor, token_a)
                    sigma = torch.full((token_a.shape[0],), sigma_hat, device=token_a.device, dtype=token_a.dtype)
                    token_repr = output_token_feature_update(
                        noise_level=self.preconditioner.c_noise(sigma),
                        accumulated=token_repr,
                        next_repr=token_a,
                    )
                elif token_a is not None:
                    token_repr = token_a
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy,
                    target=denoised,
                    weight=expanded_atom_mask,
                )
            derivative = (noisy - denoised) / sigma_hat_hp
            coords = (noisy + step_scale * (sigma_next_hp - sigma_hat_hp) * derivative).to(coords.dtype)
        return (coords, token_repr) if return_token_repr else coords


class Boltz1DenoiserAtomEncoder(nn.Module):
    r"""
    Build trunk-conditioned atom states and atom-pair biases for coordinate denoising.
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
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.single_trunk_norm = LayerNorm(single_size)
        self.single_trunk_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.pair_trunk_norm = LayerNorm(pair_size)
        self.pair_trunk_projection = nn.Linear(
            pair_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.left_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.right_atom_projection = nn.Linear(
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_input_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_hidden_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.feed_forward_output_projection = nn.Linear(
            atom_pair_hidden_size,
            atom_pair_hidden_size,
            bias=False,
        )

    def feed_forward(self, pair: Tensor) -> Tensor:
        pair = self.feed_forward_input_projection(F.relu(pair))
        pair = self.feed_forward_hidden_projection(F.relu(pair))
        return self.feed_forward_output_projection(F.relu(pair))

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        *,
        atom_to_token: Tensor,
        single_trunk: Tensor,
        pair_trunk: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        uid_keys = to_keys(space_id.unsqueeze(-1)).view(batch, num_windows, 1, num_key_atoms)
        valid = (mask_queries & mask_keys & (uid_queries == uid_keys)).to(distance_vectors.dtype).unsqueeze(-1)

        single_proj = self.single_trunk_projection(self.single_trunk_norm(single_trunk))
        atom_single = atom_single + single_proj.gather(
            1, atom_to_token.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1])
        )
        pair_from_trunk = self.pair_trunk_projection(self.pair_trunk_norm(pair_trunk))
        token_windows = pad_atoms(atom_to_token)
        query_token = token_windows.view(batch, num_windows, num_query_atoms)
        key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
        batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
        pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
        pair_bias = pair_gathered * mask_keys.unsqueeze(-1).to(distance_vectors.dtype)

        single_windows = F.relu(pad_atoms(atom_single))
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


class Boltz1SingleConditioning(nn.Module):
    r"""
    Noise-condition the concatenated trunk-single and single-input representation for Boltz-1.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_conditioning_transitions: int = 2,
        transition_expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(single_size + input_size, backend=backend)
        self.single_projection = nn.Linear(single_size + input_size, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size, backend=backend)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=transition_expansion, backend=backend)
            for _ in range(num_conditioning_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
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


class Boltz1OutputTokenFeatureUpdate(nn.Module):
    r"""
    Accumulate the diffusion-token representation across reverse-diffusion steps for confidence conditioning.
    """

    def __init__(self, *, token_size: int = 384, fourier_embedding_size: int = 256) -> None:
        super().__init__()
        hidden = 2 * token_size
        self.next_norm = LayerNorm(hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size)
        self.transition = TripleProductConditionedTransition(hidden, hidden + fourier_embedding_size)

    def forward(self, *, noise_level: Tensor, accumulated: Tensor, next_repr: Tensor) -> Tensor:
        next_repr = self.next_norm(next_repr)
        fourier = self.fourier_norm(self.noise_fourier(noise_level))
        fourier = fourier.unsqueeze(1).expand(-1, next_repr.shape[1], -1)
        conditioning = torch.cat([accumulated, fourier], dim=-1)
        return accumulated + self.transition(next_repr, conditioning)


class _Boltz1ConfidenceHead(nn.Module):
    r"""
    Shared sampled-coordinate conditioning and confidence-output tail.
    """

    diffusion_single_norm: nn.Module | None
    diffusion_single_projection: nn.Linear | None
    single_to_pair_row: nn.Linear
    single_to_pair_col: nn.Linear
    single_to_pair_product_left: nn.Linear | None
    single_to_pair_product_right: nn.Linear | None
    single_to_pair_product_out: nn.Linear | None
    boundaries: Tensor
    distance_embedding: nn.Embedding
    output_head: Boltz1ConfidenceOutputHead

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int,
        pair_size: int,
        ranking_plddt_weight: float,
        ranking_ptm_weight: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.single_size = single_size
        self.pair_size = pair_size
        self.ranking_plddt_weight = ranking_plddt_weight
        self.ranking_ptm_weight = ranking_ptm_weight

    def _initialize_sample_conditioning(
        self,
        *,
        num_distogram_bins: int,
        max_distance: float,
        add_single_to_pair_product: bool,
        use_single_diffusion: bool,
        backend: str,
    ) -> None:
        if use_single_diffusion:
            self.diffusion_single_norm = LayerNorm(2 * self.single_size, backend=backend)
            self.diffusion_single_projection = nn.Linear(2 * self.single_size, self.single_size, bias=False)
        else:
            self.diffusion_single_norm = None
            self.diffusion_single_projection = None

        self.single_to_pair_row = nn.Linear(self.input_size, self.pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(self.input_size, self.pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(self.input_size, self.pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(self.input_size, self.pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(self.pair_size, self.pair_size, bias=False)
        else:
            self.single_to_pair_product_left = None
            self.single_to_pair_product_right = None
            self.single_to_pair_product_out = None

        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, self.pair_size)

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        input: Boltz1Input,
        pred_distogram_logits: Tensor,
        single_diffusion: Tensor | None = None,
        multiplicity: int = 1,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        single_inputs, single, pair = self._prepare_representations(single_inputs, single, pair, input=input)

        single = single.repeat_interleave(multiplicity, 0)
        single_inputs = single_inputs.repeat_interleave(multiplicity, 0)
        if self.diffusion_single_projection is not None and self.diffusion_single_norm is not None:
            single = single + self.diffusion_single_projection(
                self.diffusion_single_norm(cast(Tensor, single_diffusion))
            )

        pair = pair.repeat_interleave(multiplicity, 0)
        pair = pair + self.single_to_pair_row(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        if (
            self.single_to_pair_product_left is not None
            and self.single_to_pair_product_right is not None
            and self.single_to_pair_product_out is not None
        ):
            pair = pair + self.single_to_pair_product_out(
                self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
                * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
            )

        representative_atom_index = input.token_distogram_atom_index.repeat_interleave(
            multiplicity,
            0,
        )
        representative_coordinates = coordinates.gather(
            1,
            representative_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]),
        )
        distances = torch.cdist(representative_coordinates, representative_coordinates)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = input.token_mask.repeat_interleave(multiplicity, 0)
        confidence_pair_mask = pair_mask_from_token_mask(token_mask)
        single, pair = self._refine_representations(
            single,
            pair,
            single_inputs=single_inputs,
            input=input,
            pair_mask=confidence_pair_mask,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        logits = self.output_head(single, pair)
        logits.update(
            _boltz1_confidence_scores(
                logits,
                distances=distances,
                input=input,
                pred_distogram_logits=pred_distogram_logits,
                coordinates=coordinates,
                multiplicity=multiplicity,
                ranking_plddt_weight=self.ranking_plddt_weight,
                ranking_ptm_weight=self.ranking_ptm_weight,
            )
        )
        return logits

    def _prepare_representations(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        *,
        input: Boltz1Input,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

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
        raise NotImplementedError


class Boltz1TrunkImitationConfidenceHead(_Boltz1ConfidenceHead):
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
        vocab_size: int = 33,
        use_paired_feature: bool = False,
        num_msa_blocks: int = 4,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        msa_pair_dropout: float = 0.25,
        pair_dropout: float = 0.25,
        update_single: bool = True,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        msa_triangle_attention_hidden_size: int = 32,
        num_msa_triangle_attention_heads: int = 4,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        relative_position_size: int = 139,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            input_size=input_size,
            single_size=single_size,
            pair_size=pair_size,
            ranking_plddt_weight=ranking_plddt_weight,
            ranking_ptm_weight=ranking_ptm_weight,
        )
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

        self._initialize_sample_conditioning(
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
                vocab_size=vocab_size,
                use_paired_feature=use_paired_feature,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=msa_pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_head_dim=msa_head_dim,
                triangle_attention_hidden_size=msa_triangle_attention_hidden_size,
                num_triangle_attention_heads=num_msa_triangle_attention_heads,
                backend=backend,
            )
        self.pairformer_stack: Boltz1PairformerStack | None = None
        self.pair_update_stack: Boltz1TrianglePairUpdateStack | None = None
        if update_single:
            self.pairformer_stack = Boltz1PairformerStack(
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        else:
            self.pair_update_stack = Boltz1TrianglePairUpdateStack(
                num_blocks=num_blocks,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        self.final_single_norm = LayerNorm(single_size, backend=backend)
        self.final_pair_norm = LayerNorm(pair_size, backend=backend)
        self.output_head = Boltz1ConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            compute_pae=compute_pae,
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
        if input.msa_tokens is not None:
            pair = pair + cast(Boltz1MsaTrunk, self.msa_trunk)(
                pair,
                single_inputs,
                msa_tokens=input.msa_tokens,
                has_deletion=cast(Tensor, input.has_deletion),
                deletion_value=cast(Tensor, input.deletion_value),
                is_paired=cast(Tensor, input.msa_paired),
                msa_mask=cast(Tensor, input.msa_mask),
                token_mask=input.token_mask,
                msa_max_depth=msa_max_depth,
                generator=generator,
            )
        if self.pairformer_stack is not None:
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        else:
            pair = cast(Boltz1TrianglePairUpdateStack, self.pair_update_stack)(
                pair,
                pair_mask=pair_mask,
            )
        return self.final_single_norm(single), self.final_pair_norm(pair)


class Boltz1TrunkReuseConfidenceHead(_Boltz1ConfidenceHead):
    r"""
    Normalize and refine the trunk representations supplied by the Boltz-1 model.
    """

    def __init__(
        self,
        *,
        input_size: int,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 48,
        num_pair_attention_heads: int = 16,
        triangle_attention_hidden_size: int = 32,
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
        super().__init__(
            input_size=input_size,
            single_size=single_size,
            pair_size=pair_size,
            ranking_plddt_weight=ranking_plddt_weight,
            ranking_ptm_weight=ranking_ptm_weight,
        )
        self.input_single_norm = LayerNorm(input_size, backend=backend)
        self.single_norm: LayerNorm | None = None
        if update_single:
            self.single_norm = LayerNorm(single_size, backend=backend)
        self.input_single_projection: nn.Linear | None = None
        if add_single_input_to_single:
            self.input_single_projection = nn.Linear(input_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.relative_position_embedder: RelativePositionEmbedder | None = None
        self.token_bond_projection: nn.Linear | None = None
        if add_pair_input_to_pair:
            self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
            self.token_bond_projection = nn.Linear(1, pair_size, bias=False)

        self._initialize_sample_conditioning(
            num_distogram_bins=num_distogram_bins,
            max_distance=max_distance,
            add_single_to_pair_product=add_single_to_pair_product,
            use_single_diffusion=use_single_diffusion,
            backend=backend,
        )

        self.pairformer_stack: Boltz1PairformerStack | None = None
        self.pair_update_stack: Boltz1TrianglePairUpdateStack | None = None
        if update_single:
            self.pairformer_stack = Boltz1PairformerStack(
                num_blocks=num_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        else:
            self.pair_update_stack = Boltz1TrianglePairUpdateStack(
                num_blocks=num_blocks,
                pair_size=pair_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                dropout=pair_dropout,
                backend=backend,
            )
        self.output_head = Boltz1ConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            compute_pae=compute_pae,
        )

    def _prepare_representations(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        *,
        input: Boltz1Input,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_inputs = self.input_single_norm(single_inputs)
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
        pair = cast(Boltz1TrianglePairUpdateStack, self.pair_update_stack)(
            pair,
            pair_mask=pair_mask,
        )
        return single, pair


class Boltz1ConfidenceOutputHead(nn.Module):
    r"""
    Linear projections from the confidence single/pair representations to the Boltz-1 confidence logits.
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
        logits = {
            "pde_logits": self.pde_projection(pair + pair.transpose(-3, -2)),
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }
        if self.pae_projection is not None:
            logits["pae_logits"] = self.pae_projection(pair)
        return logits


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
        "geometry_upper_bounds": input.geometry_upper_bounds.to(device=device, dtype=atom_features.dtype),
        "geometry_lower_bounds": input.geometry_lower_bounds.to(device=device, dtype=atom_features.dtype),
        "chiral_atom_index": input.chiral_atom_index.to(device=device, dtype=torch.long),
        "chiral_atom_is_r": input.chiral_atom_is_r.to(device=device, dtype=torch.bool),
        "stereo_bond_index": input.stereo_bond_index.to(device=device, dtype=torch.long),
        "stereo_bond_is_e": input.stereo_bond_is_e.to(device=device, dtype=torch.bool),
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


_Boltz1PotentialArguments: TypeAlias = tuple[Tensor, Tensor | None, Tensor | None]


class Boltz1Potential:
    r"""
    Base class for a Boltz-1x physical potential over predicted coordinates.
    """

    def __init__(self, parameters: Mapping[str, float | int | Boltz1ParameterSchedule]) -> None:
        self.parameters = parameters

    def compute_parameters(self, time: float) -> dict[str, float | int]:
        return {
            name: value.compute(time) if isinstance(value, Boltz1ParameterSchedule) else value
            for name, value in self.parameters.items()
        }

    def compute(self, coords: Tensor, feats: dict[str, Tensor], parameters: Mapping[str, float | int]) -> Tensor:
        index, args, com_args = self.compute_args(feats, parameters)
        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
        local_coords = coords
        if com_args is not None:
            com_index, atom_mask = com_args
            if not torch.any(atom_mask):
                return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
            unpad_com_index = com_index[atom_mask]
            unpad_coords = local_coords[..., atom_mask, :]
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
        index, args, com_args = self.compute_args(feats, parameters)
        if index.shape[1] == 0:
            return torch.zeros_like(coords)
        local_coords = coords
        com_index = None
        if com_args is not None:
            com_index, atom_mask = com_args
            if not torch.any(atom_mask):
                return torch.zeros_like(coords)
            unpad_com_index = com_index[atom_mask]
            unpad_coords = local_coords[..., atom_mask, :]
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
    ) -> tuple[Tensor, _Boltz1PotentialArguments, tuple[Tensor, Tensor] | None]:
        raise NotImplementedError

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
        if not compute_gradient:
            return indexed_signed_dihedral(coords, index)
        return indexed_signed_dihedral_with_gradient(coords, index)


class Boltz1AbsoluteDihedralPotential(Boltz1DihedralPotential):
    r"""
    Dihedral variant that returns the absolute angle, with the gradient sign flipped where the angle is negative.
    """

    def compute_variable(
        self, coords: Tensor, index: Tensor, *, compute_gradient: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not compute_gradient:
            return torch.abs(cast(Tensor, super().compute_variable(coords, index, compute_gradient=False)))
        phi, grad = cast(tuple[Tensor, Tensor], super().compute_variable(coords, index, compute_gradient=True))
        grad = torch.where((phi < 0)[..., None, :, None], -grad, grad)
        return torch.abs(phi), grad


class Boltz1PoseBustersPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    PoseBusters bond/angle/clash potential over molecular distance bounds.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        pair_index = feats["geometry_pair_index"][0]
        lower_bounds = feats["geometry_lower_bounds"][0].clone()
        upper_bounds = feats["geometry_upper_bounds"][0].clone()
        bond_mask = feats["geometry_bond_mask"][0]
        angle_mask = feats["geometry_angle_mask"][0]
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
        atom_vdw_radii = (feats["reference_elements"].float() @ _vdw_radii(pair_index.device).unsqueeze(-1)).squeeze(
            -1
        )[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(
            lower_bounds[~bond_mask],
            bond_cutoffs[~bond_mask],
        )
        upper_bounds[bond_mask] = torch.min(
            upper_bounds[bond_mask],
            bond_cutoffs[bond_mask],
        )
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class Boltz1ConnectionsPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Distance potential pulling cross-chain connected atom pairs within a fixed upper-bound buffer.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        pair_index = feats["connected_atom_index"][0]
        upper_bounds = torch.full((pair_index.shape[1],), float(parameters["buffer"]), device=pair_index.device)
        return pair_index, (torch.ones_like(upper_bounds), None, upper_bounds), None


class Boltz1VdwOverlapPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Van-der-Waals clash potential penalizing non-bonded, cross-chain atom pairs closer than their summed radii.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        atom_chain_index = feats["chain_index"].gather(1, feats["atom_to_token"])[0]
        atom_mask = feats["atom_mask"][0].bool()
        if not torch.any(atom_mask):
            pair_index = torch.empty(2, 0, dtype=torch.long, device=atom_chain_index.device)
            empty = torch.empty(0, device=atom_chain_index.device)
            return pair_index, (empty, empty, None), None
        chain_sizes = torch.bincount(
            atom_chain_index[atom_mask],
            minlength=int(atom_chain_index.max().item()) + 1,
        )
        single_ion_mask = (chain_sizes > 1)[atom_chain_index]
        reference_elements = feats["reference_elements"]
        atom_vdw_radii = (reference_elements.float() @ _vdw_radii(reference_elements.device).unsqueeze(-1)).squeeze(-1)[
            0
        ]
        pair_index = torch.triu_indices(
            atom_chain_index.shape[0],
            atom_chain_index.shape[0],
            1,
            device=atom_chain_index.device,
        )
        pair_mask = atom_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]
        num_chains = int(atom_chain_index.max().item()) + 1
        connected_chain_index = feats["connected_chain_index"][0]
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
        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (1.0 - float(parameters["buffer"]))
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), None


class Boltz1SymmetricChainComPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Distance potential separating the centers of mass of same-entity symmetric chain pairs.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, tuple[Tensor, Tensor] | None]:
        atom_chain_index = feats["chain_index"].gather(1, feats["atom_to_token"])[0]
        atom_mask = feats["atom_mask"][0].bool()
        pair_index = feats["symmetric_chain_index"][0]
        if pair_index.shape[1] == 0 or not torch.any(atom_mask):
            lower_bounds = torch.empty(0, dtype=torch.float32, device=atom_chain_index.device)
            return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), None
        chain_sizes = torch.bincount(
            atom_chain_index[atom_mask],
            minlength=int(atom_chain_index.max().item()) + 1,
        )
        single_ion_mask = chain_sizes > 1
        pair_index = pair_index[:, single_ion_mask[pair_index[0]] & single_ion_mask[pair_index[1]]]
        lower_bounds = torch.full((pair_index.shape[1],), float(parameters["buffer"]), device=pair_index.device)
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), (atom_chain_index, atom_mask)


class Boltz1StereoBondPotential(Boltz1AbsoluteDihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing E/Z stereo-bond orientations toward 0 or pi.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        index = feats["stereo_bond_index"][0]
        orientation = feats["stereo_bond_is_e"][0].bool()
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
        lower_bounds[orientation] = torch.pi - float(parameters["buffer"])
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = float(parameters["buffer"])
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class Boltz1ChiralAtomPotential(Boltz1DihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Signed-dihedral potential enforcing chiral-center handedness by sign of the improper dihedral.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        index = feats["chiral_atom_index"][0]
        orientation = feats["chiral_atom_is_r"][0].bool()
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
        lower_bounds[orientation] = float(parameters["buffer"])
        upper_bounds[orientation] = float("inf")
        lower_bounds[~orientation] = float("-inf")
        upper_bounds[~orientation] = -float(parameters["buffer"])
        return index, (torch.ones_like(lower_bounds), lower_bounds, upper_bounds), None


class Boltz1PlanarBondPotential(Boltz1AbsoluteDihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing planarity around double bonds via improper dihedrals near zero.
    """

    def compute_args(
        self, feats: dict[str, Tensor], parameters: Mapping[str, float | int]
    ) -> tuple[Tensor, _Boltz1PotentialArguments, None]:
        double_bond_index = feats["planar_bond_index"][0].T
        improper_map = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 3]], device=double_bond_index.device).T
        improper_index = double_bond_index[:, improper_map].swapaxes(0, 1).flatten(start_dim=1)
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            float(parameters["buffer"]),
            device=improper_index.device,
        )
        return improper_index, (torch.ones_like(upper_bounds), None, upper_bounds), None


def _boltz1_potentials(physical_guidance: bool) -> tuple[Boltz1Potential, ...]:
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
            }
        ),
        Boltz1VdwOverlapPotential(
            {
                "guidance_interval": 5,
                "guidance_weight": (
                    Boltz1PiecewiseStepSchedule(thresholds=[0.4], values=[0.125, 0.0]) if guidance_weight else 0.0
                ),
                "resampling_weight": Boltz1PiecewiseStepSchedule(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        Boltz1ConnectionsPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.15 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 2.0,
            }
        ),
        Boltz1PoseBustersPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.01 * guidance_weight,
                "resampling_weight": 0.1,
                "bond_buffer": 0.125,
                "angle_buffer": 0.125,
                "clash_buffer": 0.10,
            }
        ),
        Boltz1ChiralAtomPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.10 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        Boltz1StereoBondPotential(
            {
                "guidance_interval": 1,
                "guidance_weight": 0.05 * guidance_weight,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        Boltz1PlanarBondPotential(
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
    input: Boltz1Input,
    pred_distogram_logits: Tensor,
    coordinates: Tensor,
    multiplicity: int,
    ranking_plddt_weight: float,
    ranking_ptm_weight: float,
) -> dict[str, Tensor]:
    r"""
    Derive Boltz-1 ranking metrics (complex/interface pLDDT, PDE, pTM/ipTM, confidence score) from the logits.

    Args:
        logits: The confidence logits dict from the confidence heads.
        distances: `[B*mult, N, N]` representative-atom distances used for interface contact masks.
        input: The Boltz-1 input batch.
        pred_distogram_logits: `[B, N, N, num_bins]` trunk distogram logits for contact probabilities.
        coordinates: `[B*mult, M, 3]` sampled coordinates used for pTM frames.
        multiplicity: Number of samples per input.
        ranking_plddt_weight: Weight on complex pLDDT in the combined confidence score.
        ranking_ptm_weight: Weight on pTM/ipTM in the combined confidence score.

    Returns:
        Mapping from score name to tensor (per-token metrics and per-sample complex/interface aggregates).
    """

    plddt = _expected_metric(logits["plddt_logits"])
    token_mask = input.token_mask.repeat_interleave(multiplicity, 0).to(plddt)
    complex_plddt = (plddt * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1)

    molecule_type = _molecule_type_for_confidence(input, multiplicity, plddt.device)
    is_ligand_token = (molecule_type == 3).to(plddt)
    is_contact = (distances < 8).to(plddt)
    chain_index = input.chain_index.repeat_interleave(multiplicity, 0)
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
            coordinates=coordinates,
            input=input,
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
    coordinates: Tensor,
    input: Boltz1Input,
    multiplicity: int,
) -> dict[str, Tensor]:
    r"""
    Compute pTM, ipTM, and the ligand/protein interface ipTM variants from PAE logits.

    Args:
        pae_logits: `[B*mult, N, N, num_bins]` PAE logits.
        coordinates: `[B*mult, M, 3]` sampled coordinates used to build valid frames.
        input: The Boltz-1 input batch.
        multiplicity: Number of samples per input.

    Returns:
        Mapping with ``ptm``, ``iptm``, ``ligand_iptm``, ``protein_iptm``, and the ``[B*mult, C, C]``
        per-chain-pair ``chain_pair_iptm`` matrix.
    """

    token_mask = input.token_mask.repeat_interleave(multiplicity, 0).to(pae_logits)
    frame_mask = _boltz1_coordinate_frames(
        coordinates,
        input=input,
        multiplicity=multiplicity,
    )[
        1
    ].to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(multiplicity, 0)
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

    molecule_type = _molecule_type_for_confidence(input, multiplicity, pae_logits.device)
    is_ligand = (molecule_type == 3).to(pae_logits)
    is_protein = (molecule_type == 0).to(pae_logits)
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

    # Restore the per-chain-pair interface pTM matrix. Entry ``[i, j]``
    # is the max-over-frame interface pTM of query chain ``i``
    # against key chain ``j``; the matrix is not symmetrized, preserving the query/key roles.
    unique_chain_indices = torch.unique(chain_index).tolist()
    num_chains = len(unique_chain_indices)
    chain_pair_iptm = tm_expected.new_zeros(tm_expected.shape[0], num_chains, num_chains)
    for row, query_index in enumerate(unique_chain_indices):
        for column, key_index in enumerate(unique_chain_indices):
            chain_pair_mask = (
                pair_mask_ptm
                * (chain_index[:, None, :] == key_index).to(pae_logits)
                * (chain_index[:, :, None] == query_index).to(pae_logits)
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


def _boltz1_coordinate_frames(
    coordinates: Tensor,
    *,
    input: Boltz1Input,
    multiplicity: int,
    atom_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Construct canonical polymer and coordinate-dependent nonpolymer frames.

    Args:
        coordinates: `[B*mult, M, 3]` coordinate realizations.
        input: The Boltz-1 input batch.
        multiplicity: Number of samples per input.
        atom_mask: Optional realization-specific resolved-atom mask.

    Returns:
        Frame indices `[B*mult, N, 3]` and valid-frame mask `[B*mult, N]`.
    """

    frame_index = (
        input.frame_atom_index.repeat_interleave(
            multiplicity,
            0,
        )
        .to(
            device=coordinates.device,
            dtype=torch.long,
        )
        .clone()
    )
    chain_index = input.chain_index
    molecule_type = input.molecule_type
    atom_to_token = input.atom_to_token.long()
    atom_chain_index = chain_index.gather(1, atom_to_token)
    structural_atom_mask = input.atom_mask.bool()
    if atom_mask is None:
        atom_mask = structural_atom_mask.repeat_interleave(
            multiplicity,
            0,
        )
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
            for sample_index in range(multiplicity):
                row = batch_index * multiplicity + sample_index
                distances = torch.cdist(
                    coordinates[row, chain_atoms],
                    coordinates[row, chain_atoms],
                )
                resolved = atom_mask[row, chain_atoms]
                resolved_pair = resolved.unsqueeze(-1) & resolved.unsqueeze(-2)
                nearest = distances.masked_fill(
                    ~resolved_pair,
                    torch.inf,
                ).argsort(
                    dim=-1
                )[:, :3]
                frame_index[row, chain_tokens] = chain_atoms[nearest][:, (1, 0, 2)]

    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    frames = coordinates.gather(
        1,
        flat_index.unsqueeze(-1).expand(-1, -1, 3),
    )
    frames = frames.reshape(*frame_index.shape, 3)
    frame_resolved = atom_mask.gather(1, flat_index).reshape(frame_index.shape).all(dim=-1)
    center_index = input.token_center_atom_index.repeat_interleave(
        multiplicity,
        0,
    ).to(
        device=coordinates.device,
        dtype=torch.long,
    )
    center_resolved = atom_mask.gather(1, center_index)
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
        & input.token_mask.repeat_interleave(multiplicity, 0).to(
            device=coordinates.device,
            dtype=torch.bool,
        )
    )
    return frame_index, valid_geometry


def _molecule_type_for_confidence(input: Boltz1Input, multiplicity: int, device: torch.device) -> Tensor:
    r"""
    Repeat the per-token molecule type used to classify ligand/protein tokens for confidence.

    Args:
        input: The Boltz-1 input batch.
        multiplicity: Number of samples per input.
        device: Target device.

    Returns:
        A `[B*mult, N]` long tensor of molecule types.
    """

    return input.molecule_type.repeat_interleave(multiplicity, 0).to(device=device, dtype=torch.long)


def _decode_boltz1_fold_result(output: _Boltz1FoldOutput, input: Boltz1Input) -> FoldResult:
    r"""
    Decode one model output into structures and confidence values.

    Maps model atom order back onto the system's complete heavy-atom sites to build one ``Structure`` per
    sample, and lifts the confidence head's per-token metrics, token-pair PAE/PDE, and per-sample ranking
    scores onto values aligned to the layout's token axis.

    Args:
        output: The raw folding output with sampled coordinates and optional
            typed confidence predictions.
        input: The Boltz-1 input batch whose ``layout`` names the aligned molecular identities.

    Returns:
        A [`FoldResult`][ofoldx.results.FoldResult] with one prediction per drawn sample.
    """

    layout = cast(Boltz1Layout, input.layout)
    token_count = len(layout.token_sites)
    atom_count = len(layout.atom_sites)

    complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element not in {"H", "D", "T"}]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)

    coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
    num_samples = coordinates.shape[0]
    token_axis = layout.token_sites
    confidence = output.confidence

    def per_sample(value: Tensor | None) -> np.ndarray | None:
        return None if value is None else value[0].to(device="cpu", dtype=torch.float32).numpy()

    plddt = per_sample(None if confidence is None else confidence.plddt)
    pae = per_sample(None if confidence is None else confidence.pae)
    pde = per_sample(None if confidence is None else confidence.pde)
    complex_plddt = per_sample(None if confidence is None else confidence.complex_plddt)
    complex_iplddt = per_sample(None if confidence is None else confidence.complex_iplddt)
    complex_pde = per_sample(None if confidence is None else confidence.complex_pde)
    complex_ipde = per_sample(None if confidence is None else confidence.complex_ipde)
    confidence_score = per_sample(None if confidence is None else confidence.confidence_score)
    ptm = per_sample(None if confidence is None else confidence.ptm)
    iptm = per_sample(None if confidence is None else confidence.iptm)
    ligand_iptm = per_sample(None if confidence is None else confidence.ligand_iptm)
    protein_iptm = per_sample(None if confidence is None else confidence.protein_iptm)
    chain_pair_iptm = per_sample(None if confidence is None else confidence.chain_pair_iptm)
    chain_axis = tuple(layout.system.instances.keys())

    resolved_probability: np.ndarray | None = None
    if confidence is not None:
        resolved_probability = (
            confidence.resolved_logits[0, :, :token_count].float().softmax(dim=-1)[..., 1].cpu().numpy()
        )

    predictions: list[FoldPrediction] = []
    for sample_index in range(num_samples):
        complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        complete_coordinates[model_to_complete] = coordinates[sample_index]
        coordinate_mask[model_to_complete] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=complete_coordinates,
            coordinate_mask=coordinate_mask,
        )

        sample_scores: dict[str, Score] = {}
        if complex_plddt is not None:
            # Published pLDDT uses the shared 0-100 scale; the expected pLDDT is 0-1.
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


def _gather_boltz1_structure_target(
    target: Boltz1StructureTarget,
    input: Boltz1Input,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather complete-pool truth onto the current padded atom axis.
    """

    atom_index = target.atom_index.long()
    valid = atom_index >= 0
    gather_index = atom_index.clamp_min(0)
    coordinates = target.coordinates.gather(
        1,
        gather_index.unsqueeze(-1).expand(-1, -1, 3),
    )
    mask = target.coordinate_mask.gather(1, gather_index)
    mask = mask & valid & input.atom_mask.bool()
    coordinates = coordinates.masked_fill(~mask.unsqueeze(-1), 0)
    return coordinates, mask


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


# ---------------------------------------------------------------------------
# Tensor boundary of the Boltz-1 model graph: the pure-tensor input,
# output, molecular-identity layout, trunk representations, and pocket record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Boltz1Pocket:
    r"""
    One Boltz-1 unary pocket conditioning declaration.

    The encoding admits a single binder chain and an explicitly declared
    contact-residue list; the 6 A cutoff is the training-time contact definition and
    is not recomputed at fold time. Training-only sampling controls are not
    representable.

    Args:
        binder: The ``instance_id`` of the single binder chain.
        pocket: [`ComponentReference`][ofoldx.data.molecular.ComponentReference] values
            values naming the pocket contact residues.
    """

    binder: str
    pocket: tuple[ComponentReference, ...] = ()


@dataclass(frozen=True)
class Boltz1Layout:
    r"""
    Molecular identities aligned to one Boltz-1 tensor layout.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]


@tensorclass
class Boltz1StructureTarget:
    r"""
    Complete pre-crop structure truth for Boltz-1 training.

    Local symmetry groups name rows on the current atom axis; all coordinate
    values live in the complete target pool.

    Args:
        coordinates: Complete-assembly coordinates with shape ``(B, T, 3)``.
        coordinate_mask: Resolved complete-assembly mask with shape ``(B, T)``.
        atom_index: Current atom-to-pool mapping with shape ``(B, A)`` and
            ``-1`` padding.
        chain_span: Half-open pool chain spans with shape ``(B, C, 2)``.
        chain_equivalence_index: Entity-and-layout equivalence ids with
            shape ``(B, C)``.
        symmetry_atom_index: Current-axis atoms in each local symmetry
            group with shape ``(B, G, W)`` and ``-1`` padding.
        symmetry_permutation: Complete local permutations with shape
            ``(B, G, P, W)`` and ``-1`` padding.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    atom_index: Tensor
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
    conditioning. All tensors carry a leading batch axis; ``N`` is the token count and ``M`` the atom count.

    Args:
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
        token_distogram_atom_index: `[B, N]` distogram-atom index per token.
        residue_type_features: `[B, N, vocab_size]` residue-type one-hot features.
        msa_profile: `[B, N, vocab_size]` MSA profile feature.
        deletion_mean: `[B, N]` mean-deletion feature.
        token_bonds: `[B, N, N]` inter-token bond indicator.
        molecule_type: `[B, N]` per-token molecule type.
        geometry_pair_index: `[B, 2, P]` atom pairs with molecular distance bounds.
        geometry_bond_mask: `[B, P]` mask of directly bonded geometry pairs.
        geometry_angle_mask: `[B, P]` mask of two-bond angle endpoint pairs.
        geometry_upper_bounds: `[B, P]` upper distance bounds in angstroms.
        geometry_lower_bounds: `[B, P]` lower distance bounds in angstroms.
        chiral_atom_index: `[B, 4, C]` chiral-center improper-dihedral atom indices.
        chiral_atom_is_r: `[B, C]` handedness flag for each chiral-center constraint.
        stereo_bond_index: `[B, 4, S]` stereo-bond dihedral atom indices.
        stereo_bond_is_e: `[B, S]` E/Z flag for each stereo-bond constraint.
        planar_bond_index: `[B, 6, D]` planar double-bond atom indices.
        connected_chain_index: `[B, 2, K]` chain pairs joined by assembly bonds.
        connected_atom_index: `[B, 2, K]` atom pairs joined by assembly bonds.
        symmetric_chain_index: `[B, 2, Q]` same-entity chain-copy pairs.
        frame_atom_index: `[B, N, 3]` canonical polymer frame indices per token.
        cyclic_period: Optional `[B, N]` cyclic-period feature for relative positions.
        msa_tokens: Optional `[B, S, N]` MSA token grid.
        has_deletion: Optional `[B, S, N]` per-MSA deletion-present flag.
        deletion_value: Optional `[B, S, N]` per-MSA deletion count feature.
        msa_paired: Optional `[B, S, N]` cross-chain pairing flag.
        msa_mask: Optional `[B, S, N]` MSA validity mask.
        pocket_feature: Optional `[B, N, pocket_feature_size]` pocket one-hot conditioning.
        layout: Optional [`Boltz1Layout`][ofoldx.models.boltz1.model.Boltz1Layout] aligning the
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
    atom_token_index: Tensor
    token_center_atom_index: Tensor
    token_distogram_atom_index: Tensor
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
    chiral_atom_is_r: Tensor
    stereo_bond_index: Tensor
    stereo_bond_is_e: Tensor
    planar_bond_index: Tensor
    connected_chain_index: Tensor
    connected_atom_index: Tensor
    symmetric_chain_index: Tensor
    frame_atom_index: Tensor
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    pocket_feature: Tensor | None = None
    layout: Boltz1Layout | None = None


@tensorclass
class _Boltz1TrunkOutput:
    r"""
    Boltz-1 trunk representations returned by the recycling trunk.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


@tensorclass
class Boltz1Output:
    r"""
    Predictions from one Boltz-1 training denoising pass.

    Args:
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


@tensorclass
class Boltz1ConfidenceOutput:
    r"""
    Raw Boltz-1 confidence predictions and their official derived values.

    Args:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)``.
        distogram_logits: Detached trunk distogram logits with shape
            ``(B, N, N, D)``.
        plddt_logits: Per-token pLDDT logits with shape ``(B, S, N, D_p)``.
        pde_logits: PDE logits with shape ``(B, S, N, N, D_d)``.
        resolved_logits: Per-token resolved logits with shape ``(B, S, N, 2)``.
        plddt: Expected per-token pLDDT fractions with shape ``(B, S, N)``.
        pde: Expected PDE values with shape ``(B, S, N, N)``.
        complex_plddt: Mean complex pLDDT with shape ``(B, S)``.
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
            shape ``(B, S, C, C)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor
    plddt: Tensor
    pde: Tensor
    complex_plddt: Tensor
    complex_iplddt: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    confidence_score: Tensor
    pae_logits: Tensor | None = None
    pae: Tensor | None = None
    ptm: Tensor | None = None
    iptm: Tensor | None = None
    ligand_iptm: Tensor | None = None
    protein_iptm: Tensor | None = None
    chain_pair_iptm: Tensor | None = None


@tensorclass
class _Boltz1FoldOutput:
    r"""
    Raw predictions from one Boltz-1 folding pass.

    Args:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        confidence: Typed confidence predictions when the checkpoint contains a
            confidence head.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    confidence: Boltz1ConfidenceOutput | None


@tensorclass
class Boltz1Representations:
    r"""
    Raw Boltz-1 trunk representations surfaced by ``Boltz1Model.representations``.

    Args:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor
