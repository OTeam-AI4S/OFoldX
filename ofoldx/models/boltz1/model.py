r"""
Boltz-1 model.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.evidence import MSA_VOCAB_SIZE
from ...data.molecular import ComponentReference, MolecularSystem
from ...model import Model, _OptionalTensor, _TensorCollection
from ...modules.atom import (
    AtomPairEncoding,
    AtomToTokenMean,
    LocalAtomWindowInfo,
    LocalAtomWindowing,
    PackedAtomAttention,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
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
from ...modules.geometry import rotate_vectors, uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import PairSymmetrizedDistogramHead
from ...modules.head.frames import gather_atoms_by_index, insert_sample_axis
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
_Boltz1PotentialEvaluation: TypeAlias = tuple[
    Tensor,
    _Boltz1PotentialArguments,
    tuple[Tensor, Tensor] | tuple[Tensor, Tensor, int] | None,
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

    Examples:
        >>> import torch
        >>> from ofoldx.data import MolecularSystem, PolymerSequence
        >>> from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model, Boltz1Pipeline, Boltz1Processor
        >>> sequence = PolymerSequence.from_one_letter("G")
        >>> system = MolecularSystem.from_entities(
        ...     {"protein": sequence}, {"A": "protein"}
        ... )
        >>> processor = Boltz1Processor()
        >>> input = processor(system)
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
        ...     output = model.sample_with_confidence(
        ...         input, num_samples=2, num_sampling_steps=1, recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape)
        (1, 2, 32, 3)
        >>> result = Boltz1Pipeline(model=model, processor=processor).decode_fold(input, output)
        >>> len(result.predictions)
        2
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
            single_size
            + input_embeddings_config.vocab_size
            + MSA_VOCAB_SIZE
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
        sample_chunk_size: int | None = None,
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
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                raise ValueError("sample_chunk_size must be positive")
            sample_chunk_size = min(sample_chunk_size, num_samples)
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
        output = self.sample_with_confidence(
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
    ) -> Boltz1SamplingOutput:
        r"""
        Sample atom coordinates with explicit ``(B, S, A, 3)`` axes.

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
        return Boltz1SamplingOutput(
            coordinates=coordinates,
            batch_size=[coordinates.shape[0]],
        )

    def sample_with_confidence(
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
    ) -> Boltz1SamplingWithConfidenceOutput:
        r"""
        Run the trunk, coordinate sampler, and confidence head.

        Runs the recycling trunk once, draws ``num_samples`` reverse-diffusion samples, and (when the
        confidence head is present) scores them into confidence logits and ranking metrics. Coordinate
        tensors retain an explicit ``(batch, num_samples, ...)`` shape throughout diffusion.

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
            A [`Boltz1SamplingWithConfidenceOutput`][]
                with sampled coordinates, distogram logits, and optional typed
                confidence predictions.

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
                complex_plddt=confidence_values["complex_plddt"],
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


def _boltz1_local_atom_pair_inputs(
    atom_single: Tensor,
    reference_positions: Tensor,
    atom_mask: Tensor,
    reference_space_id: Tensor,
    windowing: LocalAtomWindowing,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, LocalAtomWindowInfo]:
    r"""Build the released centered key-minus-query Boltz-1 local-pair inputs."""

    position_windows = windowing(reference_positions)
    info = position_windows.info

    def gather(value: Tensor) -> tuple[Tensor, Tensor]:
        windows = windowing.gather(value, info)
        return windows.query, windows.key

    mask_query, mask_key = gather(atom_mask.unsqueeze(-1))
    space_query, space_key = gather(reference_space_id.unsqueeze(-1))
    single_query, single_key = gather(F.relu(atom_single))
    mask_query = mask_query.squeeze(-1).bool()
    mask_key = mask_key.squeeze(-1).bool()
    same_space = space_query.squeeze(-1).unsqueeze(-1) == space_key.squeeze(-1).unsqueeze(-2)
    valid = (mask_query.unsqueeze(-1) & mask_key.unsqueeze(-2) & same_space).to(reference_positions.dtype)
    distance_vectors = position_windows.key.unsqueeze(-3) - position_windows.query.unsqueeze(-2)
    return distance_vectors, valid.unsqueeze(-1), mask_query, mask_key, single_query, single_key, info


class Boltz1PackedReferenceAtomEncoder(nn.Module):
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
        self.pair_encoding = AtomPairEncoding(atom_hidden_size, atom_pair_hidden_size)
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_query = self.atom_feature_projection(atom_features)
        atom_single = atom_query
        distance_vectors, valid, _mask_query, _mask_key, single_query, single_key, info = (
            _boltz1_local_atom_pair_inputs(
                atom_single,
                reference_positions,
                atom_mask,
                reference_space_id,
                self.windowing,
            )
        )
        atom_pair_values = self.pair_encoding(
            distance_vectors,
            single_query.unsqueeze(-2),
            single_key.unsqueeze(-3),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=None,
            prefix_ndim=1,
        )
        atom_pair = info.pack_like(atom_single, atom_pair_values)
        return atom_query, atom_single, atom_pair


class Boltz1PackedAtomToTokenEncoder(nn.Module):
    r"""
    Run packed local atom attention and mean-pool the result to token representations.

    Args:
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        token_hidden_size: Width of the produced token representation.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.token_activation = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend, denominator_offset=1e-6)

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
    ) -> Tensor:
        atom_query = self.atom_transformer(
            atom_single,
            atom_conditioning,
            atom_pair_bias,
            atom_mask=atom_mask,
        )
        token_from_atom = self.token_activation(self.token_projection(atom_query))
        return self.atom_to_token_mean(
            token_from_atom,
            atom_to_token,
            token_reference,
            atom_mask,
        )


class Boltz1PackedDenoisingAtomToTokenEncoder(nn.Module):
    r"""
    Add noisy coordinates to packed atom states, attend locally, and pool to denoiser tokens.

    Args:
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        token_hidden_size: Half the width of the produced denoiser token representation.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        transformer_size: int | None = None,
        post_layer_norm: bool = False,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * token_hidden_size if transformer_size is None else transformer_size
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                post_layer_norm=post_layer_norm,
                use_qk_norm=use_qk_norm,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.token_projection = nn.Linear(atom_hidden_size, transformer_size, bias=False)
        self.token_activation = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend, denominator_offset=1e-6)

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
        noisy_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_query = atom_single.unsqueeze(-3) + self.noisy_position_projection(noisy_positions)
        conditioning = atom_conditioning.unsqueeze(-3)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning,
            atom_pair_bias,
            atom_mask=atom_mask,
        )
        token_from_atom = self.token_activation(self.token_projection(atom_query))
        token_repr = self.atom_to_token_mean(
            token_from_atom,
            atom_to_token,
            token_reference,
            atom_mask,
        )
        return token_repr, atom_query, conditioning


class Boltz1PackedTokenToAtomDecoder(nn.Module):
    r"""
    Broadcast token states to packed atom queries and predict per-atom coordinate updates.

    Args:
        token_hidden_size: Half the width of the incoming denoiser token representation.
        atom_hidden_size: Width of the atom hidden and conditioning channels.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the conditioned Transformer encoder.
    """

    def __init__(
        self,
        token_hidden_size: int = 384,
        atom_hidden_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        transformer_size: int | None = None,
        use_qk_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * token_hidden_size if transformer_size is None else transformer_size
        self.token_projection = nn.Linear(transformer_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_blocks,
                num_attention_heads=num_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                use_qk_norm=use_qk_norm,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.position_norm = LayerNorm(atom_hidden_size, backend=backend)
        self.position_update_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def forward(
        self,
        token_repr: Tensor,
        atom_skip: Tensor,
        conditioning_skip: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        token_projection = self.token_projection(token_repr)
        atom_query = atom_skip + self.token_to_atom_broadcast(token_projection, atom_to_token)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=atom_mask,
        )
        return self.position_update_projection(self.position_norm(atom_query))


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
        self.atom_encoder: Boltz1PackedReferenceAtomEncoder | None = None
        self.atom_pair_bias_projection: PerBlockPairBiasProjection | None = None
        self.atom_attention_encoder: Boltz1PackedAtomToTokenEncoder | None = None
        if use_atom_encoder:
            self.atom_encoder = Boltz1PackedReferenceAtomEncoder(
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
            self.atom_attention_encoder = Boltz1PackedAtomToTokenEncoder(
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
            token_from_atom = torch.zeros_like(residue_type_features[..., :1]).expand(
                *([-1] * (residue_type_features.dim() - 1)),
                self.single_size,
            )
        else:
            atom_query, atom_conditioning, atom_pair = atom_encoder(
                atom_features,
                reference_positions,
                atom_mask,
                reference_space_id,
            )
            token_from_atom = cast(Boltz1PackedAtomToTokenEncoder, atom_attention_encoder)(
                atom_query,
                atom_conditioning,
                cast(PerBlockPairBiasProjection, atom_pair_bias_projection)(atom_pair),
                atom_mask,
                atom_to_token,
                residue_type_features,
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
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.use_paired_feature = use_paired_feature
        self.msa_projection = nn.Linear(
            MSA_VOCAB_SIZE + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
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
        if msa_max_depth is not None:
            msa_max_depth = operator.index(msa_max_depth)
            if msa_max_depth < 1:
                raise ValueError("msa_max_depth must be positive")
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
            F.one_hot(msa_tokens, num_classes=MSA_VOCAB_SIZE).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(-3)
        pair_mask = pair_mask_from_token_mask(token_mask)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair, msa = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    msa,
                    pair_mask,
                    msa_mask,
                )
            else:
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
        self.msa_dropout = StructuredDropout(msa_dropout, broadcast_dims=(-2, -1))
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
        self.gradient_checkpointing = False
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

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                single, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    single,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
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
        self.gradient_checkpointing = False
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

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
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
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
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
            backend=backend,
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
        self.atom_attention_encoder = Boltz1PackedDenoisingAtomToTokenEncoder(
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
        self.atom_attention_decoder = Boltz1PackedTokenToAtomDecoder(
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
        return_token_repr: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        single, _ = self.single_conditioner(
            noise_level,
            single_trunk,
            single_inputs,
        )
        token_repr, atom_skip, conditioning_skip = self.atom_attention_encoder(
            cast(Tensor, conditioning["atom_query"]),
            cast(Tensor, conditioning["atom_conditioning"]),
            cast(Tensor, conditioning["atom_enc_bias"]),
            atom_mask,
            atom_to_token,
            single,
            noisy_positions=noisy_positions,
        )
        token_repr = token_repr + self.single_to_token_projection(self.single_to_token_norm(single))
        token_encoder_bias = cast(Tensor, conditioning["token_encoder_bias"])
        token_repr = self.token_encoder(
            token_repr,
            single,
            token_encoder_bias,
            token_mask,
        )
        token_repr = self.token_norm(token_repr)
        coordinate_update = self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            cast(Tensor, conditioning["atom_dec_bias"]),
            atom_mask,
            atom_to_token,
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
        normal = _draw_normal(
            (batch_size,),
            device=device,
            dtype=dtype,
            generator=generator,
        )
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
        return_token_repr: bool = False,
        **conditioning: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size = noisy_coordinates.shape[0]
        num_samples = noisy_coordinates.shape[-3]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device, dtype=noisy_coordinates.dtype)
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
        multiplicity: int = 1,
        sample_chunk_size: int | None = None,
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
        sample_chunk_size = multiplicity if sample_chunk_size is None else min(sample_chunk_size, multiplicity)
        dtype, device = single_trunk.dtype, single_trunk.device
        target = target_coordinates.to(dtype=dtype, device=device).unsqueeze(-3)
        target = target.expand(
            *target.shape[:-3],
            multiplicity,
            *target.shape[-2:],
        )
        # Centre and rigid-augment over ALL padded atoms (atom_pad_mask), matching upstream
        # ``center_random_augmentation``. The resolved-atom mask is a loss/alignment concern applied by the
        # criterion, not part of the corruption equation.
        augmentation_mask = atom_mask.to(dtype=dtype, device=device)
        batch_size = target_coordinates.shape[0]
        noise_batch_size = batch_size if synchronize_noise_levels else batch_size * multiplicity
        noise_level = self.noise_distribution(
            noise_batch_size,
            device=device,
            dtype=dtype,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.unsqueeze(-1).expand(batch_size, multiplicity)
        else:
            noise_level = noise_level.reshape(batch_size, multiplicity)
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
        for sample_start in range(0, multiplicity, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, multiplicity)
            chunk_num_samples = sample_end - sample_start
            noisy_chunk = torch.narrow(noisy_coordinates, -3, sample_start, chunk_num_samples)
            noise_level_chunk = torch.narrow(noise_level, -1, sample_start, chunk_num_samples)
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
        weight = mask.unsqueeze(-2).unsqueeze(-1)
        centroid = (coords * weight).sum(-2, keepdim=True) / weight.sum(-2, keepdim=True).clamp_min(1.0)
        coords = coords - centroid
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
        translation = _draw_normal(
            (*coords.shape[:-2], 1, 3),
            device=coords.device,
            dtype=coords.dtype,
            generator=generator,
        )
        return rotate_vectors(coords, rotation) + translation_scale * translation

    def _sample_ordinary(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None,
        step_scale: float,
        multiplicity: int,
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
            multiplicity,
            expanded_atom_mask.shape[-1],
        )
        coordinate_template = expanded_atom_mask.unsqueeze(-1).expand(
            *expanded_atom_mask.shape,
            3,
        )
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_current_hp = sigmas[:-1].double()
        sigma_hats_hp = sigma_current_hp * (1 + gammas[1:].double())
        sigma_next_hp_values = sigmas[1:].double()
        noise_variances = self.noise_scale**2 * (sigma_hats_hp**2 - sigma_current_hp**2)

        def randn_like(reference: Tensor) -> Tensor:
            random_device = reference.device if generator is None else generator.device
            return torch.empty_like(reference, device=random_device).normal_(generator=generator).to(reference.device)

        coords = sigmas[0] * randn_like(coordinate_template.to(dtype=dtype))
        token_repr = None
        token_a = None
        for index in range(len(sigmas) - 1):
            sigma_current, gamma = sigmas[index], gammas[index + 1]
            random_device = device if generator is None else generator.device
            rotation = uniform_random_rotation(
                math.prod(coords.shape[:-2]),
                device=random_device,
                dtype=coords.dtype,
                generator=generator,
            ).to(device)
            rotation = rotation.reshape(*coords.shape[:-2], 3, 3)
            translation = randn_like(torch.narrow(coords, -2, 0, 1))
            coords = rotate_vectors(coords - coords.mean(-2, keepdim=True), rotation) + translation

            sigma_hat = sigma_current * (1 + gamma)
            sigma_hat_hp = sigma_hats_hp[index]
            sigma_next_hp = sigma_next_hp_values[index]
            noise_var = noise_variances[index]
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)).to(coords.dtype) * randn_like(coords)
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
                        dtype=token_a.dtype,
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
                    source=noisy,
                    target=denoised,
                    weight=expanded_atom_mask,
                )
            derivative = (noisy - denoised) / sigma_hat_hp
            coords = (noisy + step_scale * (sigma_next_hp - sigma_hat_hp) * derivative).to(coords.dtype)
        return (coords, cast(Tensor, token_repr)) if return_token_repr else coords

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
        guidance_active = physical_guidance and num_guidance_steps > 0
        if not feynman_kac and not guidance_active:
            return self._sample_ordinary(
                atom_mask,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                multiplicity=multiplicity,
                generator=generator,
                return_token_repr=return_token_repr,
                conditioning=conditioning,
            )
        if isinstance(atom_mask, NestedTensor):
            raise NotImplementedError("packed Boltz-1 sampling does not yet support Feynman-Kac or potentials")
        potentials = _boltz1_potentials(physical_guidance) if feynman_kac or guidance_active else ()
        requested_multiplicity = multiplicity
        effective_multiplicity = multiplicity * num_particles if feynman_kac else multiplicity
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(effective_multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        sigma_current_hp = sigmas[:-1].double()
        sigma_hats_hp = sigma_current_hp * (1 + gammas[1:].double())
        sigma_next_hp_values = sigmas[1:].double()
        noise_variances = self.noise_scale**2 * (sigma_hats_hp**2 - sigma_current_hp**2)
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
                            potential._resolve_parameters(lowering, parameters),
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
                    batch_index * effective_multiplicity,
                    (batch_index + 1) * effective_multiplicity,
                )
                for batch_index in range(atom_mask.shape[0])
            )

        def randn() -> Tensor:
            r"""
            Draw standard-normal noise of the expanded coordinate ``shape`` on the sampler device/generator.
            """

            return _draw_normal(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        coords = sigmas[0] * randn()
        if feynman_kac:
            num_groups = atom_mask.shape[0] * requested_multiplicity
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
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
            random_device = device if generator is None else generator.device
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0],
                device=random_device,
                dtype=coords.dtype,
                generator=generator,
            ).to(device)
            translation = _draw_normal(
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

            sigma_hat = sigma_current * (1 + gamma)
            sigma_hat_hp = sigma_hats_hp[index]
            sigma_next_hp = sigma_next_hp_values[index]
            noise_var = noise_variances[index]
            noisy = coords + torch.sqrt(torch.clamp(noise_var, min=0.0)).to(coords.dtype) * randn()
            with torch.no_grad():
                noisy_explicit = noisy.reshape(
                    base_atom_mask.shape[0],
                    effective_multiplicity,
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
                                potential._resolve_parameters(lowering, parameters),
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
                    effective_multiplicity = requested_multiplicity
            if return_token_repr:
                output_token_feature_update = self.output_token_feature_update
                if output_token_feature_update is not None:
                    if token_repr is None:
                        token_repr = torch.zeros_like(cast(Tensor, token_a))
                    token_a = cast(Tensor, token_a)
                    sigma = sigma_hat.expand(token_a.shape[0]).to(device=token_a.device, dtype=token_a.dtype)
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
        coordinates = coords.reshape(base_atom_mask.shape[0], requested_multiplicity, *coords.shape[1:])
        if not return_token_repr:
            return coordinates
        token_repr = cast(Tensor, token_repr)
        return coordinates, token_repr.reshape(
            base_atom_mask.shape[0],
            requested_multiplicity,
            *token_repr.shape[1:],
        )


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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_encoding = AtomPairEncoding(atom_hidden_size, atom_pair_hidden_size)
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=False)
        self.single_trunk_norm = LayerNorm(single_size)
        self.single_trunk_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.pair_trunk_norm = LayerNorm(pair_size)
        self.pair_trunk_projection = nn.Linear(
            pair_size,
            atom_pair_hidden_size,
            bias=False,
        )
        self.token_pair_to_local_atom_pair_broadcast = TokenPairToLocalAtomPairBroadcast(
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )

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
        single_proj = self.single_trunk_projection(self.single_trunk_norm(single_trunk))
        atom_single = atom_single + self.token_to_atom_broadcast(single_proj, atom_to_token)
        distance_vectors, valid, _mask_query, mask_key, single_query, single_key, info = _boltz1_local_atom_pair_inputs(
            atom_single,
            reference_positions,
            atom_mask,
            reference_space_id,
            self.windowing,
        )
        pair_from_trunk = self.pair_trunk_projection(self.pair_trunk_norm(pair_trunk))
        pair_gathered = self.token_pair_to_local_atom_pair_broadcast(
            pair_from_trunk,
            atom_to_token,
            key_index=info.key_index,
        )
        pair_values = info.values(pair_gathered)
        pair_bias = pair_values * mask_key.unsqueeze(-2).unsqueeze(-1).to(pair_values.dtype)
        atom_pair_values = self.pair_encoding(
            distance_vectors,
            single_query.unsqueeze(-2),
            single_key.unsqueeze(-3),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=pair_bias,
            prefix_ndim=1,
        )
        atom_pair = info.pack_like(atom_single, atom_pair_values)
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

    def forward(
        self,
        noise_level: Tensor,
        single_trunk: Tensor,
        single_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
            sample_axis=True,
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
        fourier = next_repr[..., :1] * 0 + fourier.unsqueeze(-2)
        conditioning = torch.cat([accumulated, fourier], dim=-1)
        return accumulated + self.transition(next_repr, conditioning)


class Boltz1ConfidenceConditioning(nn.Module):
    r"""
    Condition confidence single and pair representations on sampled coordinates.
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
        if use_single_diffusion:
            self.diffusion_single_norm = LayerNorm(2 * single_size, backend=backend)
            self.diffusion_single_projection = nn.Linear(2 * single_size, single_size, bias=False)
        else:
            self.diffusion_single_norm = None
            self.diffusion_single_projection = None

        self.single_to_pair_row = nn.Linear(input_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(input_size, pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(input_size, pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        else:
            self.single_to_pair_product_left = None
            self.single_to_pair_product_right = None
            self.single_to_pair_product_out = None

        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        token_distogram_atom_index: Tensor,
        token_mask: Tensor,
        single_diffusion: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_samples = coordinates.shape[-3]
        single = insert_sample_axis(single, num_samples, dim=-3)
        single_inputs = insert_sample_axis(single_inputs, num_samples, dim=-3)
        if self.diffusion_single_projection is not None and self.diffusion_single_norm is not None:
            single = single + self.diffusion_single_projection(
                self.diffusion_single_norm(cast(Tensor, single_diffusion))
            )

        pair = insert_sample_axis(pair, num_samples, dim=-4)
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

        representative_coordinates = gather_atoms_by_index(
            coordinates,
            insert_sample_axis(token_distogram_atom_index, num_samples, dim=-2),
        )
        distances = torch.cdist(representative_coordinates, representative_coordinates)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        pair_mask = pair_mask_from_token_mask(insert_sample_axis(token_mask, num_samples, dim=-2))
        return single_inputs, single, pair, distances, pair_mask


class Boltz1TrunkImitationConfidenceHead(nn.Module):
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
        msa_head_dim: int = 32,
        msa_triangle_attention_hidden_size: int = 32,
        num_msa_triangle_attention_heads: int = 4,
        triangle_attention_hidden_size: int = 32,
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
            token_distogram_atom_index=input.token_distogram_atom_index,
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
            pair = cast(Boltz1TrianglePairUpdateStack, self.pair_update_stack)(
                pair,
                pair_mask=pair_mask,
            )
        return self.final_single_norm(single), self.final_pair_norm(pair)


class Boltz1TrunkReuseConfidenceHead(nn.Module):
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
        super().__init__()
        self.ranking_plddt_weight = ranking_plddt_weight
        self.ranking_ptm_weight = ranking_ptm_weight
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
            token_distogram_atom_index=input.token_distogram_atom_index,
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


def _draw_normal(
    size: tuple[int, ...] | torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Draw standard-normal values from ``generator`` and place them on ``device``.
    """

    random_device = device if generator is None else generator.device
    return torch.randn(
        size,
        device=random_device,
        dtype=dtype,
        generator=generator,
    ).to(device)


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
        return self._compute(coords, self.compute_args(feats, parameters))

    def _compute(self, coords: Tensor, arguments: _Boltz1PotentialEvaluation) -> Tensor:
        index, args, com_args = arguments
        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
        local_coords = coords
        if com_args is not None:
            com_index, atom_mask = com_args[:2]
            if len(com_args) == 2:
                if not torch.any(atom_mask):
                    return torch.zeros(coords.shape[:-2], dtype=coords.dtype, device=coords.device)
                num_coms = int(com_index[atom_mask].max().item()) + 1
            else:
                num_coms = cast(tuple[Tensor, Tensor, int], com_args)[2]
            unpad_com_index = com_index[atom_mask]
            unpad_coords = local_coords[..., atom_mask, :]
            local_coords = torch.zeros(
                (*unpad_coords.shape[:-2], num_coms, 3),
                dtype=coords.dtype,
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        value = cast(Tensor, self.compute_variable(local_coords, index, compute_gradient=False))
        energy = cast(Tensor, self.compute_function(value, *args, compute_derivative=False))
        return energy.sum(dim=-1)

    def compute_gradient(
        self,
        coords: Tensor,
        feats: dict[str, Tensor],
        parameters: Mapping[str, float | int],
    ) -> Tensor:
        return self._compute_gradient(coords, self.compute_args(feats, parameters))

    def _compute_gradient(self, coords: Tensor, arguments: _Boltz1PotentialEvaluation) -> Tensor:
        index, args, com_args = arguments
        if index.shape[1] == 0:
            return torch.zeros_like(coords)
        local_coords = coords
        com_index = None
        if com_args is not None:
            com_index, atom_mask = com_args[:2]
            if len(com_args) == 2:
                if not torch.any(atom_mask):
                    return torch.zeros_like(coords)
                num_coms = int(com_index[atom_mask].max().item()) + 1
            else:
                num_coms = cast(tuple[Tensor, Tensor, int], com_args)[2]
            unpad_com_index = com_index[atom_mask]
            unpad_coords = local_coords[..., atom_mask, :]
            local_coords = torch.zeros(
                (*unpad_coords.shape[:-2], num_coms, 3),
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
        self,
        feats: dict[str, Tensor],
        parameters: Mapping[str, float | int],
    ) -> tuple[Tensor, _Boltz1PotentialArguments, tuple[Tensor, Tensor] | None]:
        index, arguments, com_args = self._resolve_parameters(
            self._lower(feats, 0),
            parameters,
        )
        if com_args is not None:
            com_index, atom_mask, _ = com_args
            return index, arguments, (com_index, atom_mask)
        return index, arguments, None

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        raise NotImplementedError

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
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

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        pair_index = feats["geometry_pair_index"][batch_index]
        lower_bounds = feats["geometry_lower_bounds"][batch_index]
        upper_bounds = feats["geometry_upper_bounds"][batch_index]
        bond_mask = feats["geometry_bond_mask"][batch_index]
        angle_mask = feats["geometry_angle_mask"][batch_index]
        atom_vdw_radii = (feats["reference_elements"].float() @ _vdw_radii(pair_index.device).unsqueeze(-1)).squeeze(
            -1
        )[batch_index]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        return pair_index, (lower_bounds, upper_bounds, bond_mask, angle_mask, bond_cutoffs), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
    ) -> _Boltz1ResolvedPotential:
        pair_index, raw_arguments, com_args = lowering
        lower_bounds, upper_bounds, bond_mask, angle_mask, bond_cutoffs = raw_arguments
        lower_bounds = lower_bounds.clone()
        upper_bounds = upper_bounds.clone()
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
    ) -> _Boltz1ResolvedPotential:
        pair_index, (template,), com_args = lowering
        upper_bounds = template * float(parameters["buffer"])
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
        reference_elements = feats["reference_elements"]
        atom_vdw_radii = (reference_elements.float() @ _vdw_radii(reference_elements.device).unsqueeze(-1)).squeeze(-1)[
            batch_index
        ]
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
        unbuffered_bounds = atom_vdw_radii[pair_index].sum(dim=0)
        return pair_index, (unbuffered_bounds,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
    ) -> _Boltz1ResolvedPotential:
        pair_index, (unbuffered_bounds,), com_args = lowering
        lower_bounds = unbuffered_bounds * (1.0 - float(parameters["buffer"]))
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), com_args


class Boltz1SymmetricChainComPotential(Boltz1DistancePotential, Boltz1FlatBottomPotential):
    r"""
    Distance potential separating the centers of mass of same-entity symmetric chain pairs.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        atom_chain_index = feats["chain_index"].gather(1, feats["atom_to_token"])[batch_index]
        atom_mask = feats["atom_mask"][batch_index].bool()
        pair_index = feats["symmetric_chain_index"][batch_index]
        if pair_index.shape[1] == 0:
            pair_index = pair_index[:, :0]
            lower_bounds = torch.empty(0, dtype=torch.float32, device=atom_chain_index.device)
            return pair_index, (lower_bounds,), None
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
    ) -> _Boltz1ResolvedPotential:
        pair_index, (template,), com_args = lowering
        lower_bounds = template * float(parameters["buffer"])
        return pair_index, (torch.ones_like(lower_bounds), lower_bounds, None), com_args


class Boltz1StereoBondPotential(Boltz1AbsoluteDihedralPotential, Boltz1FlatBottomPotential):
    r"""
    Absolute-dihedral potential enforcing E/Z stereo-bond orientations toward 0 or pi.
    """

    def _lower(self, feats: dict[str, Tensor], batch_index: int) -> _Boltz1PotentialLowering:
        index = feats["stereo_bond_index"][batch_index]
        orientation = feats["stereo_bond_is_e"][batch_index].bool()
        return index, (orientation,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
    ) -> _Boltz1ResolvedPotential:
        index, (orientation,), com_args = lowering
        buffer = float(parameters["buffer"])
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
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
        orientation = feats["chiral_atom_is_r"][batch_index].bool()
        return index, (orientation,), None

    def _resolve_parameters(
        self,
        lowering: _Boltz1PotentialLowering,
        parameters: Mapping[str, float | int],
    ) -> _Boltz1ResolvedPotential:
        index, (orientation,), com_args = lowering
        buffer = float(parameters["buffer"])
        lower_bounds = torch.zeros(orientation.shape, device=orientation.device)
        upper_bounds = torch.zeros(orientation.shape, device=orientation.device)
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
    ) -> _Boltz1ResolvedPotential:
        improper_index, _, com_args = lowering
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            float(parameters["buffer"]),
            device=improper_index.device,
        )
        return improper_index, (torch.ones_like(upper_bounds), None, upper_bounds), com_args


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
    ranking_plddt_weight: float,
    ranking_ptm_weight: float,
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

    plddt = _expected_metric(logits["plddt_logits"], end=100.0)  # 0-100, shared with AF3/Protenix
    num_samples = coordinates.shape[-3]
    token_mask = insert_sample_axis(input.token_mask, num_samples, dim=-2).to(plddt)
    complex_plddt = (plddt * token_mask).sum(dim=-1) / token_mask.sum(dim=-1)  # bare: no token, no pLDDT

    molecule_type = _molecule_type_for_confidence(input, num_samples, plddt.device)
    is_ligand_token = (molecule_type == 3).to(plddt)
    is_contact = (distances < 8).to(plddt)
    chain_index = insert_sample_axis(input.chain_index, num_samples, dim=-2)
    is_different_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(plddt)
    token_interface_mask = torch.max(
        is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
        dim=-1,
    ).values
    iplddt_weight = is_ligand_token * 2.0 + token_interface_mask
    # A single-chain, ligand-free complex has an identically zero weight, so this is NaN by design: a monomer
    # has no interface and the interface pLDDT is undefined rather than zero.
    complex_iplddt = (plddt * token_mask * iplddt_weight).sum(dim=-1) / (token_mask * iplddt_weight).sum(dim=-1)

    pde = _expected_metric(logits["pde_logits"], end=32.0)
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
    different_token = (~same_token).to(token_mask)
    token_pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * different_token * prob_contact
    # Bare denominators: with no contacting pair (and no cross-chain contacting pair) these are undefined.
    complex_pde = (pde * token_pair_mask).sum(dim=(-2, -1)) / token_pair_mask.sum(dim=(-2, -1))
    token_interface_pair_mask = token_pair_mask * is_different_chain
    complex_ipde = (pde * token_interface_pair_mask).sum(dim=(-2, -1)) / token_interface_pair_mask.sum(dim=(-2, -1))

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
        )
        scores["pae"] = pae
        scores.update(ptm_scores)
        ptm_for_ranking = torch.where(
            torch.isclose(ptm_scores["iptm"], torch.zeros_like(ptm_scores["iptm"])),
            ptm_scores["ptm"],
            ptm_scores["iptm"],
        )
        scores["confidence_score"] = (
            ranking_plddt_weight * (complex_plddt / 100.0) + ranking_ptm_weight * ptm_for_ranking
        ) / (ranking_plddt_weight + ranking_ptm_weight)
    else:
        scores["confidence_score"] = complex_plddt / 100.0
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
) -> dict[str, Tensor]:
    r"""
    Compute pTM, ipTM, and the ligand/protein interface ipTM variants from PAE logits.

    Args:
        pae_logits: `[B, S, N, N, num_bins]` PAE logits.
        coordinates: `[B, S, M, 3]` sampled coordinates used to build valid frames.
        input: The Boltz-1 input batch.

    Returns:
        Mapping with ``ptm``, ``iptm``, ``ligand_iptm``, ``protein_iptm``, and the ``[B, S, C, C]``
        per-chain-pair ``chain_pair_iptm`` matrix.
    """

    num_samples = coordinates.shape[-3]
    token_mask = insert_sample_axis(input.token_mask, num_samples, dim=-2).to(pae_logits)
    frame_mask = _boltz1_coordinate_frames(coordinates, input=input)[1].to(pae_logits)
    chain_index = insert_sample_axis(input.chain_index, num_samples, dim=-2)
    pair_mask_ptm = frame_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * token_mask.unsqueeze(-1)
    pair_mask_iptm = pair_mask_ptm * (chain_index.unsqueeze(-2) != chain_index.unsqueeze(-1)).to(pae_logits)

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
        dim=-1,
    ).values
    iptm = torch.max(
        (tm_expected * pair_mask_iptm).sum(dim=-1) / pair_mask_iptm.sum(dim=-1).clamp_min(tiny),
        dim=-1,
    ).values

    molecule_type = _molecule_type_for_confidence(input, num_samples, pae_logits.device)
    is_ligand = (molecule_type == 3).to(pae_logits)
    is_protein = (molecule_type == 0).to(pae_logits)
    ligand_mask = pair_mask_iptm * (
        is_ligand.unsqueeze(-1) * is_protein.unsqueeze(-2) + is_protein.unsqueeze(-1) * is_ligand.unsqueeze(-2)
    )
    protein_mask = pair_mask_iptm * (is_protein.unsqueeze(-1) * is_protein.unsqueeze(-2))
    ligand_iptm = torch.max(
        (tm_expected * ligand_mask).sum(dim=-1) / ligand_mask.sum(dim=-1).clamp_min(tiny),
        dim=-1,
    ).values
    protein_iptm = torch.max(
        (tm_expected * protein_mask).sum(dim=-1) / protein_mask.sum(dim=-1).clamp_min(tiny),
        dim=-1,
    ).values

    # Restore the per-chain-pair interface pTM matrix. Entry ``[i, j]``
    # is the max-over-frame interface pTM of query chain ``i``
    # against key chain ``j``; the matrix is not symmetrized, preserving the query/key roles.
    unique_chain_indices = torch.unique(input.chain_index).tolist()
    chain_pair_rows = []
    for query_index in unique_chain_indices:
        chain_pair_columns = []
        for key_index in unique_chain_indices:
            chain_pair_mask = (
                pair_mask_ptm
                * (chain_index.unsqueeze(-2) == key_index).to(pae_logits)
                * (chain_index.unsqueeze(-1) == query_index).to(pae_logits)
            )
            chain_pair_columns.append(
                torch.max(
                    (tm_expected * chain_pair_mask).sum(dim=-1) / chain_pair_mask.sum(dim=-1).clamp_min(tiny),
                    dim=-1,
                ).values
            )
        chain_pair_rows.append(torch.stack(chain_pair_columns, dim=-1))
    chain_pair_iptm = torch.stack(chain_pair_rows, dim=-2)

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
            distances = torch.cdist(batch_coordinates, batch_coordinates)
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
    Gather canonical truth onto the current padded model atom axis.
    """

    atom_index = input.atom_index.long()
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
        pocket: `ComponentReference` values
            values naming the pocket contact residues.
    """

    binder: str
    pocket: tuple[ComponentReference, ...] = ()


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
        chain_equivalence_index: Entity-and-layout equivalence ids with
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
        token_distogram_atom_index: `[B, N]` distogram-atom index per token.
        residue_type_features: `[B, N, vocab_size]` residue-type one-hot features.
        msa_profile: `[B, N, 38]` canonical MSA profile feature.
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
        system: Canonical molecular system represented by these tensors.
        atom_index: Canonical atom index for each model atom with shape
            ``(B, M)`` and ``-1`` padding.
        token_component_index: Canonical component index for each model token
            with shape ``(B, N)``.
        token_atom_index: Canonical atom index for atom-level tokens and ``-1``
            for component-level tokens, with shape ``(B, N)``.
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
    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    pocket_feature: Tensor | None = None


class _Boltz1TrunkOutput(NamedTuple):
    r"""
    Boltz-1 trunk representations returned by the recycling trunk.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


class Boltz1Output(_TensorCollection):
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


class Boltz1SamplingOutput(_TensorCollection):
    r"""
    Predictions from one Boltz-1 reverse-diffusion sampling pass.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    coordinates: Tensor


class Boltz1ConfidenceOutput(_TensorCollection):
    r"""
    Raw Boltz-1 confidence predictions and their official derived values.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)``.
        distogram_logits: Detached trunk distogram logits with shape
            ``(B, N, N, D)``.
        plddt_logits: Per-token pLDDT logits with shape ``(B, S, N, D_p)``.
        pde_logits: PDE logits with shape ``(B, S, N, N, D_d)``.
        resolved_logits: Per-token resolved logits with shape ``(B, S, N, 2)``.
        plddt: Expected per-token pLDDT values on the ``0--100`` scale with
            shape ``(B, S, N)``.
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
    pae_logits: _OptionalTensor
    pae: _OptionalTensor
    ptm: _OptionalTensor
    iptm: _OptionalTensor
    ligand_iptm: _OptionalTensor
    protein_iptm: _OptionalTensor
    chain_pair_iptm: _OptionalTensor


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


class Boltz1Representations(_TensorCollection):
    r"""
    Raw Boltz-1 trunk representations surfaced by ``Boltz1Model.representations``.

    Attributes:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor
