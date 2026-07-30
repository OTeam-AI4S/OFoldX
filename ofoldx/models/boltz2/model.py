r"""
Boltz-2 model.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import partial
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.evidence import Template
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    PackedAtomAttention,
    PackedAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedReferenceAtomEncoder,
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
from ...modules.contact import BoltzContactConditioning
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
from ...modules.transformer import ConditionedTransformerEncoder
from ...modules.transition import Transition
from ...modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, ScoreResult
from .config import Boltz2AffinityConfig, Boltz2AffinityHeadConfig, Boltz2Config


class Boltz2Model(Model):
    r"""
    Predict biomolecular complex structures with Boltz 2.

    Accepts [`Boltz2Input`][ofoldx.models.boltz2.model.Boltz2Input] and returns
    [`Boltz2Output`][ofoldx.models.boltz2.model.Boltz2Output].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import MolecularSystem, PolymerSequence
        >>> from ofoldx.models.boltz2 import Boltz2Config, Boltz2Model, Boltz2Processor
        >>> sequence = PolymerSequence.from_one_letter("G")
        >>> system = MolecularSystem({"protein": sequence}, {"A": "protein"})
        >>> input = Boltz2Processor()(system)
        >>> config = Boltz2Config(
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
        ...             "num_pairformer_blocks": 1, "num_pair_attention_heads": 1,
        ...             "num_msa_blocks": 1, "msa_size": 8, "msa_head_dim": 4,
        ...             "num_msa_attention_heads": 1, "outer_product_hidden_size": 4,
        ...             "msa_triangle_attention_hidden_size": 4,
        ...             "num_msa_triangle_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "pair_dropout": 0.0, "msa_dropout": 0.0, "msa_pair_dropout": 0.0,
        ...         },
        ...         "template": {
        ...             "hidden_size": 8, "num_blocks": 1, "num_distogram_bins": 4,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1, "dropout": 0.0,
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
        ...             "num_pairformer_blocks": 1, "num_pair_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "num_distogram_bins": 4, "num_plddt_bins": 4,
        ...             "num_pae_bins": 4, "num_pde_bins": 4,
        ...             "pair_dropout": 0.0,
        ...         },
        ...     }
        ... )
        >>> model = Boltz2Model(config).eval()
        >>> with torch.inference_mode():
        ...     coordinates = model.sample(
        ...         input,
        ...         num_samples=1,
        ...         num_sampling_steps=2,
        ...         recycling_steps=1,
        ...         contact_guidance=False,
        ...     )
        >>> tuple(coordinates.shape)
        (1, 1, 32, 3)
    """

    config_class = Boltz2Config
    config: Boltz2Config

    def __init__(self, config: Boltz2Config | None = None) -> None:
        config = Boltz2Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
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
            msa_profile_size=input_embeddings_config.msa_profile_size,
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
        self.contact_conditioning = BoltzContactConditioning(
            pair_size, cutoff_min=config.conditioning_cutoff_min, cutoff_max=config.conditioning_cutoff_max
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
                triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                dropout=config.template.dropout,
                backend="auto",
            )
        self.msa_trunk = Boltz2MsaTrunk(
            single_size=single_size,
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
        self.pairformer_stack = Boltz2PairformerStack(
            num_blocks=trunk.num_pairformer_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
            num_triangle_attention_heads=trunk.num_triangle_attention_heads,
            num_pair_attention_heads=trunk.num_pair_attention_heads,
            dropout=trunk.pair_dropout,
            post_layer_norm=trunk.post_layer_norm,
            backend="auto",
        )
        self.distogram_head = PairSymmetrizedDistogramHead(pair_size=pair_size, num_bins=config.num_distogram_bins)

        denoiser_config = config.denoiser
        self.denoiser_conditioning = Boltz2DenoiserConditioning(
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
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
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
            triangle_attention_hidden_size=confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=confidence.num_triangle_attention_heads,
            num_distogram_bins=confidence.num_distogram_bins,
            max_distance=confidence.max_distance,
            relative_position_size=config.relative_position_size,
            num_bond_types=config.num_bond_types,
            conditioning_cutoff_min=config.conditioning_cutoff_min,
            conditioning_cutoff_max=config.conditioning_cutoff_max,
            num_plddt_bins=confidence.num_plddt_bins,
            num_pae_bins=confidence.num_pae_bins,
            num_pde_bins=confidence.num_pde_bins,
            pair_dropout=confidence.pair_dropout,
            post_layer_norm=confidence.post_layer_norm,
            normalize_single_input=confidence.normalize_single_input,
            backend="auto",
        )
        self.bfactor_head: nn.Linear | None = None
        if config.predict_bfactor:
            self.bfactor_head = nn.Linear(single_size, config.num_distogram_bins)
        self.post_init()

    def forward(
        self,
        input: Boltz2Input,
        target: Boltz2StructureTarget,
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
    ) -> Boltz2Output:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model-visible structure and evidence features.
            target: Complete pre-crop structure truth. Its conformer axis must
                have length one or ``num_samples``.
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
            generator: Optional generator for MSA row sampling, log-normal noise
                levels, rigid augmentation, and Gaussian coordinate noise.

        Returns:
            Denoised and centered target coordinates with shape ``(B, S, A, 3)``,
            noise levels with shape ``(B, S)`` in angstroms, and
            distogram logits with shape ``(B, N, N, D)``.
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
        target_positions, _ = _gather_boltz2_structure_target(
            target,
            input,
            num_samples=num_samples,
        )
        conditioning = self._build_denoiser_conditioning(representations, input)
        diffusion_output = self.coordinate_diffusion.denoise_training(
            target_positions,
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
        denoised_coordinates, augmented_target_coordinates, noise_level = diffusion_output
        batch_size = representations.single.shape[0]
        bfactor_logits = self.bfactor_head(representations.single) if self.bfactor_head is not None else None
        return Boltz2Output(  # type: ignore[call-arg]
            coordinates=_unflatten_sample_axis(denoised_coordinates, batch_size, num_samples),
            distogram_logits=representations.distogram_logits,
            augmented_target_coordinates=_unflatten_sample_axis(augmented_target_coordinates, batch_size, num_samples),
            noise_level=noise_level.reshape(batch_size, num_samples),
            bfactor_logits=bfactor_logits,
            batch_size=[batch_size],
        )

    def confidence(
        self,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2ConfidenceOutput:
        r"""
        Return raw differentiable confidence predictions for coordinate samples.

        Args:
            input: Model-visible structure and evidence features.
            coordinates: Atom coordinates with shape ``(B, S, A, 3)``.
            recycling_steps: Number of trunk recycling iterations.
            msa_max_depth: Optional MSA rows retained per MSA-module invocation.
            generator: Optional generator for MSA row sampling.
        """

        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
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
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Representations:
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
        return Boltz2Representations(  # type: ignore[call-arg]
            single=representations.single,
            pair=representations.pair,
            batch_size=[representations.single.shape[0]],
        )

    def sample(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        recycling_steps: int | None = None,
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
    ) -> Tensor:
        r"""
        Sample raw atom coordinates with explicit ``(B, S, A, 3)`` axes.
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
        return _unflatten_sample_axis(
            coordinates,
            input.token_mask.shape[0],
            num_samples,
        )

    def _confidence_from_representations(
        self,
        representations: _Boltz2TrunkOutput,
        input: Boltz2Input,
        coordinates: Tensor,
    ) -> Boltz2ConfidenceOutput:
        batch_size, num_samples = coordinates.shape[:2]
        flat_coordinates = coordinates.flatten(0, 1)
        confidence = self.confidence_head(
            representations.single_inputs,
            representations.single,
            representations.pair,
            flat_coordinates,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            token_bonds=input.token_bonds,
            type_bonds=input.type_bonds,
            contact_conditioning=input.contact_conditioning,
            contact_threshold=input.contact_threshold,
            token_to_rep_atom=input.token_to_rep_atom,
            token_mask=input.token_mask,
            cyclic_period=input.cyclic_period,
            multiplicity=num_samples,
        )
        plddt_logits = _unflatten_sample_axis(
            confidence["plddt_logits"],
            batch_size,
            num_samples,
        )
        pae_logits = _unflatten_sample_axis(
            confidence["pae_logits"],
            batch_size,
            num_samples,
        )
        pde_logits = _unflatten_sample_axis(
            confidence["pde_logits"],
            batch_size,
            num_samples,
        )
        resolved_logits = _unflatten_sample_axis(
            confidence["resolved_logits"],
            batch_size,
            num_samples,
        )
        values = _boltz2_confidence_values(
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            coordinates=coordinates,
            distogram_logits=representations.distogram_logits,
            input=input,
        )
        bfactor_logits = self.bfactor_head(representations.single) if self.bfactor_head is not None else None
        return Boltz2ConfidenceOutput(  # type: ignore[call-arg]
            coordinates=coordinates,
            distogram_logits=representations.distogram_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            bfactor_logits=bfactor_logits,
            **values,
            batch_size=[batch_size],
        )

    @torch.inference_mode()
    def fold(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
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
    ) -> FoldResult:
        r"""
        Predict structures for one molecular system with Boltz 2.

        Args:
            input: Processor output for one system, with batch size ``B = 1`` and
                molecular identities aligned to its atom and token axes.
            num_samples: Number of coordinate samples.
            generator: Optional generator for MSA row sampling, initial
                coordinates, churn noise, and per-step rigid augmentations.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            max_parallel_samples: Maximum samples evaluated by one denoiser call.
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
            raise ValueError("Boltz2Model.fold accepts one molecular system per call")
        if input.layout is None:
            raise ValueError("Boltz2Model.fold requires the canonical layout produced by its Processor")

        output = self._fold(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            max_parallel_samples=max_parallel_samples,
            feynman_kac=feynman_kac or use_potentials,
            physical_guidance=physical_guidance or use_potentials,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            num_samples=num_samples,
            generator=generator,
        )
        return _decode_boltz2_fold_result(output, input)

    def _fold(
        self,
        input: Boltz2Input,
        *,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Boltz2ConfidenceOutput:
        r"""
        Inference verb: fold (trunk) -> sample coords (diffusion) -> score (confidence).

        Runs the recycled trunk, the multi-step EDM reverse-diffusion sampler, and the confidence head. This is
        the distinct *inference* path (the training path is
        [`forward`][ofoldx.models.boltz2.model.Boltz2Model.forward]); the returned raw
        [`Boltz2ConfidenceOutput`][ofoldx.models.boltz2.model.Boltz2ConfidenceOutput]
        is the raw folding carrier; the public [`fold`][] method decodes it into
        ``FoldResult``.

        Args:
            input: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            num_samples: Number of reverse-diffusion samples to draw.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output with sampled coords, distogram logits, and confidence logits.

        """

        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        sample_coords_flat = self._sample_coords(
            representations,
            input,
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
        )
        batch_size = representations.single.shape[0]
        return self._confidence_from_representations(
            representations,
            input,
            _unflatten_sample_axis(sample_coords_flat, batch_size, num_samples),
        )

    def _run_trunk(
        self,
        input: Boltz2Input,
        *,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _Boltz2TrunkOutput:
        r"""
        Run the recycled folding trunk (template + MSA + Pairformer) and emit the trunk representations.

        Args:
            input: Typed model input.
            recycling_steps: Total number of trunk passes; defaults to the configured value. The trunk runs
                ``recycling_steps`` times.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation.
            generator: Optional generator advanced by each MSA row sample.

        Returns:
            Trunk representations and distogram logits.

        """

        recycling_steps = operator.index(self._recycling_steps if recycling_steps is None else recycling_steps)
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        single_inputs, single_init, pair_init, relative_position_encoding = self._initial_representations(input)
        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        # Boltz runs recycling_steps trunk passes.
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
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
                        beta_carbon_positions=cast(
                            Tensor,
                            input.template_beta_carbon_positions,
                        ),
                        alpha_carbon_positions=cast(
                            Tensor,
                            input.template_alpha_carbon_positions,
                        ),
                        beta_carbon_mask=cast(Tensor, input.template_beta_carbon_mask),
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

        return _Boltz2TrunkOutput(  # type: ignore[call-arg]
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            distogram_logits=self.distogram_head(pair),
            relative_position_encoding=relative_position_encoding,
            batch_size=[single_inputs.shape[0]],
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
            Flattened sampled atom coords ``(B*num_samples, M, 3)``.
        """

        conditioning = self._build_denoiser_conditioning(representations, input)
        return self.coordinate_diffusion._sample(
            input.atom_mask,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            multiplicity=num_samples,
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
    ) -> dict[str, Tensor]:
        r"""
        Precompute the per-call-invariant diffusion conditioning from the trunk representations.

        Shared by the training denoising step ([`forward`][ofoldx.models.boltz2.model.Boltz2Model.forward]) and
        the reverse-diffusion sampler ([`_sample_coords`][ofoldx.models.boltz2.model.Boltz2Model._sample_coords]).

        Args:
            representations: Trunk representations from
                [`_run_trunk`][ofoldx.models.boltz2.model.Boltz2Model._run_trunk].
            input: Typed model input.

        Returns:
            The [`Boltz2DenoiserConditioning`][ofoldx.models.boltz2.model.Boltz2DenoiserConditioning] output dict.
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

    Examples:
        >>> import torch
        >>> from ofoldx.data import (
        ...     ChemicalAtom,
        ...     ChemicalComponent,
        ...     MolecularSystem,
        ...     Molecule,
        ...     PolymerSequence,
        ...     get_component,
        ... )
        >>> from ofoldx.models.boltz2 import (
        ...     Boltz2AffinityConfig,
        ...     Boltz2AffinityHeadConfig,
        ...     Boltz2AffinityModel,
        ...     Boltz2AffinityProcessor,
        ... )
        >>> protein = PolymerSequence("protein", (get_component("GLY"),))
        >>> ligand = Molecule(
        ...     (
        ...         ChemicalComponent(
        ...             "LIG",
        ...             (ChemicalAtom("C1", "C"),),
        ...             frozenset(),
        ...             reference_coordinates=((0.0, 0.0, 0.0),),
        ...         ),
        ...     )
        ... )
        >>> system = MolecularSystem(
        ...     {"protein": protein, "ligand": ligand},
        ...     {"A": "protein", "L": "ligand"},
        ... )
        >>> input = Boltz2AffinityProcessor()(system)
        >>> config = Boltz2AffinityConfig(
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
        ...             "num_pairformer_blocks": 1, "num_pair_attention_heads": 1,
        ...             "num_msa_blocks": 1, "msa_size": 8, "msa_head_dim": 4,
        ...             "num_msa_attention_heads": 1, "outer_product_hidden_size": 4,
        ...             "msa_triangle_attention_hidden_size": 4,
        ...             "num_msa_triangle_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "pair_dropout": 0.0, "msa_dropout": 0.0, "msa_pair_dropout": 0.0,
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
        ...             "num_pairformer_blocks": 1, "num_pair_attention_heads": 1,
        ...             "triangle_attention_hidden_size": 4,
        ...             "num_triangle_attention_heads": 1,
        ...             "num_distogram_bins": 4, "num_plddt_bins": 4,
        ...             "num_pae_bins": 4, "num_pde_bins": 4,
        ...             "pair_dropout": 0.0,
        ...         },
        ...         "affinity_heads": (
        ...             Boltz2AffinityHeadConfig({
        ...                 "num_pair_update_blocks": 1,
        ...                 "num_distogram_bins": 4,
        ...                 "num_conditioning_transitions": 1,
        ...                 "triangle_attention_hidden_size": 4,
        ...                 "num_triangle_attention_heads": 1,
        ...                 "dropout": 0.0,
        ...             }),
        ...             Boltz2AffinityHeadConfig({
        ...                 "num_pair_update_blocks": 1,
        ...                 "num_distogram_bins": 4,
        ...                 "num_conditioning_transitions": 1,
        ...                 "triangle_attention_hidden_size": 4,
        ...                 "num_triangle_attention_heads": 1,
        ...                 "dropout": 0.0,
        ...             }),
        ...         ),
        ...     }
        ... )
        >>> model = Boltz2AffinityModel(config).eval()
        >>> with torch.inference_mode():
        ...     score = model.affinity(
        ...         input,
        ...         torch.zeros(1, 1, input.atom_mask.shape[-1], 3),
        ...         recycling_steps=1,
        ...     )
        >>> tuple(score.affinity_pred_value.shape)
        (1, 1, 1)
        >>> tuple(score.affinity_probability_binary.shape)
        (1, 1, 1)
    """

    config_class = Boltz2AffinityConfig
    config: Boltz2AffinityConfig

    def __init__(self, config: Boltz2AffinityConfig | None = None) -> None:
        config = Boltz2AffinityConfig() if config is None else config
        super().__init__(config)
        self.affinity_heads = nn.ModuleList(self._build_affinity_head(head) for head in config.affinity_heads)
        self.post_init(backend=self.backend)

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
            triangle_attention_hidden_size=affinity.triangle_attention_hidden_size,
            num_triangle_attention_heads=affinity.num_triangle_attention_heads,
            dropout=affinity.dropout,
            backend="auto",
        )

    def affinity(
        self,
        input: Boltz2Input,
        coordinates: Tensor,
        *,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityOutput:
        r"""
        Return raw differentiable affinity predictions for coordinate samples.

        Args:
            input: Affinity Processor output.
            coordinates: Candidate coordinates with shape ``(B, S, A, 3)``.
            recycling_steps: Number of trunk recycling iterations.
            msa_max_depth: Optional MSA rows retained per MSA-module invocation.
            molecular_weight_correction: Whether to calibrate the ensemble value.
            generator: Optional generator for MSA row sampling.
        """

        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        values = self._score_affinity(
            representations,
            input,
            coordinates,
            molecular_weight_correction=molecular_weight_correction,
        )
        return Boltz2AffinityOutput(  # type: ignore[call-arg]
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

    @torch.inference_mode()
    def score(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 5,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
    ) -> ScoreResult:
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
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            msa_max_depth: Optional MSA rows sampled independently for every
                MSA-module invocation. ``None`` retains every processed row.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            molecular_weight_correction: Whether to apply molecular-weight
                calibration to the ensemble affinity value.

        Returns:
            The selected structure and six scalar scores. Affinity values use
            ``log10(IC50 / micromolar)`` units, where lower is stronger; binder
            probabilities lie in ``[0, 1]``.

        Raises:
            ValueError: If a count is not positive, the input contains more than
                one system, or no affinity ligand is identified.
        """

        num_samples = operator.index(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if recycling_steps is not None:
            recycling_steps = operator.index(recycling_steps)
            if recycling_steps < 1:
                raise ValueError("recycling_steps must be positive")
        if num_sampling_steps is not None:
            num_sampling_steps = operator.index(num_sampling_steps)
            if num_sampling_steps < 2:
                raise ValueError("num_sampling_steps must be at least 2")
        if input.token_mask.shape[0] != 1:
            raise ValueError("Boltz2AffinityModel.score accepts one molecular system per call")
        if input.layout is None:
            raise ValueError("Boltz2AffinityModel.score requires the canonical layout produced by its Processor")
        if input.layout.affinity_instance_id is None:
            raise ValueError("Boltz2AffinityModel.score requires one canonical affinity ligand instance")

        structure_output, raw_scores = self._score(
            input,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        selected_index = int(raw_scores["selected_sample_index"][0].item())
        selected_coords = structure_output.coordinates[0, selected_index]
        layout = input.layout
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        model_coords = selected_coords[: len(layout.atom_sites)].float().cpu().numpy()
        for site, values in zip(layout.atom_sites, model_coords):
            index = complete_index[site]
            coordinates[index] = values
            coordinate_mask[index] = True
        structure = Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )

        def scalar(name: str) -> float:
            return float(raw_scores[name][0, 0, 0].item())

        # Return the six affinity scores, each scoped to the selected ligand instance. The
        # ensemble binding affinity carries the optional molecular-weight correction; the
        # two per-head values are uncorrected. The molecular-weight input echoes (ligand
        # molecular weight, molecular_weight_correction), recycling/sampling run-controls, the affinity ligand
        # atom mask (reconstructable from the affinity ligand scope), and the run provenance / RNG echo. The
        # first-stage per-sample confidence and its ranked selection live on the Pipeline's
        # ``structure_predictions``; this root pass returns only the second-pass ``target`` and the six Scores.
        ligand_id = cast(Boltz2Layout, input.layout).affinity_instance_id
        scope = frozenset({cast(str, ligand_id)})
        return ScoreResult(
            target=structure,
            scores={
                "affinity_pred_value": Score(
                    scalar("affinity_pred_value"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary": Score(
                    scalar("affinity_probability_binary"),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_pred_value1": Score(
                    scalar("affinity_pred_value1"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_pred_value2": Score(
                    scalar("affinity_pred_value2"),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary1": Score(
                    scalar("affinity_probability_binary1"),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_probability_binary2": Score(
                    scalar("affinity_probability_binary2"),
                    preferred="higher",
                    scope=scope,
                ),
            },
        )

    @torch.inference_mode()
    def _score(
        self,
        input: Boltz2Input,
        *,
        coordinates: Tensor | None = None,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Boltz2ConfidenceOutput, dict[str, Tensor]]:
        r"""
        Raw affinity protocol, optionally using caller-supplied candidate coords.

        Args:
            input: Typed Boltz-2 input. ``affinity_token_mask`` may select the ligand; otherwise ligand tokens are
                inferred from ``molecule_type == 3``.
            coordinates: Optional atom coordinates to score with shape ``(B, S, M, 3)``.
            recycling_steps: Optional trunk recycling passes. Defaults to the config value.
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
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            generator=generator,
        )
        batch_size = representations.single.shape[0]
        if coordinates is None:
            sample_coords_flat = self._sample_coords(
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
            coordinates = _unflatten_sample_axis(sample_coords_flat, batch_size, num_samples)
        else:
            num_samples = coordinates.shape[1]
        structure_prediction = self._confidence_from_representations(
            representations,
            input,
            coordinates,
        )
        iptm = structure_prediction.iptm
        selected_sample_index = iptm.argmax(dim=1)
        selected_coords = coordinates[torch.arange(batch_size, device=coordinates.device), selected_sample_index]
        scores = self._score_affinity(
            representations,
            input,
            selected_coords.unsqueeze(1),
            molecular_weight_correction=molecular_weight_correction,
        )
        scores["iptm_by_sample"] = iptm
        scores["selected_sample_index"] = selected_sample_index
        return structure_prediction, scores

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

        batch_size, multiplicity = coordinates.shape[:2]
        coords = coordinates.flatten(0, 1)
        if affinity_token_mask is None:
            affinity_token_mask = input.affinity_token_mask
        if affinity_token_mask is None:
            raise ValueError("Boltz-2 affinity scoring requires one explicit ligand token mask")
        if input.affinity_msa_profile is None or input.affinity_deletion_mean is None:
            raise ValueError("Boltz-2 affinity scoring requires its single-sequence affinity profile")
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
            "token_to_rep_atom": input.token_to_rep_atom,
            "token_mask": input.token_mask,
            "ligand_mask": ligand_mask,
            "receptor_mask": receptor_mask,
            "multiplicity": multiplicity,
        }
        flat_scores1, flat_scores2 = (head(**head_kwargs) for head in self.affinity_heads)
        scores1 = {
            name: _unflatten_sample_axis(tensor, batch_size, multiplicity) for name, tensor in flat_scores1.items()
        }
        scores2 = {
            name: _unflatten_sample_axis(tensor, batch_size, multiplicity) for name, tensor in flat_scores2.items()
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
            if affinity_mw is None:
                raise ValueError("molecular-weight correction requires the selected ligand molecular weight")
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
        vocab_size: int = 33,
        msa_profile_size: int = 34,
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_mol_type_feat: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_encoder = PackedReferenceAtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_pair_norm = LayerNorm(atom_pair_hidden_size, backend=backend)
        self.atom_pair_bias_projection = nn.Linear(
            atom_pair_hidden_size,
            num_atom_encoder_blocks * num_atom_encoder_attention_heads,
            bias=False,
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

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(msa_profile_size, single_size, bias=False)
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
        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features, reference_positions, atom_mask, reference_space_id
        )
        atom_pair_bias = self.atom_pair_bias_projection(self.atom_pair_norm(atom_pair))
        token_from_atom = self.atom_attention_encoder(
            atom_query,
            atom_conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            residue_type_features.shape[-2],
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
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        expansion: int = 4,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        template_feature_size = 2 * vocab_size + num_bins + 5
        self.template_projection = nn.Linear(template_feature_size, hidden_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.pair_update_stack = Boltz2TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            triangle_multiplication_hidden_size=hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
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
        beta_carbon_positions: Tensor,
        alpha_carbon_positions: Tensor,
        beta_carbon_mask: Tensor,
        visibility_index: Tensor,
    ) -> Tensor:
        num_tokens = residue_type_features.shape[2]
        pair_beta_carbon_mask = (beta_carbon_mask[..., :, None] * beta_carbon_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility_index[..., :, None] == visibility_index[..., None, :]).to(
            residue_type_features.dtype
        )

        beta_carbon_distances = torch.cdist(beta_carbon_positions, beta_carbon_positions)
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=beta_carbon_distances.device,
        )
        distogram = (beta_carbon_distances[..., None] > boundaries).sum(dim=-1).long()
        distogram = F.one_hot(distogram, num_classes=self.num_bins).to(residue_type_features.dtype)

        # Per-frame unit vector of CA_j in frame_i (replicated faithfully, incl. the per-component norm).
        rotation = frame_rotations.unsqueeze(2).transpose(-1, -2)
        translation = frame_translations.unsqueeze(2).unsqueeze(-1)
        targets = alpha_carbon_positions.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(rotation, targets - translation)
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

        geometry = torch.cat(
            [distogram, pair_beta_carbon_mask, unit_vector, pair_frame_mask],
            dim=-1,
        )
        geometry = geometry * same_visibility.unsqueeze(-1)
        residue_features_i = residue_type_features[:, :, :, None].expand(-1, -1, -1, num_tokens, -1)
        residue_features_j = residue_type_features[:, :, None, :].expand(-1, -1, num_tokens, -1, -1)
        return torch.cat([geometry, residue_features_i, residue_features_j], dim=-1)

    def forward(
        self,
        pair: Tensor,
        *,
        residue_type_features: Tensor,
        frame_rotations: Tensor,
        frame_translations: Tensor,
        frame_mask: Tensor,
        beta_carbon_positions: Tensor,
        alpha_carbon_positions: Tensor,
        beta_carbon_mask: Tensor,
        visibility_index: Tensor,
        template_mask: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        batch, num_templates = residue_type_features.shape[:2]
        present = template_mask.any(dim=2).to(pair.dtype)
        template_count = present.sum(dim=1).clamp(min=1)

        features = self._featurize(
            residue_type_features,
            frame_rotations,
            frame_translations,
            frame_mask,
            beta_carbon_positions,
            alpha_carbon_positions,
            beta_carbon_mask,
            visibility_index,
        )
        projected = self.template_projection(features)
        template_pair = self.pair_projection(self.pair_norm(pair[:, None])) + projected
        template_pair = template_pair.reshape(batch * num_templates, *template_pair.shape[2:])

        if pair_mask is not None:
            pair_mask = pair_mask[:, None].expand(-1, num_templates, -1, -1)
            pair_mask = pair_mask.reshape(batch * num_templates, *pair_mask.shape[2:])

        stacked = self.pair_update_stack(template_pair, pair_mask=pair_mask)
        template_pair = self.hidden_norm(template_pair + stacked)
        template_pair = template_pair.reshape(batch, num_templates, *template_pair.shape[1:])

        weight = present[:, :, None, None, None]
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
        vocab_size: int = 33,
        use_paired_feature: bool = True,
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
        self.msa_projection = nn.Linear(
            vocab_size + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            Boltz2MsaBlock(
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


class Boltz2MsaBlock(nn.Module):
    r"""
    One Boltz-2 MSA block with scalar dropout per MSA row.
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
        self.triangle_pair_update = Boltz2TrianglePairUpdateBlock(
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


class Boltz2PairformerStack(nn.Module):
    r"""
    Homogeneous stack of Boltz-2 Pairformer blocks.
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
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            Boltz2PairformerBlock(
                single_size=single_size,
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                expansion=expansion,
                dropout=dropout,
                post_layer_norm=post_layer_norm,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class Boltz2PairformerBlock(nn.Module):
    r"""
    One Boltz-2 Pairformer block.
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
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_update = Boltz2TrianglePairUpdateBlock(
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
        self.single_post_norm = LayerNorm(single_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pair = self.pair_update(pair, pair_mask=pair_mask)
        single = single + self.pair_attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return self.single_post_norm(single), pair


class Boltz2TrianglePairUpdateStack(nn.Module):
    r"""
    Homogeneous stack of Boltz-2 pair-only triangle update blocks.
    """

    def __init__(
        self,
        num_blocks: int,
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
            Boltz2TrianglePairUpdateBlock(
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


class Boltz2TrianglePairUpdateBlock(nn.Module):
    r"""
    Boltz-2 pair update with one dropout scalar per pair row or column.
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


class Boltz2DenoiserConditioning(nn.Module):
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
        self.atom_encoder = Boltz2DenoiserAtomEncoder(
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


class Boltz2Denoiser(nn.Module):
    r"""
    EDM score network: ``(noisy positions, noise level, trunk, conditioning) -> position update``.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
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
        self.single_conditioner = Boltz2SingleConditioning(
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
            post_layer_norm=post_layer_norm,
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
        self.atom_attention_decoder = Boltz2AtomAttentionDecoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_decoder_blocks=num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=num_atom_decoder_attention_heads,
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
    ) -> Tensor:
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

        return self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            conditioning["atom_dec_bias"].repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            multiplicity=multiplicity,
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
        normal = torch.empty(batch_size, device=device, dtype=dtype).normal_(generator=generator)
        return self.sigma_data * (log_mean + log_std * normal).exp()

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2 + 1e-8)

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
        **conditioning: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        batch_size = atom_mask.shape[0]
        target = target_coordinates.to(device=device, dtype=dtype).reshape(
            batch_size * multiplicity,
            target_coordinates.shape[-2],
            3,
        )
        # Centre and rigid-augment over ALL padded atoms (atom_pad_mask), matching upstream
        # ``center_random_augmentation``. The resolved-atom mask is a loss/alignment concern applied by the
        # criterion, not part of the corruption equation.
        augmentation_mask = atom_mask.to(device=device, dtype=dtype)
        augmentation_mask = augmentation_mask.repeat_interleave(multiplicity, 0)
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
        denoised = self.preconditioned_network_forward(
            noisy_coordinates,
            noise_level,
            multiplicity=multiplicity,
            atom_mask=atom_mask,
            **conditioning,
        )
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
        weight = mask[..., None]
        centroid = (coordinates * weight).sum(-2, keepdim=True) / weight.sum(-2, keepdim=True).clamp_min(1.0)
        coordinates = coordinates - centroid
        if not coordinate_augmentation:
            return coordinates
        rotation = uniform_random_rotation(
            coordinates.shape[0],
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        )
        translation = torch.empty(
            coordinates.shape[0],
            1,
            3,
            device=coordinates.device,
            dtype=coordinates.dtype,
        ).normal_(generator=generator)
        return torch.einsum("bmd,bds->bms", coordinates, rotation) + translation_scale * translation

    def preconditioned_network_forward(
        self, noisy_coordinates: Tensor, sigma: Tensor | float, *, multiplicity: int = 1, **conditioning: object
    ) -> Tensor:
        batch = noisy_coordinates.shape[0]
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device, dtype=noisy_coordinates.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        return self.preconditioner.c_skip(padded) * noisy_coordinates + self.preconditioner.c_out(padded) * update

    def sample_schedule(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        steps = operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps)
        if steps < 2:
            raise ValueError("num_sampling_steps must be at least 2")
        inv_rho = 1 / self.rho
        # `steps` is the number of sampling steps (transitions); the trailing pad adds the final sigma=0.
        ramp = torch.arange(steps, device=device, dtype=dtype) / (steps - 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    @torch.no_grad()
    def _sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        multiplicity: int = 1,
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
        multiplicity = operator.index(multiplicity)
        requested_multiplicity = multiplicity
        if feynman_kac:
            num_particles = operator.index(num_particles)
            if num_particles < 1:
                raise ValueError("num_particles must be positive")
            feynman_kac_resampling_interval = operator.index(feynman_kac_resampling_interval)
            if feynman_kac_resampling_interval < 1:
                raise ValueError("feynman_kac_resampling_interval must be positive")
        effective_multiplicity = multiplicity * num_particles if feynman_kac else multiplicity
        if max_parallel_samples is None:
            max_parallel_samples = effective_multiplicity
        else:
            max_parallel_samples = operator.index(max_parallel_samples)
            if max_parallel_samples < 1:
                raise ValueError("max_parallel_samples must be positive")
        if physical_guidance or contact_guidance:
            num_guidance_steps = operator.index(num_guidance_steps)
            if num_guidance_steps < 0:
                raise ValueError("num_guidance_steps must be non-negative")
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(effective_multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)

        def randn() -> Tensor:
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)

        coords = sigmas[0] * randn()
        if feynman_kac:
            energy_trajectory = torch.empty(
                batch_size * effective_multiplicity,
                0,
                device=device,
                dtype=dtype,
            )
            resample_weights = torch.ones(
                batch_size * requested_multiplicity,
                num_particles,
                device=device,
                dtype=dtype,
            )
        else:
            energy_trajectory = torch.empty(0, 0, device=device, dtype=dtype)
            resample_weights = torch.empty(0, 0, device=device, dtype=dtype)
        scaled_guidance_update = (
            torch.zeros(shape, device=device, dtype=dtype) if physical_guidance or contact_guidance else None
        )
        # `contact_guidance` defaults True, so the inner guidance loop would otherwise run its full
        # `num_guidance_steps` sweep of gradient evaluations every step even when the input carries no
        # constraint to guide toward. Physical potentials are coordinate-only and always contribute, but the
        # contact and forced-template fields produce exactly zero gradient when absent, so decide once whether
        # there is anything to guide (a single template `.any()` sync, hoisted out of the sampling loop).
        guidance_active = physical_guidance or (
            contact_guidance
            and (
                input.contact_pair_index.shape[-1] > 0
                or (input.template_force is not None and bool(input.template_force.any()))
            )
        )
        denoised = None
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
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms",
                    scaled_guidance_update,
                    rotation,
                )

            sigma_hat = sigma_current * (1 + gamma)
            noise_var = self.noise_scale**2 * (sigma_hat**2 - sigma_current**2)
            epsilon = torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
            noisy = coords + epsilon
            noisy_by_sample = noisy.reshape(
                batch_size,
                effective_multiplicity,
                *noisy.shape[1:],
            )
            denoised_by_sample = torch.zeros_like(noisy_by_sample)
            denoiser_conditioning = cast(
                dict[str, Tensor],
                conditioning["conditioning"],
            )
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
                    "conditioning": {
                        name: value[batch_start:batch_stop] for name, value in denoiser_conditioning.items()
                    },
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
                    effective_multiplicity,
                    sample_chunk_size,
                ):
                    sample_stop = min(
                        sample_start + sample_chunk_size,
                        effective_multiplicity,
                    )
                    sample_count = sample_stop - sample_start
                    denoised_by_sample[
                        batch_start:batch_stop,
                        sample_start:sample_stop,
                    ] = self.preconditioned_network_forward(
                        noisy_by_sample[
                            batch_start:batch_stop,
                            sample_start:sample_stop,
                        ].flatten(0, 1),
                        sigma_hat,
                        multiplicity=sample_count,
                        atom_mask=base_atom_mask[batch_start:batch_stop],
                        **batch_conditioning,
                    ).reshape(
                        batch_count,
                        sample_count,
                        *noisy.shape[1:],
                    )
            denoised = denoised_by_sample.flatten(0, 1)

            steering_t = 1.0 - index / (len(sigmas) - 1)
            should_resample = feynman_kac and (
                (index % feynman_kac_resampling_interval == 0 and noise_var > 0) or index == len(sigmas) - 2
            )
            if should_resample:
                active_physical = {0, 2, 3, 4, 5, 6}
                if steering_t <= 0.6:
                    active_physical.add(1)
                terms = _boltz2_potential_terms(
                    denoised,
                    input,
                    multiplicity=effective_multiplicity,
                    t=steering_t,
                    physical_guidance=physical_guidance,
                    contact_guidance=contact_guidance,
                    active_physical=frozenset(active_physical),
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
                if scaled_guidance_update is not None and noise_var > 0:
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
                        input,
                        multiplicity=effective_multiplicity,
                        t=steering_t,
                        guidance_step=guidance_step,
                        physical_guidance=physical_guidance,
                        contact_guidance=contact_guidance,
                    )
                denoised = denoised + guidance_update
                scaled_guidance_update = -guidance_update * step_scale * (sigma_next - sigma_hat) / sigma_hat

            if should_resample:
                sample_count = num_particles if index < len(sigmas) - 2 else 1
                offsets = num_particles * torch.arange(
                    resample_weights.shape[0],
                    device=device,
                ).unsqueeze(-1)
                resample_index = (
                    torch.multinomial(
                        resample_weights,
                        sample_count,
                        replacement=True,
                        generator=generator,
                    )
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
                    effective_multiplicity = requested_multiplicity

            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy.float(), target=denoised.float(), weight=expanded_atom_mask.float()
                ).to(denoised)
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale * (sigma_next - sigma_hat) * derivative
        return coords


class Boltz2DenoiserAtomEncoder(nn.Module):
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
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.single_trunk_norm = LayerNorm(single_size)
        self.single_trunk_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.pair_trunk_norm = LayerNorm(pair_size)
        self.pair_trunk_projection = nn.Linear(
            pair_size,
            atom_pair_hidden_size,
            bias=False,
        )
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
        single_from_trunk = single_proj.gather(1, atom_to_token.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1]))
        atom_single = atom_single + single_from_trunk * atom_mask.unsqueeze(-1).to(single_from_trunk.dtype)
        pair_from_trunk = self.pair_trunk_projection(self.pair_trunk_norm(pair_trunk))
        token_windows = pad_atoms(atom_to_token)
        query_token = token_windows.view(batch, num_windows, num_query_atoms)
        key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
        batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
        pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
        pair_bias = pair_gathered * (mask_queries & mask_keys).unsqueeze(-1).to(pair_gathered.dtype)

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


class Boltz2AtomAttentionDecoder(nn.Module):
    r"""
    Broadcast token activations back to atoms and predict a per-atom position update.
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2 * single_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            ConditionedTransformerEncoder(
                num_blocks=num_atom_decoder_blocks,
                num_attention_heads=num_atom_decoder_attention_heads,
                hidden_size=atom_hidden_size,
                conditioning_size=atom_hidden_size,
                backend=backend,
            ),
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
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
        multiplicity: int = 1,
    ) -> Tensor:
        mask = atom_mask.repeat_interleave(multiplicity, 0).bool()
        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        token_proj = self.token_projection(token_repr)
        token_to_atom = token_proj.gather(1, token_index.unsqueeze(-1).expand(-1, -1, token_proj.shape[-1]))
        atom_query = atom_skip + token_to_atom * mask.unsqueeze(-1).to(token_to_atom.dtype)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask,
            attention_layout="local",
        )
        return self.position_update_projection(self.position_norm(atom_query))


class Boltz2SingleConditioning(nn.Module):
    r"""
    Condition the (concatenated trunk+input) single representation on the diffusion noise level.
    """

    def __init__(
        self,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(hidden, backend=backend)
        self.single_projection = nn.Linear(hidden, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size, backend=backend)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=expansion, backend=backend) for _ in range(num_conditioning_transitions)
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


class Boltz2AffinityHead(nn.Module):
    r"""
    Predict scalar binding affinity and a binder-class from trunk features and predicted atom geometry.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.dist_bin_pairwise_embed = nn.Embedding(num_distogram_bins, pair_size)
        nn.init.zeros_(self.dist_bin_pairwise_embed.weight)
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            expansion=expansion,
            backend=backend,
        )
        self.pair_update_stack = Boltz2TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
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
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        ligand_mask: Tensor,
        receptor_mask: Tensor,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        pair = self.pair_projection(self.pair_norm(pair))
        pair = pair + self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0).long()
        representative = coordinates.gather(
            1,
            rep_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]),
        )
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.pairwise_conditioner(pair, self.dist_bin_pairwise_embed(distogram))

        token_mask = token_mask.repeat_interleave(multiplicity, 0).bool()
        ligand_mask = ligand_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
        receptor_mask = receptor_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
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
        self.pair_feature_input_projection = nn.Linear(pair_size, pair_size)
        self.pair_feature_output_projection = nn.Linear(pair_size, single_size)
        self.affinity_value_input_projection = nn.Linear(single_size, single_size)
        self.affinity_value_hidden_projection = nn.Linear(single_size, single_size)
        self.affinity_value_output_projection = nn.Linear(single_size, 1)
        self.affinity_score_input_projection = nn.Linear(single_size, single_size)
        self.affinity_score_hidden_projection = nn.Linear(single_size, single_size)
        self.affinity_score_output_projection = nn.Linear(single_size, 1)
        self.binder_logit_projection = nn.Linear(1, 1)

    def forward(self, pair: Tensor, affinity_pair_mask: Tensor) -> dict[str, Tensor]:
        num_tokens = affinity_pair_mask.shape[-1]
        non_diagonal = ~torch.eye(num_tokens, device=affinity_pair_mask.device, dtype=torch.bool).unsqueeze(0)
        pool_mask = (affinity_pair_mask.bool() & non_diagonal).to(dtype=pair.dtype).unsqueeze(-1)
        pooled = torch.sum(pair * pool_mask, dim=(-3, -2)) / (torch.sum(pool_mask, dim=(-3, -2)) + 1e-7)
        pooled = F.relu(self.pair_feature_input_projection(pooled))
        pooled = F.relu(self.pair_feature_output_projection(pooled))
        affinity_value = F.relu(self.affinity_value_input_projection(pooled))
        affinity_value = F.relu(self.affinity_value_hidden_projection(affinity_value))
        affinity_pred_value = self.affinity_value_output_projection(affinity_value).reshape(-1, 1)
        affinity_score = F.relu(self.affinity_score_input_projection(pooled))
        affinity_score = F.relu(self.affinity_score_hidden_projection(affinity_score))
        affinity_pred_score = self.affinity_score_output_projection(affinity_score).reshape(-1, 1)
        affinity_logits_binary = self.binder_logit_projection(affinity_pred_score).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }


class Boltz2ConfidenceHead(nn.Module):
    r"""
    Score predicted structures with per-token pLDDT / resolved / PAE / PDE confidence logits.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        relative_position_size: int = 139,
        num_bond_types: int = 7,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        pair_dropout: float = 0.25,
        post_layer_norm: bool = False,
        normalize_single_input: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_single_norm = LayerNorm(single_size, backend=backend)
        self.single_norm: LayerNorm | None = None
        if normalize_single_input:
            self.single_norm = LayerNorm(single_size, backend=backend)
        self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)

        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(num_bond_types, pair_size)
        self.contact_conditioning = BoltzContactConditioning(
            pair_size, cutoff_min=conditioning_cutoff_min, cutoff_max=conditioning_cutoff_max
        )

        self.single_to_pair_row = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)

        self.pairformer_stack = Boltz2PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            num_pair_attention_heads=num_pair_attention_heads,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            dropout=pair_dropout,
            post_layer_norm=post_layer_norm,
            backend=backend,
        )
        self.output_head = Boltz2ConfidenceOutputHead(
            single_size, pair_size, num_plddt_bins=num_plddt_bins, num_pae_bins=num_pae_bins, num_pde_bins=num_pde_bins
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
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        cyclic_period: Tensor | None = None,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
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

        single = single.repeat_interleave(multiplicity, 0)
        pair = (
            pair
            + self.single_to_pair_row(single_inputs).unsqueeze(-2)
            + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        )
        pair = pair + self.single_to_pair_product_out(
            self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
            * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        )
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        representative = coordinates.gather(1, rep_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1]))
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = token_mask.repeat_interleave(multiplicity, 0)
        pair_mask = pair_mask_from_token_mask(token_mask)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        chain_index = chain_index.repeat_interleave(multiplicity, 0)
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
        same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).float().unsqueeze(-1)
        different_chain = 1.0 - same_chain
        pae_logits = self.pae_intra_projection(pair) * same_chain + self.pae_inter_projection(pair) * different_chain
        symmetric = pair + pair.transpose(-3, -2)
        pde_logits = (
            self.pde_intra_projection(symmetric) * same_chain + self.pde_inter_projection(symmetric) * different_chain
        )
        return {
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }


def _decode_boltz2_fold_result(output: Boltz2ConfidenceOutput, input: Boltz2Input) -> FoldResult:
    r"""
    Decode one raw [`Boltz2ConfidenceOutput`][ofoldx.models.boltz2.model.Boltz2ConfidenceOutput]
    into a ``FoldResult``.

    Maps model atom order back onto the system's complete heavy-atom sites to build one ``Structure`` per
    sample, and lifts the confidence head's per-token pLDDT/resolved metrics,
    token-pair PAE/PDE, and the per-sample complex/design confidence scores onto
    values aligned to the layout's token axis.
    """

    layout = cast(Boltz2Layout, input.layout)
    token_count = len(layout.token_sites)
    atom_count = len(layout.atom_sites)

    complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)

    coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
    num_samples = coordinates.shape[0]
    # Each polymer token_site is already a ``ComponentReference`` (``(instance_id, monomer_index)``) and each
    # atomized-ligand token_site is an ``AtomSite`` -- both are valid ``MetricReference`` values, so the token
    # The metric axis follows layout.token_sites.
    token_axis = layout.token_sites
    chain_axis = tuple(layout.system.instances)

    # Per-token pLDDT (0--100) and resolved probability aligned to the token axis.
    plddt = output.plddt[0, :, :token_count].to(device="cpu", dtype=torch.float32)
    resolved_probability = output.resolved_probability[0, :, :token_count].float().cpu()

    # Expected PAE/PDE in angstrom (half-bin centers over 0-32, matching _boltz2_ptm_scores).
    pae = output.pae[0, :, :token_count, :token_count].float().cpu()
    pde = output.pde[0, :, :token_count, :token_count].float().cpu()

    # pTM / ipTM metrics over the flattened sample axis (B == 1); reuse the estimator already used by
    # _fold/_score.
    ptm = output.ptm[0].float().cpu()
    iptm = output.iptm[0].float().cpu()
    ligand_iptm = output.ligand_iptm[0].float().cpu()
    protein_iptm = output.protein_iptm[0].float().cpu()
    chain_pair_iptm = output.pair_chains_iptm[0].float().cpu()

    # Interface confidence uses iPTM unless every iPTM value is zero, in which case it falls back to pTM.
    # The confidence score combines that interface term with complex pLDDT normalized back to 0-1.
    complex_plddt = output.complex_plddt[0].float().cpu()
    complex_iplddt = output.complex_iplddt[0].float().cpu()
    complex_pde = output.complex_pde[0].float().cpu()
    complex_ipde = output.complex_ipde[0].float().cpu()
    interface = ptm if torch.allclose(iptm, torch.zeros_like(iptm)) else iptm
    confidence_score = (4.0 * (complex_plddt / 100.0) + interface) / 5.0
    has_design = bool(input.design_mask[0].any())
    design_scores = {
        "design_to_target_iptm": output.design_to_target_iptm[0].float().cpu(),
        "design_residue_iptm": output.design_residue_iptm[0].float().cpu(),
        "design_iptm": output.design_iptm[0].float().cpu(),
        "design_iiptm": output.design_iiptm[0].float().cpu(),
        "target_ptm": output.target_ptm[0].float().cpu(),
        "design_ptm": output.design_ptm[0].float().cpu(),
        "interaction_pae": output.interaction_pae[0].float().cpu(),
        "min_interaction_pae": output.min_interaction_pae[0].float().cpu(),
        "min_design_to_target_pae": output.min_design_to_target_pae[0].float().cpu(),
        "design_ipsae_min": output.design_ipsae_min[0].float().cpu(),
        "design_to_target_ipsae": output.design_to_target_ipsae[0].float().cpu(),
        "target_to_design_ipsae": output.target_to_design_ipsae[0].float().cpu(),
    }

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
        scores = {
            "plddt": Score(float(complex_plddt[sample_index]), preferred="higher"),
            "ptm": Score(float(ptm[sample_index]), preferred="higher"),
            "iptm": Score(float(iptm[sample_index]), preferred="higher"),
            "ligand_iptm": Score(float(ligand_iptm[sample_index]), preferred="higher"),
            "protein_iptm": Score(float(protein_iptm[sample_index]), preferred="higher"),
            "confidence_score": Score(float(confidence_score[sample_index]), preferred="higher"),
            "complex_iplddt": Score(float(complex_iplddt[sample_index]), preferred="higher"),
            "complex_pde": Score(float(complex_pde[sample_index]), unit="angstrom", preferred="lower"),
            "complex_ipde": Score(float(complex_ipde[sample_index]), unit="angstrom", preferred="lower"),
        }
        if has_design:
            for name, values in design_scores.items():
                if name in {
                    "interaction_pae",
                    "min_interaction_pae",
                    "min_design_to_target_pae",
                }:
                    scores[name] = Score(
                        float(values[sample_index]),
                        unit="angstrom",
                        preferred="lower",
                    )
                else:
                    scores[name] = Score(float(values[sample_index]), preferred="higher")
        predictions.append(
            FoldPrediction(
                structure=structure,
                scores=scores,
                metrics={
                    "plddt": Metric(values=plddt[sample_index].numpy(), axis=token_axis),
                    "resolved_probability": Metric(values=resolved_probability[sample_index].numpy(), axis=token_axis),
                },
                pair_metrics={
                    "pae": PairMetric(
                        values=pae[sample_index].numpy(),
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                    "pde": PairMetric(
                        values=pde[sample_index].numpy(),
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                    "chain_pair_iptm": PairMetric(
                        values=chain_pair_iptm[sample_index].numpy(),
                        row_axis=chain_axis,
                        column_axis=chain_axis,
                    ),
                },
            )
        )
    return FoldResult(predictions=tuple(predictions))


def _boltz2_expected_angstrom(logits: Tensor) -> Tensor:
    r"""
    Softmax-expected error in angstrom over half-bin centers spanning ``0--32``.
    """

    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    centers = torch.arange(start=0.5 * bin_width, end=32.0, step=bin_width, device=logits.device, dtype=logits.dtype)
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)


def _boltz2_confidence_values(
    *,
    plddt_logits: Tensor,
    pae_logits: Tensor,
    pde_logits: Tensor,
    resolved_logits: Tensor,
    coordinates: Tensor,
    distogram_logits: Tensor,
    input: Boltz2Input,
) -> dict[str, Tensor]:
    r"""
    Derive the released Boltz-2 confidence values while preserving ``(B, S)``.
    """

    batch_size, num_samples = coordinates.shape[:2]
    plddt_bins = plddt_logits.shape[-1]
    plddt_centers = (
        torch.arange(
            plddt_bins,
            device=plddt_logits.device,
            dtype=plddt_logits.dtype,
        )
        + 0.5
    ) / plddt_bins
    plddt = (plddt_logits.softmax(dim=-1) * plddt_centers).sum(dim=-1) * 100.0  # 0-100, shared with AF3/Protenix
    pae = _boltz2_expected_angstrom(pae_logits)
    pde = _boltz2_expected_angstrom(pde_logits)

    token_mask = input.token_mask[:, None].to(device=coordinates.device, dtype=coordinates.dtype)
    chain_index = input.chain_index[:, None].to(device=coordinates.device)
    molecule_type = input.molecule_type[:, None].to(device=coordinates.device)
    representative_index = input.token_to_rep_atom[:, None, :, None].expand(
        -1,
        num_samples,
        -1,
        3,
    )
    representative = coordinates.gather(2, representative_index)
    distances = torch.cdist(representative, representative)
    is_contact = (distances < 8).to(coordinates.dtype)
    different_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(coordinates.dtype)
    is_ligand = (molecule_type == 3).to(coordinates.dtype)
    interface_mask = (is_contact * different_chain * (1 - is_ligand).unsqueeze(-1)).amax(dim=-1)
    non_interface_mask = (1 - interface_mask) * (1 - is_ligand)
    interface_weight = 20 * is_ligand + 10 * interface_mask + non_interface_mask
    complex_plddt = (plddt * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1)
    complex_iplddt = (plddt * token_mask * interface_weight).sum(dim=-1) / (token_mask * interface_weight).sum(
        dim=-1
    ).clamp_min(1)

    distogram_probability = distogram_logits.softmax(dim=-1)
    contact_bin_count = min(20, distogram_probability.shape[-1])
    predicted_contact = distogram_probability[..., :contact_bin_count].sum(dim=-1)[:, None]
    token_pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    diagonal = torch.eye(
        token_pair_mask.shape[-1],
        device=token_pair_mask.device,
        dtype=token_pair_mask.dtype,
    )
    token_pair_mask = token_pair_mask * (1 - diagonal)
    contact_pair_mask = token_pair_mask * predicted_contact
    complex_pde = (pde * contact_pair_mask).sum(dim=(-2, -1)) / contact_pair_mask.sum(dim=(-2, -1))
    interface_pair_mask = contact_pair_mask * different_chain
    complex_ipde = (pde * interface_pair_mask).sum(dim=(-2, -1)) / interface_pair_mask.sum(dim=(-2, -1)).add(1e-5)

    design_mask = input.design_mask[:, None].to(device=coordinates.device, dtype=torch.bool)
    chain_design_mask = input.chain_design_mask[:, None].to(device=coordinates.device, dtype=torch.bool)
    target_mask = ~chain_design_mask
    chain_design_target_mask = token_pair_mask.bool() & (
        (chain_design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2))
    )
    min_interaction_pae = pae.masked_fill(~chain_design_target_mask, 100000.0).amin(dim=(-2, -1))
    design_target_mask = token_pair_mask.bool() & (
        (design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & design_mask.unsqueeze(-2))
    )
    interaction_pae = (pae * design_target_mask).sum(dim=(-2, -1)) / design_target_mask.sum(dim=(-2, -1)).add(1e-5)
    min_design_to_target_pae = pae.masked_fill(~design_target_mask, 100000.0).amin(dim=(-2, -1))
    has_design = design_mask.any(dim=-1)
    min_design_to_target_pae = torch.where(
        has_design,
        min_design_to_target_pae,
        min_design_to_target_pae.new_full(min_design_to_target_pae.shape, torch.nan),
    )

    ptm_scores = _boltz2_ptm_scores(
        pae_logits.flatten(0, 1),
        coordinates=coordinates.flatten(0, 1),
        input=input,
        multiplicity=num_samples,
    )
    return {
        "plddt": plddt,
        "pae": pae,
        "pde": pde,
        "resolved_probability": resolved_logits.softmax(dim=-1)[..., 0],
        "complex_plddt": complex_plddt,
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
        "pair_chains_iptm": ptm_scores["pair_chains_iptm"].reshape(
            batch_size,
            num_samples,
            *ptm_scores["pair_chains_iptm"].shape[-2:],
        ),
    }


def _boltz2_ptm_scores(
    pae_logits: Tensor,
    *,
    coordinates: Tensor,
    input: Boltz2Input,
    multiplicity: int,
) -> dict[str, Tensor]:
    r"""
    Derive the pTM and interface-pTM scores from PAE logits.
    """

    token_mask = input.token_mask.repeat_interleave(multiplicity, 0).float()
    frame_mask = _boltz2_predicted_frame_mask(
        coordinates,
        input=input,
        multiplicity=multiplicity,
    ).float()
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

    def aggregate(mask: Tensor) -> Tensor:
        return torch.max(
            (tm_expected * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-5),
            dim=1,
        ).values

    molecule_type = input.molecule_type.repeat_interleave(multiplicity, 0)
    is_ligand = (molecule_type == 3).float()
    is_protein = (molecule_type == 0).float()
    ligand_mask = pair_mask_iptm * (
        is_ligand[:, :, None] * is_protein[:, None, :] + is_protein[:, :, None] * is_ligand[:, None, :]
    )
    protein_mask = pair_mask_iptm * (is_protein[:, :, None] * is_protein[:, None, :])

    design_mask = input.design_mask.bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    chain_design_mask = torch.where(has_design, input.chain_design_mask.bool(), design_mask)
    target_mask = ~chain_design_mask
    design_mask = design_mask.repeat_interleave(multiplicity, 0)
    chain_design_mask = chain_design_mask.repeat_interleave(multiplicity, 0)
    target_mask = target_mask.repeat_interleave(multiplicity, 0)
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
    pair_chains_iptm = tm_expected.new_zeros(
        tm_expected.shape[0],
        num_chains,
        num_chains,
    )
    for chain_a in range(num_chains):
        for chain_b in range(num_chains):
            chain_pair_mask = (
                frame_mask[:, :, None]
                * (chain_index[:, None, :] == chain_a)
                * (chain_index[:, :, None] == chain_b)
                * token_mask[:, None, :]
                * token_mask[:, :, None]
            )
            pair_chains_iptm[:, chain_a, chain_b] = aggregate(chain_pair_mask)

    interaction_mask = _boltz2_design_interaction_mask(
        coordinates,
        input=input,
        multiplicity=multiplicity,
        chain_design_mask=chain_design_mask,
    )
    design_iiptm_mask = pair_mask_ptm * (
        (interaction_mask[:, :, None] & target_mask[:, None, :])
        | (target_mask[:, :, None] & interaction_mask[:, None, :])
    ).to(pae_logits)

    expected_pae = (pae_logits.softmax(dim=-1) * pae_value).sum(dim=-1)
    design_to_target_ipsae = _boltz2_ipsae_score(
        chain_design_mask,
        target_mask,
        expected_pae,
        frame_mask,
        token_mask,
    )
    target_to_design_ipsae = _boltz2_ipsae_score(
        target_mask,
        chain_design_mask,
        expected_pae,
        frame_mask,
        token_mask,
    )

    return {
        "ptm": aggregate(pair_mask_ptm),
        "iptm": aggregate(pair_mask_iptm),
        "ligand_iptm": aggregate(ligand_mask),
        "protein_iptm": aggregate(protein_mask),
        "design_to_target_iptm": aggregate(design_to_target_iptm_mask),
        "design_residue_iptm": aggregate(design_residue_iptm_mask),
        "design_iptm": aggregate(design_iptm_mask),
        "design_iiptm": aggregate(design_iiptm_mask),
        "target_ptm": aggregate(target_ptm_mask),
        "design_ptm": aggregate(design_ptm_mask),
        "design_ipsae_min": torch.minimum(design_to_target_ipsae, target_to_design_ipsae),
        "design_to_target_ipsae": design_to_target_ipsae,
        "target_to_design_ipsae": target_to_design_ipsae,
        "pair_chains_iptm": pair_chains_iptm,
    }


def _boltz2_design_interaction_mask(
    coordinates: Tensor,
    *,
    input: Boltz2Input,
    multiplicity: int,
    chain_design_mask: Tensor,
) -> Tensor:
    r"""
    Select design tokens with at least one atom within 8 Å of any target atom.
    """

    atom_to_token = input.atom_to_token.repeat_interleave(multiplicity, 0).long()
    atom_mask = input.atom_mask.repeat_interleave(multiplicity, 0).bool()
    atom_chain_design_mask = chain_design_mask.gather(1, atom_to_token) & atom_mask
    interaction_mask = chain_design_mask.new_zeros(chain_design_mask.shape)
    for sample_index in range(coordinates.shape[0]):
        design_atoms = atom_chain_design_mask[sample_index]
        target_atoms = atom_mask[sample_index] & ~design_atoms
        if design_atoms.any() and target_atoms.any():
            interacting_atoms = (
                torch.cdist(
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


def _boltz2_ipsae_score(
    source_mask: Tensor,
    target_mask: Tensor,
    pae: Tensor,
    frame_mask: Tensor,
    token_mask: Tensor,
    cutoff: float = 15.0,
) -> Tensor:
    r"""
    Compute the directional iPSAE score.
    """

    pair_mask = (
        frame_mask.unsqueeze(-1)
        * token_mask.unsqueeze(-2)
        * token_mask.unsqueeze(-1)
        * source_mask.unsqueeze(-1).to(pae)
        * target_mask.unsqueeze(-2).to(pae)
    )
    valid_pae = (pae < cutoff).to(pae) * pair_mask
    num_valid = valid_pae.sum(dim=-1)
    d0 = 1.24 * (num_valid.clamp(min=19) - 15).pow(1.0 / 3.0) - 1.8
    d0 = d0.clamp(min=1.0)
    terms = 1.0 / (1.0 + (pae / d0.unsqueeze(-1)).square())
    per_source = (terms * valid_pae).sum(dim=-1) / (num_valid + 1e-5)
    return (per_source * source_mask.to(per_source)).max(dim=-1).values


def _boltz2_predicted_frame_mask(
    coordinates: Tensor,
    *,
    input: Boltz2Input,
    multiplicity: int,
) -> Tensor:
    r"""
    Return the exact predicted-frame validity mask used by Boltz-2 pTM.
    """

    if input.frame_atom_index is None:
        raise ValueError("Boltz-2 confidence requires token frame atom indices")
    frame_index = (
        input.frame_atom_index.repeat_interleave(multiplicity, 0)
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
            for sample_index in range(multiplicity):
                row = batch_index * multiplicity + sample_index
                distances = torch.cdist(coordinates[row, chain_atoms], coordinates[row, chain_atoms])
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
    return valid_geometry & token_mask.repeat_interleave(multiplicity, 0)


def _boltz2_potential_terms(
    coordinates: Tensor,
    input: Boltz2Input,
    *,
    multiplicity: int,
    t: float,
    physical_guidance: bool,
    contact_guidance: bool,
    include_contact: bool = True,
    active_physical: frozenset[int] | None = None,
) -> tuple[tuple[Tensor, float, float, int], ...]:
    r"""
    Evaluate the released Boltz-2 physical, contact, and template energies.
    """

    batch_size = input.atom_mask.shape[0]
    coordinates = coordinates.reshape(
        batch_size,
        multiplicity,
        *coordinates.shape[1:],
    )
    term_rows = [
        _boltz2_batch_potential_terms(
            coordinates[batch_index],
            input,
            batch_index,
            t=t,
            include_contact=include_contact,
            active_physical=active_physical,
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
    input: Boltz2Input,
    *,
    multiplicity: int,
    t: float,
    guidance_step: int,
    physical_guidance: bool,
    contact_guidance: bool,
) -> Tensor:
    r"""
    Evaluate the released scheduled Boltz-2 guidance vector field.

    Boltz guidance is not, in three places, the mathematical autograd gradient
    of its reported energy. The COM field is copied from each chain center to
    every atom, contact unions retain the released softmin derivative, and
    template token fields are copied to every atom in that token.
    """

    gradient = torch.zeros_like(coordinates)
    if physical_guidance:
        if guidance_step % 4 == 0:
            gradient = gradient + 0.5 * _boltz2_symmetric_chain_com_gradient(
                coordinates,
                input,
                multiplicity=multiplicity,
                t=t,
            )
        gradient = gradient + _boltz2_local_physical_gradient(
            coordinates,
            input,
            multiplicity=multiplicity,
            t=t,
            guidance_step=guidance_step,
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
                    input,
                    multiplicity=multiplicity,
                    t=t,
                )
        if guidance_step % 2 == 0:
            gradient = gradient + 0.1 * _boltz2_template_gradient(
                coordinates,
                input,
                multiplicity=multiplicity,
            )
    return gradient


def _boltz2_local_physical_gradient(
    coordinates: Tensor,
    input: Boltz2Input,
    *,
    multiplicity: int,
    t: float,
    guidance_step: int,
) -> Tensor:
    batch_size = input.atom_mask.shape[0]
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        multiplicity,
        *coordinates.shape[1:],
    )
    rows = []
    for batch_index in range(batch_size):
        row_coordinates = coordinates_by_batch[batch_index]
        device, dtype = row_coordinates.device, row_coordinates.dtype
        row_gradient = torch.zeros_like(row_coordinates)
        atom_mask = input.atom_mask[batch_index].to(
            device=device,
            dtype=torch.bool,
        )
        atom_to_token = input.atom_to_token[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        atom_chain = (
            input.chain_index[batch_index]
            .to(
                device=device,
                dtype=torch.long,
            )
            .gather(0, atom_to_token)
        )
        radii = _boltz2_vdw_radii(device=device, dtype=dtype)
        element = input.atom_features[
            batch_index,
            :,
            4:132,
        ].to(device=device, dtype=dtype)
        atom_radii = element @ radii

        if t <= 0.4 and guidance_step % 5 == 0:
            pair_index = torch.triu_indices(
                row_coordinates.shape[-2],
                row_coordinates.shape[-2],
                1,
                device=device,
            )
            pair_mask = atom_mask[pair_index].all(dim=0)
            chain_size = torch.bincount(
                atom_chain[atom_mask],
                minlength=int(atom_chain.max().item()) + 1,
            )
            pair_ion_mask = (chain_size > 1)[atom_chain[pair_index]].all(dim=0)
            connected_chain = input.connected_chain_index[batch_index].to(
                device=device,
                dtype=torch.long,
            )
            connected_matrix = torch.eye(
                chain_size.shape[0],
                device=device,
                dtype=torch.bool,
            )
            if connected_chain.shape[1]:
                connected_matrix[
                    connected_chain[0],
                    connected_chain[1],
                ] = True
                connected_matrix[
                    connected_chain[1],
                    connected_chain[0],
                ] = True
            connected_pair = connected_matrix[
                atom_chain[pair_index[0]],
                atom_chain[pair_index[1]],
            ]
            pair_index = pair_index[
                :,
                pair_mask & pair_ion_mask & ~connected_pair,
            ]
            if pair_index.shape[1]:
                distance, distance_gradient = indexed_distance_with_gradient(
                    row_coordinates,
                    pair_index,
                )
                lower = atom_radii[pair_index].sum(dim=0) * 0.775
                slope = linear_flat_bottom_derivative(
                    distance,
                    distance.new_tensor(0.125),
                    lower,
                    distance.new_tensor(float("inf")),
                )
                row_gradient = row_gradient + (
                    _boltz2_scatter_potential_gradient(
                        row_coordinates,
                        pair_index,
                        distance_gradient,
                        slope,
                    )
                )

        connected_atom = input.connected_atom_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        if connected_atom.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                connected_atom,
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
                    connected_atom,
                    distance_gradient,
                    slope,
                )
            )

        geometry_index = input.geometry_pair_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        if geometry_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                geometry_index,
            )
            lower = (
                input.geometry_lower_bounds[batch_index]
                .to(
                    device=device,
                    dtype=dtype,
                )
                .clone()
            )
            upper = (
                input.geometry_upper_bounds[batch_index]
                .to(
                    device=device,
                    dtype=dtype,
                )
                .clone()
            )
            bond = input.geometry_bond_mask[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            angle = input.geometry_angle_mask[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            lower[bond & ~angle] *= 0.875
            upper[bond & ~angle] *= 1.125
            lower[~bond & angle] *= 0.875
            upper[~bond & angle] *= 1.125
            lower[bond & angle] *= 0.875
            upper[bond & angle] *= 1.125
            lower[~bond & ~angle] *= 0.9
            upper[~bond & ~angle] = float("inf")
            cutoff = 0.35 + atom_radii[geometry_index].mean(dim=0)
            lower[~bond] = torch.maximum(
                lower[~bond],
                cutoff[~bond],
            )
            upper[bond] = torch.minimum(
                upper[bond],
                cutoff[bond],
            )
            slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(0.01),
                lower,
                upper,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    geometry_index,
                    distance_gradient,
                    slope,
                )
            )

        chiral_index = input.chiral_atom_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        if chiral_index.shape[1]:
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates,
                chiral_index,
            )
            is_r = input.chiral_atom_is_r[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            lower = torch.where(
                is_r,
                dihedral.new_tensor(0.52360),
                dihedral.new_tensor(float("-inf")),
            )
            upper = torch.where(
                is_r,
                dihedral.new_tensor(float("inf")),
                dihedral.new_tensor(-0.52360),
            )
            slope = linear_flat_bottom_derivative(
                dihedral,
                dihedral.new_tensor(0.1),
                lower,
                upper,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    chiral_index,
                    dihedral_gradient,
                    slope,
                )
            )

        stereo_index = input.stereo_bond_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        if stereo_index.shape[1]:
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates,
                stereo_index,
            )
            absolute = dihedral.abs()
            absolute_gradient = torch.where(
                (dihedral < 0)[:, None, :, None],
                -dihedral_gradient,
                dihedral_gradient,
            )
            is_e = input.stereo_bond_is_e[batch_index].to(
                device=device,
                dtype=torch.bool,
            )
            lower = torch.where(
                is_e,
                absolute.new_tensor(torch.pi - 0.52360),
                absolute.new_tensor(float("-inf")),
            )
            upper = torch.where(
                is_e,
                absolute.new_tensor(float("inf")),
                absolute.new_tensor(0.52360),
            )
            slope = linear_flat_bottom_derivative(
                absolute,
                absolute.new_tensor(0.05),
                lower,
                upper,
            )
            row_gradient = row_gradient + (
                _boltz2_scatter_potential_gradient(
                    row_coordinates,
                    stereo_index,
                    absolute_gradient,
                    slope,
                )
            )

        planar_index = input.planar_bond_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        if planar_index.shape[1]:
            improper = torch.cat(
                (
                    planar_index[[1, 2, 3, 0]],
                    planar_index[[4, 5, 0, 3]],
                ),
                dim=1,
            )
            dihedral, dihedral_gradient = indexed_signed_dihedral_with_gradient(
                row_coordinates,
                improper,
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
                    improper,
                    absolute_gradient,
                    slope,
                )
            )
        rows.append(row_gradient)
    return torch.cat(rows)


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
    input: Boltz2Input,
    *,
    multiplicity: int,
    t: float,
) -> Tensor:
    batch_size = input.atom_mask.shape[0]
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        multiplicity,
        *coordinates.shape[1:],
    )
    rows = []
    buffer = 1.0 + 4.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
    for batch_index in range(batch_size):
        row_coordinates = coordinates_by_batch[batch_index]
        atom_mask = input.atom_mask[batch_index].to(
            device=coordinates.device,
            dtype=torch.bool,
        )
        atom_to_token = input.atom_to_token[batch_index].to(
            device=coordinates.device,
            dtype=torch.long,
        )
        atom_chain = (
            input.chain_index[batch_index]
            .to(
                device=coordinates.device,
                dtype=torch.long,
            )
            .gather(0, atom_to_token)
        )
        pair_index = input.symmetric_chain_index[batch_index].to(
            device=coordinates.device,
            dtype=torch.long,
        )
        row_gradient = torch.zeros_like(row_coordinates)
        if pair_index.shape[1] and bool(atom_mask.any()):
            num_chains = int(atom_chain.max().item()) + 1
            chain_sizes = torch.bincount(
                atom_chain[atom_mask],
                minlength=num_chains,
            )
            pair_index = pair_index[
                :,
                (chain_sizes[pair_index] > 1).all(dim=0),
            ]
            if pair_index.shape[1]:
                unpadded_chain = atom_chain[atom_mask]
                unpadded_coordinates = row_coordinates[:, atom_mask]
                centers = torch.zeros(
                    multiplicity,
                    num_chains,
                    3,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                ).scatter_reduce(
                    -2,
                    unpadded_chain[None, :, None].expand(
                        multiplicity,
                        -1,
                        3,
                    ),
                    unpadded_coordinates,
                    "mean",
                )
                distance, distance_gradient = indexed_distance_with_gradient(
                    centers,
                    pair_index,
                )
                slope = linear_flat_bottom_derivative(
                    distance,
                    distance.new_tensor(1.0),
                    distance.new_tensor(buffer),
                    distance.new_tensor(float("inf")),
                )
                center_gradient = _boltz2_scatter_potential_gradient(
                    centers,
                    pair_index,
                    distance_gradient,
                    slope,
                )
                row_gradient = center_gradient[:, atom_chain]
        rows.append(row_gradient)
    return torch.cat(rows)


def _boltz2_contact_gradient(
    coordinates: Tensor,
    input: Boltz2Input,
    *,
    multiplicity: int,
    t: float,
) -> Tensor:
    batch_size = input.atom_mask.shape[0]
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        multiplicity,
        *coordinates.shape[1:],
    )
    rows = []
    union_lambda = 8.0 - 8.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
    for batch_index in range(batch_size):
        row_coordinates = coordinates_by_batch[batch_index]
        pair_index = input.contact_pair_index[batch_index].to(
            device=coordinates.device,
            dtype=torch.long,
        )
        row_gradient = torch.zeros_like(row_coordinates)
        if pair_index.shape[1]:
            distance, distance_gradient = indexed_distance_with_gradient(
                row_coordinates,
                pair_index,
            )
            threshold = input.contact_thresholds[batch_index].to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            energy = linear_flat_bottom_energy(
                distance,
                distance.new_tensor(1.0),
                distance.new_tensor(float("-inf")),
                threshold,
            )
            energy_slope = linear_flat_bottom_derivative(
                distance,
                distance.new_tensor(1.0),
                distance.new_tensor(float("-inf")),
                threshold,
            )
            union_index = input.contact_union_index[batch_index].to(
                device=coordinates.device,
                dtype=torch.long,
            )
            coefficient = torch.zeros_like(energy)
            for union in range(int(union_index.max().item()) + 1):
                union_mask = union_index == union
                union_energy = energy[:, union_mask]
                negative_exponential = torch.exp(
                    -union_lambda * union_energy,
                )
                normalization = negative_exponential.sum(
                    dim=-1,
                    keepdim=True,
                )
                softmax = negative_exponential / normalization
                softmax = torch.where(
                    normalization == 0,
                    torch.zeros_like(softmax),
                    softmax,
                )
                union_mean = (union_energy * softmax).sum(
                    dim=-1,
                    keepdim=True,
                )
                coefficient[:, union_mask] = (
                    energy_slope[:, union_mask] * softmax * (1 + union_lambda * (union_energy - union_mean))
                )
            row_gradient = _boltz2_scatter_potential_gradient(
                row_coordinates,
                pair_index,
                distance_gradient,
                coefficient,
            )
        rows.append(row_gradient)
    return torch.cat(rows)


def _boltz2_template_gradient(
    coordinates: Tensor,
    input: Boltz2Input,
    *,
    multiplicity: int,
) -> Tensor:
    batch_size = input.atom_mask.shape[0]
    coordinates_by_batch = coordinates.reshape(
        batch_size,
        multiplicity,
        *coordinates.shape[1:],
    )
    rows = []
    for batch_index in range(batch_size):
        row_coordinates = coordinates_by_batch[batch_index]
        row_gradient = torch.zeros_like(row_coordinates)
        # An all-False row selects zero templates below, so the template loop runs zero times and the row
        # gradient stays zero; drop the per-row `.any()` guard, which forced a host sync on every step.
        if input.template_force is not None:
            force = input.template_force[batch_index].to(
                device=coordinates.device,
                dtype=torch.bool,
            )
            reference = cast(
                Tensor,
                input.template_beta_carbon_positions,
            )[batch_index, force].to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            reference_mask = cast(
                Tensor,
                input.template_beta_carbon_mask,
            )[batch_index, force].to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            threshold = cast(
                Tensor,
                input.template_force_threshold,
            )[batch_index, force].to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            representative_index = input.token_to_rep_atom[batch_index].to(
                device=coordinates.device,
                dtype=torch.long,
            )
            representative = row_coordinates[:, representative_index]
            token_gradient = torch.zeros_like(representative)
            for template_index in range(reference.shape[0]):
                repeated_reference = reference[template_index].expand(
                    multiplicity,
                    -1,
                    -1,
                )
                repeated_mask = reference_mask[template_index].expand(
                    multiplicity,
                    -1,
                )
                aligned = (
                    weighted_rigid_align(
                        source=repeated_reference.float(),
                        target=representative.float(),
                        weight=repeated_mask.float(),
                    )
                    .to(dtype=coordinates.dtype)
                    .detach()
                )
                delta = representative - aligned
                distance = torch.linalg.vector_norm(delta, dim=-1)
                direction = delta / distance.unsqueeze(-1)
                slope = linear_flat_bottom_derivative(
                    distance,
                    distance.new_tensor(1.0),
                    distance.new_tensor(float("-inf")),
                    threshold[template_index],
                )
                slope = slope * repeated_mask
                token_gradient = token_gradient + slope.unsqueeze(-1) * direction
            atom_to_token = input.atom_to_token[batch_index].to(
                device=coordinates.device,
                dtype=torch.long,
            )
            row_gradient = token_gradient[:, atom_to_token]
        rows.append(row_gradient)
    return torch.cat(rows)


def _boltz2_batch_potential_terms(
    coordinates: Tensor,
    input: Boltz2Input,
    batch_index: int,
    *,
    t: float,
    include_contact: bool,
    active_physical: frozenset[int] | None,
) -> tuple[Tensor, ...]:
    r"""
    Evaluate each potential for one batch row and every sample in that row.
    """

    device, dtype = coordinates.device, coordinates.dtype
    atom_mask = input.atom_mask[batch_index].to(device=device, dtype=torch.bool)
    atom_to_token = input.atom_to_token[batch_index].to(device=device, dtype=torch.long)
    atom_chain = (
        input.chain_index[batch_index]
        .to(device=device)
        .gather(
            0,
            atom_to_token,
        )
    )
    atom_count = coordinates.shape[-2]
    zero = coordinates.new_zeros(coordinates.shape[0])
    active_physical = frozenset(range(7)) if active_physical is None else active_physical
    if 1 in active_physical or 3 in active_physical:
        radii = _boltz2_vdw_radii(device=device, dtype=dtype)
        element = input.atom_features[
            batch_index,
            :,
            4:132,
        ].to(device=device, dtype=dtype)
        atom_radii = element @ radii
    else:
        atom_radii = coordinates.new_empty(0)

    symmetric_index = input.symmetric_chain_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 0 in active_physical and symmetric_index.shape[1]:
        num_chains = int(input.chain_index[batch_index].max().item()) + 1
        unpadded_chain = atom_chain[atom_mask]
        unpadded_coordinates = coordinates[:, atom_mask]
        centers = torch.zeros(
            coordinates.shape[0],
            num_chains,
            3,
            device=device,
            dtype=dtype,
        ).scatter_reduce(
            -2,
            unpadded_chain[None, :, None].expand(
                coordinates.shape[0],
                -1,
                3,
            ),
            unpadded_coordinates,
            "mean",
        )
        chain_size = torch.stack([(atom_mask & (atom_chain == chain)).sum() for chain in range(num_chains)])
        pair_mask = (chain_size[symmetric_index] > 1).all(dim=0)
        symmetric_distance = indexed_distance(centers, symmetric_index[:, pair_mask])
        symmetric_buffer = 1.0 + 4.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
        symmetric_energy = linear_flat_bottom_energy(
            symmetric_distance,
            symmetric_distance.new_tensor(1.0),
            symmetric_distance.new_tensor(symmetric_buffer),
            symmetric_distance.new_tensor(float("inf")),
        ).sum(dim=-1)
    else:
        symmetric_energy = zero

    if 1 in active_physical:
        pair_index = torch.triu_indices(
            atom_count,
            atom_count,
            1,
            device=device,
        )
        pair_mask = atom_mask[pair_index].all(dim=0)
        chain_size = torch.bincount(
            atom_chain[atom_mask],
            minlength=int(atom_chain.max().item()) + 1,
        )
        pair_ion_mask = (chain_size > 1)[atom_chain[pair_index]].all(dim=0)
        connected_chain = input.connected_chain_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        connected_matrix = torch.eye(
            chain_size.shape[0],
            device=device,
            dtype=torch.bool,
        )
        if connected_chain.shape[1]:
            connected_matrix[
                connected_chain[0],
                connected_chain[1],
            ] = True
            connected_matrix[
                connected_chain[1],
                connected_chain[0],
            ] = True
        connected_pair = connected_matrix[
            atom_chain[pair_index[0]],
            atom_chain[pair_index[1]],
        ]
        pair_index = pair_index[
            :,
            pair_mask & pair_ion_mask & ~connected_pair,
        ]
        if pair_index.shape[1]:
            distance = indexed_distance(coordinates, pair_index)
            lower = atom_radii[pair_index].sum(dim=0) * 0.775
            vdw_energy = linear_flat_bottom_energy(
                distance,
                distance.new_tensor(1.0),
                lower,
                distance.new_tensor(float("inf")),
            ).sum(dim=-1)
        else:
            vdw_energy = zero
    else:
        vdw_energy = zero

    connected_atom = input.connected_atom_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 2 in active_physical and connected_atom.shape[1]:
        distance = indexed_distance(coordinates, connected_atom)
        connection_energy = linear_flat_bottom_energy(
            distance,
            distance.new_tensor(1.0),
            distance.new_tensor(float("-inf")),
            distance.new_tensor(2.0),
        ).sum(dim=-1)
    else:
        connection_energy = zero

    geometry_index = input.geometry_pair_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 3 in active_physical and geometry_index.shape[1]:
        distance = indexed_distance(coordinates, geometry_index)
        lower = input.geometry_lower_bounds[batch_index].to(device=device, dtype=dtype).clone()
        upper = input.geometry_upper_bounds[batch_index].to(device=device, dtype=dtype).clone()
        bond = input.geometry_bond_mask[batch_index].to(device=device, dtype=torch.bool)
        angle = input.geometry_angle_mask[batch_index].to(device=device, dtype=torch.bool)
        lower[bond & ~angle] *= 0.875
        upper[bond & ~angle] *= 1.125
        lower[~bond & angle] *= 0.875
        upper[~bond & angle] *= 1.125
        lower[bond & angle] *= 0.875
        upper[bond & angle] *= 1.125
        lower[~bond & ~angle] *= 0.9
        upper[~bond & ~angle] = float("inf")
        cutoff = 0.35 + atom_radii[geometry_index].mean(dim=0)
        lower[~bond] = torch.maximum(lower[~bond], cutoff[~bond])
        upper[bond] = torch.minimum(upper[bond], cutoff[bond])
        pose_energy = linear_flat_bottom_energy(
            distance,
            distance.new_tensor(1.0),
            lower,
            upper,
        ).sum(dim=-1)
    else:
        pose_energy = zero

    chiral_index = input.chiral_atom_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 4 in active_physical and chiral_index.shape[1]:
        angle = indexed_signed_dihedral(coordinates, chiral_index)
        is_r = input.chiral_atom_is_r[batch_index].to(device=device, dtype=torch.bool)
        lower = torch.where(
            is_r,
            angle.new_tensor(0.52360),
            angle.new_tensor(float("-inf")),
        )
        upper = torch.where(
            is_r,
            angle.new_tensor(float("inf")),
            angle.new_tensor(-0.52360),
        )
        chiral_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            lower,
            upper,
        ).sum(dim=-1)
    else:
        chiral_energy = zero

    stereo_index = input.stereo_bond_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 5 in active_physical and stereo_index.shape[1]:
        angle = indexed_signed_dihedral(coordinates, stereo_index).abs()
        is_e = input.stereo_bond_is_e[batch_index].to(device=device, dtype=torch.bool)
        lower = torch.where(
            is_e,
            angle.new_tensor(torch.pi - 0.52360),
            angle.new_tensor(float("-inf")),
        )
        upper = torch.where(
            is_e,
            angle.new_tensor(float("inf")),
            angle.new_tensor(0.52360),
        )
        stereo_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            lower,
            upper,
        ).sum(dim=-1)
    else:
        stereo_energy = zero

    planar_index = input.planar_bond_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if 6 in active_physical and planar_index.shape[1]:
        improper = torch.cat(
            (
                planar_index[[1, 2, 3, 0]],
                planar_index[[4, 5, 0, 3]],
            ),
            dim=1,
        )
        angle = indexed_signed_dihedral(coordinates, improper).abs()
        planar_energy = linear_flat_bottom_energy(
            angle,
            angle.new_tensor(1.0),
            angle.new_tensor(float("-inf")),
            angle.new_tensor(0.26180),
        ).sum(dim=-1)
    else:
        planar_energy = zero

    forced_index = input.contact_pair_index[batch_index].to(
        device=device,
        dtype=torch.long,
    )
    if include_contact and forced_index.shape[1]:
        forced_distance = indexed_distance(coordinates, forced_index)
        forced_threshold = input.contact_thresholds[batch_index].to(
            device=device,
            dtype=dtype,
        )
        forced_energy = linear_flat_bottom_energy(
            forced_distance,
            forced_distance.new_tensor(1.0),
            forced_distance.new_tensor(float("-inf")),
            forced_threshold,
        )
        union_index = input.contact_union_index[batch_index].to(
            device=device,
            dtype=torch.long,
        )
        union_lambda = 8.0 - 8.0 * (math.exp(-2.0 * t) - 1) / (math.exp(-2.0) - 1)
        contact_energy = zero
        for union in range(int(union_index.max().item()) + 1):
            values = forced_energy[:, union_index == union]
            negative_exponential = torch.exp(-union_lambda * values)
            normalization = negative_exponential.sum(
                dim=-1,
                keepdim=True,
            )
            weights = negative_exponential / normalization
            weights = torch.where(
                normalization == 0,
                torch.zeros_like(weights),
                weights,
            )
            contact_energy = contact_energy + (values * weights).sum(dim=-1)
    else:
        contact_energy = zero

    template_energy = zero
    if include_contact and input.template_force is not None and bool(input.template_force[batch_index].any()):
        force = input.template_force[batch_index].to(device=device, dtype=torch.bool)
        reference = cast(Tensor, input.template_beta_carbon_positions)[batch_index, force].to(
            device=device,
            dtype=dtype,
        )
        reference_mask = cast(Tensor, input.template_beta_carbon_mask)[batch_index, force].to(
            device=device,
            dtype=dtype,
        )
        threshold = cast(Tensor, input.template_force_threshold)[batch_index, force].to(
            device=device,
            dtype=dtype,
        )
        representative = coordinates[:, input.token_to_rep_atom[batch_index].long()]
        for template_index in range(reference.shape[0]):
            repeated_reference = reference[template_index].expand(
                coordinates.shape[0],
                -1,
                -1,
            )
            repeated_mask = reference_mask[template_index].expand(
                coordinates.shape[0],
                -1,
            )
            aligned = (
                weighted_rigid_align(
                    source=repeated_reference.float(),
                    target=representative.float(),
                    weight=repeated_mask.float(),
                )
                .to(dtype=dtype)
                .detach()
            )
            distance = torch.linalg.vector_norm(representative - aligned, dim=-1)
            template_energy = template_energy + (
                linear_flat_bottom_energy(
                    distance,
                    distance.new_tensor(1.0),
                    distance.new_tensor(float("-inf")),
                    threshold[template_index],
                )
                * repeated_mask
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
    Gather complete-pool Boltz-2 conformers onto the current padded atom axis.
    """

    positions = target.coordinates
    if positions.ndim != 4:
        raise ValueError("Boltz2StructureTarget.coordinates must have shape (B, K, T, 3)")
    conformers = positions.shape[1]
    if conformers not in (1, num_samples):
        raise ValueError("Boltz-2 target conformer count must be one or num_samples")
    if conformers == 1:
        positions = positions.expand(-1, num_samples, -1, -1)

    atom_index = target.atom_index.long()
    safe_index = atom_index.clamp_min(0)
    gathered = positions.gather(
        2,
        safe_index[:, None, :, None].expand(-1, num_samples, -1, 3),
    )
    resolved = target.coordinate_mask.gather(1, safe_index)
    resolved = resolved & (atom_index >= 0) & input.atom_mask.bool()
    gathered = gathered.masked_fill(~resolved[:, None, :, None], 0)
    return gathered, resolved


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


# =====================================================================================================
# Boltz-2 public values and tensor boundary
#
# Family-owned scientific evidence precedes the pure-tensor input, output, and
# molecular-identity layout.
# =====================================================================================================


@dataclass(frozen=True)
class Boltz2Contact:
    r"""
    Learned Boltz-2 contact conditioning between two token-level endpoints.

    Polymer endpoints are residue occurrences and non-polymer endpoints are exact
    atom occurrences.

    Args:
        endpoint_a: First residue or non-polymer atom occurrence.
        endpoint_b: Second residue or non-polymer atom occurrence.
        max_distance: Contact threshold in angstroms.
        force: Whether the condition also contributes potential guidance.
    """

    endpoint_a: ComponentReference | AtomSite
    endpoint_b: ComponentReference | AtomSite
    max_distance: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class Boltz2Pocket:
    r"""
    Learned Boltz-2 pocket conditioning between a binder instance and contact sites.

    Args:
        binder: Molecular instance whose complete token set forms the binder side.
        contacts: Polymer residue or non-polymer atom occurrences on the pocket side.
        max_distance: Binder-to-pocket threshold in angstroms.
        force: Whether the pocket also contributes potential guidance.
    """

    binder: str
    contacts: tuple[ComponentReference | AtomSite, ...]
    max_distance: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class Boltz2Template:
    r"""
    Boltz-2 template evidence with optional potential-guidance force.

    Args:
        template: Canonical structural template evidence.
        force: Whether template distances also guide coordinate sampling.
        threshold: Flat-bottom template deviation threshold in angstroms.
            ``None`` applies force without a finite cutoff.
    """

    template: Template
    force: bool = False
    threshold: float | None = None


@dataclass(frozen=True)
class Boltz2Layout:
    r"""
    Molecular identities aligned to one Boltz-2 tensor layout.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    affinity_instance_id: str | None = None


@tensorclass
class Boltz2StructureTarget:
    r"""
    Complete pre-crop structure truth for Boltz-2.

    Args:
        coordinates: Complete-assembly conformers with shape
            ``(B, K, T, 3)``.
        coordinate_mask: Resolved complete-assembly mask with shape ``(B, T)``.
        atom_index: Current atom-to-pool map with shape ``(B, A)`` and
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
class Boltz2Input:
    r"""
    Input tensors consumed by [`Boltz2Model`][].

    Args:
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
        token_to_rep_atom: Distogram representative atom index for each token.
        residue_type_features: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level covalent adjacency.
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
        chiral_atom_is_r: Handedness flag for each chiral-center constraint.
        stereo_bond_index: Stereo-bond dihedral atom indices.
        stereo_bond_is_e: E/Z flag for each stereo-bond constraint.
        planar_bond_index: Planar double-bond atom indices.
        connected_chain_index: Chain pairs joined by assembly bonds.
        connected_atom_index: Atom pairs joined by assembly bonds.
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
            shape ``(B, T, N, 33)``.
        template_frame_rotations: Optional template frame rotations with shape
            ``(B, T, N, 3, 3)``.
        template_frame_translations: Optional template frame origins with shape
            ``(B, T, N, 3)``.
        template_frame_mask: Optional template frame validity mask with shape
            ``(B, T, N)``.
        template_beta_carbon_positions: Optional template CB/Gly-CA positions
            with shape ``(B, T, N, 3)``.
        template_alpha_carbon_positions: Optional template CA positions with
            shape ``(B, T, N, 3)``.
        template_beta_carbon_mask: Optional CB/Gly-CA validity mask with shape
            ``(B, T, N)``.
        template_mask: Optional alignment-coverage mask with shape ``(B, T, N)``.
        template_visibility_index: Optional per-chain template visibility groups
            with shape ``(B, T, N)``.
        template_force: Optional mask of templates used for potential guidance.
        template_force_threshold: Optional per-template guidance thresholds.
        affinity_token_mask: Optional mask selecting the ligand tokens to score for affinity.
        affinity_mw: Optional molecular weight for the selected ligand, used only when affinity MW correction is on.
        affinity_msa_profile: Optional single-sequence profile used by the affinity heads.
        affinity_deletion_mean: Optional deletion mean used by the affinity heads.
        layout: Molecular identities aligned to the atom and token tensors.
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
    token_to_rep_atom: Tensor
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
    chiral_atom_is_r: Tensor
    stereo_bond_index: Tensor
    stereo_bond_is_e: Tensor
    planar_bond_index: Tensor
    connected_chain_index: Tensor
    connected_atom_index: Tensor
    symmetric_chain_index: Tensor
    molecule_type: Tensor
    design_mask: Tensor
    chain_design_mask: Tensor
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
    template_beta_carbon_positions: Tensor | None = None
    template_alpha_carbon_positions: Tensor | None = None
    template_beta_carbon_mask: Tensor | None = None
    template_mask: Tensor | None = None
    template_visibility_index: Tensor | None = None
    template_force: Tensor | None = None
    template_force_threshold: Tensor | None = None
    affinity_token_mask: Tensor | None = None
    affinity_mw: Tensor | None = None
    affinity_msa_profile: Tensor | None = None
    affinity_deletion_mean: Tensor | None = None
    layout: Boltz2Layout | None = None


@tensorclass
class _Boltz2TrunkOutput:
    r"""
    Trunk outputs consumed by the diffusion / confidence heads.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor


@tensorclass
class Boltz2Representations:
    r"""
    Raw Boltz-2 trunk representations surfaced by ``Boltz2Model.representations``.

    Args:
        single: Trunk single representation `[B, N, single_size]`.
        pair: Trunk pair representation `[B, N, N, pair_size]`.
    """

    single: Tensor
    pair: Tensor


@tensorclass
class Boltz2Output:
    r"""
    Predictions from one Boltz-2 training denoising pass.

    Args:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        augmented_target_coordinates: Training target coordinates after rigid
            augmentation, with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
        bfactor_logits: Optional token-level B-factor logits with shape
            ``(B, N, D_bfactor)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor
    bfactor_logits: Tensor | None = None


@tensorclass
class Boltz2AffinityOutput:
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


@tensorclass
class Boltz2ConfidenceOutput:
    r"""
    Raw predictions from one Boltz-2 folding pass.

    Args:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        plddt_logits: Per-token confidence logits with shape
            ``(B, S, N, D_plddt)``.
        pae_logits: Token-pair aligned-error logits with shape
            ``(B, S, N, N, D_pae)``.
        pde_logits: Token-pair distance-error logits with shape
            ``(B, S, N, N, D_pde)``.
        resolved_logits: Per-token resolved-state logits with shape
            ``(B, S, N, D_resolved)``.
        bfactor_logits: Optional per-token B-factor logits with shape
            ``(B, N, D_bfactor)``.
        plddt: Expected pLDDT values on the ``0--100`` scale with shape
            ``(B, S, N)``.
        pae: Expected PAE in angstroms with shape ``(B, S, N, N)``.
        pde: Expected PDE in angstroms with shape ``(B, S, N, N)``.
        resolved_probability: Probability of resolved class zero with shape
            ``(B, S, N)``.
        complex_plddt: Mean complex pLDDT with shape ``(B, S)``.
        complex_iplddt: Interface-weighted pLDDT with shape ``(B, S)``.
        complex_pde: Distogram-contact-weighted PDE with shape ``(B, S)``.
        complex_ipde: Interface distogram-contact-weighted PDE with shape
            ``(B, S)``.
        ptm: Predicted TM score with shape ``(B, S)``.
        iptm: Interface predicted TM score with shape ``(B, S)``.
        ligand_iptm: Protein-ligand predicted TM score with shape ``(B, S)``.
        protein_iptm: Protein-protein predicted TM score with shape ``(B, S)``.
        pair_chains_iptm: Ordered chain-pair predicted TM scores with shape
            ``(B, S, C, C)``.
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
    bfactor_logits: Tensor | None
    plddt: Tensor
    pae: Tensor
    pde: Tensor
    resolved_probability: Tensor
    complex_plddt: Tensor
    complex_iplddt: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    ptm: Tensor
    iptm: Tensor
    ligand_iptm: Tensor
    protein_iptm: Tensor
    pair_chains_iptm: Tensor
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
