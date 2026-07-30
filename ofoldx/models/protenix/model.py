r"""
Protenix default, structure-conditioned, and constraint predictors.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast, overload

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn
from transformers import EsmConfig, EsmModel

from ...data.molecular import (
    HYDROGEN_ELEMENTS,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    LocalAtomGeometry,
    NormalizedKeyAtomToTokenEncoder,
)
from ...modules.backend import LayerNorm
from ...modules.backend.attention import projected_attention
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateEdmSampler,
    EdmScheduler,
)
from ...modules.embedding import (
    FactorizedPairLinear,
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation
from ...modules.head import AtomConfidenceProjector, DistogramHead, expected_plddt, tm_bin_weights
from ...modules.head.frames import gather_atoms_by_index, insert_sample_axis
from ...modules.transition import Transition
from ...modules.trunk import MsaTrunk, PairformerStack
from ...modules.trunk.msa import TrianglePairUpdateBlock
from ...results import (
    FoldPrediction,
    FoldResult,
    Metric,
    PairMetric,
    Score,
)
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixStructureConditionConfig,
    ProtenixTemplateModelConfig,
)

#: Sentinel for an omitted ``tfg_guidance`` argument, distinct from an explicit ``None``: omission inherits
#: the guidance a processor-prepared input carries, while ``None`` disables guidance (the documented meaning).
_INHERIT_TFG_GUIDANCE: Any = object()


class ProtenixModel(Model):
    r"""
    Predict biomolecular complex structures with Protenix.

    ``forward`` accepts [`ProtenixInput`][] and [`ProtenixStructureTarget`][]
    for stochastic training denoising. ``denoise`` accepts explicit noisy
    coordinates, while ``sample`` runs reverse diffusion.

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import PolymerSequence
        >>> from ofoldx.models.protenix import (
        ...     ProtenixConfig,
        ...     ProtenixModel,
        ...     ProtenixProcessor,
        ... )
        >>> config = ProtenixConfig({
        ...     "single_size": 8,
        ...     "pair_size": 8,
        ...     "trunk": {
        ...         "msa_size": 8,
        ...         "num_msa_blocks": 1,
        ...         "msa_outer_product_hidden_size": 4,
        ...         "num_msa_attention_heads": 1,
        ...         "msa_triangle_multiplication_hidden_size": 4,
        ...         "msa_triangle_attention_hidden_size": 4,
        ...         "num_msa_triangle_attention_heads": 1,
        ...         "msa_expansion": 2,
        ...         "num_pairformer_blocks": 1,
        ...         "pairformer_triangle_multiplication_hidden_size": 4,
        ...         "pairformer_triangle_attention_hidden_size": 4,
        ...         "num_pairformer_triangle_attention_heads": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "pairformer_expansion": 2,
        ...         "pairformer_dropout": 0.0,
        ...         "msa_dropout": 0.0,
        ...         "pair_dropout": 0.0,
        ...     },
        ...     "recycling_steps": 1,
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 8,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_encoder_attention_heads": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...         "num_atom_decoder_attention_heads": 1,
        ...         "num_query_atoms": 4,
        ...         "num_key_atoms": 4,
        ...     },
        ...     "distogram": {"num_bins": 4},
        ...     "confidence": {
        ...         "num_plddt_bins": 4,
        ...         "num_pae_bins": 4,
        ...         "num_pde_bins": 4,
        ...         "max_atoms_per_token": 8,
        ...         "num_blocks": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "triangle_multiplication_hidden_size": 4,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ... })
        >>> inputs = ProtenixProcessor()(PolymerSequence.from_one_letter("A"))
        >>> atom_count = inputs.atom_features.shape[1]
        >>> model = ProtenixModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.denoise(
        ...         inputs,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...         recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
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
            raise ValueError(
                "this Protenix architecture has no template graph and its Processor produces no template "
                "features; use protenix_template for the template architecture"
            )
        if config.constraints and not self._supports_constraints:
            raise ValueError(
                "this Protenix architecture has no constraint graph and its Processor exposes no constraint "
                "inputs; use protenix_constraint for the constraint architecture"
            )
        self._recycling_steps = config.recycling_steps
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
            self.esm_projection = nn.Linear(esm.hidden_size, config.token_feature_size, bias=False)
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
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
                triangle_attention_hidden_size=template.triangle_attention_hidden_size,
                num_triangle_attention_heads=template.num_triangle_attention_heads,
                expansion=template.expansion,
                dropout=template.dropout,
            )
        )
        self.msa_trunk = MsaTrunk(
            msa_size=config.trunk.msa_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            num_stages=config.trunk.num_msa_blocks,
            outer_product_hidden_size=config.trunk.msa_outer_product_hidden_size,
            num_msa_attention_heads=config.trunk.num_msa_attention_heads,
            triangle_multiplication_hidden_size=config.trunk.msa_triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=config.trunk.msa_triangle_attention_hidden_size,
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
            triangle_attention_hidden_size=config.trunk.pairformer_triangle_attention_hidden_size,
            num_triangle_attention_heads=config.trunk.num_pairformer_triangle_attention_heads,
            num_pair_attention_heads=config.trunk.num_pair_attention_heads,
            expansion=config.trunk.pairformer_expansion,
            dropout=config.trunk.pairformer_dropout,
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
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.coordinate_sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(
                max_noise_level=config.sampler.max_noise_level,
                min_noise_level=config.sampler.min_noise_level,
                rho=config.sampler.rho,
                sigma_data=config.denoiser.sigma_data,
            ),
            solver=cast(Literal["euler", "heun"], config.sampler.solver),
            churn_rate=config.sampler.churn_rate,
            min_churn_noise_level=config.sampler.min_churn_noise_level,
            noise_scale=config.sampler.noise_scale,
            step_scale=config.sampler.step_scale,
            center_random_augmentation=config.sampler.center_random_augmentation,
        )
        self._num_sampling_steps = config.sampler.num_sampling_steps
        self.confidence_head = ProtenixConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            distance_bin_start=config.confidence.distance_bin_start,
            distance_bin_end=config.confidence.distance_bin_end,
            distance_bin_step=config.confidence.distance_bin_step,
            num_plddt_bins=config.confidence.num_plddt_bins,
            num_resolved_bins=config.confidence.num_resolved_bins,
            max_atoms_per_token=config.confidence.max_atoms_per_token,
            num_pae_bins=config.confidence.num_pae_bins,
            num_pde_bins=config.confidence.num_pde_bins,
            num_blocks=config.confidence.num_blocks,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=config.confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            expansion=config.confidence.expansion,
            dropout=config.confidence.pairformer_dropout,
            single_max_abs=config.confidence.single_max_abs,
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
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        use_conditioning: bool = True,
        generator: torch.Generator | None = None,
    ) -> ProtenixOutput:
        r"""
        Denoise randomly corrupted target coordinates for diffusion training.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            target: Heavy-atom truth-pool coordinates and their current atom-axis
                mapping.
            num_samples: Number ``S`` of independently corrupted copies per input.
            synchronize_noise_levels: Whether samples from one example share sigma.
            coordinate_augmentation: Whether to apply random rigid augmentation.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.
            generator: Optional generator for rigid augmentation, noise levels,
                and Gaussian coordinate noise.

        Returns:
            Denoised and augmented target coordinates with shape ``(B, S, A, 3)``,
            noise levels with shape ``(B, S)``, one distogram, and the four
            confidence logits.

        Raises:
            ValueError: If ``num_samples`` or ``recycling_steps`` is not positive.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
            reference_positions=input.reference_positions,
        )
        target_coordinates, target_mask = _gather_protenix_structure_target(target, input)
        target_coordinates = target_coordinates.to(representations.single)
        target_mask = target_mask.to(device=target_coordinates.device)
        batch_size, atom_count = target_coordinates.shape[:2]
        flat_target = target_coordinates.repeat_interleave(num_samples, dim=0)
        flat_target_mask = target_mask.repeat_interleave(num_samples, dim=0)
        augmented_target_coordinates = _center_random_augmentation(
            flat_target,
            flat_target_mask,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise_count = batch_size if synchronize_noise_levels else batch_size * num_samples
        normal = torch.empty(
            noise_count,
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
        ).normal_(generator=generator)
        noise_level = (
            self.coordinate_denoiser.preconditioner.sigma_data * (noise_log_mean + noise_log_std * normal).exp()
        )
        if synchronize_noise_levels:
            noise_level = noise_level.repeat_interleave(num_samples)
        coordinate_noise = torch.empty_like(augmented_target_coordinates).normal_(generator=generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[:, None, None] * coordinate_noise
        augmented_target_coordinates = augmented_target_coordinates.reshape(
            batch_size,
            num_samples,
            atom_count,
            3,
        )
        noise_level = noise_level.reshape(batch_size, num_samples)
        coordinates = self.coordinate_denoiser(
            self._denoise_context(
                input,
                representations,
                reference_positions=input.reference_positions,
                use_conditioning=use_conditioning,
            ).to_denoising_input(
                noisy_coords=noisy_coordinates.reshape(
                    batch_size,
                    num_samples,
                    atom_count,
                    3,
                ),
                noise_level=noise_level,
            )
        )
        raw_output = self._assemble_output(input, representations, coordinates)
        return ProtenixOutput(  # type: ignore[call-arg]
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
        recycling_steps: int | None = None,
        use_conditioning: bool = True,
    ) -> ProtenixRawOutput:
        r"""
        Denoise atom coordinates at explicitly supplied EDM noise levels.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            noisy_coordinates: Noisy coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: EDM noise standard deviations with shape ``(B, S)`` in
                angstroms.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.

        Returns:
            Raw denoised coordinates, distogram, and confidence logits.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.
        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
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
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = _INHERIT_TFG_GUIDANCE,
    ) -> ProtenixRawOutput:
        r"""
        Run reverse diffusion and return raw coordinates and prediction heads.

        Args:
            input: Processor output for one or more molecular assemblies.
            num_samples: Number of coordinate samples. ``None`` uses five.
            generator: Optional generator for MSA subsampling, initial coordinates,
                stochastic churn, and per-step rigid augmentations.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the configured value.
            step_scale: Per-call solver-step multiplier. ``None`` uses the configured value.
            tfg_guidance: Training-free chemical guidance. Omit to inherit the guidance the input was
                prepared with; pass ``None`` to disable it.

        Returns:
            Raw sampled coordinates, distogram, and confidence logits.

        Raises:
            ValueError: If a sampling or guidance iteration count is invalid,
                concrete guidance contains no potential term, or guidance is
                paired with input that has no prepared TFG geometry.
        """

        num_samples = 5 if num_samples is None else operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        if tfg_guidance is _INHERIT_TFG_GUIDANCE:
            tfg_guidance = input.tfg_guidance  # omitted -> inherit the input's guidance; explicit None disables
        tfg_potential = None
        if tfg_guidance is not None:
            if input.tfg_features is None:
                raise ValueError("tfg_guidance requires ProtenixInput.tfg_features prepared by the processor")
            # ``tfg_features`` is a plain (non-tensorclass) bundle whose unbatched index tensors the input's
            # ``.to()`` cannot reach, so align the potential's feature buffers to the coordinates here.
            # ``.to(reference_positions)`` follows ``nn.Module.to(Tensor)``: it matches the sampling device
            # and float dtype while leaving the integer index and bool mask buffers' dtypes untouched.
            tfg_potential = ProtenixTfgPotential(input.tfg_features, tfg_guidance).to(input.reference_positions)

        reference_positions = input.reference_positions
        msa_indices = tuple(
            self._sample_msa_indices(input.msa_tokens.shape[-2], input.msa_tokens.device, generator=generator)
            for _ in range(recycling_steps)
        )
        representations = self._run_trunk(
            input,
            recycling_steps=recycling_steps,
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

    @torch.no_grad()
    def fold(
        self,
        input: ProtenixInput,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = _INHERIT_TFG_GUIDANCE,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with Protenix.

        Args:
            input: Processor output for one assembly, with batch size ``B = 1`` and
                molecular identities aligned to its atom and token axes.
            num_samples: Number of coordinate samples. ``None`` uses five.
            generator: Optional generator for MSA subsampling, initial coordinates,
                stochastic churn, and per-step rigid augmentations.
            recycling_steps: Number of trunk recycling iterations. ``None`` uses
                the configured value.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses the
                configured value.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the configured value.
            step_scale: Per-call solver-step multiplier. ``None`` uses the configured value.
            tfg_guidance: Training-free chemical guidance. Omit to inherit the guidance the input was
                prepared with; pass ``None`` to disable guidance; ``ProtenixTfgGuidance()`` uses the
                released enabled defaults.

        Returns:
            One generation-ordered prediction per sample. Structure, PAE, PDE, and
            gPDE values are in angstroms; pLDDT is on the ``0--100`` scale. Atom
            metrics follow ``input.atom_sites`` and token-pair metrics follow
            ``input.token_sites``.

        Raises:
            ValueError: If a sampling or guidance iteration count is invalid,
                concrete guidance contains no potential term, or guidance is
                paired with input that has no prepared TFG geometry.
        """

        output = self.sample(
            input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
        num_samples = output.coordinates.shape[-3]
        reference_positions = input.reference_positions

        atom_count = len(input.atom_sites)
        token_count = len(input.token_sites)
        complete_sites = [site for site, atom in input.system.iter_atoms() if atom.element not in HYDROGEN_ELEMENTS]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray(
            [complete_index[site] for site in input.atom_sites],
            dtype=np.intp,
        )

        coordinates = output.coordinates[0, :, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        plddt_logits = output.plddt_logits[0, :, :atom_count].float()
        pae_logits = output.pae_logits[0, :, :token_count, :token_count].float()
        pde_logits = output.pde_logits[0, :, :token_count, :token_count].float()
        resolved_logits = output.resolved_logits[0, :, :atom_count].float()
        distogram_logits = output.distogram_logits[0, :token_count, :token_count].float()

        atom_plddt_unit = _protenix_bin_expectation(plddt_logits, 0.0, 1.0)
        atom_plddt = 100.0 * atom_plddt_unit
        pae, pae_probability = _protenix_bin_expectation(
            pae_logits,
            0.0,
            32.0,
            return_probability=True,
        )
        pde = _protenix_bin_expectation(pde_logits, 0.0, 32.0)
        resolved_probability = resolved_logits.softmax(dim=-1)[..., 1]
        contact_probability = _protenix_contact_probability(distogram_logits)
        token_has_frame = _protenix_token_has_frame(
            reference_positions,
            reference_mask=input.reference_mask,
            reference_space_id=input.reference_space_id,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_sites=input.atom_sites,
            token_sites=input.token_sites,
        )
        token_chain = input.chain_index[0, :token_count].to(device=pae_logits.device, dtype=torch.long)
        atom_to_token = input.atom_to_token[0, :atom_count].to(device=pae_logits.device, dtype=torch.long)
        atom_is_polymer = torch.tensor(
            [
                isinstance(
                    input.system.entities[input.system.instances[site.instance_id]],
                    PolymerSequence,
                )
                for site in input.atom_sites
            ],
            dtype=torch.bool,
            device=pae_logits.device,
        )
        ptm, iptm, chain_ptm, chain_iptm, chain_pair_iptm, chain_pair_iptm_global = _protenix_tm_scores(
            pae_probability,
            token_has_frame=token_has_frame,
            token_chain=token_chain,
            atom_to_token=atom_to_token,
            atom_is_polymer=atom_is_polymer,
        )
        gpde = (pde * contact_probability).sum(dim=(-1, -2)) / contact_probability.sum()
        chain_gpde, chain_pair_gpde = _protenix_chain_gpde(
            pde,
            contact_probability=contact_probability,
            token_chain=token_chain,
        )
        chain_plddt, chain_pair_plddt = _protenix_chain_plddt(
            atom_plddt_unit,
            token_chain=token_chain,
            atom_to_token=atom_to_token,
        )
        has_clash = _protenix_has_clash(
            output.coordinates[0, :, :atom_count].float(),
            atom_to_token=atom_to_token,
            token_chain=token_chain,
            atom_is_polymer=atom_is_polymer,
        )
        disorder = torch.zeros_like(ptm)
        ranking_score = 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100.0 * has_clash.to(ptm.dtype)

        contact_numpy = contact_probability.cpu().numpy()
        token_instance_ids = tuple(
            site.instance_id if isinstance(site, AtomSite) else site[0] for site in input.token_sites
        )
        token_chain_numpy = token_chain.cpu().numpy()
        chain_axis = tuple(
            token_instance_ids[int(np.argmax(token_chain_numpy == value))] for value in np.unique(token_chain_numpy)
        )
        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            complete_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
            coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
            complete_coordinates[model_to_complete] = coordinates[sample_index]
            coordinate_mask[model_to_complete] = True
            structure = Structure(
                system=input.system,
                sites=tuple(complete_sites),
                coordinates=complete_coordinates,
                coordinate_mask=coordinate_mask,
            )
            sample_plddt = atom_plddt[sample_index].cpu().numpy()
            sample_pae = pae[sample_index].cpu().numpy()
            sample_pde = pde[sample_index].cpu().numpy()
            sample_resolved_probability = resolved_probability[sample_index].cpu().numpy()
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "mean_plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
                        "gpde": Score(float(gpde[sample_index]), unit="angstrom", preferred="lower"),
                        "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                        "iptm": Score(float(iptm[sample_index]), preferred="higher"),
                        "has_clash": Score(float(has_clash[sample_index]), preferred="lower"),
                        "disorder": Score(float(disorder[sample_index]), preferred="higher"),
                        "ranking_score": Score(float(ranking_score[sample_index]), preferred="higher"),
                    },
                    metrics={
                        "plddt": Metric(values=sample_plddt, axis=input.atom_sites),
                        "resolved_probability": Metric(
                            values=sample_resolved_probability,
                            axis=input.atom_sites,
                        ),
                        "chain_plddt": Metric(
                            values=chain_plddt[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_ptm": Metric(
                            values=chain_ptm[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_iptm": Metric(
                            values=chain_iptm[sample_index].cpu().numpy(),
                            axis=chain_axis,
                        ),
                        "chain_gpde": Metric(
                            values=chain_gpde[sample_index].cpu().numpy(),
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=sample_pae,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=sample_pde,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_numpy,
                            row_axis=input.token_sites,
                            column_axis=input.token_sites,
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_pair_iptm[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_iptm_global": PairMetric(
                            values=chain_pair_iptm_global[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_gpde": PairMetric(
                            values=chain_pair_gpde[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_plddt": PairMetric(
                            values=chain_pair_plddt[sample_index].cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )

        return FoldResult(predictions=tuple(predictions))

    def _assemble_output(
        self,
        input: ProtenixInput,
        representations: _ProtenixTrunkOutput,
        coordinates: Tensor,
    ) -> ProtenixRawOutput:
        r"""
        Compute the distogram and confidence predictions.
        """

        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            representations.single,
            pair=representations.pair,
            token_embedding=representations.token_embedding,
            pred_coords=coordinates,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
        )
        return ProtenixRawOutput(  # type: ignore[call-arg]
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
            token_embedding=representations.token_embedding,
            single=representations.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=representations.relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                reference_positions,
                input.reference_space_id,
            ),
        )

    def _run_trunk(
        self,
        input: ProtenixInput,
        *,
        recycling_steps: int,
        reference_positions: Tensor,
        msa_indices: tuple[Tensor, ...] | None = None,
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
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embeddings.build_local_geometry(
                reference_positions,
                input.reference_space_id,
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
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            recycling_steps,
            msa_indices=msa_indices,
        )
        return _ProtenixTrunkOutput(  # type: ignore[call-arg]
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            batch_size=[token_embedding.shape[0]],
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
        recycling_steps: int,
        *,
        msa_indices: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run the recycling loop.
        """

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.template_embedder is not None:
                    pair = pair + self.template_embedder(
                        pair,
                        cast(ProtenixTemplateFeatures, input.template_features),
                        same_chain_mask=input.chain_index[..., :, None] == input.chain_index[..., None, :],
                        pair_mask=pair_mask,
                    )
                indices = None if msa_indices is None else msa_indices[cycle]
                _, pair = self.msa_trunk(
                    self._msa_features(input, indices),
                    pair,
                    token_embedding,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return single, pair

    @staticmethod
    def _msa_features(input: ProtenixInput, indices: Tensor | None) -> Tensor:
        r"""
        Build the 32+1+1 MSA feature vector for selected rows.
        """

        if indices is None:
            msa_tokens = input.msa_tokens
            has_deletion = input.has_deletion
            deletion_value = input.deletion_value
        else:
            msa_tokens = torch.index_select(input.msa_tokens, -2, indices)
            has_deletion = torch.index_select(input.has_deletion, -2, indices)
            deletion_value = torch.index_select(input.deletion_value, -2, indices)
        one_hot = F.one_hot(msa_tokens.long(), num_classes=32).to(dtype=input.token_features.dtype)
        return torch.cat(
            (
                one_hot,
                has_deletion.to(one_hot).unsqueeze(-1),
                deletion_value.to(one_hot).unsqueeze(-1),
            ),
            dim=-1,
        )

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
        token_bonds = torch.broadcast_to(token_bonds, (*single.shape[:-1], single.shape[-2]))
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
        pair_mask = pair_mask_from_token_mask(input.token_mask)
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
            self.coordinate_denoiser,
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

        reference = context.reference_positions
        noise_template = reference.unsqueeze(-3).expand(
            *context.batch_shape,
            num_samples,
            reference.shape[-2],
            3,
        )
        coordinates = noise_schedule[0] * torch.empty_like(noise_template).normal_(generator=generator)
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
                self.coordinate_sampler.noise_scale
                * delta_noise
                * torch.empty_like(coordinates).normal_(generator=generator)
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
            for outer_index in range(guidance.outer_steps):
                if guidance.rho == 0.0:
                    guided_coordinates = working_coordinates
                else:
                    with torch.enable_grad():
                        differentiable_coordinates = working_coordinates.detach().requires_grad_(True)
                        denoised_for_guidance = self.coordinate_denoiser(
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

                denoised_coordinates = self.coordinate_denoiser(
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
                for _ in range(guidance.inner_steps):
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
                if outer_index + 1 < guidance.outer_steps:
                    outer_noise = torch.empty_like(next_coordinates).normal_(generator=generator)
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

        count = int(
            torch.randint(
                1,
                num_rows + 1,
                (),
                device=device,
                generator=generator,
            )
        )
        count = min(count, 16384)
        return torch.randperm(num_rows, device=device, generator=generator)[:count]


class ProtenixStructureConditionModel(ProtenixModel):
    r"""
    Predict structures with the released Protenix mini structure condition.

    The structure update is a real part of this graph and is applied before the
    MSA path on every recycling pass.

    Args:
        config: Structure-conditioned model architecture and inference
            configuration.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from ofoldx.data import AtomSite, MolecularSystem, PolymerSequence, Structure
        >>> from ofoldx.models.protenix import (
        ...     ProtenixStructureConditionConfig,
        ...     ProtenixStructureConditionModel,
        ...     ProtenixStructureConditionProcessor,
        ... )
        >>> sequence = PolymerSequence.from_one_letter("G")
        >>> system = MolecularSystem({"protein": sequence}, {"A": "protein"})
        >>> sites = tuple(AtomSite("A", 0, atom) for atom in ("N", "CA", "C", "O"))
        >>> structure = Structure(
        ...     system,
        ...     sites,
        ...     np.asarray(
        ...         [[-1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [1., 2., 0.]],
        ...         dtype=np.float32,
        ...     ),
        ...     np.ones(len(sites), dtype=np.bool_),
        ... )
        >>> input = ProtenixStructureConditionProcessor()(
        ...     structure,
        ...     structure_condition=structure,
        ... )
        >>> atom_count = input.atom_features.shape[1]
        >>> config = ProtenixStructureConditionConfig({
        ...     "single_size": 8,
        ...     "pair_size": 8,
        ...     "trunk": {
        ...         "msa_size": 8,
        ...         "num_msa_blocks": 1,
        ...         "msa_outer_product_hidden_size": 4,
        ...         "num_msa_attention_heads": 1,
        ...         "msa_triangle_multiplication_hidden_size": 4,
        ...         "msa_triangle_attention_hidden_size": 4,
        ...         "num_msa_triangle_attention_heads": 1,
        ...         "msa_expansion": 2,
        ...         "num_pairformer_blocks": 1,
        ...         "pairformer_triangle_multiplication_hidden_size": 4,
        ...         "pairformer_triangle_attention_hidden_size": 4,
        ...         "num_pairformer_triangle_attention_heads": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "pairformer_expansion": 2,
        ...         "pairformer_dropout": 0.0,
        ...         "msa_dropout": 0.0,
        ...         "pair_dropout": 0.0,
        ...     },
        ...     "recycling_steps": 1,
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 8,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_encoder_attention_heads": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...         "num_atom_decoder_attention_heads": 1,
        ...         "num_query_atoms": 4,
        ...         "num_key_atoms": 4,
        ...     },
        ...     "distogram": {"num_bins": 4},
        ...     "confidence": {
        ...         "num_plddt_bins": 4,
        ...         "num_pae_bins": 4,
        ...         "num_pde_bins": 4,
        ...         "max_atoms_per_token": 8,
        ...         "num_blocks": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "triangle_multiplication_hidden_size": 4,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ... })
        >>> model = ProtenixStructureConditionModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.denoise(
        ...         input,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...         recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
    """

    config_class = ProtenixStructureConditionConfig
    config: ProtenixStructureConditionConfig

    def __init__(self, config: ProtenixStructureConditionConfig | None = None) -> None:
        resolved = ProtenixStructureConditionConfig() if config is None else config
        super().__init__(resolved)
        self.noisy_structure_embedder = ProtenixNoisyStructureEmbedder(resolved.pair_size)
        nn.init.zeros_(self.noisy_structure_embedder.transition.feed_forward.output_projection.weight)

    def _run_recycling(
        self,
        input: ProtenixInput,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        recycling_steps: int,
        *,
        msa_indices: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run the fixed structure-conditioned recycling loop.
        """

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(recycling_steps):
            grad_enabled = torch.is_grad_enabled() and cycle == recycling_steps - 1
            with torch.set_grad_enabled(grad_enabled):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.noisy_structure_embedder(
                    pair,
                    input.structure_rep_positions,
                    input.structure_rep_mask,
                )
                if self.template_embedder is not None:
                    pair = pair + self.template_embedder(
                        pair,
                        cast(ProtenixTemplateFeatures, input.template_features),
                        same_chain_mask=input.chain_index[..., :, None] == input.chain_index[..., None, :],
                        pair_mask=pair_mask,
                    )
                indices = None if msa_indices is None else msa_indices[cycle]
                _, pair = self.msa_trunk(
                    self._msa_features(input, indices),
                    pair,
                    token_embedding,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return single, pair


class ProtenixConstraintModel(ProtenixModel):
    r"""
    Predict constrained biomolecular complex structures with Protenix.

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import PolymerSequence
        >>> from ofoldx.models.protenix import (
        ...     ProtenixConstraintConfig,
        ...     ProtenixConstraintModel,
        ...     ProtenixConstraintProcessor,
        ...     ProtenixTokenContact,
        ... )
        >>> config = ProtenixConstraintConfig({
        ...     "single_size": 8,
        ...     "pair_size": 8,
        ...     "trunk": {
        ...         "msa_size": 8,
        ...         "num_msa_blocks": 1,
        ...         "msa_outer_product_hidden_size": 4,
        ...         "num_msa_attention_heads": 1,
        ...         "msa_triangle_multiplication_hidden_size": 4,
        ...         "msa_triangle_attention_hidden_size": 4,
        ...         "num_msa_triangle_attention_heads": 1,
        ...         "msa_expansion": 2,
        ...         "num_pairformer_blocks": 1,
        ...         "pairformer_triangle_multiplication_hidden_size": 4,
        ...         "pairformer_triangle_attention_hidden_size": 4,
        ...         "num_pairformer_triangle_attention_heads": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "pairformer_expansion": 2,
        ...         "pairformer_dropout": 0.0,
        ...         "msa_dropout": 0.0,
        ...         "pair_dropout": 0.0,
        ...     },
        ...     "recycling_steps": 1,
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 8,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_encoder_attention_heads": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...         "num_atom_decoder_attention_heads": 1,
        ...         "num_query_atoms": 4,
        ...         "num_key_atoms": 4,
        ...     },
        ...     "distogram": {"num_bins": 4},
        ...     "confidence": {
        ...         "num_plddt_bins": 4,
        ...         "num_pae_bins": 4,
        ...         "num_pde_bins": 4,
        ...         "max_atoms_per_token": 8,
        ...         "num_blocks": 1,
        ...         "num_pair_attention_heads": 1,
        ...         "triangle_multiplication_hidden_size": 4,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ... })
        >>> inputs = ProtenixConstraintProcessor()(
        ...     PolymerSequence.from_one_letter("AC"),
        ...     token_contacts=(ProtenixTokenContact(("A", 0), ("A", 1), 8.0),),
        ... )
        >>> atom_count = inputs.atom_features.shape[1]
        >>> model = ProtenixConstraintModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.denoise(
        ...         inputs,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...         recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
    """

    config_class = ProtenixConstraintConfig
    config: ProtenixConstraintConfig
    _supports_constraints = True

    def __init__(self, config: ProtenixConstraintConfig | None = None) -> None:
        resolved = ProtenixConstraintConfig() if config is None else config
        if not resolved.constraints:
            raise ValueError("protenix_constraint requires constraints=True; it is the constraint architecture")
        super().__init__(resolved)


class ProtenixTemplateModel(ProtenixModel):
    r"""
    Base Protenix graph with the template embedder enabled -- the ``protenix_template`` architecture.

    Identical to [`ProtenixModel`][] except its config defaults ``template`` on, so a fresh build always
    constructs the template embedder that its ``ProtenixTemplateProcessor`` feeds, rather than silently
    dropping template features.
    """

    config_class = ProtenixTemplateModelConfig
    config: ProtenixTemplateModelConfig
    _supports_template = True

    def __init__(self, config: ProtenixTemplateModelConfig | None = None) -> None:
        resolved = ProtenixTemplateModelConfig() if config is None else config
        if resolved.template is None:
            raise ValueError("protenix_template requires a template graph; config.template must not be None")
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
            raise ValueError("Protenix noisy structure embedding requires an even pair_size")
        half_size = pair_size // 2
        lower_bins = torch.linspace(min_distance, max_distance, num_bins)
        upper_bins = torch.cat((lower_bins[1:], lower_bins.new_tensor([1e6])))
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.pair_norm = LayerNorm(pair_size)
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

        distance = torch.cdist(
            structure_rep_positions.to(torch.float32),
            structure_rep_positions.to(torch.float32),
        ).unsqueeze(-1)
        if add_noise > 0.0:
            distance = (
                distance
                + torch.randn(
                    distance.shape,
                    device=distance.device,
                    dtype=distance.dtype,
                    generator=generator,
                )
                * add_noise
            )
        valid_pair = (structure_rep_mask.bool().unsqueeze(-1) & structure_rep_mask.bool().unsqueeze(-2)).unsqueeze(-1)
        distance_bins = ((distance > self.lower_bins.to(distance)) & (distance < self.upper_bins.to(distance))).to(
            dtype=pair.dtype
        )
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
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 385,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = NormalizedKeyAtomToTokenEncoder(
            token_hidden_size=single_size,
            atom_feature_size=atom_feature_size,
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
        num_tokens: int,
        atom_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> Tensor:
        encoder_reference_charges = torch.arcsinh(reference_charges)
        atom_encoder_output = self.atom_encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=encoder_reference_charges,
            reference_mask=reference_mask,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            local_geometry=local_geometry,
        )
        token_single = atom_encoder_output.token_hidden

        token_tail = token_features[..., self.single_size :].to(token_single)
        token_embedding = torch.cat([token_single, token_tail], dim=-1)

        if token_mask is not None:
            token_embedding = token_embedding * token_mask.unsqueeze(-1).to(dtype=token_embedding.dtype)
        return token_embedding

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


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
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Template transition expansion factor.
        dropout: Dropout probability inside each template block.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 32,
        num_distogram_bins: int = 39,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
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
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=hidden_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
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
            for block in self.blocks:
                template_pair = block(template_pair, pair_mask=pair_mask)
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
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
        self.encoder = ProtenixConstraintSubstructureEncoder(
            hidden_size=hidden_size,
            num_blocks=1,
            num_attention_heads=4,
            transition_size=hidden_size * 4,
            dropout=0.1,
            backend=backend,
        )
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, substructure: Tensor) -> Tensor:
        shape = substructure.shape
        flattened = substructure.reshape(-1, shape[-3] * shape[-2], shape[-1])
        hidden = self.encoder(self.input_projection(flattened))
        return self.output_projection(hidden).reshape(*shape[:-1], -1)


class ProtenixConstraintSubstructureEncoder(nn.Module):
    r"""
    Transformer encoder over flattened token-pair substructure features.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        transition_size: int,
        dropout: float,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            ProtenixConstraintSubstructureBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                transition_size=transition_size,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class ProtenixConstraintSubstructureBlock(nn.Module):
    r"""
    Post-norm self-attention and transition residual unit.
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
        self.attention = ProtenixConstraintSubstructureAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            backend=backend,
        )
        self.attention_output_dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        self.transition = ProtenixConstraintSubstructureTransition(
            hidden_size=hidden_size,
            transition_size=transition_size,
            dropout=dropout,
        )
        self.transition_output_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.attention_norm(hidden + self.attention_output_dropout(self.attention(hidden)))
        return self.transition_norm(hidden + self.transition_output_dropout(self.transition(hidden)))


class ProtenixConstraintSubstructureAttention(nn.Module):
    r"""
    Multi-head self-attention with separate query, key, and value projections.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout
        self.backend = backend
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        shape = (*hidden.shape[:-1], self.num_attention_heads, self.head_dim)
        query = self.query_projection(hidden).reshape(shape).transpose(-3, -2)
        key = self.key_projection(hidden).reshape(shape).transpose(-3, -2)
        value = self.value_projection(hidden).reshape(shape).transpose(-3, -2)
        update = projected_attention(
            query,
            key,
            value,
            attention_mask=None,
            backend=self.backend,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output_projection(update.transpose(-3, -2).flatten(-2))


class ProtenixConstraintSubstructureTransition(nn.Module):
    r"""
    ReLU transition update with activation dropout.
    """

    def __init__(self, *, hidden_size: int, transition_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, transition_size)
        self.activation_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(transition_size, hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.input_projection(hidden)
        hidden = self.activation_dropout(F.relu(hidden))
        return self.output_projection(hidden)


class ProtenixConfidenceHead(nn.Module):
    r"""
    Predict Protenix pLDDT, PAE, PDE, and resolved logits.

    Args:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        token_feature_size: Width of the packed per-token feature vector.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        num_blocks: Number of Pairformer blocks in the confidence stack.
        num_pair_attention_heads: Number of pair-attention heads in the Pairformer stack.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication updates.
        triangle_attention_hidden_size: Per-head width of the triangle-attention updates.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor in the Pairformer transitions.
        dropout: Pairformer dropout probability.
        single_max_abs: Optional clamp on the single representation before scoring; ``None`` disables it.
        backend: Public backend request forwarded to the Pairformer and atom projector.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.single_norm = LayerNorm(single_size, backend=backend)
        self.pair_embedder = ProtenixConfidencePairEmbedder(
            pair_size=pair_size,
            token_feature_size=token_feature_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
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
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            backend=backend,
        )
        self.pair_projector = ProtenixPairConfidenceProjector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Return atom-level and token-pair confidence logits for every predicted sample.
        """

        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
        single = self.single_norm(single)
        representative_coords = gather_atoms_by_index(
            pred_coords,
            insert_sample_axis(token_to_rep_atom, pred_coords.shape[-3]),
        )

        plddt_logits = []
        pae_logits = []
        pde_logits = []
        resolved_logits = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_pair = self.pair_embedder(pair, token_embedding, representative_coords_sample)
            sample_single, sample_pair = self.pairformer_stack(single, sample_pair, pair_mask=pair_mask)
            sample_plddt, sample_resolved = self.atom_projector(sample_single, atom_to_token, atom_token_index)
            sample_pae, sample_pde = self.pair_projector(sample_pair)
            plddt_logits.append(sample_plddt)
            pae_logits.append(sample_pae)
            pde_logits.append(sample_pde)
            resolved_logits.append(sample_resolved)

        return (
            torch.stack(plddt_logits, dim=-3),
            torch.stack(pae_logits, dim=-4),
            torch.stack(pde_logits, dim=-4),
            torch.stack(resolved_logits, dim=-3),
        )


class ProtenixConfidencePairEmbedder(nn.Module):
    r"""
    Add Protenix token and representative-atom distance features to a pair state.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.left_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.right_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        buffer_device = self.left_token_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        lower_bins = torch.arange(
            distance_bin_start,
            distance_bin_end,
            distance_bin_step,
            device=buffer_device,
            dtype=self.left_token_projection.weight.dtype,
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins)
        self.register_buffer("upper_bins", upper_bins)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)
        self.distance_projection = nn.Linear(1, pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        # torch.cdist does not implement CUDA bf16; keep the cast local to this distance calculation.
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] < self.upper_bins)).to(
            pair.dtype
        )
        pair = pair + self.distance_one_hot_projection(distance_bins)
        pair = pair + self.distance_projection(distance.to(pair.dtype).unsqueeze(-1))
        return pair


class ProtenixPairConfidenceProjector(nn.Module):
    r"""
    Project Protenix pair representations to PAE and symmetric PDE logits.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_norm = LayerNorm(pair_size)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        pae_pair = pair.to(dtype=self.pae_projection.weight.dtype)
        pde_pair = pair.to(dtype=self.pde_projection.weight.dtype)
        pae_logits = self.pae_projection(self.pae_norm(pae_pair))
        pde_logits = self.pde_projection(self.pde_norm(pde_pair + pde_pair.transpose(-2, -3)))
        return pae_logits, pde_logits


@dataclass(frozen=True)
class ProtenixExponentialInterpolation:
    r"""
    Exponentially interpolate one scalar TFG parameter over normalized diffusion time.

    Protenix uses time one for the first, noisiest transition and approaches
    time zero at the end of sampling.

    Args:
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

    Args:
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

    Args:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.0015


@dataclass(frozen=True)
class ProtenixInterchainBondTerm:
    r"""
    Configure the covalent interchain-bond TFG term.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
        interval: Evaluate every this many diffusion transitions.
        weight: Energy weight or normalized-time schedule.
    """

    interval: int = 1
    weight: float | Callable[[float], float] = 0.12


@dataclass(frozen=True)
class ProtenixLinearBondTerm:
    r"""
    Configure the linear-bond TFG term.

    Args:
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

    Args:
        rho: Strength of the gradient propagated through the denoiser to noisy
            coordinates.
        mu: Direct clean-coordinate log-probability ascent step size.
        monte_carlo_std: Standard deviation of clean-coordinate perturbations
            used for Monte Carlo smoothing.
        monte_carlo_samples: Number of Monte Carlo perturbations when the
            standard deviation is nonzero.
        outer_steps: TFG predictor-corrector iterations per diffusion transition.
        inner_steps: Direct clean-coordinate refinement iterations per outer step.
        projection_outer_steps: Passes over the chiral-then-pairwise projection order.
        projection_inner_steps: Projection updates for each enabled projection term.
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
    monte_carlo_samples: int = 1
    outer_steps: int = 1
    inner_steps: int = 20
    projection_outer_steps: int = 2
    projection_inner_steps: int = 10
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
                features.chiral_index,
                features.chiral_orientation,
                features.atom_chain,
            )
        )
        self.stereo_bond = (
            None
            if guidance.stereo_bond is None
            else ProtenixStereoBondPotential(
                features.stereo_bond_index,
                features.stereo_bond_orientation,
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
        monte_carlo_count = monte_carlo_noise.shape[0]
        perturbed = (coordinates.unsqueeze(0) + monte_carlo_noise).reshape(
            -1,
            *coordinates.shape[-2:],
        )
        log_probability = -self(
            perturbed,
            time=time,
            step_index=step_index,
        ).reshape(monte_carlo_count, *coordinates.shape[:-2])
        if monte_carlo_count == 1:
            return log_probability.squeeze(0)
        return torch.logsumexp(log_probability, dim=0) - math.log(monte_carlo_count)

    def log_probability_and_gradient(
        self,
        coordinates: Tensor,
        monte_carlo_noise: Tensor,
        *,
        time: float,
        step_index: int,
    ) -> tuple[Tensor, Tensor]:
        monte_carlo_count = monte_carlo_noise.shape[0]
        perturbed = (coordinates.unsqueeze(0) + monte_carlo_noise).reshape(
            -1,
            *coordinates.shape[-2:],
        )
        energy, energy_gradient = self.energy_and_gradient(
            perturbed,
            time=time,
            step_index=step_index,
        )
        energy = energy.reshape(monte_carlo_count, *coordinates.shape[:-2])
        energy_gradient = energy_gradient.reshape(monte_carlo_count, *coordinates.shape)
        log_probability = -energy
        if monte_carlo_count == 1:
            return log_probability.squeeze(0), -energy_gradient.squeeze(0)
        average = torch.logsumexp(log_probability, dim=0) - math.log(monte_carlo_count)
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
        delta = torch.zeros_like(coordinates)
        chiral = self.guidance.chiral_atom
        pairwise = self.guidance.pairwise_distance
        for _ in range(self.guidance.projection_outer_steps):
            if (
                chiral is not None
                and self.chiral_atom is not None
                and chiral.enable_projection
                and step_index % chiral.interval == 0
            ):
                buffer = _protenix_tfg_scalar(chiral.buffer, time)
                for _ in range(self.guidance.projection_inner_steps):
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
                for _ in range(self.guidance.projection_inner_steps):
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
        self.register_buffer("pair_index", pair_index)
        self.register_buffer("equilibrium_distance", equilibrium_distance)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.225) -> Tensor:
        if self.pair_index.numel() == 0:
            return _protenix_zero_energy(coordinates)

        batch_shape = coordinates.shape[:-2]
        num_atoms = coordinates.shape[-2]
        flattened_batch = math.prod(batch_shape) if batch_shape else 1
        flattened_coordinates = coordinates.reshape(flattened_batch, num_atoms, 3)
        distances = _protenix_distances(flattened_coordinates, self.pair_index)
        equilibrium = self.equilibrium_distance.to(distances)
        active = distances < equilibrium.unsqueeze(0) * (1.0 - buffer)
        batch_index, pair_index = active.nonzero(as_tuple=True)
        if batch_index.numel() == 0:
            return _protenix_zero_energy(coordinates)

        active_equilibrium = equilibrium.index_select(0, pair_index)
        active_distance = distances[batch_index, pair_index]
        difference = active_distance - active_equilibrium
        first_gaussian = -0.0356 * torch.exp(-((difference / 0.5) ** 2))
        second_gaussian = -0.00516 * torch.exp(-(((difference - 3.0) / 2.0) ** 2))
        repulsion = 0.840 * torch.where(difference < 0.0, difference**2, 0.0)
        energy = torch.zeros(
            (flattened_batch,),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        energy.scatter_add_(0, batch_index, first_gaussian + second_gaussian + repulsion)
        if not batch_shape:
            return energy[0]
        return energy.reshape(*batch_shape)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.225,
    ) -> tuple[Tensor, Tensor]:
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
    ) -> None:
        super().__init__()
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("force_constant", force_constant)
        self.register_buffer("sign", sign)

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        angle = _protenix_dihedral(coordinates, self.atom_index)
        order = torch.arange(1, 7, device=coordinates.device, dtype=coordinates.dtype)
        energy = self.force_constant.to(coordinates) * (
            1.0 + self.sign.to(coordinates) * torch.cos(angle.unsqueeze(-1) * order)
        )
        return energy.sum(dim=(-1, -2))

    def energy_and_gradient(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = _protenix_dihedral_and_gradient(coordinates, self.atom_index)
        order = torch.arange(1, 7, device=coordinates.device, dtype=coordinates.dtype)
        force_constant = self.force_constant.to(coordinates)
        sign = self.sign.to(coordinates)
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
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor, *, buffer: float = 2.0) -> Tensor:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        distances = _protenix_distances(coordinates, self.atom_index)
        return torch.clamp_min(distances - buffer, 0.0).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 2.0,
    ) -> tuple[Tensor, Tensor]:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        distances, distance_gradient = _protenix_distance_and_gradient(coordinates, self.atom_index)
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
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        lower_bound, upper_bound = self._distance_bounds(
            bond_buffer=bond_buffer,
            angle_buffer=angle_buffer,
            clash_buffer=clash_buffer,
        )
        distances = _protenix_distances(coordinates, self.atom_index)
        below = torch.clamp_min(lower_bound.to(distances) - distances, 0.0)
        above = torch.clamp_min(distances - upper_bound.to(distances), 0.0)
        return (0.5 * (below**2 + above**2)).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        bond_buffer: float = 0.05,
        angle_buffer: float = 0.05,
        clash_buffer: float = 0.05,
    ) -> tuple[Tensor, Tensor]:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        lower_bound, upper_bound = self._distance_bounds(
            bond_buffer=bond_buffer,
            angle_buffer=angle_buffer,
            clash_buffer=clash_buffer,
        )
        distances, distance_gradient = _protenix_distance_and_gradient(coordinates, self.atom_index)
        below = lower_bound.to(distances) - distances
        above = distances - upper_bound.to(distances)
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
        vdw_limit = self.vdw_limit.to(lower_bound)
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
        lower = lower_bound[selected].to(coordinates)
        upper = upper_bound[selected].to(coordinates)
        distances, distance_gradient = _protenix_distance_and_gradient(coordinates, atom_index)
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
    ) -> None:
        super().__init__()
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("orientation", orientation)
        self.register_buffer("atom_chain", atom_chain)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.34906) -> Tensor:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        signed_angle = _protenix_dihedral(coordinates, self.atom_index) * self.orientation.to(coordinates)
        return torch.clamp_min(buffer - signed_angle, 0.0).sum(dim=-1)

    def energy_and_gradient(
        self,
        coordinates: Tensor,
        *,
        buffer: float = 0.34906,
    ) -> tuple[Tensor, Tensor]:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = _protenix_dihedral_and_gradient(coordinates, self.atom_index)
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

        if self.atom_index.numel() == 0:
            return torch.zeros_like(coordinates)
        angle, angle_gradient = _protenix_dihedral_and_gradient(coordinates, self.atom_index)
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
    ) -> None:
        super().__init__()
        self.register_buffer("atom_index", atom_index)
        self.register_buffer("orientation", orientation)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.52360) -> Tensor:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        angle = _protenix_dihedral(coordinates, self.atom_index).abs()
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
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        signed_angle, signed_gradient = _protenix_dihedral_and_gradient(coordinates, self.atom_index)
        sign = torch.where(signed_angle < 0, -1.0, 1.0)
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
        derivative = torch.where(below, -1.0, torch.where(above, 1.0, 0.0))
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
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates)
        return _protenix_planar_improper_energy(coordinates, self.atom_index).sum(dim=-1)

    def energy_and_gradient(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
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
        self.register_buffer("atom_index", atom_index)

    def forward(self, coordinates: Tensor, *, buffer: float = 0.08726646259) -> Tensor:
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
        if self.atom_index.numel() == 0:
            return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)
        angle, angle_gradient = _protenix_angle_and_gradient(coordinates, self.atom_index)
        violation = math.pi - buffer - angle
        active = violation > 0
        gradient = _protenix_aggregate_atom_gradient(
            coordinates,
            self.atom_index,
            angle_gradient,
            torch.where(active, -1.0, 0.0),
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


@overload
def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: Literal[False] = False,
) -> Tensor: ...


@overload
def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def _protenix_bin_expectation(
    logits: Tensor,
    minimum: float,
    maximum: float,
    *,
    return_probability: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    Decode pLDDT/PAE/PDE categorical logits.
    """

    probability = logits.softmax(dim=-1)
    values = probability @ _protenix_bin_centers(logits, minimum, maximum, logits.shape[-1])
    if return_probability:
        return values, probability
    return values


def _protenix_contact_probability(distogram_logits: Tensor) -> Tensor:
    r"""
    Decode the probability of distance below 8 Å.
    """

    centers = _protenix_bin_centers(
        distogram_logits,
        2.3125,
        21.6875,
        distogram_logits.shape[-1],
    )
    return distogram_logits.softmax(dim=-1)[..., centers < 8.0].sum(dim=-1)


def _protenix_tm_scores(
    pae_probability: Tensor,
    *,
    token_has_frame: Tensor,
    token_chain: Tensor,
    atom_to_token: Tensor,
    atom_is_polymer: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Compute global and chain TM summaries.
    """

    token_has_frame = token_has_frame.to(device=pae_probability.device, dtype=torch.bool)
    token_chain = token_chain.to(device=pae_probability.device, dtype=torch.long)

    def calculate(probability: Tensor, has_frame: Tensor, chain: Tensor) -> tuple[Tensor, Tensor]:
        if not bool(has_frame.any()):
            zeros = probability.new_zeros(probability.shape[:-3])
            return zeros, zeros
        num_tokens = has_frame.shape[0]
        centers = _protenix_bin_centers(probability, 0.0, 32.0, probability.shape[-1])
        token_pair_tm = (probability * tm_bin_weights(centers, num_tokens)).sum(dim=-1)
        ptm_score = token_pair_tm.mean(dim=-1)[..., has_frame].max(dim=-1).values
        different_chain = chain[None, :] != chain[:, None]
        iptm_per_alignment = (token_pair_tm * different_chain).sum(dim=-1) / (
            torch.finfo(token_pair_tm.dtype).tiny + different_chain.sum(dim=-1)
        )
        iptm_score = iptm_per_alignment[..., has_frame].max(dim=-1).values
        return ptm_score, iptm_score

    ptm, iptm = calculate(pae_probability, token_has_frame, token_chain)
    chains = torch.unique(token_chain)
    chain_masks = tuple(token_chain == chain for chain in chains)
    batch_shape = pae_probability.shape[:-3]
    chain_ptm = pae_probability.new_zeros(batch_shape + (len(chains),))
    chain_pair_iptm = pae_probability.new_zeros(batch_shape + (len(chains), len(chains)))
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
            chain_pair_iptm[..., first_index, second_index] = pair_iptm
            chain_pair_iptm[..., second_index, first_index] = pair_iptm

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
    chain_has_frame = tuple(bool((mask & token_has_frame).any()) for mask in chain_masks)
    chain_iptm = pae_probability.new_zeros(batch_shape + (len(chains),))
    for chain_index in range(len(chains)):
        values = [
            chain_pair_iptm[..., first_index, second_index]
            for first_index in range(len(chains))
            for second_index in range(len(chains))
            if (first_index == chain_index or second_index == chain_index)
            and first_index != second_index
            and chain_has_frame[first_index]
        ]
        if values:
            chain_iptm[..., chain_index] = torch.stack(values, dim=-1).mean(dim=-1)

    chain_is_ligand = tuple(token_is_ligand[mask].sum() >= mask.sum() // 2 for mask in chain_masks)
    chain_pair_iptm_global = torch.zeros_like(chain_pair_iptm)
    for first_index in range(len(chains)):
        for second_index in range(len(chains)):
            if first_index == second_index:
                continue
            if chain_is_ligand[first_index]:
                value = chain_iptm[..., first_index]
            elif chain_is_ligand[second_index]:
                value = chain_iptm[..., second_index]
            else:
                value = 0.5 * (chain_iptm[..., first_index] + chain_iptm[..., second_index])
            chain_pair_iptm_global[..., first_index, second_index] = value
    return ptm, iptm, chain_ptm, chain_iptm, chain_pair_iptm, chain_pair_iptm_global


def _protenix_chain_gpde(
    token_pair_pde: Tensor,
    *,
    contact_probability: Tensor,
    token_chain: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute chain and chain-pair gPDE summaries.
    """

    token_chain = token_chain.to(device=token_pair_pde.device, dtype=torch.long)
    contact_probability = contact_probability.to(device=token_pair_pde.device)
    chains = torch.unique(token_chain)
    chain_gpde = token_pair_pde.new_zeros(token_pair_pde.shape[:-2] + (len(chains),))
    chain_pair_gpde = token_pair_pde.new_zeros(token_pair_pde.shape[:-2] + (len(chains), len(chains)))

    def calculate(row_mask: Tensor, column_mask: Tensor) -> Tensor:
        contact = contact_probability[..., row_mask, :][..., column_mask]
        pde = token_pair_pde[..., row_mask, :][..., column_mask]
        return (pde * contact).sum(dim=(-1, -2)) / (contact.sum(dim=(-1, -2)) + torch.finfo(contact.dtype).tiny)

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
    Compute chain and chain-pair pLDDT summaries.
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


def _protenix_token_has_frame(
    reference_positions: Tensor,
    *,
    reference_mask: Tensor,
    reference_space_id: Tensor,
    token_to_rep_atom: Tensor,
    atom_sites: tuple[AtomSite, ...],
    token_sites: tuple[AtomSite | tuple[str, int], ...],
) -> Tensor:
    r"""
    Compute token-frame validity from the reference conformer.
    """

    atom_count = len(atom_sites)
    reference_positions = reference_positions[0, :atom_count].float()
    reference_mask = reference_mask[0, :atom_count].bool()
    reference_space = reference_space_id[0, :atom_count]
    representative_atom = token_to_rep_atom[0, : len(token_sites)].long()
    has_frame = torch.zeros(len(token_sites), dtype=torch.bool, device=reference_positions.device)
    for token_index, token_site in enumerate(token_sites):
        if type(token_site) is tuple:
            has_frame[token_index] = True
            continue
        center = representative_atom[token_index]
        same_reference = torch.where(reference_space == reference_space[center])[0]
        neighbors = same_reference[same_reference != center]
        if neighbors.numel() < 2:
            continue
        squared_distance = ((reference_positions[neighbors] - reference_positions[center]) ** 2).sum(dim=-1)
        nearest = neighbors[torch.argsort(squared_distance, stable=True)[:2]]
        frame_atoms = torch.cat((nearest[:1], center.reshape(1), nearest[1:]))
        if not bool(reference_mask[frame_atoms].all()):
            continue
        first = reference_positions[nearest[0]] - reference_positions[center]
        second = reference_positions[nearest[1]] - reference_positions[center]
        norm_product = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
        if bool(torch.isclose(norm_product, norm_product.new_zeros(()))):
            continue
        angle = torch.rad2deg(torch.acos(torch.dot(first, second).div(norm_product).clamp(-1.0, 1.0)))
        has_frame[token_index] = (angle > 25.0) & (angle < 155.0)
    return has_frame


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
            distances = torch.cdist(coordinates[:, first_mask], coordinates[:, second_mask])
            clashes = (distances < 1.1).sum(dim=(-1, -2))
            shorter_chain = min(int(first_mask.sum()), int(second_mask.sum()))
            has_clash |= (clashes > 100) | (clashes / shorter_chain > 0.5)
    return has_clash


def _validate_protenix_tfg_guidance(guidance: ProtenixTfgGuidance) -> None:
    if guidance.monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive")
    if guidance.outer_steps < 1:
        raise ValueError("outer_steps must be positive")
    if guidance.inner_steps < 0:
        raise ValueError("inner_steps cannot be negative")
    if guidance.projection_outer_steps < 0:
        raise ValueError("projection_outer_steps cannot be negative")
    if guidance.projection_inner_steps < 0:
        raise ValueError("projection_inner_steps cannot be negative")

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
        raise ValueError("ProtenixTfgGuidance requires at least one potential term")
    for term in terms:
        if term is not None and term.interval < 1:
            raise ValueError(f"{type(term).__name__}.interval must be positive; use None to remove a term")


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
    noise = torch.empty(
        (guidance.monte_carlo_samples, *coordinates.shape),
        device=coordinates.device,
        dtype=coordinates.dtype,
    ).normal_(generator=generator)
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
    return coordinates.sum(dim=(-1, -2)) * 0.0


def _protenix_distances(coordinates: Tensor, atom_index: Tensor) -> Tensor:
    difference = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[1])
    return torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-8)


def _protenix_distance_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    difference = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[1])
    distance = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-8)
    direction = difference / distance.unsqueeze(-1)
    return distance, torch.stack((direction, -direction), dim=-3)


def _protenix_vina_energy_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
    equilibrium_distance: Tensor,
    *,
    buffer: float,
) -> tuple[Tensor, Tensor]:
    if atom_index.numel() == 0:
        return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)

    batch_shape = coordinates.shape[:-2]
    num_atoms = coordinates.shape[-2]
    flattened_batch = math.prod(batch_shape) if batch_shape else 1
    flattened_coordinates = coordinates.reshape(flattened_batch, num_atoms, 3)
    distances = _protenix_distances(flattened_coordinates, atom_index)
    equilibrium = equilibrium_distance.to(distances)
    active = distances < equilibrium.unsqueeze(0) * (1.0 - buffer)
    batch_index, pair_index = active.nonzero(as_tuple=True)
    if batch_index.numel() == 0:
        return _protenix_zero_energy(coordinates), torch.zeros_like(coordinates)

    first_atom = atom_index[0].index_select(0, pair_index)
    second_atom = atom_index[1].index_select(0, pair_index)
    active_equilibrium = equilibrium.index_select(0, pair_index)
    active_distance = distances[batch_index, pair_index]
    difference = active_distance - active_equilibrium
    normalized_difference = difference / 0.5
    first_gaussian = -0.0356 * torch.exp(-(normalized_difference**2))
    second_gaussian = -0.00516 * torch.exp(-(((difference - 3.0) / 2.0) ** 2))
    repulsion = 0.840 * torch.where(difference < 0.0, difference**2, 0.0)

    energy = torch.zeros(
        (flattened_batch,),
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    energy.scatter_add_(0, batch_index, first_gaussian + second_gaussian + repulsion)

    displacement = flattened_coordinates[batch_index, first_atom] - flattened_coordinates[batch_index, second_atom]
    direction = displacement / torch.linalg.vector_norm(
        displacement,
        dim=-1,
    ).clamp_min(
        1e-8
    ).unsqueeze(-1)
    distance_derivative = (
        -2.0 * first_gaussian * normalized_difference * (1.0 / 0.5)
        - 0.5 * second_gaussian * (difference - 3.0)
        + 0.840 * torch.where(difference < 0.0, 2.0 * difference, 0.0)
    )
    first_gradient = distance_derivative.unsqueeze(-1) * direction
    flattened_gradient = torch.zeros(
        (flattened_batch * num_atoms, 3),
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    flat_first_atom = batch_index * num_atoms + first_atom
    flat_second_atom = batch_index * num_atoms + second_atom
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
    if batch_shape:
        return energy.reshape(*batch_shape), gradient
    return energy[0], gradient


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
    angle = torch.atan2(
        torch.linalg.vector_norm(cross, dim=-1),
        torch.linalg.vecdot(first, second) + 1e-8,
    )
    first_square = torch.linalg.vecdot(first, first) + 1e-8
    second_square = torch.linalg.vecdot(second, second) + 1e-8
    cross_norm = torch.linalg.vector_norm(cross, dim=-1) + 1e-8
    first_gradient = torch.linalg.cross(first, cross, dim=-1) / (first_square * cross_norm).unsqueeze(-1)
    third_gradient = torch.linalg.cross(cross, second, dim=-1) / (second_square * cross_norm).unsqueeze(-1)
    second_gradient = -first_gradient - third_gradient
    return angle, torch.stack(
        (first_gradient, second_gradient, third_gradient),
        dim=-3,
    )


def _protenix_dihedral(coordinates: Tensor, atom_index: Tensor) -> Tensor:
    first = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[0])
    middle = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[2])
    last = coordinates.index_select(-2, atom_index[3]) - coordinates.index_select(-2, atom_index[2])
    first_normal = torch.linalg.cross(first, middle, dim=-1)
    second_normal = torch.linalg.cross(middle, last, dim=-1)
    normal_cross = torch.linalg.cross(first_normal, second_normal, dim=-1)
    angle = torch.atan2(
        torch.linalg.vector_norm(normal_cross, dim=-1),
        torch.linalg.vecdot(first_normal, second_normal) + 1e-8,
    )
    return -angle * torch.sign(torch.linalg.vecdot(first, second_normal))


def _protenix_dihedral_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    first = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[0])
    middle = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[2])
    last = coordinates.index_select(-2, atom_index[3]) - coordinates.index_select(-2, atom_index[2])
    first_normal = torch.linalg.cross(first, middle, dim=-1)
    second_normal = torch.linalg.cross(middle, last, dim=-1)
    normal_cross = torch.linalg.cross(first_normal, second_normal, dim=-1)
    angle = torch.atan2(
        torch.linalg.vector_norm(normal_cross, dim=-1),
        torch.linalg.vecdot(first_normal, second_normal) + 1e-8,
    )
    angle = -angle * torch.sign(torch.linalg.vecdot(first, second_normal))

    first_normal_square = torch.linalg.vecdot(first_normal, first_normal) + 1e-8
    second_normal_square = torch.linalg.vecdot(second_normal, second_normal) + 1e-8
    middle_square = torch.linalg.vecdot(middle, middle)
    inverse_middle = torch.rsqrt(middle_square + 1e-8)
    inverse_middle_square = torch.square(inverse_middle)
    middle_norm = middle_square * inverse_middle
    first_gradient = middle_norm.unsqueeze(-1) * first_normal / first_normal_square.unsqueeze(-1)
    last_gradient = -middle_norm.unsqueeze(-1) * second_normal / second_normal_square.unsqueeze(-1)
    first_projection = torch.linalg.vecdot(first, middle) * inverse_middle_square
    last_projection = torch.linalg.vecdot(last, middle) * inverse_middle_square
    correction = first_projection.unsqueeze(-1) * first_gradient - last_projection.unsqueeze(-1) * last_gradient
    second_gradient = first_gradient - correction
    third_gradient = last_gradient + correction
    gradient = torch.stack(
        (
            first_gradient,
            -second_gradient,
            -third_gradient,
            last_gradient,
        ),
        dim=-3,
    )
    return angle, gradient


def _protenix_planar_improper_energy(coordinates: Tensor, atom_index: Tensor) -> Tensor:
    first = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[2])
    second = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[2])
    third = coordinates.index_select(-2, atom_index[3]) - coordinates.index_select(-2, atom_index[2])
    first_square = torch.linalg.vecdot(first, first)
    second_square = torch.linalg.vecdot(second, second)
    third_square = torch.linalg.vecdot(third, third)
    valid_distance = (first_square > 1e-10) & (second_square > 1e-10) & (third_square > 1e-10)
    first = first / torch.sqrt(first_square).unsqueeze(-1).clamp_min(1e-8)
    second = second / torch.sqrt(second_square).unsqueeze(-1).clamp_min(1e-8)
    third = third / torch.sqrt(third_square).unsqueeze(-1).clamp_min(1e-8)
    normal = torch.linalg.cross(-first, second, dim=-1)
    normal_square = torch.linalg.vecdot(normal, normal)
    valid = valid_distance & (normal_square > 1e-10)
    normal = normal / torch.sqrt(normal_square).unsqueeze(-1).clamp_min(1e-8)
    cosine = torch.where(valid, torch.linalg.vecdot(normal, third), 0.0).clamp(-1.0, 1.0)
    sine = torch.sqrt((1.0 - cosine**2).clamp_min(0.0)).clamp_min(1e-8)
    return 1.0 - sine


def _protenix_planar_improper_energy_and_gradient(
    coordinates: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    first = coordinates.index_select(-2, atom_index[0]) - coordinates.index_select(-2, atom_index[2])
    second = coordinates.index_select(-2, atom_index[1]) - coordinates.index_select(-2, atom_index[2])
    third = coordinates.index_select(-2, atom_index[3]) - coordinates.index_select(-2, atom_index[2])
    first_square = torch.linalg.vecdot(first, first)
    second_square = torch.linalg.vecdot(second, second)
    third_square = torch.linalg.vecdot(third, third)
    valid_distance = (first_square > 1e-10) & (second_square > 1e-10) & (third_square > 1e-10)
    first = first / torch.sqrt(first_square).unsqueeze(-1).clamp_min(1e-8)
    second = second / torch.sqrt(second_square).unsqueeze(-1).clamp_min(1e-8)
    third = third / torch.sqrt(third_square).unsqueeze(-1).clamp_min(1e-8)
    normal = torch.linalg.cross(-first, second, dim=-1)
    normal_square = torch.linalg.vecdot(normal, normal)
    valid = valid_distance & (normal_square > 1e-10)
    normal = normal / torch.sqrt(normal_square).unsqueeze(-1).clamp_min(1e-8)
    cosine = torch.where(valid, torch.linalg.vecdot(normal, third), 0.0).clamp(-1.0, 1.0)
    sine = torch.sqrt((1.0 - cosine**2).clamp_min(0.0)).clamp_min(1e-8)
    energy = 1.0 - sine

    cosine_theta = torch.linalg.vecdot(first, second).clamp(-1.0, 1.0)
    sine_theta_square = (1.0 - cosine_theta**2).clamp_min(1e-12)
    sine_theta = torch.sqrt(sine_theta_square).clamp_min(1e-8)
    first_cross = torch.linalg.cross(third, second, dim=-1)
    second_cross = torch.linalg.cross(first, third, dim=-1)
    third_cross = torch.linalg.cross(second, first, dim=-1)
    first_term = sine * sine_theta
    second_term = cosine / (sine * sine_theta_square)
    first_gradient = (
        first_cross / first_term.unsqueeze(-1)
        - (first - second * cosine_theta.unsqueeze(-1)) * second_term.unsqueeze(-1)
    ) / torch.sqrt(first_square).unsqueeze(-1).clamp_min(1e-8)
    second_gradient = (
        second_cross / first_term.unsqueeze(-1)
        - (second - first * cosine_theta.unsqueeze(-1)) * second_term.unsqueeze(-1)
    ) / torch.sqrt(second_square).unsqueeze(-1).clamp_min(1e-8)
    third_gradient = (third_cross / first_term.unsqueeze(-1) - third * (cosine / sine).unsqueeze(-1)) / torch.sqrt(
        third_square
    ).unsqueeze(-1).clamp_min(1e-8)
    center_gradient = -(first_gradient + second_gradient + third_gradient)
    gradient = torch.stack(
        (
            first_gradient,
            second_gradient,
            center_gradient,
            third_gradient,
        ),
        dim=-3,
    ) * cosine.unsqueeze(-2).unsqueeze(-1)
    gradient = torch.where(
        valid.unsqueeze(-2).unsqueeze(-1),
        gradient,
        0.0,
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
    gradient = torch.zeros_like(coordinates)
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
    violation = violation.reshape(flattened_batch, num_constraints)
    active = active.reshape(flattened_batch, num_constraints)
    value_gradient = value_gradient.reshape(
        flattened_batch,
        atom_index.shape[0],
        num_constraints,
        3,
    )
    delta = torch.zeros(
        (flattened_batch, num_atoms, 3),
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    for batch_index in range(flattened_batch):
        selected = active[batch_index]
        num_selected = int(selected.sum())
        if num_selected == 0:
            continue
        selected_index = atom_index[:, selected]
        selected_gradient = value_gradient[batch_index, :, selected, :]
        atom_gradient = torch.zeros(
            (num_selected, num_atoms, 3),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        rows = torch.arange(num_selected, device=coordinates.device)
        for endpoint in range(atom_index.shape[0]):
            atom_gradient[rows, selected_index[endpoint]] += selected_gradient[endpoint]
        jacobian = atom_gradient.reshape(num_selected, num_atoms * 3)
        work_dtype = torch.float32 if coordinates.dtype in {torch.float16, torch.bfloat16} else coordinates.dtype
        jacobian = jacobian.to(work_dtype)
        right_hand_side = violation[batch_index, selected].to(work_dtype)
        gram = jacobian @ jacobian.T
        gram = gram + 1e-8 * torch.eye(
            num_selected,
            device=coordinates.device,
            dtype=work_dtype,
        )
        multiplier = torch.linalg.solve(gram, right_hand_side)
        delta[batch_index] = (-(jacobian.T @ multiplier)).to(coordinates).reshape(num_atoms, 3)
    return delta.reshape(*coordinates.shape)


def _protenix_preserve_chain_radius(
    coordinates: Tensor,
    delta: Tensor,
    *,
    atom_index: Tensor,
    atom_chain: Tensor,
) -> Tensor:
    affected = torch.zeros(coordinates.shape[-2], dtype=torch.bool, device=coordinates.device)
    affected[atom_index.flatten()] = True
    corrected = delta.clone()
    for chain in torch.unique(atom_chain[affected]):
        selected = affected & (atom_chain == chain)
        original = coordinates[..., selected, :]
        proposed = original + delta[..., selected, :]
        original_center = original.mean(dim=-2, keepdim=True)
        proposed_center = proposed.mean(dim=-2, keepdim=True)
        original_radius = ((original - original_center) ** 2).sum(dim=-1).mean(dim=-1, keepdim=True)
        proposed_radius = ((proposed - proposed_center) ** 2).sum(dim=-1).mean(dim=-1, keepdim=True)
        scale = torch.sqrt(original_radius / proposed_radius.clamp_min(1e-12)).unsqueeze(-1)
        corrected[..., selected, :] = (proposed - proposed_center) * scale + original_center - original
    return corrected


def _gather_protenix_structure_target(
    target: ProtenixStructureTarget,
    input: ProtenixInput,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather heavy-atom truth onto the current atom axis.
    """

    atom_index = target.atom_index.long()
    valid = atom_index >= 0
    gather_index = atom_index.clamp_min(0)
    coordinates = target.coordinates.gather(
        1,
        gather_index.unsqueeze(-1).expand(-1, -1, 3),
    )
    coordinate_mask = target.coordinate_mask.gather(1, gather_index)
    coordinate_mask = coordinate_mask & valid & input.atom_mask.bool()
    coordinates = coordinates.masked_fill(~coordinate_mask.unsqueeze(-1), 0)
    return coordinates, coordinate_mask


def _center_random_augmentation(
    coordinates: Tensor,
    coordinate_mask: Tensor,
    *,
    augmentation: bool,
    translation_scale: float,
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Center masked coordinates and optionally apply one rigid transform per sample.
    """

    weight = coordinate_mask.unsqueeze(-1).to(dtype=coordinates.dtype)
    centroid = (coordinates * weight).sum(dim=-2, keepdim=True) / weight.sum(
        dim=-2,
        keepdim=True,
    ).clamp_min(1.0)
    centered = coordinates - centroid
    if not augmentation:
        return centered
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
    return torch.einsum("bmd,bds->bms", centered, rotation) + translation_scale * translation


# =====================================================================================
# Typed tensor and semantic boundary of the Protenix graph.
# =====================================================================================


@tensorclass
class ProtenixTemplateFeatures:
    r"""
    Fixed-four template features consumed by [`ProtenixTemplateEmbedder`][].

    Args:
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


@dataclass(kw_only=True, eq=False, repr=False)
class ProtenixTfgFeatures:
    r"""
    Static molecular geometry consumed by Protenix training-free guidance.

    Index tensors are unbatched and use the model atom axis.

    Args:
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


@tensorclass(frozen=True)
class _ProtenixTrunkOutput:
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

    Args:
        atom_a: First atom occurrence.
        atom_b: Second atom occurrence.
        max_distance: Maximum distance in angstroms.
    """

    atom_a: AtomSite
    atom_b: AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if self.atom_a == self.atom_b:
            raise ValueError("a Protenix atom contact needs two distinct atoms")
        if self.max_distance <= 0:
            raise ValueError(
                "ProtenixAtomContact max_distance must be positive; [0, 0] is indistinguishable from no contact"
            )


@dataclass(frozen=True)
class ProtenixTokenContact:
    r"""
    Set an upper distance bound between token-level endpoints.

    Args:
        endpoint_a: First residue or atom endpoint.
        endpoint_b: Second residue or atom endpoint.
        max_distance: Maximum token-center distance in angstroms.

    Raises:
        ValueError: If both endpoints are atoms or ``max_distance`` is not positive.
    """

    endpoint_a: ComponentReference | AtomSite
    endpoint_b: ComponentReference | AtomSite
    max_distance: float

    def __post_init__(self) -> None:
        if isinstance(self.endpoint_a, AtomSite) and isinstance(self.endpoint_b, AtomSite):
            raise ValueError(
                "a two-atom Protenix contact is the atom channel; use ProtenixAtomContact instead of "
                "ProtenixTokenContact"
            )
        if self.max_distance <= 0:
            raise ValueError(
                "ProtenixTokenContact max_distance must be positive; [0, 0] is indistinguishable from no contact"
            )


@dataclass(frozen=True)
class ProtenixPocket:
    r"""
    Condition one binder chain to contact a set of residues.

    Args:
        binder: Molecular instance ID of the binder chain.
        contact_residues: Target residue occurrences outside the binder chain.
        max_distance: Maximum binder-to-residue distance in angstroms.
    """

    binder: str
    contact_residues: tuple[ComponentReference, ...]
    max_distance: float

    def __post_init__(self) -> None:
        if not self.contact_residues:
            raise ValueError("ProtenixPocket requires at least one contact residue")
        if self.max_distance <= 0:
            raise ValueError("ProtenixPocket max_distance must be positive")
        for residue in self.contact_residues:
            if residue[0] == self.binder:
                raise ValueError("a Protenix pocket residue cannot belong to the binder chain")


@tensorclass
class ProtenixStructureTarget:
    r"""
    Heavy-atom structure truth for Protenix training.

    Args:
        coordinates: Heavy-atom truth coordinates with shape ``(B, T, 3)`` in
            angstroms.
        coordinate_mask: Resolved-coordinate mask with shape ``(B, T)``.
        atom_index: Current model-atom to truth-pool mapping with shape
            ``(B, A)``; ``-1`` marks padding.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    atom_index: Tensor


@tensorclass
class ProtenixInput:
    r"""
    Tensor and molecular-layout input for Protenix.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.

    Args:
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
        token_to_rep_atom: Representative atom index for each token with shape
            ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, C_token)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        msa_tokens: MSA token IDs with shape ``(B, M, N)``.
        has_deletion: MSA deletion flags with shape ``(B, M, N)``.
        deletion_value: Transformed MSA deletion values with shape ``(B, M, N)``.
        esm_tokens: ESM2 sequence token IDs with shape ``(B, E, L)``.
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
        tfg_guidance: The guidance ``tfg_features`` was prepared for; ``sample`` defaults to it when no
            guidance is passed explicitly, so a processor-prepared input carries its own guidance.
        system: Molecular system represented by the tensors.
        atom_sites: Atom occurrences aligned to the atom axis.
        token_sites: Molecular occurrences aligned to the token axis.
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
    token_to_rep_atom: Tensor
    token_features: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    token_bonds: Tensor
    msa_tokens: Tensor
    has_deletion: Tensor
    deletion_value: Tensor
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
    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | ComponentReference, ...]
    tfg_guidance: ProtenixTfgGuidance | None = None


@tensorclass
class ProtenixRawOutput:
    r"""
    Raw predictions from one Protenix denoising or sampling call.

    Args:
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

        return expected_plddt(self.plddt_logits)


@tensorclass
class ProtenixOutput(ProtenixRawOutput):
    r"""
    Predictions from one Protenix training denoising pass.

    Args:
        augmented_target_coordinates: Training targets after rigid augmentation,
            with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
    """

    augmented_target_coordinates: Tensor
    noise_level: Tensor
