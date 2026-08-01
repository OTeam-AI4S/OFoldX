r"""
AlphaFold 3 top-level model.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...compilation import HomogeneousBlockRegions
from ...data.evidence import MSA_VOCAB_SIZE
from ...data.molecular import MolecularSystem
from ...data.tokenization import TOKEN_INDEX, TOKEN_VOCAB_SIZE
from ...model import Model, _OptionalTensor, _TensorCollection
from ...modules.atom import (
    AtomToTokenEncoder,
    CoordinateAtomToTokenEncoder,
    LocalAtomGeometry,
    TokenToAtomDecoder,
)
from ...modules.attention import PairBiasedSelfAttention, TriangleAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.backend import categorical_msa_projection as dispatch_categorical_msa_projection
from ...modules.backend import resolve_config_backend
from ...modules.backend import triangle_attention as dispatch_triangle_attention
from ...modules.denoiser import DenoisingTokenRefiner
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoiseFn,
    CoordinateDenoisingInput,
    CoordinateDenoisingOutput,
    CoordinateEdmSampler,
    EdmPreconditioner,
    EdmScheduler,
    denoise_coordinates,
)
from ...modules.diffusion.conditioning import condition_pair
from ...modules.embedding import (
    FrozenFourierEmbedding,
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.functional import pair_mask_from_msa_mask, pair_mask_from_token_mask
from ...modules.head import (
    AtomConfidenceProjector,
    DistogramHead,
    expected_plddt,
    tm_bin_weights,
)
from ...modules.head.frames import gather_atoms_by_index, insert_sample_axis
from ...modules.transition import Transition
from ...modules.trunk.msa import (
    MsaUpdateBlock,
    PostProjectionOuterProductMean,
)
from .config import AlphaFold3Config

# Atom-derived token state + target residue type + canonical MSA profile + deletion mean.
_TOKEN_FEATURE_SIZE = 384 + TOKEN_VOCAB_SIZE + MSA_VOCAB_SIZE + 1
_AF3_OFFICIAL_VOCAB_GROUPS = (
    (TOKEN_INDEX["A"],),
    (TOKEN_INDEX["R"],),
    (TOKEN_INDEX["N"],),
    (TOKEN_INDEX["D"], TOKEN_INDEX["B"]),
    (TOKEN_INDEX["C"], TOKEN_INDEX["U"]),
    (TOKEN_INDEX["Q"],),
    (TOKEN_INDEX["E"], TOKEN_INDEX["Z"]),
    (TOKEN_INDEX["G"],),
    (TOKEN_INDEX["H"],),
    (TOKEN_INDEX["I"],),
    (TOKEN_INDEX["L"],),
    (TOKEN_INDEX["K"],),
    (TOKEN_INDEX["M"],),
    (TOKEN_INDEX["F"],),
    (TOKEN_INDEX["P"],),
    (TOKEN_INDEX["S"],),
    (TOKEN_INDEX["T"],),
    (TOKEN_INDEX["W"],),
    (TOKEN_INDEX["Y"],),
    (TOKEN_INDEX["V"],),
    (TOKEN_INDEX["X"], TOKEN_INDEX["J"], TOKEN_INDEX["O"], TOKEN_INDEX["OTHER"]),
    (TOKEN_INDEX["-"],),
    (TOKEN_INDEX["RNA_A"],),
    (TOKEN_INDEX["RNA_G"],),
    (TOKEN_INDEX["RNA_C"],),
    (TOKEN_INDEX["RNA_U"],),
    (TOKEN_INDEX["DNA_A"],),
    (TOKEN_INDEX["DNA_G"],),
    (TOKEN_INDEX["DNA_C"],),
    (TOKEN_INDEX["DNA_T"],),
    (TOKEN_INDEX["RNA_N"], TOKEN_INDEX["DNA_N"]),
)
_AF3_CONDITIONING_TOKEN_FEATURE_SIZE = 384 + 31 + 31 + 1


def _alphafold3_conditioning_token_embedding(token_embedding: Tensor) -> Tensor:
    r"""Collapse the unified token/profile alphabets onto the official 31-class conditioner axis."""

    atom = token_embedding[..., :384]
    token = token_embedding[..., 384 : 384 + TOKEN_VOCAB_SIZE]
    profile = token_embedding[..., 384 + TOKEN_VOCAB_SIZE : -1]

    def collapse(value: Tensor) -> Tensor:
        groups = []
        for indices in _AF3_OFFICIAL_VOCAB_GROUPS:
            group = value[..., indices[0] : indices[0] + 1]
            for index in indices[1:]:
                group = group + value[..., index : index + 1]
            groups.append(group)
        return torch.cat(groups, dim=-1)

    return torch.cat((atom, collapse(token), collapse(profile), token_embedding[..., -1:]), dim=-1)


class AlphaFold3Model(Model):
    r"""
    Predict biomolecular complex structures with AlphaFold 3.

    Training ``forward`` accepts [`AlphaFold3Input`][] and
    [`AlphaFold3StructureTarget`][] and returns [`AlphaFold3Output`][] by
    default, or a minimal [`CoordinateDenoisingOutput`][] for a diffusion-only
    objective. Explicit denoising and sampling return [`AlphaFold3SamplingOutput`][];
    [`AlphaFold3Pipeline`][ofoldx.models.alphafold3.AlphaFold3Pipeline]
    prepares inputs and decodes sampling tensors into public results.

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import PolymerSequence
        >>> from ofoldx.models.alphafold3 import (
        ...     AlphaFold3Config, AlphaFold3Model, AlphaFold3Pipeline, AlphaFold3Processor,
        ... )
        >>> config = AlphaFold3Config({
        ...     "single_size": 16,
        ...     "pair_size": 16,
        ...     "template": {
        ...         "hidden_size": 8,
        ...         "num_blocks": 0,
        ...         "triangle_attention_hidden_size": 4,
        ...         "num_triangle_attention_heads": 1,
        ...     },
        ...     "trunk": {"num_pairformer_blocks": 1},
        ...     "recycling_steps": 1,
        ...     "msa_max_depth": 1,
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 16,
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
        >>> processor = AlphaFold3Processor()
        >>> inputs = processor(PolymerSequence.from_one_letter("A"))
        >>> atom_count = inputs.atom_features.shape[1]
        >>> model = AlphaFold3Model(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.denoise(
        ...         inputs,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
        >>> result = AlphaFold3Pipeline(model=model, processor=processor).decode_fold(inputs, output)
        >>> len(result.predictions)
        1
    """

    config_class = AlphaFold3Config
    config: AlphaFold3Config

    def __init__(self, config: AlphaFold3Config | None = None) -> None:
        config = AlphaFold3Config() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self._msa_max_depth = config.msa_max_depth
        self.input_embeddings = AlphaFold3InputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.single_init_projection = nn.Linear(_TOKEN_FEATURE_SIZE, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(_TOKEN_FEATURE_SIZE, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(_TOKEN_FEATURE_SIZE, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.template_embedder = AlphaFold3TemplateEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.template.hidden_size,
            num_blocks=config.template.num_blocks,
            vocab_size=config.template.vocab_size,
            num_distogram_bins=config.template.num_distogram_bins,
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            expansion=config.template.expansion,
        )
        self.msa_trunk = AlphaFold3MsaTrunk(
            pair_size=config.pair_size,
        )
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pairformer_stack = AlphaFold3PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
        )
        self.coordinate_denoiser = AlphaFold3CoordinateDenoiser(
            num_encoder_blocks=config.denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=config.denoiser.num_token_encoder_attention_heads,
            sigma_data=config.denoiser.sigma_data,
            atom_hidden_size=config.denoiser.atom_hidden_size,
            atom_pair_hidden_size=config.denoiser.atom_pair_hidden_size,
            token_hidden_size=config.denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
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
        self.confidence_head = AlphaFold3ConfidenceHead(
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
            num_blocks=config.confidence.num_blocks,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=config.confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            expansion=config.confidence.expansion,
            single_max_abs=config.confidence.single_max_abs,
        )
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins, bias=False)
        # Atom cross-attention adaptively normalizes queries and keys from the raw hidden state.
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Zero-initialize the recycling and PAE/PDE output projections.

        Returns:
            ``None``.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)

    def forward(
        self,
        input: AlphaFold3Input,
        target: AlphaFold3StructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        compute_auxiliary_heads: bool = True,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        use_conditioning: bool = True,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3Output | CoordinateDenoisingOutput:
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
            use_conditioning: Whether trunk single and pair values condition the
                diffusion head. The atom path always receives the trunk single.
            recycling_steps: Number of complete trunk passes. ``None`` uses
                the configured value.
            generator: Optional generator for MSA realization, rigid augmentation,
                noise levels, and Gaussian coordinate noise.

        Returns:
            The complete [`AlphaFold3Output`][] by default. When
                ``compute_auxiliary_heads=False``, only the denoised coordinates,
                augmented targets, and noise levels required by a diffusion objective
                are returned.

        Raises:
            ValueError: If ``num_samples``, ``sample_chunk_size``, or
                ``recycling_steps`` is not positive.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(input, recycling_steps=recycling_steps, generator=generator)
        target_coordinates, target_mask = _gather_alphafold3_structure_target(target, input)
        target_coordinates = target_coordinates.to(representations.single)
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
        normal = torch.empty_like(noise_template).normal_(generator=generator)
        noise_level = (
            self.coordinate_denoiser.preconditioner.sigma_data * (noise_log_mean + noise_log_std * normal).exp()
        )
        if synchronize_noise_levels:
            noise_level = noise_level.expand(-1, num_samples)
        coordinate_noise = torch.empty_like(augmented_target_coordinates).normal_(generator=generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * coordinate_noise
        denoise_context = self._denoise_context(
            input,
            representations,
            use_conditioning=use_conditioning,
        )
        coordinates = denoise_coordinates(
            self._prepare_denoise(denoise_context),
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
        return AlphaFold3Output(
            single=raw_output.single,
            pair=raw_output.pair,
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
        input: AlphaFold3Input,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        use_conditioning: bool = True,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3SamplingOutput:
        r"""
        Denoise atom coordinates at explicitly supplied EDM noise levels.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            noisy_coordinates: Noisy coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: EDM noise standard deviations with shape ``(B, S)`` in
                angstroms.
            use_conditioning: Whether trunk single and pair values condition the
                diffusion head. The atom path always receives the trunk single.
            recycling_steps: Number of complete trunk passes. ``None`` uses
                the configured value.
            generator: Optional generator for per-recycle MSA realization.

        Returns:
            Raw denoised coordinates, trunk representations, distogram logits,
                and confidence logits. No diffusion sampling is performed.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.
        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        representations = self._run_trunk(input, recycling_steps=recycling_steps, generator=generator)
        denoise_context = self._denoise_context(
            input,
            representations,
            use_conditioning=use_conditioning,
        )
        coordinates = self._prepare_denoise(denoise_context)(
            denoise_context.to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return self._assemble_output(input, representations, coordinates)

    @torch.inference_mode()
    def sample(
        self,
        input: AlphaFold3Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int | None = None,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3SamplingOutput:
        r"""
        Run the recycled trunk, coordinate sampling, and prediction heads.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of coordinate samples. ``None`` uses five.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses
                the configured value.
            generator: Optional generator for MSA realization and coordinate diffusion.

        Returns:
            Final trunk representations, coordinate samples, distogram logits,
                and confidence logits.

        Raises:
            ValueError: If a count is not positive.
        """

        num_samples = 5 if num_samples is None else num_samples
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be positive")
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        representations = self._run_trunk(input, recycling_steps=recycling_steps, generator=generator)
        denoise_context = self._denoise_context(input, representations)
        pred_coords = self.coordinate_sampler(
            self._prepare_denoise(denoise_context),
            context=denoise_context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )
        return self._assemble_output(input, representations, pred_coords)

    def confidence_summary(
        self,
        output: AlphaFold3SamplingOutput,
        input: AlphaFold3Input,
        *,
        max_error_bin: float = 31.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Compute pTM, ipTM, and their weighted average from PAE logits.

        Args:
            output: Predictions whose PAE logits have shape ``(B, S, N, N, D)``.
            input: The corresponding model input. ``token_mask`` and
                ``chain_index`` have shape ``(B, N)``.
            max_error_bin: Upper edge of the resolved PAE bins in angstroms.

        Returns:
            ptm: pTM scores with shape ``(B, S)``.
            iptm: ipTM scores with shape ``(B, S)``.
            ptm_iptm_average: ``0.8 * ipTM + 0.2 * pTM`` with shape ``(B, S)``.
                For single-chain systems, uses pTM in place of undefined ipTM.
        """

        num_tokens = output.pae_logits.shape[-2]
        frame_mask = input.token_mask.to(torch.bool)
        # Broadcast token-axis fields across the per-sample logit axis so the summary is per-sample.
        sample_axes = output.pae_logits.dim() - 3
        chain_index = input.chain_index
        for _ in range(sample_axes - chain_index.dim() + 1):
            chain_index = chain_index.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(1)
        return alphafold3_confidence_summary(
            output.pae_logits,
            frame_mask=frame_mask.expand(*output.pae_logits.shape[:-3], num_tokens),
            chain_index=chain_index.expand(*output.pae_logits.shape[:-3], num_tokens),
            max_error_bin=max_error_bin,
        )

    def _assemble_output(
        self,
        input: AlphaFold3Input,
        representations: _AlphaFold3TrunkOutput,
        coordinates: Tensor,
    ) -> AlphaFold3SamplingOutput:
        r"""
        Compute distogram and confidence predictions.
        """

        token_embedding = representations.token_embedding
        single = representations.single
        pair = representations.pair
        confidence_dtype = self.confidence_head.pair_embedder.left_token_projection.weight.dtype
        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            single.to(dtype=confidence_dtype),
            pair=pair.to(dtype=confidence_dtype),
            token_embedding=token_embedding.to(dtype=confidence_dtype),
            pred_coords=coordinates,
            token_to_rep_atom=input.token_to_rep_atom,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
        )
        with torch.autocast(device_type=pair.device.type, enabled=False):
            distogram_logits = self.distogram_head(pair.to(dtype=self.distogram_head.projection.weight.dtype))
        return AlphaFold3SamplingOutput(
            single=single,
            pair=pair,
            coordinates=coordinates,
            distogram_logits=distogram_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            batch_size=[single.shape[0]],
        )

    def _denoise_context(
        self,
        input: AlphaFold3Input,
        representations: _AlphaFold3TrunkOutput,
        *,
        use_conditioning: bool = True,
    ) -> CoordinateDenoiseContext:
        r"""
        Assemble conditioning shared by ``forward`` and the sampler.
        """

        conditioning_single = representations.single if use_conditioning else 0 * representations.single
        conditioning_pair = representations.pair if use_conditioning else 0 * representations.pair
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=_alphafold3_conditioning_token_embedding(representations.token_embedding),
            single=representations.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=representations.relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                input.reference_positions,
                input.reference_space_id,
                input.atom_mask,
            ),
        )

    def _prepare_denoise(self, context: CoordinateDenoiseContext) -> CoordinateDenoiseFn:
        r"""
        Prepare the FP32 diffusion head independently of trunk autocast.
        """

        with torch.autocast(device_type=context.device.type, enabled=False):
            denoise = self.coordinate_denoiser.prepare(context)

        def denoise_fp32(inputs: CoordinateDenoisingInput) -> Tensor:
            with torch.autocast(device_type=inputs.noisy_coords.device.type, enabled=False):
                return denoise(inputs)

        return denoise_fp32

    def _run_trunk(
        self,
        input: AlphaFold3Input,
        *,
        recycling_steps: int,
        generator: torch.Generator | None,
    ) -> _AlphaFold3TrunkOutput:
        r"""
        Build the token embedding and run the recycled trunk to final single/pair representations.

        Args:
            input: Typed model input.
            recycling_steps: Number of complete trunk passes to run.
            generator: Generator used to realize the MSA independently for each pass.

        Returns:
            Named token embedding, single, pair, and relative-position representations.
        """

        atom_dtype = self.input_embeddings.atom_encoder.atom_conditioner.atom_feature_projection.weight.dtype
        reference_positions = input.reference_positions.to(dtype=atom_dtype)
        token_embedding = self.input_embeddings(
            input.atom_features.to(dtype=atom_dtype),
            token_features=input.token_features,
            reference_positions=reference_positions,
            reference_charges=input.reference_charges.to(dtype=atom_dtype),
            reference_mask=input.reference_mask,
            atom_to_token=input.atom_to_token,
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=self.input_embeddings.build_local_geometry(
                reference_positions,
                input.reference_space_id,
                input.atom_mask,
            ),
        )
        # Per-atom conditioning remains float32 until entering the Evoformer projection dtype.
        token_embedding = token_embedding.to(dtype=self.single_init_projection.weight.dtype)

        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            dtype=single_init.dtype,
        )
        pair_init = self._initial_pair(token_embedding, input, relative_features)

        pair_mask = pair_mask_from_token_mask(input.token_mask)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            recycling_steps=recycling_steps,
            generator=generator,
        )

        return _AlphaFold3TrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
        )

    def _run_recycling(
        self,
        input: AlphaFold3Input,
        token_embedding: Tensor,
        single_init: Tensor,
        pair_init: Tensor,
        pair_mask: Tensor,
        *,
        recycling_steps: int,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run the differentiable recycling loop.
        """

        template_features = input.template_features
        if template_features is not None:
            template_mask = template_features.template_mask
            optional_features = {} if template_mask is None else {"template_mask": template_mask}
            template_features = AlphaFold3TemplateFeatures(
                distogram=template_features.distogram.to(dtype=pair_init.dtype),
                pseudo_beta_mask=template_features.pseudo_beta_mask.to(dtype=pair_init.dtype),
                residue_type_ids=template_features.residue_type_ids,
                unit_vector=template_features.unit_vector.to(dtype=pair_init.dtype),
                backbone_frame_mask=template_features.backbone_frame_mask.to(dtype=pair_init.dtype),
                batch_size=[template_features.distogram.shape[0]],
                **optional_features,
            )
        msa_tokens = input.msa_tokens
        msa_deletion_counts = input.msa_deletion_counts
        msa_mask = input.msa_mask

        # Recycled representations are stored in float32 and cast to the activation
        # dtype at the start of each pass.
        single = torch.zeros_like(single_init, dtype=torch.float32)
        pair = torch.zeros_like(pair_init, dtype=torch.float32)
        for _ in range(recycling_steps):
            pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair.to(dtype=pair_init.dtype)))
            if template_features is not None:
                pair = pair + self.template_embedder(pair, template_features, pair_mask=pair_mask)
            realized_msa_tokens, realized_deletion_counts, realized_msa_mask = _realize_alphafold3_msa(
                msa_tokens,
                msa_deletion_counts,
                msa_mask,
                max_depth=self._msa_max_depth,
                generator=generator,
            )
            realized_deletion_counts = realized_deletion_counts.to(torch.float32)
            realized_msa_continuous_features = torch.stack(
                (
                    (realized_deletion_counts > 0).to(torch.float32),
                    torch.atan(realized_deletion_counts / 3.0) * (2.0 / torch.pi),
                ),
                dim=-1,
            ).to(pair_init.dtype)
            _, pair = self.msa_trunk(
                realized_msa_tokens,
                realized_msa_continuous_features,
                pair,
                token_embedding,
                msa_mask=realized_msa_mask,
                pair_mask=pair_mask,
            )
            single = single_init + self.single_recycle_projection(
                self.single_recycle_norm(single.to(dtype=single_init.dtype))
            )
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
            single = single.float()
            pair = pair.float()

        return single, pair

    def _initial_pair(
        self,
        token_embedding: torch.Tensor,
        input: AlphaFold3Input,
        relative_features: RelativePositionBuckets,
    ) -> torch.Tensor:
        r"""
        Build the initial pair representation from single projections, relative positions, and token bonds.

        Args:
            token_embedding: Initial target-feature representation ``(..., N_token, token_feature_size)``.
            input: Typed model input (provides token bonds and the token mask).
            relative_features: Relative-position bucket features for the pair embedder.

        Returns:
            Masked initial pair representation ``(..., N_token, N_token, pair_size)``.
        """

        left = self.pair_left_projection(token_embedding).unsqueeze(-2)
        right = self.pair_right_projection(token_embedding).unsqueeze(-3)
        token_bonds = input.token_bonds.to(device=token_embedding.device, dtype=token_embedding.dtype)
        token_bonds = token_bonds.unsqueeze(-1)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds)

        pair_mask = pair_mask_from_token_mask(input.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)


class _PreparedAlphaFold3DenoisingContext(NamedTuple):
    single: Tensor
    pair: Tensor
    atom_conditioning: Tensor
    atom_encoder_packed_bias: Tensor
    atom_decoder_packed_bias: Tensor


class AlphaFold3CoordinateDenoiser(nn.Module):
    r"""
    AlphaFold 3 coordinate-denoising composition.
    """

    def __init__(
        self,
        num_encoder_blocks: int = 24,
        num_encoder_attention_heads: int = 16,
        *,
        sigma_data: float = 16.0,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        token_hidden_size: int = 768,
        single_size: int = 384,
        pair_size: int = 128,
        atom_feature_size: int = 385,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = AlphaFold3DenoisingConditioner(
            pair_input_size=pair_size + 139,
            pair_size=pair_size,
            single_conditioning_size=single_size + _AF3_CONDITIONING_TOKEN_FEATURE_SIZE,
            single_size=single_size,
            sigma_data=sigma_data,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            atom_feature_size=atom_feature_size,
            backend=backend,
        )
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.atom_decoder = TokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )

    _gradient_checkpointing_func: Callable[..., Any]

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        r"""Build the local atom windows consumed by this denoiser."""

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)

    def _prepare_context(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> _PreparedAlphaFold3DenoisingContext:
        single, pair = self.conditioner.prepare_context(
            token_embedding=inputs.token_embedding,
            single=inputs.conditioning_single,
            pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
        )

        encoder = self.atom_encoder
        atom_conditioning, atom_pair = encoder.prepare_context(
            inputs.atom_features,
            reference_positions=inputs.reference_positions,
            reference_charges=torch.arcsinh(inputs.reference_charges),
            reference_mask=inputs.reference_mask,
            single=inputs.single,
            pair=pair,
            local_geometry=inputs.local_geometry,
            atom_to_token=inputs.atom_to_token,
        )
        return _PreparedAlphaFold3DenoisingContext(
            single,
            pair,
            atom_conditioning,
            encoder.atom_self_attention.project_pair_bias(atom_pair),
            self.atom_decoder.atom_self_attention.project_pair_bias(atom_pair),
        )

    def _checkpointed_context(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> _PreparedAlphaFold3DenoisingContext:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return self._gradient_checkpointing_func(self._prepare_context, inputs)
        return self._prepare_context(inputs)

    def prepare(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> CoordinateDenoiseFn:
        r"""Prepare one differentiable denoising invocation shared across samples."""

        prepared = self._checkpointed_context(inputs)

        def denoise(step_inputs: CoordinateDenoisingInput) -> Tensor:
            return self._forward_prepared(step_inputs, prepared)

        return denoise

    def _predict_coordinate_update_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedAlphaFold3DenoisingContext,
    ) -> Tensor:
        scaled_coordinates = self.preconditioner.scale_noisy_coords(
            inputs.noisy_coords,
            inputs.noise_level,
        )
        denoising_single = self.conditioner.condition_noise(inputs.noise_level, prepared.single)
        encoder_args = (
            scaled_coordinates,
            prepared.atom_conditioning,
            prepared.atom_encoder_packed_bias,
            inputs.atom_mask.to(dtype=inputs.atom_features.dtype),
            inputs.atom_to_token,
            denoising_single,
            inputs.local_geometry.window_info.key_index,
        )
        atom_tensors = self.atom_encoder.forward_prepared(*encoder_args)
        token_hidden = self.token_refiner(
            atom_tensors[0],
            denoising_single,
            prepared.pair,
        )
        return self.atom_decoder.forward_prepared(
            atom_tensors[1],
            prepared.atom_conditioning,
            prepared.atom_decoder_packed_bias,
            token_hidden,
            inputs.atom_to_token,
            inputs.atom_mask,
            inputs.local_geometry.window_info.key_index,
        )

    def _forward_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedAlphaFold3DenoisingContext,
    ) -> Tensor:
        if inputs.noisy_coords.shape[-1] != 3:
            raise ValueError("noisy coordinates must have three spatial channels")
        coordinate_update = self._predict_coordinate_update_prepared(inputs, prepared)
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )

    def forward(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Denoise atom coordinates with the AlphaFold 3 diffusion-head equation.
        """

        return self._forward_prepared(inputs, self._checkpointed_context(inputs))


class AlphaFold3DenoisingConditioner(nn.Module):
    r"""
    Concatenate raw relative-position buckets into AlphaFold 3 denoising pair conditioning.
    """

    def __init__(
        self,
        pair_input_size: int = 267,
        pair_size: int = 128,
        single_conditioning_size: int = 384 + _AF3_CONDITIONING_TOKEN_FEATURE_SIZE,
        single_size: int = 384,
        noise_embedding_size: int = 256,
        sigma_data: float = 16.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(Transition(pair_size, expansion=2, backend=backend) for _ in range(2))
        self.single_conditioning_norm = LayerNorm(single_conditioning_size, bias=False, backend=backend)
        self.single_conditioning_projection = nn.Linear(single_conditioning_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(noise_embedding_size)
        self.noise_norm = LayerNorm(noise_embedding_size, bias=False, backend=backend)
        self.noise_projection = nn.Linear(noise_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(Transition(single_size, expansion=2, backend=backend) for _ in range(2))

    def forward(
        self,
        noise_level: Tensor,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> tuple[Tensor, Tensor]:
        single, pair = self.prepare_context(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
        )
        return self.condition_noise(noise_level, single), pair

    def condition_noise(self, noise_level: Tensor, single: Tensor) -> Tensor:
        r"""Add sample-specific noise conditioning to a prepared single representation."""

        noise = self.noise_norm(self.noise_embedding(torch.log(noise_level / self.sigma_data) / 4).to(single.dtype))
        single = single.unsqueeze(-3) + self.noise_projection(noise).unsqueeze(-2)
        for transition in self.single_transitions:
            single = single + transition(single)
        return single

    def prepare_context(
        self,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> tuple[Tensor, Tensor]:
        r"""Prepare the single base and pair conditioning shared across samples."""

        pair = condition_pair(
            pair,
            _relative_position_one_hot(relative_features),
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=self.pair_transitions,
        )
        single = self.single_conditioning_projection(
            self.single_conditioning_norm(torch.cat([single, token_embedding], dim=-1))
        )
        return single, pair


class AlphaFold3InputEmbeddings(nn.Module):
    r"""
    Build token input embeddings by encoding atoms to tokens and appending the token-feature tail.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 385,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = AtomToTokenEncoder(
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
        local_geometry: LocalAtomGeometry,
        atom_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        atom_dtype = self.atom_encoder.atom_conditioner.atom_feature_projection.weight.dtype
        # Reference features remain FP32 even when the Evoformer uses a lower precision.
        with torch.autocast(device_type=reference_positions.device.type, enabled=False):
            atom_encoder_output = self.atom_encoder(
                atom_features.to(dtype=atom_dtype),
                reference_positions=reference_positions.to(dtype=atom_dtype),
                reference_charges=torch.arcsinh(reference_charges.to(dtype=atom_dtype)),
                reference_mask=reference_mask,
                atom_mask=atom_mask,
                atom_to_token=atom_to_token,
                token_reference=token_features,
                local_geometry=local_geometry,
            )
        token_single = atom_encoder_output.token_hidden

        token_tail = token_features[..., self.single_size :].to(token_single)
        token_embedding = torch.cat([token_single, token_tail], dim=-1)

        if token_mask is not None:
            token_embedding = token_embedding * token_mask.unsqueeze(-1).to(dtype=token_embedding.dtype)
        return token_embedding

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor,
    ) -> LocalAtomGeometry:
        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)


class AlphaFold3TemplateEmbedder(nn.Module):
    r"""
    Embed AlphaFold 3 template-derived pair features into the trunk pair representation.

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks; ``0`` disables the embedder.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_distogram_bins: Number of distogram bins in the packed template features.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Transition expansion factor inside each triangle update block.
        backend: Backend selector forwarded to the triangle update blocks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 0,
        vocab_size: int = TOKEN_VOCAB_SIZE,
        num_distogram_bins: int = 39,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.template_projection = FactorizedPairLinear(
            pair_size=num_distogram_bins + 5,
            token_size=vocab_size,
            output_size=hidden_size,
            pair_prefix_size=num_distogram_bins + 1,
            bias=False,
            backend=backend,
        )
        self.pair_update_stack = AlphaFold3TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            triangle_multiplication_hidden_size=hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(
        self,
        pair: Tensor,
        templates: AlphaFold3TemplateFeatures,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return template pair updates for an input pair representation.

        Args:
            pair: ``[..., N_token, N_token, pair_size]`` trunk pair representation.
            templates: Pre-packed AlphaFold 3 template features.
            pair_mask: Optional ``[..., N_token, N_token]`` pair validity mask forwarded to the
                triangle update blocks.

        Returns:
            ``[..., N_token, N_token, pair_size]`` template pair update. The
            update is zero when ``num_blocks == 0`` or the template count is zero.
        """

        if self.num_blocks == 0 or templates.distogram.shape[-4] == 0:
            return pair.new_zeros(pair.shape)

        normalized_pair = self.pair_norm(pair)
        template_updates = []
        for template_index in range(templates.distogram.shape[-4]):
            template_pair = self._single_template_update(
                normalized_pair,
                templates=templates,
                template_index=template_index,
                pair_mask=pair_mask,
            )
            if templates.template_mask is not None:
                mask = templates.template_mask[..., template_index].to(
                    device=template_pair.device,
                    dtype=template_pair.dtype,
                )
                template_pair = template_pair * mask.reshape(*mask.shape, 1, 1, 1)
            template_updates.append(template_pair)

        update = torch.stack(template_updates, dim=-4).mean(dim=-4)
        return self.output_projection(self.activation(update))

    def _single_template_update(
        self,
        pair: Tensor,
        *,
        templates: AlphaFold3TemplateFeatures,
        template_index: int,
        pair_mask: Tensor | None,
    ) -> Tensor:
        r"""
        Embed and refine one AlphaFold 3 template in the hidden pair channel.
        """

        geom, residue_one_hot = self._pack_features(templates, template_index)
        template_pair = self.pair_projection(pair)
        template_pair = template_pair + self.template_projection(geom, residue_one_hot)
        template_pair = self.pair_update_stack(template_pair, pair_mask=pair_mask)
        return self.hidden_norm(template_pair)

    def _pack_features(self, templates: AlphaFold3TemplateFeatures, template_index: int) -> tuple[Tensor, Tensor]:
        r"""
        Return geometric pair features and residue one-hot features for one template.
        """

        expected_residue_shape = templates.distogram.shape[:-2]
        if templates.residue_type_ids.shape != expected_residue_shape:
            raise ValueError(
                f"residue_type_ids must have shape {expected_residue_shape}; got {templates.residue_type_ids.shape}"
            )
        residue_type_ids = templates.residue_type_ids[..., template_index, :]
        residue_one_hot = F.one_hot(residue_type_ids, num_classes=self.vocab_size).to(dtype=templates.distogram.dtype)

        geom = torch.cat(
            [
                templates.distogram[..., template_index, :, :, :],
                templates.pseudo_beta_mask[..., template_index, :, :].unsqueeze(-1),
                templates.unit_vector[..., template_index, :, :, :],
                templates.backbone_frame_mask[..., template_index, :, :].unsqueeze(-1),
            ],
            dim=-1,
        )
        return geom, residue_one_hot


class AlphaFold3BinnedDistancePairEmbedder(nn.Module):
    r"""
    Add token features and binned representative-atom distances to an AlphaFold 3 pair state.
    """

    def __init__(
        self,
        pair_size: int = 128,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.left_token_projection = nn.Linear(_TOKEN_FEATURE_SIZE, pair_size, bias=False)
        self.right_token_projection = nn.Linear(_TOKEN_FEATURE_SIZE, pair_size, bias=False)
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
        self.register_buffer("lower_bins", lower_bins, persistent=False)
        self.register_buffer("upper_bins", upper_bins, persistent=False)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] < self.upper_bins)).to(
            pair.dtype
        )
        return pair + self.distance_one_hot_projection(distance_bins)


class AlphaFold3MsaTrunk(HomogeneousBlockRegions, nn.Module):
    r"""
    AlphaFold 3 MSA trunk with homogeneous family-owned joint blocks and a final pair-only update.

    ``num_stages`` counts both the joint MSA/pair blocks and the distinct final pair-only stage.
    AlphaFold 3 runs no structured dropout and refines the pair with its own ending-node
    triangle-bias equation.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_stages: int = 4,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(MSA_VOCAB_SIZE + 2, msa_size, bias=False)
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "eager"),
            public_default="eager",
        )
        self.token_projection = nn.Linear(_TOKEN_FEATURE_SIZE, msa_size, bias=False)
        self.num_stages = num_stages
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            AlphaFold3MsaPairBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_stages - 1)
        )
        self.final_pair_update = (
            AlphaFold3FinalMsaPairUpdate(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                backend=backend,
            )
            if num_stages
            else None
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(
        self,
        msa_tokens: Tensor,
        msa_continuous_features: Tensor,
        pair: Tensor,
        token_embedding: Tensor | None = None,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        msa = dispatch_categorical_msa_projection(
            msa_tokens,
            msa_continuous_features,
            self.msa_projection.weight,
            backend=self.backend,
        )
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask)
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
        if self.final_pair_update is not None:
            pair = self.final_pair_update(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


class AlphaFold3MsaPairBlock(nn.Module):
    r"""
    Joint AlphaFold 3 MSA/pair block with the family-owned triangle-bias equation.

    Uses the joint outer-product/MSA/pair residual order with dropout-free
    AlphaFold 3 updates.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.msa_update = MsaUpdateBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            expansion=expansion,
            dropout=0.0,
            backend=backend,
        )
        self.triangle_pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        msa = self.msa_update(msa, pair, pair_mask=pair_mask)
        return msa, self.triangle_pair_update(pair, pair_mask=pair_mask)


class AlphaFold3FinalMsaPairUpdate(nn.Module):
    r"""
    Final AlphaFold 3 MSA-stage update that refines only the pair representation.

    Uses outer-product mean followed by the dropout-free AlphaFold 3 pair update.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        return self.triangle_pair_update(pair, pair_mask=pair_mask)


class AlphaFold3PairformerStack(HomogeneousBlockRegions, nn.Module):
    r"""
    Homogeneous stack of AlphaFold 3 Pairformer blocks.

    Runs homogeneous family-owned blocks in sequence.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            AlphaFold3PairformerBlock(
                single_size=single_size,
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                num_pair_attention_heads=num_pair_attention_heads,
                expansion=expansion,
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


class AlphaFold3PairformerBlock(nn.Module):
    r"""
    One AlphaFold 3 Pairformer block with the family-owned triangle-bias equation.

    Combines the dropout-free AlphaFold 3 pair update with pair-biased single
    attention and a single transition.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_update = AlphaFold3TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
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
        return single + self.single_transition(single), pair


class AlphaFold3TrianglePairUpdateStack(HomogeneousBlockRegions, nn.Module):
    r"""Homogeneous stack of AlphaFold 3 pair-only triangle update blocks."""

    def __init__(
        self,
        num_blocks: int,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            AlphaFold3TrianglePairUpdateBlock(
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
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


class AlphaFold3TrianglePairUpdateBlock(nn.Module):
    r"""
    AlphaFold 3 pair update with its exact ending-node triangle-bias equation.

    AlphaFold 3 runs the four triangle branches and the pair transition without dropout, and
    biases the ending-node attention from the pre-transpose pair orientation.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
        self.rowwise_dropout = nn.Identity()
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
        self.triangle_attention_ending_node = AlphaFold3EndingNodeTriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, pair_mask=pair_mask)
        pair = pair + self.triangle_attention_starting_node(pair, pair_mask=pair_mask)
        end_pair_mask = None if pair_mask is None else pair_mask.transpose(-1, -2)
        ending_update = self.triangle_attention_ending_node(
            pair.transpose(-3, -2),
            pair_mask=end_pair_mask,
        )
        pair = pair + ending_update.transpose(-3, -2)
        return pair + self.transition(pair)


class AlphaFold3EndingNodeTriangleAttention(TriangleAttention):
    r"""
    AlphaFold 3 ending-node triangle attention with bias from the pre-transpose pair orientation.
    """

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        normalized = self.pair_norm(pair)
        bias_source = normalized.transpose(-3, -2)
        projected_bias = self.bias_projection(bias_source)
        triangle_bias = projected_bias.movedim(-1, -3).unsqueeze(-4)
        return dispatch_triangle_attention(
            normalized,
            triangle_bias,
            pair_mask,
            projection=self.attention.projection,
            backend=self.attention.backend,
        )


class AlphaFold3ConfidenceHead(nn.Module):
    r"""
    AlphaFold 3 confidence-prediction composition.
    """

    def __init__(
        self,
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
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.pair_embedder = AlphaFold3BinnedDistancePairEmbedder(
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.pairformer_stack = AlphaFold3PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            expansion=expansion,
            backend=backend,
        )
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            backend=backend,
        )
        self.pair_projector = AlphaFold3PostProjectionSymmetricPairConfidenceProjector(
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
        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
        representative_coords = gather_atoms_by_index(
            pred_coords,
            insert_sample_axis(token_to_rep_atom, pred_coords.shape[-3]),
        )

        outputs = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_pair = self.pair_embedder(pair, token_embedding, representative_coords_sample)
            sample_single, sample_pair = self.pairformer_stack(single, sample_pair, pair_mask=pair_mask)
            sample_plddt, sample_resolved = self.atom_projector(sample_single, atom_to_token, atom_token_index)
            sample_pae, sample_pde = self.pair_projector(sample_pair)
            outputs.append((sample_plddt, sample_pae, sample_pde, sample_resolved))

        plddt_logits, pae_logits, pde_logits, resolved_logits = zip(*outputs)
        return (
            torch.stack(plddt_logits, dim=-3),
            torch.stack(pae_logits, dim=-4),
            torch.stack(pde_logits, dim=-4),
            torch.stack(resolved_logits, dim=-3),
        )


class AlphaFold3PostProjectionSymmetricPairConfidenceProjector(nn.Module):
    r"""
    Project AlphaFold 3 pair states and symmetrize PDE after nonlinear normalization.
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
        left_distance_logits = self.pde_projection(self.pde_norm(pde_pair))
        return pae_logits, left_distance_logits + left_distance_logits.transpose(-2, -3)


def _relative_position_one_hot(buckets: RelativePositionBuckets) -> Tensor:
    residue = F.one_hot(buckets.residue_bucket.long(), num_classes=buckets.residue_size)
    token = F.one_hot(buckets.token_bucket.long(), num_classes=buckets.residue_size)
    same_entity = buckets.same_entity.unsqueeze(-1)
    chain = F.one_hot(buckets.chain_bucket.long(), num_classes=buckets.chain_size)
    return torch.cat([residue, token, same_entity, chain], dim=-1).to(dtype=buckets.dtype)


def _realize_alphafold3_msa(
    msa_tokens: Tensor,
    deletion_counts: Tensor,
    msa_mask: Tensor,
    *,
    max_depth: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Independently permute valid MSA rows and gather one aligned realization.
    """

    valid_rows = msa_mask.to(torch.bool).any(dim=-1)
    permutation_keys = torch.empty_like(valid_rows, dtype=torch.float32).uniform_(generator=generator)
    permutation = permutation_keys.masked_fill(~valid_rows, -1.0).argsort(dim=-1, descending=True)
    selected_rows = permutation[..., : min(msa_tokens.shape[-2], max_depth)]
    row_indices = selected_rows.unsqueeze(-1)

    return (
        torch.take_along_dim(msa_tokens, row_indices, dim=-2),
        torch.take_along_dim(deletion_counts, row_indices, dim=-2),
        torch.take_along_dim(msa_mask, row_indices, dim=-2),
    )


# ipTM contributes 0.8 and pTM contributes 0.2 to the combined score.
_IPTM_WEIGHT = 0.8


def alphafold3_pae_bin_centers(
    num_bins: int, max_error_bin: float = 31.0, *, device: torch.device | None = None
) -> Tensor:
    r"""
    Return the PAE-bin center distances used to take the expected aligned error.

    The first ``num_bins - 1`` centers evenly tile ``[0, max_error_bin]`` at
    half-step offsets; a final catch-all bin is appended one step beyond.

    Args:
        num_bins: Number of PAE logit bins.
        max_error_bin: Upper edge of the resolved (non-catch-all) bins in angstroms.
        device: Optional device for the returned tensor.

    Returns:
        ``[num_bins]`` bin-center distances in Angstrom.

    Raises:
        ValueError: If ``num_bins`` is less than two.
    """

    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step], dim=0)


def alphafold3_tm_adjusted_pae(
    pae_logits: Tensor,
    *,
    pair_mask: Tensor,
    chain_index: Tensor,
    interface: bool,
    max_error_bin: float = 31.0,
) -> Tensor:
    r"""
    Return the per-token-pair TM-adjusted PAE from PAE logits.

    Softmaxes the logits into a distance-error distribution, weights each bin by its TM-score term
    ``1 / (1 + (d / d0)^2)``, and sums over bins. The TM-score length scale ``d0`` uses the interface
    token count (both chains of the pair) when ``interface`` is set, else the global
    token count. ipTM and pTM therefore differ only in normalization counts.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, use the interface token count for ``d0``; else the global count.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        ``[..., N_token, N_token]`` TM-adjusted PAE.
    """

    num_bins = pae_logits.shape[-1]
    centers = alphafold3_pae_bin_centers(num_bins, max_error_bin, device=pae_logits.device).to(pae_logits.dtype)
    probs = torch.softmax(pae_logits.float(), dim=-1).to(pae_logits.dtype)
    pair_mask = pair_mask.to(pae_logits.dtype)

    same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
    if interface:
        num_chain_tokens = (same_chain.to(pae_logits.dtype) * pair_mask).sum(dim=-1)
        num_tokens = num_chain_tokens.unsqueeze(-1) + num_chain_tokens.unsqueeze(-2)
        # Tokens from one chain contribute once to the interface length.
        num_tokens = num_tokens - same_chain.to(pae_logits.dtype) * torch.floor(num_tokens / 2)
        num_tokens = num_tokens * pair_mask
    else:
        num_tokens = pair_mask.new_full(pair_mask.shape[:-2], float(pair_mask.shape[-1])).unsqueeze(-1).unsqueeze(-1)
        num_tokens = num_tokens.expand_as(pair_mask)

    return (probs * tm_bin_weights(centers, num_tokens)).sum(dim=-1)


def alphafold3_predicted_tm_score(
    adjusted_pae: Tensor, *, pair_mask: Tensor, chain_index: Tensor, interface: bool
) -> Tensor:
    r"""
    Reduce a TM-adjusted PAE matrix to a scalar pTM or ipTM.

    Restricts to cross-chain pairs when ``interface`` is set, row-normalizes the mask, takes the masked
    per-row average of the adjusted PAE, and returns the best row. Returns ``NaN`` when the (interface)
    mask is empty; single-chain systems therefore have no ipTM.

    Args:
        adjusted_pae: ``[..., N_token, N_token]`` TM-adjusted PAE.
        pair_mask: ``[..., N_token, N_token]`` boolean mask of valid frame pairs.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, compute ipTM over cross-chain pairs; else pTM over all pairs.

    Returns:
        ``[...]`` pTM or ipTM score.
    """

    mask = pair_mask.to(adjusted_pae.dtype)
    if interface:
        mask = mask * (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(adjusted_pae.dtype)
    normed = mask / (torch.finfo(mask.dtype).tiny + mask.sum(dim=-1, keepdim=True))
    per_alignment = (adjusted_pae * normed).sum(dim=-1)
    score = per_alignment.max(dim=-1).values
    empty = mask.sum(dim=(-2, -1)) == 0
    return torch.where(empty, torch.full_like(score, float("nan")), score)


def alphafold3_confidence_summary(
    pae_logits: Tensor,
    *,
    frame_mask: Tensor,
    chain_index: Tensor,
    max_error_bin: float = 31.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pTM, ipTM, and the ptm_iptm_average from PAE logits.

    The third element is ``0.8 * ipTM + 0.2 * pTM``, with pTM substituted for
    NaN ipTM on single-chain systems. The final ranking score additionally includes
    DSSP disorder and clash terms at the
    [`AlphaFold3Pipeline.decode_fold`][ofoldx.models.alphafold3.AlphaFold3Pipeline.decode_fold]
    result boundary.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        frame_mask: ``[..., N_token]`` boolean mask of tokens with a valid frame.
        chain_index: ``[..., N_token]`` chain index per token.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        Scalar pTM / ipTM / ptm_iptm_average summaries broadcast over the leading batch/sample axes.
    """

    frame_mask = frame_mask.to(torch.bool)
    pair_mask = frame_mask.unsqueeze(-1) & frame_mask.unsqueeze(-2)
    global_adjusted = alphafold3_tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, chain_index=chain_index, interface=False, max_error_bin=max_error_bin
    )
    interface_adjusted = alphafold3_tm_adjusted_pae(
        pae_logits, pair_mask=pair_mask, chain_index=chain_index, interface=True, max_error_bin=max_error_bin
    )
    ptm = alphafold3_predicted_tm_score(global_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=False)
    iptm = alphafold3_predicted_tm_score(
        interface_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=True
    )
    ptm_iptm_average = torch.where(torch.isnan(iptm), ptm, _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm)
    return ptm, iptm, ptm_iptm_average


# DSSP and geometry terms evaluated after coordinates are decoded.
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALIZATION_WEIGHT = 100.0


def alphafold3_ranking_score(ptm: float, iptm: float, fraction_disordered: float, has_clash: bool) -> float:
    r"""
    Return the AlphaFold 3 ranking score for one sample.

    ``ptm_iptm_average + 0.5 * fraction_disordered - 100 * has_clash``, with pTM
    substituted for the ipTM term when ipTM is ``NaN``.

    Args:
        ptm: The predicted global TM score.
        iptm: The interface predicted TM score (``NaN`` for single-chain systems).
        fraction_disordered: The AlphaFold-RSA disorder fraction in ``[0, 1]``.
        has_clash: Whether the structure has at least one clashing chain.

    Returns:
        The scalar ranking score (higher is better).
    """

    if np.isnan(iptm):
        ptm_iptm_average = ptm
    else:
        ptm_iptm_average = _IPTM_WEIGHT * iptm + (1.0 - _IPTM_WEIGHT) * ptm
    return (
        ptm_iptm_average
        + _FRACTION_DISORDERED_WEIGHT * fraction_disordered
        - _CLASH_PENALIZATION_WEIGHT * float(has_clash)
    )


def _gather_alphafold3_structure_target(
    target: AlphaFold3StructureTarget,
    input: AlphaFold3Input,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather full-system truth onto the current model atom axis.
    """

    atom_index = input.atom_index.long()
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


# ===========================================================================
# Typed tensor and molecular-identity records for AlphaFold 3.
# ===========================================================================


class AlphaFold3TemplateFeatures(_TensorCollection):
    r"""
    Template features consumed by `AlphaFold3TemplateEmbedder`.

    Every tensor carries a ``num_templates`` axis.

    Attributes:
        distogram: ``[..., num_templates, N_token, N_token, num_distogram_bins]`` template pair distogram
            features.
        pseudo_beta_mask: ``[..., num_templates, N_token, N_token]`` template pseudo-beta validity mask.
        residue_type_ids: Integer template residue type ids ``[..., num_templates, N_token]``.
        unit_vector: ``[..., num_templates, N_token, N_token, 3]`` template pair unit-vector features.
        backbone_frame_mask: ``[..., num_templates, N_token, N_token]`` template backbone-frame validity mask.
        template_mask: Optional ``[..., num_templates]`` per-template validity mask.
    """

    distogram: Tensor
    pseudo_beta_mask: Tensor
    residue_type_ids: Tensor
    unit_vector: Tensor
    backbone_frame_mask: Tensor
    template_mask: _OptionalTensor


class _AlphaFold3TrunkOutput(NamedTuple):
    r"""
    Named representations returned by the AlphaFold 3 recycling trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@tensorclass
class AlphaFold3StructureTarget:
    r"""
    Full canonical atom-axis structure truth for AlphaFold 3 training.

    Attributes:
        coordinates: Full-system atom coordinates with shape ``(B, A_system, 3)`` in
            angstroms.
        coordinate_mask: Resolved-coordinate mask with shape ``(B, A_system)``.
    """

    coordinates: Tensor
    coordinate_mask: Tensor


@tensorclass
class AlphaFold3Input:
    r"""
    Tensor and molecular-layout input for AlphaFold 3.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, ``M`` MSA depth, and
    ``T`` template count. ``B`` is explicit in the family tensors; the carrier's
    tensorclass batch size stays empty so the ragged canonical system remains a
    directly nested, recursively movable tensor graph.

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
        token_to_rep_atom: Representative atom index for each token with shape
            ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, C_token)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        msa_tokens: Canonical MSA token IDs with shape ``(B, M, N)``.
        msa_deletion_counts: Raw int32 MSA deletion counts with shape ``(B, M, N)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        template_features: Optional features for ``T`` structural templates.
        system: Molecular system represented by the tensors.
        atom_index: Canonical atom index aligned to the model atom axis, with
            shape ``(B, A)``.
        token_component_index: Canonical component index aligned to tokens, with
            shape ``(B, N)``.
        token_atom_index: Canonical atom index for atom-level tokens, with shape
            ``(B, N)``; ``-1`` marks a residue/component token.
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
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
    template_features: AlphaFold3TemplateFeatures | None = None


class AlphaFold3SamplingOutput(_TensorCollection):
    r"""
    Predictions from one AlphaFold 3 denoising or sampling call.

    Attributes:
        single: Final trunk single representation with shape ``(B, N, 384)``.
        pair: Final trunk pair representation with shape ``(B, N, N, 128)``.
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

    single: Tensor
    pair: Tensor
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


class AlphaFold3Output(AlphaFold3SamplingOutput):
    r"""
    Predictions from one AlphaFold 3 training denoising pass.

    Attributes:
        augmented_target_coordinates: Training targets after rigid augmentation,
            with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Training noise standard deviations with shape ``(B, S)`` in
            angstroms.
    """

    augmented_target_coordinates: Tensor
    noise_level: Tensor
