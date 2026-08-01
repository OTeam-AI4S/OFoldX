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
AlphaFold 3 top-level model.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...compilation import HomogeneousBlockRegions
from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...metrics import plddt_from_logits, predicted_tm_score, tm_weight_from_pae
from ...model import Model, TensorCollection
from ...modules.atom.encoders import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenEncoder,
    CoordinateAtomToTokenEncoder,
    TokenToAtomDecoder,
    TrunkToAtomConditioning,
)
from ...modules.atom.window import ClampedLocalAtomWindowing, LocalAtomGeometry
from ...modules.attention import PerBlockPairBiasProjection
from ...modules.backend import LayerNorm
from ...modules.backend import categorical_msa_projection as dispatch_categorical_msa_projection
from ...modules.backend import resolve_config_backend
from ...modules.backend.normalization import layer_norm_projection
from ...modules.denoiser import DenoisingTokenRefiner
from ...modules.denoiser.coordinate import DenoisingConditioner
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoiseFn,
    CoordinateDenoisingInput,
    CoordinateDenoisingOutput,
    CoordinateEdmSampler,
    EdmPreconditioner,
    EdmScheduler,
    HeunCoordinateEdmSampler,
    denoise_coordinates,
)
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.functional import pair_mask_from_msa_mask, pair_mask_from_token_mask
from ...modules.head import ConfidenceHead, DistogramHead
from ...modules.msa import MsaUpdateBlock, PostProjectionOuterProductMean
from ...modules.pair import TrianglePairUpdateBlock, TrianglePairUpdateStack
from ...modules.pairformer import PairformerBlock, PairformerStack
from ...modules.transformer import AtomCrossAttentionTransformerEncoderBlock
from ..processing import (
    gather_model_coordinates,
    project_model_categories,
    project_model_coordinates,
    project_model_counts,
    project_model_features,
)
from ..vocabulary import map_token_feature_planes
from .config import AlphaFold3Config

# Atom-derived token state + target residue type + MSA profile + deletion mean, both planes on the 72 canonical classes.
_TOKEN_FEATURE_SIZE = 384 + 2 * len(TOKENS) + 1

#: The upstream 31-class AlphaFold 3 residue alphabet (target feature, profile, MSA and template rows) in canonical
#: spelling; upstream's one unknown nucleic row ``N`` is ``RNA_N``.
ALPHAFOLD3_SOURCE_TOKENS: tuple[str, ...] = (
    *"ARNDCQEGHILKMFPSTWYV",
    "X",
    "-",
    *(f"RNA_{x}" for x in "AGCU"),
    *(f"DNA_{x}" for x in "AGCT"),
    "RNA_N",
)
#: Canonical classes upstream reads as another of its rows (its letter tables, ``data/msa_features.py``): B as D, Z as
#: E, U as C, J and O as X, every per-atom and ligand token (``OTHER``) as UNK, and every other nucleic class as N.
ALPHAFOLD3_SOURCE_ALIASES: dict[str, str] = {
    "B": "D",
    "Z": "E",
    "U": "C",
    "J": "X",
    "O": "X",
    "OTHER": "X",
    **{
        token: "RNA_N"
        for token in TOKENS
        if token.startswith(("RNA_", "DNA_")) and token not in ALPHAFOLD3_SOURCE_TOKENS
    },
}
_AF3_CONDITIONING_TOKEN_FEATURE_SIZE = 384 + 2 * len(ALPHAFOLD3_SOURCE_TOKENS) + 1


def alphafold3_source_map() -> Tensor:
    r"""
    ``[31, 72]``: per canonical class (column), the one-hot of the upstream row it is (``ALPHAFOLD3_SOURCE_TOKENS``) or
    is read as (``ALPHAFOLD3_SOURCE_ALIASES``); specials and shared symbols other than the gap read zero.
    """

    index = {token: row for row, token in enumerate(ALPHAFOLD3_SOURCE_TOKENS)}
    mapping = torch.zeros(len(ALPHAFOLD3_SOURCE_TOKENS), len(TOKENS))
    for column, token in enumerate(TOKENS):
        row = index.get(ALPHAFOLD3_SOURCE_ALIASES.get(token, token))
        if row is not None:
            mapping[row, column] = 1.0
    return mapping


_SOURCE_MAP = alphafold3_source_map()


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
    """

    config_class = AlphaFold3Config
    config: AlphaFold3Config

    def __init__(self, config: AlphaFold3Config | None = None) -> None:
        config = AlphaFold3Config() if config is None else config
        super().__init__(config)
        self._num_recycles = config.num_recycles
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
            triangle_attention_head_size=config.template.triangle_attention_head_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            expansion=config.template.expansion,
        )
        self.msa_trunk = AlphaFold3MsaTrunk(
            pair_size=config.pair_size,
        )
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pairformer_stack = PairformerStack(
            dropout=0.0,
            block=partial(PairformerBlock, ending_bias_pair_dims=(-2, -3)),
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
        self.confidence_head = ConfidenceHead(
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
            dropout=0.0,
            block=partial(PairformerBlock, ending_bias_pair_dims=(-2, -3)),
            num_blocks=config.confidence.num_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_head_size=config.confidence.triangle_attention_head_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            expansion=config.confidence.expansion,
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
        num_recycles: int | None = None,
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
            num_recycles: Number of complete trunk passes. ``None`` uses
                the configured value.
            generator: Optional generator for MSA realization, rigid augmentation,
                noise levels, and Gaussian coordinate noise.

        Returns:
            The complete [`AlphaFold3Output`][] by default. When
                ``compute_auxiliary_heads=False``, only the denoised coordinates,
                augmented targets, and noise levels required by a diffusion objective
                are returned.

        Raises:
            InvalidRequestError: If ``num_samples`` or ``num_recycles`` is not positive.
        """
        input = normalize_alphafold3_input(
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
        representations = self._run_trunk(input, num_recycles=num_recycles, generator=generator)
        target_coordinates, target_mask = _gather_alphafold3_structure_target(target, input)
        target_coordinates = project_model_coordinates(
            target_coordinates, target_mask, dtype=representations.single.dtype, feature="AlphaFold3 training target"
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
            self.coordinate_denoiser.prepare(denoise_context),
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
        num_recycles: int | None = None,
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
            num_recycles: Number of complete trunk passes. ``None`` uses
                the configured value.
            generator: Optional generator for per-recycle MSA realization.

        Returns:
            Raw denoised coordinates, trunk representations, distogram logits,
                and confidence logits. No diffusion sampling is performed.

        Raises:
            InvalidRequestError: If ``num_recycles`` is not positive.
        """
        input = normalize_alphafold3_input(
            input,
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            max_atoms_per_token=self.confidence_head.atom_projector.plddt_weight.shape[0],
        )

        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        representations = self._run_trunk(input, num_recycles=num_recycles, generator=generator)
        denoise_context = self._denoise_context(
            input,
            representations,
            use_conditioning=use_conditioning,
        )
        coordinates = self.coordinate_denoiser.prepare(denoise_context)(
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
        num_recycles: int | None = None,
        num_samples: int | None = None,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3SamplingOutput:
        r"""
        Run the recycled trunk, coordinate sampling, and prediction heads.

        Args:
            input: Model features with batch, token, and atom axes ``(B, N, A)``.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of coordinate samples. ``None`` uses five.
            num_sampling_steps: Number of diffusion transitions. ``None`` uses
                the configured value.
            generator: Optional generator for MSA realization and coordinate diffusion.

        Returns:
            Final trunk representations, coordinate samples, distogram logits,
                and confidence logits.

        Raises:
            InvalidRequestError: If a count is not positive.
        """
        input = normalize_alphafold3_input(
            input,
            dtype=self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype,
            max_atoms_per_token=self.confidence_head.atom_projector.plddt_weight.shape[0],
        )

        num_samples = 5 if num_samples is None else num_samples
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if num_sampling_steps < 1:
            fail("value_invalid", RequestRef("num_sampling_steps", str(num_sampling_steps)))
        representations = self._run_trunk(input, num_recycles=num_recycles, generator=generator)
        denoise_context = self._denoise_context(input, representations)
        pred_coords = self.coordinate_sampler(
            self.coordinate_denoiser.prepare(denoise_context),
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

        frame_mask = input.frame_mask.to(torch.bool)
        token_mask = input.token_mask.to(torch.bool)
        # Broadcast token-axis fields across the per-sample logit axis so the summary is per-sample.
        sample_axes = output.pae_logits.dim() - 3
        chain_index = input.chain_index
        for _ in range(sample_axes - chain_index.dim() + 1):
            chain_index = chain_index.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(1)
            token_mask = token_mask.unsqueeze(1)
        return alphafold3_confidence_summary(
            output.pae_logits,
            frame_mask=frame_mask,
            token_mask=token_mask,
            chain_index=chain_index,
            max_error_bin=max_error_bin,
            backend=self.backend,
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
        confidence_dtype = self.confidence_head.pair_embedder.pair_left_projection.weight.dtype
        plddt_logits, pae_logits, pde_logits, resolved_logits = self.confidence_head(
            single.to(dtype=confidence_dtype),
            pair=pair.to(dtype=confidence_dtype),
            token_embedding=token_embedding.to(dtype=confidence_dtype),
            pred_coords=coordinates,
            token_representative_atom_index=input.token_representative_atom_index,
            atom_to_token=input.atom_to_token,
            atom_token_index=input.atom_token_index,
            pairformer_stack=self.confidence_pairformer_stack,
        )
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
            token_embedding=map_token_feature_planes(representations.token_embedding, _SOURCE_MAP, 384),
            single=representations.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=representations.relative_features,
            local_geometry=self.coordinate_denoiser.windowing(
                input.reference_positions,
                input.reference_space_id,
                input.atom_mask,
                input.reference_mask,
            ),
        )

    def _run_trunk(
        self,
        input: AlphaFold3Input,
        *,
        num_recycles: int,
        generator: torch.Generator | None,
    ) -> _AlphaFold3TrunkOutput:
        r"""
        Build the token embedding and run the recycled trunk to final single/pair representations.

        Args:
            input: Typed model input.
            num_recycles: Number of complete trunk passes to run.
            generator: Generator used to realize the MSA independently for each pass.

        Returns:
            Named token embedding, single, pair, and relative-position representations.
        """

        atom_dtype = self.input_embeddings.atom_conditioner.atom_feature_projection.weight.dtype
        reference_positions = input.reference_positions
        token_embedding = self.input_embeddings(
            input.atom_features.to(dtype=atom_dtype),
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

        pair_mask = pair_mask_from_token_mask(input.token_mask, backend=self.backend)
        single, pair = self._run_recycling(
            input,
            token_embedding,
            single_init,
            pair_init,
            pair_mask,
            num_recycles=num_recycles,
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
        num_recycles: int,
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

        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for _ in range(num_recycles):
            pair = pair_init + layer_norm_projection(
                pair.to(dtype=pair_init.dtype), norm=self.pair_recycle_norm, projection=self.pair_recycle_projection
            )
            if template_features is not None:
                pair = pair + self.template_embedder(
                    pair,
                    template_features,
                    same_chain_mask=input.chain_index[..., :, None] == input.chain_index[..., None, :],
                    pair_mask=pair_mask,
                )
            realized_msa_tokens, realized_deletion_counts, realized_msa_mask = _realize_alphafold3_msa(
                msa_tokens,
                msa_deletion_counts,
                msa_mask,
                max_depth=self._msa_max_depth,
                generator=generator,
            )
            realized_deletion_counts = realized_deletion_counts.to(torch.float64)
            realized_msa_continuous_features = torch.stack(
                (
                    (realized_deletion_counts > 0).to(pair_init.dtype),
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
            single = single_init + layer_norm_projection(
                single.to(dtype=single_init.dtype),
                norm=self.single_recycle_norm,
                projection=self.single_recycle_projection,
            )
            single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
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

        pair_mask = pair_mask_from_token_mask(input.token_mask, backend=self.backend)
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
        self.conditioner = DenoisingConditioner(
            pair_input_size=pair_size + 139,
            relative_position_embedder=_relative_position_one_hot,
            pair_size=pair_size,
            single_conditioning_size=single_size + _AF3_CONDITIONING_TOKEN_FEATURE_SIZE,
            single_size=single_size,
            backend=backend,
        )
        self.windowing = ClampedLocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_conditioner = AtomConditioner(atom_feature_size, atom_hidden_size)
        self.trunk_conditioning = TrunkToAtomConditioning(
            single_size,
            pair_size,
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
            backend=backend,
        )
        self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
        self.atom_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            bias=False,
            backend=backend,
        )
        self.atom_decoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            bias=False,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            atom_hidden_size,
            token_hidden_size,
            num_atom_encoder_blocks,
            num_atom_encoder_attention_heads,
            backend,
            block=AtomCrossAttentionTransformerEncoderBlock,
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
            token_hidden_size,
            atom_hidden_size,
            num_atom_decoder_blocks,
            num_atom_decoder_attention_heads,
            backend,
            block=AtomCrossAttentionTransformerEncoderBlock,
            bias=False,
        )

    _gradient_checkpointing_func: Callable[..., Any]

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
        atom_conditioning = self.atom_conditioner(
            inputs.atom_features,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
        )
        atom_conditioning, local_pair = self.trunk_conditioning(
            atom_conditioning,
            inputs.single,
            pair,
            inputs.atom_to_token,
            inputs.local_geometry.window_info,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, inputs.local_geometry, local_pair)
        return _PreparedAlphaFold3DenoisingContext(
            single,
            pair,
            atom_conditioning,
            self.atom_encoder_pair_bias_projection(atom_pair),
            self.atom_decoder_pair_bias_projection(atom_pair),
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
        r"""Prepare shared conditioning and return denoising with owned output storage."""

        inputs = _normalize_alphafold3_denoising(
            inputs, dtype=self.atom_conditioner.atom_feature_projection.weight.dtype
        )
        prepared = self._checkpointed_context(inputs)

        def denoise(step_inputs: CoordinateDenoisingInput) -> Tensor:
            return self(step_inputs, prepared=prepared).clone()

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
        denoising_single = self.conditioner.condition_noise(
            self.preconditioner.c_noise(inputs.noise_level), prepared.single
        )
        window_info = inputs.local_geometry.window_info
        token_hidden, atom_query = self.atom_encoder(
            scaled_coordinates,
            prepared.atom_conditioning,
            prepared.atom_conditioning,
            prepared.atom_encoder_packed_bias,
            inputs.atom_mask,
            inputs.atom_to_token,
            denoising_single,
            window_info,
        )
        token_hidden = self.token_refiner(token_hidden, denoising_single, prepared.pair)
        return self.atom_decoder(
            token_hidden,
            atom_query,
            prepared.atom_conditioning,
            prepared.atom_decoder_packed_bias,
            inputs.atom_mask,
            inputs.atom_to_token,
            window_info,
        )

    def _forward_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedAlphaFold3DenoisingContext,
    ) -> Tensor:
        inputs = copy.copy(inputs)
        inputs.noisy_coords = project_model_coordinates(
            inputs.noisy_coords,
            inputs.atom_mask.to(torch.bool).unsqueeze(-2)
            & torch.ones_like(inputs.noisy_coords[..., 0], dtype=torch.bool),
            dtype=self.atom_conditioner.atom_feature_projection.weight.dtype,
            feature="AlphaFold3 noisy coordinates",
        )
        coordinate_update = self._predict_coordinate_update_prepared(inputs, prepared)
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )

    def forward(
        self,
        inputs: CoordinateDenoisingInput,
        *,
        prepared: _PreparedAlphaFold3DenoisingContext | None = None,
    ) -> Tensor:
        r"""
        Denoise atom coordinates with the AlphaFold 3 diffusion-head equation.

        Args:
            inputs: Coordinates, noise levels and their conditioning tensors.
            prepared: Invocation-local conditioning used by the callable returned
                by ``prepare``. ``None`` derives conditioning from ``inputs``.
        """

        if prepared is None:
            inputs = cast(
                CoordinateDenoisingInput,
                _normalize_alphafold3_denoising(
                    inputs, dtype=self.atom_conditioner.atom_feature_projection.weight.dtype
                ),
            )
            prepared = self._checkpointed_context(inputs)
        return self._forward_prepared(inputs, prepared)


class AlphaFold3InputEmbeddings(nn.Module):
    r"""
    Build token input embeddings by encoding atoms to tokens and appending the token-feature tail.

    The clamped [`windowing`][ofoldx.modules.atom.window.ClampedLocalAtomWindowing] builds the local atom
    geometry the embeddings consume. The reference atoms are conditioned, their local pairs encoded and
    projected to every block's attention bias, and the attended atoms mean-pooled to tokens.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 385,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.windowing = ClampedLocalAtomWindowing(32, 128)
        self.atom_conditioner = AtomConditioner(atom_feature_size, 128)
        self.atom_pair_encoder = AtomPairEncoder(128, 16)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=16,
            num_blocks=3,
            num_attention_heads=4,
            bias=False,
        )
        self.atom_encoder = AtomToTokenEncoder(128, single_size, 3, 4, block=AtomCrossAttentionTransformerEncoderBlock)

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
            feature="AlphaFold3 embedding references",
        )
        atom_features = project_model_features(
            atom_features,
            atom_mask.to(torch.bool)[..., None],
            dtype=atom_dtype,
            feature="AlphaFold3 embedding atom features",
        )
        reference_charges = project_model_features(
            reference_charges,
            atom_mask.to(torch.bool)[..., None],
            dtype=atom_dtype,
            feature="AlphaFold3 embedding charges",
        )
        token_features = project_model_features(
            token_features,
            (
                torch.ones_like(token_features[..., 0], dtype=torch.bool)
                if token_mask is None
                else token_mask.to(torch.bool)
            )[..., None],
            dtype=atom_dtype,
            feature="AlphaFold3 embedding token features",
        )
        atom_conditioning = self.atom_conditioner(
            atom_features.to(dtype=atom_dtype),
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


class AlphaFold3TemplateEmbedder(nn.Module):
    r"""
    Embed AlphaFold 3 template-derived pair features into the trunk pair representation.

    Args:
        pair_size: Width of the trunk pair representation this update is added into.
        hidden_size: Width of the internal template pair channel.
        num_blocks: Number of pair-only triangle update blocks; ``0`` disables the embedder.
        vocab_size: Residue-type vocabulary size (one block per side of the pair).
        num_distogram_bins: Number of distogram bins in the packed template features.
        triangle_attention_head_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Transition expansion factor inside each triangle update block.
        backend: Backend selector forwarded to the triangle update blocks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 0,
        vocab_size: int = len(TOKENS),
        num_distogram_bins: int = 39,
        triangle_attention_head_size: int = 32,
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
        self.pair_update_stack = TrianglePairUpdateStack(
            dropout=0.0,
            block=partial(TrianglePairUpdateBlock, ending_bias_pair_dims=(-2, -3)),
            num_blocks=num_blocks,
            pair_size=hidden_size,
            triangle_multiplication_hidden_size=hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
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
        *,
        same_chain_mask: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return template pair updates for an input pair representation.

        Args:
            pair: ``[..., N_token, N_token, pair_size]`` trunk pair representation.
            templates: Pre-packed AlphaFold 3 template features.
            same_chain_mask: Architectural same-target-instance pair restriction.
            pair_mask: Optional ``[..., N_token, N_token]`` pair validity mask forwarded to the
                triangle update blocks.

        Returns:
            ``[..., N_token, N_token, pair_size]`` template pair update. The
            update is zero when ``num_blocks == 0`` or the template count is zero.
        """
        templates = _project_alphafold3_templates(
            templates, dtype=pair.dtype, pair_mask=pair_mask, vocab_size=self.vocab_size
        )

        if self.num_blocks == 0 or templates.distogram.shape[-4] == 0:
            return pair.new_zeros(pair.shape)

        projected_pair = layer_norm_projection(pair, norm=self.pair_norm, projection=self.pair_projection)
        geometry_mask = same_chain_mask.bool()
        if pair_mask is not None:
            geometry_mask = geometry_mask & pair_mask.bool()
        template_updates = []
        for template_index in range(templates.distogram.shape[-4]):
            template_pair = self._single_template_update(
                projected_pair,
                templates=templates,
                template_index=template_index,
                pair_mask=pair_mask,
                geometry_mask=geometry_mask,
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
        projected_pair: Tensor,
        *,
        templates: AlphaFold3TemplateFeatures,
        template_index: int,
        pair_mask: Tensor | None,
        geometry_mask: Tensor,
    ) -> Tensor:
        r"""
        Embed and refine one AlphaFold 3 template in the hidden pair channel.
        """

        geom, residue_one_hot = self._pack_features(templates, template_index)
        geom = geom.masked_fill(~geometry_mask.unsqueeze(-1), 0)
        template_pair = projected_pair + self.template_projection(geom, residue_one_hot)
        template_pair = self.pair_update_stack(template_pair, pair_mask=pair_mask)
        return self.hidden_norm(template_pair)

    def _pack_features(self, templates: AlphaFold3TemplateFeatures, template_index: int) -> tuple[Tensor, Tensor]:
        r"""
        Return geometric pair features and residue one-hot features for one template.
        """

        expected_residue_shape = templates.distogram.shape[:-2]
        if templates.residue_type_ids.shape != expected_residue_shape:
            fail("value_invalid", RequestRef("residue_type_ids", str(tuple(templates.residue_type_ids.shape))))
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
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(len(TOKENS) + 2, msa_size, bias=False)
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
                triangle_attention_head_size=triangle_attention_head_size,
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
                triangle_attention_head_size=triangle_attention_head_size,
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
        available = torch.ones_like(msa_tokens, dtype=torch.bool) if msa_mask is None else msa_mask.bool()
        msa_tokens = project_model_categories(
            msa_tokens, available, num_classes=self.msa_projection.in_features - 2, feature="AlphaFold3 MSA tokens"
        )
        msa_continuous_features = project_model_features(
            msa_continuous_features, available[..., None], dtype=pair.dtype, feature="AlphaFold3 MSA features"
        )
        msa = dispatch_categorical_msa_projection(
            msa_tokens,
            msa_continuous_features,
            self.msa_projection.weight,
            backend=self.backend,
        )
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask, backend=self.backend)
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
        triangle_attention_head_size: int = 32,
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
        self.triangle_pair_update = TrianglePairUpdateBlock(
            dropout=0.0,
            ending_bias_pair_dims=(-2, -3),
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
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
        triangle_attention_head_size: int = 32,
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
        self.triangle_pair_update = TrianglePairUpdateBlock(
            dropout=0.0,
            ending_bias_pair_dims=(-2, -3),
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
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
        InvalidRequestError: If ``num_bins`` is less than three.
    """

    if num_bins < 3:
        fail("value_invalid", RequestRef("num_bins", str(num_bins)))
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step], dim=0)


def alphafold3_tm_adjusted_pae(
    pae_logits: Tensor,
    *,
    token_mask: Tensor,
    chain_index: Tensor,
    interface: bool,
    max_error_bin: float = 31.0,
) -> Tensor:
    r"""
    Return the per-token-pair TM-adjusted PAE from PAE logits.

    Softmaxes the logits into a distance-error distribution, weights each bin by its TM-score term
    ``1 / (1 + (d / d0)^2)``, and sums over bins. The TM-score length scale ``d0`` uses the interface
    token count (both chains of the pair) when ``interface`` is set, else the global
    token count. Global and chain-union calibrations differ only in their actual
    token counts; selecting cross-chain scoring targets is a separate reduction.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        token_mask: ``[..., N_token]`` boolean mask of present tokens, including frameless points.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, use the interface token count for ``d0``; else the global count.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        ``[..., N_token, N_token]`` TM-adjusted PAE.
    """

    num_bins = pae_logits.shape[-1]
    centers = alphafold3_pae_bin_centers(num_bins, max_error_bin, device=pae_logits.device).to(pae_logits.dtype)
    token_mask = token_mask.to(torch.bool)
    same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
    if interface:
        num_chain_tokens = (same_chain & token_mask.unsqueeze(-2)).sum(dim=-1)
        num_tokens = num_chain_tokens.unsqueeze(-1) + num_chain_tokens.unsqueeze(-2)
        num_tokens = num_tokens - torch.where(same_chain, num_chain_tokens.unsqueeze(-1), 0)
        num_tokens = num_tokens * (token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2))
    else:
        num_tokens = token_mask.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
    return tm_weight_from_pae(pae_logits, num_tokens.to(pae_logits.dtype), bin_centers=centers)


def alphafold3_predicted_tm_score(
    adjusted_pae: Tensor,
    *,
    pair_mask: Tensor,
    chain_index: Tensor,
    interface: bool,
    backend: str = "auto",
) -> Tensor:
    r"""
    Reduce a TM-adjusted PAE matrix to a scalar pTM or ipTM.

    Restricts to cross-chain pairs when ``interface`` is set, row-normalizes the mask, takes the masked
    per-row average of the adjusted PAE, and returns the best row. Returns ``NaN`` when the (interface)
    mask is empty; single-chain systems therefore have no ipTM.

    Args:
        adjusted_pae: ``[..., N_token, N_token]`` TM-adjusted PAE.
        pair_mask: ``[..., N_token, N_token]`` eligibility of source frames and target points.
        chain_index: ``[..., N_token]`` chain index per token.
        interface: If ``True``, compute ipTM over cross-chain pairs; else pTM over all pairs.

    Returns:
        ``[...]`` pTM or ipTM score.
    """

    mask = pair_mask.to(adjusted_pae.dtype)
    if interface:
        mask = mask * (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(adjusted_pae.dtype)
    score = predicted_tm_score(adjusted_pae, pair_weights=mask, backend=backend)
    empty = mask.sum(dim=(-2, -1)) == 0
    return torch.where(empty, torch.full_like(score, float("nan")), score)


def alphafold3_confidence_summary(
    pae_logits: Tensor,
    *,
    frame_mask: Tensor,
    token_mask: Tensor,
    chain_index: Tensor,
    max_error_bin: float = 31.0,
    backend: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pTM, ipTM, and the ptm_iptm_average from PAE logits.

    The third element is ``0.8 * ipTM + 0.2 * pTM``, with pTM substituted for
    NaN ipTM on single-chain systems. The final ranking score additionally includes
    DSSP disorder and clash terms at the
    [`AlphaFold3Pipeline.postprocess`][ofoldx.models.alphafold3.AlphaFold3Pipeline.postprocess]
    result boundary.

    Args:
        pae_logits: ``[..., N_token, N_token, num_bins]`` predicted-aligned-error logits.
        frame_mask: ``[..., N_token]`` boolean mask of tokens with a valid frame.
        token_mask: ``[..., N_token]`` boolean mask of present target points and counted tokens.
        chain_index: ``[..., N_token]`` chain index per token.
        max_error_bin: Upper edge of the resolved PAE bins in angstroms.

    Returns:
        Scalar pTM / ipTM / ptm_iptm_average summaries broadcast over the leading batch/sample axes.
    """

    frame_mask = frame_mask.to(torch.bool)
    token_mask = token_mask.to(torch.bool)
    pair_mask = (frame_mask & token_mask).unsqueeze(-1) & token_mask.unsqueeze(-2)
    pae_logits = torch.where(pair_mask.unsqueeze(-1), pae_logits, 0)
    global_adjusted = alphafold3_tm_adjusted_pae(
        pae_logits, token_mask=token_mask, chain_index=chain_index, interface=False, max_error_bin=max_error_bin
    )
    ptm = alphafold3_predicted_tm_score(
        global_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=False, backend=backend
    )
    iptm = alphafold3_predicted_tm_score(
        global_adjusted, pair_mask=pair_mask, chain_index=chain_index, interface=True, backend=backend
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

    return gather_model_coordinates(
        target.coordinates,
        target.coordinate_mask,
        input.atom_index,
        input.atom_mask,
    )


# ===========================================================================
# Typed tensor and molecular-identity records for AlphaFold 3.
# ===========================================================================


class AlphaFold3TemplateFeatures(TensorClass):
    r"""
    Template features consumed by `AlphaFold3TemplateEmbedder`.

    Every tensor carries a ``num_templates`` axis.

    Attributes:
        distogram: ``[..., num_templates, N_token, N_token, num_distogram_bins]`` template pair distogram
            features.
        pseudo_beta_mask: ``[..., num_templates, N_token, N_token]`` template
            pseudo-beta pair validity, including physical placement frames.
        residue_type_ids: Integer template residue type ids ``[..., num_templates, N_token]``.
        unit_vector: ``[..., num_templates, N_token, N_token, 3]`` template pair unit-vector features.
        backbone_frame_mask: ``[..., num_templates, N_token, N_token]`` template
            backbone-frame pair validity, including physical placement frames.
        template_mask: Optional ``[..., num_templates]`` per-template validity mask.
    """

    distogram: Tensor
    pseudo_beta_mask: Tensor
    residue_type_ids: Tensor
    unit_vector: Tensor
    backbone_frame_mask: Tensor
    template_mask: Tensor | None = None


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
    Tensor input for AlphaFold 3 on the prepared slot axis.

    ``B`` is batch size, ``N`` token count, ``A`` atom (slot) count, ``M`` MSA depth and ``T`` template count. ``B`` is
    explicit in the family tensors; the carrier's tensorclass batch size stays empty.

    Attributes:
        token_mask: Valid-token mask with shape ``(B, N)``.
        frame_mask: Reference-based confidence-frame eligibility with shape ``(B, N)``.
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
        token_features: Per-token input features with shape ``(B, N, C_token)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        msa_tokens: Canonical MSA classes with shape ``(B, M, N)``.
        msa_deletion_counts: Raw int32 MSA deletion counts with shape ``(B, M, N)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        template_features: Optional features for ``T`` structural templates.
        atom_index: The canonical target atom of every slot, with shape ``(B, A)`` (the training gather).
    """

    token_mask: Tensor
    frame_mask: Tensor
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
    token_features: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    token_bonds: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    atom_index: Tensor
    template_features: AlphaFold3TemplateFeatures | None = None


def _project_alphafold3_templates(
    templates: AlphaFold3TemplateFeatures,
    *,
    dtype: torch.dtype | None = None,
    pair_mask: Tensor | None = None,
    token_mask: Tensor | None = None,
    vocab_size: int = len(TOKENS),
) -> AlphaFold3TemplateFeatures:
    """Exclude unavailable derived template payload before numerical lowering."""
    template = cast(AlphaFold3TemplateFeatures, cast(TensorClass, templates).clone(recurse=False))
    pair_valid = template.pseudo_beta_mask.to(torch.bool)
    frame_valid = template.backbone_frame_mask.to(torch.bool)
    if pair_mask is not None:
        pair_valid = pair_valid & pair_mask.to(torch.bool).unsqueeze(-3)
        frame_valid = frame_valid & pair_mask.to(torch.bool).unsqueeze(-3)
    template_valid = template.template_mask
    categories_valid = torch.ones_like(template.residue_type_ids, dtype=torch.bool)
    if template_valid is not None:
        categories_valid = categories_valid & template_valid.bool()[..., None]
    if token_mask is not None:
        categories_valid = categories_valid & token_mask.bool().unsqueeze(-2)
    template.residue_type_ids = project_model_categories(
        template.residue_type_ids, categories_valid, num_classes=vocab_size, feature="AlphaFold3 template classes"
    )
    if template_valid is not None:
        pair_valid = pair_valid & template_valid.to(torch.bool)[..., None, None]
        frame_valid = frame_valid & template_valid.to(torch.bool)[..., None, None]
    template.distogram = project_model_features(
        template.distogram, pair_valid[..., None], dtype=dtype, feature="AlphaFold3 template distogram"
    )
    template.unit_vector = project_model_features(
        template.unit_vector, frame_valid[..., None], dtype=dtype, feature="AlphaFold3 template vectors"
    )
    return template


def normalize_alphafold3_input(
    input: AlphaFold3Input, *, dtype: torch.dtype | None = None, max_atoms_per_token: int | None = None
) -> AlphaFold3Input:
    """Return finite model features without changing source facts or prepared maps.

    Reference availability excludes reference payload; atom validity excludes
    padded feature rows. Template pair masks already include physical placement
    eligibility and remain the authority for their derived descriptors.
    """
    result = cast(AlphaFold3Input, cast(TensorClass, input).clone(recurse=False))
    atom_mask = input.atom_mask.to(torch.bool)
    offsets = torch.where(atom_mask, input.atom_token_index, 0)
    nonnegative = (offsets >= 0).all()
    if torch.compiler.is_compiling():
        torch._assert_async(nonnegative, "AlphaFold3 atom offsets must be nonnegative")
    elif not bool(nonnegative):
        fail("value_invalid", RequestRef("atom_token_index", "negative"))
    if max_atoms_per_token is not None:
        representable = (offsets < max_atoms_per_token).all()
        if torch.compiler.is_compiling():
            torch._assert_async(representable, "AlphaFold3 atom offsets exceed confidence capacity")
        elif not bool(representable):
            fail("model_capacity", RequestRef("atom_token_index", f"not below {max_atoms_per_token}"))
    result.atom_token_index = offsets
    token_mask = input.token_mask.to(torch.bool)
    result.reference_positions = project_model_coordinates(
        input.reference_positions,
        input.reference_mask.to(torch.bool) & atom_mask,
        dtype=dtype,
        feature="AlphaFold3 reference positions",
    )
    result.atom_features = project_model_features(
        input.atom_features, atom_mask[..., None], dtype=dtype, feature="AlphaFold3 atom features"
    )
    result.reference_charges = project_model_features(
        input.reference_charges, atom_mask[..., None], dtype=dtype, feature="AlphaFold3 reference charges"
    )
    result.token_features = project_model_features(
        input.token_features, token_mask[..., None], dtype=dtype, feature="AlphaFold3 token features"
    )
    result.token_bonds = project_model_features(
        input.token_bonds,
        token_mask[..., :, None] & token_mask[..., None, :],
        dtype=dtype,
        feature="AlphaFold3 token relations",
    )
    if input.template_features is not None:
        result.template_features = _project_alphafold3_templates(
            input.template_features,
            dtype=dtype,
            pair_mask=token_mask[..., :, None] & token_mask[..., None, :],
            token_mask=token_mask,
        )
    msa_available = input.msa_mask.bool() & token_mask.unsqueeze(-2)
    result.msa_tokens = project_model_categories(
        input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="AlphaFold3 MSA tokens"
    )
    result.msa_deletion_counts = project_model_counts(
        input.msa_deletion_counts, msa_available, feature="AlphaFold3 MSA deletion counts"
    )
    result.msa_mask = msa_available
    return result


def _normalize_alphafold3_denoising(
    inputs: CoordinateDenoiseContext | CoordinateDenoisingInput, *, dtype: torch.dtype
) -> CoordinateDenoiseContext | CoordinateDenoisingInput:
    atom_mask = inputs.atom_mask.to(torch.bool)
    reference_positions = project_model_coordinates(
        inputs.reference_positions,
        inputs.reference_mask.to(torch.bool) & atom_mask,
        dtype=dtype,
        feature="AlphaFold3 denoising references",
    )
    atom_features = project_model_features(
        inputs.atom_features, atom_mask[..., None], dtype=dtype, feature="AlphaFold3 denoising atom features"
    )
    reference_charges = project_model_features(
        inputs.reference_charges, atom_mask[..., None], dtype=dtype, feature="AlphaFold3 denoising charges"
    )
    normalized = copy.copy(inputs)
    normalized.reference_positions = reference_positions
    normalized.atom_features = atom_features
    normalized.reference_charges = reference_charges
    if isinstance(normalized, CoordinateDenoisingInput):
        normalized.noisy_coords = project_model_coordinates(
            normalized.noisy_coords,
            atom_mask.unsqueeze(-2) & torch.ones_like(normalized.noisy_coords[..., 0], dtype=torch.bool),
            dtype=dtype,
            feature="AlphaFold3 noisy coordinates",
        )
    return normalized


class AlphaFold3SamplingOutput(TensorCollection):
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

        return plddt_from_logits(self.plddt_logits)


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
