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
PxDesign coordinate-design model graph.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.atom import AtomConditioner, AtomPairEncoder, AtomToTokenEncoder, LocalAtomWindowing
from ...modules.attention import PerBlockPairBiasProjection
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoiseFn,
    EdmScheduler,
)
from ...modules.diffusion.sampling import draw_normal
from ...modules.embedding import RelativePositionBuckets, generate_relative_position_features
from ...modules.functional import weighted_mean
from ...modules.geometry import uniform_random_rotation
from ...modules.transformer import NormalizedKeyAtomCrossAttentionTransformerEncoderBlock
from ..processing import project_model_coordinates, project_model_features
from .config import PxDesignConfig


class PxDesignModel(Model):
    r"""
    Generate protein backbones with PXDesign.

    Accepts [`PxDesignInput`][ofoldx.models.pxdesign.model.PxDesignInput] and returns
    [`PxDesignSamplingOutput`][ofoldx.models.pxdesign.model.PxDesignSamplingOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = PxDesignConfig
    config: PxDesignConfig

    def __init__(self, config: PxDesignConfig | None = None) -> None:
        config = PxDesignConfig() if config is None else config
        super().__init__(config)
        self.condition_embeddings = PxDesignConditionEmbeddings(config)

        denoiser = config.denoiser
        self.coordinate_denoiser = CoordinateDenoiser(
            num_encoder_blocks=denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=denoiser.num_token_encoder_attention_heads,
            sigma_data=denoiser.sigma_data,
            atom_hidden_size=denoiser.atom_hidden_size,
            atom_pair_hidden_size=denoiser.atom_pair_hidden_size,
            token_hidden_size=denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=denoiser.num_query_atoms,
            num_key_atoms=denoiser.num_key_atoms,
        )
        self.coordinate_sampler = PxDesignSampler(config)
        self.post_init()

    def forward(
        self,
        input: PxDesignInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
    ) -> PxDesignSamplingOutput:
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        ``B`` is batch size, ``S`` is the coordinate-sample axis, ``N`` is token count,
        and ``A`` is atom count.

        Args:
            input: Processed target-binder design input.
            noisy_coordinates: Noisy atom coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: Diffusion noise levels with shape ``(B, S)``.

        Returns:
            A [`PxDesignSamplingOutput`][] with denoised ``coordinates`` of shape
                ``(B, S, A, 3)`` in angstroms.
        """

        input = normalize_pxdesign_input(input, dtype=self.condition_embeddings.input_projection.weight.dtype)
        token_embedding, single, pair, relative_features = self.encode_condition(input)
        context = self._denoise_context(input, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_denoiser.prepare(context)(
            context.to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return PxDesignSamplingOutput(coordinates=coordinates, batch_size=[coordinates.shape[0]])

    @torch.inference_mode()
    def sample(
        self,
        input: PxDesignInput,
        *,
        num_samples: int = 100,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        generator: torch.Generator | None = None,
    ) -> PxDesignSamplingOutput:
        r"""
        Sample target-binder backbone coordinates.

        Args:
            input: Processed design input with ``B`` batch items and ``A`` atoms.
            num_samples: Number of coordinate samples per batch item.
            num_sampling_steps: Number of EDM sampling steps.
            sample_chunk_size: Maximum number of samples denoised together. ``None``
                processes all samples together.
            generator: Random generator used for reference-conformer augmentation and
                coordinate sampling.

        Returns:
            A [`PxDesignSamplingOutput`][] with ``coordinates`` of shape
                ``(B, num_samples, A, 3)`` in angstroms.
        """

        input = normalize_pxdesign_input(input, dtype=self.condition_embeddings.input_projection.weight.dtype)
        input = _augment_reference_conformers(input, generator)
        input = normalize_pxdesign_input(input, dtype=self.condition_embeddings.input_projection.weight.dtype)
        token_embedding, single, pair, relative_features = self.encode_condition(input)
        context = self._denoise_context(input, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_sampler(
            self.coordinate_denoiser.prepare(context),
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        return PxDesignSamplingOutput(coordinates=coordinates, batch_size=[coordinates.shape[0]])

    def encode_condition(
        self,
        input: PxDesignInput,
    ) -> tuple[Tensor, Tensor, Tensor, RelativePositionBuckets]:
        r"""
        Encode target, binder, and hotspot conditions.

        ``B`` is batch size and ``N`` is token count.

        Args:
            input: Processed design input.

        Returns:
            token: Token embedding with shape ``(B, N, C_token)``.
            single: Single representation with shape ``(B, N, C_single)``.
            pair: Pair representation with shape ``(B, N, N, C_pair)``.
            relative: Integer pairwise buckets with leading shape ``(B, N, N)``.
        """

        input = normalize_pxdesign_input(input, dtype=self.condition_embeddings.input_projection.weight.dtype)
        token_embedding, pair = self.condition_embeddings(input)
        single = token_embedding.new_zeros(*token_embedding.shape[:-1], self.config.single_size)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            dtype=token_embedding.dtype,
        )
        return token_embedding, single, pair, relative_features

    def _denoise_context(
        self,
        input: PxDesignInput,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> CoordinateDenoiseContext:
        r"""
        Assemble the conditioning shared by ``forward`` and ``sample``.
        """

        local_geometry = self.coordinate_denoiser.windowing(
            input.reference_positions,
            input.reference_space_id,
            input.atom_mask,
            input.reference_mask,
        )
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=token_embedding,
            single=single,
            conditioning_single=single,
            conditioning_pair=pair,
            relative_features=relative_features,
            local_geometry=local_geometry,
        )


class PxDesignConditionEmbeddings(nn.Module):
    r"""
    Embed PxDesign atom, token, and conditional-template features.

    The centred [`windowing`][ofoldx.modules.atom.window.LocalAtomWindowing] builds the local atom geometry
    of the reference atoms, which are conditioned, have their local pairs encoded and projected to every
    block's attention bias, and are mean-pooled to tokens after local attention.
    """

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        embedder = config.condition_embeddings
        self.windowing = LocalAtomWindowing(embedder.num_query_atoms, embedder.num_key_atoms)
        self.atom_conditioner = AtomConditioner(config.atom_feature_size, embedder.atom_hidden_size)
        self.atom_pair_encoder = AtomPairEncoder(embedder.atom_hidden_size, embedder.atom_pair_hidden_size)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=embedder.atom_pair_hidden_size,
            num_blocks=embedder.num_atom_encoder_blocks,
            num_attention_heads=embedder.num_atom_encoder_attention_heads,
            bias=False,
        )
        self.atom_encoder = AtomToTokenEncoder(
            embedder.atom_hidden_size,
            embedder.token_hidden_size,
            embedder.num_atom_encoder_blocks,
            embedder.num_atom_encoder_attention_heads,
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
        )
        input_size = (
            embedder.token_hidden_size
            + embedder.vocab_size
            + embedder.design_feature_size
            + embedder.plddt_size
            + embedder.hotspot_size
            + 2 * embedder.add_feature_size
        )
        self.input_projection = nn.Linear(input_size, config.token_feature_size)
        self.template_embedder = PxDesignConditionalTemplateEmbedder(
            num_template_bins=config.num_template_bins,
            pair_size=config.pair_size,
        )
        self.add_feature_size = embedder.add_feature_size

    def forward(self, input: PxDesignInput) -> tuple[Tensor, Tensor]:
        input = normalize_pxdesign_input(input, dtype=self.input_projection.weight.dtype)
        local_geometry = self.windowing(
            input.reference_positions,
            input.reference_space_id,
            input.atom_mask,
            input.reference_mask,
        )
        atom_conditioning = self.atom_conditioner(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, local_geometry)
        token_hidden = self.atom_encoder(
            atom_conditioning,
            self.atom_pair_bias_projection(atom_pair),
            input.atom_mask,
            input.atom_to_token,
            input.residue_type_features,
            local_geometry.window_info,
        )
        add_feature = F.one_hot(
            torch.zeros(
                input.residue_type_features.shape[:-1],
                device=input.residue_type_features.device,
                dtype=torch.long,
            ),
            num_classes=self.add_feature_size,
        ).to(dtype=token_hidden.dtype)
        token_embedding = self.input_projection(
            torch.cat(
                (
                    token_hidden,
                    input.residue_type_features.to(token_hidden),
                    input.design_features.to(token_hidden),
                    input.plddt.to(token_hidden),
                    input.hotspot.to(token_hidden),
                    add_feature,
                    add_feature,
                ),
                dim=-1,
            )
        )
        pair = self.template_embedder(
            input.template_distance_bins,
            input.template_pair_mask,
        )
        return token_embedding, pair.to(token_embedding)


class PxDesignConditionalTemplateEmbedder(nn.Module):
    r"""
    Embed conditional-template distance-bin IDs.
    """

    def __init__(self, *, num_template_bins: int = 64, pair_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_template_bins + 1, pair_size)

    def forward(self, template_distance_bins: Tensor, template_pair_mask: Tensor) -> Tensor:
        embedding_ids = template_pair_mask.to(dtype=torch.long) * (template_distance_bins.to(dtype=torch.long) + 1)
        return self.embedding(embedding_ids)


class PxDesignSampler(nn.Module):
    r"""
    The PxDesign EDM predictor-corrector sampling loop.
    """

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        sampler = config.sampler
        self.scheduler = EdmScheduler(
            max_noise_level=sampler.max_noise_level,
            min_noise_level=sampler.min_noise_level,
            rho=sampler.rho,
            sigma_data=config.denoiser.sigma_data,
        )
        self.churn_rate = sampler.churn_rate
        self.min_churn_noise_level = sampler.min_churn_noise_level
        self.noise_scale = sampler.noise_scale
        self.min_step_scale = sampler.min_step_scale
        self.max_step_scale = sampler.max_step_scale
        self.step_scale_schedule = sampler.step_scale_schedule

    def forward(
        self,
        denoise: CoordinateDenoiseFn,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if sample_chunk_size is not None and sample_chunk_size < 1:
            fail("value_invalid", RequestRef("sample_chunk_size", str(sample_chunk_size)))
        noise_schedule = self.scheduler(
            num_sampling_steps=num_sampling_steps,
            device=context.device,
            dtype=context.dtype,
        )
        chunk_size = num_samples if sample_chunk_size is None else min(sample_chunk_size, num_samples)
        reference = context.reference_positions
        coordinates = noise_schedule[0] * draw_normal(
            (*context.batch_shape, num_samples, reference.shape[-2], 3),
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        observed = context.atom_mask.bool()[..., None, :] & torch.ones_like(coordinates[..., 0], dtype=torch.bool)
        coordinates = project_model_coordinates(coordinates, observed, feature="PXDesign sampled coordinates")
        schedule_length = noise_schedule.shape[0]

        for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            coordinates = project_model_coordinates(coordinates, observed, feature="PXDesign sampler coordinates")
            center = weighted_mean(coordinates.transpose(-1, -2), observed.unsqueeze(-2)).unsqueeze(-2)
            center = torch.where(observed.any(-1, keepdim=True).unsqueeze(-1), center, 0)
            coordinates = project_model_coordinates(
                coordinates - center, observed, feature="PXDesign centered coordinates"
            )
            batch_shape = coordinates.shape[:-2]
            random_device = coordinates.device if generator is None else generator.device
            rotations = (
                uniform_random_rotation(
                    math.prod(batch_shape),
                    device=random_device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                .to(coordinates.device)
                .reshape(*batch_shape, 3, 3)
            )
            translations = draw_normal(
                (*batch_shape, 3),
                device=coordinates.device,
                dtype=coordinates.dtype,
                generator=generator,
            )
            coordinates = torch.einsum("...ij,...nj->...ni", rotations, coordinates)
            coordinates = project_model_coordinates(
                coordinates + translations.unsqueeze(-2),
                observed,
                feature="PXDesign rigid augmentation",
            )

            gamma = torch.where(
                sigma_next > self.min_churn_noise_level,
                sigma_next.new_tensor(self.churn_rate),
                sigma_next.new_zeros(()),
            )
            sigma_hat = sigma_prev * (gamma + 1)
            churn = draw_normal(
                coordinates.shape,
                device=coordinates.device,
                dtype=coordinates.dtype,
                generator=generator,
            )
            noisy_coordinates = project_model_coordinates(
                coordinates + (self.noise_scale * torch.sqrt(sigma_hat**2 - sigma_prev**2) * churn),
                observed,
                feature="PXDesign churn coordinates",
            )

            fraction = step_index / schedule_length
            minimum = float(self.min_step_scale)
            maximum = float(self.max_step_scale)
            if self.step_scale_schedule == "const":
                step_scale = minimum
            elif self.step_scale_schedule == "linear":
                step_scale = minimum + (maximum - minimum) * fraction
            elif self.step_scale_schedule == "poly":
                step_scale = minimum + (maximum - minimum) * fraction**2
            elif self.step_scale_schedule == "cos":
                step_scale = minimum + 0.5 * (maximum - minimum) * (1 - math.cos(math.pi * fraction))
            elif self.step_scale_schedule == "piecewise":
                step_scale = minimum if fraction < 0.5 else maximum
            elif self.step_scale_schedule == "piecewise_65":
                step_scale = minimum if fraction < 0.65 else maximum
            elif self.step_scale_schedule == "piecewise_70":
                step_scale = minimum if fraction < 0.70 else maximum
            else:
                fail("value_invalid", RequestRef("step_scale_schedule", self.step_scale_schedule))

            chunks: list[Tensor] = []
            for sample_start in range(0, num_samples, chunk_size):
                num_chunk_samples = min(chunk_size, num_samples - sample_start)
                sample_end = sample_start + num_chunk_samples
                chunk_noisy_coordinates = noisy_coordinates[..., sample_start:sample_end, :, :]
                leading_shape = noisy_coordinates.shape[:-3]
                sample_count = chunk_noisy_coordinates.shape[-3]
                noise_level = sigma_hat.reshape((1,) * (len(leading_shape) + 1)).expand(
                    *leading_shape,
                    sample_count,
                )
                denoised_coordinates = denoise(
                    context.to_denoising_input(
                        noisy_coords=chunk_noisy_coordinates,
                        noise_level=noise_level,
                    )
                )

                derivative = (chunk_noisy_coordinates - denoised_coordinates) / noise_level[..., None, None]
                chunks.append(chunk_noisy_coordinates + step_scale * (sigma_next - sigma_hat) * derivative)
            coordinates = torch.cat(chunks, dim=-3)

        return project_model_coordinates(coordinates, observed, feature="PXDesign sampled output")


def _augment_reference_conformers(
    input: PxDesignInput,
    generator: torch.Generator | None,
) -> PxDesignInput:
    r"""
    Apply the independent per-residue reference augmentation.
    """

    positions = project_model_coordinates(
        input.reference_positions,
        input.reference_mask.bool() & input.atom_mask.bool(),
        feature="PXDesign augmentation references",
    ).clone()
    flat_positions = positions.reshape(-1, positions.shape[-2], 3)
    flat_spaces = input.reference_space_id.reshape(-1, input.reference_space_id.shape[-1])
    flat_masks = (input.reference_mask.bool() & input.atom_mask.bool()).reshape(-1, positions.shape[-2])
    active_spaces: list[tuple[int, Tensor]] = []
    for sample_index, sample_spaces in enumerate(flat_spaces):
        for space_id in torch.unique(sample_spaces):
            if bool(((sample_spaces == space_id) & flat_masks[sample_index]).any()):
                active_spaces.append((sample_index, space_id))

    if active_spaces:
        random_device = positions.device if generator is None else generator.device
        translations = (
            2
            * torch.rand(
                len(active_spaces),
                3,
                device=random_device,
                dtype=positions.dtype,
                generator=generator,
            ).to(positions.device)
            - 1
        )
        rotations = uniform_random_rotation(
            len(active_spaces),
            device=random_device,
            dtype=positions.dtype,
            generator=generator,
        ).to(positions.device)
        for (sample_index, space_id), translation, rotation in zip(active_spaces, translations, rotations):
            sample_positions = flat_positions[sample_index]
            selected = (flat_spaces[sample_index] == space_id) & flat_masks[sample_index]
            values = sample_positions[selected]
            values = values - values.mean(dim=0)
            augmented = (values + translation) @ rotation.transpose(-1, -2)
            sample_positions[selected] = augmented
    result = cast(PxDesignInput, cast(TensorClass, input).clone(recurse=False))
    result.reference_positions = positions
    return result


# ===========================================================================
# PxDesign tensor inputs and outputs.
# ===========================================================================


@tensorclass
class PxDesignInput:
    r"""
    Tensor and molecular-layout input for PxDesign.

    ``B`` is batch size, ``N`` token count, and ``A`` atom count.

    Attributes:
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
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        residue_type_features: One-hot canonical-class features with shape
            ``(B, N, 72)``.
        design_features: Generated-token indicator with shape ``(B, N, 1)``.
        plddt: Per-token confidence condition with shape ``(B, N, 1)``.
        hotspot: Per-token hotspot condition with shape ``(B, N, 1)``.
        template_distance_bins: Template distance features aligned to token pairs.
        template_pair_mask: Valid-template-pair mask aligned to token pairs.
    """

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
    reference_mask: Tensor
    reference_charges: Tensor
    residue_type_features: Tensor
    design_features: Tensor
    plddt: Tensor
    hotspot: Tensor
    template_distance_bins: Tensor
    template_pair_mask: Tensor


class PxDesignSamplingOutput(TensorCollection):
    r"""
    Predictions from one PxDesign denoising or sampling call.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    coordinates: Tensor


def normalize_pxdesign_input(input: PxDesignInput, *, dtype: torch.dtype | None = None) -> PxDesignInput:
    """Project family fields without reinterpreting prepared operation bindings."""
    result = cast(PxDesignInput, cast(TensorClass, input).clone(recurse=False))
    valid = input.atom_mask.bool()
    result.reference_positions = project_model_coordinates(
        input.reference_positions,
        valid & input.reference_mask.bool(),
        dtype=dtype,
        feature="PXDesign references",
    )
    for name in ("atom_features", "reference_charges"):
        values = getattr(input, name)
        setattr(
            result,
            name,
            project_model_features(
                values,
                valid[..., None],
                dtype=dtype,
                feature=f"PXDesign {name}",
            ),
        )
    # The token axis is unpadded. Independent raw inputs must provide complete
    # finite token features; reference availability does not erase chemistry.
    for name in ("residue_type_features", "design_features", "plddt", "hotspot"):
        values = getattr(input, name)
        setattr(
            result,
            name,
            project_model_features(
                values,
                torch.ones_like(values, dtype=torch.bool),
                dtype=dtype,
                feature=f"PXDesign {name}",
            ),
        )
    return result
