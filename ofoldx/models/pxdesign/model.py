r"""
PxDesign coordinate-design model graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data import Hotspot, PlacementMap, PlacementSpec
from ...data.molecular import AtomSite, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import NormalizedKeyAtomToTokenEncoder
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, EdmScheduler
from ...modules.embedding import RelativePositionBuckets, generate_relative_position_features
from ...modules.geometry import uniform_random_rotation
from .config import PxDesignConfig


class PxDesignModel(Model):
    r"""
    Generate protein backbones with PXDesign.

    Accepts [`PxDesignInput`][ofoldx.models.pxdesign.model.PxDesignInput] and returns
    [`PxDesignOutput`][ofoldx.models.pxdesign.model.PxDesignOutput].

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
    ) -> PxDesignOutput:
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
            A [`PxDesignOutput`][] with denoised ``coordinates`` of shape
            ``(B, S, A, 3)`` in angstroms.
        """

        token_embedding, single, pair, relative_features = self.encode_condition(input)
        context = self._denoise_context(input, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_denoiser(
            context.to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return PxDesignOutput(coordinates)

    @torch.inference_mode()
    def sample(
        self,
        input: PxDesignInput,
        *,
        num_samples: int = 100,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        generator: torch.Generator | None = None,
    ) -> PxDesignOutput:
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
            A [`PxDesignOutput`][] with ``coordinates`` of shape
            ``(B, num_samples, A, 3)`` in angstroms.
        """

        input = _augment_reference_conformers(input, generator)
        token_embedding, single, pair, relative_features = self.encode_condition(input)
        context = self._denoise_context(input, token_embedding, single, pair, relative_features)
        coordinates = self.coordinate_sampler(
            self.coordinate_denoiser,
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        return PxDesignOutput(coordinates)

    @torch.inference_mode()
    def design(
        self,
        input: PxDesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
    ) -> PxDesignDesignResult:
        r"""
        Generate decoded target-binder backbone candidates.

        Args:
            input: Processed design input with batch size one.
            num_candidates: Number of backbone candidates.
            generator: Random generator used for reference-conformer augmentation and
                coordinate sampling.
            num_sampling_steps: Number of EDM sampling steps.
            sample_chunk_size: Maximum number of candidates denoised together. ``None``
                processes all candidates together.

        Returns:
            A [`PxDesignDesignResult`][] containing ``num_candidates`` structures in
            sampling order. Structure coordinates are in angstroms.
        """

        if input.reference_positions.shape[:-2] != (1,):
            raise ValueError("PxDesignModel.design requires exactly one biological input")
        output = self.sample(
            input,
            num_samples=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        candidates = tuple(
            PxDesignDesignCandidate(
                structure=Structure(
                    system=input.layout.system,
                    sites=input.layout.sites,
                    coordinates=output.coordinates[0, index].detach().to(device="cpu", dtype=torch.float32).numpy(),
                    coordinate_mask=np.ones(len(input.layout.sites), dtype=np.bool_),
                ),
            )
            for index in range(num_candidates)
        )
        # This model does not produce candidate-level confidence values.
        return PxDesignDesignResult(candidates=candidates)

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
            A tuple ``(token, single, pair, relative)``. ``token`` and ``single`` have
            shapes ``(B, N, C_token)`` and ``(B, N, C_single)``; ``pair`` has shape
            ``(B, N, N, C_pair)``; ``relative`` contains integer pairwise buckets with
            leading shape ``(B, N, N)``.
        """

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

        local_geometry = self.coordinate_denoiser.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
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
            pair=pair,
            relative_features=relative_features,
            local_geometry=local_geometry,
        )


class PxDesignConditionEmbeddings(nn.Module):
    r"""
    Embed PxDesign atom, token, and conditional-template features.
    """

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        embedder = config.condition_embeddings
        self.atom_encoder = NormalizedKeyAtomToTokenEncoder(
            token_hidden_size=embedder.token_hidden_size,
            atom_hidden_size=embedder.atom_hidden_size,
            atom_pair_hidden_size=embedder.atom_pair_hidden_size,
            num_blocks=embedder.num_atom_encoder_blocks,
            num_attention_heads=embedder.num_atom_encoder_attention_heads,
            num_query_atoms=embedder.num_query_atoms,
            num_key_atoms=embedder.num_key_atoms,
            atom_feature_size=config.atom_feature_size,
        )
        input_size = (
            embedder.token_hidden_size
            + embedder.vocab_size
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
        local_geometry = self.atom_encoder.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
        )
        atom_output = self.atom_encoder(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=torch.arcsinh(input.reference_charges),
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            atom_to_token=input.atom_to_token,
            num_tokens=input.residue_type_features.shape[-2],
            local_geometry=local_geometry,
        )
        add_feature = F.one_hot(
            torch.zeros(
                input.residue_type_features.shape[:-1],
                device=input.residue_type_features.device,
                dtype=torch.long,
            ),
            num_classes=self.add_feature_size,
        ).to(dtype=atom_output.token_hidden.dtype)
        token_embedding = self.input_projection(
            torch.cat(
                (
                    atom_output.token_hidden,
                    input.residue_type_features.to(atom_output.token_hidden),
                    input.plddt.to(atom_output.token_hidden),
                    input.hotspot.to(atom_output.token_hidden),
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
        self.step_scale_min = sampler.step_scale_min
        self.step_scale_max = sampler.step_scale_max
        self.step_scale_schedule = sampler.step_scale_schedule

    def forward(
        self,
        denoise: nn.Module,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sample_chunk_size is not None and sample_chunk_size < 1:
            raise ValueError("sample_chunk_size must be positive")
        noise_schedule = self.scheduler(
            num_sampling_steps=num_sampling_steps,
            device=context.device,
            dtype=context.dtype,
        )
        chunk_size = num_samples if sample_chunk_size is None else sample_chunk_size
        chunks: list[Tensor] = []
        for sample_start in range(0, num_samples, chunk_size):
            chunk_num_samples = min(chunk_size, num_samples - sample_start)
            reference = context.reference_positions.to(device=context.device, dtype=context.dtype)
            noise = torch.randn(
                *context.batch_shape,
                chunk_num_samples,
                reference.shape[-2],
                3,
                device=context.device,
                dtype=context.dtype,
                generator=generator,
            )
            coordinates = noise_schedule[0] * noise
            schedule_length = noise_schedule.shape[0]

            for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
                coordinates = coordinates - coordinates.mean(dim=-2, keepdim=True)
                batch_shape = coordinates.shape[:-2]
                rotations = uniform_random_rotation(
                    math.prod(batch_shape),
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                ).reshape(*batch_shape, 3, 3)
                translations = torch.randn(
                    *batch_shape,
                    3,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                coordinates = torch.einsum("...ij,...nj->...ni", rotations, coordinates)
                coordinates = coordinates + translations.unsqueeze(-2)

                gamma = torch.where(
                    sigma_next > self.min_churn_noise_level,
                    sigma_next.new_tensor(self.churn_rate),
                    sigma_next.new_zeros(()),
                )
                sigma_hat = sigma_prev * (gamma + 1)
                churn = torch.randn(
                    coordinates.shape,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                noisy_coordinates = coordinates + (self.noise_scale * torch.sqrt(sigma_hat**2 - sigma_prev**2) * churn)
                leading_shape = noisy_coordinates.shape[:-3]
                sample_count = noisy_coordinates.shape[-3]
                noise_level = sigma_hat.reshape((1,) * (len(leading_shape) + 1)).expand(
                    *leading_shape,
                    sample_count,
                )
                denoised_coordinates = denoise(
                    context.to_denoising_input(
                        noisy_coords=noisy_coordinates,
                        noise_level=noise_level.to(dtype=noisy_coordinates.dtype),
                    )
                )

                fraction = step_index / schedule_length
                minimum = float(self.step_scale_min)
                maximum = float(self.step_scale_max)
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
                    raise ValueError(f"unsupported PxDesign step_scale_schedule: {self.step_scale_schedule!r}")

                derivative = (noisy_coordinates - denoised_coordinates) / noise_level[..., None, None]
                coordinates = noisy_coordinates + step_scale * (sigma_next - sigma_hat) * derivative
            chunks.append(coordinates)

        return torch.cat(chunks, dim=-3)


def _augment_reference_conformers(
    input: PxDesignInput,
    generator: torch.Generator | None,
) -> PxDesignInput:
    r"""
    Apply the independent per-residue reference augmentation.
    """

    positions = input.reference_positions.clone()
    flat_positions = positions.reshape(-1, positions.shape[-2], 3)
    flat_spaces = input.reference_space_id.reshape(-1, input.reference_space_id.shape[-1])
    for sample_positions, sample_spaces in zip(flat_positions, flat_spaces):
        for space_id in torch.unique(sample_spaces):
            selected = sample_spaces == space_id
            values = sample_positions[selected]
            augmentation_dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
            values = values.to(augmentation_dtype)
            values = values - values.mean(dim=0)
            translation = (
                2
                * torch.rand(
                    3,
                    device=values.device,
                    dtype=augmentation_dtype,
                    generator=generator,
                )
                - 1
            )
            rotation = uniform_random_rotation(
                1,
                device=values.device,
                dtype=augmentation_dtype,
                generator=generator,
            )[0]
            augmented = (values + translation) @ rotation.transpose(-1, -2)
            sample_positions[selected] = augmented.to(sample_positions.dtype)
    return replace(input, reference_positions=positions)


# ============================================================================
# PxDesign tensor inputs and outputs
# ============================================================================
# ``PlacementSpec`` and ``PlacementMap`` are shared records in ``ofoldx.data``. ``Hotspot`` is
# the shared distance-free per-residue label consumed by PxDesign.


@dataclass(frozen=True)
class PxDesignLayout:
    r"""
    Molecular identities aligned to PxDesign token and atom tensors.

    Args:
        system: Molecular system represented by the tensors.
        sites: Atom occurrences aligned to the atom axis.
        placement: Placement declaration used to build the output.
        conditions: Hotspot conditions applied to the design.
        placement_map: Realized component occurrences for every placement.
    """

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    placement: PlacementSpec
    conditions: tuple[Hotspot, ...]
    placement_map: PlacementMap


@dataclass(kw_only=True, eq=False)
class PxDesignInput:
    r"""
    Tensor and molecular-layout input for PxDesign.

    ``B`` is batch size, ``N`` token count, and ``A`` atom count.

    Args:
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
        residue_type_features: One-hot residue-type features with shape
            ``(B, N, 36)``.
        plddt: Per-token confidence condition with shape ``(B, N, 1)``.
        hotspot: Per-token hotspot condition with shape ``(B, N, 1)``.
        template_distance_bins: Template distance features aligned to token pairs.
        template_pair_mask: Valid-template-pair mask aligned to token pairs.
        layout: Molecular identities and conditions aligned to the tensor axes.
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
    plddt: Tensor
    hotspot: Tensor
    template_distance_bins: Tensor
    template_pair_mask: Tensor
    layout: PxDesignLayout


@dataclass(eq=False)
class PxDesignOutput:
    r"""
    Predictions from one PxDesign denoising or sampling call.

    Args:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    coordinates: Tensor


@dataclass(eq=False)
class PxDesignDesignCandidate:
    r"""
    One coordinate-bearing target-binder backbone candidate.

    Args:
        structure: Decoded backbone structure. Unassigned binder residues retain their unknown
            component identity.
    """

    structure: Structure


@dataclass(eq=False)
class PxDesignDesignResult:
    r"""
    A generation-ordered collection of PxDesign candidates.

    Args:
        candidates: Candidates in generation order.
    """

    candidates: tuple[PxDesignDesignCandidate, ...]
