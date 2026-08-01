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
Elucidated-diffusion-model coordinate sampling.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..atom import LocalAtomGeometry
from ..atom.coordinates import project_coordinates
from ..batch import insert_sample_axis
from ..embedding import RelativePositionBuckets
from ..functional import weighted_mean
from ..geometry import apply_inverse_rigid_transform, apply_rigid_transform, uniform_random_rotation
from .schedule import EdmScheduler
from .types import CoordinateDenoiseFn, CoordinateDenoisingInput, StepDenoiseFn


@dataclass(eq=False)
class CoordinateDenoiseContext:
    r"""
    Conditioning tensors shared by coordinate-denoising steps.

    Attributes:
        atom_features: Per-atom features with shape ``[..., N_atom, C_atom]``.
        atom_to_token: Token index for each atom with shape ``[..., N_atom]``.
        reference_positions: Centered reference coordinates with shape
            ``[..., N_atom, 3]``.
        reference_charges: Formal charges with shape ``[..., N_atom, 1]``.
        reference_mask: Reference-coordinate mask with shape ``[..., N_atom]``.
        atom_mask: Atom validity mask with shape ``[..., N_atom]``.
        token_embedding: Token embeddings with shape ``[..., N_token, C_token]``.
        single: Trunk single representation used by the atom path, with shape
            ``[..., N_token, C_single]``.
        conditioning_single: Single representation supplied to diffusion
            conditioning, with shape ``[..., N_token, C_single]``.
        conditioning_pair: Pair representation supplied to diffusion
            conditioning, with shape ``[..., N_token, N_token, C_pair]``.
        relative_features: Relative-position feature bundle.
        local_geometry: Local atom-window geometry.
    """

    atom_features: Tensor
    atom_to_token: Tensor
    reference_positions: Tensor
    reference_charges: Tensor
    reference_mask: Tensor
    atom_mask: Tensor
    token_embedding: Tensor
    single: Tensor
    conditioning_single: Tensor
    conditioning_pair: Tensor
    relative_features: RelativePositionBuckets
    local_geometry: LocalAtomGeometry

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Leading dimensions before the token and channel axes.

        Returns:
            The shared leading shape of the context tensors.
        """

        return self.token_embedding.shape[:-2]

    @property
    def device(self) -> torch.device:
        r"""
        Device of the context tensors.

        Returns:
            The device of the token embedding.
        """

        return self.token_embedding.device

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Floating-point dtype of the context tensors.

        Returns:
            The dtype of the token embedding.
        """

        return self.token_embedding.dtype

    @property
    def num_atoms(self) -> int:
        r"""
        Number of atoms in the context.

        Returns:
            The size of the atom axis.
        """

        return self.atom_to_token.size(-1)

    def to_denoising_input(
        self,
        noisy_coords: Tensor,
        noise_level: Tensor,
    ) -> CoordinateDenoisingInput:
        r"""
        Combine this context with one step's noisy coordinates and noise level.

        Args:
            noisy_coords: Noisy coordinates with shape
                ``[..., N_sample, N_atom, 3]``.
            noise_level: Noise levels with shape ``[..., N_sample]``.
        Returns:
            A denoising input carrying the context and step-specific tensors.
        """

        return CoordinateDenoisingInput(
            atom_features=self.atom_features,
            atom_to_token=self.atom_to_token,
            token_embedding=self.token_embedding,
            single=self.single,
            conditioning_single=self.conditioning_single,
            conditioning_pair=self.conditioning_pair,
            relative_features=self.relative_features,
            noise_level=noise_level,
            reference_positions=self.reference_positions,
            reference_charges=self.reference_charges,
            reference_mask=self.reference_mask,
            atom_mask=self.atom_mask,
            noisy_coords=noisy_coords,
            local_geometry=self.local_geometry,
        )


def denoise_coordinates(
    denoise: CoordinateDenoiseFn,
    *,
    context: CoordinateDenoiseContext,
    noisy_coords: Tensor,
    noise_level: Tensor,
    sample_chunk_size: int | None = None,
) -> Tensor:
    r"""
    Denoise coordinate samples, optionally bounding the samples processed together.

    Callers draw target augmentation, noise levels, and coordinate noise and build
    the shared context before this function. Chunking therefore does not change the
    corrupted coordinate inputs or repeat trunk computation.

    Args:
        denoise: Coordinate denoiser.
        context: Conditioning shared by all samples.
        noisy_coords: Noisy coordinates with shape ``[..., S, N_atom, 3]``.
        noise_level: Noise levels with shape ``[..., S]``.
        sample_chunk_size: Maximum number of samples denoised together. ``None``
            denoises all samples in one call.

    Returns:
        Denoised coordinates with the same shape as ``noisy_coords``.

    Raises:
        ValueError: If ``sample_chunk_size`` is not positive.
    """

    if sample_chunk_size is None:
        return denoise(context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level))
    sample_chunk_size = operator.index(sample_chunk_size)
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive")
    if sample_chunk_size >= noisy_coords.shape[-3]:
        return denoise(context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level))
    return torch.cat(
        [
            denoise(context.to_denoising_input(noisy_coords=coords, noise_level=sigma))
            for coords, sigma in zip(
                noisy_coords.split(sample_chunk_size, dim=-3),
                noise_level.split(sample_chunk_size, dim=-1),
            )
        ],
        dim=-3,
    )


class CoordinateEdmSampler(nn.Module):
    r"""
    Sample coordinates with an EDM reverse-diffusion process.

    Every step moves the coordinates into a fresh random frame
    (``_augment``), injects churn noise,
    denoises, and takes one Euler step, which
    ``_correct`` may refine; sampling then
    continues from the frame
    ``_unaugment`` returns.

    Args:
        scheduler: Noise schedule to sample along; a default
            [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] is built when ``None``.
        churn_rate: Fraction of extra noise injected before each step to make sampling stochastic.
        min_churn_noise_level: Churn is only applied while the next noise level exceeds this threshold.
        noise_scale: Multiplier on the injected churn noise standard deviation.
        step_scale: Multiplier applied to each solver step.
    """

    def __init__(
        self,
        *,
        scheduler: EdmScheduler | None = None,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.scheduler = EdmScheduler() if scheduler is None else scheduler
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale

    def augment_target(
        self,
        coords: Tensor,
        *,
        num_samples: int,
        atom_mask: Tensor | None = None,
        augmentation: bool = True,
        translation_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Center coordinates and optionally apply random rigid transforms.

        Args:
            coords: Coordinates with shape ``[..., N_atom, 3]``.
            num_samples: Number of augmented copies to draw.
            atom_mask: Optional ``[..., N_atom]`` validity mask used for centering.
            augmentation: Whether to apply a random rigid transform after centering.
            translation_scale: Standard deviation of the random translation.
            generator: Random generator used for augmentation.

        Returns:
            Coordinates with shape ``[..., num_samples, N_atom, 3]``.
        """

        augmented, _ = self._center_random_augmentation(
            coords,
            num_samples=num_samples,
            translation_scale=translation_scale,
            center_only=not augmentation,
            atom_mask=atom_mask,
            generator=generator,
        )
        return augmented

    def forward(
        self,
        denoise: CoordinateDenoiseFn,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int = 1,
        num_sampling_steps: int = 200,
        sample_chunk_size: int | None = None,
        initial_coords: Tensor | None = None,
        condition_coords: Tensor | None = None,
        condition_mask: Tensor | None = None,
        max_noise_level: float | None = None,
        churn_rate: float | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Sample coordinates by repeatedly applying ``denoise``.

        Args:
            denoise: Denoiser mapping a
                [`CoordinateDenoisingInput`][ofoldx.modules.diffusion.types.CoordinateDenoisingInput] to denoised
                coordinates.
            context: Static conditioning tensors shared across steps.
            num_samples: Number of samples to draw per batch element.
            num_sampling_steps: Number of reverse-diffusion steps.
            sample_chunk_size: Maximum number of candidates denoised together. Random
                draws still advance over the complete candidate axis.
            initial_coords: Optional starting coordinates with shape
                ``[..., N_atom, 3]`` or ``[..., num_samples, N_atom, 3]``.
                Gaussian noise is drawn when omitted.
            condition_coords: Optional fixed coordinates pinned after every denoising step.
            condition_mask: Optional ``[..., N_atom]`` mask selecting coordinates
                to pin from ``condition_coords``.
            max_noise_level: Optional per-call initial scheduler noise override.
            churn_rate: Optional per-call extra-noise fraction.
            noise_scale: Optional per-call churn-noise multiplier.
            step_scale: Optional per-call solver-step multiplier.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Sampled coordinates with shape ``[..., num_samples, N_atom, 3]``.
        """

        def step_denoise(noisy_coords: Tensor, noise_level: Tensor) -> Tensor:
            return denoise(context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level))

        return self._sample(
            step_denoise,
            reference_positions=context.reference_positions,
            atom_mask=context.atom_mask,
            device=context.device,
            dtype=context.dtype,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            initial_coords=initial_coords,
            condition_coords=condition_coords,
            condition_mask=condition_mask,
            max_noise_level=max_noise_level,
            churn_rate=self.churn_rate if churn_rate is None else churn_rate,
            noise_scale=self.noise_scale if noise_scale is None else noise_scale,
            step_scale=self.step_scale if step_scale is None else step_scale,
            generator=generator,
        )

    def _sample(
        self,
        denoise: StepDenoiseFn,
        *,
        reference_positions: Tensor,
        atom_mask: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None,
        initial_coords: Tensor | None,
        condition_coords: Tensor | None,
        condition_mask: Tensor | None,
        max_noise_level: float | None,
        churn_rate: float,
        noise_scale: float,
        step_scale: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Drive the reverse-diffusion loop, optionally chunked, and return coordinates.

        Args:
            denoise: Denoiser called at each step with noisy coordinates and noise levels.
            reference_positions: ``[..., N_atom, 3]`` coordinates whose shape seeds the sampled
                coordinate axis.
            atom_mask: Optional ``[..., N_atom]`` mask used when centering.
            device: Device of the noise schedule.
            dtype: Floating-point dtype of the noise schedule.
            num_samples: Total number of samples to draw.
            num_sampling_steps: Number of reverse-diffusion steps.
            sample_chunk_size: Optional maximum number of candidates passed to the
                denoiser together.
            initial_coords: Optional starting coords.
            condition_coords: Optional fixed coordinates pinned after each step.
            condition_mask: Optional atom mask selecting fixed coordinates.
            max_noise_level: Optional per-call initial scheduler noise override.
            churn_rate: Extra-noise fraction for this call.
            noise_scale: Churn-noise multiplier for this call.
            step_scale: Solver-step multiplier for this call.
            generator: Optional RNG.
        Returns:
            Sampled coordinates.

        Raises:
            ValueError: If ``num_samples`` or ``sample_chunk_size`` is not positive.
        """

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if sample_chunk_size is not None and sample_chunk_size < 1:
            raise ValueError(f"sample_chunk_size must be positive, got {sample_chunk_size}")
        if (condition_coords is None) != (condition_mask is None):
            raise ValueError("condition_coords and condition_mask must be provided together")
        noise_schedule = self.scheduler(
            num_sampling_steps=num_sampling_steps,
            device=device,
            dtype=dtype,
            max_noise_level=max_noise_level,
        )

        coords = self._initial_coords(
            reference_positions=reference_positions,
            num_samples=num_samples,
            noise_level=noise_schedule[0],
            initial_coords=initial_coords,
            generator=generator,
        )
        if condition_mask is not None and condition_coords is not None:
            coords = self._apply_condition(coords, condition_coords, condition_mask)

        chunk_size = num_samples if sample_chunk_size is None else min(sample_chunk_size, num_samples)
        num_transitions = noise_schedule.size(0) - 1
        for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            coords, augmentation_transform = self._augment(coords, atom_mask=atom_mask, generator=generator)

            step_condition_coords = condition_coords
            if condition_mask is not None and step_condition_coords is not None:
                if augmentation_transform is not None:
                    step_condition_coords = self._apply_augmentation(
                        step_condition_coords,
                        augmentation_transform,
                        target=coords,
                    )
                coords = self._apply_condition(coords, step_condition_coords, condition_mask)

            # Randomness is drawn for the complete candidate axis before denoiser
            # chunking.  ``sample_chunk_size`` therefore changes only peak
            # denoiser memory, never the advancing generator stream.
            noise = draw_normal_like(coords, generator)
            chunks: list[Tensor] = []
            for sample_start in range(0, num_samples, chunk_size):
                num_chunk_samples = min(chunk_size, num_samples - sample_start)
                chunk_coords = torch.narrow(coords, -3, sample_start, num_chunk_samples)
                chunk_noise = torch.narrow(noise, -3, sample_start, num_chunk_samples)
                chunk_condition_coords = self._coords_chunk(
                    step_condition_coords,
                    reference_positions=reference_positions,
                    num_samples=num_samples,
                    sample_start=sample_start,
                    num_chunk_samples=num_chunk_samples,
                    name="condition_coords",
                )
                chunk_coords = self._step(
                    denoise,
                    chunk_coords,
                    sigma_prev,
                    sigma_next,
                    atom_mask=atom_mask,
                    noise=chunk_noise,
                    condition_coords=chunk_condition_coords,
                    condition_mask=condition_mask,
                    is_terminal=step_index == num_transitions - 1,
                    churn_rate=churn_rate,
                    noise_scale=noise_scale,
                    step_scale=step_scale,
                )
                if condition_mask is not None and chunk_condition_coords is not None:
                    chunk_coords = self._apply_condition(
                        chunk_coords,
                        chunk_condition_coords,
                        condition_mask,
                    )
                chunks.append(chunk_coords)
            coords = torch.cat(chunks, dim=-3)

            coords = self._unaugment(coords, augmentation_transform)
            if condition_mask is not None and condition_coords is not None:
                coords = self._apply_condition(coords, condition_coords, condition_mask)
        return coords

    @staticmethod
    def _initial_coords(
        *,
        reference_positions: Tensor,
        num_samples: int,
        noise_level: Tensor,
        initial_coords: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Seed the sampling loop with either validated ``initial_coords`` or fresh noise at ``noise_level``.

        Args:
            reference_positions: ``[..., N_atom, 3]`` coordinates supplying the batch shape, atom
                count, device, and dtype of the seed.
            num_samples: Number of samples to seed.
            noise_level: First noise level, scaling the fresh Gaussian noise.
            initial_coords: Optional user-provided starting coords.
            generator: Optional RNG.

        Returns:
            Seed coords of shape `[..., num_samples, N_atom, 3]`.

        Raises:
            ValueError: If ``initial_coords`` has an unexpected shape.
        """

        reference = reference_positions
        unsampled_shape = tuple(reference.shape)
        expected_shape = (*unsampled_shape[:-2], num_samples, *unsampled_shape[-2:])
        if initial_coords is not None:
            if tuple(initial_coords.shape) == tuple(unsampled_shape):
                initial_coords = insert_sample_axis(initial_coords, dim=-3, num_samples=num_samples)
            elif tuple(initial_coords.shape) != tuple(expected_shape):
                raise ValueError(
                    f"initial_coords must have shape {expected_shape} or {unsampled_shape}, "
                    f"got {tuple(initial_coords.shape)}"
                )
            return initial_coords
        # Derive the noise shape from a real conditioning tensor (reference_positions is
        # [..., N_atom, 3]) rather than a literal that embeds the symbolic atom count, so
        # torch.compile with dynamic shapes can trace this. Insert the sample axis at -3.
        noise_template = insert_sample_axis(reference, dim=-3, num_samples=num_samples)
        noise = draw_normal_like(noise_template, generator)
        return noise_level * noise

    @staticmethod
    def _coords_chunk(
        coords: Tensor | None,
        *,
        reference_positions: Tensor,
        num_samples: int,
        sample_start: int,
        num_chunk_samples: int,
        name: str,
    ) -> Tensor | None:
        r"""
        Slice sampled coordinates for one chunk, or pass unsampled coordinates through.

        Args:
            coords: Optional sampled or unsampled coordinates.
            reference_positions: ``[..., N_atom, 3]`` coordinates supplying the expected shapes.
            num_samples: Total number of samples across all chunks.
            sample_start: Index of this chunk's first sample.
            num_chunk_samples: Number of samples in this chunk.
            name: Argument name used in shape errors.

        Returns:
            The chunk's coordinates, or ``None`` when ``coords`` is ``None``.

        Raises:
            ValueError: If ``coords`` has an unexpected shape.
        """

        if coords is None:
            return None

        reference_shape = tuple(reference_positions.shape)
        unsampled_shape = reference_shape
        if tuple(coords.shape) == tuple(unsampled_shape):
            return coords

        sampled_shape = (*reference_shape[:-2], num_samples, *reference_shape[-2:])
        if tuple(coords.shape) != tuple(sampled_shape):
            raise ValueError(f"{name} must have shape {sampled_shape} or {unsampled_shape}, got {tuple(coords.shape)}")

        return torch.narrow(coords, -3, sample_start, num_chunk_samples)

    def _step(
        self,
        denoise: StepDenoiseFn,
        coords: Tensor,
        sigma_prev: Tensor,
        sigma_next: Tensor,
        *,
        atom_mask: Tensor | None = None,
        noise: Tensor,
        condition_coords: Tensor | None = None,
        condition_mask: Tensor | None = None,
        is_terminal: bool = False,
        churn_rate: float,
        noise_scale: float,
        step_scale: float,
    ) -> Tensor:
        r"""
        Run one EDM coordinate update from ``sigma_prev`` to ``sigma_next``.

        Optionally injects churn noise, denoises, realigns through
        [`_align`][ofoldx.modules.diffusion.sampling.CoordinateEdmSampler._align], and advances with the
        Euler step, which ``_correct`` may
        refine.

        Args:
            denoise: Denoiser called on the noisy coords.
            coords: Current coords of shape `[..., N_sample, N_atom, 3]`.
            sigma_prev: Noise level at the start of the step.
            sigma_next: Noise level at the end of the step.
            atom_mask: Optional ``[..., N_atom]`` mask forwarded to ``_align``.
            noise: Pre-drawn churn noise sampled after coordinate augmentation.
            condition_coords: Optional coords to keep fixed during denoising, already in the current frame.
            condition_mask: Optional atom mask selecting the fixed coords.
            is_terminal: Whether this step ends the schedule.
            churn_rate: Extra-noise fraction for this call.
            noise_scale: Churn-noise multiplier for this call.
            step_scale: Solver-step multiplier for this call.

        Returns:
            Updated coords of shape `[..., N_sample, N_atom, 3]`.
        """

        churn = torch.where(
            sigma_next > self.min_churn_noise_level,
            sigma_next.new_tensor(churn_rate),
            sigma_next.new_zeros(()),
        )
        sigma_hat = sigma_prev * (churn + 1)
        delta_noise = torch.sqrt(sigma_hat**2 - sigma_prev**2)

        if condition_mask is not None:
            noise = torch.where(
                _atom_mask_for_coords(condition_mask, noise).bool(),
                torch.zeros_like(noise),
                noise,
            )
        noisy_coords = coords + noise_scale * delta_noise * noise
        if condition_mask is not None and condition_coords is not None:
            noisy_coords = self._apply_condition(noisy_coords, condition_coords, condition_mask)

        # Broadcast over every axis ahead of the atom and coordinate axes, so this step works for
        # both an explicit ``[..., N_sample, N_atom, 3]`` layout and a flattened sample axis.
        level_shape = coords.shape[:-2]
        noise_level = sigma_hat.reshape((1,) * len(level_shape)).expand(level_shape)
        denoised_coords = denoise(noisy_coords, noise_level)
        noisy_coords = self._align(noisy_coords, denoised_coords, atom_mask=atom_mask)

        delta = self._derivative(noisy_coords, denoised_coords, noise_level)
        dt = sigma_next - sigma_hat
        coords_next = noisy_coords + step_scale * dt[..., None, None] * delta
        if condition_mask is not None and condition_coords is not None:
            coords_next = self._apply_condition(coords_next, condition_coords, condition_mask)
        return self._correct(
            denoise,
            noisy_coords,
            coords_next,
            delta,
            dt=dt,
            sigma_next=sigma_next,
            condition_coords=condition_coords,
            condition_mask=condition_mask,
            is_terminal=is_terminal,
            step_scale=step_scale,
        )

    def _correct(
        self,
        denoise: StepDenoiseFn,
        noisy_coords: Tensor,
        coords_next: Tensor,
        derivative: Tensor,
        *,
        dt: Tensor,
        sigma_next: Tensor,
        condition_coords: Tensor | None,
        condition_mask: Tensor | None,
        is_terminal: bool,
        step_scale: float,
    ) -> Tensor:
        r"""
        Return the coordinates that end a step, given its Euler prediction.

        This sampler ends every step at the Euler prediction. A sampler that corrects the prediction
        overrides this.

        Args:
            denoise: Denoiser called with coordinates and their noise levels.
            noisy_coords: Churn-noised coordinates the step started from, after alignment.
            coords_next: The Euler prediction, with any fixed coordinates pinned.
            derivative: The EDM derivative at ``noisy_coords``.
            dt: Noise-level difference the step covers.
            sigma_next: Noise level at the end of the step.
            condition_coords: Optional coords to keep fixed, already in the current frame.
            condition_mask: Optional atom mask selecting the fixed coords.
            is_terminal: Whether this step ends the schedule.
            step_scale: Solver-step multiplier for this call.

        Returns:
            ``coords_next`` unchanged.
        """

        del denoise, noisy_coords, derivative, dt, sigma_next, condition_coords, condition_mask, is_terminal
        del step_scale
        return coords_next

    @staticmethod
    def _derivative(noisy_coords: Tensor, denoised_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""
        Return the EDM ODE derivative ``(noisy - denoised) / sigma`` used by the solver step.
        """

        return (noisy_coords - denoised_coords) / noise_level[..., None, None]

    def _align(self, noisy_coords: Tensor, denoised_coords: Tensor, *, atom_mask: Tensor | None) -> Tensor:
        r"""
        Return the noisy coordinates the solver step should differentiate against.

        This sampler differentiates against the noisy coordinates exactly as drawn. A family whose
        released sampler rigidly realigns them onto the denoiser output each step overrides this.

        Args:
            noisy_coords: Churn-noised coordinates of shape `[..., N_sample, N_atom, 3]`.
            denoised_coords: Denoiser output with the same shape.
            atom_mask: Optional ``[..., N_atom]`` atom validity mask.

        Returns:
            ``noisy_coords`` unchanged.
        """

        del denoised_coords, atom_mask
        return noisy_coords

    def _augment(
        self,
        coords: Tensor,
        *,
        atom_mask: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        r"""
        Move the coordinates into the frame one step runs in.

        This sampler centers the coordinates and applies a fresh random rotation and translation before every
        step. A sampler that keeps one frame throughout overrides this.

        Args:
            coords: Coordinates of shape `[..., N_sample, N_atom, 3]`.
            atom_mask: Optional ``[..., N_atom]`` mask used when centering.
            generator: Optional RNG for the random rotation and translation.

        Returns:
            coords: The coordinates in the step's frame.
            transform: The applied ``(rotation, translation, center)``, or ``None`` when the frame is unchanged.
        """

        coords, transform = self._center_random_augmentation(
            coords,
            num_samples=1,
            atom_mask=atom_mask,
            generator=generator,
        )
        return coords.squeeze(dim=-3), transform

    def _unaugment(self, coords: Tensor, transform: tuple[Tensor, Tensor, Tensor] | None) -> Tensor:
        r"""
        Return the coordinates of a finished step in the frame sampling continues from.

        This sampler continues from the frame the step ran in. A sampler that undoes each step's
        ``_augment`` overrides this.

        Args:
            coords: Coordinates of shape `[..., N_sample, N_atom, 3]` after the step.
            transform: The transform the step's ``_augment`` applied, or ``None``.

        Returns:
            ``coords`` unchanged.
        """

        del transform
        return coords

    @staticmethod
    def _center_random_augmentation(
        coords: Tensor,
        num_samples: int = 1,
        translation_scale: float = 1.0,
        center_only: bool = False,
        atom_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        r"""
        Center and optionally rigidly augment coordinates, retaining the inverse transform.
        """

        device = coords.device

        if atom_mask is None:
            observed = torch.ones_like(coords[..., 0], dtype=torch.bool)
        else:
            if atom_mask.dtype != torch.bool:
                raise TypeError("coordinate augmentation atom_mask must be boolean")
            observed = _atom_mask_for_coords(atom_mask, coords).squeeze(-1) & torch.ones_like(
                coords[..., 0], dtype=torch.bool
            )
        coords = project_coordinates(coords, observed)
        center = weighted_mean(coords.transpose(-1, -2), observed.unsqueeze(-2)).unsqueeze(-2)
        center = torch.where(observed.any(-1, keepdim=True).unsqueeze(-1), center, 0)
        coords = project_coordinates(coords - center, observed)

        coords = insert_sample_axis(coords, dim=-3, num_samples=num_samples)
        center = insert_sample_axis(center.squeeze(-2), dim=-2, num_samples=num_samples)
        if center_only:
            return coords, None

        num_augmentations = math.prod(coords.shape[:-2])
        batch_size_shape = coords.shape[:-3]
        random_device = device if generator is None else generator.device
        rotation = (
            uniform_random_rotation(
                num_samples=num_augmentations,
                device=random_device,
                dtype=coords.dtype,
                generator=generator,
            )
            .to(device)
            .reshape(*batch_size_shape, num_samples, 3, 3)
        )
        translation = translation_scale * draw_normal_like(center, generator)
        augmented = apply_rigid_transform(coords, rotation, translation)

        augmented = project_coordinates(augmented, insert_sample_axis(observed, dim=-2, num_samples=num_samples))
        return augmented, (rotation, translation, center)

    @staticmethod
    def _reverse_augmentation(coords: Tensor, transform: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        rotation, translation, center = transform
        if rotation.shape[-3] == 1:
            rotation = rotation.squeeze(-3)
            translation = translation.squeeze(-2)
            center = center.squeeze(-2)
        coords = apply_inverse_rigid_transform(coords, rotation, translation)
        return coords + center[..., None, :]

    @staticmethod
    def _apply_augmentation(
        coords: Tensor,
        transform: tuple[Tensor, Tensor, Tensor],
        *,
        target: Tensor,
    ) -> Tensor:
        rotation, translation, center = transform
        if coords.dim() == target.dim() - 1:
            coords = insert_sample_axis(coords, dim=-3, num_samples=target.shape[-3])
        if rotation.shape[-3] == 1 and rotation.dim() == coords.dim() + 1:
            rotation = rotation.squeeze(-3)
            translation = translation.squeeze(-2)
            center = center.squeeze(-2)
        coords = coords.to(device=target.device) - center[..., None, :]
        return apply_rigid_transform(coords, rotation, translation)

    @staticmethod
    def _apply_condition(coords: Tensor, condition_coords: Tensor, condition_mask: Tensor) -> Tensor:
        if condition_coords.dim() == coords.dim() - 1:
            condition_coords = condition_coords.unsqueeze(-3)
        return torch.where(
            _atom_mask_for_coords(condition_mask, coords).bool(),
            condition_coords.to(device=coords.device),
            coords,
        )


class HeunCoordinateEdmSampler(CoordinateEdmSampler):
    r"""
    EDM sampler whose steps follow Heun's second-order method.

    After the Euler prediction the denoiser runs once more at the predicted coordinates, and the step
    advances along the mean of the two derivatives. The terminal step, which ends at noise level zero,
    stays an Euler step.
    """

    def _correct(
        self,
        denoise: StepDenoiseFn,
        noisy_coords: Tensor,
        coords_next: Tensor,
        derivative: Tensor,
        *,
        dt: Tensor,
        sigma_next: Tensor,
        condition_coords: Tensor | None,
        condition_mask: Tensor | None,
        is_terminal: bool,
        step_scale: float,
    ) -> Tensor:
        if is_terminal:
            return coords_next
        level_shape = coords_next.shape[:-2]
        next_noise_level = sigma_next.reshape((1,) * len(level_shape)).expand(level_shape)
        next_denoised_coords = denoise(coords_next, next_noise_level)
        next_derivative = self._derivative(coords_next, next_denoised_coords, next_noise_level)
        coords_next = noisy_coords + step_scale * dt[..., None, None] * 0.5 * (derivative + next_derivative)
        if condition_mask is not None and condition_coords is not None:
            coords_next = self._apply_condition(coords_next, condition_coords, condition_mask)
        return coords_next


class UnaugmentedCoordinateEdmSampler(CoordinateEdmSampler):
    r"""
    EDM sampler that keeps one coordinate frame throughout: no step centers, rotates, or translates the
    coordinates.
    """

    def _augment(
        self,
        coords: Tensor,
        *,
        atom_mask: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        del atom_mask, generator
        return coords, None


def draw_normal(
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


def draw_normal_like(reference: Tensor, generator: torch.Generator | None) -> Tensor:
    r"""
    Draw standard-normal values with the reference layout on the generator device.

    Restore the reference device after drawing; dtype and memory layout follow
    ``empty_like(reference)``.
    """

    random_device = reference.device if generator is None else generator.device
    return torch.empty_like(reference, device=random_device).normal_(generator=generator).to(reference.device)


def _atom_mask_for_coords(atom_mask: Tensor, coords: Tensor) -> Tensor:
    r"""
    Broadcast ``atom_mask`` to align with the atom and coordinate axes of ``coords``.

    Args:
        atom_mask: Validity mask whose last axis must match the atom axis of ``coords``.
        coords: Coordinate tensor of shape `[..., N_atom, 3]`.

    Returns:
        The mask reshaped to `[..., N_atom, 1]`.

    """

    while atom_mask.dim() < coords.dim() - 1:
        atom_mask = atom_mask.unsqueeze(-2)
    return atom_mask.unsqueeze(-1)
