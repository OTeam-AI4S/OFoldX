r"""
Elucidated-diffusion-model coordinate sampling.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..atom import LocalAtomGeometry
from ..embedding import RelativePositionBuckets
from ..geometry import uniform_random_rotation
from .schedule import EdmScheduler
from .types import CoordinateDenoiseFn, CoordinateDenoisingInput


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

    Args:
        scheduler: Noise schedule to sample along; a default
            [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] is built when ``None``.
        solver: ODE solver for the update, either ``"euler"`` (first order) or ``"heun"`` (second order).
        churn_rate: Fraction of extra noise injected before each step to make sampling stochastic.
        min_churn_noise_level: Churn is only applied while the next noise level exceeds this threshold.
        noise_scale: Multiplier on the injected churn noise standard deviation.
        step_scale: Multiplier applied to each solver step.
        center_random_augmentation: If ``True``, apply a random rigid augmentation (plus centering) each step;
            otherwise only center.
        reverse_augmentation: If ``True``, reverse each per-step random rigid augmentation after the solver update.

    Raises:
        ValueError: If ``solver`` is not ``"euler"`` or ``"heun"``.
    """

    def __init__(
        self,
        *,
        scheduler: EdmScheduler | None = None,
        solver: Literal["euler", "heun"] = "euler",
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        center_random_augmentation: bool = True,
        reverse_augmentation: bool = False,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"unsupported EDM solver: {solver!r}")
        self.scheduler = EdmScheduler() if scheduler is None else scheduler
        self.solver = solver
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.center_random_augmentation = center_random_augmentation
        self.reverse_augmentation = reverse_augmentation

    def augment_target(
        self,
        coords: Tensor,
        *,
        num_samples: int,
        atom_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Center coordinates and optionally apply random rigid transforms.

        Args:
            coords: Coordinates with shape ``[..., N_atom, 3]``.
            num_samples: Number of augmented copies to draw.
            atom_mask: Optional ``[..., N_atom]`` validity mask used for centering.
            generator: Random generator used for augmentation.

        Returns:
            Coordinates with shape ``[..., num_samples, N_atom, 3]``.
        """

        augmented, _ = self._center_random_augmentation(
            coords,
            num_samples=num_samples,
            center_only=not self.center_random_augmentation,
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
            step_scale: Optional per-call solver-step multiplier.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Sampled coordinates with shape ``[..., num_samples, N_atom, 3]``.
        """

        return self._sample(
            denoise,
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            initial_coords=initial_coords,
            condition_coords=condition_coords,
            condition_mask=condition_mask,
            max_noise_level=max_noise_level,
            churn_rate=self.churn_rate if churn_rate is None else churn_rate,
            step_scale=self.step_scale if step_scale is None else step_scale,
            generator=generator,
        )

    def _sample(
        self,
        denoise: CoordinateDenoiseFn,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None,
        initial_coords: Tensor | None,
        condition_coords: Tensor | None,
        condition_mask: Tensor | None,
        max_noise_level: float | None,
        churn_rate: float,
        step_scale: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Drive the reverse-diffusion loop, optionally chunked, and return coordinates.

        Args:
            denoise: Denoiser called at each step.
            context: Static conditioning tensors.
            num_samples: Total number of samples to draw.
            num_sampling_steps: Number of reverse-diffusion steps.
            sample_chunk_size: Optional maximum number of candidates passed to the
                denoiser together.
            initial_coords: Optional starting coords.
            condition_coords: Optional fixed coordinates pinned after each step.
            condition_mask: Optional atom mask selecting fixed coordinates.
            max_noise_level: Optional per-call initial scheduler noise override.
            churn_rate: Extra-noise fraction for this call.
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
            device=context.device,
            dtype=context.dtype,
            max_noise_level=max_noise_level,
        )

        coords = self._initial_coords(
            context=context,
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
            augmentation_transform = None
            if self.center_random_augmentation:
                coords, augmentation_transform = self._center_random_augmentation(
                    coords,
                    num_samples=1,
                    atom_mask=context.atom_mask,
                    generator=generator,
                )
                coords = coords.squeeze(dim=-3)

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
            noise = _draw_normal_like(coords, generator)
            chunks: list[Tensor] = []
            for sample_start in range(0, num_samples, chunk_size):
                chunk_num_samples = min(chunk_size, num_samples - sample_start)
                sample_end = sample_start + chunk_num_samples
                chunk_coords = coords[..., sample_start:sample_end, :, :]
                chunk_noise = noise[..., sample_start:sample_end, :, :]
                chunk_condition_coords = self._coords_chunk(
                    step_condition_coords,
                    context=context,
                    num_samples=num_samples,
                    sample_start=sample_start,
                    chunk_num_samples=chunk_num_samples,
                    name="condition_coords",
                )
                chunk_coords = self._step(
                    denoise,
                    chunk_coords,
                    sigma_prev,
                    sigma_next,
                    context=context,
                    noise=chunk_noise,
                    condition_coords=chunk_condition_coords,
                    condition_mask=condition_mask,
                    is_terminal=step_index == num_transitions - 1,
                    churn_rate=churn_rate,
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

            if self.reverse_augmentation and augmentation_transform is not None:
                coords = self._reverse_augmentation(coords, augmentation_transform)
            if condition_mask is not None and condition_coords is not None:
                coords = self._apply_condition(coords, condition_coords, condition_mask)
        return coords

    @staticmethod
    def _initial_coords(
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        noise_level: Tensor,
        initial_coords: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Seed the sampling loop with either validated ``initial_coords`` or fresh noise at ``noise_level``.

        Args:
            context: Static conditioning tensors (source of batch shape / atom count / device / dtype).
            num_samples: Number of samples to seed.
            noise_level: First noise level, scaling the fresh Gaussian noise.
            initial_coords: Optional user-provided starting coords.
            generator: Optional RNG.

        Returns:
            Seed coords of shape `[..., num_samples, N_atom, 3]`.

        Raises:
            ValueError: If ``initial_coords`` has an unexpected shape.
        """

        if initial_coords is not None:
            expected_shape = (*context.batch_shape, num_samples, context.num_atoms, 3)
            unsampled_shape = (*context.batch_shape, context.num_atoms, 3)
            if tuple(initial_coords.shape) == tuple(unsampled_shape):
                initial_coords = initial_coords.unsqueeze(dim=-3).expand(*expected_shape)
            elif tuple(initial_coords.shape) != tuple(expected_shape):
                raise ValueError(
                    f"initial_coords must have shape {expected_shape} or {unsampled_shape}, "
                    f"got {tuple(initial_coords.shape)}"
                )
            return initial_coords
        # Derive the noise shape from a real conditioning tensor (reference_positions is
        # [..., N_atom, 3]) rather than a literal that embeds the symbolic atom count, so
        # torch.compile with dynamic shapes can trace this. Insert the sample axis at -3.
        reference = context.reference_positions
        noise_template = reference.unsqueeze(-3).expand(*context.batch_shape, num_samples, reference.shape[-2], 3)
        noise = _draw_normal_like(noise_template, generator)
        return noise_level * noise

    @staticmethod
    def _coords_chunk(
        coords: Tensor | None,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        sample_start: int,
        chunk_num_samples: int,
        name: str,
    ) -> Tensor | None:
        r"""
        Slice sampled coordinates for one chunk, or pass unsampled coordinates through.

        Args:
            coords: Optional sampled or unsampled coordinates.
            context: Static conditioning tensors.
            num_samples: Total number of samples across all chunks.
            sample_start: Index of this chunk's first sample.
            chunk_num_samples: Number of samples in this chunk.
            name: Argument name used in shape errors.

        Returns:
            The chunk's coordinates, or ``None`` when ``coords`` is ``None``.

        Raises:
            ValueError: If ``coords`` has an unexpected shape.
        """

        if coords is None:
            return None

        unsampled_shape = (*context.batch_shape, context.num_atoms, 3)
        if tuple(coords.shape) == tuple(unsampled_shape):
            return coords

        sampled_shape = (*context.batch_shape, num_samples, context.num_atoms, 3)
        if tuple(coords.shape) != tuple(sampled_shape):
            raise ValueError(f"{name} must have shape {sampled_shape} or {unsampled_shape}, got {tuple(coords.shape)}")

        sample_end = sample_start + chunk_num_samples
        return coords[..., sample_start:sample_end, :, :]

    def _step(
        self,
        denoise: CoordinateDenoiseFn,
        coords: Tensor,
        sigma_prev: Tensor,
        sigma_next: Tensor,
        *,
        context: CoordinateDenoiseContext,
        noise: Tensor,
        condition_coords: Tensor | None = None,
        condition_mask: Tensor | None = None,
        is_terminal: bool = False,
        churn_rate: float,
        step_scale: float,
    ) -> Tensor:
        r"""
        Run one EDM coordinate update from ``sigma_prev`` to ``sigma_next``.

        Optionally injects churn noise, denoises, and advances with the Euler step (or the Heun correction unless
        this is the terminal step).

        Args:
            denoise: Denoiser called on the noisy coords.
            coords: Current coords of shape `[..., N_sample, N_atom, 3]`.
            sigma_prev: Noise level at the start of the step.
            sigma_next: Noise level at the end of the step.
            context: Static conditioning tensors.
            noise: Pre-drawn churn noise sampled after coordinate augmentation.
            condition_coords: Optional coords to keep fixed during denoising, already in the current frame.
            condition_mask: Optional atom mask selecting the fixed coords.
            is_terminal: If ``True``, skip the Heun correction even for the Heun solver.
            churn_rate: Extra-noise fraction for this call.
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
        noisy_coords = coords + self.noise_scale * delta_noise * noise
        if condition_mask is not None and condition_coords is not None:
            noisy_coords = self._apply_condition(noisy_coords, condition_coords, condition_mask)

        batch_shape = coords.shape[:-3]
        num_samples = coords.shape[-3]
        noise_level = (
            sigma_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, num_samples).to(coords.dtype)
        )
        denoised_coords = self._denoise(
            denoise,
            context=context,
            noisy_coords=noisy_coords,
            noise_level=noise_level,
        )

        delta = self._derivative(noisy_coords, denoised_coords, noise_level)
        dt = sigma_next - sigma_hat
        coords_next = noisy_coords + step_scale * dt[..., None, None] * delta
        if condition_mask is not None and condition_coords is not None:
            coords_next = self._apply_condition(coords_next, condition_coords, condition_mask)
        if self.solver == "euler" or is_terminal:
            return coords_next

        next_noise_level = (
            sigma_next.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, num_samples).to(coords.dtype)
        )
        next_denoised_coords = self._denoise(
            denoise,
            context=context,
            noisy_coords=coords_next,
            noise_level=next_noise_level,
        )
        next_delta = self._derivative(coords_next, next_denoised_coords, next_noise_level)
        coords_next = noisy_coords + step_scale * dt[..., None, None] * 0.5 * (delta + next_delta)
        if condition_mask is not None and condition_coords is not None:
            coords_next = self._apply_condition(coords_next, condition_coords, condition_mask)
        return coords_next

    @staticmethod
    def _derivative(noisy_coords: Tensor, denoised_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""
        Return the EDM ODE derivative ``(noisy - denoised) / sigma`` used by the solver step.
        """

        return (noisy_coords - denoised_coords) / noise_level[..., None, None]

    def _denoise(
        self,
        denoise: CoordinateDenoiseFn,
        *,
        context: CoordinateDenoiseContext,
        noisy_coords: Tensor,
        noise_level: Tensor,
    ) -> Tensor:
        r"""
        Call ``denoise`` on the per-step denoising input built from ``context``.
        """

        return denoise(context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level))

    @staticmethod
    def _center_random_augmentation(
        coords: Tensor,
        num_samples: int = 1,
        translation_scale: float = 1.0,
        center_only: bool = False,
        atom_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
        eps: float = 1e-12,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        r"""
        Center and optionally rigidly augment coordinates, retaining the inverse transform.
        """

        num_atoms = coords.size(-2)
        device = coords.device

        if atom_mask is None:
            center = torch.mean(input=coords, dim=-2, keepdim=True)
            coords = coords - center
        else:
            atom_mask = _atom_mask_for_coords(atom_mask, coords)
            center = (coords * atom_mask).sum(dim=-2, keepdim=True) / atom_mask.sum(dim=-2, keepdim=True).clamp_min(eps)
            coords = coords - center

        coords = _expand_at_dim(coords, dim=-3, size=num_samples)
        center = _expand_at_dim(center.squeeze(-2), dim=-2, size=num_samples)
        if center_only:
            return coords, None

        num_augmentations = torch.numel(coords[..., 0, 0])
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
        translation = translation_scale * _draw_normal_like(center, generator)
        augmented = _rot_vec_mul(
            rotation=_expand_at_dim(rotation, dim=-3, size=num_atoms),
            vectors=coords,
        )
        augmented = augmented + translation[..., None, :]

        if atom_mask is not None:
            augmented = augmented * atom_mask.unsqueeze(-3)
        return augmented, (rotation, translation, center)

    @staticmethod
    def _reverse_augmentation(coords: Tensor, transform: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        rotation, translation, center = transform
        if rotation.shape[-3] == 1:
            rotation = rotation.squeeze(-3)
            translation = translation.squeeze(-2)
            center = center.squeeze(-2)
        coords = coords - translation[..., None, :]
        coords = _rot_vec_mul(
            rotation=_expand_at_dim(rotation.transpose(-1, -2), dim=-3, size=coords.shape[-2]),
            vectors=coords,
        )
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
            coords = coords.unsqueeze(-3).expand_as(target)
        if rotation.shape[-3] == 1 and rotation.dim() == coords.dim() + 1:
            rotation = rotation.squeeze(-3)
            translation = translation.squeeze(-2)
            center = center.squeeze(-2)
        coords = coords.to(dtype=target.dtype) - center[..., None, :]
        augmented = _rot_vec_mul(
            rotation=_expand_at_dim(rotation, dim=-3, size=coords.shape[-2]),
            vectors=coords,
        )
        return augmented + translation[..., None, :]

    @staticmethod
    def _apply_condition(coords: Tensor, condition_coords: Tensor, condition_mask: Tensor) -> Tensor:
        if condition_coords.dim() == coords.dim() - 1:
            condition_coords = condition_coords.unsqueeze(-3)
        return torch.where(
            _atom_mask_for_coords(condition_mask, coords).bool(),
            condition_coords.to(dtype=coords.dtype),
            coords,
        )


def _rot_vec_mul(rotation: Tensor, vectors: Tensor) -> Tensor:
    r"""
    Apply a batched `[..., 3, 3]` rotation to `[..., 3]` vectors without materializing a matmul.

    Args:
        rotation: Rotation matrices of shape `[..., 3, 3]`.
        vectors: Vectors of shape `[..., 3]` to rotate.

    Returns:
        Rotated vectors of shape `[..., 3]`.
    """

    x, y, z = torch.unbind(input=vectors, dim=-1)
    return torch.stack(
        tensors=[
            rotation[..., 0, 0] * x + rotation[..., 0, 1] * y + rotation[..., 0, 2] * z,
            rotation[..., 1, 0] * x + rotation[..., 1, 1] * y + rotation[..., 1, 2] * z,
            rotation[..., 2, 0] * x + rotation[..., 2, 1] * y + rotation[..., 2, 2] * z,
        ],
        dim=-1,
    )


def _draw_normal_like(reference: Tensor, generator: torch.Generator | None) -> Tensor:
    r"""
    Draw standard-normal values shaped like ``reference`` on the generator device.
    """

    random_device = reference.device if generator is None else generator.device
    return torch.empty_like(reference, device=random_device).normal_(generator=generator).to(reference.device)


def _expand_at_dim(tensor: Tensor, dim: int, size: int) -> Tensor:
    r"""
    Insert a new axis of length ``size`` at ``dim`` via unsqueeze + expand (no data copy).

    Args:
        tensor: Tensor to expand.
        dim: Position (may be negative) at which to insert the new axis.
        size: Length of the inserted axis.

    Returns:
        A view of ``tensor`` with the new broadcast axis.
    """

    tensor = tensor.unsqueeze(dim=dim)
    if dim < 0:
        dim = tensor.dim() + dim
    before_shape = tensor.shape[:dim]
    after_shape = tensor.shape[dim + 1 :]
    return tensor.expand(*before_shape, size, *after_shape)


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
