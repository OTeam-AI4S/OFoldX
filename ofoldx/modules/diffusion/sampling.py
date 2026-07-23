r"""
EDM coordinate sampling.

The reverse-diffusion loop that turns Gaussian noise into atom coords by repeatedly calling a caller-supplied
denoiser along an [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] noise schedule. Supports the
Euler and Heun solvers, stochastic churn, per-step center-random augmentation, and chunked sampling of many draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..atom import LocalAtomGeometry
from ..embedding import RelativePositionBuckets
from ..geometry import uniform_random_rotation
from .schedule import EdmScheduler
from .types import CoordinateDenoiseFn, CoordinateDenoisingInput


@dataclass
class CoordinateDenoiseContext:
    r"""
    Step-invariant conditioning tensors shared across every coordinate-denoising step.

    Assembled once before the sampling loop and reused for all steps: each step calls
    [`to_denoising_input`][ofoldx.modules.diffusion.sampling.CoordinateDenoiseContext.to_denoising_input] to fold
    in that step's ``noisy_coords`` / ``noise_level`` and produce a
    [`CoordinateDenoisingInput`][ofoldx.modules.diffusion.types.CoordinateDenoisingInput]. The convenience
    properties read leading batch shape, device, dtype, and atom count off ``token_embedding`` / ``atom_to_token``.

    Args:
        atom_features: Per-atom input features.
        atom_to_token: Token index for each atom.
        reference_positions: Centered reference-conformer atom coords.
        reference_charges: Per-atom reference formal charges, shape `[..., N_atom, 1]`.
        reference_mask: Boolean mask for atoms with valid reference-conformer coordinates.
        atom_mask: Boolean atom validity mask.
        token_embedding: Token embedding before trunk updates.
        single: Token single representation.
        pair: Token pair representation.
        relative_features: Relative-position feature bundle.
        local_geometry: Optional local atom-pair geometry.
    """

    atom_features: Tensor
    atom_to_token: Tensor
    reference_positions: Tensor
    reference_charges: Tensor
    reference_mask: Tensor
    atom_mask: Tensor
    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets
    local_geometry: LocalAtomGeometry | None = None

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Leading batch dims of ``token_embedding`` (its shape minus the trailing ``(N_token, C)``).

        Returns:
            The batch portion of ``token_embedding``'s shape.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.embedding.position import RelativePositionBuckets
            >>> buckets = RelativePositionBuckets(
            ...     residue_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     token_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     same_entity=torch.zeros(1, 2, 2, dtype=torch.bool),
            ...     chain_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     residue_size=4,
            ...     chain_size=2,
            ...     dtype=torch.float32,
            ... )
            >>> context = CoordinateDenoiseContext(
            ...     atom_features=torch.zeros(1, 3, 8),
            ...     atom_to_token=torch.zeros(1, 3, dtype=torch.long),
            ...     reference_positions=torch.zeros(1, 3, 3),
            ...     reference_charges=torch.zeros(1, 3),
            ...     reference_mask=torch.ones(1, 3, dtype=torch.bool),
            ...     atom_mask=torch.ones(1, 3, dtype=torch.bool),
            ...     token_embedding=torch.zeros(1, 2, 4),
            ...     single=torch.zeros(1, 2, 4),
            ...     pair=torch.zeros(1, 2, 2, 4),
            ...     relative_features=buckets,
            ... )
            >>> tuple(context.batch_shape)
            (1,)
        """

        return self.token_embedding.shape[:-2]

    @property
    def device(self) -> torch.device:
        r"""
        Device of ``token_embedding``, taken as the device of the whole context.

        Returns:
            The device the conditioning tensors live on.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.embedding.position import RelativePositionBuckets
            >>> buckets = RelativePositionBuckets(
            ...     residue_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     token_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     same_entity=torch.zeros(1, 2, 2, dtype=torch.bool),
            ...     chain_bucket=torch.zeros(1, 2, 2, dtype=torch.long),
            ...     residue_size=4,
            ...     chain_size=2,
            ...     dtype=torch.float32,
            ... )
            >>> context = CoordinateDenoiseContext(
            ...     atom_features=torch.zeros(1, 3, 8),
            ...     atom_to_token=torch.zeros(1, 3, dtype=torch.long),
            ...     reference_positions=torch.zeros(1, 3, 3),
            ...     reference_charges=torch.zeros(1, 3),
            ...     reference_mask=torch.ones(1, 3, dtype=torch.bool),
            ...     atom_mask=torch.ones(1, 3, dtype=torch.bool),
            ...     token_embedding=torch.zeros(1, 2, 4),
            ...     single=torch.zeros(1, 2, 4),
            ...     pair=torch.zeros(1, 2, 2, 4),
            ...     relative_features=buckets,
            ... )
            >>> context.device
            device(type='cpu')
        """

        return self.token_embedding.device

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Dtype of ``token_embedding``, taken as the dtype of the whole context.

        Returns:
            The floating dtype of the conditioning tensors.
        """

        return self.token_embedding.dtype

    @property
    def num_atoms(self) -> int:
        r"""
        Number of atoms ``N_atom`` (length of the trailing axis of ``atom_to_token``).

        Returns:
            The atom count ``N_atom``.
        """

        return self.atom_to_token.size(-1)

    def to_denoising_input(self, noisy_coords: Tensor, noise_level: Tensor) -> CoordinateDenoisingInput:
        r"""
        Fold step-specific noisy coords and noise levels into the static context.

        Produces the per-step [`CoordinateDenoisingInput`][ofoldx.modules.diffusion.types.CoordinateDenoisingInput]
        the denoiser is called with, copying every conditioning tensor through unchanged.

        Args:
            noisy_coords: Noisy atom coords for this step, shape `[..., N_sample, N_atom, 3]`.
            noise_level: EDM noise level for this step, shape `[..., N_sample]`.

        Returns:
            A [`CoordinateDenoisingInput`][ofoldx.modules.diffusion.types.CoordinateDenoisingInput] carrying the
            static conditioning plus this step's ``noisy_coords`` and ``noise_level``.
        """

        return CoordinateDenoisingInput(
            atom_features=self.atom_features,
            atom_to_token=self.atom_to_token,
            token_embedding=self.token_embedding,
            single=self.single,
            pair=self.pair,
            relative_features=self.relative_features,
            noise_level=noise_level,
            reference_positions=self.reference_positions,
            reference_charges=self.reference_charges,
            reference_mask=self.reference_mask,
            atom_mask=self.atom_mask,
            noisy_coords=noisy_coords,
            local_geometry=self.local_geometry,
        )


class CoordinateEdmSampler(nn.Module):
    r"""
    EDM reverse-diffusion coordinate sampler with its noise schedule and stochastic update rule.

    Wraps an [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] and repeatedly applies a
    caller-supplied denoising callable over the noise
    schedule. Each step optionally injects stochastic "churn" noise, then advances the coords with either the Euler
    or Heun solver; per-step center-random augmentation and chunked multi-sample draws are supported.

    Args:
        scheduler: Noise schedule to sample along; a default
            [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] is built when ``None``.
        solver: ODE solver for the update, either ``"euler"`` (first order) or ``"heun"`` (second order).
        churn_rate: Fraction of extra noise injected before each step to make sampling stochastic.
        min_churn_noise_level: Churn is only applied while the next noise level exceeds this threshold.
        noise_scale: Multiplier on the injected churn noise standard deviation.
        step_scale: Multiplier on each solver step (the "gradient scaling" trick that over-steps the ODE).
        center_random_augmentation: If ``True``, apply a random rigid augmentation (plus centering) each step;
            otherwise only center.
        reverse_augmentation: If ``True``, reverse each per-step random rigid augmentation after the solver update.

    Raises:
        ValueError: If ``solver`` is not ``"euler"`` or ``"heun"``.

    Examples:
        >>> sampler = CoordinateEdmSampler(solver="euler")
        >>> sampler.solver
        'euler'
        >>> CoordinateEdmSampler(solver="rk4")
        Traceback (most recent call last):
            ...
        ValueError: unsupported EDM solver: 'rk4'
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
        Return ``num_samples`` centered, randomly rigid-augmented copies of ``coords``.

        Used by single-step diffusion training to produce per-sample ground-truth targets. Honors
        ``center_random_augmentation``: when disabled, only centering is applied (the example below relies on that to
        stay deterministic).

        Args:
            coords: Ground-truth coords of shape `[..., N_atom, 3]`.
            num_samples: Number of augmented copies to draw.
            atom_mask: Optional `[..., N_atom]` validity mask used for masked centering.
            generator: Optional RNG for reproducible augmentation.

        Returns:
            Augmented coords of shape `[..., num_samples, N_atom, 3]`.

        Examples:
            >>> import torch
            >>> sampler = CoordinateEdmSampler(center_random_augmentation=False)
            >>> coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            >>> sampler.augment_target(coords, num_samples=2).shape
            torch.Size([2, 2, 3])
            >>> sampler.augment_target(coords, num_samples=1).round(decimals=4).tolist()
            [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Run the EDM coordinate sampling loop and return the final coords.

        Draws ``num_samples`` structures per batch element (optionally in chunks of ``sample_chunk_size`` to bound
        memory), walking the full noise schedule with the configured solver and calling ``denoise`` at each step.

        Args:
            denoise: Denoiser mapping a
                [`CoordinateDenoisingInput`][ofoldx.modules.diffusion.types.CoordinateDenoisingInput] to denoised
                coords.
            context: Static conditioning tensors shared across steps.
            num_samples: Number of samples to draw per batch element.
            num_sampling_steps: Number of reverse-diffusion steps.
            sample_chunk_size: If set, sample in chunks of this many draws to cap peak memory.
            initial_coords: Optional starting coords of shape `[..., N_atom, 3]` or `[..., num_samples, N_atom, 3]`;
                fresh Gaussian noise is drawn when ``None``.
            condition_coords: Optional fixed coordinates pinned after every denoising step.
            condition_mask: Optional atom mask selecting coordinates to pin from ``condition_coords``.
            max_noise_level: Optional per-call initial scheduler noise override.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Sampled coords of shape `[..., num_samples, N_atom, 3]`.

        Examples:
            >>> import torch  # doctest: +SKIP
            >>> sampler = CoordinateEdmSampler()  # doctest: +SKIP
            >>> coords = sampler(denoise, context=context, num_samples=4, num_sampling_steps=200)  # doctest: +SKIP
            >>> coords.shape  # doctest: +SKIP
            torch.Size([1, 4, 128, 3])
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
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Drive the reverse-diffusion loop, optionally chunked, and return coordinates.

        Args:
            denoise: Denoiser called at each step.
            context: Static conditioning tensors.
            num_samples: Total number of samples to draw.
            num_sampling_steps: Number of reverse-diffusion steps.
            sample_chunk_size: Optional chunk size for memory-bounded sampling.
            initial_coords: Optional starting coords.
            condition_coords: Optional fixed coordinates pinned after each step.
            condition_mask: Optional atom mask selecting fixed coordinates.
            max_noise_level: Optional per-call initial scheduler noise override.
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

        def denoise_chunk(
            chunk_num_samples: int,
            chunk_initial_coords: Tensor | None,
            *,
            sample_start: int = 0,
        ) -> Tensor:
            r"""
            Run the full reverse-diffusion loop for one chunk of ``chunk_num_samples`` samples.

            Seeds ``chunk_num_samples`` noisy structures from ``chunk_initial_coords`` (or from the
            schedule's first noise level), then walks every consecutive ``(sigma_prev, sigma_next)``
            pair of ``noise_schedule`` applying optional center-random-augmentation and one
            ``_step`` update.

            Args:
                chunk_num_samples: Number of samples to draw in this chunk.
                chunk_initial_coords: Optional starting coords for the chunk; seeded from the schedule's
                    first noise level when ``None``.

            Returns:
                Coordinates shaped `[..., chunk_num_samples, N_atom, 3]`.
            """

            coords = self._initial_coords(
                context=context,
                num_samples=chunk_num_samples,
                noise_level=noise_schedule[0],
                initial_coords=chunk_initial_coords,
                generator=generator,
            )
            chunk_condition_coords = self._coords_chunk(
                condition_coords,
                context=context,
                num_samples=num_samples,
                sample_start=sample_start,
                chunk_num_samples=chunk_num_samples,
                name="condition_coords",
            )
            if condition_mask is not None and chunk_condition_coords is not None:
                coords = self._apply_condition(coords, chunk_condition_coords, condition_mask)
            num_transitions = noise_schedule.size(0) - 1
            for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
                augmentation_transform = None
                if self.center_random_augmentation:
                    coords, augmentation_transform = self._center_random_augmentation(
                        coords,
                        num_samples=1,
                        generator=generator,
                    )
                    coords = coords.squeeze(dim=-3)
                step_condition_coords = chunk_condition_coords
                if condition_mask is not None and chunk_condition_coords is not None:
                    if augmentation_transform is not None:
                        step_condition_coords = self._apply_augmentation(
                            chunk_condition_coords,
                            augmentation_transform,
                            target=coords,
                        )
                    coords = self._apply_condition(coords, step_condition_coords, condition_mask)
                noise = torch.empty_like(coords).normal_(generator=generator)
                coords = self._step(
                    denoise,
                    coords,
                    sigma_prev,
                    sigma_next,
                    context=context,
                    noise=noise,
                    condition_coords=step_condition_coords,
                    condition_mask=condition_mask,
                    is_terminal=step_index == num_transitions - 1,
                )
                if condition_mask is not None and step_condition_coords is not None:
                    coords = self._apply_condition(coords, step_condition_coords, condition_mask)
                if self.reverse_augmentation and augmentation_transform is not None:
                    coords = self._reverse_augmentation(coords, augmentation_transform)
                if condition_mask is not None and chunk_condition_coords is not None:
                    coords = self._apply_condition(coords, chunk_condition_coords, condition_mask)
            return coords

        if sample_chunk_size is None:
            return denoise_chunk(num_samples, initial_coords, sample_start=0)

        chunks = []
        num_chunks = num_samples // sample_chunk_size + (num_samples % sample_chunk_size != 0)
        for i in range(num_chunks):
            chunk_num_samples = sample_chunk_size if i < num_chunks - 1 else num_samples - i * sample_chunk_size
            chunk_initial_coords = self._coords_chunk(
                initial_coords,
                context=context,
                num_samples=num_samples,
                sample_start=i * sample_chunk_size,
                chunk_num_samples=chunk_num_samples,
                name="initial_coords",
            )
            coords = denoise_chunk(chunk_num_samples, chunk_initial_coords, sample_start=i * sample_chunk_size)
            chunks.append(coords)

        return torch.cat(chunks, dim=-3)

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
        noise = torch.empty_like(noise_template).normal_(generator=generator)
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
            raise ValueError(
                f"{name} must have shape {sampled_shape} or {unsampled_shape}, " f"got {tuple(coords.shape)}"
            )

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

        Returns:
            Updated coords of shape `[..., N_sample, N_atom, 3]`.
        """

        churn = torch.where(
            sigma_next > self.min_churn_noise_level,
            sigma_next.new_tensor(self.churn_rate),
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
        coords_next = noisy_coords + self.step_scale * dt[..., None, None] * delta
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
        coords_next = noisy_coords + self.step_scale * dt[..., None, None] * 0.5 * (delta + next_delta)
        if condition_mask is not None and condition_coords is not None:
            coords_next = self._apply_condition(coords_next, condition_coords, condition_mask)
        return coords_next

    @staticmethod
    def _derivative(noisy_coords: Tensor, denoised_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""Return the EDM ODE derivative ``(noisy - denoised) / sigma`` used by the solver step."""
        return (noisy_coords - denoised_coords) / noise_level[..., None, None]

    def _denoise(
        self,
        denoise: CoordinateDenoiseFn,
        *,
        context: CoordinateDenoiseContext,
        noisy_coords: Tensor,
        noise_level: Tensor,
    ) -> Tensor:
        r"""Call ``denoise`` on the per-step denoising input built from ``context``."""
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
        r"""Center and optionally rigidly augment coordinates, retaining the inverse transform."""

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
        rotation = uniform_random_rotation(
            num_samples=num_augmentations,
            device=device,
            dtype=coords.dtype,
            generator=generator,
        ).reshape(*batch_size_shape, num_samples, 3, 3)
        translation = translation_scale * torch.randn(
            size=(*batch_size_shape, num_samples, 3),
            generator=generator,
            device=device,
            dtype=coords.dtype,
        )
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
