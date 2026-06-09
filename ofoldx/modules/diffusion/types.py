r"""
Typed diffusion input contracts.

The dataclass here is the frozen-shape contract handed to a coordinate denoiser: it bundles the step-invariant
conditioning (atom features, trunk single/pair, relative-position features) with the per-step ``noisy_coords`` and
``noise_level`` so the denoiser signature stays a single typed argument.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..atom import LocalAtomGeometry
from ..embedding.position import RelativePositionBuckets


@dataclass(slots=True)
class CoordinateDenoisingInput:
    r"""
    Typed input bundle for a single coordinate-denoising step.

    Groups the static conditioning tensors together with the step-specific ``noisy_coords`` and ``noise_level`` that a
    [`CoordinateDenoiseFn`][ofoldx.modules.diffusion.sampling.CoordinateDenoiseFn] consumes. Built per step by
    [`CoordinateDenoiseContext.to_denoising_input`][ofoldx.modules.diffusion.sampling.CoordinateDenoiseContext].

    Attributes:
        atom_features: Per-atom input features.
        atom_to_token: Token index for each atom.
        token_embedding: Token embedding before trunk updates.
        single: Token single representation.
        pair: Token pair representation.
        relative_features: Relative-position feature bundle.
        noise_level: EDM noise level for the denoising step.
        reference_positions: Centered reference-conformer atom coords.
        reference_charges: Per-atom reference formal charges.
        atom_mask: Boolean atom validity mask.
        noisy_coords: Noisy atom coords to denoise.
        local_geometry: Optional local atom-pair geometry.
    """

    atom_features: Tensor
    atom_to_token: Tensor
    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets
    noise_level: Tensor
    reference_positions: Tensor
    reference_charges: Tensor
    atom_mask: Tensor
    noisy_coords: Tensor
    local_geometry: LocalAtomGeometry | None
