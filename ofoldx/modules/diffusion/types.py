r"""Coordinate-denoising input types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from torch import Tensor

from ..atom import LocalAtomGeometry
from ..embedding.position import RelativePositionBuckets


@dataclass
class CoordinateDenoisingInput:
    r"""
    Inputs for one coordinate-denoising step.

    Static conditioning accompanies the step-specific ``noisy_coords`` and
    ``noise_level``. Instances are built by
    [`CoordinateDenoiseContext.to_denoising_input`][ofoldx.modules.diffusion.sampling.CoordinateDenoiseContext].

    Args:
        atom_features: Per-atom input features.
        atom_to_token: Token index for each atom.
        token_embedding: Token embedding before trunk updates.
        single: Token single representation.
        pair: Token pair representation.
        relative_features: Relative-position feature bundle.
        noise_level: EDM noise level for the denoising step.
        reference_positions: Centered reference-conformer atom coords.
        reference_charges: Per-atom reference formal charges, shape `[..., N_atom, 1]`.
        reference_mask: Boolean mask for atoms with valid reference-conformer coordinates.
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
    reference_mask: Tensor
    atom_mask: Tensor
    noisy_coords: Tensor
    local_geometry: LocalAtomGeometry | None


CoordinateDenoiseFn: TypeAlias = Callable[[CoordinateDenoisingInput], Tensor]
