r"""
Coordinate-denoising input types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from tensordict import TypedTensorDict
from torch import Tensor

from ..atom import LocalAtomGeometry
from ..embedding.position import RelativePositionBuckets


class CoordinateDenoisingOutput(TypedTensorDict, dict):
    r"""Minimal tensors produced by a stochastic coordinate-denoising pass.

    Attributes:
        coordinates: Denoised coordinates with shape ``[..., S, N_atom, 3]``.
        augmented_target_coordinates: Target coordinates after random rigid
            augmentation, with shape ``[..., S, N_atom, 3]``.
        noise_level: Noise levels with shape ``[..., S]``.
    """

    coordinates: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor


@dataclass(eq=False)
class CoordinateDenoisingInput:
    r"""
    Inputs for one coordinate-denoising step.

    Attributes:
        atom_features: Per-atom features with shape ``[..., N_atom, C_atom]``.
        atom_to_token: Token index for each atom with shape ``[..., N_atom]``.
        token_embedding: Token embeddings with shape ``[..., N_token, C_token]``.
        single: Trunk single representation used by the atom path, with shape
            ``[..., N_token, C_single]``.
        conditioning_single: Single representation supplied to diffusion
            conditioning, with shape ``[..., N_token, C_single]``.
        conditioning_pair: Pair representation supplied to diffusion
            conditioning, with shape ``[..., N_token, N_token, C_pair]``.
        relative_features: Relative-position feature bundle.
        noise_level: Noise levels with shape ``[..., N_sample]``.
        reference_positions: Centered reference coordinates with shape
            ``[..., N_atom, 3]``.
        reference_charges: Formal charges with shape ``[..., N_atom, 1]``.
        reference_mask: Reference-coordinate mask with shape ``[..., N_atom]``.
        atom_mask: Atom validity mask with shape ``[..., N_atom]``.
        noisy_coords: Noisy coordinates with shape
            ``[..., N_sample, N_atom, 3]``.
        local_geometry: Local atom-window geometry.
    """

    atom_features: Tensor
    atom_to_token: Tensor
    token_embedding: Tensor
    single: Tensor
    conditioning_single: Tensor
    conditioning_pair: Tensor
    relative_features: RelativePositionBuckets
    noise_level: Tensor
    reference_positions: Tensor
    reference_charges: Tensor
    reference_mask: Tensor
    atom_mask: Tensor
    noisy_coords: Tensor
    local_geometry: LocalAtomGeometry


#: Callable that predicts denoised coordinates for one diffusion step.
CoordinateDenoiseFn: TypeAlias = Callable[[CoordinateDenoisingInput], Tensor]
