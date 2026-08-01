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

#: Callable that predicts denoised coordinates from noisy coordinates and their noise levels alone.
#: A sampler drives this signature so it does not depend on any one family's conditioning bundle;
#: a caller holding a [`CoordinateDenoiseContext`][ofoldx.modules.diffusion.sampling.CoordinateDenoiseContext]
#: closes over it to produce one.
StepDenoiseFn: TypeAlias = Callable[[Tensor, Tensor], Tensor]
