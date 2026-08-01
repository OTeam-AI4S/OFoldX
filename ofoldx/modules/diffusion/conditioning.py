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
Diffusion-conditioning components.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import torch
from torch import Tensor, nn

from ..backend import LayerNorm
from ..backend.normalization import layer_norm_projection
from ..embedding.fourier import FrozenFourierEmbedding
from ..transition import Transition


class SingleConditioning(nn.Module):
    r"""
    Condition the trunk and input single representations on the diffusion noise level.

    The concatenated single representations are normalized and projected to ``output_size``, the
    projected Fourier features of the noise level are added along a sample axis inserted before the
    token axis, and residual transitions refine the result.

    Args:
        input_size: Width of the concatenated trunk and input single representations.
        output_size: Width of the conditioned single representation.
        fourier_embedding_size: Width of the noise-level Fourier features.
        num_transitions: Number of residual transitions.
        expansion: Transition expansion factor.
        backend: Backend selector forwarded to the normalizations and transitions.
        norm_bias: Whether single and noise normalization have an additive bias.
        single_projection_bias: Whether the prepared single projection has a bias.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.diffusion import SingleConditioning
        >>> conditioning = SingleConditioning(12, 8, fourier_embedding_size=4)
        >>> conditioning(torch.rand(2, 3), torch.randn(2, 5, 8), torch.randn(2, 5, 4)).shape
        torch.Size([2, 3, 5, 8])
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        fourier_embedding_size: int = 256,
        num_transitions: int = 2,
        expansion: int = 2,
        backend: str = "auto",
        norm_bias: bool = True,
        single_projection_bias: bool = True,
    ) -> None:
        super().__init__()
        self.single_norm = LayerNorm(input_size, bias=norm_bias, backend=backend)
        self.single_projection = nn.Linear(input_size, output_size, bias=single_projection_bias)
        self.noise_embedding = FrozenFourierEmbedding(fourier_embedding_size)
        self.noise_norm = LayerNorm(fourier_embedding_size, bias=norm_bias, backend=backend)
        self.noise_projection = nn.Linear(fourier_embedding_size, output_size, bias=False)
        self.transitions = nn.ModuleList(
            Transition(output_size, expansion=expansion, backend=backend) for _ in range(num_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> Tensor:
        r"""
        Args:
            noise_level: Noise levels, already scaled by the caller, `[..., N_sample]`.
            single_trunk: Trunk single representation `[..., N_token, C_trunk]`.
            single_inputs: Input single representation `[..., N_token, C_inputs]`.

        Returns:
            Conditioned single representation `[..., N_sample, N_token, output_size]`.
        """

        return self.condition_noise(noise_level, self.prepare(single_trunk, single_inputs))

    def prepare(self, single_trunk: Tensor, single_inputs: Tensor) -> Tensor:
        r"""Project sample-independent trunk and input representations onto the conditioning axis."""

        return layer_norm_projection(
            torch.cat([single_trunk, single_inputs], dim=-1),
            norm=self.single_norm,
            projection=self.single_projection,
        )

    def condition_noise(self, noise_level: Tensor, single: Tensor) -> Tensor:
        r"""Refine a prepared single representation with caller-scaled noise Fourier features."""

        noise = layer_norm_projection(
            self.noise_embedding(noise_level),
            norm=self.noise_norm,
            projection=self.noise_projection,
        )
        single = single.unsqueeze(-3) + noise.unsqueeze(-2)
        for transition in self.transitions:
            single = single + transition(single)
        return single


class PairwiseConditioning(nn.Module):
    r"""
    Condition a pair representation with relative-position features.

    Args:
        pair_size: Width of the pair representation.
        relative_position_size: Width of the relative-position features concatenated with the pair.
        num_transitions: Number of transition blocks refining the conditioned pair.
        expansion: Inner-width expansion factor of each transition block.
        backend: Public backend request forwarded to every transition.
    """

    def __init__(
        self,
        pair_size: int = 128,
        relative_position_size: int = 128,
        num_transitions: int = 2,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_norm = LayerNorm(pair_size + relative_position_size, backend=backend)
        self.input_projection = nn.Linear(
            pair_size + relative_position_size,
            pair_size,
            bias=False,
        )
        self.transitions = nn.ModuleList(
            Transition(pair_size, expansion=expansion, backend=backend) for _ in range(num_transitions)
        )

    def forward(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        r"""
        Apply relative-position conditioning to a pair representation.

        Args:
            pair_trunk: Trunk pair representation ``[..., N_token, N_token, pair_size]``.
            relative_position_encoding: Relative-position features
                ``[..., N_token, N_token, relative_position_size]``.

        Returns:
            Conditioned pair representation ``[..., N_token, N_token, pair_size]``.
        """

        return condition_pair(
            pair_trunk,
            relative_position_encoding,
            norm=self.input_norm,
            projection=self.input_projection,
            transitions=self.transitions,
        )


def condition_pair(
    pair_trunk: Tensor,
    relative_position_encoding: Tensor,
    *,
    norm: LayerNorm,
    projection: nn.Linear,
    transitions: Iterable[nn.Module],
    checkpoint: Callable[..., Tensor] | None = None,
) -> Tensor:
    r"""
    Condition a pair representation on relative-position features.

    Args:
        pair_trunk: ``[..., N_token, N_token, C_pair]`` trunk pair representation.
        relative_position_encoding: Embedded relative-position features with matching pair axes.
        norm: LayerNorm for concatenated pair features.
        projection: Linear projection to the output pair width.
        transitions: Transition modules applied residually.
        checkpoint: Optional callable applied independently to the input projection
            and each transition, supplied by the containing module.

    Returns:
        Conditioned pair representation with shape
            ``[..., N_token, N_token, C_pair]``.
    """

    project = partial(_project_pair, norm=norm, projection=projection)
    if checkpoint is None:
        pair = project(pair_trunk, relative_position_encoding)
    else:
        pair = checkpoint(project, pair_trunk, relative_position_encoding)
    for transition in transitions:
        pair = pair + (transition(pair) if checkpoint is None else checkpoint(transition, pair))
    return pair


def _project_pair(
    pair: Tensor,
    relative_position_encoding: Tensor,
    *,
    norm: LayerNorm,
    projection: nn.Linear,
) -> Tensor:
    return layer_norm_projection(
        torch.cat([pair, relative_position_encoding], dim=-1), norm=norm, projection=projection
    )
