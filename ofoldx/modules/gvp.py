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
Geometric vector perceptrons, normalization and whole-vector channel dropout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GvpValue = tuple[Tensor, Tensor]
Activation = Callable[[Tensor], Tensor] | None


class Gvp(nn.Module):
    r"""
    Geometric vector perceptron over scalar and vector channels.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        *,
        scalar_activation: Activation = F.silu,
        vector_activation: Activation = None,
    ) -> None:
        super().__init__()
        input_scalar_size, input_vector_size = input_size
        output_scalar_size, output_vector_size = output_size
        self.input_vector_size = input_vector_size
        self.output_vector_size = output_vector_size
        if input_vector_size:
            hidden_vector_size = max(input_vector_size, output_vector_size)
            self.hidden_vector_projection = nn.Linear(
                input_vector_size,
                hidden_vector_size,
                bias=False,
            )
            self.scalar_projection = nn.Linear(
                hidden_vector_size + input_scalar_size,
                output_scalar_size,
            )
            if output_vector_size:
                self.output_vector_projection = nn.Linear(
                    hidden_vector_size,
                    output_vector_size,
                    bias=False,
                )
                self.vector_gate_projection = nn.Linear(
                    output_scalar_size,
                    output_vector_size,
                )
        else:
            self.scalar_projection = nn.Linear(input_scalar_size, output_scalar_size)
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        vector: Tensor | None = None
        if self.input_vector_size:
            scalar, vector = cast(GvpValue, input)
            vector = vector.transpose(-1, -2)
            hidden_vector = self.hidden_vector_projection(vector)
            vector_norm = _norm_no_nan(hidden_vector, dim=-2)
            scalar = self.scalar_projection(torch.cat((scalar, vector_norm), dim=-1))
            if self.output_vector_size:
                vector = self.output_vector_projection(hidden_vector).transpose(-1, -2)
                gate_input = scalar if self.vector_activation is None else self.vector_activation(scalar)
                gate = torch.sigmoid(self.vector_gate_projection(gate_input))
                vector = vector * gate.unsqueeze(-1)
        else:
            scalar = self.scalar_projection(cast(Tensor, input))
            if self.output_vector_size:
                vector = torch.zeros(
                    *scalar.shape[:-1],
                    self.output_vector_size,
                    3,
                    device=scalar.device,
                    dtype=scalar.dtype,
                )
        if self.scalar_activation is not None:
            scalar = self.scalar_activation(scalar)
        if self.output_vector_size:
            return scalar, cast(Tensor, vector)
        return scalar


class GvpDropout(nn.Module):
    r"""
    Scalar dropout plus whole-vector-channel dropout.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.scalar_dropout = nn.Dropout(probability)
        self.probability = probability

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        if isinstance(input, Tensor):
            return self.scalar_dropout(input)
        scalar, vector = input
        mask = F.dropout(
            torch.ones(vector.shape[:-1], device=vector.device, dtype=vector.dtype),
            p=self.probability,
            training=self.training,
        ).unsqueeze(-1)
        return self.scalar_dropout(scalar), mask * vector


class GvpLayerNorm(nn.Module):
    r"""
    Scalar layer normalization and vector-channel RMS normalization.
    """

    def __init__(self, feature_size: tuple[int, int]) -> None:
        super().__init__()
        self.vector_size = feature_size[1]
        self.scalar_norm = nn.LayerNorm(feature_size[0])

    def forward(self, input: Tensor | GvpValue) -> Tensor | GvpValue:
        if not self.vector_size:
            return self.scalar_norm(cast(Tensor, input))
        scalar, vector = cast(GvpValue, input)
        vector_norm = _norm_no_nan(vector, dim=-1, keepdim=True)
        vector_norm = torch.linalg.vector_norm(vector_norm / vector_norm.shape[-2] ** 0.5, dim=-2, keepdim=True)
        return self.scalar_norm(scalar), vector / vector_norm


def _norm_no_nan(
    value: Tensor,
    *,
    dim: int,
    keepdim: bool = False,
) -> Tensor:
    return torch.linalg.vector_norm(value, dim=dim, keepdim=keepdim).clamp_min(1e-4)
