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
Feed-forward computations shared across model architectures.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN


class FeedForward(nn.Module):
    r"""
    Apply affine hidden stages with dropout and a named activation, then project.

    Each hidden projection is followed by its ``norm`` factory's normalization,
    then dropout and activation. Residual addition and output dropout belong to
    the enclosing transition or block. Hidden affine maps use ``bias``; an
    omitted ``output_bias`` follows it.
    ``num_hidden_layers`` is positive and counts hidden projections, excluding the final output
    projection. The default identity normalization makes each hidden stage
    ``activation(Dropout(Linear(hidden)))``.
    ``activation`` selects the ordinary Transformers ``ACT2FN`` function.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        num_hidden_layers: int = 1,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "relu",
        norm: Callable[[int], nn.Module] = nn.Identity,
        output_bias: bool | None = None,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        output_size = input_size if output_size is None else output_size
        output_bias = bias if output_bias is None else output_bias
        self.input_projection = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_norm = norm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[activation]
        self.intermediate_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=bias) for _ in range(num_hidden_layers - 1)
        )
        self.intermediate_norms = nn.ModuleList(norm(hidden_size) for _ in range(num_hidden_layers - 1))
        self.intermediate_dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_hidden_layers - 1))
        self.output_projection = nn.Linear(hidden_size, output_size, bias=output_bias)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.activation(self.input_dropout(self.input_norm(self.input_projection(hidden))))
        for projection, norm, dropout in zip(
            self.intermediate_projections, self.intermediate_norms, self.intermediate_dropouts, strict=True
        ):
            hidden = self.activation(dropout(norm(projection(hidden))))
        return self.output_projection(hidden)


class SwiGluFeedForward(nn.Module):
    r"""
    Project, apply a SwiGLU gate, and project to the output width.

    Normalization, conditioning, dropout, and residual addition belong to the
    enclosing transition or block.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        output_size = input_size if output_size is None else output_size
        self.gate_projection = nn.Linear(input_size, hidden_size, bias=bias)
        self.value_projection = nn.Linear(input_size, hidden_size, bias=bias)
        self.output_projection = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.gate_projection(hidden)) * self.value_projection(hidden))
