r"""
Feed-forward computations shared across model architectures.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


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
