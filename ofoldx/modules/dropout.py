r"""
Structured dropout modules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StructuredDropout(nn.Module):
    r"""
    Apply dropout with a mask shared across one or more tensor axes.

    Args:
        probability: Probability of dropping an element from the sampled mask.
        broadcast_dims: Axes along which one mask value is shared.
    """

    def __init__(self, probability: float = 0.15, *, broadcast_dims: int | tuple[int, ...]) -> None:
        super().__init__()
        self.broadcast_dims = (broadcast_dims,) if isinstance(broadcast_dims, int) else broadcast_dims
        self.dropout = nn.Dropout(probability)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Apply an expectation-preserving structured dropout mask.

        Args:
            hidden: Input tensor.

        Returns:
            Dropped and rescaled tensor, or the input in evaluation mode.
        """

        if not self.training or self.dropout.p == 0:
            return hidden
        shape_carrier = hidden
        for dim in self.broadcast_dims:
            shape_carrier = torch.narrow(shape_carrier, dim, 0, 1)
        return hidden * self.dropout(torch.ones_like(shape_carrier))
