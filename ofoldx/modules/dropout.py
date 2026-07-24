r"""
Structured dropout modules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StructuredDropout(nn.Module):
    r"""
    Apply dropout with an optional mask-broadcast axis.

    Args:
        probability: Probability of dropping an element from the sampled mask.
        broadcast_dim: Optional axis along which one mask value is shared.
    """

    def __init__(self, probability: float = 0.15, *, broadcast_dim: int | None = None) -> None:
        super().__init__()
        self.probability = probability
        self.broadcast_dim = broadcast_dim

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Apply an expectation-preserving structured dropout mask.

        Args:
            hidden: Input tensor.

        Returns:
            Dropped and rescaled tensor, or the input in evaluation mode.
        """

        if not self.training or self.probability == 0:
            return hidden
        shape = list(hidden.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = (torch.rand(shape, device=hidden.device) >= self.probability).to(hidden.dtype)
        return mask * hidden / (1.0 - self.probability)


class RowwiseDropout(nn.Module):
    r"""
    Apply dropout with a mask shared across the row axis.

    For an input shaped ``(..., num_rows, num_columns, hidden_size)``, the sampled
    mask has shape ``(..., 1, num_columns, hidden_size)`` and is broadcast to every
    row. Evaluation mode and zero probability are identities.

    Args:
        probability: Dropout probability for each element of the broadcast mask.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(probability)

    def forward(self, hidden: Tensor) -> Tensor:
        r"""
        Apply a dropout mask broadcast across the ``-3`` axis.

        Args:
            hidden: Tensor whose row axis is ``-3``.

        Returns:
            A tensor with the same shape as ``hidden``.
        """

        if not self.training or self.dropout.p == 0:
            return hidden
        shape = list(hidden.shape)
        shape[-3] = 1
        return hidden * self.dropout(hidden.new_ones(shape))
