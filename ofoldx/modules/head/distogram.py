r"""
Pairwise distance prediction.
"""

from torch import Tensor, nn


class DistogramHead(nn.Module):
    r"""Project pair representations to symmetric distance-bin logits."""

    def __init__(self, pair_size: int = 128, num_bins: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_size, num_bins, bias=bias)

    def forward(self, pair: Tensor) -> Tensor:
        logits = self.projection(pair)
        return logits + logits.transpose(-2, -3)


class PairSymmetrizedDistogramHead(nn.Module):
    r"""Symmetrize pair representations, then project them to distance-bin logits."""

    def __init__(self, pair_size: int = 128, num_bins: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_size, num_bins, bias=bias)

    def forward(self, pair: Tensor) -> Tensor:
        return self.projection(pair + pair.transpose(-2, -3))
