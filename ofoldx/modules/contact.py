r"""Contact-conditioning operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .embedding.fourier import FrozenFourierEmbedding


class ContactConditioning(nn.Module):
    r"""Encode contact classes and distance thresholds as an additive pair bias."""

    def __init__(
        self,
        pair_size: int,
        *,
        cutoff_min: float,
        cutoff_max: float,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.threshold_fourier = FrozenFourierEmbedding(pair_size)
        self.constraint_encoder = nn.Linear(pair_size + num_classes - 1, pair_size)
        self.unspecified_bias = nn.Parameter(torch.zeros(pair_size))
        self.unselected_bias = nn.Parameter(torch.zeros(pair_size))

    def forward(self, contact_class: Tensor, contact_threshold: Tensor) -> Tensor:
        threshold = (contact_threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        encoded = self.constraint_encoder(
            torch.cat([contact_class[..., 2:], threshold.unsqueeze(-1), self.threshold_fourier(threshold)], dim=-1)
        )
        active = 1 - contact_class[..., :2].sum(dim=-1, keepdim=True)
        return (
            encoded * active
            + self.unspecified_bias * contact_class[..., :1]
            + self.unselected_bias * contact_class[..., 1:2]
        )
