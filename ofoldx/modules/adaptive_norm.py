r"""
Adaptive layer normalization modules.
"""

from __future__ import annotations

from torch import Tensor, nn


class BoltzAdaptiveLayerNorm(nn.Module):
    r"""
    Boltz-style AdaLN: affine-free hidden norm, conditioning norm, sigmoid scale, and bias shift.

    The submodule names intentionally match the previous fused ``oops.AdaptiveLayerNorm`` layout used by the
    checkpoint converters: ``conditioning_norm``, ``gate_projection``, and ``shift_projection``.
    """

    def __init__(self, hidden_size: int, conditioning_size: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, bias=False)
        self.conditioning_norm = nn.LayerNorm(conditioning_size, bias=False)
        self.gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.shift_projection = nn.Linear(conditioning_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        hidden = self.hidden_norm(hidden)
        conditioning = self.conditioning_norm(conditioning)
        return self.sigmoid(self.gate_projection(conditioning)) * hidden + self.shift_projection(conditioning)
