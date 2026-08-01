r"""
Pair-derived attention-bias projections.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..backend import LayerNorm


class PerBlockPairBiasProjection(nn.Module):
    r"""
    Project one pair representation to packed per-block attention-bias logits.

    Each block owns an affine layer normalization and a bias-free projection to
    its attention heads. The projected logits are concatenated over blocks.

    Args:
        pair_size: Width of the pair representation.
        num_blocks: Number of attention blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        *,
        pair_size: int,
        num_blocks: int,
        num_attention_heads: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=True, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )

    def forward(self, pair: Tensor) -> Tensor:
        return torch.cat(
            [projection(norm(pair)) for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)],
            dim=-1,
        )
