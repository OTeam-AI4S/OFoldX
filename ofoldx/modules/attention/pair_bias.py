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
Pair-derived attention biases and self-attention.
"""

from __future__ import annotations

from typing import cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import LayerNorm
from ..backend.normalization import layer_norm_projection
from .multihead import GatedMultiheadAttentionProjection, MultiheadAttention


class PerBlockPairBiasProjection(nn.Module):
    r"""
    Project one pair representation to packed per-block attention-bias logits.

    Each block owns a layer normalization and a bias-free projection to its attention heads. The
    projected logits are concatenated over blocks, block-major on the last axis.

    Args:
        pair_size: Width of the pair representation.
        num_blocks: Number of attention blocks.
        num_attention_heads: Number of attention heads per block.
        bias: Whether every block's layer normalization has a learnable bias.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Examples:
        >>> import torch
        >>> from ofoldx.modules.attention import PerBlockPairBiasProjection
        >>> projection = PerBlockPairBiasProjection(pair_size=4, num_blocks=3, num_attention_heads=2)
        >>> projection(torch.randn(1, 5, 5, 4)).shape
        torch.Size([1, 5, 5, 6])
    """

    def __init__(
        self,
        *,
        pair_size: int,
        num_blocks: int,
        num_attention_heads: int,
        bias: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=bias, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )

    def forward(self, pair: Tensor | NestedTensor) -> Tensor | NestedTensor:
        r"""
        Project ``pair`` to the attention-bias logits of every block.

        Args:
            pair: Pair representation ``[..., pair_size]``.

        Returns:
            Logits ``[..., num_blocks * num_attention_heads]``.
        """

        if not self.bias_projections:
            return pair[..., :0]
        return torch.cat(
            [
                layer_norm_projection(pair, norm=cast(LayerNorm, norm), projection=cast(nn.Linear, projection))
                for norm, projection in zip(self.pair_norms, self.bias_projections, strict=True)
            ],
            dim=-1,
        )


class PairBiasedSelfAttention(nn.Module):
    r"""
    Self-attention over the single representation with the pair representation as per-head bias.

    Layer-norms the single and pair inputs, projects the normalized pair to a per-head additive
    attention bias, and runs gated self-attention. It returns an additive attention update; the
    caller block owns the outer residual.

    Args:
        hidden_size: Width of the single-representation token channel.
        pair_size: Width of the pair-representation channel used to derive the bias.
        num_attention_heads: Number of attention heads.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        hidden_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, bias=True, backend=backend)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(hidden_size, num_attention_heads=num_attention_heads, query_bias=True),
            backend=backend,
        )

    def forward(
        self,
        hidden: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Layer-norm the inputs, project the pair bias, and return an additive attention update.

        The normalized ``pair`` is linearly projected to ``num_attention_heads`` channels and moved to
        `[..., num_attention_heads, N, N]` as the additive per-head bias; ``hidden`` then attends over its ``N``
        tokens under that bias.

        Args:
            hidden: Single representation `[..., N, hidden_size]`.
            pair: Pair representation `[..., N, N, pair_size]` supplying the attention bias.
            pair_mask: Optional boolean keep-mask `[..., N, N]` or `[..., N]`; ``False`` key positions
                are excluded from the softmax.

        Returns:
            Additive attention update `[..., N, hidden_size]`; the caller owns the outer residual.
        """

        normalized = self.hidden_norm(hidden)
        bias = layer_norm_projection(pair, norm=self.pair_norm, projection=self.bias_projection).movedim(-1, -3)
        return self.attention(normalized, biases=(bias,), attention_mask=pair_mask)
