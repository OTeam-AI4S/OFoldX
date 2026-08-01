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
Triangle self-attention over pair representations.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..backend import LayerNorm
from ..backend.attention import triangle_attention as dispatch_triangle_attention
from .multihead import GatedMultiheadAttentionProjection, MultiheadAttention


class TriangleAttention(nn.Module):
    r"""
    Triangle self-attention over the pair representation with a pair-derived per-head bias.

    Layer-norms the pair representation, attends along the key axis with a per-head triangle bias
    that is shared across the starting-node (row) axis, and applies a gated update. This is the
    attention half of a Pairformer triangle block; the row/column direction is fixed by how the caller
    orients ``pair`` before the call.

    Args:
        pair_size: Width of the pair-representation channel (query/key/value and output width).
        head_size: Per-head width of the attention.
        num_attention_heads: Number of attention heads.
        output_bias: Whether the attention output projection carries a learnable bias.
        gate_bias: Whether the output-gate projection carries a learnable bias.
        gate_bias_init: Initial value for the gate bias when ``gate_bias`` is set.
        bias_pair_dims: Source pair axes mapped to the bias query/key axes,
            independently of the oriented query/key/value pair input.
        backend: Operator provider; see [ofoldx.modules.backend][]. The ``oops`` backend uses the
            dedicated OOps-owned ``triangle_attention`` implementation.
    """

    def __init__(
        self,
        pair_size: int = 128,
        head_size: int = 32,
        num_attention_heads: int = 4,
        *,
        output_bias: bool = False,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
        bias_pair_dims: tuple[int, int] = (-3, -2),
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.bias_pair_dims = bias_pair_dims
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(
                pair_size,
                num_attention_heads=num_attention_heads,
                head_size=head_size,
                query_bias=False,
                output_bias=output_bias,
                gate_bias=gate_bias,
                gate_bias_init=gate_bias_init,
            ),
            backend=backend,
        )

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Layer-norm the pair, attend along the key axis with a per-head triangle bias, and gate the update.

        The triangle bias is a per-head projection of the normalized pair under
        ``bias_pair_dims``, broadcast across the starting-node row axis. The
        default uses ``b_{jk} = Linear(norm(pair)_{jk})``. Masked keys get a
        ``-inf`` additive bias so they are excluded from the softmax.

        Args:
            pair: Pair representation `[..., N, N, pair_size]`.
            pair_mask: Optional boolean keep-mask `[..., N, N]`; ``False`` key positions are excluded.

        Returns:
            Additive attention update `[..., N, N, pair_size]`; the caller owns the outer residual.
        """

        normalized = self.pair_norm(pair)
        bias_source = normalized.movedim(self.bias_pair_dims, (-3, -2))
        projected_bias = self.bias_projection(bias_source)  # (..., n_query, n_key, num_attention_heads)
        triangle_bias = projected_bias.movedim(-1, -3).unsqueeze(-4)  # (..., 1, num_attention_heads, n_query, n_key)
        return dispatch_triangle_attention(
            normalized,
            triangle_bias,
            pair_mask,
            projection=self.attention.projection,
            backend=self.attention.backend,
        )
