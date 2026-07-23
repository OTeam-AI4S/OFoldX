r"""
Encoder modules built from self-attention blocks.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..backend import LayerNorm
from .multihead import GatedMultiheadAttention


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
        operator_backend: Operator provider; see [ofoldx.modules.backend][].

    Examples:
        >>> import torch
        >>> attention = PairBiasedSelfAttention(
        ...     hidden_size=8, pair_size=4, num_attention_heads=2, operator_backend="torch"
        ... )
        >>> _ = torch.manual_seed(0)
        >>> hidden = torch.randn(1, 3, 8)
        >>> pair = torch.randn(1, 3, 3, 4)
        >>> attention(hidden, pair).shape
        torch.Size([1, 3, 8])
    """

    def __init__(
        self,
        hidden_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.hidden_norm = LayerNorm(hidden_size, operator_backend=operator_backend)
        self.attention = GatedMultiheadAttention(
            hidden_size,
            num_attention_heads=num_attention_heads,
            query_bias=True,
            operator_backend=operator_backend,
        )
        self.pair_norm = LayerNorm(pair_size, bias=True, operator_backend=operator_backend)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)

    def forward(
        self,
        hidden: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Layer-norm the inputs, project the pair bias, and return an additive attention update.

        The normalized ``pair`` is linearly projected to ``num_attention_heads`` channels and moved to
        `[..., num_heads, N, N]` as the additive per-head bias; ``hidden`` then attends over its ``N``
        tokens under that bias.

        Args:
            hidden: Single representation `[..., N, hidden_size]`.
            pair: Pair representation `[..., N, N, pair_size]` supplying the attention bias.
            pair_mask: Optional boolean keep-mask `[..., N, N]` or `[..., N]`; ``False`` key positions
                are excluded from the softmax.

        Returns:
            Additive attention update `[..., N, hidden_size]`; the caller owns the outer residual.

        Examples:
            >>> import torch
            >>> attention = PairBiasedSelfAttention(
            ...     hidden_size=8, pair_size=4, num_attention_heads=2, operator_backend="torch"
            ... )
            >>> _ = torch.manual_seed(0)
            >>> hidden = torch.randn(1, 3, 8)
            >>> pair = torch.randn(1, 3, 3, 4)
            >>> attention(hidden, pair).shape
            torch.Size([1, 3, 8])
        """

        normalized = self.hidden_norm(hidden)
        pair_normalized = self.pair_norm(pair)
        bias = self.bias_projection(pair_normalized).movedim(-1, -3)
        return self.attention(normalized, biases=(bias,), attention_mask=pair_mask)
