r"""
Pairformer stack modules.

The Pairformer refines a single representation and a pair representation together, with no MSA
axis. Each block runs the pair triangle updates (multiplication + attention + transition) and,
when a single channel is present, updates the single representation with pair-biased
self-attention. Setting ``single_size = 0`` yields a pair-only stack.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..attention import (
    PairBiasedSelfAttention,
    TriangleAttention,
    TriangleMultiplication,
)
from ..transition import RowwiseDropout, Transition


class PairformerStack(nn.Module):
    r"""
    Stack of [`PairformerBlock`][ofoldx.modules.trunk.pairformer.PairformerBlock] run in sequence.

    The stack applies ``num_blocks`` identical blocks in sequence. When ``single_size`` is ``0``
    every block is pair-only and the stack accepts (and returns) ``None`` for the single representation.

    Args:
        num_blocks: Number of Pairformer blocks in the stack.
        single_size: Width of the single representation; ``0`` builds a pair-only stack.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair rather than the oriented attention input.
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> stack = PairformerStack(
        ...     num_blocks=1, single_size=8, pair_size=4,
        ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, num_pair_attention_heads=2,
        ...     expansion=2, dropout=0.0,
        ... ).eval()
        >>> single, pair = torch.randn(1, 3, 8), torch.randn(1, 3, 3, 4)
        >>> out_single, out_pair = stack(single, pair)
        >>> out_single.shape, out_pair.shape
        (torch.Size([1, 3, 8]), torch.Size([1, 3, 3, 4]))
    """

    def __init__(
        self,
        num_blocks: int = 48,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        ending_node_transpose_bias: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.single_size = single_size
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    single_size=single_size,
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    num_pair_attention_heads=num_pair_attention_heads,
                    expansion=expansion,
                    dropout=dropout,
                    ending_node_transpose_bias=ending_node_transpose_bias,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ],
        )

    def forward(
        self, single: Tensor | None, pair: Tensor, pair_mask: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor]:
        r"""
        Run every [`PairformerBlock`][ofoldx.modules.trunk.pairformer.PairformerBlock] in sequence.

        A pair-only stack (``single_size == 0``) is invoked with ``single=None`` and returns
        ``single`` as ``None``.

        Args:
            single: Single representation `[B, N, single_size]`, or ``None`` for a pair-only stack.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Refined ``(single, pair)`` of shapes `[B, N, single_size]` (or ``None``) and
            `[B, N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> stack = PairformerStack(
            ...     num_blocks=1, single_size=0, pair_size=4,
            ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2, dropout=0.0,
            ... ).eval()
            >>> single, pair = stack(None, torch.randn(1, 3, 3, 4))
            >>> single is None, pair.shape
            (True, torch.Size([1, 3, 3, 4]))
        """

        if self.single_size == 0:
            for block in self.blocks:
                _, pair = block(None, pair, pair_mask=pair_mask)
            return None, pair

        for block in self.blocks:
            single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class PairformerBlock(nn.Module):
    r"""
    One Pairformer block that jointly refines the single and pair representations.

    The pair representation is updated by outgoing then incoming triangle multiplication, starting- then
    ending-node triangle attention (each row-wise dropped and added back), and a pair transition. When
    ``single_size > 0`` the single representation is then updated by pair-biased self-attention followed by
    a single transition; ``single_size == 0`` makes the block pair-only.

    Args:
        single_size: Width of the single representation; ``0`` builds a pair-only block.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Feed-forward expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        ending_node_transpose_bias: Whether ending-node attention derives its pair bias from the
            pre-transpose pair rather than the oriented attention input.
        operator_backend: Operator backend selector forwarded to every submodule.

    Examples:
        >>> import torch
        >>> block = PairformerBlock(
        ...     single_size=8, pair_size=4,
        ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
        ...     num_triangle_attention_heads=2, num_pair_attention_heads=2,
        ...     expansion=2, dropout=0.0,
        ... ).eval()
        >>> single, pair = block(torch.randn(1, 3, 8), torch.randn(1, 3, 3, 4))
        >>> single.shape, pair.shape
        (torch.Size([1, 3, 8]), torch.Size([1, 3, 3, 4]))
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        ending_node_transpose_bias: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            operator_backend=operator_backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            transpose_bias=ending_node_transpose_bias,
            operator_backend=operator_backend,
        )
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.pair_transition = Transition(pair_size, expansion=expansion, operator_backend=operator_backend)
        self.pair_attention: PairBiasedSelfAttention | None
        self.single_transition: Transition | None
        if single_size > 0:
            self.pair_attention = PairBiasedSelfAttention(
                hidden_size=single_size,
                pair_size=pair_size,
                num_attention_heads=num_pair_attention_heads,
                operator_backend=operator_backend,
            )
            self.single_transition = Transition(single_size, expansion=expansion, operator_backend=operator_backend)
        else:
            self.pair_attention = None
            self.single_transition = None

    def forward(
        self, single: Tensor | None, pair: Tensor, pair_mask: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor]:
        r"""
        Apply one Pairformer update with residual connections.

        The pair representation is refined by outgoing then incoming triangle multiplication, starting-
        then ending-node triangle attention (each row-wise dropped and added back), and a pair transition.
        When ``single_size > 0`` the single representation is then updated by pair-biased self-attention
        and a single transition.

        Args:
            single: Single representation `[B, N, single_size]`. Required when ``single_size > 0``;
                pass ``None`` for a pair-only block.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated ``(single, pair)`` of shapes `[B, N, single_size]` (or ``None``) and
            `[B, N, N, pair_size]`.

        Examples:
            >>> import torch
            >>> block = PairformerBlock(
            ...     single_size=0, pair_size=4,
            ...     triangle_multiplication_hidden_size=4, triangle_attention_hidden_size=2,
            ...     num_triangle_attention_heads=2, expansion=2, dropout=0.0,
            ... ).eval()
            >>> single, pair = block(None, torch.randn(1, 3, 3, 4))
            >>> single is None, pair.shape
            (True, torch.Size([1, 3, 3, 4]))
        """

        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask=pair_mask))
        end_pair_mask = pair_mask.transpose(-2, -1) if pair_mask is not None else None
        pair = pair + self.rowwise_dropout(
            self.triangle_attention_ending_node(pair.transpose(-3, -2), pair_mask=end_pair_mask).transpose(
                -3,
                -2,
            ),
        )
        pair = pair + self.pair_transition(pair)
        if self.pair_attention is not None and self.single_transition is not None:
            single = single + self.pair_attention(single, pair, pair_mask=pair_mask)
            single = single + self.single_transition(single)
        return single, pair
