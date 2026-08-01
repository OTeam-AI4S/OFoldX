r"""
Pairformer blocks and stacks.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, is_grad_enabled, nn

from ...compilation import HomogeneousBlockRegions
from ..attention import PairBiasedSelfAttention
from ..transition import Transition
from .msa import TrianglePairUpdateBlock


class PairformerStack(HomogeneousBlockRegions, nn.Module):
    r"""
    Stack of [`PairformerBlock`][ofoldx.modules.trunk.pairformer.PairformerBlock] run in sequence.

    The stack applies ``num_blocks`` identical joint single/pair blocks in sequence.

    Args:
        num_blocks: Number of Pairformer blocks in the stack.
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        backend: Backend selector forwarded to every submodule.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
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
                    backend=backend,
                )
                for _ in range(num_blocks)
            ],
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        r"""
        Run every [`PairformerBlock`][ofoldx.modules.trunk.pairformer.PairformerBlock] in sequence.

        Args:
            single: Single representation `[B, N, single_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Refined ``(single, pair)`` of shapes `[B, N, single_size]` and `[B, N, N, pair_size]`.
        """

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and is_grad_enabled():
                single, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    single,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
                single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class PairformerBlock(nn.Module):
    r"""
    One Pairformer block that jointly refines the single and pair representations.

    Extends the exact pair-only triangle update with pair-biased single self-attention and a
    single transition.

    Args:
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Feed-forward expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        backend: Backend selector forwarded to every submodule.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.pair_attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        r"""
        Apply one joint pair and single Pairformer update.

        Args:
            single: Single representation `[B, N, single_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated ``(single, pair)`` of shapes `[B, N, single_size]` and `[B, N, N, pair_size]`.
        """

        pair = self.pair_update(pair, pair_mask=pair_mask)
        single = single + self.pair_attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return single, pair
