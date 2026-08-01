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
Pairformer blocks and stacks.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from operator import index

from torch import Tensor, is_grad_enabled, nn

from ..compilation import HomogeneousBlockRegions
from .attention import PairBiasedSelfAttention
from .backend import LayerNorm
from .pair import LineDropoutTrianglePairUpdateBlock, TrianglePairUpdateBlock
from .transition import Transition


class PairformerBlock(nn.Module):
    r"""
    One Pairformer block that jointly refines the single and pair representations.

    Extends the exact pair-only triangle update with pair-biased single self-attention and a
    single transition.

    Args:
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Feed-forward expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        backend: Backend selector forwarded to every submodule.
        ending_bias_pair_dims: Source pair axes mapped to ending attention's bias query/key axes.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        ending_bias_pair_dims: tuple[int, int] = (-3, -2),
    ) -> None:
        super().__init__()
        self.pair_update = self.build_pair_update(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
            ending_bias_pair_dims=ending_bias_pair_dims,
        )
        self.pair_attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)

    def build_pair_update(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        expansion: int,
        dropout: float,
        backend: str,
        ending_bias_pair_dims: tuple[int, int] = (-3, -2),
    ) -> TrianglePairUpdateBlock:
        r"""
        Build the pair-only triangle update this block runs before its single update.

        Returns:
            A [`TrianglePairUpdateBlock`][ofoldx.modules.pair.TrianglePairUpdateBlock].
        """

        return TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
            ending_bias_pair_dims=ending_bias_pair_dims,
        )

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


class PairformerStack(HomogeneousBlockRegions, nn.Module):
    r"""
    Stack of [`PairformerBlock`][ofoldx.modules.pairformer.PairformerBlock] run in sequence.

    The stack applies ``num_blocks`` identical joint single/pair blocks in sequence.

    Args:
        num_blocks: Number of Pairformer blocks in the stack.
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Expansion factor of every transition.
        dropout: Row-wise dropout probability applied to each pair residual.
        backend: Backend selector forwarded to every submodule.
        checkpoint_group_size: Maximum blocks per outer recomputation group. One
            keeps only the per-block checkpoints. Inactive without checkpointing.
        block: Pairformer block factory. Omitted uses ``PairformerBlock``.

    Raises:
        ValueError: If ``checkpoint_group_size`` is not positive.
    """

    def __init__(
        self,
        num_blocks: int = 48,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        checkpoint_group_size: int = 1,
        block: Callable[..., PairformerBlock] = PairformerBlock,
    ) -> None:
        super().__init__()
        self.checkpoint_group_size = index(checkpoint_group_size)
        if self.checkpoint_group_size < 1:
            raise ValueError("checkpoint_group_size must be positive")
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                block(
                    single_size=single_size,
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_head_size=triangle_attention_head_size,
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
        Run every [`PairformerBlock`][ofoldx.modules.pairformer.PairformerBlock] in sequence.

        Args:
            single: Single representation `[B, N, single_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Refined ``(single, pair)`` of shapes `[B, N, single_size]` and `[B, N, N, pair_size]`.
        """

        if not (self.gradient_checkpointing and self.training and is_grad_enabled()):
            for block in self.blocks:
                single, pair = block(single, pair, pair_mask=pair_mask)
            return single, pair

        checkpoint = self._gradient_checkpointing_func
        for start in range(0, len(self.blocks), self.checkpoint_group_size):
            stop = min(start + self.checkpoint_group_size, len(self.blocks))
            if stop - start == 1:
                single, pair = checkpoint(self.blocks[start].__call__, single, pair, pair_mask)
            else:
                run_group = partial(self._run_checkpointed_group, start=start, stop=stop, checkpoint=checkpoint)
                single, pair = checkpoint(run_group, single, pair, pair_mask)
        return single, pair

    def _run_checkpointed_group(
        self,
        single: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None,
        *,
        start: int,
        stop: int,
        checkpoint: Callable[..., tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        for block_index in range(start, stop):
            single, pair = checkpoint(self.blocks[block_index].__call__, single, pair, pair_mask)
        return single, pair


class LineDropoutPairformerBlock(PairformerBlock):
    r"""
    Pairformer block whose pair update drops whole rows and columns of the pair representation.

    Identical to [`PairformerBlock`][ofoldx.modules.pairformer.PairformerBlock] except for
    the dropout granularity of its pair update; see
    [`LineDropoutTrianglePairUpdateBlock`][ofoldx.modules.pair.LineDropoutTrianglePairUpdateBlock].
    """

    def build_pair_update(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        expansion: int,
        dropout: float,
        backend: str,
        ending_bias_pair_dims: tuple[int, int] = (-3, -2),
    ) -> TrianglePairUpdateBlock:
        r"""
        Returns:
            A [`LineDropoutTrianglePairUpdateBlock`][ofoldx.modules.pair.LineDropoutTrianglePairUpdateBlock].
        """

        return LineDropoutTrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
            ending_bias_pair_dims=ending_bias_pair_dims,
        )


class PostNormLineDropoutPairformerBlock(LineDropoutPairformerBlock):
    r"""
    Line-dropout Pairformer block that layer-normalizes its updated single representation.

    The normalization is a learned affine layer applied after the single transition, so a block with
    it has parameters a block without it has no equivalent for.

    Args:
        single_size: Width of the single representation.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        num_pair_attention_heads: Number of heads in the pair-biased single self-attention.
        expansion: Feed-forward expansion factor of every transition.
        dropout: Probability of dropping a whole pair row or column from each pair residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            single_size=single_size,
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.single_post_norm = LayerNorm(single_size, backend=backend)

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        r"""
        Apply one Pairformer update, then normalize the updated single representation.

        Args:
            single: Single representation `[B, N, single_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated ``(single, pair)`` of shapes `[B, N, single_size]` and `[B, N, N, pair_size]`.
        """

        single, pair = super().forward(single, pair, pair_mask=pair_mask)
        return self.single_post_norm(single), pair
