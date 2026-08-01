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
Evoformer blocks and stacks for joint MSA/pair refinement.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .attention import (
    MsaColumnAttention,
    MsaColumnGlobalAttention,
    MsaRowAttentionWithPairBias,
    TriangleAttention,
)
from .dropout import StructuredDropout
from .msa import PostProjectionOuterProductMean
from .pair import TriangleMultiplication
from .transition import FeedForwardTransition


class EvoformerMsaBlock(nn.Module):
    r"""
    MSA-track block owning row, column, and transition residuals.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation used as row-attention bias.
        num_row_attention_heads: Number of MSA row-attention heads.
        num_column_attention_heads: Number of MSA column-attention heads.
        transition_expansion: Expansion factor of the MSA transition.
        dropout: Row-wise dropout probability on row-attention updates.
        backend: Backend used by the attention and transition normalizations.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_row_attention_heads: int,
        num_column_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.msa_row_attention_with_pair_bias = MsaRowAttentionWithPairBias(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_row_attention_heads,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.msa_column_attention = MsaColumnAttention(
            msa_size=msa_size,
            num_attention_heads=num_column_attention_heads,
            backend=backend,
        )
        self.transition = FeedForwardTransition(msa_size, expansion=transition_expansion, backend=backend)

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> Tensor:
        r"""
        Apply row attention, column attention, and transition residuals in order.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            pair: Pair representation with shape ``(..., N_residue, N_residue, C_pair)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.
            pair_mask: Pair validity mask with shape ``(..., N_residue, N_residue)``.

        Returns:
            Updated MSA representation with the same shape as ``msa``.
        """

        msa = msa + self.rowwise_dropout(
            self.msa_row_attention_with_pair_bias(msa, pair, msa_mask=msa_mask),
        )
        msa = msa + self.msa_column_attention(msa, msa_mask=msa_mask)
        return msa + self.transition(msa, msa_mask)


class ExtraMsaTrackBlock(nn.Module):
    r"""
    Extra-MSA track update with row attention, global column attention, and transition.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_row_attention_heads: int,
        num_column_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.msa_row_attention_with_pair_bias = MsaRowAttentionWithPairBias(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_row_attention_heads,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.msa_column_global_attention = MsaColumnGlobalAttention(
            msa_size=msa_size,
            num_attention_heads=num_column_attention_heads,
            backend=backend,
        )
        self.transition = FeedForwardTransition(msa_size, expansion=transition_expansion, backend=backend)

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> Tensor:
        msa = msa + self.rowwise_dropout(
            self.msa_row_attention_with_pair_bias(msa, pair, msa_mask=msa_mask),
        )
        msa = msa + self.msa_column_global_attention(msa, msa_mask=msa_mask)
        return msa + self.transition(msa, msa_mask)


class EvoformerPairBlock(nn.Module):
    r"""
    Pair-track block owning triangle and transition residuals.

    Args:
        pair_size: Width of the pair representation.
        triangle_multiplication_outgoing_hidden_size: Hidden width of outgoing triangle multiplication.
        triangle_multiplication_incoming_hidden_size: Hidden width of incoming triangle multiplication.
        triangle_attention_head_size: Per-head width of triangle attention.
        num_triangle_attention_starting_node_heads: Number of starting-node triangle-attention heads.
        num_triangle_attention_ending_node_heads: Number of ending-node triangle-attention heads.
        transition_expansion: Expansion factor of the pair transition.
        dropout: Row-wise dropout probability on pair updates.
        backend: Backend used by triangle operations and transition normalization.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_outgoing_hidden_size: int,
        triangle_multiplication_incoming_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_starting_node_heads: int,
        num_triangle_attention_ending_node_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_outgoing_hidden_size,
            direction="outgoing",
            projection_bias=True,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_incoming_hidden_size,
            direction="incoming",
            projection_bias=True,
            backend=backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_starting_node_heads,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_ending_node_heads,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )
        self.transition = FeedForwardTransition(pair_size, expansion=transition_expansion, backend=backend)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        r"""
        Apply outgoing, incoming, starting, ending, and transition residuals.

        Args:
            pair: Pair representation with shape ``(..., N_residue, N_residue, C_pair)``.
            pair_mask: Pair validity mask with shape ``(..., N_residue, N_residue)``.

        Returns:
            Updated pair representation with the same shape as ``pair``.
        """

        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        # TriangleAttention preserves the input residue-axis orientation. Ending-node attention
        # therefore transposes the pair and mask residue axes around the call, then restores them.
        end_pair_mask = pair_mask.transpose(-2, -1)
        ending_update = self.triangle_attention_ending_node(pair.transpose(-3, -2), end_pair_mask)
        pair = pair + self.rowwise_dropout(ending_update).transpose(-3, -2)
        return pair + self.transition(pair, pair_mask)


class EvoformerBlock(nn.Module):
    r"""
    Own one ordered Evoformer update of the MSA and pair tracks.

    The block updates the MSA, applies outer-product mean, then updates the pair track.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_row_attention_heads: Number of MSA row-attention heads.
        num_msa_column_attention_heads: Number of MSA column-attention heads.
        triangle_multiplication_outgoing_hidden_size: Hidden width of outgoing triangle multiplication.
        triangle_multiplication_incoming_hidden_size: Hidden width of incoming triangle multiplication.
        triangle_attention_head_size: Per-head width of triangle attention.
        num_triangle_attention_starting_node_heads: Number of starting-node triangle-attention heads.
        num_triangle_attention_ending_node_heads: Number of ending-node triangle-attention heads.
        msa_transition_expansion: Expansion factor of the MSA transition.
        pair_transition_expansion: Expansion factor of the pair transition.
        msa_dropout: Row-wise dropout probability on MSA residuals.
        pair_dropout: Row-wise dropout probability on pair residuals.
        backend: Backend forwarded to every dispatched unit.
        msa_block: Concrete MSA-track block class; defaults to ``EvoformerMsaBlock``.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_row_attention_heads: int,
        num_msa_column_attention_heads: int,
        triangle_multiplication_outgoing_hidden_size: int,
        triangle_multiplication_incoming_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_starting_node_heads: int,
        num_triangle_attention_ending_node_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
        msa_block: type[nn.Module] = EvoformerMsaBlock,
    ) -> None:
        super().__init__()
        self.msa_block = msa_block(
            msa_size=msa_size,
            pair_size=pair_size,
            num_row_attention_heads=num_msa_row_attention_heads,
            num_column_attention_heads=num_msa_column_attention_heads,
            transition_expansion=msa_transition_expansion,
            dropout=msa_dropout,
            backend=backend,
        )
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            pair_size=pair_size,
            hidden_size=outer_product_hidden_size,
            backend=backend,
            projection_bias=True,
        )
        self.pair_block = EvoformerPairBlock(
            pair_size=pair_size,
            triangle_multiplication_outgoing_hidden_size=triangle_multiplication_outgoing_hidden_size,
            triangle_multiplication_incoming_hidden_size=triangle_multiplication_incoming_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_starting_node_heads=num_triangle_attention_starting_node_heads,
            num_triangle_attention_ending_node_heads=num_triangle_attention_ending_node_heads,
            transition_expansion=pair_transition_expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Apply the ordered MSA, outer-product, and pair updates.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            pair: Pair representation with shape ``(..., N_residue, N_residue, C_pair)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.
            pair_mask: Pair validity mask with shape ``(..., N_residue, N_residue)``.

        Returns:
            Updated MSA and pair representations with unchanged shapes.
        """

        msa = self.msa_block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.pair_block(pair, pair_mask=pair_mask)
        return msa, pair


class OuterProductFirstEvoformerBlock(EvoformerBlock):
    r"""
    Update the pair with outer-product mean before applying the MSA and pair tracks.

    The inherited constructor owns the same named modules as ``EvoformerBlock``.
    This override preserves the distinct scientific operation order.
    """

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(msa, msa_mask)
        msa = self.msa_block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        pair = self.pair_block(pair, pair_mask=pair_mask)
        return msa, pair


class EvoformerStack(nn.Module):
    r"""
    Refine MSA and pair representations through homogeneous Evoformer blocks.

    MSA states use ``[..., sequences, residues, msa_size]`` and pair states use
    ``[..., residues, residues, pair_size]``. Each block preserves both layouts.

    Args:
        num_blocks: Number of homogeneous Evoformer blocks.
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_row_attention_heads: Number of MSA row-attention heads.
        num_msa_column_attention_heads: Number of MSA column-attention heads.
        triangle_multiplication_outgoing_hidden_size: Hidden width of outgoing triangle multiplication.
        triangle_multiplication_incoming_hidden_size: Hidden width of incoming triangle multiplication.
        triangle_attention_head_size: Per-head width of triangle attention.
        num_triangle_attention_starting_node_heads: Number of starting-node triangle-attention heads.
        num_triangle_attention_ending_node_heads: Number of ending-node triangle-attention heads.
        msa_transition_expansion: Expansion factor of the MSA transition.
        pair_transition_expansion: Expansion factor of the pair transition.
        msa_dropout: Row-wise dropout probability on MSA residuals.
        pair_dropout: Row-wise dropout probability on pair residuals.
        backend: Backend forwarded to every dispatched unit.
        block: Concrete ordered block class; defaults to ``EvoformerBlock``.
        msa_block: Concrete MSA-track class passed to every block; defaults to ``EvoformerMsaBlock``.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_row_attention_heads: int,
        num_msa_column_attention_heads: int,
        triangle_multiplication_outgoing_hidden_size: int,
        triangle_multiplication_incoming_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_starting_node_heads: int,
        num_triangle_attention_ending_node_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
        block: type[EvoformerBlock] = EvoformerBlock,
        msa_block: type[nn.Module] = EvoformerMsaBlock,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                block(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    num_msa_row_attention_heads=num_msa_row_attention_heads,
                    num_msa_column_attention_heads=num_msa_column_attention_heads,
                    triangle_multiplication_outgoing_hidden_size=triangle_multiplication_outgoing_hidden_size,
                    triangle_multiplication_incoming_hidden_size=triangle_multiplication_incoming_hidden_size,
                    triangle_attention_head_size=triangle_attention_head_size,
                    num_triangle_attention_starting_node_heads=num_triangle_attention_starting_node_heads,
                    num_triangle_attention_ending_node_heads=num_triangle_attention_ending_node_heads,
                    msa_transition_expansion=msa_transition_expansion,
                    pair_transition_expansion=pair_transition_expansion,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    backend=backend,
                    msa_block=msa_block,
                )
                for _ in range(num_blocks)
            ],
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.blocks:
            return msa, pair
        if msa_mask is None:
            msa_mask = msa.new_ones(msa.shape[:-1])
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                msa, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    msa,
                    pair,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )
            else:
                msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair
