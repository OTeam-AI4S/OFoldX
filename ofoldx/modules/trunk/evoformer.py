r"""
Evoformer representation refinement shared by AlphaFold-derived models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attention import GatedMultiheadAttention, TriangleAttention, TriangleMultiplication
from ..backend import LayerNorm
from ..dropout import RowwiseDropout
from .msa import OuterProductMean


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
        num_msa_attention_heads: Number of MSA attention heads.
        triangle_multiplication_hidden_size: Hidden width of triangle multiplication.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        msa_transition_expansion: Expansion factor of the MSA transition.
        pair_transition_expansion: Expansion factor of the pair transition.
        msa_dropout: Row-wise dropout probability on MSA residuals.
        pair_dropout: Row-wise dropout probability on pair residuals.
        backend: Backend used by attention and pair updates.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    msa_transition_expansion=msa_transition_expansion,
                    pair_transition_expansion=pair_transition_expansion,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ],
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Refine MSA and pair representations through every block.

        Args:
            msa: MSA representation with shape ``(..., M, N, C_msa)``.
            pair: Pair representation with shape ``(..., N, N, C_pair)``.
            msa_mask: Optional MSA validity mask with shape ``(..., M, N)``.
            pair_mask: Optional pair validity mask with shape ``(..., N, N)``.

        Returns:
            Refined MSA and pair representations with the same shapes as the inputs.
        """

        if not self.blocks:
            return msa, pair
        if msa_mask is None:
            msa_mask = msa.new_ones(msa.shape[:-1])
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


class ExtraMsaEvoformerStack(EvoformerStack):
    r"""
    Homogeneous stack of AlphaFold 2 extra-MSA blocks.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            ExtraMsaEvoformerBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                msa_transition_expansion=msa_transition_expansion,
                pair_transition_expansion=pair_transition_expansion,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )


class OuterProductFirstEvoformerStack(EvoformerStack):
    r"""
    Homogeneous Evoformer stack whose blocks apply outer-product mean before MSA attention.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            OuterProductFirstEvoformerBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                msa_transition_expansion=msa_transition_expansion,
                pair_transition_expansion=pair_transition_expansion,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )


class OuterProductFirstExtraMsaEvoformerStack(EvoformerStack):
    r"""
    Extra-MSA stack whose blocks apply outer-product mean before global-column MSA attention.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            OuterProductFirstExtraMsaEvoformerBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                msa_transition_expansion=msa_transition_expansion,
                pair_transition_expansion=pair_transition_expansion,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            for _ in range(num_blocks)
        )


class EvoformerBlock(nn.Module):
    r"""
    Own one ordered Evoformer update of the MSA and pair tracks.

    The block updates the MSA, applies outer-product mean, then updates the pair track.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of MSA attention heads.
        triangle_multiplication_hidden_size: Hidden width of triangle multiplication.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        msa_transition_expansion: Expansion factor of the MSA transition.
        pair_transition_expansion: Expansion factor of the pair transition.
        msa_dropout: Row-wise dropout probability on MSA residuals.
        pair_dropout: Row-wise dropout probability on pair residuals.
        backend: Backend used by attention and pair updates.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.outer_product_mean = EvoformerOuterProductMean(
            msa_size=msa_size,
            pair_size=pair_size,
            hidden_size=outer_product_hidden_size,
            backend=backend,
        )
        self.msa_block = EvoformerMsaBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            transition_expansion=msa_transition_expansion,
            dropout=msa_dropout,
            backend=backend,
        )
        self.pair_block = EvoformerPairBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
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


class ExtraMsaEvoformerBlock(nn.Module):
    r"""
    One extra-MSA update with global column attention and outer-product mean after the MSA update.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.outer_product_mean = EvoformerOuterProductMean(
            msa_size=msa_size,
            pair_size=pair_size,
            hidden_size=outer_product_hidden_size,
            backend=backend,
        )
        self.msa_block = ExtraMsaTrackBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            transition_expansion=msa_transition_expansion,
            dropout=msa_dropout,
            backend=backend,
        )
        self.pair_block = EvoformerPairBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=pair_transition_expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        msa = self.msa_block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.pair_block(pair, pair_mask=pair_mask)
        return msa, pair


class OuterProductFirstEvoformerBlock(EvoformerBlock):
    r"""
    One ordinary-MSA Evoformer block with outer-product mean before the MSA update.
    """

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(msa, msa_mask)
        msa = self.msa_block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        pair = self.pair_block(pair, pair_mask=pair_mask)
        return msa, pair


class OuterProductFirstExtraMsaEvoformerBlock(ExtraMsaEvoformerBlock):
    r"""
    One extra-MSA block with outer-product mean before global-column MSA attention.
    """

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(msa, msa_mask)
        msa = self.msa_block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        pair = self.pair_block(pair, pair_mask=pair_mask)
        return msa, pair


class EvoformerMsaBlock(nn.Module):
    r"""
    MSA-track block owning row, column, and transition residuals.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation used as row-attention bias.
        num_attention_heads: Number of MSA attention heads.
        transition_expansion: Expansion factor of the MSA transition.
        dropout: Row-wise dropout probability on row-attention updates.
        backend: Backend used by MSA attention.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.msa_row_attention_with_pair_bias = MsaRowAttentionWithPairBias(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            backend=backend,
        )
        self.msa_column_attention = MsaColumnAttention(
            msa_size=msa_size,
            num_attention_heads=num_attention_heads,
            backend=backend,
        )
        self.transition = EvoformerTransition(msa_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

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
        num_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.msa_row_attention_with_pair_bias = MsaRowAttentionWithPairBias(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            backend=backend,
        )
        self.msa_column_global_attention = MsaColumnGlobalAttention(
            msa_size=msa_size,
            num_attention_heads=num_attention_heads,
        )
        self.transition = EvoformerTransition(msa_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

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
        triangle_multiplication_hidden_size: Hidden width of triangle multiplication.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        transition_expansion: Expansion factor of the pair transition.
        dropout: Row-wise dropout probability on pair updates.
        backend: Backend used by triangle operations.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = EvoformerTriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            backend=backend,
        )
        self.triangle_multiplication_incoming = EvoformerTriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            backend=backend,
        )
        self.triangle_attention_starting_node = EvoformerTriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.triangle_attention_ending_node = EvoformerTriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            backend=backend,
        )
        self.transition = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

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
        pair = pair + self.rowwise_dropout(
            self.triangle_attention_ending_node(pair.transpose(-3, -2), end_pair_mask).transpose(-3, -2),
        )
        return pair + self.transition(pair, pair_mask)


class MsaRowAttentionWithPairBias(nn.Module):
    r"""
    Attend over residues within each MSA row using pair-derived head biases.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of row-attention heads.
        backend: Backend used by gated attention.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_attention_heads: int,
        backend: str,
    ) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            msa_size,
            num_attention_heads=num_attention_heads,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor) -> Tensor:
        r"""
        Return an additive MSA row-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            pair: Pair representation with shape ``(..., N_residue, N_residue, C_pair)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Row-attention update with the same shape as ``msa``.
        """

        msa = self.query_norm(msa)
        pair = self.pair_norm(pair)
        pair_bias = self.pair_bias_projection(pair).movedim(-1, -3).unsqueeze(-4)
        return self.attention(msa, biases=(pair_bias,), attention_mask=msa_mask.unsqueeze(-2).bool())


class MsaColumnAttention(nn.Module):
    r"""
    Attend over MSA rows independently at each residue position.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of column-attention heads.
        backend: Backend used by gated attention.
    """

    def __init__(self, *, msa_size: int, num_attention_heads: int, backend: str) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size, backend=backend)
        self.attention = GatedMultiheadAttention(
            msa_size,
            num_attention_heads=num_attention_heads,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )

    def forward(self, msa: Tensor, *, msa_mask: Tensor) -> Tensor:
        r"""
        Return an additive MSA column-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Column-attention update with the same shape as ``msa``.
        """

        msa = msa.transpose(-3, -2)
        msa_mask = msa_mask.transpose(-2, -1)
        output = self.attention(self.query_norm(msa), attention_mask=msa_mask.unsqueeze(-2).bool())
        return output.transpose(-3, -2)


class MsaColumnGlobalAttention(nn.Module):
    r"""
    Attend over MSA rows from one mask-weighted global query per residue.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of global-attention heads.
    """

    def __init__(self, *, msa_size: int, num_attention_heads: int) -> None:
        super().__init__()
        if msa_size % num_attention_heads != 0:
            raise ValueError(f"msa_size ({msa_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        self.query_norm = LayerNorm(msa_size)
        self.attention = GlobalAttention(
            hidden_size=msa_size,
            num_attention_heads=num_attention_heads,
            head_dim=msa_size // num_attention_heads,
        )

    def forward(self, msa: Tensor, *, msa_mask: Tensor) -> Tensor:
        r"""
        Return an additive global column-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Global column-attention update with the same shape as ``msa``.
        """

        msa = msa.transpose(-3, -2)
        msa_mask = msa_mask.transpose(-2, -1)
        output = self.attention(self.query_norm(msa), msa_mask.unsqueeze(-1))
        return output.transpose(-3, -2)


class GlobalAttention(nn.Module):
    r"""
    Apply mask-weighted global-query attention along an input axis.

    Args:
        hidden_size: Width of input and output states.
        num_attention_heads: Number of global-query heads.
        head_dim: Width of each attention head.
    """

    def __init__(self, *, hidden_size: int, num_attention_heads: int, head_dim: int) -> None:
        super().__init__()
        self.projection = GlobalAttentionProjection(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )

    def forward(self, query: Tensor, mask: Tensor) -> Tensor:
        r"""
        Project a mask-weighted global query and return gated attended values.

        Args:
            query: Input states with shape ``(..., N, C)``.
            mask: Valid-input mask broadcastable to ``(..., N, 1)``.

        Returns:
            Gated attention updates with the same shape as ``query``.
        """

        return self.projection(query, mask)


class GlobalAttentionProjection(nn.Module):
    r"""
    Own the query, key, value, gate, and output projections for global attention.

    Args:
        hidden_size: Width of input and output states.
        num_attention_heads: Number of global-query heads.
        head_dim: Width of each attention head.
    """

    def __init__(self, *, hidden_size: int, num_attention_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        inner_size = num_attention_heads * head_dim
        self.query_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.value_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.output_projection = nn.Linear(inner_size, hidden_size)
        self.gate_projection = nn.Linear(hidden_size, inner_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query: Tensor, mask: Tensor) -> Tensor:
        r"""
        Attend from the mask-weighted mean query and gate the broadcast result.

        Args:
            query: Input states with shape ``(..., N, C)``.
            mask: Valid-input mask broadcastable to ``(..., N, 1)``.

        Returns:
            Gated attention updates with the same shape as ``query``.
        """

        mask = mask.to(dtype=query.dtype)
        mask_count = mask.sum(dim=-2).clamp(min=1.0)
        query_avg = (query * mask).sum(dim=-2) / mask_count
        q = self.query_projection(query_avg).unflatten(-1, (self.num_attention_heads, self.head_dim))
        k = self.key_projection(query)
        v = self.value_projection(query)
        q = q * (self.head_dim**-0.5)
        logits = torch.einsum("...hd,...kd->...hk", q, k)
        logits = logits.masked_fill(~mask.squeeze(-1).bool().unsqueeze(-2), -1e5)
        weights = torch.softmax(logits, dim=-1)
        weighted_avg = torch.einsum("...hk,...kd->...hd", weights, v)
        gate = self.sigmoid(self.gate_projection(query)).unflatten(-1, (self.num_attention_heads, self.head_dim))
        output = weighted_avg.unsqueeze(-3) * gate
        return self.output_projection(output.flatten(-2))


class EvoformerTransition(nn.Module):
    r"""
    Pre-normalized two-projection ReLU transition returning an update.

    Args:
        hidden_size: Width of input and output representations.
        expansion: Intermediate-width multiplier.
    """

    def __init__(self, hidden_size: int, *, expansion: int) -> None:
        super().__init__()
        self.input_norm = LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, hidden_size * expansion)
        self.output_projection = nn.Linear(hidden_size * expansion, hidden_size)

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""
        Return a masked additive transition update.

        Args:
            hidden: Hidden representation with ``hidden_size`` channels.
            mask: Optional validity mask matching ``hidden`` without its channel axis.

        Returns:
            Transition update with the same shape as ``hidden``.
        """

        output = self.output_projection(F.relu(self.input_projection(self.input_norm(hidden))))
        if mask is None:
            return output
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class EvoformerOuterProductMean(OuterProductMean):
    r"""
    Evoformer outer-product mean with biased left and right MSA projections.

    It uses pre-normalization, divides by ``count + eps`` after projection, and
    returns an additive pair update.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the returned pair update.
        hidden_size: Width of each projected MSA factor.
        eps: Epsilon added to the valid-sequence count.
        backend: Backend used by the outer-product contraction.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        eps: float = 1e-3,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            msa_size=msa_size,
            hidden_size=hidden_size,
            pair_size=pair_size,
            eps=eps,
            divide_before_projection=False,
            backend=backend,
        )
        self.left_projection = nn.Linear(msa_size, hidden_size)
        self.right_projection = nn.Linear(msa_size, hidden_size)


class EvoformerTriangleMultiplication(TriangleMultiplication):
    r"""
    Evoformer triangle multiplication with biased value, gate, and output projections.

    The outgoing and incoming contractions are ``"...ikh,...jkh->...ijh"`` and
    ``"...kih,...kjh->...ijh"``, respectively. The left operand owns output row
    ``i``, the right operand owns output column ``j``, and the method returns
    an additive pair update.

    Args:
        pair_size: Width of the pair representation.
        hidden_size: Width of projected multiplicative factors.
        outgoing: Whether to contract outgoing rather than incoming edges.
        backend: Backend used by the triangle contraction.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        hidden_size: int,
        outgoing: bool,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            pair_size=pair_size,
            hidden_size=hidden_size,
            outgoing=outgoing,
            backend=backend,
        )
        self.left_projection = nn.Linear(pair_size, hidden_size)
        self.left_gate_projection = nn.Linear(pair_size, hidden_size)
        self.right_projection = nn.Linear(pair_size, hidden_size)
        self.right_gate_projection = nn.Linear(pair_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, pair_size)
        self.gate_projection = nn.Linear(pair_size, pair_size)


class EvoformerTriangleAttention(TriangleAttention):
    r"""
    Evoformer triangle self-attention with biased output and gate projections.

    A gate bias of ``1.0`` opens the output gate near initialization. The caller
    selects starting- or ending-node attention by orienting ``pair``; this module
    preserves the orientation it receives.

    Args:
        pair_size: Width of the pair representation.
        head_dim: Width of each triangle-attention head.
        num_attention_heads: Number of triangle-attention heads.
        backend: Backend used by attention.
    """

    def __init__(self, *, pair_size: int, head_dim: int, num_attention_heads: int, backend: str) -> None:
        super().__init__(
            pair_size=pair_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )
