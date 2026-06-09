r"""
AlphaFold 2 style Evoformer blocks.

This module owns OFoldX's canonical Evoformer implementation. The block order and
parameter-bearing submodule boundaries follow the official AlphaFold 2
``EvoformerIteration`` closely enough for AF2 checkpoints to load through a thin
architecture-specific alias layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ..attention import GatedMultiheadAttention
from ..functional import checkpoint_blocks
from ..transition import RowwiseDropout


class EvoformerStack(nn.Module):
    r"""
    Stack of official-order AlphaFold 2 Evoformer iterations.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        outer_product_first: bool,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        is_extra_msa: bool,
        operator_backend: str,
        blocks_per_checkpoint: int | None,
    ) -> None:
        super().__init__()
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    outer_product_first=outer_product_first,
                    num_msa_attention_heads=num_msa_attention_heads,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    msa_transition_expansion=msa_transition_expansion,
                    pair_transition_expansion=pair_transition_expansion,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    is_extra_msa=is_extra_msa,
                    operator_backend=operator_backend,
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
        if not self.blocks:
            return msa, pair
        if msa_mask is None:
            msa_mask = msa.new_ones(msa.shape[:-1])
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        blocks = [
            lambda msa, pair, block=block: block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
            for block in self.blocks
        ]
        msa, pair = checkpoint_blocks(blocks, (msa, pair), self.blocks_per_checkpoint)
        return msa, pair


class EvoformerBlock(nn.Module):
    r"""
    One AF2 Evoformer iteration in official module order.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        outer_product_hidden_size: int,
        outer_product_first: bool,
        num_msa_attention_heads: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        msa_transition_expansion: int,
        pair_transition_expansion: int,
        msa_dropout: float,
        pair_dropout: float,
        is_extra_msa: bool,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.outer_product_first = outer_product_first
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            pair_size=pair_size,
            hidden_size=outer_product_hidden_size,
        )
        self.msa_stack = MsaStack(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            transition_expansion=msa_transition_expansion,
            dropout=msa_dropout,
            is_extra_msa=is_extra_msa,
            operator_backend=operator_backend,
        )
        self.pair_stack = PairStack(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=pair_transition_expansion,
            dropout=pair_dropout,
            operator_backend=operator_backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.outer_product_first:
            pair = pair + self.outer_product_mean(msa, msa_mask)
        msa = self.msa_stack(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        if not self.outer_product_first:
            pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.pair_stack(pair, pair_mask=pair_mask)
        return msa, pair


class MsaStack(nn.Module):
    r"""
    AF2 MSA update stack for main-MSA or extra-MSA Evoformer blocks.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        is_extra_msa: bool,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.msa_row_attention_with_pair_bias = MsaRowAttentionWithPairBias(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            operator_backend=operator_backend,
        )
        if is_extra_msa:
            self.msa_column_global_attention: nn.Module = MsaColumnGlobalAttention(
                msa_size=msa_size,
                num_attention_heads=num_attention_heads,
            )
            self.msa_column_attention: nn.Module | None = None
        else:
            self.msa_column_attention = MsaColumnAttention(
                msa_size=msa_size,
                num_attention_heads=num_attention_heads,
                operator_backend=operator_backend,
            )
            self.msa_column_global_attention = None
        self.feed_forward = EvoformerTransition(msa_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> Tensor:
        msa = msa + self.rowwise_dropout(
            self.msa_row_attention_with_pair_bias(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask),
        )
        if self.msa_column_attention is not None:
            msa = msa + self.msa_column_attention(msa, msa_mask=msa_mask)
        if self.msa_column_global_attention is not None:
            msa = msa + self.msa_column_global_attention(msa, msa_mask=msa_mask)
        return msa + self.feed_forward(msa, msa_mask)


class PairStack(nn.Module):
    r"""
    AF2 pair update stack: triangle multiplication, triangle attention, transition.
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
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="outgoing",
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="incoming",
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_row",
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_column",
            operator_backend=operator_backend,
        )
        self.feed_forward = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_ending_node(pair, pair_mask))
        return pair + self.feed_forward(pair, pair_mask)


class MsaRowAttentionWithPairBias(nn.Module):
    r"""
    AF2 MSA row attention with per-head pair bias.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_attention_heads: int,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size)
        self.feat_2d_norm = LayerNorm(pair_size)
        self.feat_2d_weights = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            msa_size,
            num_attention_heads=num_attention_heads,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            operator_backend=operator_backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor, pair_mask: Tensor) -> Tensor:
        del pair_mask
        msa = self.query_norm(msa)
        pair = self.feat_2d_norm(pair)
        pair_bias = self.feat_2d_weights(pair).movedim(-1, -3).unsqueeze(-4)
        return self.attention(msa, biases=(pair_bias,), attention_mask=msa_mask.unsqueeze(-2).bool())


class MsaColumnAttention(nn.Module):
    r"""
    AF2 main-MSA column attention.
    """

    def __init__(self, *, msa_size: int, num_attention_heads: int, operator_backend: str) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size)
        self.attention = GatedMultiheadAttention(
            msa_size,
            num_attention_heads=num_attention_heads,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            operator_backend=operator_backend,
        )

    def forward(self, msa: Tensor, *, msa_mask: Tensor) -> Tensor:
        msa = msa.transpose(-3, -2)
        msa_mask = msa_mask.transpose(-2, -1)
        output = self.attention(self.query_norm(msa), attention_mask=msa_mask.unsqueeze(-2).bool())
        return output.transpose(-3, -2)


class MsaColumnGlobalAttention(nn.Module):
    r"""
    AF2 extra-MSA column global attention.
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
        msa = msa.transpose(-3, -2)
        msa_mask = msa_mask.transpose(-2, -1)
        output = self.attention(self.query_norm(msa), msa_mask.unsqueeze(-1))
        return output.transpose(-3, -2)


class GlobalAttention(nn.Module):
    r"""
    AF2 global attention used by the extra-MSA stack.
    """

    def __init__(self, *, hidden_size: int, num_attention_heads: int, head_dim: int) -> None:
        super().__init__()
        self.projection = GlobalAttentionProjection(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
        )

    def forward(self, query: Tensor, mask: Tensor) -> Tensor:
        return self.projection(query, mask)


class GlobalAttentionProjection(nn.Module):
    r"""
    Parameter-bearing AF2 global attention projection.
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
    AF2 transition block.
    """

    def __init__(self, hidden_size: int, *, expansion: int) -> None:
        super().__init__()
        self.input_layer_norm = LayerNorm(hidden_size)
        self.transition1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.transition2 = nn.Linear(hidden_size * expansion, hidden_size)

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        output = self.transition2(F.relu(self.transition1(self.input_layer_norm(hidden))))
        if mask is None:
            return output
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class OuterProductMean(nn.Module):
    r"""
    AF2 outer-product mean.
    """

    def __init__(self, *, msa_size: int, pair_size: int, hidden_size: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps
        self.layer_norm_input = LayerNorm(msa_size)
        self.left_projection = nn.Linear(msa_size, hidden_size)
        self.right_projection = nn.Linear(msa_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor) -> Tensor:
        mask = msa_mask.unsqueeze(-1).to(dtype=msa.dtype)
        msa = self.layer_norm_input(msa)
        left = self.left_projection(msa) * mask
        right = self.right_projection(msa) * mask
        outer = torch.einsum("...sic,...sjd->...ijcd", left, right)
        norm = torch.einsum("...si,...sj->...ij", msa_mask.to(dtype=msa.dtype), msa_mask.to(dtype=msa.dtype))
        projected = self.output_projection(outer.flatten(-2))
        return projected / (norm.unsqueeze(-1) + self.eps)


class TriangleMultiplication(nn.Module):
    r"""
    AF2 triangle multiplication update.
    """

    def __init__(self, *, pair_size: int, hidden_size: int, equation: str) -> None:
        super().__init__()
        self.equation = equation
        self.layer_norm_input = LayerNorm(pair_size)
        self.left_projection = nn.Linear(pair_size, hidden_size)
        self.right_projection = nn.Linear(pair_size, hidden_size)
        self.left_gate = nn.Linear(pair_size, hidden_size)
        self.right_gate = nn.Linear(pair_size, hidden_size)
        self.center_layer_norm = LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, pair_size)
        self.gating_linear = nn.Linear(pair_size, pair_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        mask = pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        normalized = self.layer_norm_input(pair)
        left = self.left_projection(normalized) * self.sigmoid(self.left_gate(normalized)) * mask
        right = self.right_projection(normalized) * self.sigmoid(self.right_gate(normalized)) * mask
        if self.equation == "outgoing":
            update = torch.einsum("...ikc,...jkc->...ijc", left, right)
        else:
            # Incoming pairs left->i and right->j over the contracted k (AF2/OpenFold
            # combine order). The prior "...kjc,...kic->..." swapped i<->j, yielding the
            # transpose of the update; whole-model parity vs upstream OpenFold caught it.
            update = torch.einsum("...kic,...kjc->...ijc", left, right)
        update = self.output_projection(self.center_layer_norm(update))
        return update * self.sigmoid(self.gating_linear(normalized))


class TriangleAttention(nn.Module):
    r"""
    AF2 triangle attention update.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        head_dim: int,
        num_attention_heads: int,
        orientation: str,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.orientation = orientation
        self.query_norm = LayerNorm(pair_size)
        self.feat_2d_weights = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            pair_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            operator_backend=operator_backend,
        )

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        if self.orientation == "per_column":
            pair = pair.transpose(-3, -2)
            pair_mask = pair_mask.transpose(-2, -1)
        normalized = self.query_norm(pair)
        pair_bias = self.feat_2d_weights(normalized).movedim(-1, -3).unsqueeze(-4)
        output = self.attention(normalized, biases=(pair_bias,), attention_mask=pair_mask.unsqueeze(-2).bool())
        if self.orientation == "per_column":
            output = output.transpose(-3, -2)
        return output
