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
from torch import Tensor, nn

from ..attention import GatedMultiheadAttention, TriangleAttention, TriangleMultiplication
from ..backend import LayerNorm
from ..transition import RowwiseDropout
from .msa import OuterProductMean


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
    ) -> None:
        super().__init__()
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
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
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
        self.outer_product_mean = AlphaFold2OuterProductMean(
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
        self.triangle_multiplication_outgoing = AlphaFold2TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
        )
        self.triangle_multiplication_incoming = AlphaFold2TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
        )
        self.triangle_attention_starting_node = AlphaFold2TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = AlphaFold2TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.feed_forward = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        # The unified TriangleAttention never transposes internally (TRUNK_EINSUM_LOCK.md §3): the
        # ending node attends over the transposed residue axes, so the caller flips ``(-3, -2)`` (and
        # the mask's last two) around the call and flips the output back.
        end_pair_mask = pair_mask.transpose(-2, -1)
        pair = pair + self.rowwise_dropout(
            self.triangle_attention_ending_node(pair.transpose(-3, -2), end_pair_mask).transpose(-3, -2),
        )
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


class AlphaFold2OuterProductMean(OuterProductMean):
    r"""
    AF2 outer-product mean.

    AlphaFold 2 variant of the shared bias-free
    [`OuterProductMean`][ofoldx.modules.trunk.msa.OuterProductMean]: the left/right MSA
    projections carry a learned bias (all AF3-lineage checkpoints leave them bias-free). The
    biased projections force the ``eager`` backend, since the fused ``oops`` kernel takes no bias
    tensors. Everything else — pre-LayerNorm, divide-after-projection (``count + eps``) normalisation
    and the affine output projection — is inherited unchanged from the shared module, whose ``forward``
    already forwards ``left_projection.bias`` / ``right_projection.bias`` to the eager backend path.
    """

    def __init__(self, *, msa_size: int, pair_size: int, hidden_size: int, eps: float = 1e-3) -> None:
        super().__init__(
            msa_size=msa_size,
            hidden_size=hidden_size,
            pair_size=pair_size,
            eps=eps,
            divide_before_projection=False,
            operator_backend="eager",
        )
        self.left_projection = nn.Linear(msa_size, hidden_size)
        self.right_projection = nn.Linear(msa_size, hidden_size)


class AlphaFold2TriangleMultiplication(TriangleMultiplication):
    r"""
    AlphaFold 2 triangle multiplicative update.

    AlphaFold 2 variant of the shared bias-free
    [`TriangleMultiplication`][ofoldx.modules.attention.triangle.TriangleMultiplication]: all six
    projections (left/right value, left/right gate, output, and the input gate) carry a learned bias,
    where every AF3-lineage checkpoint leaves them bias-free. The biased projections force the
    ``eager`` backend, since the fused ``oops`` kernel takes no bias tensors. Everything else — the
    canonical incoming/outgoing contraction ``"...kih,...kjh->...ijh"`` / ``"...ikh,...jkh->...ijh"``
    (left owns row ``i``, right owns column ``j``), the input/output LayerNorms and the gated output
    projection — is inherited unchanged from the shared module, whose ``forward`` already forwards
    each projection's ``.bias`` to the eager backend path.

    Because the shared module fixes the operand order (left→``i``, right→``j``) for both directions,
    the AlphaFold 2 *incoming* converter must load DeepMind/OpenFold ``left_projection``/``left_gate``
    into this module's **right** slot and ``right_projection``/``right_gate`` into the **left** slot
    (value and gate move together). The outgoing converter loads raw. See
    ``ofoldx/models/alphafold2/converters``.
    """

    def __init__(self, *, pair_size: int, hidden_size: int, outgoing: bool) -> None:
        super().__init__(
            pair_size=pair_size,
            hidden_size=hidden_size,
            outgoing=outgoing,
            operator_backend="eager",
        )
        self.left_projection = nn.Linear(pair_size, hidden_size)
        self.left_gate_projection = nn.Linear(pair_size, hidden_size)
        self.right_projection = nn.Linear(pair_size, hidden_size)
        self.right_gate_projection = nn.Linear(pair_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, pair_size)
        self.gate_projection = nn.Linear(pair_size, pair_size)


class AlphaFold2TriangleAttention(TriangleAttention):
    r"""
    AlphaFold 2 triangle self-attention update.

    AlphaFold 2 variant of the shared bias-free
    [`TriangleAttention`][ofoldx.modules.attention.triangle.TriangleAttention]: the attention output
    projection and the output gate carry a learned bias (``gate_bias_init=1.0`` opens the gate near
    init), where every AF3-lineage checkpoint leaves them bias-free. The biased gate/output keep the
    module off the flash ``triangle_attention`` kernel, which requires bias-free projections; the
    attention core still dispatches to whatever ``operator_backend`` resolves to, applying the gate and
    output bias in the (backend-agnostic) head merge. ``transpose_bias`` stays ``False`` — the OpenFold
    convention, matching DeepMind's own AlphaFold 2 ending node — and the row/column direction is set by
    the caller orienting ``pair`` (never inside this module). See TRUNK_EINSUM_LOCK.md §3.
    """

    def __init__(self, *, pair_size: int, head_dim: int, num_attention_heads: int, operator_backend: str) -> None:
        super().__init__(
            pair_size=pair_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            transpose_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            operator_backend=operator_backend,
        )
