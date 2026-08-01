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
Concrete conditioned Transformer-encoder equations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import torch
from torch import Tensor, nn

from ..compilation import HomogeneousBlockRegions
from .adaptive_norm import AdaptiveLayerNorm
from .attention.multihead import (
    GatedMultiheadAttentionProjection,
    MultiheadAttention,
    QueryKeyNormGatedMultiheadAttentionProjection,
)
from .backend import LayerNorm, pair_norm_projection
from .transition import ConditionedTransition, TripleProductConditionedTransition


class ConditionedTransformerEncoder(HomogeneousBlockRegions, nn.Module):
    r"""
    Homogeneous conditioned Transformer-encoder blocks with one packed pair bias.

    Every block is an instance of ``block``, built as ``block(hidden_size, conditioning_size,
    num_attention_heads, backend)``. The pair bias stores all blocks' heads on its last axis, and each
    block attends under its own slice.

    Args:
        num_blocks: Number of blocks.
        num_attention_heads: Attention heads per block.
        hidden_size: Width of the hidden state.
        conditioning_size: Width of the per-position conditioning.
        backend: Backend selector forwarded to every block.
        block: Block class every block instantiates.
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int,
        conditioning_size: int,
        backend: str = "auto",
        *,
        block: Callable[[int, int, int, str], nn.Module],
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            block(hidden_size, conditioning_size, num_attention_heads, backend) for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        r"""Refine hidden states under per-position validity.

        ``mask`` follows the query positions. The actual ``to_keys`` index map
        gathers the same validity as the key/value states; a singleton query
        axis makes the resulting key validity explicit to each child block.
        """

        if not self.blocks:
            return hidden
        attention_mask = None
        if mask is not None:
            key_validity = to_keys(mask.unsqueeze(-1)).squeeze(-1) if to_keys is not None else mask
            attention_mask = key_validity.unsqueeze(-2)
        checkpoint = (
            self._gradient_checkpointing_func
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
            else None
        )
        # Each block's bias is concatenated on the last axis: [..., num_blocks * width].
        bias = torch.unflatten(bias, -1, (len(self.blocks), bias.shape[-1] // len(self.blocks)))
        for index, block in enumerate(self.blocks):
            if checkpoint is not None:
                hidden = checkpoint(block.__call__, hidden, conditioning, bias[..., index, :], attention_mask, to_keys)
            else:
                hidden = block(hidden, conditioning, bias[..., index, :], attention_mask, to_keys)
        return hidden


def _run_pair_biased_block(
    hidden: Tensor,
    conditioning: Tensor,
    pair: Tensor,
    pair_mask: Tensor | None,
    *,
    block: ConditionedTransformerEncoderBlock,
    norm: LayerNorm,
    projection: nn.Linear,
) -> Tensor:
    # The block accepts head-last bias and moves it back to head-major for attention.
    # Keep this as a view so OOps can write that final head-major layout directly.
    bias = pair_norm_projection(
        pair,
        norm.weight,
        norm.bias,
        projection.weight,
        norm.eps,
        output_layout="head_major",
        backend=norm.backend,
    )
    missing_prefix_dims = len(hidden.shape[:-2]) - len(pair.shape[:-3])
    for _ in range(missing_prefix_dims):
        bias = bias.unsqueeze(-4)
    bias = bias.movedim(-3, -1)
    return block(hidden, conditioning, bias, pair_mask, to_keys=None)


class PairBiasedConditionedTransformerEncoder(HomogeneousBlockRegions, nn.Module):
    r"""
    Conditioned Transformer encoder with learned per-block pair attention biases.

    Projects the pair representation to per-block attention-bias logits (``pair_norms`` / ``bias_projections``)
    and runs its homogeneous
    [`ConditionedTransformerEncoderBlock`][ofoldx.modules.transformer.ConditionedTransformerEncoderBlock]s,
    each with a single adaptive norm and a double-product transition.

    Args:
        num_blocks: Number of token-encoder blocks; each contributes one Transformer-encoder block. ``0`` makes the
            module the identity on the tokens.
        token_hidden_size: Width of the per-token channel carried through the encoder.
        conditioning_size: Width of the conditioning signal driving the adaptive norm.
        pair_size: Width of the pair representation projected to the attention bias.
        num_attention_heads: Number of attention heads per transformer block.
        backend: Operator provider; see [ofoldx.modules.backend][].
    """

    def __init__(
        self,
        num_blocks: int = 24,
        token_hidden_size: int = 768,
        conditioning_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.pair_norms = nn.ModuleList(LayerNorm(pair_size, bias=False, backend=backend) for _ in range(num_blocks))
        self.bias_projections = nn.ModuleList(
            nn.Linear(pair_size, num_attention_heads, bias=False) for _ in range(num_blocks)
        )
        self.blocks = nn.ModuleList(
            ConditionedTransformerEncoderBlock(
                token_hidden_size,
                conditioning_size,
                num_attention_heads,
                backend,
                transition=partial(ConditionedTransition, expansion=2),
            )
            for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(
        self,
        token_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Norm and project the pair representation lazily into each block's per-head bias, then run the transformer.

        Each block ``i`` maps ``pair`` through its own ``pair_norms[i]`` /
        ``bias_projections[i]`` immediately before that block executes. Under
        gradient checkpointing, the pair projection is recomputed together with
        its consuming block instead of retaining every block's projected bias.
        With zero blocks the module is the identity on ``token_hidden``.

        Args:
            token_hidden: Per-token features `[..., N_token, token_hidden_size]`.
            conditioning: Conditioning signal `[..., N_token, conditioning_size]` for the adaptive norm.
            pair: Pair representation `[..., N_token, N_token, pair_size]` supplying the attention bias.
            pair_mask: Optional keep mask `[..., N_token, N_token]`; ``False`` pairs are excluded.
                Key validity uses the explicit query-broadcast shape `[..., 1, N_token]`.

        Returns:
            Refined token features `[..., N_token, token_hidden_size]`.
        """

        if not self.pair_norms:  # zero blocks -> identity (matches an empty transformer stack)
            return token_hidden
        checkpoint = (
            self._gradient_checkpointing_func
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
            else None
        )
        hidden = token_hidden
        for block, norm, projection in zip(
            self.blocks,
            self.pair_norms,
            self.bias_projections,
            strict=True,
        ):
            run_block = partial(
                _run_pair_biased_block,
                block=cast(ConditionedTransformerEncoderBlock, block),
                norm=cast(LayerNorm, norm),
                projection=cast(nn.Linear, projection),
            )
            if checkpoint is not None:
                hidden = checkpoint(run_block, hidden, conditioning, pair, pair_mask)
            else:
                hidden = run_block(hidden, conditioning, pair, pair_mask)
        return hidden


class ConditionedTransformerEncoderBlock(nn.Module):
    r"""
    Pair-biased encoder self-attention followed by its actual conditioned transition.

    The concrete ``transition`` factory defaults to the triple-product operation.

    The self-attention head projection is built by
    [`attention_projection`][ofoldx.modules.transformer.ConditionedTransformerEncoderBlock.attention_projection].
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        *,
        transition: Callable[..., nn.Module] = TripleProductConditionedTransition,
    ) -> None:
        super().__init__()
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.attention = MultiheadAttention(
            self.attention_projection(hidden_size, num_attention_heads, backend),
            backend=backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.transition = transition(
            hidden_size,
            conditioning_size,
            backend=backend,
        )

    def attention_projection(
        self, hidden_size: int, num_attention_heads: int, backend: str
    ) -> GatedMultiheadAttentionProjection:
        r"""
        Build the gated self-attention head projection with a query bias.

        Args:
            hidden_size: Width of the hidden state.
            num_attention_heads: Number of attention heads.
            backend: Operator provider of normalizations inside the projection.

        Returns:
            The head projection of the block's attention.
        """

        return GatedMultiheadAttentionProjection(hidden_size, num_attention_heads, query_bias=True)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        attention_mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        r"""Run the learned block with an explicit query-key attention mask.

        Key-only validity carries a singleton query axis. ``to_keys`` maps the
        key/value states; the supplied attention mask already names their axes.
        """

        return conditioned_transformer_block(
            hidden,
            conditioning,
            bias,
            attention_mask,
            to_keys,
            norm=self.conditioned_norm,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            transition=self.transition,
        )


class PostNormConditionedTransformerEncoderBlock(ConditionedTransformerEncoderBlock):
    r"""
    Conditioned Transformer-encoder block that layer-normalizes its output.
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        *,
        transition: Callable[..., nn.Module] = TripleProductConditionedTransition,
    ) -> None:
        super().__init__(hidden_size, conditioning_size, num_attention_heads, backend, transition=transition)
        self.post_norm = LayerNorm(hidden_size, backend=backend)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        attention_mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        return self.post_norm(super().forward(hidden, conditioning, bias, attention_mask, to_keys))


class QueryKeyNormConditionedTransformerEncoderBlock(ConditionedTransformerEncoderBlock):
    r"""
    Conditioned Transformer-encoder block whose attention layer-normalizes every query and key head.
    """

    def attention_projection(
        self, hidden_size: int, num_attention_heads: int, backend: str
    ) -> GatedMultiheadAttentionProjection:
        r"""
        Build the gated self-attention head projection with a query bias and normalized query/key heads.
        """

        return QueryKeyNormGatedMultiheadAttentionProjection(
            hidden_size, num_attention_heads, query_bias=True, backend=backend
        )


class PostNormQueryKeyNormConditionedTransformerEncoderBlock(
    PostNormConditionedTransformerEncoderBlock, QueryKeyNormConditionedTransformerEncoderBlock
):
    r"""
    Query/key-normalized conditioned Transformer-encoder block that layer-normalizes its output.
    """


class AtomCrossAttentionTransformerEncoderBlock(nn.Module):
    r"""
    Double-product atom cross-attention with independently normalized raw keys.
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        num_attention_heads: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.key_value_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(hidden_size, num_attention_heads, query_bias=True),
            backend=backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.transition = ConditionedTransition(
            hidden_size,
            conditioning_size,
            expansion=2,
            backend=backend,
        )

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        attention_mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        hidden = hidden + _attention_update(
            normalized,
            self.normalize_key_values(hidden, normalized, conditioning),
            conditioning,
            bias,
            attention_mask,
            to_keys,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
        )
        return hidden + self.transition(hidden, conditioning)

    def normalize_key_values(self, hidden: Tensor, normalized: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Normalize the state the gathered keys and values are projected from.

        This block normalizes the raw block input, independently of the queries.

        Args:
            hidden: Block input `[..., N, hidden_size]`.
            normalized: Query-normalized block input `[..., N, hidden_size]`.
            conditioning: Per-position conditioning `[..., N, conditioning_size]`.

        Returns:
            The key and value source state `[..., N, hidden_size]`.
        """

        return self.key_value_norm(hidden, conditioning)


class NormalizedKeyAtomCrossAttentionTransformerEncoderBlock(AtomCrossAttentionTransformerEncoderBlock):
    r"""
    Double-product atom cross-attention whose key normalization starts from the query-normalized state.
    """

    def normalize_key_values(self, hidden: Tensor, normalized: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Normalize the query-normalized block input again, so keys and values carry both normalizations.
        """

        return self.key_value_norm(normalized, conditioning)


def conditioned_transformer_block(
    hidden: Tensor,
    conditioning: Tensor,
    bias: Tensor,
    attention_mask: Tensor | None = None,
    to_keys: Callable[[Tensor], Tensor] | None = None,
    *,
    norm: AdaptiveLayerNorm,
    attention: MultiheadAttention,
    output_gate_projection: nn.Linear,
    transition: nn.Module,
) -> Tensor:
    r"""Apply the complete conditioned attention and transition residual block."""

    normalized = norm(hidden, conditioning)
    hidden = hidden + _attention_update(
        normalized,
        normalized,
        conditioning,
        bias,
        attention_mask,
        to_keys,
        attention=attention,
        output_gate_projection=output_gate_projection,
    )
    return hidden + transition(hidden, conditioning)


def _attention_update(
    query: Tensor,
    key_base: Tensor,
    conditioning: Tensor,
    bias: Tensor,
    attention_mask: Tensor | None,
    to_keys: Callable[[Tensor], Tensor] | None,
    *,
    attention: MultiheadAttention,
    output_gate_projection: nn.Linear,
) -> Tensor:
    key = to_keys(key_base) if to_keys is not None else key_base
    missing_bias_prefix = query.dim() - 2 - (bias.dim() - 3)
    for _ in range(missing_bias_prefix):
        bias = bias.unsqueeze(-4)
    update = attention(
        query,
        key=key,
        biases=(bias.movedim(-1, -3),),
        attention_mask=attention_mask,
    )
    return output_gate_projection(conditioning).sigmoid() * update
