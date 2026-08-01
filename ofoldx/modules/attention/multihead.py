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
Multi-head attention primitives shared across the trunk, atom, and diffusion stacks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import LayerNorm, resolve_config_backend
from ..backend.attention import multihead_attention
from ..backend.attention import projected_attention as dispatch_projected_attention


class MultiheadAttention(nn.Module):
    r"""
    Multi-head attention with additive pair biases over a learned head projection.

    The supplied [`MultiheadAttentionProjection`][ofoldx.modules.attention.multihead.MultiheadAttentionProjection]
    provides the learned head projections. Higher-level modules supply the pair-derived ``biases``.

    Args:
        projection: The query/key/value/output head projection.
        backend: Operator provider; attention uses SDPA as its default
            PyTorch implementation.
        dropout: Dropout probability applied to attention weights while training.
    """

    def __init__(
        self,
        projection: MultiheadAttentionProjection,
        backend: str = "auto",
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "sdpa", "eager"), public_default="sdpa")
        self.projection = projection
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        biases: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Run multi-head attention, dispatching to the configured backend.

        Projects q/k/v into heads, computes biased and masked attention via the selected OOps or
        PyTorch implementation, then merges the heads through the projection.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`; defaults to ``query`` (self-attention).
            value: Value tokens `[..., kv_tokens, value_size]`; defaults to ``key``.
            biases: Additive pair biases, each broadcastable to `[..., heads, q_tokens, kv_tokens]`.
            attention_mask: Boolean pair-first keep-mask. ``(q_tokens,
                kv_tokens)`` is a shared pair mask, ``(kv_tokens,)`` is shared
                key validity, and prefix-specific key validity uses
                ``(..., kv_tokens)`` when unambiguous or
                ``(..., 1, kv_tokens)`` otherwise. A head-specific mask retains
                every prefix position, followed by the head axis.

        Returns:
            Updated hidden states `[..., q_tokens, hidden_size]`.
        """

        return multihead_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            projection=self.projection,
            backend=self.backend,
            dropout_p=self.dropout if self.training else 0.0,
        )


class ProjectedAttention(nn.Module):
    r"""
    Scaled dot-product attention over already projected head-major tensors.

    This parameter-free Module keeps layout construction and learned projections at
    their mathematical owner while moving the attention core behind the common
    backend policy. The eager implementation uses float32 intermediates for half
    and bfloat16 inputs, preserves float64, returns the value dtype, and zeros
    query rows with no valid key.

    Args:
        dropout: Dropout probability applied to attention weights while training.
        backend: Operator provider; attention uses SDPA as its default PyTorch
            implementation.
    """

    backend: str

    def __init__(self, dropout: float = 0.0, *, backend: str = "auto") -> None:
        super().__init__()
        self.dropout = dropout
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "sdpa", "eager"),
            public_default="sdpa",
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        biases: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
        scale: float | None = None,
    ) -> Tensor:
        r"""
        Attend from projected queries to projected keys and values.

        Args:
            query: Head-major query tensor
                ``(..., num_attention_heads, num_query_tokens, head_size)``.
            key: Head-major key tensor
                ``(..., num_attention_heads, num_key_tokens, head_size)``.
            value: Head-major value tensor
                ``(..., num_attention_heads, num_key_tokens, value_size)``.
            biases: Optional additive biases broadcastable to
                ``(..., num_attention_heads, num_query_tokens, num_key_tokens)``.
            attention_mask: Optional boolean pair-first keep-mask. A trailing
                ``(num_query_tokens, num_key_tokens)`` payload is a pair mask,
                ``(num_key_tokens,)`` is shared key validity, and an explicit
                singleton query axis disambiguates prefix-specific key validity.
                Head-specific masks retain every prefix position, followed by
                the head axis.
            scale: Optional multiplier applied to query-key logits. ``None`` uses
                ``head_size ** -0.5``.

        Returns:
            Head-major attention values with shape
                ``(..., num_attention_heads, num_query_tokens, value_size)``.
        """

        return dispatch_projected_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            backend=self.backend,
            dropout_p=self.dropout if self.training else 0.0,
            scale=scale,
        )


class GlobalAttention(nn.Module):
    r"""
    Apply mask-weighted global-query attention along an input axis.

    Args:
        hidden_size: Width of input and output states.
        num_attention_heads: Number of global-query heads.
        head_size: Width of each attention head.
        backend: Backend used by the projected attention operation.
    """

    def __init__(self, *, hidden_size: int, num_attention_heads: int, head_size: int, backend: str = "auto") -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        inner_size = num_attention_heads * head_size
        self.query_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, head_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, head_size, bias=False)
        self.attention = ProjectedAttention(backend=backend)
        self.gate_projection = nn.Linear(hidden_size, inner_size)
        self.sigmoid = nn.Sigmoid()
        self.output_projection = nn.Linear(inner_size, hidden_size)

    def forward(self, query: Tensor, mask: Tensor | None = None) -> Tensor:
        r"""
        Attend from the mask-weighted mean query and gate the broadcast result.

        Args:
            query: Input states with shape ``(..., N, C)``.
            mask: Optional valid-input mask broadcastable to ``(..., N, 1)``.
                Boolean false excludes the payload before all arithmetic.
                Numeric masks are coefficients; finite zero-coefficient values
                retain sensitivity to that coefficient. Nested inputs require
                an aligned NestedTensor mask.

        Returns:
            Gated attention updates with the same shape as ``query``.
        """

        if isinstance(query, NestedTensor):
            if mask is not None and not isinstance(mask, NestedTensor):
                raise ValueError("nested global attention requires a NestedTensor mask")
            nested_query = query
            batch_dim = 0 if nested_query.batch_first else 1
            masks = (None,) * len(nested_query) if mask is None else mask.unbind(batch_dim)
            return NestedTensor(
                [self(part, validity) for part, validity in zip(nested_query.unbind(batch_dim), masks, strict=True)],
                ragged_dims=nested_query.ragged_dims,
                batch_first=nested_query.batch_first,
                padding_value=nested_query.padding_value,
                mask_value=nested_query.mask_value,
            )
        if mask is None:
            query_avg = query.mean(dim=-2)
        else:
            mask = mask.expand_as(query[..., :1])
            if mask.dtype == torch.bool:
                query = torch.where(mask, query, 0)
                mean_values = query
            else:
                # A finite observation at zero coefficient still determines
                # the derivative with respect to that coefficient.
                mean_values = torch.where((mask != 0) | torch.isfinite(query).all(-1, keepdim=True), query, 0)
                query = torch.where(mask != 0, query, 0)
            coefficients = mask.double()
            mask_count = coefficients.sum(dim=-2).clamp(min=1.0)
            query_avg = ((mean_values.double() * coefficients).sum(dim=-2) / mask_count).to(query.dtype)
        q = self.query_projection(query_avg).unflatten(-1, (self.num_attention_heads, self.head_size)).unsqueeze(-2)
        head_shape = (*((-1,) * (query.dim() - 2)), self.num_attention_heads, -1, -1)
        k = self.key_projection(query).unsqueeze(-3).expand(head_shape)
        v = self.value_projection(query).unsqueeze(-3).expand(head_shape)
        weighted_avg = self.attention(
            q,
            k,
            v,
            attention_mask=None if mask is None else mask.squeeze(-1).bool().unsqueeze(-2),
        ).squeeze(-2)
        gate = self.sigmoid(self.gate_projection(query)).unflatten(-1, (self.num_attention_heads, self.head_size))
        output = weighted_avg.unsqueeze(-3) * gate
        return self.output_projection(output.flatten(-2))


class MultiheadAttentionProjection(nn.Module):
    r"""
    Project query/key/value tensors into attention heads and merge head outputs back to hidden states.

    Owns the learned query/key/value/output projections so that
    [`MultiheadAttention`][ofoldx.modules.attention.multihead.MultiheadAttention] and the backend
    dispatch share one projection while the attention core stays backend-agnostic. Variants change the
    equation by overriding a seam:
    [`project_heads`][ofoldx.modules.attention.multihead.MultiheadAttentionProjection.project_heads]
    for the per-head queries, keys, and values, and
    [`merge`][ofoldx.modules.attention.multihead.MultiheadAttentionProjection.merge] for the head merge.

    Args:
        hidden_size: Width of the output token channel.
        num_attention_heads: Number of attention heads.
        head_size: Per-head width. Defaults to ``hidden_size // num_attention_heads``.
        query_size: Width of the query input. Defaults to ``hidden_size``.
        key_size: Width of the key input. Defaults to ``hidden_size``.
        value_size: Width of the value input. Defaults to ``hidden_size``.
        query_bias: Whether the query projection carries a learnable bias.
        output_bias: Whether the output projection carries a learnable bias.
        value_bias: Whether the value projection carries a learnable bias.

    Raises:
        ValueError: If ``head_size`` is not given and ``hidden_size`` is not divisible by
            ``num_attention_heads``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 16,
        head_size: int | None = None,
        query_size: int | None = None,
        key_size: int | None = None,
        value_size: int | None = None,
        query_bias: bool = True,
        output_bias: bool = False,
        *,
        value_bias: bool = False,
    ) -> None:
        super().__init__()
        if head_size is None and hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads if head_size is None else head_size
        self.query_size = hidden_size if query_size is None else query_size
        self.key_size = hidden_size if key_size is None else key_size
        self.value_size = hidden_size if value_size is None else value_size
        self.inner_size = self.head_size * self.num_attention_heads

        self.query_projection = nn.Linear(self.query_size, self.inner_size, bias=query_bias)
        self.key_projection = nn.Linear(self.key_size, self.inner_size, bias=False)
        self.value_projection = nn.Linear(self.value_size, self.inner_size, bias=value_bias)
        self.output_projection = nn.Linear(self.inner_size, hidden_size, bias=output_bias)

    def project_heads(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v and split their channels into heads.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            query: Query projection in `[..., q_tokens, heads, head_size]` layout.
            key: Key projection in `[..., kv_tokens, heads, head_size]` layout.
            value: Value projection in `[..., kv_tokens, heads, head_size]` layout.
        """

        heads = (self.num_attention_heads, self.head_size)
        query_projection = self.query_projection(query)
        key_projection = self.key_projection(key)
        value_projection = self.value_projection(value)
        if isinstance(query_projection, NestedTensor):
            return (
                query_projection.unflatten(-1, heads),
                key_projection.unflatten(-1, heads),
                value_projection.unflatten(-1, heads),
            )
        return (
            query_projection.view(*query.shape[:-1], *heads),
            key_projection.view(*key.shape[:-1], *heads),
            value_projection.view(*value.shape[:-1], *heads),
        )

    def project_unscaled(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v into head-major heads without scaling the query.

        The unscaled variant is used by backends that fold the ``1 / sqrt(head_size)`` scaling into
        their own attention core.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            query: Query projection in `[..., heads, tokens, head_size]` layout,
                with no query scaling applied.
            key: Key projection in `[..., heads, tokens, head_size]` layout.
            value: Value projection in `[..., heads, tokens, head_size]` layout.
        """

        query, key, value = self.project_heads(query, key, value)
        return query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)

    def project(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v into head-major heads and scale the query by ``1 / sqrt(head_size)``.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            query: Query projection in `[..., heads, tokens, head_size]` layout,
                already divided by ``sqrt(head_size)``.
            key: Key projection in `[..., heads, tokens, head_size]` layout.
            value: Value projection in `[..., heads, tokens, head_size]` layout.
        """

        query, key, value = self.project_unscaled(query, key, value)
        return query / math.sqrt(self.head_size), key, value

    def merge(self, output: Tensor, query: Tensor) -> Tensor:
        r"""
        Merge per-head attention outputs back to hidden states.

        Args:
            output: Attention output in `[..., tokens, heads, head_size]` layout (heads adjacent to
                ``head_size``, i.e. the attention core's output already transposed back).
            query: The original query `[..., tokens, query_size]`, read by merges that depend on it.

        Returns:
            Merged hidden states `[..., tokens, hidden_size]`.
        """

        if isinstance(output, NestedTensor):
            output = output.flatten(-2)
        else:
            output = output.reshape(*output.shape[:-2], self.inner_size)
        return self.output_projection(output)


class GatedMultiheadAttentionProjection(MultiheadAttentionProjection):
    r"""
    Multi-head attention projection whose merge gates every head output by a sigmoid of the query.

    Args:
        hidden_size: Width of the output token channel.
        num_attention_heads: Number of attention heads.
        head_size: Per-head width. Defaults to ``hidden_size // num_attention_heads``.
        query_size: Width of the query input. Defaults to ``hidden_size``.
        key_size: Width of the key input. Defaults to ``hidden_size``.
        value_size: Width of the value input. Defaults to ``hidden_size``.
        query_bias: Whether the query projection carries a learnable bias.
        output_bias: Whether the output projection carries a learnable bias.
        gate_bias: Whether the gate projection carries a learnable bias.
        gate_bias_init: Initial value of the gate bias when ``gate_bias`` is set.
        value_bias: Whether the value projection carries a learnable bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 16,
        head_size: int | None = None,
        query_size: int | None = None,
        key_size: int | None = None,
        value_size: int | None = None,
        query_bias: bool = True,
        output_bias: bool = False,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
        *,
        value_bias: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            num_attention_heads,
            head_size,
            query_size,
            key_size,
            value_size,
            query_bias,
            output_bias,
            value_bias=value_bias,
        )
        self.gate_projection = nn.Linear(self.query_size, self.inner_size, bias=gate_bias)
        if self.gate_projection.bias is not None:
            nn.init.constant_(self.gate_projection.bias, gate_bias_init)

    def merge(self, output: Tensor, query: Tensor) -> Tensor:
        r"""
        Gate the per-head attention outputs by ``sigmoid(gate_projection(query))``, then merge them.

        Args:
            output: Attention output in `[..., tokens, heads, head_size]` layout.
            query: The original query `[..., tokens, query_size]` the gate is computed from.

        Returns:
            Merged hidden states `[..., tokens, hidden_size]`.
        """

        gate = self.gate_projection(query).sigmoid()
        if isinstance(gate, NestedTensor):
            gate = gate.unflatten(-1, (self.num_attention_heads, self.head_size))
        else:
            gate = gate.view(*query.shape[:-1], self.num_attention_heads, self.head_size)
        return super().merge(output * gate, query)


class QueryKeyNormGatedMultiheadAttentionProjection(GatedMultiheadAttentionProjection):
    r"""
    Gated multi-head attention projection that layer-normalizes every projected query and key head.

    Args:
        hidden_size: Width of the output token channel.
        num_attention_heads: Number of attention heads.
        head_size: Per-head width. Defaults to ``hidden_size // num_attention_heads``.
        query_size: Width of the query input. Defaults to ``hidden_size``.
        key_size: Width of the key input. Defaults to ``hidden_size``.
        value_size: Width of the value input. Defaults to ``hidden_size``.
        query_bias: Whether the query projection carries a learnable bias.
        output_bias: Whether the output projection carries a learnable bias.
        gate_bias: Whether the gate projection carries a learnable bias.
        gate_bias_init: Initial value of the gate bias when ``gate_bias`` is set.
        backend: Operator provider of the query and key head normalizations.
        value_bias: Whether the value projection carries a learnable bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 16,
        head_size: int | None = None,
        query_size: int | None = None,
        key_size: int | None = None,
        value_size: int | None = None,
        query_bias: bool = True,
        output_bias: bool = False,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
        backend: str = "auto",
        *,
        value_bias: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            num_attention_heads,
            head_size,
            query_size,
            key_size,
            value_size,
            query_bias,
            output_bias,
            gate_bias,
            gate_bias_init,
            value_bias=value_bias,
        )
        self.query_norm = LayerNorm(self.head_size, backend=backend)
        self.key_norm = LayerNorm(self.head_size, backend=backend)

    def project_heads(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v into heads and layer-normalize every query and key head.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            query: Normalized query heads in `[..., q_tokens, heads, head_size]` layout.
            key: Normalized key heads in `[..., kv_tokens, heads, head_size]` layout.
            value: Value heads in `[..., kv_tokens, heads, head_size]` layout.
        """

        query, key, value = super().project_heads(query, key, value)
        return self.query_norm(query), self.key_norm(key), value
