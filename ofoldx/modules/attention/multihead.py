r"""
Multi-head attention primitives shared across the trunk, atom, and diffusion stacks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch import Tensor, nn

from ..backend import resolve_config_backend
from ..backend.attention import gated_multihead_attention, projected_attention as dispatch_projected_attention


class GatedMultiheadAttention(nn.Module):
    r"""
    Full multi-head attention primitive with optional output gating and additive pair bias.

    Wraps a [`MultiheadAttentionProjection`][ofoldx.modules.attention.multihead.MultiheadAttentionProjection]
    and dispatches the scaled-dot-product core to the resolved backend
    through the selected operator provider. This is the shared attention block used by the trunk,
    atom, and diffusion stacks; higher-level modules supply the pair-derived ``biases``.

    Args:
        hidden_size: Width of the input/output token channel.
        num_attention_heads: Number of attention heads; must divide ``hidden_size`` unless
            ``head_dim`` is set explicitly.
        head_dim: Per-head dimension. Defaults to ``hidden_size // num_attention_heads``.
        query_size: Width of the query input. Defaults to ``hidden_size``.
        key_size: Width of the key input. Defaults to ``hidden_size``.
        value_size: Width of the value input. Defaults to ``hidden_size``.
        query_bias: Whether the query projection carries a learnable bias.
        output_bias: Whether the output projection carries a learnable bias.
        gating: Whether to apply the sigmoid output gate.
        gate_bias: Whether the gate projection carries a learnable bias.
        gate_bias_init: Initial value for the gate bias when ``gate_bias`` is set.
        backend: Operator provider; attention uses SDPA as its default
            PyTorch implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 16,
        head_dim: int | None = None,
        query_size: int | None = None,
        key_size: int | None = None,
        value_size: int | None = None,
        query_bias: bool = True,
        output_bias: bool = False,
        gating: bool = True,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "sdpa", "eager"), public_default="sdpa")
        self.projection = MultiheadAttentionProjection(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            query_size=query_size,
            key_size=key_size,
            value_size=value_size,
            query_bias=query_bias,
            output_bias=output_bias,
            gating=gating,
            gate_bias=gate_bias,
            gate_bias_init=gate_bias_init,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        biases: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Run gated multi-head attention, dispatching to the configured backend.

        Projects q/k/v into heads, computes biased and masked attention via the selected OOps or
        PyTorch implementation, then merges heads applying the optional sigmoid gate.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`; defaults to ``query`` (self-attention).
            value: Value tokens `[..., kv_tokens, value_size]`; defaults to ``key``.
            biases: Additive pair biases, each broadcastable to `[..., heads, q_tokens, kv_tokens]`.
            attention_mask: Boolean keep-mask broadcastable to `[..., q_tokens, kv_tokens]` or
                `[..., kv_tokens]`; ``False`` positions are excluded from the softmax.

        Returns:
            Updated hidden states `[..., q_tokens, hidden_size]`.
        """

        return gated_multihead_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            projection=self.projection,
            backend=self.backend,
        )


class ProjectedAttention(nn.Module):
    r"""
    Scaled dot-product attention over already projected head-major tensors.

    This parameter-free Module keeps layout construction and learned projections at
    their mathematical owner while moving the attention core behind the common
    backend policy. The eager implementation accumulates logits and weighted values
    in float32, returns the value dtype, and zeros query rows with no valid key.

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
                ``(..., num_heads, num_query_tokens, head_size)``.
            key: Head-major key tensor
                ``(..., num_heads, num_key_tokens, head_size)``.
            value: Head-major value tensor
                ``(..., num_heads, num_key_tokens, value_size)``.
            biases: Optional additive biases broadcastable to
                ``(..., num_heads, num_query_tokens, num_key_tokens)``.
            attention_mask: Optional boolean keep-mask. A full-rank mask
                includes the head axis, one fewer axis omits it, and a
                shorter mask supplies only key validity.
            scale: Optional multiplier applied to query-key logits. ``None`` uses
                ``head_size ** -0.5``.

        Returns:
            Head-major attention values with shape
            ``(..., num_heads, num_query_tokens, value_size)``.
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


class MultiheadAttentionProjection(nn.Module):
    r"""
    Project query/key/value tensors into attention heads and merge head outputs back to hidden states.

    Owns the learnable linear projections (query/key/value/output plus an optional query bias and
    output gate) so that [`GatedMultiheadAttention`][ofoldx.modules.attention.multihead.GatedMultiheadAttention]
    and the backend dispatch can share one projection module while keeping the attention core
    backend-agnostic.

    Args:
        hidden_size: Width of the output token channel.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimension. Defaults to ``hidden_size // num_attention_heads``.
        query_size: Width of the query input. Defaults to ``hidden_size``.
        key_size: Width of the key input. Defaults to ``hidden_size``.
        value_size: Width of the value input. Defaults to ``hidden_size``.
        query_bias: Whether the query projection carries a learnable bias.
        output_bias: Whether the output projection carries a learnable bias.
        gating: Whether to build and apply the sigmoid output gate.
        gate_bias: Whether the gate projection carries a learnable bias.
        gate_bias_init: Initial value for the gate projection bias when ``gate_bias`` is enabled.

    Raises:
        ValueError: If ``head_dim`` is not given and ``hidden_size`` is not divisible by
            ``num_attention_heads``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 16,
        head_dim: int | None = None,
        query_size: int | None = None,
        key_size: int | None = None,
        value_size: int | None = None,
        query_bias: bool = True,
        output_bias: bool = False,
        gating: bool = True,
        gate_bias: bool = False,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim is None and hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.query_size = hidden_size if query_size is None else query_size
        self.key_size = hidden_size if key_size is None else key_size
        self.value_size = hidden_size if value_size is None else value_size
        self.inner_size = self.head_dim * self.num_attention_heads
        self.gating = gating

        if query_bias:
            self.query_projection = nn.Linear(self.query_size, self.inner_size)
        else:
            self.query_projection = nn.Linear(self.query_size, self.inner_size, bias=False)
        self.key_projection = nn.Linear(self.key_size, self.inner_size, bias=False)
        self.value_projection = nn.Linear(self.value_size, self.inner_size, bias=False)
        self.output_projection = nn.Linear(self.inner_size, hidden_size, bias=output_bias)
        if self.gating:
            self.gate_projection = nn.Linear(self.query_size, self.inner_size, bias=gate_bias)
            if self.gate_projection.bias is not None:
                nn.init.constant_(self.gate_projection.bias, gate_bias_init)
        self.sigmoid = nn.Sigmoid()

    def project(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v into heads and scale the query by ``1 / sqrt(head_dim)``.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            The query, key, and value projections, each in `[..., heads, tokens, head_dim]` layout,
            with the query already divided by ``sqrt(head_dim)``.
        """

        scale = math.sqrt(self.head_dim)
        query_projection = self.query_projection(query).view(*query.shape[:-1], self.num_attention_heads, self.head_dim)
        key_projection = self.key_projection(key).view(*key.shape[:-1], self.num_attention_heads, self.head_dim)
        value_projection = self.value_projection(value).view(*value.shape[:-1], self.num_attention_heads, self.head_dim)
        query_projection = query_projection.transpose(-2, -3) / scale
        key_projection = key_projection.transpose(-2, -3)
        value_projection = value_projection.transpose(-2, -3)
        return query_projection, key_projection, value_projection

    def project_unscaled(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Project q/k/v into heads without scaling the query.

        The unscaled variant is used by backends that fold the ``1 / sqrt(head_dim)`` scaling into
        their own attention core.

        Args:
            query: Query tokens `[..., q_tokens, query_size]`.
            key: Key tokens `[..., kv_tokens, key_size]`.
            value: Value tokens `[..., kv_tokens, value_size]`.

        Returns:
            The query, key, and value projections, each in `[..., heads, tokens, head_dim]` layout,
            with no query scaling applied.
        """

        query_projection = self.query_projection(query).view(*query.shape[:-1], self.num_attention_heads, self.head_dim)
        key_projection = self.key_projection(key).view(*key.shape[:-1], self.num_attention_heads, self.head_dim)
        value_projection = self.value_projection(value).view(*value.shape[:-1], self.num_attention_heads, self.head_dim)
        return (
            query_projection.transpose(-2, -3),
            key_projection.transpose(-2, -3),
            value_projection.transpose(-2, -3),
        )

    def merge(self, output: Tensor, query: Tensor) -> Tensor:
        r"""
        Merge per-head attention outputs back to hidden states, applying the optional sigmoid gate.

        Args:
            output: Attention output in `[..., tokens, heads, head_dim]` layout (heads adjacent to
                ``head_dim``, i.e. the attention core's output already transposed back).
            query: The original query `[..., tokens, query_size]`, used to compute the output gate.

        Returns:
            Merged hidden states `[..., tokens, hidden_size]`.
        """

        if self.gating:
            gate = self.sigmoid(self.gate_projection(query)).view(
                *query.shape[:-1], self.num_attention_heads, self.head_dim
            )
            output = output * gate
        output = output.reshape(*output.shape[:-2], self.inner_size)
        return self.output_projection(output)
