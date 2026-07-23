r"""
Conditioned transformer stack shared by atom attention and diffusion token transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn

from .adaptive_norm import AdaptiveLayerNorm
from .attention.multihead import GatedMultiheadAttention
from .transition import ConditionedTransition

TransitionKind = Literal["double", "triple"]
CrossAttentionKeySource = Literal["raw", "normalized"]


class ConditionedTransformer(nn.Module):
    r"""
    Stack of [ConditionedTransformerBlock][ofoldx.modules.transformer.ConditionedTransformerBlock]
    that slices the precomputed per-block pair bias.

    The pair bias is computed once and packed as ``num_blocks * heads`` channels on the last axis;
    the stack hands each block its own per-head slice, so no per-block bias projection is needed.

    Args:
        num_blocks: Number of transformer blocks.
        num_attention_heads: Number of attention heads per block.
        hidden_size: Width of the token/atom channel that flows through the stack.
        conditioning_size: Width of the conditioning channel; defaults to ``hidden_size``.
        operator_backend: Which operator backend the attention runs; see
            [ofoldx.modules.backend][].
        cross_attention_mode: Whether each block adds a second adaptive norm on the key/value path.
        cross_attention_key_source: Whether key/value normalization starts from the raw hidden state
            or the query-normalized state.
        transition: Which conditioned transition each block uses, ``"triple"`` or ``"double"``.

    Examples:
        >>> import torch
        >>> transformer = ConditionedTransformer(
        ...     num_blocks=2,
        ...     num_attention_heads=2,
        ...     hidden_size=8,
        ...     conditioning_size=4,
        ...     operator_backend="eager",
        ... )
        >>> hidden = torch.randn(1, 5, 8)
        >>> conditioning = torch.randn(1, 5, 4)
        >>> bias = torch.randn(1, 5, 5, 2 * 2)  # num_blocks * num_attention_heads
        >>> transformer(hidden, conditioning, bias).shape
        torch.Size([1, 5, 8])
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int = 384,
        conditioning_size: int | None = None,
        operator_backend: str = "auto",
        cross_attention_mode: bool = False,
        cross_attention_key_source: CrossAttentionKeySource = "raw",
        transition: TransitionKind = "triple",
    ) -> None:
        super().__init__()
        conditioning_size = hidden_size if conditioning_size is None else conditioning_size
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            ConditionedTransformerBlock(
                hidden_size,
                conditioning_size,
                num_attention_heads,
                operator_backend,
                cross_attention_mode=cross_attention_mode,
                cross_attention_key_source=cross_attention_key_source,
                transition=transition,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        r"""
        Run the block stack, feeding each block its own slice of the packed pair bias.

        ``bias`` of shape `[..., q, k, num_blocks * heads]` is reshaped to
        `[..., q, k, num_blocks, heads]`, and the ``index``-th head slice is handed to block
        ``index``; ``hidden``, ``conditioning``, ``mask``, and ``to_keys`` are forwarded unchanged.
        Args:
            hidden: `[..., q, hidden_size]` token/atom activations.
            conditioning: `[..., q, conditioning_size]` conditioning activations.
            bias: `[..., q, k, num_blocks * heads]` precomputed per-head attention bias, packed for
                the whole stack on the last axis.
            mask: Optional `[..., k]` key-validity mask, forwarded to every block.
            to_keys: Optional windowed-key selector that gathers the key set for local attention;
                when ``None`` each block self-attends over the full axis.

        Returns:
            Updated hidden activations of shape `[..., q, hidden_size]`.

        Examples:
            >>> import torch
            >>> transformer = ConditionedTransformer(
            ...     num_blocks=2, num_attention_heads=2, hidden_size=8,
            ...     conditioning_size=4, operator_backend="eager",
            ... )
            >>> hidden = torch.randn(1, 5, 8)
            >>> conditioning = torch.randn(1, 5, 4)
            >>> bias = torch.randn(1, 5, 5, 4)
            >>> transformer(hidden, conditioning, bias).shape
            torch.Size([1, 5, 8])
        """
        # The precomputed bias packs (num_blocks * heads) on the last axis; one slice feeds each block.
        *prefix, packed = bias.shape
        num_blocks = len(self.blocks)
        bias = bias.view(*prefix, num_blocks, packed // num_blocks)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, conditioning, bias[..., index, :], mask, to_keys)
        return hidden


class ConditionedTransformerBlock(nn.Module):
    r"""
    One conditioned-transformer block: gated pair-biased attention plus a conditioned transition.

    The pair bias is supplied precomputed, already projected to per-head logits. ``to_keys``
    selects the windowed key set for local atom attention; when ``None`` the block self-attends
    over the full token/atom axis. ``cross_attention_mode`` adds a second adaptive norm on the
    key/value path. ``cross_attention_key_source`` selects whether that norm consumes the raw hidden
    or the already query-normalized activation. ``transition`` selects either the
    [TripleProductConditionedTransition][ofoldx.modules.transformer.TripleProductConditionedTransition] or the
    double-product [ConditionedTransition][ofoldx.modules.transition.ConditionedTransition].

    Args:
        hidden_size: Width of the token/atom channel.
        conditioning_size: Width of the conditioning channel driving the adaptive norms and gates.
        num_attention_heads: Number of attention heads.
        operator_backend: Which operator backend the attention runs; see
            [ofoldx.modules.backend][].
        cross_attention_mode: Whether to add a second adaptive norm on the key/value path.
        cross_attention_key_source: Whether key/value normalization starts from the raw hidden state
            or the query-normalized state.
        transition: Which conditioned transition to use, ``"triple"`` or ``"double"``.

    Raises:
        ValueError: If ``transition`` is neither ``"triple"`` nor ``"double"``.

    Examples:
        >>> import torch
        >>> block = ConditionedTransformerBlock(8, 4, num_attention_heads=2, operator_backend="eager")
        >>> hidden = torch.randn(1, 5, 8)
        >>> conditioning = torch.randn(1, 5, 4)
        >>> bias = torch.randn(1, 5, 5, 2)  # per-head bias for this block
        >>> block(hidden, conditioning, bias).shape
        torch.Size([1, 5, 8])
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        num_attention_heads: int,
        operator_backend: str = "auto",
        cross_attention_mode: bool = False,
        cross_attention_key_source: CrossAttentionKeySource = "raw",
        transition: TransitionKind = "triple",
    ) -> None:
        super().__init__()
        self.cross_attention_mode = cross_attention_mode
        self.cross_attention_key_source = cross_attention_key_source
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size)
        if cross_attention_mode:
            # A second adaptive norm on the key/value path.
            self.key_value_norm = AdaptiveLayerNorm(hidden_size, conditioning_size)
        self.attention = GatedMultiheadAttention(
            hidden_size,
            num_attention_heads=num_attention_heads,
            query_bias=True,
            gating=True,
            operator_backend=operator_backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        if transition == "triple":
            self.transition: nn.Module = TripleProductConditionedTransition(hidden_size, conditioning_size)
        elif transition == "double":
            self.transition = ConditionedTransition(hidden_size, conditioning_size, expansion=2)
        else:
            raise ValueError(f"unknown transition: {transition!r} (expected 'triple' or 'double')")
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        r"""
        Adaptive-norm, gated pair-biased attention, then conditioned transition, both added residually.

        The input is conditioned-normed to form the query; in cross-attention mode a second adaptive
        norm produces the key base. ``to_keys`` (if given) gathers the windowed key set, so keys have
        length ``k``, and key masks are gathered the same way, while pair masks already have
        `[..., q, k]` shape. The attention output is scaled by a sigmoid gate projected from
        ``conditioning`` and added to ``hidden``; the transition output is then added.

        Args:
            hidden: `[..., q, hidden_size]` query-side token/atom activations.
            conditioning: `[..., q, conditioning_size]` conditioning activations.
            bias: `[..., q, k, heads]` per-head attention bias for this block; internally moved to
                `[..., heads, q, k]` for the attention op.
            mask: Optional key-validity mask, either `[..., k]` or a pair mask `[..., q, k]`.
            to_keys: Optional windowed-key selector; when ``None`` the block self-attends over the
                full axis.

        Returns:
            Updated hidden activations of shape `[..., q, hidden_size]`.

        Examples:
            >>> import torch
            >>> block = ConditionedTransformerBlock(8, 4, num_attention_heads=2, operator_backend="eager")
            >>> hidden = torch.randn(1, 5, 8)
            >>> conditioning = torch.randn(1, 5, 4)
            >>> bias = torch.randn(1, 5, 5, 2)
            >>> block(hidden, conditioning, bias).shape
            torch.Size([1, 5, 8])
        """
        normalized = self.conditioned_norm(hidden, conditioning)
        # Cross-attention normalizes the key/value path independently. Normalizing before ``to_keys``
        # lets every gathered key use its own conditioning; ``cross_attention_key_source`` selects
        # whether that normalization starts from the raw or query-normalized hidden state.
        if self.cross_attention_mode:
            key_source = normalized if self.cross_attention_key_source == "normalized" else hidden
            key_base = self.key_value_norm(key_source, conditioning)
        else:
            key_base = normalized
        key_input = to_keys(key_base) if to_keys is not None else key_base
        attention_mask = None
        if mask is not None:
            key_length = key_input.shape[-2]
            if to_keys is not None and mask.shape[-1] != key_length:
                mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
                attention_mask = mask.unsqueeze(-2).expand(*mask.shape[:-1], normalized.shape[-2], key_length)
            else:
                attention_mask = _attention_mask_for_query_keys(
                    mask,
                    query_length=normalized.shape[-2],
                    key_length=key_length,
                )
        # bias arrives as (..., q, k, heads); GatedMultiheadAttention wants the head axis at -3.
        attention = self.attention(
            normalized,
            key=key_input,
            biases=(bias.movedim(-1, -3),),
            attention_mask=attention_mask,
        )
        gate = self.sigmoid(self.output_gate_projection(conditioning))
        hidden = hidden + gate * attention
        return hidden + self.transition(hidden, conditioning)


class TripleProductConditionedTransition(nn.Module):
    r"""
    Conditioning-gated SwiGLU transition whose activation is multiplied by a third expansion branch.

    Unlike the double-product
    [ConditionedTransition][ofoldx.modules.transition.ConditionedTransition], the SwiGLU
    activation is multiplied by a third ``expand_projection`` branch.

    Args:
        hidden_size: Input and output hidden width.
        conditioning_size: Conditioning tensor width driving the adaptive norm and output gate.
        expansion: Hidden expansion factor; the inner width is ``expansion * hidden_size``.

    Examples:
        >>> import torch
        >>> transition = TripleProductConditionedTransition(8, 4, expansion=2)
        >>> hidden = torch.randn(2, 3, 8)
        >>> conditioning = torch.randn(2, 3, 4)
        >>> transition(hidden, conditioning).shape
        torch.Size([2, 3, 8])
    """

    def __init__(self, hidden_size: int, conditioning_size: int, expansion: int = 2) -> None:
        super().__init__()
        inner_size = hidden_size * expansion
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size)
        self.gate_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.expand_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, hidden_size, bias=False)
        self.conditioning_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Triple-product SwiGLU transition with a conditioning-gated output.

        The input is conditioned-normed, then ``silu(gate) * value`` is multiplied by the
        ``expand_projection`` branch, projected back to ``hidden_size``, and scaled by a sigmoid
        gate projected from ``conditioning``.

        Args:
            hidden: `[..., hidden_size]` input activations.
            conditioning: `[..., conditioning_size]` conditioning activations.

        Returns:
            Transition output of shape `[..., hidden_size]`.

        Examples:
            >>> import torch
            >>> transition = TripleProductConditionedTransition(8, 4, expansion=2)
            >>> hidden = torch.randn(1, 5, 8)
            >>> conditioning = torch.randn(1, 5, 4)
            >>> transition(hidden, conditioning).shape
            torch.Size([1, 5, 8])
        """
        normalized = self.conditioned_norm(hidden, conditioning)
        gate = self.gate_projection(normalized)
        value = self.value_projection(normalized)
        activated = (self.silu(gate) * value) * self.expand_projection(normalized)
        conditioning_gate = self.sigmoid(self.conditioning_gate_projection(conditioning))
        return conditioning_gate * self.output_projection(activated)


def _attention_mask_for_query_keys(mask: Tensor, *, query_length: int, key_length: int) -> Tensor:
    r"""
    Broadcast a key or pair mask to the ``(query_length, key_length)`` attention shape.

    Args:
        mask: Either a pair mask already shaped `[..., query_length, key_length]` or a key mask
            `[..., key_length]` that is broadcast across the query axis.
        query_length: Expected query axis length.
        key_length: Expected key axis length.

    Returns:
        Attention mask of shape `[..., query_length, key_length]`.

    Raises:
        ValueError: If ``mask`` matches neither the pair shape nor the key shape.
    """
    if mask.shape[-2:] == (query_length, key_length):
        return mask
    if mask.shape[-1:] == (key_length,):
        return mask.unsqueeze(-2).expand(*mask.shape[:-1], query_length, key_length)
    raise ValueError(
        "transformer mask must have shape broadcastable to "
        f"(..., {query_length}, {key_length}) or (..., {key_length}), got {tuple(mask.shape)}"
    )
