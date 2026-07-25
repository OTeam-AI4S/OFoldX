r"""
Concrete conditioned Transformer-encoder equations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor, nn

from .adaptive_norm import AdaptiveLayerNorm
from .attention.multihead import GatedMultiheadAttention
from .transition import ConditionedTransition


class ConditionedTransformerEncoder(nn.Module):
    r"""
    Homogeneous triple-product conditioned Transformer-encoder blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int = 384,
        conditioning_size: int | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        conditioning_size = hidden_size if conditioning_size is None else conditioning_size
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            ConditionedTransformerEncoderBlock(
                hidden_size,
                conditioning_size,
                num_attention_heads,
                backend,
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
        return _run_blocks(self.blocks, hidden, conditioning, bias, mask, to_keys)


class DenoisingTransformerEncoder(nn.Module):
    r"""
    Homogeneous double-product conditioned Transformer-encoder blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int = 384,
        conditioning_size: int | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        conditioning_size = hidden_size if conditioning_size is None else conditioning_size
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            DenoisingTransformerEncoderBlock(
                hidden_size,
                conditioning_size,
                num_attention_heads,
                backend,
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
        return _run_blocks(self.blocks, hidden, conditioning, bias, mask, to_keys)


class AtomCrossAttentionTransformer(nn.Module):
    r"""
    Atom cross-attention blocks whose key normalization starts from raw hidden states.
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int,
        conditioning_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            AtomCrossAttentionTransformerBlock(
                hidden_size,
                conditioning_size,
                num_attention_heads,
                backend,
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
        return _run_blocks(self.blocks, hidden, conditioning, bias, mask, to_keys)


class NormalizedKeyAtomCrossAttentionTransformer(nn.Module):
    r"""
    Atom cross-attention blocks whose key normalization starts from normalized queries.
    """

    def __init__(
        self,
        num_blocks: int,
        num_attention_heads: int,
        hidden_size: int,
        conditioning_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            NormalizedKeyAtomCrossAttentionTransformerBlock(
                hidden_size,
                conditioning_size,
                num_attention_heads,
                backend,
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
        return _run_blocks(self.blocks, hidden, conditioning, bias, mask, to_keys)


class ConditionedTransformerEncoderBlock(nn.Module):
    r"""
    Pair-biased encoder self-attention followed by a triple-product conditioned transition.
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
        self.attention = GatedMultiheadAttention(
            hidden_size,
            num_attention_heads=num_attention_heads,
            query_bias=True,
            gating=True,
            backend=backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.transition = TripleProductConditionedTransition(
            hidden_size,
            conditioning_size,
            backend=backend,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        hidden = hidden + _attention_update(
            normalized,
            normalized,
            conditioning,
            bias,
            mask,
            to_keys,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            sigmoid=self.sigmoid,
        )
        return hidden + self.transition(hidden, conditioning)


class DenoisingTransformerEncoderBlock(nn.Module):
    r"""
    Pair-biased encoder self-attention followed by a double-product conditioned transition.
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
        self.attention = GatedMultiheadAttention(
            hidden_size,
            num_attention_heads=num_attention_heads,
            query_bias=True,
            gating=True,
            backend=backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.transition = ConditionedTransition(
            hidden_size,
            conditioning_size,
            expansion=2,
            backend=backend,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        hidden = hidden + _attention_update(
            normalized,
            normalized,
            conditioning,
            bias,
            mask,
            to_keys,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            sigmoid=self.sigmoid,
        )
        return hidden + self.transition(hidden, conditioning)


class AtomCrossAttentionTransformerBlock(DenoisingTransformerEncoderBlock):
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
        super().__init__(hidden_size, conditioning_size, num_attention_heads, backend)
        self.key_value_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        key_base = self.key_value_norm(hidden, conditioning)
        hidden = hidden + _attention_update(
            normalized,
            key_base,
            conditioning,
            bias,
            mask,
            to_keys,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            sigmoid=self.sigmoid,
        )
        return hidden + self.transition(hidden, conditioning)


class NormalizedKeyAtomCrossAttentionTransformerBlock(AtomCrossAttentionTransformerBlock):
    r"""
    Double-product atom cross-attention whose key normalization starts from the query-normalized state.
    """

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        bias: Tensor,
        mask: Tensor | None = None,
        to_keys: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        key_base = self.key_value_norm(normalized, conditioning)
        hidden = hidden + _attention_update(
            normalized,
            key_base,
            conditioning,
            bias,
            mask,
            to_keys,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            sigmoid=self.sigmoid,
        )
        return hidden + self.transition(hidden, conditioning)


class TripleProductConditionedTransition(nn.Module):
    r"""
    Conditioning-gated SwiGLU transition multiplied by a third expansion branch.
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        inner_size = hidden_size * expansion
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.gate_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.expand_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, hidden_size, bias=False)
        self.conditioning_gate_projection = nn.Linear(conditioning_size, hidden_size)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        normalized = self.conditioned_norm(hidden, conditioning)
        activated = (
            self.silu(self.gate_projection(normalized))
            * self.value_projection(normalized)
            * self.expand_projection(normalized)
        )
        gate = self.sigmoid(self.conditioning_gate_projection(conditioning))
        return gate * self.output_projection(activated)


def _run_blocks(
    blocks: Iterable[nn.Module],
    hidden: Tensor,
    conditioning: Tensor,
    bias: Tensor,
    mask: Tensor | None,
    to_keys: Callable[[Tensor], Tensor] | None,
) -> Tensor:
    blocks = tuple(blocks)
    if not blocks:
        return hidden
    *prefix, packed = bias.shape
    bias = bias.view(*prefix, len(blocks), packed // len(blocks))
    for index, block in enumerate(blocks):
        hidden = block(hidden, conditioning, bias[..., index, :], mask, to_keys)
    return hidden


def _attention_update(
    query: Tensor,
    key_base: Tensor,
    conditioning: Tensor,
    bias: Tensor,
    mask: Tensor | None,
    to_keys: Callable[[Tensor], Tensor] | None,
    *,
    attention: GatedMultiheadAttention,
    output_gate_projection: nn.Linear,
    sigmoid: nn.Sigmoid,
) -> Tensor:
    key = to_keys(key_base) if to_keys is not None else key_base
    attention_mask = None
    if mask is not None:
        key_length = key.shape[-2]
        if to_keys is not None and mask.shape[-1] != key_length:
            key_mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
            attention_mask = key_mask.unsqueeze(-2).expand(*key_mask.shape[:-1], query.shape[-2], key_length)
        else:
            attention_mask = _attention_mask_for_query_keys(
                mask,
                query_length=query.shape[-2],
                key_length=key_length,
            )
    update = attention(
        query,
        key=key,
        biases=(bias.movedim(-1, -3),),
        attention_mask=attention_mask,
    )
    return sigmoid(output_gate_projection(conditioning)) * update


def _attention_mask_for_query_keys(mask: Tensor, *, query_length: int, key_length: int) -> Tensor:
    if mask.shape[-2:] == (query_length, key_length):
        return mask
    if mask.shape[-1:] == (key_length,):
        return mask.unsqueeze(-2).expand(*mask.shape[:-1], query_length, key_length)
    raise ValueError(
        "transformer mask must have shape broadcastable to "
        f"(..., {query_length}, {key_length}) or (..., {key_length}), got {tuple(mask.shape)}"
    )
