r"""
Backend-dispatched attention operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ._apex import apex_evoformer_attention, apex_evoformer_attention_supported
from ._cueq import cueq_triangle_attention
from ._deepspeed import deepspeed_evoformer_attention
from ._oops import oops_kernel
from .resolve import resolve_tensor_backend

_oops_gated_pair_bias_attention = oops_kernel("gated_pair_bias_attention")
_oops_pair_bias_attention = oops_kernel("pair_bias_attention")
_oops_triangle_attention = oops_kernel("triangle_attention")

if TYPE_CHECKING:
    from ..attention.multihead import MultiheadAttentionProjection


def gated_multihead_attention(
    query: Tensor,
    key: Tensor | None,
    value: Tensor | None,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    projection: MultiheadAttentionProjection,
    backend: str,
) -> Tensor:
    r"""
    Project q/k/v via ``projection``, run attention on ``backend``, and merge heads.

    Dispatches attention to the selected provider when that provider implements
    the concrete tensor contract, otherwise to PyTorch SDPA. ``biases`` are
    summed into one additive pair bias. ``attention_mask`` follows the common
    pair-first mask contract and is normalized before provider selection.

    Args:
        query: ``(..., q_tokens, query_size)`` query input.
        key: ``(..., k_tokens, key_size)`` key input; defaults to ``query`` when ``None``.
        value: ``(..., k_tokens, value_size)`` value input; defaults to ``key`` when ``None``.
        biases: Optional additive pair biases, each broadcastable to ``(..., heads, q_tokens, k_tokens)``.
        attention_mask: Optional boolean pair-first keep-mask. ``(q_tokens,
            k_tokens)`` is a shared pair mask, ``(k_tokens,)`` is shared key
            validity, and prefix-specific key validity uses ``(..., k_tokens)``
            when unambiguous or ``(..., 1, k_tokens)`` otherwise. An explicit
            head-specific form retains every prefix position followed by the
            head axis.
        projection: The q/k/v (and optional gate/output) projection owned by the calling module.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``(..., q_tokens, hidden_size)`` attention output with heads merged.
    """

    key = query if key is None else key
    value = key if value is None else value
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    supported: tuple[str, ...]
    if isinstance(query, NestedTensor):
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError("nested gated attention requires NestedTensor query/key/value")
        if attention_mask is not None and not isinstance(attention_mask, NestedTensor):
            raise ValueError("nested gated attention requires a NestedTensor mask")
        supported = ("oops",)
    else:
        attention_mask = _normalize_attention_mask(
            attention_mask,
            prefix_shape=query.shape[:-2],
            num_heads=projection.num_attention_heads,
            query_length=query.shape[-2],
            key_length=key.shape[-2],
        )
        oops_mask_supported = (
            attention_mask is None or attention_mask.dim() != query.dim() + 1 or attention_mask.shape[-3] == 1
        )
        supported = (
            ("oops", "apex", "deepspeed", "sdpa", "eager")
            if oops_mask_supported
            else ("apex", "deepspeed", "sdpa", "eager")
        )
    backend = resolve_tensor_backend(
        backend,
        tensors=(query, key, value, attention_mask, *(biases or ())),
        supported=supported,
        public_default="sdpa",
    )
    if isinstance(query, NestedTensor) and backend != "oops":
        raise NotImplementedError("NestedTensor gated attention requires the OOps backend")
    if backend in {"eager", "apex", "deepspeed"}:
        query_projection, key_projection, value_projection = projection.project_unscaled(query, key, value)
    else:
        query_projection, key_projection, value_projection = projection.project(query, key, value)
    if backend == "eager":
        output = _eager_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
        )
    elif backend == "oops":
        can_use_fused_gated_output = (
            projection.gating and projection.output_projection.bias is None and projection.gate_projection.bias is None
        )
        if can_use_fused_gated_output:
            gated_output = _oops_attention_pair_bias(
                query,
                query_projection,
                key_projection,
                value_projection,
                biases=biases,
                attention_mask=attention_mask,
                projection=projection,
            )
            if gated_output is not None:
                return gated_output
        output = _oops_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
        )
    elif backend == "apex":
        apex_output = _apex_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
        )
        output = (
            _sdpa_attention(
                query_projection,
                key_projection,
                value_projection,
                biases=biases,
                attention_mask=attention_mask,
                scale=None,
            )
            if apex_output is None
            else apex_output
        )
    elif backend == "deepspeed":
        deepspeed_output = _deepspeed_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
        )
        output = (
            _sdpa_attention(
                query_projection,
                key_projection,
                value_projection,
                biases=biases,
                attention_mask=attention_mask,
                scale=None,
            )
            if deepspeed_output is None
            else deepspeed_output
        )
    else:
        output = _sdpa_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
            scale=1.0,
        )
    output = _zero_fully_masked_rows(output, attention_mask)
    output = output.transpose(-2, -3)
    return projection.merge(output, query)


def projected_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    biases: Sequence[Tensor] | None = None,
    attention_mask: Tensor | None,
    backend: str,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> Tensor:
    r"""
    Apply attention to projected head-major query, key, and value tensors.

    Args:
        query: Query tensor with shape
            ``(..., num_heads, num_query_tokens, head_size)``.
        key: Key tensor with shape
            ``(..., num_heads, num_key_tokens, head_size)``.
        value: Value tensor with shape
            ``(..., num_heads, num_key_tokens, value_size)``.
        biases: Optional additive pair biases broadcastable to
            ``(..., num_heads, num_query_tokens, num_key_tokens)``.
        attention_mask: Optional boolean pair-first keep-mask. A trailing
            ``(num_query_tokens, num_key_tokens)`` payload is a pair mask;
            ``(num_key_tokens,)`` is shared key validity, and an explicit
            singleton query axis disambiguates prefix-specific key validity.
            Head-specific masks retain every prefix position followed by the
            head axis.
        backend: Backend used for the attention operation.
        dropout_p: Dropout probability applied to attention weights.
        scale: Optional multiplier applied to query-key logits. ``None`` uses
            ``head_size ** -0.5``.

    Returns:
        Attention values with shape
            ``(..., num_heads, num_query_tokens, value_size)``.
    """

    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    nested_query = isinstance(query, NestedTensor)
    supported: tuple[str, ...]
    if nested_query:
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError("nested projected attention requires NestedTensor query/key/value")
        if attention_mask is not None and not isinstance(attention_mask, NestedTensor):
            raise ValueError("nested projected attention requires a NestedTensor mask")
    else:
        attention_mask = _normalize_attention_mask(
            attention_mask,
            prefix_shape=query.shape[:-3],
            num_heads=query.shape[-3],
            query_length=query.shape[-2],
            key_length=key.shape[-2],
        )
    oops_supported = (
        dropout_p == 0.0
        and value.shape[-1] == query.shape[-1]
        and (attention_mask is None or attention_mask.dim() != query.dim() or attention_mask.shape[-3] == 1)
    )
    if nested_query:
        if not oops_supported:
            raise NotImplementedError(
                "NestedTensor projected attention requires zero dropout and matching query/value head sizes"
            )
        supported = ("oops",)
    else:
        supported = ("oops", "sdpa", "eager") if oops_supported else ("sdpa", "eager")
    resolved = resolve_tensor_backend(
        backend,
        tensors=(query, key, value, attention_mask, *(biases or ())),
        supported=supported,
        public_default="sdpa",
    )
    if nested_query and resolved != "oops":
        raise NotImplementedError("NestedTensor projected attention requires the OOps backend")
    if resolved == "eager":
        output = _eager_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
        )
    elif resolved == "oops":
        output = _oops_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            scale=scale,
        )
    else:
        output = _sdpa_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
        )
    return _zero_fully_masked_rows(output, attention_mask)


def triangle_attention(
    pair: Tensor,
    triangle_bias: Tensor,
    attention_mask: Tensor | None,
    *,
    projection: MultiheadAttentionProjection,
    backend: str,
) -> Tensor:
    r"""
    Row-batched triangle self-attention over the pair representation.

    Projects the (already layer-normed) ``pair`` into per-row attention heads and attends along the
    key/column axis with the pre-computed per-head ``triangle_bias``. When the resolved backend is
    ``oops`` and the output and gate projections are bias-free,
    the dedicated OOps-owned ``triangle_attention`` implementation is used, which consumes the ``(..., rows, cols,
    heads, head_dim)`` q/k/v layout and the row-broadcast ``(..., rows|1, heads, cols, cols)`` bias
    directly instead of flattening the row axis into a generic pair-bias attention. Every other backend
    (``sdpa``/``eager``, or ``oops`` with biased projections) folds the key mask into an
    additive bias and delegates to [`gated_multihead_attention`][], equivalent to attending on
    ``pair`` with ``biases=[mask_bias, triangle_bias]``.

    Args:
        pair: Layer-normed pair representation ``(..., rows, cols, pair_size)``.
        triangle_bias: Per-head triangle bias ``(..., 1, heads, cols, cols)`` (row axis broadcast).
        attention_mask: Optional boolean keep-mask ``(..., rows, cols)``; ``False`` keys are excluded.
        projection: The q/k/v/gate/output projection owned by the calling triangle-attention module.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        Gated pair update ``(..., rows, cols, pair_size)``.
    """

    resolved = resolve_tensor_backend(
        backend,
        tensors=(pair, triangle_bias, attention_mask),
        supported=("oops", "cueq", "apex", "deepspeed", "sdpa", "eager"),
        public_default="sdpa",
    )
    can_flash = (
        resolved == "oops"
        and projection.gating
        and not projection.use_qk_norm
        and projection.output_projection.bias is None
        and projection.gate_projection.bias is None
    )
    if can_flash:
        return _oops_triangle_flash(pair, triangle_bias, attention_mask, projection=projection)
    if resolved == "cueq" and pair.is_cuda:
        return _cueq_triangle_flash(pair, triangle_bias, attention_mask, projection=projection)
    key_mask = None if attention_mask is None else attention_mask.unsqueeze(-2)
    return gated_multihead_attention(
        pair,
        None,
        None,
        biases=(triangle_bias,),
        attention_mask=key_mask,
        projection=projection,
        backend=backend,
    )


def _oops_triangle_flash(
    pair: Tensor,
    triangle_bias: Tensor,
    attention_mask: Tensor | None,
    *,
    projection: MultiheadAttentionProjection,
) -> Tensor:
    r"""
    Run the OOps-owned ``triangle_attention`` implementation and apply the sigmoid output gate.

    The OOps callable accepts q/k/v in ``(..., rows, cols, heads, head_dim)`` layout (the natural
    projection output, no head-front transpose), the mask is the boolean
    ``(..., rows, cols)`` keep-mask, and the bias is the row-broadcast ``(..., rows|1, heads, cols,
    cols)`` tensor. The ``1/sqrt(head_dim)`` scaling is passed directly to OOps dispatch, so the
    projections are left unscaled; the gate and output projection are then applied by ``projection.merge``.
    """

    heads = projection.num_attention_heads
    head_dim = projection.head_dim
    query = projection.query_projection(pair)
    key = projection.key_projection(pair)
    value = projection.value_projection(pair)
    if isinstance(pair, NestedTensor):
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))
    else:
        query = query.view(*pair.shape[:-1], heads, head_dim)
        key = key.view(*pair.shape[:-1], heads, head_dim)
        value = value.view(*pair.shape[:-1], heads, head_dim)
    mask = attention_mask.bool() if attention_mask is not None else None
    output = _oops_triangle_attention(
        query,
        key,
        value,
        mask,
        triangle_bias.to(dtype=pair.dtype),
        scale=head_dim**-0.5,
    )
    return projection.merge(cast(Tensor, output), pair)


def _cueq_triangle_flash(
    pair: Tensor,
    triangle_bias: Tensor,
    attention_mask: Tensor | None,
    *,
    projection: MultiheadAttentionProjection,
) -> Tensor:
    r"""
    Run cuEquivariance triangle attention and apply OFoldX's output gate.
    """

    prefix_shape = pair.shape[:-3]
    rows, columns = pair.shape[-3:-1]
    heads, head_dim = projection.num_attention_heads, projection.head_dim
    query, key, value = projection.project_unscaled(pair, pair, pair)
    query = query.reshape(-1, rows, heads, columns, head_dim)
    key = key.reshape(-1, rows, heads, columns, head_dim)
    value = value.reshape(-1, rows, heads, columns, head_dim)
    target_bias_shape = (*prefix_shape, 1, heads, columns, columns)
    bias = _broadcast_to_attention_shape(triangle_bias, target_bias_shape, tail_dims=4)
    mask = None
    if attention_mask is not None:
        target_mask_shape = (*prefix_shape, rows, columns)
        mask = _broadcast_to_attention_shape(attention_mask.bool(), target_mask_shape, tail_dims=2)
        mask = mask.reshape(-1, rows, 1, 1, columns)
    output = cueq_triangle_attention(
        query,
        key,
        value,
        bias.reshape(-1, 1, heads, columns, columns),
        mask,
        scale=head_dim**-0.5,
    )
    if mask is not None:
        output = output.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
    output = output.reshape(*prefix_shape, rows, heads, columns, head_dim)
    output = output.transpose(-2, -3)
    return projection.merge(output, pair)


def _normalize_attention_mask(
    attention_mask: Tensor | None,
    *,
    prefix_shape: torch.Size,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> Tensor | None:
    r"""
    Normalize a public pair-first keep-mask before provider dispatch.

    A mask without a head axis is returned as ``(..., q|1, k)``; an
    explicitly head-specific mask is returned as ``(..., heads|1, q|1, k)``.
    Prefix axes are expanded to ``prefix_shape`` while singleton query and
    head axes are retained for provider capability checks. Head-specific
    forms include every prefix position so the head axis cannot change
    ownership when two dimension sizes happen to coincide.
    """

    if attention_mask is None:
        return None
    mask = attention_mask.bool()

    if mask.dim() >= 2 and mask.shape[-2] in {1, query_length} and mask.shape[-1] in {1, key_length}:
        query_payload = 1 if mask.shape[-2] == 1 else query_length
        if mask.dim() - 2 <= len(prefix_shape):
            normalized = _try_broadcast_to_attention_shape(
                mask,
                (*prefix_shape, query_payload, key_length),
                tail_dims=2,
            )
            if normalized is not None:
                return normalized

        if mask.dim() == len(prefix_shape) + 3 and mask.shape[-3] in {1, num_heads}:
            head_payload = 1 if mask.shape[-3] == 1 else num_heads
            normalized = _try_broadcast_to_attention_shape(
                mask,
                (*prefix_shape, head_payload, query_payload, key_length),
                tail_dims=3,
            )
            if normalized is not None:
                return normalized

    if mask.dim() == len(prefix_shape) + 2 and mask.shape[-2] in {1, num_heads} and mask.shape[-1] in {1, key_length}:
        head_payload = 1 if mask.shape[-2] == 1 else num_heads
        normalized = _try_broadcast_to_attention_shape(
            mask,
            (*prefix_shape, head_payload, key_length),
            tail_dims=2,
        )
        if normalized is not None:
            return normalized.unsqueeze(-2)

    if mask.dim() >= 1 and mask.shape[-1] in {1, key_length} and mask.dim() - 1 <= len(prefix_shape):
        normalized = _try_broadcast_to_attention_shape(
            mask,
            (*prefix_shape, key_length),
            tail_dims=1,
        )
        if normalized is not None:
            return normalized.unsqueeze(-2)

    raise ValueError(
        "attention_mask must be a pair-first boolean mask with trailing "
        f"({query_length}, {key_length}), (..., 1, {key_length}), "
        f"(..., {num_heads}, {key_length}), or ({key_length},); "
        f"got {tuple(attention_mask.shape)}"
    )


def _try_broadcast_to_attention_shape(
    tensor: Tensor,
    target_shape: tuple[int, ...],
    *,
    tail_dims: int,
) -> Tensor | None:
    r"""
    Return the ordinary broadcast to ``target_shape``, or ``None`` when incompatible.
    """

    try:
        return _broadcast_to_attention_shape(tensor, target_shape, tail_dims=tail_dims)
    except RuntimeError:
        return None


def _eager_attention(
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> Tensor:
    r"""
    Pure PyTorch attention with explicit float32 logits and softmax.

    Unlike SDPA, this path explicitly performs the attention logits and softmax
    in float32. Masked keys are excluded after additive biases are applied, so
    no finite public bias can revive them. It is selected only by an explicit
    ``backend="eager"`` request.
    """

    head_dim = query_projection.shape[-1]
    scale = head_dim**-0.5 if scale is None else scale
    logits = torch.einsum("...hid,...hjd->...hij", query_projection.float(), key_projection.float())
    logits = logits * scale
    if biases is not None:
        for bias in biases:
            if bias.dim() == logits.dim() - 1:
                bias = bias.unsqueeze(-3)
            logits = logits + bias.to(dtype=torch.float32)
    if attention_mask is not None:
        mask = _broadcast_attention_mask(
            query_projection,
            attention_mask,
            key_length=key_projection.shape[-2],
        )
        logits.masked_fill_(~mask, torch.finfo(logits.dtype).min)
    attention = logits.softmax(dim=-1)
    if dropout_p:
        attention = F.dropout(attention, p=dropout_p, training=True)
    return torch.einsum("...hij,...hjd->...hid", attention, value_projection.float()).to(value_projection.dtype)


def _sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    dropout_p: float = 0.0,
    scale: float | None,
) -> Tensor:
    r"""
    Run PyTorch scaled-dot-product attention.
    """

    attention_bias: Tensor | None
    if not biases and attention_mask is not None:
        attention_bias = _attention_mask_view(query, attention_mask)
    else:
        attention_bias = _sum_biases(
            _attention_biases(
                query,
                key,
                biases=biases,
                attention_mask=attention_mask,
            )
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
        dropout_p=dropout_p,
        scale=scale,
    )


def _supports_evoformer_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
) -> bool:
    r"""
    Return whether q/k/v and mask have the shared Evoformer-kernel contract.
    """

    if query.dim() != 5 or not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if query.shape != key.shape or key.shape != value.shape:
        return False
    if attention_mask is None or attention_mask.dim() <= 3:
        return True
    return attention_mask.dim() == 4 and attention_mask.shape[-2] == 1


def _evoformer_key_mask(
    attention_mask: Tensor,
    query: Tensor,
) -> Tensor:
    r"""
    Broadcast a key-only mask to ``(batch, outer, 1, 1, keys)``.
    """

    if attention_mask.dim() == 4:
        attention_mask = attention_mask.squeeze(-2)
    target_shape = (query.shape[0], query.shape[1], query.shape[-2])
    mask = _broadcast_to_attention_shape(attention_mask.bool(), target_shape, tail_dims=1)
    return mask.reshape(query.shape[0], query.shape[1], 1, 1, query.shape[-2])


def _evoformer_pair_bias(
    query: Tensor,
    key: Tensor,
    biases: Sequence[Tensor] | None,
) -> Tensor | None:
    r"""
    Broadcast pair biases to the layout shared by Apex and DeepSpeed.
    """

    if not biases:
        return None
    target_shape = (query.shape[0], 1, query.shape[-3], query.shape[-2], key.shape[-2])
    broadcast_biases = [_broadcast_to_attention_shape(bias, target_shape, tail_dims=3) for bias in biases]
    return _sum_biases(broadcast_biases)


def _apex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
) -> Tensor | None:
    r"""
    Run Apex OpenFold attention when its scheduler accepts the exact shape.
    """

    if not _supports_evoformer_attention(query, key, value, attention_mask):
        return None
    training = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value, *(biases or ())))
    if not apex_evoformer_attention_supported(query.shape, training=training):
        return None
    bias = _evoformer_pair_bias(query, key, biases)
    if bias is None:
        bias = query.new_zeros(query.shape[0], 1, query.shape[-3], query.shape[-2], key.shape[-2])
    mask = None
    if attention_mask is not None:
        mask = _evoformer_key_mask(attention_mask, query).to(dtype=query.dtype)
    return apex_evoformer_attention(
        query,
        key,
        value,
        mask,
        bias,
        training=training,
    )


def _deepspeed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
) -> Tensor | None:
    r"""
    Run DS4Sci Evoformer attention when its tensor contract is satisfied.
    """

    if not _supports_evoformer_attention(query, key, value, attention_mask):
        return None
    if query.shape[-2] <= 16 or query.shape[-1] > 64:
        return None
    pair_bias = _evoformer_pair_bias(query, key, biases)
    if pair_bias is not None:
        pair_bias = pair_bias.to(dtype=query.dtype)
    mask_bias = None
    if attention_mask is not None:
        mask = _evoformer_key_mask(attention_mask, query).to(dtype=query.dtype)
        mask_bias = (mask - 1) * 1e9
    deepspeed_biases: list[Tensor | None]
    if pair_bias is None:
        deepspeed_biases = [] if mask_bias is None else [mask_bias]
    else:
        deepspeed_biases = [mask_bias, pair_bias]
    output = deepspeed_evoformer_attention(
        query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2), deepspeed_biases
    )
    return output.transpose(-3, -2)


def _oops_attention_pair_bias(
    query: Tensor,
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    projection: MultiheadAttentionProjection,
) -> Tensor | None:
    r"""
    Run the OOps-owned gated pair-bias attention implementation.

    Flattens the leading batch dims, builds the summed pair bias and mask, and delegates gate and output
    composition to OOps dispatch.

    Args:
        query: ``(..., q_tokens, query_size)`` gating input passed to the OOps callable.
        query_projection, key_projection: Head-split q/k of shape
            ``(..., heads, tokens, head_dim)``.
        value_projection: Head-split values of shape
            ``(..., heads, tokens, value_size)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.
        projection: The projection supplying the gate/output weights and ``hidden_size``.

    Returns:
        ``(..., q_tokens, hidden_size)`` output, or ``None`` when there is no pair bias to feed the
        gated OOps callable (the caller then falls back to the ungated OOps path).
    """
    if isinstance(query_projection, NestedTensor):
        if not all(isinstance(tensor, NestedTensor) for tensor in (query, key_projection, value_projection)):
            raise ValueError("nested OOps gated attention requires NestedTensor single and q/k/v")
        pair_bias = _sum_biases(biases or ())
        if pair_bias is None:
            return None
        if not isinstance(pair_bias, NestedTensor):
            raise ValueError("nested OOps gated attention requires NestedTensor pair bias")
        if attention_mask is not None and not isinstance(attention_mask, NestedTensor):
            raise ValueError("nested OOps gated attention requires a NestedTensor mask")
        if pair_bias.dtype != query_projection.dtype:
            pair_bias = pair_bias.to(dtype=query_projection.dtype)
        return cast(
            Tensor,
            _oops_gated_pair_bias_attention(
                single=query,
                query=query_projection,
                key=key_projection,
                value=value_projection,
                mask=attention_mask,
                pair_bias=pair_bias,
                gate_weight=projection.gate_projection.weight,
                output_weight=projection.output_projection.weight,
                scale=1.0,
            ),
        )

    if query_projection.dim() == 5:
        pair_bias = _sum_biases(biases or ())
        if pair_bias is None:
            return None
        if isinstance(pair_bias, NestedTensor):
            raise ValueError("dense OOps gated attention cannot use NestedTensor pair bias")
        if attention_mask is not None and attention_mask.shape[-2] == 1:
            attention_mask = attention_mask.squeeze(-2)
        return cast(
            Tensor,
            _oops_gated_pair_bias_attention(
                single=query,
                query=query_projection,
                key=key_projection,
                value=value_projection,
                mask=attention_mask,
                pair_bias=pair_bias.to(dtype=query_projection.dtype),
                gate_weight=projection.gate_projection.weight,
                output_weight=projection.output_projection.weight,
                scale=1.0,
            ),
        )

    prefix_shape = query_projection.shape[:-3]
    num_heads, query_length, head_dim = query_projection.shape[-3:]
    key_length = key_projection.shape[-2]
    query_flat = query_projection.reshape(-1, num_heads, query_length, head_dim)
    key_flat = key_projection.reshape(-1, num_heads, key_length, head_dim)
    value_flat = value_projection.reshape(-1, num_heads, key_length, head_dim)
    flat_prefix_size = query_flat.shape[0]
    pair_bias = _oops_pair_bias(
        biases,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        num_heads=num_heads,
        query_length=query_length,
        key_length=key_length,
        dtype=query_projection.dtype,
    )
    if pair_bias is None:
        return None
    mask = _oops_attention_mask(
        attention_mask,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        query_length=query_length,
        key_length=key_length,
    )
    query_input = query.reshape(flat_prefix_size, query_length, query.shape[-1])
    output = _oops_gated_pair_bias_attention(
        single=query_input,
        query=query_flat,
        key=key_flat,
        value=value_flat,
        mask=mask,
        pair_bias=pair_bias,
        gate_weight=projection.gate_projection.weight,
        output_weight=projection.output_projection.weight,
        scale=1.0,
    )
    return cast(Tensor, output).reshape(*query.shape[:-2], query_length, projection.hidden_size)


def _oops_attention(
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    scale: float | None = 1.0,
) -> Tensor:
    r"""
    Run the ungated OOps pair-bias attention path.

    Flattens the leading batch dims, builds the summed pair bias and mask, calls the OOps-owned pair-bias
    implementation, and restores the original prefix shape.

    Args:
        query_projection, key_projection: Head-split q/k of shape
            ``(..., heads, tokens, head_dim)``.
        value_projection: Head-split values of shape
            ``(..., heads, tokens, value_size)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.

    Returns:
        ``(..., heads, q_tokens, value_size)`` attention output before the head merge.
    """
    if isinstance(query_projection, NestedTensor):
        if not isinstance(key_projection, NestedTensor) or not isinstance(value_projection, NestedTensor):
            raise ValueError("nested OOps attention requires NestedTensor q/k/v")
        pair_bias = _sum_biases(biases or ())
        if pair_bias is not None:
            if not isinstance(pair_bias, NestedTensor):
                raise ValueError("nested OOps attention requires NestedTensor pair bias")
            if pair_bias.dtype != query_projection.dtype:
                pair_bias = pair_bias.to(dtype=query_projection.dtype)
        if attention_mask is not None and not isinstance(attention_mask, NestedTensor):
            raise ValueError("nested OOps attention requires a NestedTensor mask")
        return cast(
            Tensor,
            _oops_pair_bias_attention(
                query_projection,
                key_projection,
                value_projection,
                mask=attention_mask,
                pair_bias=pair_bias,
                scale=scale,
            ),
        )

    if query_projection.dim() == 5:
        pair_bias = _sum_biases(biases or ())
        if isinstance(pair_bias, NestedTensor):
            raise ValueError("dense OOps attention cannot use NestedTensor pair bias")
        return cast(
            Tensor,
            _oops_pair_bias_attention(
                query_projection,
                key_projection,
                value_projection,
                mask=attention_mask,
                pair_bias=None if pair_bias is None else pair_bias.to(dtype=query_projection.dtype),
                scale=scale,
            ),
        )

    prefix_shape = query_projection.shape[:-3]
    num_heads, query_length, head_dim = query_projection.shape[-3:]
    key_length = key_projection.shape[-2]
    value_size = value_projection.shape[-1]
    query_flat = query_projection.reshape(-1, num_heads, query_length, head_dim)
    key_flat = key_projection.reshape(-1, num_heads, key_length, head_dim)
    value_flat = value_projection.reshape(-1, num_heads, key_length, value_size)
    flat_prefix_size = query_flat.shape[0]
    pair_bias = _oops_pair_bias(
        biases,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        num_heads=num_heads,
        query_length=query_length,
        key_length=key_length,
        dtype=query_projection.dtype,
    )
    mask = _oops_attention_mask(
        attention_mask,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        query_length=query_length,
        key_length=key_length,
    )
    output = _oops_pair_bias_attention(query_flat, key_flat, value_flat, mask=mask, pair_bias=pair_bias, scale=scale)
    return cast(Tensor, output).reshape(*prefix_shape, num_heads, query_length, value_size)


def _oops_pair_bias(
    biases: Sequence[Tensor] | None,
    *,
    prefix_shape: torch.Size,
    flat_prefix_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
) -> Tensor | None:
    r"""
    Broadcast and sum the pair biases into the flat ``(flat_prefix, heads, q, k)`` bias consumed by OOps.

    Args:
        biases: Optional additive pair biases; ``None``/empty yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).
        num_heads, query_length, key_length: Attention head count and q/k token counts.
        dtype: Target dtype for the biases.

    Returns:
        ``(flat_prefix_size, heads, q, k)`` summed pair bias, or ``None`` when there is no bias.
    """
    if not biases:
        return None
    target_shape = (*prefix_shape, num_heads, query_length, key_length)
    broadcast_biases = [
        _broadcast_to_attention_shape(bias.to(dtype=dtype), target_shape, tail_dims=3) for bias in biases
    ]
    pair_bias = _sum_biases(broadcast_biases)
    if pair_bias is None:
        return None
    return pair_bias.reshape(flat_prefix_size, num_heads, query_length, key_length)


def _oops_attention_mask(
    attention_mask: Tensor | None,
    *,
    prefix_shape: torch.Size,
    flat_prefix_size: int,
    query_length: int,
    key_length: int,
) -> Tensor | None:
    r"""
    Broadcast a normalized attention mask to the flat mask consumed by OOps.

    The public entry point has already normalized a mask without a head axis
    to ``(..., q|1, k)``. A singleton explicit head axis is also accepted and
    removed here; masks with distinct head values are excluded before OOps
    dispatch.

    Args:
        attention_mask: Optional boolean mask; ``None`` yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).
        query_length, key_length: Query and key token counts.

    Returns:
        The flattened boolean mask ``(flat_prefix, q, k)`` or the key-only
        form ``(flat_prefix, k)``, or ``None``.

    Raises:
        ValueError: If ``attention_mask`` is not a normalized head-independent mask.
    """
    if attention_mask is None:
        return None
    attention_mask = attention_mask.bool()
    prefix_ndim = len(prefix_shape)
    if attention_mask.dim() == prefix_ndim + 3 and attention_mask.shape[-3] == 1:
        attention_mask = attention_mask.squeeze(-3)
    if attention_mask.dim() == prefix_ndim + 2:
        if attention_mask.shape[-2] == 1:
            target_shape = (*prefix_shape, 1, key_length)
            return _broadcast_to_attention_shape(attention_mask, target_shape, tail_dims=2).reshape(
                flat_prefix_size, key_length
            )
        target_shape = (*prefix_shape, query_length, key_length)
        return _broadcast_to_attention_shape(attention_mask, target_shape, tail_dims=2).reshape(
            flat_prefix_size, query_length, key_length
        )
    raise ValueError(
        "normalized attention_mask must have shape "
        f"(..., {query_length}|1, {key_length}), got {tuple(attention_mask.shape)}"
    )


def _attention_biases(
    query_projection: Tensor,
    key_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
) -> list[Tensor]:
    r"""
    Assemble the additive bias list (pair biases plus a masking bias) for PyTorch SDPA.

    Casts and head-broadcasts each pair bias, then prepends a full large-negative additive mask derived
    from ``attention_mask``.

    Args:
        query_projection, key_projection: Head-split q/k of shape ``(..., heads, tokens, head_dim)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.
    Returns:
        The list of additive biases, with the masking bias (if any) at index 0.
    """
    attention_biases: list[Tensor] = []
    if biases is not None:
        for bias in biases:
            if bias.dim() == query_projection.dim() - 1:
                bias = bias.unsqueeze(-3)
            attention_biases.append(bias.to(dtype=query_projection.dtype))

    if attention_mask is None:
        return attention_biases

    attention_mask = _broadcast_attention_mask(
        query_projection,
        attention_mask,
        key_length=key_projection.shape[-2],
    )

    mask_bias = torch.full(
        (*query_projection.shape[:-1], key_projection.shape[-2]),
        torch.finfo(query_projection.dtype).min,
        dtype=query_projection.dtype,
        device=query_projection.device,
    )
    mask_bias.masked_fill_(attention_mask, 0)
    attention_biases.insert(0, mask_bias)
    return attention_biases


def _sum_biases(biases: Sequence[Tensor]) -> Tensor | None:
    r"""
    Sum a sequence of broadcastable bias tensors into one, or return ``None`` when empty.

    Args:
        biases: The biases to add together.

    Returns:
        The elementwise sum, or ``None`` when ``biases`` is empty.
    """
    if not biases:
        return None
    total = biases[0]
    for bias in biases[1:]:
        total = total + bias
    return total


def _zero_fully_masked_rows(
    output: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    r"""
    Zero pre-projection attention values for queries with no valid key.

    ``output`` has head-major shape ``(..., heads, queries, value_size)`` and
    the public mask has already been normalized. Zeroing here preserves any
    bias subsequently owned by an output projection.
    """

    if attention_mask is None:
        return output
    valid_rows = attention_mask.any(dim=-1, keepdim=True)
    if attention_mask.dim() == output.dim() - 1:
        valid_rows = valid_rows.unsqueeze(-3)
    elif attention_mask.dim() != output.dim():
        raise ValueError(
            f"normalized attention_mask with shape {tuple(attention_mask.shape)} does not match "
            f"attention output with shape {tuple(output.shape)}"
        )
    return output.masked_fill(~valid_rows, 0)


def _attention_mask_view(reference: Tensor, attention_mask: Tensor) -> Tensor:
    r"""
    Insert the omitted head axis without materializing broadcast attention weights.

    ``reference`` has head-major shape ``(..., heads, queries, channels)`` and
    ``attention_mask`` has already been normalized.
    """

    if attention_mask.dim() == reference.dim():
        return attention_mask
    if attention_mask.dim() == reference.dim() - 1:
        return attention_mask.unsqueeze(-3)
    raise ValueError(
        f"normalized attention_mask with shape {tuple(attention_mask.shape)} does not match "
        f"attention reference with shape {tuple(reference.shape)}"
    )


def _broadcast_attention_mask(
    reference: Tensor,
    attention_mask: Tensor,
    *,
    key_length: int,
) -> Tensor:
    r"""
    Expand a normalized keep-mask to head-major attention weights.

    ``reference`` has shape ``(..., heads, queries, channels)``. Normalization
    has already made prefix axes explicit, so a full-rank mask includes the
    head axis and a mask with one fewer axis omits it.
    """

    prefix_shape = reference.shape[:-3]
    num_heads, query_length = reference.shape[-3:-1]
    target_shape = (*prefix_shape, num_heads, query_length, key_length)
    mask = attention_mask.bool()

    if mask.dim() == reference.dim():
        try:
            return torch.broadcast_to(mask, target_shape)
        except RuntimeError as error:
            raise ValueError(
                f"attention_mask with shape {tuple(attention_mask.shape)} is not broadcastable "
                f"to attention weights with shape {target_shape}"
            ) from error

    if mask.dim() == reference.dim() - 1:
        target_without_heads = (*prefix_shape, query_length, key_length)
        try:
            mask = torch.broadcast_to(mask, target_without_heads)
        except RuntimeError as error:
            raise ValueError(
                f"attention_mask with shape {tuple(attention_mask.shape)} is not broadcastable "
                f"to attention weights with shape {target_shape}"
            ) from error
        return torch.broadcast_to(mask.unsqueeze(-3), target_shape)

    raise ValueError(
        f"normalized attention_mask with shape {tuple(attention_mask.shape)} does not match "
        f"attention weights with shape {target_shape}"
    )


def _broadcast_to_attention_shape(tensor: Tensor, target_shape: tuple[int, ...], *, tail_dims: int) -> Tensor:
    r"""
    Broadcast ``tensor`` to ``target_shape``, matching the last ``tail_dims`` axes exactly.

    The trailing ``tail_dims`` axes are the attention payload. Missing prefix axes are inserted
    immediately before them and ordinary PyTorch broadcasting performs the expansion.

    Args:
        tensor: The tensor to broadcast.
        target_shape: The desired output shape.
        tail_dims: The number of trailing axes that must line up with ``target_shape``.

    Returns:
        A view of ``tensor`` broadcast to ``target_shape``.

    """
    if tuple(tensor.shape) == target_shape:
        return tensor
    source_prefix = tensor.shape[:-tail_dims]
    source_tail = tensor.shape[-tail_dims:]
    view_shape = (*source_prefix, *((1,) * (len(target_shape) - tensor.dim())), *source_tail)
    return torch.broadcast_to(tensor.reshape(view_shape), target_shape)
