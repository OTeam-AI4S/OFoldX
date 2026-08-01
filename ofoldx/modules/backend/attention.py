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
Backend-dispatched attention operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ..._oops import OOpsUnsupportedError, oops_kernel
from ._apex import apex_evoformer_attention, apex_evoformer_attention_supported
from ._cueq import cueq_triangle_attention
from ._deepspeed import deepspeed_evoformer_attention
from .resolve import DEEPSPEED_AVAILABLE, resolve_tensor_backend

_oops_gated_pair_bias_attention = oops_kernel("gated_pair_bias_attention")
_oops_pair_bias_attention = oops_kernel("pair_bias_attention")
_oops_triangle_attention = oops_kernel("triangle_attention")

if TYPE_CHECKING:
    from ..attention.multihead import GatedMultiheadAttentionProjection, MultiheadAttentionProjection


def multihead_attention(
    query: Tensor,
    key: Tensor | None,
    value: Tensor | None,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    projection: MultiheadAttentionProjection,
    backend: str,
    dropout_p: float = 0.0,
) -> Tensor:
    r"""
    Project q/k/v via ``projection``, run attention on ``backend``, and merge heads.

    Dispatches attention to the selected provider when that provider implements
    the concrete tensor contract, otherwise to PyTorch SDPA. ``biases`` are
    summed into one additive pair bias. ``attention_mask`` follows the common
    pair-first mask contract and is normalized before provider selection. On
    OOps, a [`GatedMultiheadAttentionProjection`][ofoldx.modules.attention.multihead.GatedMultiheadAttentionProjection]
    with bias-free gate and output projections and a pair bias runs the fused
    gated pair-bias kernel, which computes its merge.

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
        projection: The head projection owned by the calling module.
        backend: Operator provider; see [ofoldx.modules.backend][].
        dropout_p: Dropout probability applied to attention weights.

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
            raise ValueError("nested multi-head attention requires NestedTensor query/key/value")
        if attention_mask is not None and not isinstance(attention_mask, NestedTensor):
            raise ValueError("nested multi-head attention requires a NestedTensor mask")
        attention_mask = _normalize_nested_mask(
            attention_mask,
            prefix_rank=query.dim() - 2,
            num_heads=projection.num_attention_heads,
            num_queries=query.shape[-2],
            num_keys=key.shape[-2],
        )
        supported = ("sdpa", "eager") if dropout_p else ("oops", "sdpa", "eager")
    else:
        attention_mask = _normalize_attention_mask(
            attention_mask,
            prefix_shape=query.shape[:-2],
            num_attention_heads=projection.num_attention_heads,
            num_queries=query.shape[-2],
            num_keys=key.shape[-2],
        )
        oops_mask_supported = (
            attention_mask is None or attention_mask.dim() != query.dim() + 1 or attention_mask.shape[-3] == 1
        )
        if dropout_p:
            supported = ("sdpa", "eager")
        else:
            supported = (
                ("oops", "apex", "deepspeed", "sdpa", "eager")
                if oops_mask_supported
                else ("apex", "deepspeed", "sdpa", "eager")
            )
    # The projection module imports this one, so its class is imported where the OOps operator is chosen.
    from ..attention.multihead import GatedMultiheadAttentionProjection

    gated = (
        bool(biases)
        and isinstance(projection, GatedMultiheadAttentionProjection)
        and projection.output_projection.bias is None
        and projection.gate_projection.bias is None
    )
    automatic = backend == "auto"
    backend = resolve_tensor_backend(
        backend,
        tensors=(query, key, value, attention_mask, *(biases or ())),
        supported=supported,
        public_default="sdpa",
        oops=_oops_gated_pair_bias_attention if gated else _oops_pair_bias_attention,
    )
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
            dropout_p=dropout_p,
        )
    elif backend == "oops":
        try:
            if gated:
                gated_output = _oops_attention_pair_bias(
                    query,
                    query_projection,
                    key_projection,
                    value_projection,
                    biases=biases,
                    attention_mask=attention_mask,
                    projection=cast(GatedMultiheadAttentionProjection, projection),
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
        except OOpsUnsupportedError:
            if not automatic:
                raise
            output = _sdpa_attention(
                query_projection,
                key_projection,
                value_projection,
                biases=biases,
                attention_mask=attention_mask,
                scale=1.0,
            )
    elif backend in {"apex", "deepspeed"}:
        provider_output = None
        if backend == "apex":
            provider_output = _apex_attention(
                query_projection, key_projection, value_projection, biases=biases, attention_mask=attention_mask
            )
        if provider_output is None and (backend == "deepspeed" or (automatic and DEEPSPEED_AVAILABLE)):
            provider_output = _deepspeed_attention(
                query_projection, key_projection, value_projection, biases=biases, attention_mask=attention_mask
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
            if provider_output is None
            else provider_output
        )
    else:
        output = _sdpa_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
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
            ``(..., num_attention_heads, num_query_tokens, head_size)``.
        key: Key tensor with shape
            ``(..., num_attention_heads, num_key_tokens, head_size)``.
        value: Value tensor with shape
            ``(..., num_attention_heads, num_key_tokens, value_size)``.
        biases: Optional additive pair biases broadcastable to
            ``(..., num_attention_heads, num_query_tokens, num_key_tokens)``.
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
            ``(..., num_attention_heads, num_query_tokens, value_size)``.
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
        attention_mask = _normalize_nested_mask(
            attention_mask,
            prefix_rank=query.dim() - 3,
            num_heads=query.shape[-3],
            num_queries=query.shape[-2],
            num_keys=key.shape[-2],
        )
    else:
        attention_mask = _normalize_attention_mask(
            attention_mask,
            prefix_shape=query.shape[:-3],
            num_attention_heads=query.shape[-3],
            num_queries=query.shape[-2],
            num_keys=key.shape[-2],
        )
    oops_supported = (
        dropout_p == 0.0
        and value.shape[-1] == query.shape[-1]
        and (attention_mask is None or attention_mask.dim() != query.dim() or attention_mask.shape[-3] == 1)
    )
    supported = ("oops", "sdpa", "eager") if oops_supported else ("sdpa", "eager")
    resolved = resolve_tensor_backend(
        backend,
        tensors=(query, key, value, attention_mask, *(biases or ())),
        supported=supported,
        public_default="sdpa",
        oops=_oops_pair_bias_attention,
    )
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
        try:
            output = _oops_attention(
                query,
                key,
                value,
                biases=biases,
                attention_mask=attention_mask,
                scale=scale,
            )
        except OOpsUnsupportedError:
            if backend != "auto":
                raise
            output = _sdpa_attention(
                query,
                key,
                value,
                biases=biases,
                attention_mask=attention_mask,
                dropout_p=dropout_p,
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
    ``oops``, the dedicated OOps-owned ``triangle_attention`` implementation computes the attention
    from the ``(..., rows, cols, heads, head_size)`` q/k/v layout and the row-broadcast
    ``(..., rows|1, heads, cols, cols)`` bias directly instead of flattening the row axis into a
    generic pair-bias attention, and ``projection`` merges its heads. ``cueq`` on CUDA likewise uses
    the cuEquivariance kernel. Every other backend folds the key mask into an additive bias and
    delegates to [`multihead_attention`][], equivalent to attending on ``pair`` with
    ``biases=[mask_bias, triangle_bias]``.

    Args:
        pair: Layer-normed pair representation ``(..., rows, cols, pair_size)``.
        triangle_bias: Per-head triangle bias ``(..., 1, heads, cols, cols)`` (row axis broadcast).
        attention_mask: Optional boolean keep-mask ``(..., rows, cols)``; ``False`` keys are excluded.
        projection: The head projection owned by the calling triangle-attention module.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        Pair update ``(..., rows, cols, pair_size)``.
    """

    resolved = resolve_tensor_backend(
        backend,
        tensors=(pair, triangle_bias, attention_mask),
        supported=(
            ("oops", "apex", "deepspeed", "sdpa", "eager")
            if isinstance(pair, NestedTensor)
            else ("oops", "cueq", "apex", "deepspeed", "sdpa", "eager")
        ),
        public_default="sdpa",
        oops=_oops_triangle_attention,
    )
    if resolved == "oops":
        return _oops_triangle_flash(pair, triangle_bias, attention_mask, projection=projection)
    if resolved == "cueq" and pair.is_cuda:
        return _cueq_triangle_flash(pair, triangle_bias, attention_mask, projection=projection)
    key_mask = None if attention_mask is None else attention_mask.unsqueeze(-2)
    return multihead_attention(
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
    Run the OOps-owned ``triangle_attention`` implementation and merge its heads.

    The OOps callable takes q/k/v in the ``project_heads`` layout ``(..., rows, cols, heads,
    head_size)`` (no head-front transpose), the boolean ``(..., rows, cols)`` keep-mask, and the
    row-broadcast ``(..., rows|1, heads, cols, cols)`` bias. The ``1/sqrt(head_size)`` scaling is
    passed directly to OOps dispatch, so the projections are left unscaled. The kernel computes the
    attention alone, so ``projection.merge`` applies any gate and the output projection, biases
    included.
    """

    query, key, value = projection.project_heads(pair, pair, pair)
    mask = attention_mask.bool() if attention_mask is not None else None
    bias = triangle_bias.to(dtype=pair.dtype)
    if not isinstance(pair, NestedTensor):
        prefix = pair.shape[:-3]
        rows, columns = pair.shape[-3:-1]
        heads, channels = projection.num_attention_heads, projection.head_size
        query = query.reshape(-1, rows, columns, heads, channels)
        key = key.reshape(-1, rows, columns, heads, channels)
        value = value.reshape(-1, rows, columns, heads, channels)
        bias = _broadcast_to_attention_shape(bias, (*prefix, 1, heads, columns, columns), tail_dims=4)
        bias = bias.reshape(-1, 1, heads, columns, columns)
        if mask is not None:
            mask = _broadcast_to_attention_shape(mask, (*prefix, rows, columns), tail_dims=2)
            mask = mask.reshape(-1, rows, columns)
    output = cast(Tensor, _oops_triangle_attention(query, key, value, mask, bias, scale=projection.head_size**-0.5))
    if not isinstance(pair, NestedTensor):
        output = output.reshape(*pair.shape[:-3], *output.shape[-4:])
    return projection.merge(output, pair)


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
    num_rows, num_columns = pair.shape[-3:-1]
    num_attention_heads, head_size = projection.num_attention_heads, projection.head_size
    query, key, value = projection.project_unscaled(pair, pair, pair)
    query = query.reshape(-1, num_rows, num_attention_heads, num_columns, head_size)
    key = key.reshape(-1, num_rows, num_attention_heads, num_columns, head_size)
    value = value.reshape(-1, num_rows, num_attention_heads, num_columns, head_size)
    target_bias_shape = (*prefix_shape, 1, num_attention_heads, num_columns, num_columns)
    bias = _broadcast_to_attention_shape(triangle_bias, target_bias_shape, tail_dims=4)
    mask = None
    if attention_mask is not None:
        target_mask_shape = (*prefix_shape, num_rows, num_columns)
        mask = _broadcast_to_attention_shape(attention_mask.bool(), target_mask_shape, tail_dims=2)
        mask = mask.reshape(-1, num_rows, 1, 1, num_columns)
    output = cueq_triangle_attention(
        query,
        key,
        value,
        bias.reshape(-1, 1, num_attention_heads, num_columns, num_columns),
        mask,
        scale=head_size**-0.5,
    )
    if mask is not None:
        output = output.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
    output = output.reshape(*prefix_shape, num_rows, num_attention_heads, num_columns, head_size)
    output = output.transpose(-2, -3)
    return projection.merge(output, pair)


def _normalize_attention_mask(
    attention_mask: Tensor | None,
    *,
    prefix_shape: torch.Size,
    num_attention_heads: int,
    num_queries: int,
    num_keys: int,
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

    if mask.dim() >= 2 and mask.shape[-2] in {1, num_queries} and mask.shape[-1] in {1, num_keys}:
        query_payload = 1 if mask.shape[-2] == 1 else num_queries
        if mask.dim() - 2 <= len(prefix_shape):
            normalized = _try_broadcast_to_attention_shape(
                mask,
                (*prefix_shape, query_payload, num_keys),
                tail_dims=2,
            )
            if normalized is not None:
                return normalized

        if mask.dim() == len(prefix_shape) + 3 and mask.shape[-3] in {1, num_attention_heads}:
            head_payload = 1 if mask.shape[-3] == 1 else num_attention_heads
            normalized = _try_broadcast_to_attention_shape(
                mask,
                (*prefix_shape, head_payload, query_payload, num_keys),
                tail_dims=3,
            )
            if normalized is not None:
                return normalized

    if (
        mask.dim() == len(prefix_shape) + 2
        and mask.shape[-2] in {1, num_attention_heads}
        and mask.shape[-1] in {1, num_keys}
    ):
        head_payload = 1 if mask.shape[-2] == 1 else num_attention_heads
        normalized = _try_broadcast_to_attention_shape(
            mask,
            (*prefix_shape, head_payload, num_keys),
            tail_dims=2,
        )
        if normalized is not None:
            return normalized.unsqueeze(-2)

    if mask.dim() >= 1 and mask.shape[-1] in {1, num_keys} and mask.dim() - 1 <= len(prefix_shape):
        normalized = _try_broadcast_to_attention_shape(
            mask,
            (*prefix_shape, num_keys),
            tail_dims=1,
        )
        if normalized is not None:
            return normalized.unsqueeze(-2)

    raise ValueError(
        "attention_mask must be a pair-first boolean mask with trailing "
        f"({num_queries}, {num_keys}), (..., 1, {num_keys}), "
        f"(..., {num_attention_heads}, {num_keys}), or ({num_keys},); "
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


def _size_in(size: int, *candidates: int) -> bool:
    r"""Compare an axis size by equality; hashing a symbolic size would specialize it under compilation."""

    return any(size == candidate for candidate in candidates)


def _normalize_nested_mask(
    mask: Tensor | None,
    *,
    prefix_rank: int,
    num_heads: int,
    num_queries: int,
    num_keys: int,
) -> Tensor | None:
    """Declare key, query and optional head axes without padding ragged masks."""
    if mask is None:
        return None
    if not _size_in(mask.shape[-1], 1, num_keys):
        raise ValueError("nested attention mask key axis must match key tokens")
    if mask.dim() == prefix_rank + 1:
        return mask.unsqueeze(-2)
    if mask.dim() == prefix_rank + 2:
        if _size_in(mask.shape[-2], 1, num_queries):
            return mask
        if mask.shape[-2] == num_heads:
            return mask.unsqueeze(-2)
    if (
        mask.dim() == prefix_rank + 3
        and _size_in(mask.shape[-3], 1, num_heads)
        and _size_in(mask.shape[-2], 1, num_queries)
    ):
        return mask
    raise ValueError("nested attention mask must declare key validity or pair-first query/key axes")


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
    Pure PyTorch attention with math-backend accumulation precision.

    Half and bfloat16 inputs accumulate in float32; float64 remains float64.
    Masked keys are excluded after additive biases are applied. Nested matrix
    products declare the query and key ragged axes of their actual results;
    softmax and dropout operate on the resulting packed attention weights.
    """

    head_size = query_projection.shape[-1]
    scale = head_size**-0.5 if scale is None else scale
    dtype = torch.float32 if query_projection.dtype in (torch.float16, torch.bfloat16) else query_projection.dtype
    with torch.autocast(device_type=query_projection.device.type, enabled=False):
        logits: Tensor
        if isinstance(query_projection, NestedTensor):
            if not isinstance(key_projection, NestedTensor) or not isinstance(value_projection, NestedTensor):
                raise ValueError("nested eager attention requires NestedTensor q/k/v")
            nested_query = query_projection
            nested_key = key_projection
            nested_value = value_projection
            query_projection = nested_query
            key_projection = nested_key
            value_projection = nested_value
            query_rank = nested_query.dim() - 1
            key_rank = nested_key.dim() - 1
            pair_rank = max(query_rank, key_rank)
            pair_ragged_dims = {
                axis + pair_rank - query_rank for axis in nested_query.ragged_dims if axis < query_rank - 1
            }
            pair_ragged_dims.update(
                axis + pair_rank - key_rank for axis in nested_key.ragged_dims if axis < key_rank - 2
            )
            if key_rank - 2 in nested_key.ragged_dims:
                pair_ragged_dims.add(pair_rank - 1)
            logits = NestedTensor(
                [
                    query_part.to(dtype) @ key_part.to(dtype).transpose(-1, -2)
                    for query_part, key_part in zip(
                        nested_query.unbind(0 if nested_query.batch_first else 1),
                        nested_key.unbind(0 if nested_key.batch_first else 1),
                        strict=True,
                    )
                ],
                ragged_dims=tuple(sorted(pair_ragged_dims)),
                batch_first=nested_query.batch_first,
            )
        else:
            logits = query_projection.to(dtype) @ key_projection.to(dtype).transpose(-1, -2)
        logits = logits * scale
        if biases is not None:
            for bias in biases:
                logits = logits + bias.to(dtype=dtype)
        if attention_mask is not None:
            if isinstance(query_projection, NestedTensor):
                mask = attention_mask.unsqueeze(-3) if attention_mask.dim() == logits.dim() - 1 else attention_mask
            else:
                mask = _broadcast_attention_mask(
                    query_projection,
                    attention_mask,
                    num_keys=key_projection.shape[-2],
                )
            logits = logits.masked_fill(~mask, float("-inf"))
            logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
        attention = logits.softmax(dim=-1)
        if dropout_p:
            attention = F.dropout(attention, p=dropout_p, training=True)
        if isinstance(query_projection, NestedTensor) and isinstance(attention, NestedTensor):
            output_ragged_dims = tuple(axis for axis in attention.ragged_dims if axis != attention.dim() - 2)
            return NestedTensor(
                [
                    (attention_part @ value_part.to(dtype)).to(value_projection.dtype)
                    for attention_part, value_part in zip(
                        attention.unbind(0 if attention.batch_first else 1),
                        value_projection.unbind(0 if cast(NestedTensor, value_projection).batch_first else 1),
                        strict=True,
                    )
                ],
                ragged_dims=output_ragged_dims,
                batch_first=query_projection.batch_first,
                padding_value=query_projection.padding_value,
                mask_value=query_projection.mask_value,
            )
        return (attention @ value_projection.to(dtype)).to(value_projection.dtype)


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
    Run PyTorch scaled-dot-product attention, retaining packed sequence boundaries.
    """

    if isinstance(query, NestedTensor):
        if not isinstance(key, NestedTensor) or not isinstance(value, NestedTensor):
            raise ValueError("nested SDPA attention requires NestedTensor q/k/v")
        native_packed = (
            query.is_cuda
            and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and not biases
            and attention_mask is None
            and all(
                tensor.dim() == 4
                and tensor.batch_first
                and tensor.ragged_dims == (1,)
                and tensor.packed_dim_order == (1, 0, 2)
                for tensor in (query, key, value)
            )
            and query.shape[-3] == key.shape[-3] == value.shape[-3]
            and query.shape[-1] == key.shape[-1] == value.shape[-1]
        )
        if native_packed:
            # Flat kernels cannot express this per-sequence K/V invariant with
            # ordinary tensor shapes; retain the boundary correspondence explicitly.
            torch._assert_async(
                (key.packed_offsets() == value.packed_offsets()).all(),
                "packed attention requires matching key/value sequence boundaries",
            )
            query_offsets = query.packed_offsets(device=query.device, dtype=torch.int32)
            key_offsets = key.packed_offsets(device=key.device, dtype=torch.int32)
            with torch.autocast(device_type=query.device.type, enabled=False):
                output = torch.ops.aten._efficient_attention_forward(
                    query.concat.unsqueeze(0),
                    key.concat.unsqueeze(0),
                    value.concat.unsqueeze(0),
                    None,
                    query_offsets,
                    key_offsets,
                    query.shape[-2],
                    key.shape[-2],
                    dropout_p,
                    0,
                    query.requires_grad or key.requires_grad or value.requires_grad,
                    scale=scale,
                )[0]
            return query.packed_like(output.squeeze(0))
        return _eager_attention(
            query,
            key,
            value,
            biases=biases,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
        )

    attention_bias: Tensor | None
    if not biases and attention_mask is not None:
        attention_bias = _attention_mask_view(query, attention_mask)
    else:
        dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
        attention_bias = _sum_biases([bias.to(dtype=dtype) for bias in biases or ()])
        if attention_mask is not None:
            mask = _broadcast_attention_mask(query, attention_mask, num_keys=key.shape[-2])
            attention_bias = cast(Tensor, attention_bias).masked_fill(~mask, float("-inf"))
    mixed_cuda_bias = (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and attention_bias is not None
        and attention_bias.dtype == torch.float32
    )
    if mixed_cuda_bias and not torch.compiler.is_compiling():
        try:
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_bias, dropout_p=dropout_p, scale=scale
            )
        except RuntimeError as error:
            if dropout_p or "invalid dtype for bias - should match query's dtype" not in str(error):
                raise
        # Keep the ordinary native/autocast path when it succeeds. Retry only this mixed-dtype CUDA failure with
        # zero dropout: a failed dropout call may already have advanced the device's random stream.
        with torch.autocast(device_type="cuda", enabled=False):
            return F.scaled_dot_product_attention(
                query.float(),
                key.float(),
                value.float(),
                attn_mask=cast(Tensor, attention_bias).contiguous(),
                dropout_p=dropout_p,
                scale=scale,
            ).to(dtype=value.dtype)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=dropout_p, scale=scale)


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
    projection: GatedMultiheadAttentionProjection,
) -> Tensor | None:
    r"""
    Run the OOps-owned gated pair-bias attention implementation.

    Flattens the leading batch dims, builds the summed pair bias and mask, and delegates gate and output
    composition to OOps dispatch.

    Args:
        query: ``(..., q_tokens, query_size)`` gating input passed to the OOps callable.
        query_projection, key_projection: Head-split q/k of shape
            ``(..., heads, tokens, head_size)``.
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
        return cast(
            Tensor,
            _oops_gated_pair_bias_attention(
                single=query,
                query=query_projection,
                key=key_projection,
                value=value_projection,
                mask=_oops_dense_attention_mask(attention_mask, prefix_rank=2),
                pair_bias=pair_bias.to(dtype=query_projection.dtype),
                gate_weight=projection.gate_projection.weight,
                output_weight=projection.output_projection.weight,
                scale=1.0,
            ),
        )

    prefix_shape = query_projection.shape[:-3]
    num_attention_heads, num_queries, head_size = query_projection.shape[-3:]
    num_keys = key_projection.shape[-2]
    query_flat = query_projection.reshape(-1, num_attention_heads, num_queries, head_size)
    key_flat = key_projection.reshape(-1, num_attention_heads, num_keys, head_size)
    value_flat = value_projection.reshape(-1, num_attention_heads, num_keys, head_size)
    flat_prefix_size = query_flat.shape[0]
    pair_bias = _oops_pair_bias(
        biases,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        num_attention_heads=num_attention_heads,
        num_queries=num_queries,
        num_keys=num_keys,
        dtype=query_projection.dtype,
    )
    if pair_bias is None:
        return None
    mask = _oops_attention_mask(attention_mask, prefix_shape=prefix_shape, flat_prefix_size=flat_prefix_size)
    query_input = query.reshape(flat_prefix_size, num_queries, query.shape[-1])
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
    return cast(Tensor, output).reshape(*query.shape[:-2], num_queries, projection.hidden_size)


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
            ``(..., heads, tokens, head_size)``.
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
                mask=_oops_dense_attention_mask(attention_mask, prefix_rank=2),
                pair_bias=None if pair_bias is None else pair_bias.to(dtype=query_projection.dtype),
                scale=scale,
            ),
        )

    prefix_shape = query_projection.shape[:-3]
    num_attention_heads, num_queries, head_size = query_projection.shape[-3:]
    num_keys = key_projection.shape[-2]
    value_size = value_projection.shape[-1]
    query_flat = query_projection.reshape(-1, num_attention_heads, num_queries, head_size)
    key_flat = key_projection.reshape(-1, num_attention_heads, num_keys, head_size)
    value_flat = value_projection.reshape(-1, num_attention_heads, num_keys, value_size)
    flat_prefix_size = query_flat.shape[0]
    pair_bias = _oops_pair_bias(
        biases,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        num_attention_heads=num_attention_heads,
        num_queries=num_queries,
        num_keys=num_keys,
        dtype=query_projection.dtype,
    )
    mask = _oops_attention_mask(attention_mask, prefix_shape=prefix_shape, flat_prefix_size=flat_prefix_size)
    output = _oops_pair_bias_attention(query_flat, key_flat, value_flat, mask=mask, pair_bias=pair_bias, scale=scale)
    return cast(Tensor, output).reshape(*prefix_shape, num_attention_heads, num_queries, value_size)


def _oops_pair_bias(
    biases: Sequence[Tensor] | None,
    *,
    prefix_shape: torch.Size,
    flat_prefix_size: int,
    num_attention_heads: int,
    num_queries: int,
    num_keys: int,
    dtype: torch.dtype,
) -> Tensor | None:
    r"""
    Broadcast and sum the pair biases into the flat ``(flat_prefix, heads, q, k)`` bias consumed by OOps.

    Args:
        biases: Optional additive pair biases; ``None``/empty yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).
        num_attention_heads, query_length, key_length: Attention head count and q/k token counts.
        dtype: Target dtype for the biases.

    Returns:
        ``(flat_prefix_size, heads, q, k)`` summed pair bias, or ``None`` when there is no bias.
    """
    if not biases:
        return None
    target_shape = (*prefix_shape, num_attention_heads, num_queries, num_keys)
    broadcast_biases = [torch.broadcast_to(bias.to(dtype=dtype), target_shape) for bias in biases]
    pair_bias = _sum_biases(broadcast_biases)
    if pair_bias is None:
        return None
    return pair_bias.reshape(flat_prefix_size, num_attention_heads, num_queries, num_keys)


def _oops_attention_mask(
    attention_mask: Tensor | None,
    *,
    prefix_shape: torch.Size,
    flat_prefix_size: int,
) -> Tensor | None:
    r"""
    Flatten the batch prefix of a normalized attention mask for the flattened OOps call.

    The public entry point has already normalized the mask to ``(..., q|1, k)``, or to
    ``(..., 1, q|1, k)`` when the caller gave an explicit head axis; masks with distinct
    head values are excluded before OOps dispatch. Shared head axes are removed,
    and a unit query axis becomes key-only validity before the prefix is flattened.

    Args:
        attention_mask: Optional normalized boolean mask; ``None`` yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).

    Returns:
        The mask with ``prefix_shape`` flattened to ``flat_prefix_size``, or ``None``.
    """
    if attention_mask is None:
        return None
    mask = _oops_dense_attention_mask(attention_mask, prefix_rank=len(prefix_shape))
    assert mask is not None
    return mask.reshape(flat_prefix_size, *mask.shape[len(prefix_shape) :])


def _oops_dense_attention_mask(attention_mask: Tensor | None, *, prefix_rank: int) -> Tensor | None:
    """Lower normalized shared-head/key masks to the public OOps calling convention."""
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.dim() == prefix_rank + 3:
        if mask.shape[-3] != 1:
            raise ValueError("OOps attention does not admit distinct head masks")
        mask = mask.squeeze(-3)
    if mask.shape[-2] == 1:
        mask = mask.squeeze(-2)
    return mask


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
    output = output.masked_fill(~valid_rows, 0)
    return output


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
    num_keys: int,
) -> Tensor:
    r"""
    Expand a normalized keep-mask to head-major attention weights.

    ``reference`` has shape ``(..., heads, queries, channels)``. Normalization
    has already made prefix axes explicit, so a full-rank mask includes the
    head axis and a mask with one fewer axis omits it.
    """

    prefix_shape = reference.shape[:-3]
    num_attention_heads, num_queries = reference.shape[-3:-1]
    target_shape = (*prefix_shape, num_attention_heads, num_queries, num_keys)
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
        target_without_heads = (*prefix_shape, num_queries, num_keys)
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
