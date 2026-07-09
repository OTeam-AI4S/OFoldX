r"""
Backend dispatch for gated multi-head (pair-bias) attention.

The dispatch owns every backend's calling convention for attention in one place: OOps fused
pair-bias kernels (with an optional gated fast path), DeepSpeed ``DS4Sci_EvoformerAttention``, and the
pure-PyTorch ``scaled_dot_product_attention`` reference. [`GatedMultiheadAttention`][] stays
backend-agnostic — it owns the q/k/v projection and passes it in. New backends (cuEquivariance /
NVIDIA) add a branch in [`gated_multihead_attention`][] plus any helper they need.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import OOpsUnsupportedError, oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_gated_pair_bias_attention = oops_kernel("gated_pair_bias_attention")
_oops_pair_bias_attention = oops_kernel("pair_bias_attention")

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
    operator_backend: str,
) -> Tensor:
    r"""
    Project q/k/v via ``projection``, run attention on ``operator_backend``, and merge heads.

    Dispatches the attention math to the resolved backend: the OOps fused pair-bias kernels (with a gated
    fast path when ``projection.gating``), DeepSpeed ``DS4Sci_EvoformerAttention``, or the pure-PyTorch
    ``scaled_dot_product_attention`` reference. ``biases`` are summed into a single additive pair bias and
    ``attention_mask`` becomes a large-negative additive mask; both are broadcast to the attention shape.

    Args:
        query: ``(..., q_tokens, query_size)`` query input.
        key: ``(..., k_tokens, key_size)`` key input; defaults to ``query`` when ``None``.
        value: ``(..., k_tokens, value_size)`` value input; defaults to ``key`` when ``None``.
        biases: Optional additive pair biases, each broadcastable to ``(..., heads, q_tokens, k_tokens)``.
        attention_mask: Optional boolean mask broadcastable to ``(..., q_tokens, k_tokens)`` or
            ``(..., k_tokens)``; ``False`` positions are masked out.
        projection: The q/k/v (and optional gate/output) projection owned by the calling module.
    operator_backend: Already-resolved concrete backend, one of ``"oops"`` / ``"deepspeed"`` /
            ``"sdpa"`` / ``"eager"``.

    Returns:
        ``(..., q_tokens, hidden_size)`` attention output with heads merged.

    Raises:
        OOpsUnsupportedError: If ``operator_backend`` is ``"oops"`` on non-CUDA tensors.
        RuntimeError: If ``operator_backend`` is ``"deepspeed"`` but deepspeed4science is unavailable.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> from ofoldx.modules.attention.multihead import MultiheadAttentionProjection
        >>> from ofoldx.modules.backend.attention import gated_multihead_attention
        >>> projection = MultiheadAttentionProjection(hidden_size=8, num_attention_heads=2)
        >>> hidden = torch.randn(1, 4, 8)
        >>> out = gated_multihead_attention(
        ...     hidden, None, None,
        ...     biases=None, attention_mask=None,
        ...     projection=projection, operator_backend="sdpa",
        ... )
        >>> out.shape
        torch.Size([1, 4, 8])
    """

    key = query if key is None else key
    value = key if value is None else value
    operator_backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(query, key, value, attention_mask, *(biases or ())),
        supported=("oops", "sdpa", "deepspeed", "eager"),
        public_default="sdpa",
    )
    if operator_backend in {"deepspeed", "eager"}:
        query_projection, key_projection, value_projection = projection.project_unscaled(query, key, value)
    else:
        query_projection, key_projection, value_projection = projection.project(query, key, value)
    if operator_backend == "eager":
        output = _eager_attention(
            query_projection,
            key_projection,
            value_projection,
            biases=biases,
            attention_mask=attention_mask,
        )
    elif operator_backend == "deepspeed":
        attention_biases = _attention_biases(
            query_projection,
            key_projection,
            biases=biases,
            attention_mask=attention_mask,
            operator_backend=operator_backend,
        )
        output = _deepspeed_attention(query_projection, key_projection, value_projection, attention_biases)
    elif operator_backend == "oops":
        if not query_projection.is_cuda:
            raise OOpsUnsupportedError(
                "GatedMultiheadAttention backend 'oops' requires CUDA tensors; use operator_backend='sdpa'"
            )
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
    else:
        attention_biases = _attention_biases(
            query_projection,
            key_projection,
            biases=biases,
            attention_mask=attention_mask,
            operator_backend=operator_backend,
        )
        output = F.scaled_dot_product_attention(
            query_projection,
            key_projection,
            value_projection,
            attn_mask=_sum_biases(attention_biases),
            dropout_p=0.0,
            scale=1.0,
        )
    output = output.transpose(-2, -3)
    return projection.merge(output, query)


def _eager_attention(
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    inf: float = 1e6,
) -> Tensor:
    r"""
    Pure PyTorch attention path matching Boltz's reference ``AttentionPairBias`` numerics.

    Unlike SDPA, this path explicitly performs the attention logits and softmax in float32 and uses the
    same ``-1e6`` additive mask convention as the Boltz implementation. It is selected only by an explicit
    ``operator_backend="eager"`` request.
    """

    head_dim = query_projection.shape[-1]
    logits = torch.einsum("...hid,...hjd->...hij", query_projection.float(), key_projection.float())
    logits = logits / (head_dim**0.5)
    if biases is not None:
        for bias in biases:
            if bias.dim() == logits.dim() - 1:
                bias = bias.unsqueeze(-3)
            logits = logits + bias.to(device=logits.device, dtype=torch.float32)
    if attention_mask is not None:
        mask = attention_mask.to(device=logits.device).bool()
        if mask.dim() == logits.dim() - 1:
            mask = mask.unsqueeze(-3)
        logits = logits + (~mask).to(dtype=torch.float32) * -inf
    attention = logits.softmax(dim=-1)
    return torch.einsum("...hij,...hjd->...hid", attention, value_projection.float()).to(value_projection.dtype)


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
    Run the fused OOps gated pair-bias attention fast path.

    Flattens the leading batch dims, builds the summed pair bias and mask, and delegates to the fused
    gated kernel that fuses the sigmoid output gate and output projection.

    Args:
        query: ``(..., q_tokens, query_size)`` gating input passed to the fused kernel.
        query_projection, key_projection, value_projection: Head-split q/k/v of shape
            ``(..., heads, tokens, head_dim)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.
        projection: The projection supplying the gate/output weights and ``hidden_size``.

    Returns:
        ``(..., q_tokens, hidden_size)`` output, or ``None`` when there is no pair bias to feed the
        fused kernel (the caller then falls back to the ungated OOps path).
    """
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
        device=query_projection.device,
    )
    if pair_bias is None:
        return None
    mask = _oops_attention_mask(
        attention_mask,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        query_length=query_length,
        key_length=key_length,
        device=query_projection.device,
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
) -> Tensor:
    r"""
    Run the ungated OOps pair-bias attention path.

    Flattens the leading batch dims, builds the summed pair bias and mask, calls the fused pair-bias
    kernel, and restores the original prefix shape.

    Args:
        query_projection, key_projection, value_projection: Head-split q/k/v of shape
            ``(..., heads, tokens, head_dim)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.

    Returns:
        ``(..., heads, q_tokens, head_dim)`` attention output before the head merge.
    """
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
        device=query_projection.device,
    )
    mask = _oops_attention_mask(
        attention_mask,
        prefix_shape=prefix_shape,
        flat_prefix_size=flat_prefix_size,
        query_length=query_length,
        key_length=key_length,
        device=query_projection.device,
    )
    output = _oops_pair_bias_attention(query_flat, key_flat, value_flat, mask=mask, pair_bias=pair_bias, scale=1.0)
    return cast(Tensor, output).reshape(*prefix_shape, num_heads, query_length, head_dim)


def _oops_pair_bias(
    biases: Sequence[Tensor] | None,
    *,
    prefix_shape: torch.Size,
    flat_prefix_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    r"""
    Broadcast and sum the pair biases into the flat ``(flat_prefix, heads, q, k)`` bias the kernel wants.

    Args:
        biases: Optional additive pair biases; ``None``/empty yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).
        num_heads, query_length, key_length: Attention head count and q/k token counts.
        dtype, device: Target dtype/device for the biases.

    Returns:
        ``(flat_prefix_size, heads, q, k)`` summed pair bias, or ``None`` when there is no bias.
    """
    if not biases:
        return None
    target_shape = (*prefix_shape, num_heads, query_length, key_length)
    broadcast_biases = [
        _broadcast_to_attention_shape(bias.to(dtype=dtype, device=device), target_shape, tail_dims=3) for bias in biases
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
    device: torch.device,
) -> Tensor | None:
    r"""
    Broadcast the attention mask to the flat boolean mask the OOps kernel expects.

    Accepts a per-query-key mask ``(..., q, k)`` or a key-only mask ``(..., k)`` and flattens the leading
    batch dims accordingly.

    Args:
        attention_mask: Optional boolean mask; ``None`` yields ``None``.
        prefix_shape: The leading batch dims before flattening.
        flat_prefix_size: Product of ``prefix_shape`` (the flattened batch size).
        query_length, key_length: Query and key token counts.
        device: Target device for the mask.

    Returns:
        The flattened boolean mask (``(flat_prefix, q, k)`` or ``(flat_prefix, k)``), or ``None``.

    Raises:
        ValueError: If ``attention_mask`` is not broadcastable to ``(..., q, k)`` or ``(..., k)``.
    """
    if attention_mask is None:
        return None
    attention_mask = attention_mask.to(device=device).bool()
    if attention_mask.shape[-2:] == (query_length, key_length):
        target_shape = (*prefix_shape, query_length, key_length)
        return _broadcast_to_attention_shape(attention_mask, target_shape, tail_dims=2).reshape(
            flat_prefix_size, query_length, key_length
        )
    elif attention_mask.shape[-1:] == (key_length,):
        target_shape = (*prefix_shape, key_length)
        return _broadcast_to_attention_shape(attention_mask, target_shape, tail_dims=1).reshape(
            flat_prefix_size, key_length
        )
    else:
        raise ValueError(
            "attention_mask must have shape broadcastable to "
            f"(..., {query_length}, {key_length}) or (..., {key_length}), got {tuple(attention_mask.shape)}"
        )


def _deepspeed_attention(
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    biases: list[Tensor],
) -> Tensor:
    r"""
    Run DeepSpeed ``DS4Sci_EvoformerAttention``, adapting layout, dtype, and bias arity.

    Transposes to the kernel's ``(..., tokens, heads, head_dim)`` layout, collapses batch dims to the
    two the kernel accepts, casts to bfloat16 when the input is not already half precision, and pads the
    bias list so a lone pair bias becomes ``[None, bias]``.

    Args:
        query_projection, key_projection, value_projection: Head-split q/k/v of shape
            ``(..., heads, tokens, head_dim)``.
        biases: Additive biases (mask bias first, then pair biases).

    Returns:
        ``(..., heads, q_tokens, head_dim)`` attention output in the original dtype and prefix shape.

    Raises:
        RuntimeError: If deepspeed4science is not importable.
    """
    try:
        from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
    except ImportError as error:
        raise RuntimeError("operator_backend='deepspeed' requires deepspeed4science") from error

    original_dtype = query_projection.dtype
    query = query_projection.transpose(-2, -3)
    key = key_projection.transpose(-2, -3)
    value = value_projection.transpose(-2, -3)
    original_shape = query.shape

    if len(original_shape[:-3]) != 2:
        query = _reshape_deepspeed_attention_dims(query)
        key = _reshape_deepspeed_attention_dims(key)
        value = _reshape_deepspeed_attention_dims(value)
        biases = [_reshape_deepspeed_attention_dims(bias) for bias in biases]

    deepspeed_biases: list[Tensor | None] = list(biases)
    if len(deepspeed_biases) == 1 and deepspeed_biases[0] is not None and deepspeed_biases[0].shape[-3] != 1:
        deepspeed_biases = [None, deepspeed_biases[0]]

    if original_dtype not in {torch.bfloat16, torch.float16}:
        output = DS4Sci_EvoformerAttention(
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
            value.to(torch.bfloat16),
            [None if bias is None else bias.to(torch.bfloat16) for bias in deepspeed_biases],
        ).to(original_dtype)
    else:
        output = DS4Sci_EvoformerAttention(query, key, value, deepspeed_biases)

    output = output.reshape(original_shape)
    return output.transpose(-2, -3)


def _attention_biases(
    query_projection: Tensor,
    key_projection: Tensor,
    *,
    biases: Sequence[Tensor] | None,
    attention_mask: Tensor | None,
    operator_backend: str,
) -> list[Tensor]:
    r"""
    Assemble the additive bias list (pair biases plus a masking bias) for the SDPA/DeepSpeed paths.

    Casts and head-broadcasts each pair bias, then prepends a bias derived from ``attention_mask``: a
    per-key ``(1, 1, k)`` masking bias for DeepSpeed, or a full large-negative additive mask otherwise.

    Args:
        query_projection, key_projection: Head-split q/k of shape ``(..., heads, tokens, head_dim)``.
        biases: Optional additive pair biases.
        attention_mask: Optional boolean attention mask.
        operator_backend: The resolved backend, selecting the masking-bias form.

    Returns:
        The list of additive biases, with the masking bias (if any) at index 0.
    """
    attention_biases: list[Tensor] = []
    if biases is not None:
        for bias in biases:
            if bias.dim() == query_projection.dim() - 1:
                bias = bias.unsqueeze(-3)
            attention_biases.append(bias.to(dtype=query_projection.dtype, device=query_projection.device))

    if attention_mask is None:
        return attention_biases

    if operator_backend == "deepspeed":
        attention_biases.insert(
            0,
            _deepspeed_key_mask_bias(
                query_projection,
                key_projection,
                attention_mask=attention_mask,
            ),
        )
        return attention_biases

    attention_mask = attention_mask.to(device=query_projection.device)
    if attention_mask.dim() == query_projection.dim() - 1:
        attention_mask = attention_mask.unsqueeze(-3)

    attention_mask = attention_mask.bool()
    mask_bias = torch.zeros(
        *query_projection.shape[:-1],
        key_projection.shape[-2],
        dtype=query_projection.dtype,
        device=query_projection.device,
    )
    mask_bias = mask_bias.masked_fill(~attention_mask, torch.finfo(query_projection.dtype).min)
    attention_biases.insert(0, mask_bias)
    return attention_biases


def _deepspeed_key_mask_bias(
    query_projection: Tensor,
    key_projection: Tensor,
    *,
    attention_mask: Tensor,
) -> Tensor:
    r"""
    Build the DeepSpeed key-masking bias of shape ``(..., 1, 1, k)`` from a boolean attention mask.

    Reduces a per-query-key mask to per-key with ``any`` over the query axis, then maps masked-out keys
    to the dtype minimum so the kernel drops them.

    Args:
        query_projection, key_projection: Head-split q/k of shape ``(..., heads, tokens, head_dim)``.
        attention_mask: Boolean attention mask over keys (optionally per-query-key).

    Returns:
        ``(..., 1, 1, k)`` additive masking bias.
    """
    attention_mask = attention_mask.to(device=query_projection.device).bool()
    if attention_mask.dim() == query_projection.dim() - 1:
        attention_mask = attention_mask.any(dim=-2)
    mask_bias = torch.zeros(
        *query_projection.shape[:-3],
        1,
        1,
        key_projection.shape[-2],
        dtype=query_projection.dtype,
        device=query_projection.device,
    )
    return mask_bias.masked_fill(~attention_mask.unsqueeze(-2).unsqueeze(-2), torch.finfo(query_projection.dtype).min)


def _reshape_deepspeed_attention_dims(tensor: Tensor) -> Tensor:
    r"""
    Coerce the leading batch dims to the exactly-two the DeepSpeed kernel accepts.

    Prepends singleton dims when there are fewer than two, or collapses the middle dims into one when
    there are more; leaves the trailing three ``(tokens, heads, head_dim)`` dims untouched.

    Args:
        tensor: A ``(..., tokens, heads, head_dim)`` q/k/v/bias tensor.

    Returns:
        The same tensor reshaped to exactly two leading batch dims.
    """
    num_batch_dims = len(tensor.shape[:-3])
    if num_batch_dims < 2:
        return tensor.reshape(*((1,) * (2 - num_batch_dims)), *tensor.shape)
    if num_batch_dims > 2:
        return tensor.reshape(tensor.shape[0], -1, *tensor.shape[-3:])
    return tensor


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


def _broadcast_to_attention_shape(tensor: Tensor, target_shape: tuple[int, ...], *, tail_dims: int) -> Tensor:
    r"""
    Broadcast ``tensor`` to ``target_shape``, matching the last ``tail_dims`` axes exactly.

    The trailing ``tail_dims`` axes must already match ``target_shape`` (up to size-1 broadcasting); the
    leading axes are right-aligned and expanded. This is stricter than plain ``broadcast_to`` so a
    mis-shaped bias/mask fails loudly rather than silently mis-aligning.

    Args:
        tensor: The tensor to broadcast.
        target_shape: The desired output shape.
        tail_dims: The number of trailing axes that must line up with ``target_shape``.

    Returns:
        A view of ``tensor`` broadcast to ``target_shape``.

    Raises:
        ValueError: If ``tensor`` cannot be broadcast to ``target_shape`` under these rules.
    """
    if tuple(tensor.shape) == target_shape:
        return tensor
    if tensor.dim() > len(target_shape):
        raise ValueError(f"cannot broadcast tensor shape {tuple(tensor.shape)} to {target_shape}")
    if tensor.dim() < tail_dims:
        raise ValueError(f"cannot broadcast tensor shape {tuple(tensor.shape)} to {target_shape}")
    source_prefix = tensor.shape[:-tail_dims]
    source_tail = tensor.shape[-tail_dims:]
    target_prefix = target_shape[:-tail_dims]
    target_tail = target_shape[-tail_dims:]
    if len(source_prefix) > len(target_prefix):
        raise ValueError(f"cannot broadcast tensor shape {tuple(tensor.shape)} to {target_shape}")
    if any(source not in (1, target) for source, target in zip(source_prefix, target_prefix, strict=False)):
        raise ValueError(f"cannot broadcast tensor shape {tuple(tensor.shape)} to {target_shape}")
    if any(source not in (1, target) for source, target in zip(source_tail, target_tail, strict=True)):
        raise ValueError(f"cannot broadcast tensor shape {tuple(tensor.shape)} to {target_shape}")
    view_shape = (*source_prefix, *((1,) * (len(target_prefix) - len(source_prefix))), *source_tail)
    return torch.broadcast_to(tensor.reshape(view_shape), target_shape)
