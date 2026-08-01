r"""
Calling-convention adapter for DeepSpeed scientific operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor


def deepspeed_evoformer_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    biases: Sequence[Tensor | None],
) -> Tensor:
    r"""
    Run the DeepSpeed4Science Evoformer attention operator.
    """

    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    return cast(
        Tensor,
        DS4Sci_EvoformerAttention(query.contiguous(), key.contiguous(), value.contiguous(), list(biases)),
    )


def deepspeed_layer_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    r"""
    Run DeepSpeed's Triton inference LayerNorm operator.
    """

    from deepspeed.ops.transformer.inference.triton.layer_norm import layer_norm

    if weight is None:
        weight = input.new_ones(input.shape[-1])
    if bias is None:
        bias = input.new_zeros(input.shape[-1])
    return cast(
        Tensor,
        layer_norm(input.contiguous(), weight.contiguous(), bias.contiguous(), eps),
    )
