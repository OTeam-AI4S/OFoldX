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
