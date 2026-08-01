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
Calling-convention adapters for cuEquivariance operators.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor


def cueq_triangle_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor,
    mask: Tensor | None,
    *,
    scale: float,
) -> Tensor:
    r"""
    Run cuEquivariance triangle attention.
    """

    from cuequivariance_torch import triangle_attention

    return cast(Tensor, triangle_attention(query, key, value, bias, mask=mask, scale=scale))


def cueq_triangle_multiplication(
    pair: Tensor,
    *,
    direction: str,
    mask: Tensor | None,
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    input_weight: Tensor,
    input_bias: Tensor | None,
    input_gate_weight: Tensor,
    input_gate_bias: Tensor | None,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None,
    output_gate_weight: Tensor,
    output_gate_bias: Tensor | None,
    eps: float,
) -> Tensor:
    r"""
    Run cuEquivariance triangle multiplicative update.
    """

    from cuequivariance_torch import triangle_multiplicative_update

    return cast(
        Tensor,
        triangle_multiplicative_update(
            pair,
            direction=direction,
            mask=mask,
            norm_in_weight=input_norm_weight,
            norm_in_bias=input_norm_bias,
            p_in_weight=input_weight,
            p_in_bias=input_bias,
            g_in_weight=input_gate_weight,
            g_in_bias=input_gate_bias,
            norm_out_weight=output_norm_weight,
            norm_out_bias=output_norm_bias,
            p_out_weight=output_weight,
            p_out_bias=output_bias,
            g_out_weight=output_gate_weight,
            g_out_bias=output_gate_bias,
            eps=eps,
        ),
    )
