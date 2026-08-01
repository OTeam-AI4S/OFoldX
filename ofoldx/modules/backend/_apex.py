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
Calling-convention adapters for NVIDIA Apex operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor


def apex_layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor | None:
    r"""
    Apply Apex OpenFold Triton LayerNorm when its affine contract is available.
    """

    if weight is None or bias is None:
        return None

    from apex.contrib.openfold_triton import LayerNormSmallShapeOptImpl

    return cast(
        Tensor,
        LayerNormSmallShapeOptImpl.apply(
            input.contiguous(), tuple(normalized_shape), weight.contiguous(), bias.contiguous(), eps
        ),
    )


def apex_evoformer_attention_supported(shape: Sequence[int], *, training: bool) -> bool:
    r"""
    Return whether Apex's OpenFold Triton kernel schedules this attention shape.
    """

    from apex.contrib.openfold_triton import CanSchTriMHA

    return bool(CanSchTriMHA(list(shape), has_bias=True, inf=1e9, training=training))


def apex_evoformer_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    bias: Tensor,
    *,
    training: bool,
) -> Tensor:
    r"""
    Run Apex's OpenFold Triton attention kernel.
    """

    from apex.contrib.openfold_triton import AttnTri

    return cast(
        Tensor,
        AttnTri(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            None if mask is None else mask.contiguous(),
            bias.contiguous(),
            1e9,
            training,
        ),
    )
