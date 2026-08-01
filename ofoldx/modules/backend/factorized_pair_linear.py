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
Backend dispatch for factorized pair linear projection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ..._oops import oops_kernel
from .resolve import resolve_tensor_backend

_oops_factorized_pair_linear = cast(Callable[..., Tensor], oops_kernel("factorized_linear"))

_SUPPORTED = ("oops", "eager")


def factorized_pair_linear(
    pair_features: Tensor,
    token_features: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    pair_prefix_size: int,
    backend: str,
) -> Tensor:
    r"""
    Project pair features and one token feature track used on both pair axes.

    ``weight`` follows the packed column order
    ``[pair_prefix, token_i, token_j, pair_suffix]``.

    Args:
        pair_features: Pairwise features with shape ``[..., N, N, P]``.
        token_features: Per-token features with shape ``[..., N, T]``.
        weight: Linear projection weight with shape ``[C, P + 2 * T]``.
        bias: Optional linear projection bias with shape ``[C]``.
        pair_prefix_size: Number of pair-feature columns before the token blocks.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        Projected pair features with shape ``[..., N, N, C]``.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(pair_features, token_features, weight, bias),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_factorized_pair_linear,
    )
    if backend == "oops":
        return _oops_factorized_pair_linear(
            pair_features,
            token_features,
            weight,
            bias,
            pair_prefix_size=pair_prefix_size,
        )

    token_size = token_features.shape[-1]
    left_end = pair_prefix_size + token_size
    right_end = left_end + token_size
    pair_weight = torch.cat((weight[:, :pair_prefix_size], weight[:, right_end:]), dim=-1)
    return (
        F.linear(pair_features, pair_weight)
        + F.linear(token_features, weight[:, pair_prefix_size:left_end]).unsqueeze(-2)
        + F.linear(token_features, weight[:, left_end:right_end], bias).unsqueeze(-3)
    )
