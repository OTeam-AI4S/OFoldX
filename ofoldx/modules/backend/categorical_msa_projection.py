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
Backend-dispatched categorical MSA projection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch.nn.functional as F
from torch import Tensor

from ..._oops import oops_kernel
from .resolve import resolve_tensor_backend

_oops_categorical_msa_projection = cast(Callable[..., Tensor], oops_kernel("categorical_msa_projection"))


def categorical_msa_projection(
    tokens: Tensor,
    continuous_features: Tensor,
    weight: Tensor,
    *,
    backend: str,
) -> Tensor:
    r"""
    Project categorical MSA tokens and two continuous features without materializing one-hot vectors.

    Args:
        tokens: MSA token indices with shape ``[..., M, N]``.
        continuous_features: Continuous MSA features with shape ``[..., M, N, 2]``.
        weight: Packed projection weight with shape ``[C, V + 2]``.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        Projected MSA features with shape ``[..., M, N, C]``.
    """

    expected_continuous_shape = (*tokens.shape, 2)
    if continuous_features.shape != expected_continuous_shape:
        raise ValueError(
            f"continuous_features must have shape {expected_continuous_shape}; "
            f"got {tuple(continuous_features.shape)}"
        )

    backend = resolve_tensor_backend(
        backend,
        tensors=(tokens, continuous_features, weight),
        supported=("oops", "eager"),
        public_default="eager",
        oops=_oops_categorical_msa_projection,
    )
    if backend == "oops":
        return _oops_categorical_msa_projection(tokens, continuous_features, weight)

    continuous_projection = F.linear(
        continuous_features,
        weight[:, -2:],
    )
    categorical_projection = F.embedding(tokens, weight[:, :-2].T)
    return categorical_projection.to(dtype=continuous_projection.dtype) + continuous_projection
