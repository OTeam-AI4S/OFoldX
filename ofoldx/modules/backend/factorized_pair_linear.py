r"""
Backend dispatch for factorized pair linear projection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_backend

_oops_factorized_pair_linear = cast(Callable[..., Tensor], oops_kernel("factorized_linear"))


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
        supported=("oops", "eager"),
        public_default="eager",
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
