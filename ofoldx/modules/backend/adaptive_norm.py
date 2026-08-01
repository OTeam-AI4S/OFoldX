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
Backend-dispatched adaptive layer normalization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ..._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_adaptive_layer_norm = cast(Callable[..., Tensor], oops_kernel("adaptive_layer_norm"))

_SUPPORTED = ("oops", "eager")


def adaptive_layer_norm(
    hidden: Tensor,
    conditioning: Tensor,
    *,
    conditioning_norm_weight: Tensor,
    projection_weight: Tensor,
    scale_bias: Tensor,
    input_norm_eps: float,
    conditioning_norm_eps: float,
    backend: str,
) -> Tensor:
    r"""
    Apply adaptive layer norm ``sigmoid(scale) * norm(hidden) + shift`` on the selected backend.

    ``hidden`` is layer-normed affine-free; ``conditioning`` is layer-normed with
    ``conditioning_norm_weight`` (no bias). The normed conditioning is projected once through the fused
    ``projection_weight`` and split into the raw ``scale`` and ``shift`` halves; ``scale_bias`` is added
    to the scale half before the sigmoid gate. The ``"oops"`` backend delegates to OOps dispatch;
    ``"eager"`` runs the equivalent pure-PyTorch composition.

    Args:
        hidden: ``[..., hidden_size]`` input activations.
        conditioning: ``[..., conditioning_size]`` conditioning activations.
        conditioning_norm_weight: ``[conditioning_size]`` LayerNorm scale for the conditioning norm.
        projection_weight: ``[2 * hidden_size, conditioning_size]`` fused bias-free scale/shift projection
            weight; the first ``hidden_size`` rows produce the scale, the remainder the shift.
        scale_bias: ``[hidden_size]`` additive bias applied to the scale half before the sigmoid.
        input_norm_eps: LayerNorm epsilon for ``hidden``.
        conditioning_norm_eps: LayerNorm epsilon for ``conditioning``.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., hidden_size]`` conditioned activations.
    """

    resolved = resolve_tensor_backend(
        backend,
        tensors=(hidden, conditioning),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_adaptive_layer_norm,
    )
    if resolved == "oops":
        return _oops_adaptive_layer_norm(
            hidden,
            conditioning,
            conditioning_projection_weight=projection_weight,
            conditioning_scale_bias=scale_bias,
            conditioning_norm_weight=conditioning_norm_weight,
            input_norm_eps=input_norm_eps,
            conditioning_norm_eps=conditioning_norm_eps,
        )
    hidden_normed = layer_norm(
        hidden,
        (hidden.shape[-1],),
        None,
        None,
        input_norm_eps,
        backend=backend,
    )
    conditioning_normed = layer_norm(
        conditioning,
        (conditioning.shape[-1],),
        conditioning_norm_weight,
        None,
        conditioning_norm_eps,
        backend=backend,
    )
    scale_weight, shift_weight = projection_weight.chunk(2, dim=0)
    scale = F.linear(conditioning_normed, scale_weight)
    shift = F.linear(conditioning_normed, shift_weight)
    return torch.sigmoid(scale + scale_bias) * hidden_normed + shift
