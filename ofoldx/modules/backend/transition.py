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
Backend-dispatched SwiGLU transition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch.nn.functional as F
from torch import Tensor

from ..._oops import oops_kernel
from .normalization import layer_norm
from .resolve import resolve_tensor_backend

_oops_transition = cast(Callable[..., Tensor], oops_kernel("transition"))

_SUPPORTED = ("oops", "eager")


def transition(
    hidden: Tensor,
    *,
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    input_norm_eps: float,
    backend: str,
) -> Tensor:
    r"""
    Apply the SwiGLU transition ``down(silu(gate(norm(x))) * up(norm(x)))`` on the selected backend.

    Pre-normalizes ``hidden`` with LayerNorm, gates the up-projection by the SiLU
    of the gate-projection, and projects back down. Shared LayerNorm supports
    half activations with float32 normalization parameters using its native
    centered-statistics accumulation. The ``"oops"`` backend delegates to the
    OOps-owned implementation; other backends retain the same mathematical
    equation through ordinary PyTorch operations.

    Args:
        hidden: ``[..., hidden_size]`` input activations.
        input_norm_weight: ``[hidden_size]`` LayerNorm scale for the pre-normalization.
        input_norm_bias: ``[hidden_size]`` LayerNorm shift for the pre-normalization.
        gate_weight: ``[intermediate_size, hidden_size]`` bias-free gate projection weight.
        up_weight: ``[intermediate_size, hidden_size]`` bias-free up projection weight.
        down_weight: ``[hidden_size, intermediate_size]`` bias-free down projection weight.
        input_norm_eps: LayerNorm epsilon.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., hidden_size]`` transformed activations.
    """

    resolved = resolve_tensor_backend(
        backend,
        tensors=(hidden,),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_transition,
    )
    if resolved == "oops":
        return _oops_transition(
            hidden,
            input_norm_weight=input_norm_weight,
            input_norm_bias=input_norm_bias,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            input_norm_eps=input_norm_eps,
        )
    normed = layer_norm(
        hidden,
        (hidden.shape[-1],),
        input_norm_weight,
        input_norm_bias,
        input_norm_eps,
        backend=backend,
    )
    return F.linear(F.silu(F.linear(normed, gate_weight)) * F.linear(normed, up_weight), down_weight)
