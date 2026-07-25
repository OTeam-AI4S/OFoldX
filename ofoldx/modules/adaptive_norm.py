r"""
Adaptive layer normalization modules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backend import LayerNorm, adaptive_layer_norm, resolve_config_backend


class AdaptiveLayerNorm(nn.Module):
    r"""
    Config-driven AdaLN: affine-free hidden norm, conditioning norm, sigmoid scale, and bias shift.

    The scale and shift projections are held as one fused ``conditioning_projection``
    (``[2 * hidden_size, conditioning_size]``, bias-free) plus a ``conditioning_scale_bias``
    (``[hidden_size]``). The selected backend consumes this packed parameter
    layout directly while preserving one tensor contract across eager and accelerated paths.

    Args:
        hidden_size: Width of the hidden representation (affine-free normalized).
        conditioning_size: Width of the conditioning vector driving the scale and shift.
        backend: Which backend to run; see [ofoldx.modules.backend][].
    """

    def __init__(self, hidden_size: int, conditioning_size: int, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.conditioning_norm = LayerNorm(conditioning_size, bias=False, backend=backend)
        self.conditioning_projection = nn.Linear(conditioning_size, hidden_size * 2, bias=False)
        self.conditioning_scale_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        r"""
        Adaptively normalize ``hidden`` under the ``conditioning`` scale and shift.

        Args:
            hidden: `[..., hidden_size]` hidden representation.
            conditioning: `[..., conditioning_size]` conditioning vector.

        Returns:
            The `[..., hidden_size]` conditioned representation.
        """

        return adaptive_layer_norm(
            hidden,
            conditioning,
            conditioning_norm_weight=self.conditioning_norm.weight,
            projection_weight=self.conditioning_projection.weight,
            scale_bias=self.conditioning_scale_bias,
            input_norm_eps=self.conditioning_norm.eps,
            conditioning_norm_eps=self.conditioning_norm.eps,
            backend=self.backend,
        )
