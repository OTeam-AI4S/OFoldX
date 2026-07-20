r"""
Adaptive layer normalization modules.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .backend import adaptive_layer_norm, resolve_config_operator_backend


class AdaptiveLayerNorm(nn.Module):
    r"""
    Config-driven AdaLN: affine-free hidden norm, conditioning norm, sigmoid scale, and bias shift.

    The scale and shift projections are held as one fused ``conditioning_projection``
    (``[2 * hidden_size, conditioning_size]``, bias-free) plus a ``conditioning_scale_bias``
    (``[hidden_size]``); this is exactly the layout the checkpoint converters emit (the scale/shift
    weights concatenated, the scale bias kept separate), so the operator dispatches to the fused OOps
    ``adaptive_layer_norm`` kernel with a byte-identical eager fallback. Legacy checkpoints that still
    carry the separate ``gate_projection``/``shift_projection`` tensors are folded into the fused layout
    on load, so both key conventions load unchanged.

    Args:
        hidden_size: Width of the hidden representation (affine-free normalized).
        conditioning_size: Width of the conditioning vector driving the scale and shift.
        operator_backend: Which operator backend to run; see [ofoldx.modules.backend][].

    Examples:
        >>> import torch
        >>> norm = AdaptiveLayerNorm(8, 4, operator_backend="eager")
        >>> norm(torch.randn(2, 3, 8), torch.randn(2, 3, 4)).shape
        torch.Size([2, 3, 8])
    """

    def __init__(self, hidden_size: int, conditioning_size: int, operator_backend: str = "auto") -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        self.conditioning_norm = nn.LayerNorm(conditioning_size, bias=False)
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
            eps=self.conditioning_norm.eps,
            operator_backend=self.operator_backend,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        r"""
        Fold legacy separate ``gate_projection``/``shift_projection`` tensors into the fused layout.

        Converters that emit the fused ``conditioning_projection.weight`` + ``conditioning_scale_bias``
        keys load natively; older checkpoints carrying the two separate projections are concatenated here
        (scale weights atop shift weights, the scale bias moved to ``conditioning_scale_bias``) so both key
        conventions load into the same module unchanged.
        """

        gate_weight_key = f"{prefix}gate_projection.weight"
        shift_weight_key = f"{prefix}shift_projection.weight"
        if gate_weight_key in state_dict and shift_weight_key in state_dict:
            gate_weight = state_dict.pop(gate_weight_key)
            shift_weight = state_dict.pop(shift_weight_key)
            state_dict[f"{prefix}conditioning_projection.weight"] = torch.cat([gate_weight, shift_weight], dim=0)
            gate_bias_key = f"{prefix}gate_projection.bias"
            if gate_bias_key in state_dict:
                state_dict[f"{prefix}conditioning_scale_bias"] = state_dict.pop(gate_bias_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
