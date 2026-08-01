r"""
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .adaptive_norm import adaptive_layer_norm
from .factorized_pair_linear import factorized_pair_linear
from .normalization import LayerNorm, layer_norm
from .pair_norm_projection import pair_norm_projection
from .resolve import Backend, resolve_config_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication

__all__ = [
    "Backend",
    "LayerNorm",
    "adaptive_layer_norm",
    "factorized_pair_linear",
    "layer_norm",
    "pair_norm_projection",
    "resolve_config_backend",
    "transition",
    "triangle_multiplication",
]
