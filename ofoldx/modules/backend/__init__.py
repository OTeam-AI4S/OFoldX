r"""
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .adaptive_norm import adaptive_layer_norm
from .factorized_pair_linear import factorized_pair_linear
from .normalization import LayerNorm, layer_norm
from .resolve import Backend, resolve_config_backend
from .transition import transition

__all__ = [
    "Backend",
    "LayerNorm",
    "adaptive_layer_norm",
    "factorized_pair_linear",
    "layer_norm",
    "resolve_config_backend",
    "transition",
]
