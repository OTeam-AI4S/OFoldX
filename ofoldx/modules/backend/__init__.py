r"""
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .normalization import LayerNorm, layer_norm
from .resolve import Backend, resolve_config_backend

__all__ = [
    "Backend",
    "LayerNorm",
    "layer_norm",
    "resolve_config_backend",
]
