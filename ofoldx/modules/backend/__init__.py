r"""
Operator-backend dispatch layer.

Every dispatched operator has a public PyTorch implementation. The
private OOps adapter exposes OOps-owned implementations and dispatch; models select it
through the shared dispatcher without changing tensor contracts or state dicts.
"""

from __future__ import annotations

from ._oops import LayerNorm
from .adaptive_norm import adaptive_layer_norm
from .atom_window import atom_to_token_mean, token_pair_to_atom_window_gather, token_to_atom_broadcast
from .outer_product_mean import outer_product_mean
from .pair_weighted_average import pair_weighted_average
from .resolve import resolve_config_operator_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication
