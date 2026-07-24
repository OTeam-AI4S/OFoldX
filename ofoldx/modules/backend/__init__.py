r"""
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .adaptive_norm import adaptive_layer_norm
from .atom_window import atom_to_token_mean, token_pair_to_atom_window_gather, token_to_atom_broadcast
from .factorized_pair_linear import factorized_pair_linear
from .normalization import LayerNorm, layer_norm
from .outer_product_mean import outer_product_mean
from .pair_weighted_average import pair_weighted_average
from .resolve import resolve_config_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication
