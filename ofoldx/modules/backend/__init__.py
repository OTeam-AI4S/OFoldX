r"""
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .adaptive_norm import adaptive_layer_norm
from .atom_window import atom_to_token_mean, token_pair_to_atom_window_gather, token_to_atom_broadcast
from .categorical_msa_projection import categorical_msa_projection
from .equivariant import Se3Basis, SegmentSoftmaxCsr, SegmentSumCsr, WeightedTensorProduct
from .factorized_pair_linear import factorized_pair_linear
from .normalization import LayerNorm, layer_norm
from .outer_product_mean import outer_product_mean
from .pair_norm_projection import pair_norm_projection
from .pair_weighted_average import pair_weighted_average
from .resolve import Backend, resolve_config_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication

__all__ = [
    "Backend",
    "LayerNorm",
    "Se3Basis",
    "SegmentSoftmaxCsr",
    "SegmentSumCsr",
    "WeightedTensorProduct",
    "adaptive_layer_norm",
    "atom_to_token_mean",
    "categorical_msa_projection",
    "factorized_pair_linear",
    "layer_norm",
    "outer_product_mean",
    "pair_norm_projection",
    "pair_weighted_average",
    "resolve_config_backend",
    "token_pair_to_atom_window_gather",
    "token_to_atom_broadcast",
    "transition",
    "triangle_multiplication",
]
