r"""
Operator-backend dispatch layer.

Each operator that has more than one implementation (OOps fused kernels, pure-PyTorch eager, torch
SDPA, DeepSpeed, and later cuEquivariance / NVIDIA) gets one dispatch function here. The function
resolves ``operator_backend`` (including ``"auto"``) and adapts each backend's calling convention, so
modules pass normalized inputs and never branch on the backend themselves.
"""

from __future__ import annotations

from ._oops import LayerNorm, oops_kernel
from .atom_window import atom_to_token_mean, token_pair_to_atom_window_gather, token_to_atom_broadcast
from .outer_product_mean import outer_product_mean
from .pair_weighted_average import pair_weighted_average
from .resolve import resolve_config_operator_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication

__all__ = [
    "LayerNorm",
    "atom_to_token_mean",
    "oops_kernel",
    "outer_product_mean",
    "pair_weighted_average",
    "resolve_config_operator_backend",
    "token_pair_to_atom_window_gather",
    "token_to_atom_broadcast",
    "transition",
    "triangle_multiplication",
]
