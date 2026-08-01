r"""
Shared mathematical modules.
"""

from __future__ import annotations

from .attention import GatedMultiheadAttention, MultiheadAttentionProjection
from .dropout import StructuredDropout
from .feed_forward import SwiGluFeedForward
from .geometry import rigid_from_three_points, uniform_random_rotation, virtual_beta_carbon, weighted_rigid_align
from .mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from .symmetry import match_symmetric_truth
from .transition import ConditionedTransition, Transition

__all__ = [
    "AMINO_ACID_COMPONENT_IDS",
    "AMINO_ACID_TOKEN_BY_COMPONENT",
    "AMINO_ACID_UNKNOWN_TOKEN",
    "ConditionedTransition",
    "GatedMultiheadAttention",
    "MultiheadAttentionProjection",
    "StructuredDropout",
    "SwiGluFeedForward",
    "Transition",
    "match_symmetric_truth",
    "rigid_from_three_points",
    "uniform_random_rotation",
    "virtual_beta_carbon",
    "weighted_rigid_align",
]
