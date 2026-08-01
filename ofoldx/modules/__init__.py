r"""
Shared mathematical modules.
"""

from __future__ import annotations

from .attention import GatedMultiheadAttention, MultiheadAttentionProjection
from .dropout import StructuredDropout
from .feed_forward import SwiGluFeedForward
from .transition import ConditionedTransition, Transition

__all__ = [
    "ConditionedTransition",
    "GatedMultiheadAttention",
    "MultiheadAttentionProjection",
    "StructuredDropout",
    "SwiGluFeedForward",
    "Transition",
]
