r"""
Shared building blocks (attention, transitions, geometry) reused across model families.

This package holds only code shared by at least two families; it is the canonical reference for
naming and API. Its public surface is re-exported here for convenient import:

- [GatedMultiheadAttention][ofoldx.modules.attention.GatedMultiheadAttention] and
  [MultiheadAttentionProjection][ofoldx.modules.attention.MultiheadAttentionProjection]
- [Transition][ofoldx.modules.transition.Transition] and
  [ConditionedTransition][ofoldx.modules.transition.ConditionedTransition]
- [weighted_rigid_align][ofoldx.modules.geometry.weighted_rigid_align] and
  [uniform_random_rotation][ofoldx.modules.geometry.uniform_random_rotation]

Examples:
    >>> from ofoldx.modules import Transition, weighted_rigid_align
"""

from __future__ import annotations

from .attention import GatedMultiheadAttention, MultiheadAttentionProjection
from .geometry import uniform_random_rotation, weighted_rigid_align
from .mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from .transition import ConditionedTransition, Transition
