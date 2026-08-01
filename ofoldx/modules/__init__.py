# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Shared mathematical modules.
"""

from __future__ import annotations

from .attention import GatedMultiheadAttentionProjection, MultiheadAttention, MultiheadAttentionProjection
from .dropout import StructuredDropout
from .feed_forward import FeedForward, SwiGluFeedForward
from .geometry import (
    atom37_to_torsion_angles,
    dihedral_angle,
    frames_to_atom14_positions,
    rotation_from_two_vectors,
    torsion_angles_to_frames,
    uniform_random_rotation,
    virtual_beta_carbon,
    weighted_rigid_align,
)
from .transition import ConditionedTransition, FeedForwardTransition, Transition
