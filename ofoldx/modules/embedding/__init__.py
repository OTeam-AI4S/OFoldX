r"""
Reusable input-embedding components.
"""

from __future__ import annotations

from .factorized import FactorizedPairLinear
from .fourier import FrozenFourierEmbedding
from .position import RelativePositionBuckets, RelativePositionEmbedder, generate_relative_position_features
from .template import TemplateEmbedder, TemplateFeatures
