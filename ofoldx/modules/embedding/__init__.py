r"""
Embedding modules that turn model inputs into learned representations.

This package groups reusable input-side embedders: frozen
random Fourier features
([`FrozenFourierEmbedding`][ofoldx.modules.embedding.fourier.FrozenFourierEmbedding]), relative-position encoding
([`RelativePositionEmbedder`][ofoldx.modules.embedding.position.RelativePositionEmbedder]),
and template embedding
([`TemplateEmbedder`][ofoldx.modules.embedding.template.TemplateEmbedder]).
"""

from __future__ import annotations

from .fourier import FrozenFourierEmbedding
from .position import RelativePositionBuckets, RelativePositionEmbedder, generate_relative_position_features
from .template import TemplateEmbedder, TemplateFeatures
