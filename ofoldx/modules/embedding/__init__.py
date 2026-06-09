r"""
Embedding modules that turn model inputs into learned representations.

This package groups the input-side embedders shared across model families:
feature packing helpers ([`pack_atom_features`][ofoldx.modules.embedding.features.pack_atom_features],
[`pack_token_tail`][ofoldx.modules.embedding.features.pack_token_tail]), frozen random
Fourier features ([`FrozenFourierEmbedding`][ofoldx.modules.embedding.fourier.FrozenFourierEmbedding]),
relative-position encoding
([`RelativePositionEmbedder`][ofoldx.modules.embedding.position.RelativePositionEmbedder]),
and template embedding
([`TemplateEmbedder`][ofoldx.modules.embedding.template.TemplateEmbedder]).
"""

from __future__ import annotations

from .features import pack_atom_features, pack_token_tail
from .fourier import FrozenFourierEmbedding
from .position import RelativePositionBuckets, RelativePositionEmbedder, generate_relative_position_features
from .template import TemplateEmbedder, TemplateFeatures

__all__ = [
    "FrozenFourierEmbedding",
    "RelativePositionBuckets",
    "RelativePositionEmbedder",
    "TemplateEmbedder",
    "TemplateFeatures",
    "generate_relative_position_features",
    "pack_atom_features",
    "pack_token_tail",
]
