r"""
AlphaFold 3 model.
"""

from __future__ import annotations

from ..registry import MODELS
from .config import AlphaFold3Config, AlphaFold3ProcessorConfig
from .model import (
    AlphaFold3FoldResult,
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3Output,
    AlphaFold3SeedGroup,
)
from .processor import AlphaFold3Processor

__all__ = [
    "AlphaFold3Config",
    "AlphaFold3ProcessorConfig",
    "AlphaFold3Model",
    "AlphaFold3Processor",
    "AlphaFold3Input",
    "AlphaFold3Output",
    "AlphaFold3FoldResult",
    "AlphaFold3SeedGroup",
]

# ``fold`` returns the family-owned ``AlphaFold3FoldResult`` (seed groups preserved), which is
# not losslessly wrappable by the generic standard fold Pipeline, so no standard task is
# registered -- ``fold`` is called directly as a family method (frozen DESIGN 7.2).
MODELS.register(
    "ofoldx.alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
    standard_tasks=(),
)
