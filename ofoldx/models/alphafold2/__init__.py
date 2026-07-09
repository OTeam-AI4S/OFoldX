r"""
AlphaFold 2 model family.
"""

from __future__ import annotations

from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerConfig,
    AlphaFold2EvoformerIterationConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InputEmbeddingConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleHeadConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2Representations,
    AlphaFold2StructureModuleOutput,
)
from .processor import AlphaFold2Processor

__all__ = [
    "AlphaFold2Config",
    "AlphaFold2EvoformerConfig",
    "AlphaFold2EvoformerIterationConfig",
    "AlphaFold2HeadsConfig",
    "AlphaFold2InputEmbeddingConfig",
    "AlphaFold2Input",
    "AlphaFold2Model",
    "AlphaFold2Processor",
    "AlphaFold2RecyclingConfig",
    "AlphaFold2Representations",
    "AlphaFold2StructureModuleHeadConfig",
    "AlphaFold2StructureModuleOutput",
    "AlphaFold2TemplateConfig",
]
