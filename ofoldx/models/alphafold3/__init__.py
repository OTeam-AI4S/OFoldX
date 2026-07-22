r"""AlphaFold 3 structure-prediction implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import AlphaFold3Config, AlphaFold3ProcessorConfig
from .model import (
    AlphaFold3FoldResult,
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3Output,
    AlphaFold3SeedGroup,
)
from .pipeline import AlphaFold3Pipeline
from .processor import AlphaFold3Processor

# Seed-group results require the concrete folding pipeline.
MODELS.register_model(
    "ofoldx.alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
    processor_config=AlphaFold3ProcessorConfig,
    standard_tasks=(),
)

PIPELINES.register_pipeline(
    "ofoldx.alphafold3",
    pipeline=AlphaFold3Pipeline,
    tasks=("fold",),
    architectures=("ofoldx.alphafold3",),
)
