r"""Boltz-1 structure-prediction implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz1ConfidenceConfig,
    Boltz1Config,
    Boltz1DiffusionConfig,
    Boltz1InputEmbedderConfig,
    Boltz1SamplerConfig,
    Boltz1TrunkConfig,
)
from .model import (
    Boltz1Embeddings,
    Boltz1Input,
    Boltz1Model,
    Boltz1Output,
    Boltz1Pocket,
)
from .pipeline import Boltz1Pipeline
from .processor import Boltz1Processor

MODELS.register_model(
    "ofoldx.boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
    processor_config=None,
    standard_tasks=("fold",),
)
PIPELINES.register_pipeline(
    "ofoldx.boltz1",
    pipeline=Boltz1Pipeline,
    tasks=("fold",),
    architectures=("ofoldx.boltz1",),
)
