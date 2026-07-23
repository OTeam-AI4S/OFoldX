r"""Boltz-1 structure-prediction implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz1ConfidenceConfig,
    Boltz1Config,
    Boltz1DenoiserConfig,
    Boltz1InputEmbeddingsConfig,
    Boltz1SamplerConfig,
    Boltz1TrunkConfig,
)
from .model import (
    Boltz1Input,
    Boltz1Model,
    Boltz1Output,
    Boltz1Pocket,
    Boltz1Representations,
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
