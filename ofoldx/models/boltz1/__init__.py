r"""
Boltz-1 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
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

_MODELS.register(
    "ofoldx.boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
)
_PIPELINES.register(
    "ofoldx.boltz1",
    pipeline=Boltz1Pipeline,
)
