r"""Boltz-1 structure-prediction model, its canonical Processor, and its registration."""

from __future__ import annotations

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

__all__ = [
    "Boltz1Config",
    "Boltz1ConfidenceConfig",
    "Boltz1DiffusionConfig",
    "Boltz1InputEmbedderConfig",
    "Boltz1SamplerConfig",
    "Boltz1TrunkConfig",
    "Boltz1Model",
    "Boltz1Processor",
    "Boltz1Pipeline",
    "Boltz1Input",
    "Boltz1Output",
    "Boltz1Embeddings",
    "Boltz1Pocket",
]

# One registration story: the family resolves through the generic standard fold Pipeline for the common
# ``fold`` case. The concrete ``Boltz1Pipeline`` (family-owned pocket + Boltz-1x steering) is exported here
# but is not registered under this ID until the concrete-Pipeline registry route freezes (DESIGN Section 7.2).
MODELS.register(
    "ofoldx.boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
    standard_tasks=("fold",),
)
