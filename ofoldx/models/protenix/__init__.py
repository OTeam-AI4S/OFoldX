r"""Protenix v0.5 default and constraint implementations."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixConstraintProcessorConfig,
    ProtenixProcessorConfig,
)
from .model import (
    ProtenixAtomContact,
    ProtenixConstraintModel,
    ProtenixInput,
    ProtenixModel,
    ProtenixOutput,
    ProtenixPocket,
    ProtenixTokenContact,
)
from .pipeline import ProtenixConstraintPipeline, ProtenixPipeline
from .processor import ProtenixConstraintProcessor, ProtenixProcessor

MODELS.register_model(
    "ofoldx.protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
    processor_config=ProtenixProcessorConfig,
    standard_tasks=("fold",),
)
MODELS.register_model(
    "ofoldx.protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
    processor_config=ProtenixConstraintProcessorConfig,
    standard_tasks=("fold",),
)
PIPELINES.register_pipeline(
    "ofoldx.protenix",
    pipeline=ProtenixPipeline,
    tasks=("fold",),
    architectures=("ofoldx.protenix",),
)
PIPELINES.register_pipeline(
    "ofoldx.protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
    tasks=("fold",),
    architectures=("ofoldx.protenix_constraint",),
)
