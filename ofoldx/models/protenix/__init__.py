r"""
Protenix structure-prediction implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
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

_MODELS.register_model(
    "ofoldx.protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
    processor_config=ProtenixProcessorConfig,
    standard_tasks=("fold",),
)
_MODELS.register_model(
    "ofoldx.protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
    processor_config=ProtenixConstraintProcessorConfig,
    standard_tasks=("fold",),
)
_PIPELINES.register_pipeline(
    "ofoldx.protenix",
    pipeline=ProtenixPipeline,
    tasks=("fold",),
    architectures=("ofoldx.protenix",),
)
_PIPELINES.register_pipeline(
    "ofoldx.protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
    tasks=("fold",),
    architectures=("ofoldx.protenix_constraint",),
)
