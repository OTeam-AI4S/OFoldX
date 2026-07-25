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

_MODELS.register(
    "ofoldx.protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
)
_MODELS.register(
    "ofoldx.protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
)
_PIPELINES.register(
    "ofoldx.protenix",
    pipeline=ProtenixPipeline,
)
_PIPELINES.register(
    "ofoldx.protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
)
