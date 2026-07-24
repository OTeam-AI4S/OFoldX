r"""
PxDesign protein-structure design implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    PxDesignConditionEmbeddingsConfig,
    PxDesignConfig,
    PxDesignDenoiserConfig,
    PxDesignProcessorConfig,
    PxDesignSamplerConfig,
)
from .model import (
    PxDesignDesignCandidate,
    PxDesignDesignResult,
    PxDesignInput,
    PxDesignLayout,
    PxDesignModel,
    PxDesignOutput,
)
from .pipeline import PxDesignPipeline
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

_MODELS.register_model(
    "ofoldx.pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
    processor_config=PxDesignProcessorConfig,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.pxdesign",
    pipeline=PxDesignPipeline,
    tasks=("design",),
    architectures=("ofoldx.pxdesign",),
)
