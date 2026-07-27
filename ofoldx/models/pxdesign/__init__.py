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
    PxDesignBackboneCandidate,
    PxDesignBackboneResult,
    PxDesignInput,
    PxDesignLayout,
    PxDesignModel,
    PxDesignOutput,
)
from .pipeline import (
    PxDesignBackbonePipeline,
    PxDesignCandidate,
    PxDesignFilterResult,
    PxDesignPipeline,
    PxDesignResult,
)
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

_MODELS.register(
    "pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
)
_PIPELINES.register(
    "pxdesign_backbone",
    pipeline=PxDesignBackbonePipeline,
)
_PIPELINES.register("pxdesign", pipeline=PxDesignPipeline)
