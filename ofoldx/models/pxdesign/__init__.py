r"""PxDesign protein-structure design implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
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

# Coordinate-bearing results require the concrete design pipeline.
MODELS.register_model(
    "ofoldx.pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
    processor_config=PxDesignProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.pxdesign",
    pipeline=PxDesignPipeline,
    tasks=("design",),
    architectures=("ofoldx.pxdesign",),
)
