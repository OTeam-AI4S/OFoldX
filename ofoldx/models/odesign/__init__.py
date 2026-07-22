r"""ODesign biomolecular structure-design implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    ODesignConfig,
    ODesignConstraintConfig,
    ODesignDenoiserConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignDesignCandidate,
    ODesignDesignResult,
    ODesignInput,
    ODesignModel,
    ODesignOutput,
)
from .pipeline import ODesignPipeline
from .processor import ODesignProcessor

# Coordinate generation and child sequence designers require the concrete multi-pair pipeline.
MODELS.register_model(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
    processor_config=ODesignProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.odesign",
    pipeline=ODesignPipeline,
    tasks=("design",),
    repository_loader=ODesignPipeline.from_repository,
)
