r"""
ODesign biomolecular structure-design implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    ODesignConfig,
    ODesignConstraintEmbeddingsConfig,
    ODesignDenoiserConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignDesignCandidate,
    ODesignDesignedResidue,
    ODesignDesignResult,
    ODesignFixedResidue,
    ODesignInput,
    ODesignModel,
    ODesignOutput,
    ODesignPlacements,
    ODesignReferenceCoordinates,
    ODesignResidueSpec,
)
from .pipeline import ODesignPipeline
from .processor import ODesignProcessor, from_design_mask, to_design_mask

_MODELS.register_model(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
    processor_config=ODesignProcessorConfig,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.odesign",
    pipeline=ODesignPipeline,
    tasks=("design",),
    repository_loader=ODesignPipeline.from_repository,
)
