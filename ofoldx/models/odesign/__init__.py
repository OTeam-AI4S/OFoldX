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
    ODesignDesignedResidue,
    ODesignFixedResidue,
    ODesignInput,
    ODesignModel,
    ODesignOutput,
    ODesignPlacements,
    ODesignReferenceCoordinates,
    ODesignResidueSpec,
    ODesignSamplingOutput,
)
from .pipeline import ODesignDesignCandidate, ODesignDesignResult, ODesignPipeline
from .processor import ODesignProcessor, from_design_mask, to_design_mask

_MODELS.register(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
)
_PIPELINES.register(
    "ofoldx.odesign",
    pipeline=ODesignPipeline,
)
