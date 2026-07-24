r"""
BoltzGen structure-design and inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    BoltzGenConfig,
    BoltzGenDenoiserConfig,
    BoltzGenInputEmbeddingsConfig,
    BoltzGenInverseFoldingConfig,
    BoltzGenInverseFoldingModuleConfig,
    BoltzGenSamplerConfig,
    BoltzGenStructureDesignConfig,
    BoltzGenTemplateConfig,
    BoltzGenTrunkConfig,
)
from .model import (
    BoltzGenBinding,
    BoltzGenContact,
    BoltzGenDesignCandidate,
    BoltzGenDesignMask,
    BoltzGenDesignResult,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingModel,
    BoltzGenInverseFoldingOutput,
    BoltzGenModel,
    BoltzGenOutput,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenSymmetry,
)
from .pipeline import BoltzGenFoldModel, BoltzGenPipeline, BoltzGenScoreModel, BoltzGenStageProcessor
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor
from .projection import DesignMaskResidue, from_design_mask, to_design_mask

_MODELS.register_model(
    "ofoldx.boltzgen",
    model=BoltzGenModel,
    processor=BoltzGenProcessor,
    processor_config=None,
    standard_tasks=(),
)
_MODELS.register_model(
    "ofoldx.boltzgen_ifold",
    model=BoltzGenInverseFoldingModel,
    processor=BoltzGenInverseFoldingProcessor,
    processor_config=None,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.boltzgen",
    pipeline=BoltzGenPipeline,
    tasks=("design",),
    repository_loader=BoltzGenPipeline.from_repository,
)
