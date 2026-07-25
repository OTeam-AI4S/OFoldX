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
    BoltzGenDesignMask,
    BoltzGenDesignOutput,
    BoltzGenDirectionalContact,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingDesignOutput,
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
from .pipeline import BoltzGenDesignCandidate, BoltzGenDesignResult, BoltzGenPipeline
from .processor import (
    BoltzGenInverseFoldingProcessor,
    BoltzGenProcessor,
    DesignMaskResidue,
    from_design_mask,
    to_design_mask,
)

_MODELS.register(
    "ofoldx.boltzgen",
    model=BoltzGenModel,
    processor=BoltzGenProcessor,
)
_MODELS.register(
    "ofoldx.boltzgen_ifold",
    model=BoltzGenInverseFoldingModel,
    processor=BoltzGenInverseFoldingProcessor,
)
_PIPELINES.register(
    "ofoldx.boltzgen",
    pipeline=BoltzGenPipeline,
)
