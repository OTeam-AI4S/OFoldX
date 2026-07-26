r"""
BoltzGen structure-design, confidence, and inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    BoltzGenConfidenceConfig,
    BoltzGenConfidenceHeadConfig,
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
    BoltzGenConfidenceDesignOutput,
    BoltzGenConfidenceModel,
    BoltzGenConfidenceOutput,
    BoltzGenCovalentBond,
    BoltzGenDesignMask,
    BoltzGenDesignOutput,
    BoltzGenDirectionalContact,
    BoltzGenInput,
    BoltzGenInverseFoldingDesignOutput,
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
    BoltzGenStructureTarget,
    BoltzGenSymmetry,
)
from .objectives import (
    BoltzGenConfidenceLoss,
    BoltzGenDiffusionLoss,
    b_factor_loss,
    bond_loss,
    confidence_loss,
    diffusion_loss,
    distogram_loss,
    pae_loss,
    pae_regression_loss,
    pde_loss,
    pde_regression_loss,
    plddt_loss,
    plddt_regression_loss,
    residue_type_loss,
    resolved_loss,
    smooth_lddt_loss,
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
    "boltzgen",
    model=BoltzGenModel,
    processor=BoltzGenProcessor,
)
_MODELS.register(
    "boltzgen_confidence",
    model=BoltzGenConfidenceModel,
    processor=BoltzGenProcessor,
)
_MODELS.register(
    "boltzgen_ifold",
    model=BoltzGenInverseFoldingModel,
    processor=BoltzGenInverseFoldingProcessor,
)
_PIPELINES.register(
    "boltzgen",
    pipeline=BoltzGenPipeline,
)
