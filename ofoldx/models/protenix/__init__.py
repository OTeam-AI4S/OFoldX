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
    ProtenixEsmConfig,
    ProtenixProcessorConfig,
    ProtenixTemplateConfig,
    ProtenixTrunkConfig,
    ProtenixVariant,
)
from .model import (
    ProtenixAtomContact,
    ProtenixChiralAtomPotential,
    ProtenixChiralAtomTerm,
    ProtenixConstraintModel,
    ProtenixExperimentalTorsionPotential,
    ProtenixExperimentalTorsionTerm,
    ProtenixExponentialInterpolation,
    ProtenixInput,
    ProtenixInterchainBondPotential,
    ProtenixInterchainBondTerm,
    ProtenixLinearBondPotential,
    ProtenixLinearBondTerm,
    ProtenixModel,
    ProtenixOutput,
    ProtenixPairwiseDistancePotential,
    ProtenixPairwiseDistanceTerm,
    ProtenixPlanarImproperPotential,
    ProtenixPlanarImproperTerm,
    ProtenixPocket,
    ProtenixRawOutput,
    ProtenixStereoBondPotential,
    ProtenixStereoBondTerm,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTfgFeatures,
    ProtenixTfgGuidance,
    ProtenixTfgPotential,
    ProtenixTokenContact,
    ProtenixVinaStericPotential,
    ProtenixVinaStericTerm,
)
from .pipeline import ProtenixConstraintPipeline, ProtenixPipeline, ProtenixTemplatePipeline
from .processor import ProtenixConstraintProcessor, ProtenixProcessor, ProtenixTemplateProcessor

_MODELS.register(
    "protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
)
_MODELS.register(
    "protenix_template",
    model=ProtenixModel,
    processor=ProtenixTemplateProcessor,
)
_MODELS.register(
    "protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
)
_PIPELINES.register(
    "protenix",
    pipeline=ProtenixPipeline,
)
_PIPELINES.register(
    "protenix_template",
    pipeline=ProtenixTemplatePipeline,
)
_PIPELINES.register(
    "protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
)
