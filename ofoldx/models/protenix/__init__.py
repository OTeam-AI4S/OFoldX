r"""
Protenix structure-prediction implementations.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixConstraintProcessorConfig,
    ProtenixEsmConfig,
    ProtenixProcessorConfig,
    ProtenixStructureConditionConfig,
    ProtenixTemplateConfig,
    ProtenixTemplateModelConfig,
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
    ProtenixNoisyStructureEmbedder,
    ProtenixOutput,
    ProtenixPairwiseDistancePotential,
    ProtenixPairwiseDistanceTerm,
    ProtenixPlanarImproperPotential,
    ProtenixPlanarImproperTerm,
    ProtenixPocket,
    ProtenixRawOutput,
    ProtenixStereoBondPotential,
    ProtenixStereoBondTerm,
    ProtenixStructureConditionModel,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTemplateModel,
    ProtenixTfgFeatures,
    ProtenixTfgGuidance,
    ProtenixTfgPotential,
    ProtenixTokenContact,
    ProtenixVinaStericPotential,
    ProtenixVinaStericTerm,
)
from .pipeline import (
    ProtenixConstraintPipeline,
    ProtenixPipeline,
    ProtenixStructureConditionPipeline,
    ProtenixTemplatePipeline,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
)

_MODELS.register(
    "protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
_MODELS.register(
    "protenix_template",
    model=ProtenixTemplateModel,
    processor=ProtenixTemplateProcessor,
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
_MODELS.register(
    "protenix_structure_condition",
    model=ProtenixStructureConditionModel,
    processor=ProtenixStructureConditionProcessor,
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
_MODELS.register(
    "protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
    criterion=denoising_criterion_factory,
    strategy="denoising",
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
    "protenix_structure_condition",
    pipeline=ProtenixStructureConditionPipeline,
)
_PIPELINES.register(
    "protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
)
