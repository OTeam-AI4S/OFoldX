r"""
Protenix structure-prediction implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    ProtenixConfidenceConfig,
    ProtenixConfig,
    ProtenixConstraintConfig,
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
    ProtenixSamplingOutput,
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
    protenix_confidence_summary,
    protenix_contact_probability,
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
    protenix_token_layout,
)

__all__ = [
    "ProtenixAtomContact",
    "ProtenixChiralAtomPotential",
    "ProtenixChiralAtomTerm",
    "ProtenixConfidenceConfig",
    "ProtenixConfig",
    "ProtenixConstraintConfig",
    "ProtenixConstraintModel",
    "ProtenixConstraintPipeline",
    "ProtenixConstraintProcessor",
    "ProtenixEsmConfig",
    "ProtenixExperimentalTorsionPotential",
    "ProtenixExperimentalTorsionTerm",
    "ProtenixExponentialInterpolation",
    "ProtenixInput",
    "ProtenixInterchainBondPotential",
    "ProtenixInterchainBondTerm",
    "ProtenixLinearBondPotential",
    "ProtenixLinearBondTerm",
    "ProtenixModel",
    "ProtenixNoisyStructureEmbedder",
    "ProtenixOutput",
    "ProtenixPairwiseDistancePotential",
    "ProtenixPairwiseDistanceTerm",
    "ProtenixPipeline",
    "ProtenixPlanarImproperPotential",
    "ProtenixPlanarImproperTerm",
    "ProtenixPocket",
    "ProtenixProcessor",
    "ProtenixProcessorConfig",
    "ProtenixSamplingOutput",
    "ProtenixStereoBondPotential",
    "ProtenixStereoBondTerm",
    "ProtenixStructureConditionConfig",
    "ProtenixStructureConditionModel",
    "ProtenixStructureConditionPipeline",
    "ProtenixStructureConditionProcessor",
    "ProtenixStructureTarget",
    "ProtenixTemplateConfig",
    "ProtenixTemplateFeatures",
    "ProtenixTemplateModel",
    "ProtenixTemplateModelConfig",
    "ProtenixTemplatePipeline",
    "ProtenixTemplateProcessor",
    "ProtenixTfgFeatures",
    "ProtenixTfgGuidance",
    "ProtenixTfgPotential",
    "ProtenixTokenContact",
    "ProtenixTrunkConfig",
    "ProtenixVariant",
    "ProtenixVinaStericPotential",
    "ProtenixVinaStericTerm",
    "protenix_confidence_summary",
    "protenix_contact_probability",
    "protenix_token_layout",
]

MODELS.register(
    "protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
)
MODELS.register(
    "protenix_template",
    model=ProtenixTemplateModel,
    processor=ProtenixTemplateProcessor,
)
MODELS.register(
    "protenix_structure_condition",
    model=ProtenixStructureConditionModel,
    processor=ProtenixStructureConditionProcessor,
)
MODELS.register(
    "protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
)
PIPELINES.register(
    "protenix",
    pipeline=ProtenixPipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", ProtenixPipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", ProtenixModel.sample),
            PipelineExecutionStep("decode", "pipeline", ProtenixPipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "protenix_template",
    pipeline=ProtenixTemplatePipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", ProtenixTemplatePipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", ProtenixTemplateModel.sample),
            PipelineExecutionStep("decode", "pipeline", ProtenixTemplatePipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "protenix_structure_condition",
    pipeline=ProtenixStructureConditionPipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", ProtenixStructureConditionPipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", ProtenixStructureConditionModel.sample),
            PipelineExecutionStep("decode", "pipeline", ProtenixStructureConditionPipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", ProtenixConstraintPipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", ProtenixConstraintModel.sample),
            PipelineExecutionStep("decode", "pipeline", ProtenixConstraintPipeline.decode_fold),
        )
    },
)
