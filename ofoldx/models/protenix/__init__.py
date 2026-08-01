# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Protenix structure-prediction implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
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
    ProtenixPipelineInput,
    ProtenixStructureConditionPipeline,
    ProtenixTemplatePipeline,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
)

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
    model_architecture=ProtenixPipeline.model_architecture,
)
PIPELINES.register(
    "protenix_template",
    pipeline=ProtenixTemplatePipeline,
    model_architecture=ProtenixTemplatePipeline.model_architecture,
)
PIPELINES.register(
    "protenix_structure_condition",
    pipeline=ProtenixStructureConditionPipeline,
    model_architecture=ProtenixStructureConditionPipeline.model_architecture,
)
PIPELINES.register(
    "protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
    model_architecture=ProtenixConstraintPipeline.model_architecture,
)
