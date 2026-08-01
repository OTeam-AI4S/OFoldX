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
BoltzGen structure design, inverse folding, analysis, and released composition.
"""

from __future__ import annotations

from ...analysis import AnalysisValue, CandidateFilter, CandidateRanking
from ...modules.diffusion.loss import bond_loss, smooth_lddt_loss
from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    BoltzGenConfidenceConfig,
    BoltzGenConfidenceHeadConfig,
    BoltzGenConfig,
    BoltzGenDenoiserConfig,
    BoltzGenInputEmbeddingsConfig,
    BoltzGenInverseFoldingConfig,
    BoltzGenInverseFoldingModuleConfig,
    BoltzGenMiniformerConfig,
    BoltzGenMiniformerResidueTypeConfig,
    BoltzGenProcessorConfig,
    BoltzGenResidueTypeConfig,
    BoltzGenSamplerConfig,
    BoltzGenStructureDesignConfig,
    BoltzGenTemplateConfig,
    BoltzGenTrunkConfig,
)
from .model import (
    BoltzGenBinding,
    BoltzGenConfidenceModel,
    BoltzGenConfidenceOutput,
    BoltzGenContact,
    BoltzGenDesignMask,
    BoltzGenInput,
    BoltzGenInverseFoldingDesignOutput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingModel,
    BoltzGenInverseFoldingOutput,
    BoltzGenMiniformerModel,
    BoltzGenMiniformerResidueTypeModel,
    BoltzGenModel,
    BoltzGenOutput,
    BoltzGenResidueTypeModel,
    BoltzGenResidueTypeOutput,
    BoltzGenResidueTypeSamplingOutput,
    BoltzGenSamplingOutput,
    BoltzGenSamplingWithConfidenceOutput,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenStructureTarget,
    BoltzGenSymmetry,
)
from .objective import (
    BoltzGenConfidenceLoss,
    BoltzGenDiffusionLoss,
    BoltzGenLoss,
    b_factor_loss,
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
)
from .pipeline import (
    BoltzGenAtomicAnalysis,
    BoltzGenBackbonePipeline,
    BoltzGenBackbonePipelineInput,
    BoltzGenCandidateAnalysis,
    BoltzGenCandidateResult,
    BoltzGenDesignCandidate,
    BoltzGenDesignCheckpoint,
    BoltzGenDesignedChainAnalysis,
    BoltzGenDesignResult,
    BoltzGenFilterOutcome,
    BoltzGenInverseFoldingPipeline,
    BoltzGenInverseFoldingPipelineInput,
    BoltzGenMiniformerPipeline,
    BoltzGenPipeline,
    BoltzGenPipelineInput,
    BoltzGenProtocol,
)
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor

MODELS.register(
    "boltzgen",
    model=BoltzGenModel,
    processor=BoltzGenProcessor,
)
MODELS.register(
    "boltzgen_confidence",
    model=BoltzGenConfidenceModel,
    processor=BoltzGenProcessor,
)
MODELS.register(
    "boltzgen_residue_type",
    model=BoltzGenResidueTypeModel,
    processor=BoltzGenProcessor,
)
MODELS.register(
    "boltzgen_miniformer",
    model=BoltzGenMiniformerModel,
    processor=BoltzGenProcessor,
)
MODELS.register(
    "boltzgen_miniformer_residue_type",
    model=BoltzGenMiniformerResidueTypeModel,
    processor=BoltzGenProcessor,
)
MODELS.register(
    "boltzgen_ifold",
    model=BoltzGenInverseFoldingModel,
    processor=BoltzGenInverseFoldingProcessor,
)
PIPELINES.register(
    "boltzgen_backbone",
    pipeline=BoltzGenBackbonePipeline,
    model_architecture=BoltzGenBackbonePipeline.model_architecture,
)
PIPELINES.register(
    "boltzgen_inverse_fold",
    pipeline=BoltzGenInverseFoldingPipeline,
    model_architecture=BoltzGenInverseFoldingPipeline.model_architecture,
)
PIPELINES.register(
    "boltzgen",
    pipeline=BoltzGenPipeline,
)
PIPELINES.register(
    "boltzgen_miniformer",
    pipeline=BoltzGenMiniformerPipeline,
    model_architecture=BoltzGenMiniformerPipeline.model_architecture,
)
