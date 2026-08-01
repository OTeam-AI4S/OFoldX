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
RFdiffusion protein-backbone design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    RfDiffusionConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionSe3Config,
)
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_MASK_TOKEN_ID,
    RFDIFFUSION_RESIDUE_TYPE_INDEX,
    RFDIFFUSION_RESIDUE_TYPES,
    RFDIFFUSION_UNKNOWN_TOKEN_ID,
    RfDiffusionInput,
    RfDiffusionModel,
    RfDiffusionOutput,
    RfDiffusionPotentialInput,
    RfDiffusionSamplingOutput,
    RfDiffusionTrajectory,
    rfdiffusion_residue_token_id,
)
from .pipeline import (
    RfDiffusionDesignCandidate,
    RfDiffusionDesignResult,
    RfDiffusionPipeline,
    RfDiffusionPipelineInput,
)
from .processor import (
    RfDiffusionCoordination,
    RfDiffusionFlatBottomRadialPotential,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionProcessor,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
    from_contig_string,
    to_contig_string,
)

MODELS.register(
    "rfdiffusion",
    model=RfDiffusionModel,
    processor=RfDiffusionProcessor,
)

PIPELINES.register(
    "rfdiffusion",
    pipeline=RfDiffusionPipeline,
    model_architecture=RfDiffusionPipeline.model_architecture,
)
