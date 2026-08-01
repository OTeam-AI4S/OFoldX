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
ODesign biomolecular structure-design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    ODesignConfig,
    ODesignConstraintEmbeddingsConfig,
    ODesignDenoiserConfig,
    ODesignLigandProcessorConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignInput,
    ODesignLigandInput,
    ODesignModel,
    ODesignOutput,
    ODesignRawOutput,
    ODesignSamplingOutput,
    ODesignStructureTarget,
)
from .objective import (
    ODesignLoss,
    covalent_bond_distance_loss,
    distogram_loss,
    odesign_loss,
    smooth_lddt_loss,
    token_bond_loss,
    weighted_denoising_loss,
)
from .pipeline import (
    ODesignBackboneComponent,
    ODesignBackboneSample,
    ODesignDnaPipeline,
    ODesignLigandCandidate,
    ODesignLigandCoordinateComponent,
    ODesignLigandCoordinateSample,
    ODesignLigandPipeline,
    ODesignLigandPipelineInput,
    ODesignLigandResult,
    ODesignOInvFoldProteinPipeline,
    ODesignOInvFoldRnaPipeline,
    ODesignPolymerCandidate,
    ODesignPolymerPipelineInput,
    ODesignPolymerResult,
    ODesignProteinPipeline,
    ODesignRnaPipeline,
    ODesignUnboundLinker,
    designed_chains,
)
from .processor import (
    ODESIGN_GATING,
    ODESIGN_LAYOUT,
    ODESIGN_SYMMETRY,
    ODesignLigandProcessor,
    ODesignLigandTarget,
    ODesignProcessor,
    designable_components,
)

MODELS.register(
    "odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
)
MODELS.register(
    "odesign_ligand",
    model=ODesignModel,
    processor=ODesignLigandProcessor,
)
PIPELINES.register(
    "odesign_protein",
    pipeline=ODesignProteinPipeline,
)
PIPELINES.register(
    "odesign_dna",
    pipeline=ODesignDnaPipeline,
)
PIPELINES.register(
    "odesign_rna",
    pipeline=ODesignRnaPipeline,
)
PIPELINES.register(
    "odesign_ligand",
    pipeline=ODesignLigandPipeline,
)
PIPELINES.register(
    "odesign_oinvfold_protein",
    pipeline=ODesignOInvFoldProteinPipeline,
)
PIPELINES.register(
    "odesign_oinvfold_rna",
    pipeline=ODesignOInvFoldRnaPipeline,
)
