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
Boltz-2 folding and affinity implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz2AffinityConfig,
    Boltz2AffinityHeadConfig,
    Boltz2ConfidenceConfig,
    Boltz2Config,
    Boltz2DenoiserConfig,
    Boltz2InputEmbeddingsConfig,
    Boltz2ProcessorConfig,
    Boltz2SamplerConfig,
    Boltz2TemplateConfig,
    Boltz2TrunkConfig,
)
from .model import (
    Boltz2AffinityModel,
    Boltz2AffinityOutput,
    Boltz2AffinityScoreOutput,
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Output,
    Boltz2Pocket,
    Boltz2Representations,
    Boltz2SamplingOutput,
    Boltz2StructureTarget,
    Boltz2Template,
)
from .pipeline import (
    Boltz2AffinityPipeline,
    Boltz2AffinityPipelineInput,
    Boltz2AffinityPipelineOutput,
    Boltz2AffinityResult,
    Boltz2AffinityScorePipeline,
    Boltz2Pipeline,
)
from .processor import Boltz2AffinityProcessor, Boltz2Processor, standardize_boltz2_affinity_smiles

MODELS.register(
    "boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
)
MODELS.register(
    "boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
)
PIPELINES.register(
    "boltz2",
    pipeline=Boltz2Pipeline,
    model_architecture=Boltz2Pipeline.model_architecture,
)
PIPELINES.register(
    "boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
)
PIPELINES.register(
    "boltz2_affinity_score",
    pipeline=Boltz2AffinityScorePipeline,
    model_architecture=Boltz2AffinityScorePipeline.model_architecture,
)
