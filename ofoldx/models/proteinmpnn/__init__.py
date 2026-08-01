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
ProteinMPNN inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import (
    ProteinMpnnCaFeatureEncoder,
    ProteinMpnnDesignOutput,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnOutput,
    ProteinMpnnSamplingOutput,
)
from .pipeline import (
    ProteinMpnnDesignAndPackPipeline,
    ProteinMpnnDesignAndPackPipelineInput,
    ProteinMpnnDesignResult,
    ProteinMpnnPipeline,
    ProteinMpnnScorePipeline,
)
from .processor import ProteinMpnnProcessor

MODELS.register(
    "proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
)
PIPELINES.register(
    "proteinmpnn",
    pipeline=ProteinMpnnPipeline,
    model_architecture=ProteinMpnnPipeline.model_architecture,
)
PIPELINES.register(
    "proteinmpnn_design_and_pack",
    pipeline=ProteinMpnnDesignAndPackPipeline,
)

PIPELINES.register(
    "proteinmpnn_score",
    pipeline=ProteinMpnnScorePipeline,
    model_architecture=ProteinMpnnScorePipeline.model_architecture,
)
