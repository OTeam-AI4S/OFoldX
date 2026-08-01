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
Boltz-1 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz1ConfidenceConfig,
    Boltz1Config,
    Boltz1DenoiserConfig,
    Boltz1InputEmbeddingsConfig,
    Boltz1ProcessorConfig,
    Boltz1SamplerConfig,
    Boltz1TrunkConfig,
)
from .model import (
    Boltz1ConfidenceOutput,
    Boltz1Input,
    Boltz1Model,
    Boltz1Output,
    Boltz1Pocket,
    Boltz1Representations,
    Boltz1SamplingOutput,
    Boltz1SamplingWithConfidenceOutput,
    Boltz1StructureTarget,
)
from .pipeline import Boltz1Pipeline
from .processor import Boltz1Processor

MODELS.register(
    "boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
)
PIPELINES.register(
    "boltz1",
    pipeline=Boltz1Pipeline,
    model_architecture=Boltz1Pipeline.model_architecture,
)
