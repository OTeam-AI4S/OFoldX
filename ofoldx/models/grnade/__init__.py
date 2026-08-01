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
gRNAde RNA inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import GrnadeConfig, GrnadeV032ProcessorConfig
from .model import GRNADE_VOCABULARY, GrnadeInput, GrnadeModel, GrnadeOutput, GrnadeSamplingOutput
from .objective import GrnadeLoss
from .pipeline import GrnadeDesignResult, GrnadePipeline, GrnadeV032Pipeline
from .processor import GrnadeProcessor, GrnadeV032Processor

MODELS.register(
    "grnade",
    model=GrnadeModel,
    processor=GrnadeProcessor,
)
PIPELINES.register(
    "grnade",
    pipeline=GrnadePipeline,
    model_architecture=GrnadePipeline.model_architecture,
)
MODELS.register(
    "grnade_v0_3_2",
    model=GrnadeModel,
    processor=GrnadeV032Processor,
)
PIPELINES.register(
    "grnade_v0_3_2",
    pipeline=GrnadeV032Pipeline,
    model_architecture=GrnadeV032Pipeline.model_architecture,
)
