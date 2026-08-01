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
NA-MPNN mixed-polymer inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    NaMpnnFeatureEncoder,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
)
from .objective import NaMpnnLoss, smoothed_loss
from .pipeline import NaMpnnDesignResult, NaMpnnPipeline, NaMpnnScorePipeline
from .processor import NaMpnnProcessor

MODELS.register(
    "nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
)
PIPELINES.register(
    "nampnn",
    pipeline=NaMpnnPipeline,
    model_architecture=NaMpnnPipeline.model_architecture,
)

PIPELINES.register(
    "nampnn_score",
    pipeline=NaMpnnScorePipeline,
    model_architecture=NaMpnnScorePipeline.model_architecture,
)
