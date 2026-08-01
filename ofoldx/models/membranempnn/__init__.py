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

r"""Membrane-label-conditioned MPNN sequence design."""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..proteinmpnn.model import ProteinMpnnInput, ProteinMpnnOutput, ProteinMpnnSamplingOutput
from ..registry import MODELS
from .config import MembraneLabelMode, MembraneMpnnConfig, MembraneMpnnProcessorConfig
from .model import MembraneMpnnModel
from .pipeline import (
    MembraneMpnnDesignAndPackPipeline,
    MembraneMpnnDesignResult,
    MembraneMpnnPipeline,
    MembraneMpnnScorePipeline,
)
from .processor import MembraneMpnnProcessor

MODELS.register("membranempnn", model=MembraneMpnnModel, processor=MembraneMpnnProcessor)
PIPELINES.register(
    "membranempnn",
    pipeline=MembraneMpnnPipeline,
    model_architecture=MembraneMpnnPipeline.model_architecture,
)
PIPELINES.register(
    "membranempnn_design_and_pack",
    pipeline=MembraneMpnnDesignAndPackPipeline,
)

PIPELINES.register(
    "membranempnn_score",
    pipeline=MembraneMpnnScorePipeline,
    model_architecture=MembraneMpnnScorePipeline.model_architecture,
)
