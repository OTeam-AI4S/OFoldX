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
PxDesign protein-structure design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    PxDesignConditionEmbeddingsConfig,
    PxDesignConfig,
    PxDesignDenoiserConfig,
    PxDesignProcessorConfig,
    PxDesignSamplerConfig,
)
from .model import (
    PxDesignInput,
    PxDesignModel,
    PxDesignSamplingOutput,
)
from .pipeline import (
    PxDesignBackbonePipeline,
    PxDesignBackboneResult,
    PxDesignCandidate,
    PxDesignFilterResult,
    PxDesignPipeline,
    PxDesignPipelineInput,
    PxDesignPipelineOutput,
    PxDesignResult,
    generated_components,
)
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

MODELS.register(
    "pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
)
PIPELINES.register(
    "pxdesign_backbone",
    pipeline=PxDesignBackbonePipeline,
    model_architecture=PxDesignBackbonePipeline.model_architecture,
)
PIPELINES.register(
    "pxdesign",
    pipeline=PxDesignPipeline,
)
