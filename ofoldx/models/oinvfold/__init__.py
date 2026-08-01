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
OInvFold polymer and ligand inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    OInvFoldLigandConfig,
    OInvFoldPolymerConfig,
    OInvFoldPolymerProcessorConfig,
)
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OInvFoldGraph,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldOutput,
    OInvFoldPolymerInput,
    OInvFoldPolymerModel,
)
from .objective import OInvFoldLoss
from .pipeline import (
    OInvFoldLigandCandidate,
    OInvFoldLigandPipeline,
    OInvFoldLigandResult,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerResult,
)
from .processor import (
    OInvFoldLigandProcessor,
    OInvFoldPolymerProcessor,
)

MODELS.register(
    "oinvfold",
    model=OInvFoldPolymerModel,
    processor=OInvFoldPolymerProcessor,
)
MODELS.register(
    "oinvfold_ligand",
    model=OInvFoldLigandModel,
    processor=OInvFoldLigandProcessor,
)
PIPELINES.register(
    "oinvfold",
    pipeline=OInvFoldPolymerPipeline,
    model_architecture=OInvFoldPolymerPipeline.model_architecture,
)
PIPELINES.register(
    "oinvfold_ligand",
    pipeline=OInvFoldLigandPipeline,
    model_architecture=OInvFoldLigandPipeline.model_architecture,
)
