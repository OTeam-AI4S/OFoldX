r"""
OInvFold polymer and ligand inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    OInvFoldConfig,
    OInvFoldLigandConfig,
    OInvFoldProcessorConfig,
)
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OINVFOLD_POLYMER_TOKENS,
    OInvFoldGraph,
    OInvFoldInput,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldModel,
    OInvFoldOutput,
)
from .objectives import criterion
from .pipeline import (
    OInvFoldCandidate,
    OInvFoldLigandCandidate,
    OInvFoldLigandPipeline,
    OInvFoldLigandResult,
    OInvFoldPipeline,
    OInvFoldResult,
)
from .processor import (
    OInvFoldLigandProcessor,
    OInvFoldProcessor,
)

__all__ = [
    "OINVFOLD_LIGAND_ELEMENTS",
    "OINVFOLD_POLYMER_TOKENS",
    "OInvFoldCandidate",
    "OInvFoldConfig",
    "OInvFoldGraph",
    "OInvFoldInput",
    "OInvFoldLigandCandidate",
    "OInvFoldLigandConfig",
    "OInvFoldLigandInput",
    "OInvFoldLigandModel",
    "OInvFoldLigandPipeline",
    "OInvFoldLigandProcessor",
    "OInvFoldLigandResult",
    "OInvFoldModel",
    "OInvFoldOutput",
    "OInvFoldPipeline",
    "OInvFoldProcessor",
    "OInvFoldProcessorConfig",
    "OInvFoldResult",
    "criterion",
]

_MODELS.register(
    "oinvfold",
    model=OInvFoldModel,
    processor=OInvFoldProcessor,
)
_MODELS.register(
    "oinvfold_ligand",
    model=OInvFoldLigandModel,
    processor=OInvFoldLigandProcessor,
)
_PIPELINES.register(
    "oinvfold",
    pipeline=OInvFoldPipeline,
)
_PIPELINES.register(
    "oinvfold_ligand",
    pipeline=OInvFoldLigandPipeline,
)
