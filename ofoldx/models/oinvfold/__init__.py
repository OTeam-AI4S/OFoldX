r"""
OInvFold polymer and ligand inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    OInvFoldLigandConfig,
    OInvFoldPolymerConfig,
    OInvFoldPolymerProcessorConfig,
)
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OINVFOLD_POLYMER_TOKENS,
    OInvFoldGraph,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldOutput,
    OInvFoldPolymerInput,
    OInvFoldPolymerModel,
)
from .objectives import criterion_factory, masked_nll_criterion
from .pipeline import (
    OInvFoldLigandCandidate,
    OInvFoldLigandPipeline,
    OInvFoldLigandResult,
    OInvFoldPolymerCandidate,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerResult,
)
from .processor import (
    OInvFoldLigandProcessor,
    OInvFoldLigandSlots,
    OInvFoldPolymerProcessor,
)

_MODELS.register(
    "oinvfold",
    model=OInvFoldPolymerModel,
    processor=OInvFoldPolymerProcessor,
)
_MODELS.register(
    "oinvfold_ligand",
    model=OInvFoldLigandModel,
    processor=OInvFoldLigandProcessor,
)
_PIPELINES.register(
    "oinvfold",
    pipeline=OInvFoldPolymerPipeline,
)
_PIPELINES.register(
    "oinvfold_ligand",
    pipeline=OInvFoldLigandPipeline,
)
