r"""
OInvFold polymer and ligand inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
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
from .objectives import criterion
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
    OInvFoldPolymerProcessor,
)

__all__ = [
    "OINVFOLD_LIGAND_ELEMENTS",
    "OINVFOLD_POLYMER_TOKENS",
    "OInvFoldGraph",
    "OInvFoldLigandCandidate",
    "OInvFoldLigandConfig",
    "OInvFoldLigandInput",
    "OInvFoldLigandModel",
    "OInvFoldLigandPipeline",
    "OInvFoldLigandProcessor",
    "OInvFoldLigandResult",
    "OInvFoldOutput",
    "OInvFoldPolymerCandidate",
    "OInvFoldPolymerConfig",
    "OInvFoldPolymerInput",
    "OInvFoldPolymerModel",
    "OInvFoldPolymerPipeline",
    "OInvFoldPolymerProcessor",
    "OInvFoldPolymerProcessorConfig",
    "OInvFoldPolymerResult",
    "criterion",
]

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
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", OInvFoldPolymerPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", OInvFoldPolymerModel.forward),
            PipelineExecutionStep("decode", "pipeline", OInvFoldPolymerPipeline.decode_design),
        )
    },
)
PIPELINES.register(
    "oinvfold_ligand",
    pipeline=OInvFoldLigandPipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", OInvFoldLigandPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", OInvFoldLigandModel.forward),
            PipelineExecutionStep("decode", "pipeline", OInvFoldLigandPipeline.decode_design),
        )
    },
)
