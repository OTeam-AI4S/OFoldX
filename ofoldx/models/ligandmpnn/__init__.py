r"""LigandMPNN protein-sequence design implementation."""

from __future__ import annotations

from typing import cast

from ...modules.mpnn import tied_groups_from_ids
from ...pipelines.registry import PIPELINES
from ..registry import MODELS, Task
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import (
    LigandMpnnFeatures,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnOutput,
    LigandMpnnSamplingOutput,
    sidechain_atom_types,
)
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

MODELS.register_model(
    "ofoldx.ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
    processor_config=LigandMpnnProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.ligandmpnn",
    pipeline=LigandMpnnPipeline,
    tasks=cast(tuple[Task], ("design",)),
    architectures=("ofoldx.ligandmpnn",),
)
