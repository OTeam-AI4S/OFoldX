r"""
LigandMpnn protein/ligand sequence-design model.

Re-exports the public surface of the OFoldX-native LigandMpnn sequence family. Raw forward, sampling, and
the four probability operations return the shared carriers in
[`ofoldx.models.mpnn`][ofoldx.models.mpnn]; ``design`` returns the shared
[`DesignResult`][ofoldx.results.DesignResult]. The context-aware designer and its design-and-pack
composition are the frozen §4.8/§4.8.1 pair.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import LigandMpnnFeatures, LigandMpnnInput, LigandMpnnModel, tied_groups_from_ids
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

MODELS.register(
    "ofoldx.ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.ligandmpnn",
    pipeline=LigandMpnnPipeline,
)
