r"""The four explicit standard task Pipeline classes and the public authoring surface.

The pipeline-authoring extension points are PUBLIC: an implementation's ``pipeline.py``
subclasses [`PairedPipeline`][] (single Model/Processor pair) or
[`CompositePipeline`][] (multi-pair, declaring a ``pair_layout`` of
[`CompositePair`][] slots), and resolves its RNG through [`resolve_generator`][]
(using the Processor's declared ``device``).
"""

from __future__ import annotations

from .composite import CompositePair, CompositePipeline
from .design import DesignPipeline
from .fold import FoldPipeline
from .generator import PipelineProcessor, resolve_generator
from .pack import PackPipeline
from .paired import PairedPipeline
from .registry import PIPELINES
from .score import ScorePipeline
