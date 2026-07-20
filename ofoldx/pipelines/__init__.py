r"""The four explicit standard task Pipeline classes and the public authoring surface.

The pipeline-authoring extension points are PUBLIC: a model family's ``pipeline.py``
subclasses :class:`PairedPipeline` (single Model/Processor pair) or
:class:`CompositePipeline` (multi-pair, declaring a ``_PAIR_LAYOUT`` of
:class:`CompositePair` slots), and resolves its RNG through :func:`resolve_generator`
(reading placement via :func:`inputs_device`).
"""

from __future__ import annotations

from ._composite import CompositePair, CompositePipeline
from ._paired import PairedPipeline
from .design import DesignPipeline
from .fold import FoldPipeline
from .generator import inputs_device, resolve_generator
from .pack import PackPipeline
from .registry import PIPELINES
from .score import ScorePipeline
