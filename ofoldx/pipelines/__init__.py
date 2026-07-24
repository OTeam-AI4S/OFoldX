r"""
Task pipelines and pipeline-authoring interfaces.
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
