r"""
Task pipelines and pipeline-authoring interfaces.
"""

from __future__ import annotations

from .composite import CompositePair, CompositePipeline
from .generator import PipelineProcessor, resolve_generator
from .paired import PairedPipeline
from .registry import PIPELINES
