r"""The four explicit standard task Pipeline classes."""

from __future__ import annotations

from ._generator import resolve_generator
from .design import DesignPipeline
from .fold import FoldPipeline
from .pack import PackPipeline
from .registry import PIPELINES
from .score import ScorePipeline
