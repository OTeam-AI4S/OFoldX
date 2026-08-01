# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Open biomolecular structure models.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._version import __version__ as __version__  # noqa: F401
    from ._version import version
    from .api import pack, pipeline, score
    from .artifacts import save_model
    from .auto import (
        AutoModel,
        AutoPipelineForDesign,
        AutoPipelineForFold,
        AutoPipelineForPack,
        AutoPipelineForScore,
        AutoProcessor,
    )
    from .chemistry import PolymerSequence
    from .data import MolecularSystem, Structure
    from .model import Model
    from .models.conditions import Hotspot, SymbolicBond
    from .modules.backend import Backend
    from .msa import MSA
    from .pipelines import Pipeline
    from .processor import Processor
    from .results import (
        AutoregressiveProbabilityResult,
        DesignAndPackResult,
        DesignCandidate,
        DesignResult,
        FoldPrediction,
        FoldResult,
        Metric,
        MetricReference,
        PackPrediction,
        PackResult,
        PairMetric,
        Score,
        Scored,
        ScoreResult,
        best,
    )
    from .templates import Template


_EXPORTS = {
    "__version__": ("._version", "__version__"),
    "version": ("._version", "version"),
    "pack": (".api", "pack"),
    "pipeline": (".api", "pipeline"),
    "score": (".api", "score"),
    "save_model": (".artifacts", "save_model"),
    "AutoPipelineForFold": (".auto", "AutoPipelineForFold"),
    "AutoPipelineForDesign": (".auto", "AutoPipelineForDesign"),
    "AutoPipelineForScore": (".auto", "AutoPipelineForScore"),
    "AutoPipelineForPack": (".auto", "AutoPipelineForPack"),
    "AutoModel": (".auto", "AutoModel"),
    "AutoProcessor": (".auto", "AutoProcessor"),
    "MSA": (".msa", "MSA"),
    "Hotspot": (".models.conditions", "Hotspot"),
    "MolecularSystem": (".data", "MolecularSystem"),
    "PolymerSequence": (".chemistry", "PolymerSequence"),
    "Structure": (".data", "Structure"),
    "SymbolicBond": (".models.conditions", "SymbolicBond"),
    "Template": (".templates", "Template"),
    "Pipeline": (".pipelines", "Pipeline"),
    "Model": (".model", "Model"),
    "Backend": (".modules.backend", "Backend"),
    "Processor": (".processor", "Processor"),
    "AutoregressiveProbabilityResult": (".results", "AutoregressiveProbabilityResult"),
    "DesignAndPackResult": (".results", "DesignAndPackResult"),
    "DesignCandidate": (".results", "DesignCandidate"),
    "DesignResult": (".results", "DesignResult"),
    "FoldPrediction": (".results", "FoldPrediction"),
    "FoldResult": (".results", "FoldResult"),
    "Metric": (".results", "Metric"),
    "MetricReference": (".results", "MetricReference"),
    "PackPrediction": (".results", "PackPrediction"),
    "PackResult": (".results", "PackResult"),
    "PairMetric": (".results", "PairMetric"),
    "Score": (".results", "Score"),
    "Scored": (".results", "Scored"),
    "ScoreResult": (".results", "ScoreResult"),
    "best": (".results", "best"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
