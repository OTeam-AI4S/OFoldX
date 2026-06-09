r"""
Structure-to-model-input processors.

This package holds only cross-family processing machinery: the processor registry and
``AutoProcessor`` dispatch, the ``ProcessorMixin`` base, and the shared featurizers (structure
tokenization, MSA, templates). Model-specific processors live in ``ofoldx.models.<family>.processor``
and self-register with :data:`PROCESSORS`; resolve them via [`AutoProcessor`][] or import them
directly from their model package.

Examples:
    >>> from ofoldx.processors import AutoProcessor
    >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
    >>> model_input = processor("complex.cif")  # doctest: +SKIP
"""

from __future__ import annotations

from .auto import AutoProcessor
from .mixin import ProcessorMixin
from .msa import ParsedMsa, attach_msa, featurize_msa, parse_a3m, parse_csv, parse_msa
from .msa_tools import (
    Hmmalign,
    Hmmbuild,
    Hmmsearch,
    Jackhmmer,
    MsaTool,
    MsaToolError,
    MsaToolResult,
    Nhmmer,
    run_msa_tools,
)
from .registry import PROCESSORS
from .structure import StructureProcessor
from .template import TemplateFilterConfig, TemplateHit, filter_hits, parse_hmmsearch_hits, search_templates
from .utils import StructuralFields, StructureInput, StructureToken, structure_from_input, tokenize_structure

__all__ = [
    "attach_msa",
    "AutoProcessor",
    "featurize_msa",
    "filter_hits",
    "Hmmalign",
    "Hmmbuild",
    "Hmmsearch",
    "Jackhmmer",
    "MsaTool",
    "MsaToolError",
    "MsaToolResult",
    "Nhmmer",
    "parse_a3m",
    "parse_csv",
    "parse_hmmsearch_hits",
    "parse_msa",
    "ParsedMsa",
    "ProcessorMixin",
    "PROCESSORS",
    "run_msa_tools",
    "search_templates",
    "StructuralFields",
    "structure_from_input",
    "StructureInput",
    "StructureProcessor",
    "StructureToken",
    "TemplateFilterConfig",
    "TemplateHit",
    "tokenize_structure",
]
