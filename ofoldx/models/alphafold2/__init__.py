r"""
AlphaFold 2 model family.

One directory / one package exporting TWO registered architecture IDs that share
the Evoformer / extra-MSA / StructureModule trunk, its heads, and the AF2 geometry
utilities:

- ``ofoldx.alphafold2`` -- monomer and monomer-pTM;
- ``ofoldx.alphafold2_multimer`` -- AlphaFold-Multimer.

The two IDs exist because their Processor call signature and required input/output
type differ: the monomer Processor lowers one unbound protein instance with an
entity-bound ``MSA``; the multimer Processor lowers a multi-instance
``MolecularSystem`` with a paired ``MSACollection`` and requires the chain-identity
features (``chain_index`` / ``entity_index`` / ``symmetry_index``) the monomer
contract cannot express (Principle 7).

``standard_tasks`` is ``()`` on BOTH pairs today: the family is intentionally
absent from the public registry until the released stochastic MSA/recycling/
ensemble inference path and complete public-output parity land (see README).
``fold`` is the target task for both; it flips to ``("fold",)`` at the publication
gate together with real parity in ``scripts/parity/alphafold2.py``.
"""

from __future__ import annotations

from ..registry import register_model as _register_model
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerConfig,
    AlphaFold2EvoformerIterationConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InputEmbeddingConfig,
    AlphaFold2MSAConfig,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerHeadsConfig,
    AlphaFold2MultimerInputEmbeddingConfig,
    AlphaFold2MultimerProcessorConfig,
    AlphaFold2MultimerTemplateConfig,
    AlphaFold2ProcessorConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleHeadConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2Input,
    AlphaFold2Layout,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerLayout,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    predicted_interface_tm_score,
)
from .processor import AlphaFold2MultimerProcessor, AlphaFold2Processor

_register_model(
    "ofoldx.alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
    standard_tasks=(),
)
_register_model(
    "ofoldx.alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
    standard_tasks=(),
)
