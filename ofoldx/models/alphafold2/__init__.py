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

``standard_tasks`` is ``("fold",)`` on BOTH pairs: monomer and monomer-pTM and
AlphaFold-Multimer fold are published through the generic
[`FoldPipeline`][ofoldx.pipelines.FoldPipeline]. The Model owns the released
stochastic realization (per-sample MSA resample / BERT-mask / cluster / crop,
per-recycle resampling under ``resample_msa_in_recycling``, and ``num_samples``
ensemble draws off the injected root generator); the Processor lowers the raw
alignment pool; ``fold`` returns a canonical
[`FoldResult`][ofoldx.results.FoldResult]. No concrete family Pipeline is
registered -- the standard fold route binds Model and Processor directly (the
Boltz-1 pattern). Bit-exact OpenFold parity signoff is tracked separately in
``scripts/parity/alphafold2.py`` and runs on the GPU parity environment.
"""

from __future__ import annotations

from ..registry import MODELS
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
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    predicted_interface_tm_score,
)
from .processor import AlphaFold2MultimerProcessor, AlphaFold2Processor

__all__ = [
    # Configs
    "AlphaFold2Config",
    "AlphaFold2EvoformerConfig",
    "AlphaFold2EvoformerIterationConfig",
    "AlphaFold2HeadsConfig",
    "AlphaFold2InputEmbeddingConfig",
    "AlphaFold2MSAConfig",
    "AlphaFold2ProcessorConfig",
    "AlphaFold2RecyclingConfig",
    "AlphaFold2StructureModuleHeadConfig",
    "AlphaFold2TemplateConfig",
    "AlphaFold2MultimerConfig",
    "AlphaFold2MultimerHeadsConfig",
    "AlphaFold2MultimerInputEmbeddingConfig",
    "AlphaFold2MultimerProcessorConfig",
    "AlphaFold2MultimerTemplateConfig",
    # Models
    "AlphaFold2Model",
    "AlphaFold2MultimerModel",
    # Processors
    "AlphaFold2Processor",
    "AlphaFold2MultimerProcessor",
    # Input / output types
    "AlphaFold2Input",
    "AlphaFold2Output",
    "AlphaFold2MultimerInput",
    "AlphaFold2StructureModuleOutput",
    # Public helpers
    "predicted_interface_tm_score",
]

MODELS.register(
    "ofoldx.alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
    standard_tasks=("fold",),
)
MODELS.register(
    "ofoldx.alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
    standard_tasks=("fold",),
)
