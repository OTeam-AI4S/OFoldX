r"""ESMFold2 architecture, processor, tensor contracts, and scientific modules."""

from ..registry import register_model as _register_model
from .config import (
    ESMFold2AtomAttentionConfig,
    ESMFold2ConfidenceHeadConfig,
    ESMFold2Config,
    ESMFold2DiffusionModuleConfig,
    ESMFold2ESMCConfig,
    ESMFold2FoldingTrunkConfig,
    ESMFold2InputsEmbedderConfig,
    ESMFold2LMEncoderConfig,
    ESMFold2MSAEncoderConfig,
    ESMFold2ParcaeConfig,
    ESMFold2ProcessorConfig,
    ESMFold2StructureHeadConfig,
)
from .model import (
    ConfidenceHead,
    DiffusionModule,
    DiffusionStructureHead,
    ESMFold2AtomDecoder,
    ESMFold2AtomEncoder,
    ESMFold2DesignCandidate,
    ESMFold2DesignResult,
    ESMFold2ExperimentalConfig,
    ESMFold2ExperimentalModel,
    ESMFold2ExperimentalProcessor,
    ESMFold2Input,
    ESMFold2Layout,
    ESMFold2Model,
    ESMFold2Output,
    FoldingTrunk,
    InputsEmbedder,
    LanguageModelPairEncoder,
    MSAEncoder,
)
from .pipeline import ESMFold2Pipeline
from .processor import ESMFold2Processor

_register_model(
    "ofoldx.esmfold2",
    model=ESMFold2Model,
    processor=ESMFold2Processor,
    standard_tasks={"fold"},
)

# The second architecture id `ofoldx.esmfold2_experimental` (the released binder-inversion
# notebook's `type="experimental"` recurrent graph, a different graph and confidence contract
# than `ESMFold2Model`) follows the two-arch-one-directory pattern of `ofoldx.alphafold2` /
# `ofoldx.alphafold2_multimer` and `ofoldx.protenix` / `ofoldx.protenix_constraint`. Its
# stub Config/Model/Processor are reserved in `experimental.py`, but its paired
# `_register_model` call is DEFERRED — the recurrent-graph checkpoints are not local and the
# graph is not implemented this wave. When those weights and the implementation land it is
# added here as ordinary explicit registration (DESIGN.md 7.2), never silently loaded as a
# variant of the released model:
#
#     _register_model(
#         "ofoldx.esmfold2_experimental",
#         model=ESMFold2ExperimentalModel,
#         processor=ESMFold2ExperimentalProcessor,
#         standard_tasks=(),
#     )
#
# The concrete `ESMFold2Pipeline` registration (`register_pipeline("ofoldx.esmfold2", ...)`
# + `pipeline.json` marker) and the built-in lazy-import index entry are likewise deferred
# until the ESMFold2/-Fast trunk and experimental binder weights and the published
# MultiMolecule ESMC asset land and `design` is fully implemented (Principle 13; README
# registry caution).
