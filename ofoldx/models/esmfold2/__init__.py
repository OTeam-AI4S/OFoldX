r"""ESMFold2 architecture, processor, tensor contracts, and scientific modules."""

from ..registry import register_model as _register_model
from .config import (
    ESMFold2AtomAttentionConfig,
    ESMFold2ConfidenceHeadConfig,
    ESMFold2Config,
    ESMFold2DiffusionModuleConfig,
    ESMFold2ESMCConfig,
    ESMFold2ExperimentalConfidenceHeadConfig,
    ESMFold2ExperimentalConfig,
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
    ESMFold2ExperimentalConfidenceHead,
    ESMFold2ExperimentalModel,
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
from .processor import ESMFold2ExperimentalProcessor, ESMFold2Processor

_register_model(
    "ofoldx.esmfold2",
    model=ESMFold2Model,
    processor=ESMFold2Processor,
    standard_tasks={"fold"},
)

# The second architecture id `ofoldx.esmfold2_experimental` (the released binder-inversion
# notebook's `type="experimental"` recurrent graph, a genuinely different graph and
# confidence contract than `ESMFold2Model`) follows the two-arch-one-directory pattern of
# `ofoldx.alphafold2` / `ofoldx.alphafold2_multimer` and `ofoldx.protenix` /
# `ofoldx.protenix_constraint`. It carries no standard task: its published surface is the
# family-owned binder-design verb `ESMFold2Pipeline.design`, not a shared standard task.
_register_model(
    "ofoldx.esmfold2_experimental",
    model=ESMFold2ExperimentalModel,
    processor=ESMFold2ExperimentalProcessor,
    standard_tasks=(),
)
