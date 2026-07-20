r"""ESMFold2 architecture, processor, tensor contracts, and scientific modules."""

from ...pipelines.registry import register_pipeline as _register_pipeline
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
from .pipeline import ESMFold2Pipeline, cdr_distogram_iptm_proxy
from .processor import (
    ESMFold2ExperimentalProcessor,
    ESMFold2Processor,
    antibody_cdr_indices,
    antibody_cdr_prompt,
    is_antibody_sequence,
)

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

# One concrete Pipeline binds either architecture's pair and owns both released biological verbs:
# ``fold`` (the released ``ESMFold2Model`` pair) and the family-owned ``design`` binder-inversion verb
# (the experimental ``ESMFold2ExperimentalModel`` pair). Registering it closes the design
# repository->task loop: an ``ofoldx.esmfold2_experimental`` repo carrying the
# ``pipeline.json = {"pipeline": "ofoldx.esmfold2"}`` marker routes ``pipeline("design", model=dir)``
# here (a single class-owned pair, so the factory takes the one-pair (model, processor) call — no
# ``_PAIR_LAYOUT``). The released fold repos stay marker-less and route through the generic standard
# ``fold`` Pipeline (``standard_tasks={"fold"}``) exactly as before.
_register_pipeline(
    "ofoldx.esmfold2",
    pipeline=ESMFold2Pipeline,
)
