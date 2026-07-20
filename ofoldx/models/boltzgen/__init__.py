r"""BoltzGen structure-design and inverse-folding family, its Processors, Pipeline, and registration."""

from __future__ import annotations

from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
from .config import (
    BoltzGenConfig,
    BoltzGenDiffusionConfig,
    BoltzGenIFoldConfig,
    BoltzGenIFoldModuleConfig,
    BoltzGenInputEmbedderConfig,
    BoltzGenSamplerConfig,
    BoltzGenStructureDesignConfig,
    BoltzGenTemplateConfig,
    BoltzGenTrunkConfig,
)
from .convert import convert_boltzgen_checkpoint, convert_boltzgen_ifold_checkpoint
from .model import (
    BoltzGenBinding,
    BoltzGenCandidate,
    BoltzGenContact,
    BoltzGenDesignMask,
    BoltzGenIFoldInput,
    BoltzGenIFoldLayout,
    BoltzGenIFoldModel,
    BoltzGenIFoldOutput,
    BoltzGenInput,
    BoltzGenLayout,
    BoltzGenModel,
    BoltzGenOutput,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenResult,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenSymmetry,
    SymbolicResidueSite,
    symbolic_residue_aliases,
)
from .pipeline import BoltzGenPipeline
from .processor import BoltzGenIFoldProcessor, BoltzGenProcessor
from .projection import DesignMaskResidue, from_design_mask, to_design_mask

# Two registered pairs, one per architecture (their Processor call signature and input type
# differ, satisfying the §7 "separate arch only on signature/type difference" test). The three
# structure checkpoints (diverse / adherence / structuretrained-small) are weight variants of
# the one ``BoltzGenModel`` architecture — one arch, multiple checkpoints, not separate pairs.
#
# ``standard_tasks=()`` for both: BoltzGen design is a multi-stage family operation, not a
# single-Model standard verb. The ``design`` verb resolves through the concrete family
# ``BoltzGenPipeline``, a DESIGN §7.4 multi-pair Pipeline: the structure-design pair owns the
# runnable root, the inverse-fold pair lives at ``ifold/``, and the optional read-only boltz2
# ``fold/`` / ``affinity/`` stages are optional sub-pairs (declared in its ``_PAIR_LAYOUT``). It is
# registered below, so a marked repository's ``design`` routes through the factory's multi-pair
# loader (``from_repository``) instead of the one-pair ``(model, processor)`` call.
_register_model(
    "ofoldx.boltzgen",
    model=BoltzGenModel,
    processor=BoltzGenProcessor,
    standard_tasks=(),
)
_register_model(
    "ofoldx.boltzgen_ifold",
    model=BoltzGenIFoldModel,
    processor=BoltzGenIFoldProcessor,
    standard_tasks=(),
)
_register_pipeline(
    "ofoldx.boltzgen",
    pipeline=BoltzGenPipeline,
)
