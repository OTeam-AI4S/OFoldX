r"""
ODesign biomolecular structure-design implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    ODesignConfig,
    ODesignConstraintEmbeddingsConfig,
    ODesignDenoiserConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignInput,
    ODesignLigandInput,
    ODesignModel,
    ODesignOutput,
    ODesignRawOutput,
    ODesignSamplingOutput,
    ODesignStructureTarget,
)
from .objectives import (
    covalent_bond_distance_loss,
    distogram_loss,
    odesign_loss,
    smooth_lddt_loss,
    token_bond_loss,
    weighted_denoising_loss,
)
from .pipeline import (
    ODesignDnaPipeline,
    ODesignLigandCandidate,
    ODesignLigandPipeline,
    ODesignLigandResult,
    ODesignOInvFoldProteinPipeline,
    ODesignOInvFoldRnaPipeline,
    ODesignPolymerCandidate,
    ODesignPolymerResult,
    ODesignProteinPipeline,
    ODesignRnaPipeline,
)
from .processor import (
    ODesignLigandProcessor,
    ODesignProcessor,
    ODesignResidueSpec,
    from_design_mask,
    to_design_mask,
)

__all__ = [
    "ODesignConfig",
    "ODesignConstraintEmbeddingsConfig",
    "ODesignDenoiserConfig",
    "ODesignDnaPipeline",
    "ODesignInput",
    "ODesignLigandCandidate",
    "ODesignLigandInput",
    "ODesignLigandPipeline",
    "ODesignLigandProcessor",
    "ODesignLigandResult",
    "ODesignModel",
    "ODesignOInvFoldProteinPipeline",
    "ODesignOInvFoldRnaPipeline",
    "ODesignOutput",
    "ODesignPairwiseConfig",
    "ODesignPolymerCandidate",
    "ODesignPolymerResult",
    "ODesignProcessor",
    "ODesignProcessorConfig",
    "ODesignProteinPipeline",
    "ODesignRawOutput",
    "ODesignResidueSpec",
    "ODesignRnaPipeline",
    "ODesignSamplingOutput",
    "ODesignStructureTarget",
    "ODesignTrunkConfig",
    "covalent_bond_distance_loss",
    "distogram_loss",
    "from_design_mask",
    "odesign_loss",
    "smooth_lddt_loss",
    "to_design_mask",
    "token_bond_loss",
    "weighted_denoising_loss",
]

_MODELS.register(
    "odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
)
_MODELS.register(
    "odesign_ligand",
    model=ODesignModel,
    processor=ODesignLigandProcessor,
)
_PIPELINES.register("odesign_protein", pipeline=ODesignProteinPipeline)
_PIPELINES.register("odesign_dna", pipeline=ODesignDnaPipeline)
_PIPELINES.register("odesign_rna", pipeline=ODesignRnaPipeline)
_PIPELINES.register("odesign_ligand", pipeline=ODesignLigandPipeline)
_PIPELINES.register(
    "odesign_oinvfold_protein",
    pipeline=ODesignOInvFoldProteinPipeline,
)
_PIPELINES.register(
    "odesign_oinvfold_rna",
    pipeline=ODesignOInvFoldRnaPipeline,
)
