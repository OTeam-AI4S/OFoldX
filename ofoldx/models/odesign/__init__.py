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
    ODesignLigandProcessorConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignDesignedResidue,
    ODesignFixedResidue,
    ODesignGeneratedAtoms,
    ODesignInput,
    ODesignLigandBackbone,
    ODesignLigandInput,
    ODesignLigandLayout,
    ODesignModel,
    ODesignNucleicAcidIdentity,
    ODesignOutput,
    ODesignPlacements,
    ODesignProvisionalAtomSlot,
    ODesignRawOutput,
    ODesignReferenceCoordinates,
    ODesignResidueSpec,
    ODesignSamplingOutput,
    ODesignStructureTarget,
    realize_nucleic_acid_identity,
)
from .pipeline import (
    ODesignGrnadeChainPrediction,
    ODesignLigandDesignCandidate,
    ODesignLigandDesignResult,
    ODesignLigandPipeline,
    ODesignMpnnChainPrediction,
    ODesignNucleicAcidPipeline,
    ODesignOInvFoldProteinPipeline,
    ODesignOInvFoldRnaPipeline,
    ODesignPolymerChainPrediction,
    ODesignPolymerDesignCandidate,
    ODesignPolymerDesignResult,
    ODesignProteinPipeline,
)
from .processor import ODesignLigandProcessor, ODesignProcessor, from_design_mask, to_design_mask

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
_PIPELINES.register(
    "odesign_protein",
    pipeline=ODesignProteinPipeline,
)
_PIPELINES.register(
    "odesign_nucleic_acid",
    pipeline=ODesignNucleicAcidPipeline,
)
_PIPELINES.register(
    "odesign_ligand",
    pipeline=ODesignLigandPipeline,
)
_PIPELINES.register(
    "odesign_oinvfold_protein",
    pipeline=ODesignOInvFoldProteinPipeline,
)
_PIPELINES.register(
    "odesign_oinvfold_rna",
    pipeline=ODesignOInvFoldRnaPipeline,
)
