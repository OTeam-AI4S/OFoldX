r"""
ODesign biomolecular structure-design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
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
    ODesignBackboneComponent,
    ODesignDnaPipeline,
    ODesignLigandBackboneComponent,
    ODesignLigandBackboneSample,
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
    "ODesignBackboneComponent",
    "ODesignConfig",
    "ODesignConstraintEmbeddingsConfig",
    "ODesignDenoiserConfig",
    "ODesignDnaPipeline",
    "ODesignInput",
    "ODesignLigandBackboneComponent",
    "ODesignLigandBackboneSample",
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

MODELS.register(
    "odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
)
MODELS.register(
    "odesign_ligand",
    model=ODesignModel,
    processor=ODesignLigandProcessor,
)
PIPELINES.register(
    "odesign_protein",
    pipeline=ODesignProteinPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignProteinPipeline.design),)},
)
PIPELINES.register(
    "odesign_dna",
    pipeline=ODesignDnaPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignDnaPipeline.design),)},
)
PIPELINES.register(
    "odesign_rna",
    pipeline=ODesignRnaPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignRnaPipeline.design),)},
)
PIPELINES.register(
    "odesign_ligand",
    pipeline=ODesignLigandPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignLigandPipeline.design),)},
)
PIPELINES.register(
    "odesign_oinvfold_protein",
    pipeline=ODesignOInvFoldProteinPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignOInvFoldProteinPipeline.design),)},
)
PIPELINES.register(
    "odesign_oinvfold_rna",
    pipeline=ODesignOInvFoldRnaPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ODesignOInvFoldRnaPipeline.design),)},
)
