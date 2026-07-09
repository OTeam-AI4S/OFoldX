r"""
Public model package: architecture entry points, auto factories, and typed IO contracts.

Re-exports every registered architecture's model, config, input, and generator classes alongside the
[`AutoConfig`][ofoldx.models.auto.AutoConfig] / [`AutoModel`][ofoldx.models.auto.AutoModel]
factories and the shared typed output contracts, so downstream code imports them from a single namespace.
"""

from __future__ import annotations

from .alphafold2 import AlphaFold2Config, AlphaFold2Input, AlphaFold2Model
from .alphafold3 import AlphaFold3Config, AlphaFold3Input, AlphaFold3Model
from .auto import AutoConfig, AutoModel
from .boltz1 import Boltz1Config, Boltz1Input, Boltz1Model, Boltz1Output
from .boltz2 import Boltz2AffinityConfig, Boltz2AffinityModel, Boltz2Config, Boltz2Input, Boltz2Model
from .boltzgen import (
    BoltzGenAdherenceConfig,
    BoltzGenConfig,
    BoltzGenDesigner,
    BoltzGenDiverseConfig,
    BoltzGenIFoldConfig,
    BoltzGenInput,
    BoltzGenInverseFoldDesigner,
    BoltzGenStructureTrainedSmallConfig,
)
from .contract import (
    AllAtomDesignCandidate,
    BaseInput,
    BindingAffinityOutput,
    DesignCandidate,
    DesignOutput,
    SequenceDesignCandidate,
    StructureDesignCandidate,
    StructurePredictionOutput,
    register_pytree_dataclass,
)
from .esmfold2 import ESMFold2Config, ESMFold2Input, ESMFold2Model, ESMFold2Processor
from .ligandmpnn import LigandMPNNConfig, LigandMPNNDesigner, LigandMPNNInput
from .mixin import ModelMixin, no_init_weights
from .nampnn import NAMPNNConfig, NAMPNNDesigner, NAMPNNInput, NAMPNNProcessor
from .odesign import ODesignConfig, ODesignDesignConfig, ODesignDesigner, ODesignInput, ODesignModel
from .proteinmpnn import ProteinMPNNConfig, ProteinMPNNDesigner, ProteinMPNNInput
from .protenix import (
    ProtenixConfig,
    ProtenixConstraintFeatures,
    ProtenixInput,
    ProtenixModel,
)
from .pxdesign import PxDesignConfig, PxDesignDesigner, PxDesignInput
from .rfdiffusion import RfDiffusionConfig, RfDiffusionDesigner, RfDiffusionInput

__all__ = [
    "AlphaFold2Config",
    "AlphaFold2Input",
    "AlphaFold2Model",
    "AlphaFold3Config",
    "AlphaFold3Input",
    "AlphaFold3Model",
    "AutoConfig",
    "AutoModel",
    "BaseInput",
    "BindingAffinityOutput",
    "Boltz1Config",
    "Boltz1Input",
    "Boltz1Model",
    "Boltz1Output",
    "Boltz2AffinityConfig",
    "Boltz2AffinityModel",
    "Boltz2Config",
    "Boltz2Input",
    "Boltz2Model",
    "BoltzGenAdherenceConfig",
    "BoltzGenConfig",
    "BoltzGenDesigner",
    "BoltzGenDiverseConfig",
    "BoltzGenIFoldConfig",
    "BoltzGenInverseFoldDesigner",
    "BoltzGenInput",
    "BoltzGenStructureTrainedSmallConfig",
    "AllAtomDesignCandidate",
    "DesignCandidate",
    "DesignOutput",
    "SequenceDesignCandidate",
    "StructureDesignCandidate",
    "ESMFold2Config",
    "ESMFold2Input",
    "ESMFold2Model",
    "ESMFold2Processor",
    "LigandMPNNConfig",
    "LigandMPNNDesigner",
    "LigandMPNNInput",
    "ModelMixin",
    "NAMPNNConfig",
    "NAMPNNDesigner",
    "NAMPNNInput",
    "NAMPNNProcessor",
    "ODesignConfig",
    "ODesignDesignConfig",
    "ODesignDesigner",
    "ODesignInput",
    "ODesignModel",
    "ProteinMPNNConfig",
    "ProteinMPNNDesigner",
    "ProteinMPNNInput",
    "ProtenixConfig",
    "ProtenixConstraintFeatures",
    "ProtenixInput",
    "ProtenixModel",
    "PxDesignConfig",
    "PxDesignDesigner",
    "PxDesignInput",
    "RfDiffusionConfig",
    "RfDiffusionDesigner",
    "RfDiffusionInput",
    "StructurePredictionOutput",
    "no_init_weights",
    "register_pytree_dataclass",
]
