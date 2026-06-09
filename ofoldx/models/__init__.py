r"""
Public model package: architecture entry points, auto factories, and typed IO contracts.

Re-exports every registered architecture's model, config, input, and generator classes alongside the
[`AutoConfig`][ofoldx.models.auto.AutoConfig] / [`AutoModel`][ofoldx.models.auto.AutoModel]
factories and the shared typed output contracts, so downstream code imports them from a single namespace.
"""

from __future__ import annotations

from .alphafold2 import AlphaFold2Config, AlphaFold2Input, AlphaFold2Model
from .alphafold3 import AlphaFold3Config, AlphaFold3Input, AlphaFold3Model
from .auto import (
    AutoConfig,
    AutoModel,
    AutoModelForBindingAffinityPrediction,
    AutoModelForDesignGeneration,
    AutoModelForStructurePrediction,
)
from .boltz1 import Boltz1Config, Boltz1Input, Boltz1Model, Boltz1Output
from .boltz2 import Boltz2AffinityConfig, Boltz2AffinityModel, Boltz2Config, Boltz2Input, Boltz2Model
from .boltzgen import (
    BoltzGenAdherenceConfig,
    BoltzGenConfig,
    BoltzGenDesignGenerator,
    BoltzGenDiverseConfig,
    BoltzGenIFoldConfig,
    BoltzGenIFoldGenerator,
    BoltzGenInput,
    BoltzGenStructureTrainedSmallConfig,
)
from .contract import (
    BaseInput,
    BindingAffinityPredictionOutput,
    DesignCandidate,
    DesignGeneratorOutput,
    DesignOutput,
    StructurePredictionOutput,
)
from .esmfold2 import ESMFold2Config, ESMFold2Input, ESMFold2Model, ESMFold2Processor
from .ligandmpnn import LigandMPNNConfig, LigandMPNNDesignGenerator, LigandMPNNInput
from .mixin import ModelMixin, no_init_weights
from .odesign import ODesignConfig, ODesignDesignConfig, ODesignDesignGenerator, ODesignInput, ODesignModel
from .proteinmpnn import ProteinMPNNConfig, ProteinMPNNDesignGenerator, ProteinMPNNInput
from .protenix import (
    ProtenixConfig,
    ProtenixConstraintFeatures,
    ProtenixInput,
    ProtenixModel,
)
from .pxdesign import PxDesignConfig, PxDesignDesignGenerator, PxDesignInput
from .rfdiffusion import RfDiffusionConfig, RfDiffusionDesignGenerator, RfDiffusionInput

__all__ = [
    "AlphaFold2Config",
    "AlphaFold2Input",
    "AlphaFold2Model",
    "AlphaFold3Config",
    "AlphaFold3Input",
    "AlphaFold3Model",
    "AutoConfig",
    "AutoModel",
    "AutoModelForBindingAffinityPrediction",
    "AutoModelForDesignGeneration",
    "AutoModelForStructurePrediction",
    "BaseInput",
    "BindingAffinityPredictionOutput",
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
    "BoltzGenDesignGenerator",
    "BoltzGenDiverseConfig",
    "BoltzGenIFoldConfig",
    "BoltzGenIFoldGenerator",
    "BoltzGenInput",
    "BoltzGenStructureTrainedSmallConfig",
    "DesignCandidate",
    "DesignGeneratorOutput",
    "DesignOutput",
    "ESMFold2Config",
    "ESMFold2Input",
    "ESMFold2Model",
    "ESMFold2Processor",
    "LigandMPNNConfig",
    "LigandMPNNDesignGenerator",
    "LigandMPNNInput",
    "ModelMixin",
    "ODesignConfig",
    "ODesignDesignConfig",
    "ODesignDesignGenerator",
    "ODesignInput",
    "ODesignModel",
    "ProteinMPNNConfig",
    "ProteinMPNNDesignGenerator",
    "ProteinMPNNInput",
    "ProtenixConfig",
    "ProtenixConstraintFeatures",
    "ProtenixInput",
    "ProtenixModel",
    "PxDesignConfig",
    "PxDesignDesignGenerator",
    "PxDesignInput",
    "RfDiffusionConfig",
    "RfDiffusionDesignGenerator",
    "RfDiffusionInput",
    "StructurePredictionOutput",
    "no_init_weights",
]
