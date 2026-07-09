r"""
RfDiffusion design-generator configuration.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import (
    RfDiffusionConfig,
    RfDiffusionContigMapConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionInferenceConfig,
    RfDiffusionLoggingConfig,
    RfDiffusionModelConfig,
    RfDiffusionPotentialsConfig,
    RfDiffusionPPIConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionScaffoldGuidedConfig,
    RfDiffusionSE3Config,
    RfDiffusionTopKSE3Config,
)
from .model import (
    ContigMap,
    Denoiser,
    Diffuser,
    EuclideanDiffuser,
    IGSO3Diffuser,
    NativeRfDiffusionSampler,
    PotentialManager,
    RfDiffusionDesigner,
    RfDiffusionInput,
    RfDiffusionRepresentations,
    RoseTTAFoldModule,
    SymmetryGenerator,
    coords_to_t2d,
    get_beta_schedule,
)
from .processor import RfDiffusionProcessor

_CONVERT_EXPORTS = {
    "config_from_rfdiffusion_checkpoint",
    "convert_rfdiffusion_checkpoint",
    "convert_rfdiffusion_state_dict",
    "load_rfdiffusion_checkpoint",
}

# PDB readers, parsing, and featurization all live in the processor; resolved lazily so importing this
# package stays free of an import cycle through the processor's heavier dependencies.
_PROCESSOR_EXPORTS = {"parse_pdb", "parse_pdb_lines", "process_target"}

__all__ = [
    "RfDiffusionConfig",
    "RfDiffusionContigMapConfig",
    "RfDiffusionDenoiserConfig",
    "RfDiffusionDiffuserConfig",
    "RfDiffusionDesigner",
    "RfDiffusionInferenceConfig",
    "RfDiffusionInput",
    "RfDiffusionLoggingConfig",
    "RfDiffusionModelConfig",
    "RfDiffusionPPIConfig",
    "RfDiffusionPotentialsConfig",
    "RfDiffusionPreprocessConfig",
    "RfDiffusionProcessor",
    "RfDiffusionRepresentations",
    "RfDiffusionSE3Config",
    "RfDiffusionScaffoldGuidedConfig",
    "RfDiffusionTopKSE3Config",
    "RoseTTAFoldModule",
    "ContigMap",
    "Denoiser",
    "Diffuser",
    "EuclideanDiffuser",
    "IGSO3Diffuser",
    "NativeRfDiffusionSampler",
    "PotentialManager",
    "SymmetryGenerator",
    "config_from_rfdiffusion_checkpoint",
    "convert_rfdiffusion_checkpoint",
    "convert_rfdiffusion_state_dict",
    "get_beta_schedule",
    "load_rfdiffusion_checkpoint",
    "parse_pdb",
    "parse_pdb_lines",
    "process_target",
    "coords_to_t2d",
]


def __getattr__(name: str) -> Any:
    if name in _CONVERT_EXPORTS:
        return getattr(import_module(f"{__name__}.convert"), name)
    if name in _PROCESSOR_EXPORTS:
        return getattr(import_module(f"{__name__}.processor"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
