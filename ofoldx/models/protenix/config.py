r"""
Configuration for Protenix folding and constraint architectures.
"""

from __future__ import annotations

from chanfig import Config

from ..config import (
    ConfidenceConfig,
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
    TemplateConfig,
    TrunkConfig,
)


class ProtenixConfig(Config):
    r"""
    Configure the Protenix base or four-branch constraint graph.

    Args:
        token_feature_size: Width of raw token features.
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of raw atom features.
        relative_position_size: Width of relative-position features.
        constraints: Whether to build the jointly trained four-branch constraint variant.
        template: Template embedding configuration.
        trunk: Pairformer trunk configuration.
        recycling_steps: Number of trunk recycling passes.
        denoiser: Coordinate denoiser configuration.
        sampler: Reverse-diffusion sampler configuration.
        distogram: Distogram head configuration.
        confidence: Confidence head configuration.
    """

    token_feature_size: int = 449
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    constraints: bool = False
    template: TemplateConfig = TemplateConfig()
    trunk: TrunkConfig = TrunkConfig()
    recycling_steps: int = 10
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: DistogramConfig = DistogramConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()


class ProtenixConstraintConfig(ProtenixConfig):
    r"""
    Configure the four-branch Protenix constraint graph.

    Args:
        constraints: Whether to build the jointly trained four-branch constraint embedder; fixed to ``True`` here.
    """

    constraints: bool = True


class ProtenixProcessorConfig(Config):
    r"""
    Configure Protenix input preprocessing.

    Args:
        constraints: Whether the paired processor exposes constraint inputs.
        max_msa_rows: Maximum MSA depth included in model inputs.
        augment_reference_positions: Whether to randomly rotate and translate reference coordinates.
    """

    constraints: bool = False
    max_msa_rows: int = 16384
    augment_reference_positions: bool = True


class ProtenixConstraintProcessorConfig(ProtenixProcessorConfig):
    r"""
    Configure Protenix constraint preprocessing.

    Args:
        constraints: Whether constraint inputs are exposed; fixed to ``True`` here.
    """

    constraints: bool = True
