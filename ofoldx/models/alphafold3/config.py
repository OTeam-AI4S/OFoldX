r"""
AlphaFold 3 configuration.
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


class AlphaFold3Config(Config):
    r"""Configure the [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model] graph.

    Args:
        token_feature_size: Width of raw token features.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of raw atom features.
        relative_position_size: Width of relative-position features.
        template: Two-block template stack with the 31-token AlphaFold 3 vocabulary and four 16-wide triangle heads.
        trunk: Pairformer trunk configuration.
        recycling_steps: Number of trunk recycling passes.
        denoiser: Atom/token diffusion denoiser configuration.
        sampler: EDM reverse-diffusion sampler configuration.
        distogram: Distogram head configuration.
        confidence: Confidence head with single-representation clamping disabled for AlphaFold 3.
        rank_metric_eps: Denominator epsilon for the contact-weighted PAE ranking metric.
    """

    token_feature_size: int = 447
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    template: TemplateConfig = TemplateConfig(
        num_blocks=2,
        vocab_size=31,
        triangle_attention_hidden_size=16,
        num_triangle_attention_heads=4,
    )
    trunk: TrunkConfig = TrunkConfig()
    recycling_steps: int = 4
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: DistogramConfig = DistogramConfig()
    # Confidence scoring consumes the trunk single representation without magnitude clipping.
    confidence: ConfidenceConfig = ConfidenceConfig(single_max_abs=None)
    rank_metric_eps: float = 1e-6


class AlphaFold3ProcessorConfig(Config):
    r"""Configure AlphaFold 3 input lowering.

    Args:
        max_msa_rows: Maximum assembled depth of paired and unpaired MSA rows.
        ccd_revision: wwPDB snapshot date used to reproducibly resolve bare CCD component codes.
    """

    max_msa_rows: int = 16384
    ccd_revision: str = "2022-09-28"
