r"""
AlphaFold 3 configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..config import (
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
)


class AlphaFold3TemplateConfig(Config):
    r"""
    Configure the AlphaFold 3 template embedder.

    Args:
        hidden_size: Width of the template pair representation.
        num_blocks: Number of template pair-update blocks.
        vocab_size: Number of template token types.
        num_distogram_bins: Number of template distance bins.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Transition expansion factor.
    """

    hidden_size: int = 64
    num_blocks: int = 2
    vocab_size: int = 31
    num_distogram_bins: int = 39
    triangle_attention_hidden_size: int = 16
    num_triangle_attention_heads: int = 4
    expansion: int = 2


class AlphaFold3TrunkConfig(Config):
    r"""
    Configure the homogeneous AlphaFold 3 Pairformer trunk.

    Args:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
    """

    num_pairformer_blocks: int = 48


class AlphaFold3ConfidenceConfig(Config):
    r"""
    Configure the AlphaFold 3 confidence head.

    Args:
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-atom bins.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        distance_bin_start: Lower edge of the confidence distance grid.
        distance_bin_end: Upper edge of the confidence distance grid.
        distance_bin_step: Spacing between confidence distance bins.
        max_atoms_per_token: Maximum atoms represented for one token.
        single_max_abs: Optional clamp on single-representation magnitudes.
        num_blocks: Number of confidence Pairformer blocks.
        num_pair_attention_heads: Number of pair-biased attention heads.
        triangle_multiplication_hidden_size: Hidden width of triangle multiplication.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Transition expansion factor.
    """

    num_plddt_bins: int = 50
    num_resolved_bins: int = 2
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    distance_bin_start: float = 3.25
    distance_bin_end: float = 52.0
    distance_bin_step: float = 1.25
    max_atoms_per_token: int = 24
    single_max_abs: float | None = None
    num_blocks: int = 4
    num_pair_attention_heads: int = 16
    triangle_multiplication_hidden_size: int = 128
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 4


class AlphaFold3Config(Config):
    r"""
    Configure the [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model] graph.

    Args:
        token_feature_size: Width of raw token features.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of raw atom features.
        relative_position_size: Width of relative-position features.
        template: Two-block template stack with the 31-token AlphaFold 3 vocabulary and four 16-wide triangle heads.
        trunk: Pairformer trunk configuration.
        recycling_steps: Number of complete trunk passes.
        msa_max_depth: Maximum MSA depth realized independently for each trunk pass.
        denoiser: Atom/token diffusion denoiser configuration.
        sampler: EDM reverse-diffusion sampler configuration.
        distogram: Distogram head configuration.
        confidence: Confidence head with single-representation clamping disabled for AlphaFold 3.
        rank_metric_eps: Denominator epsilon for the contact-weighted PAE ranking metric.

    Examples:
        >>> from ofoldx.models.alphafold3 import AlphaFold3Config
        >>> config = AlphaFold3Config(recycling_steps=3)
        >>> config.recycling_steps
        3
    """

    token_feature_size: int = 447
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    template: AlphaFold3TemplateConfig = AlphaFold3TemplateConfig()
    trunk: AlphaFold3TrunkConfig = AlphaFold3TrunkConfig()
    recycling_steps: int = 11
    msa_max_depth: int = 1024
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: DistogramConfig = DistogramConfig()
    # Confidence scoring consumes the trunk single representation without magnitude clipping.
    confidence: AlphaFold3ConfidenceConfig = AlphaFold3ConfidenceConfig()
    rank_metric_eps: float = 1e-6


class AlphaFold3ProcessorConfig(Config):
    r"""
    Configure AlphaFold 3 input preprocessing.

    Args:
        max_msa_rows: Maximum assembled depth of paired and unpaired MSA rows.
        ccd_revision: wwPDB snapshot date used to reproducibly resolve bare CCD component codes.
    """

    max_msa_rows: int = 16384
    ccd_revision: str = "2022-09-28"
