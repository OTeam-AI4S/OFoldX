# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
AlphaFold 3 configuration.
"""

from __future__ import annotations

from chanfig import Config

from ...data.tokenization import TOKENS
from ..config import (
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
)


class AlphaFold3TemplateConfig(Config):
    r"""
    Configure the AlphaFold 3 template embedder.

    Attributes:
        hidden_size: Width of the template pair representation.
        num_blocks: Number of template pair-update blocks.
        vocab_size: Number of template residue classes (the canonical classes).
        num_distogram_bins: Number of template distance bins.
        triangle_attention_head_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Transition expansion factor.
    """

    hidden_size: int = 64
    num_blocks: int = 2
    vocab_size: int = len(TOKENS)
    num_distogram_bins: int = 39
    triangle_attention_head_size: int = 16
    num_triangle_attention_heads: int = 4
    expansion: int = 2


class AlphaFold3TrunkConfig(Config):
    r"""
    Configure the homogeneous AlphaFold 3 Pairformer trunk.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
    """

    num_pairformer_blocks: int = 48


class AlphaFold3ConfidenceConfig(Config):
    r"""
    Configure the AlphaFold 3 confidence head.

    Attributes:
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
        triangle_attention_head_size: Per-head width of triangle attention.
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
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 4


class AlphaFold3DistogramConfig(DistogramConfig):
    r"""Configure the distance-category breaks used by AlphaFold3 contact scoring."""

    first_break: float = 2.3125
    last_break: float = 21.6875


class AlphaFold3Config(Config):
    r"""
    Configure the [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model] graph.

    Attributes:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of raw atom features.
        relative_position_size: Width of relative-position features.
        template: Two-block template stack over the canonical classes with four 16-wide triangle heads.
        trunk: Pairformer trunk configuration.
        num_recycles: Number of complete trunk passes.
        msa_max_depth: Maximum MSA depth realized independently for each trunk pass.
        denoiser: Atom/token diffusion denoiser configuration.
        sampler: EDM reverse-diffusion sampler configuration.
        distogram: Distogram head configuration.
        confidence: Confidence head with single-representation clamping disabled for AlphaFold 3.
        rank_metric_eps: Denominator epsilon for the contact-weighted PAE ranking metric.

    Examples:
        >>> from ofoldx.models.alphafold3 import AlphaFold3Config
        >>> config = AlphaFold3Config(num_recycles=3)
        >>> config.num_recycles
        3
    """

    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    template: AlphaFold3TemplateConfig = AlphaFold3TemplateConfig()
    trunk: AlphaFold3TrunkConfig = AlphaFold3TrunkConfig()
    num_recycles: int = 11
    msa_max_depth: int = 1024
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: AlphaFold3DistogramConfig = AlphaFold3DistogramConfig()
    # Confidence scoring consumes the trunk single representation without magnitude clipping.
    confidence: AlphaFold3ConfidenceConfig = AlphaFold3ConfidenceConfig()
    rank_metric_eps: float = 1e-6


class AlphaFold3ProcessorConfig(Config):
    r"""Configure AlphaFold 3 input processing.

    Attributes:
        seed: Seed used to embed missing reference conformers.
    """

    seed: int = 42
