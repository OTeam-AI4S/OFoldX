r"""
Configuration for Protenix folding and constraint architectures.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config

from ...data import TOKEN_VOCAB_SIZE
from ..config import (
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
)

ProtenixVariant = Literal[
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_mini_tmpl_v0.5.0",
    "protenix_tiny_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
]


class ProtenixTrunkConfig(Config):
    r"""
    Configure the exact MSA and Pairformer axes of a Protenix trunk.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
        pairformer_dropout: Dropout probability applied inside each Pairformer block.
        msa_dropout: Dropout probability applied along the MSA path.
        pair_dropout: Dropout probability applied along the pair path.
        msa_size: Width of the projected MSA representation.
        num_msa_stages: Number of MSA stages, including the final pair-only update.
        msa_outer_product_hidden_size: Hidden width of the MSA outer-product mean.
        num_msa_attention_heads: Number of heads in MSA pair-weighted averaging.
        msa_attention_head_dim: Per-head value width in MSA pair-weighted averaging. ``None`` derives it from
            ``msa_size`` and ``num_msa_attention_heads``.
        msa_triangle_multiplication_hidden_size: Hidden width of MSA-path triangle multiplication.
        msa_triangle_attention_hidden_size: Per-head width of MSA-path triangle attention.
        num_msa_triangle_attention_heads: Number of MSA-path triangle-attention heads.
        msa_expansion: Feed-forward expansion factor inside the MSA stack.
        pairformer_triangle_multiplication_hidden_size: Hidden width of Pairformer triangle multiplication.
        pairformer_triangle_attention_hidden_size: Per-head width of Pairformer triangle attention.
        num_pairformer_triangle_attention_heads: Number of Pairformer triangle-attention heads.
        num_pair_attention_heads: Number of heads in pair-biased single attention.
        pairformer_expansion: Feed-forward expansion factor inside Pairformer blocks.
    """

    num_pairformer_blocks: int = 48
    pairformer_dropout: float = 0.25
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    msa_size: int = 64
    num_msa_stages: int = 4
    msa_outer_product_hidden_size: int = 32
    num_msa_attention_heads: int = 8
    msa_attention_head_dim: int | None = None
    msa_triangle_multiplication_hidden_size: int = 128
    msa_triangle_attention_hidden_size: int = 32
    num_msa_triangle_attention_heads: int = 4
    msa_expansion: int = 4
    pairformer_triangle_multiplication_hidden_size: int = 128
    pairformer_triangle_attention_hidden_size: int = 32
    num_pairformer_triangle_attention_heads: int = 4
    num_pair_attention_heads: int = 16
    pairformer_expansion: int = 4


class ProtenixTemplateConfig(Config):
    r"""
    Configure the released Protenix template graph used by base v1 and v2.

    Attributes:
        hidden_size: Width of the template pair embedding.
        num_blocks: Number of template Pairformer blocks.
        vocab_size: Number of template token types.
        num_distogram_bins: Number of template distogram distance bins.
        triangle_multiplication_hidden_size: Hidden width of template triangle multiplication.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Feed-forward expansion factor inside the template transition.
        dropout: Dropout probability applied inside the template stack.

    Examples:
        >>> from ofoldx.models.protenix import ProtenixTemplateConfig
        >>> ProtenixTemplateConfig().vocab_size
        38
    """

    hidden_size: int = 64
    num_blocks: int = 2
    vocab_size: int = TOKEN_VOCAB_SIZE
    num_distogram_bins: int = 39
    triangle_multiplication_hidden_size: int = 128
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 2
    dropout: float = 0.25


class ProtenixEsmConfig(Config):
    r"""
    Configure the frozen ESM2 t36/3B encoder used by mini ESM and mini ISM.

    ESM and ISM use this same graph and differ only in packaged weights.
    """

    vocab_size: int = 33
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 40
    intermediate_size: int = 10240
    max_position_embeddings: int = 1026
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    pad_token_id: int = 1
    mask_token_id: int = 32
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: Literal["rotary"] = "rotary"
    emb_layer_norm_before: bool = False
    token_dropout: bool = True


class ProtenixConfidenceConfig(Config):
    r"""
    Configure Protenix confidence metrics and their Pairformer trunk.

    Attributes:
        num_plddt_bins: Number of pLDDT confidence bins.
        num_resolved_bins: Number of resolved-atom bins.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        distance_bin_start: Lower edge of the confidence distance grid, in angstroms.
        distance_bin_end: Upper edge of the confidence distance grid, in angstroms.
        distance_bin_step: Spacing between confidence distance bins, in angstroms.
        max_atoms_per_token: Maximum number of atoms represented for a single token.
        single_max_abs: Optional clamp on the absolute value of single-representation
            features; ``None`` disables it.
        num_blocks: Number of Pairformer blocks in the confidence trunk.
        num_pair_attention_heads: Number of attention heads used for pair-biased attention.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication update.
        triangle_attention_hidden_size: Per-head hidden width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor inside the Pairformer transition.
        pairformer_dropout: Dropout probability applied inside the confidence Pairformer.
    """

    num_plddt_bins: int = 50
    num_resolved_bins: int = 2
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    distance_bin_start: float = 3.25
    distance_bin_end: float = 52.0
    distance_bin_step: float = 1.25
    max_atoms_per_token: int = 24
    single_max_abs: float | None = 512.0
    num_blocks: int = 4
    num_pair_attention_heads: int = 16
    triangle_multiplication_hidden_size: int = 128
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 4
    pairformer_dropout: float = 0.0


class ProtenixConfig(Config):
    r"""
    Configure one released Protenix graph.

    Attributes:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of raw atom features.
        relative_position_size: Width of relative-position features.
        constraints: Whether to build the jointly trained four-branch constraint variant.
        trunk: Pairformer trunk configuration.
        template: Optional family-local template graph.
        esm: Optional frozen ESM2 encoder graph.
        recycling_steps: Number of trunk recycling passes.
        denoiser: Coordinate denoiser configuration.
        sampler: Reverse-diffusion sampler configuration.
        distogram: Distogram head configuration.
        confidence: Confidence head configuration.

    Examples:
        >>> from ofoldx.models.protenix import ProtenixConfig
        >>> config = ProtenixConfig(recycling_steps=4)
        >>> config.recycling_steps
        4
    """

    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    constraints: bool = False
    trunk: ProtenixTrunkConfig = ProtenixTrunkConfig()
    template: ProtenixTemplateConfig | None = None
    esm: ProtenixEsmConfig | None = None
    recycling_steps: int = 10
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: DistogramConfig = DistogramConfig()
    confidence: ProtenixConfidenceConfig = ProtenixConfidenceConfig()

    @staticmethod
    def from_variant(variant: ProtenixVariant) -> ProtenixConfig:
        r"""
        Construct the exact persistent graph configuration for one released checkpoint variant.

        The two v1 releases return the same graph. ESM and ISM return the same
        Protenix graph and differ only in the companion weights selected explicitly
        by conversion.

        Args:
            variant: One of the ten released Protenix checkpoint names.

        Returns:
            The matching graph and inference configuration.

        Raises:
            ValueError: If ``variant`` is not a released checkpoint name.
        """

        if variant == "protenix_base_default_v0.5.0":
            return ProtenixConfig()
        if variant == "protenix_base_constraint_v0.5.0":
            return ProtenixConstraintConfig()
        if variant == "protenix_mini_tmpl_v0.5.0":
            return ProtenixStructureConditionConfig()
        if variant in {
            "protenix_mini_default_v0.5.0",
            "protenix_mini_esm_v0.5.0",
            "protenix_mini_ism_v0.5.0",
        }:
            return ProtenixConfig(
                trunk=ProtenixTrunkConfig(
                    num_msa_stages=1,
                    num_pairformer_blocks=16,
                ),
                esm=(
                    ProtenixEsmConfig() if variant in {"protenix_mini_esm_v0.5.0", "protenix_mini_ism_v0.5.0"} else None
                ),
                recycling_steps=4,
                denoiser=DenoiserConfig(
                    num_token_encoder_blocks=8,
                    num_atom_encoder_blocks=1,
                    num_atom_decoder_blocks=1,
                ),
                sampler=SamplerConfig(
                    num_sampling_steps=5,
                    churn_rate=0.0,
                    step_scale=1.0,
                ),
            )
        if variant == "protenix_tiny_default_v0.5.0":
            return ProtenixConfig(
                trunk=ProtenixTrunkConfig(
                    num_msa_stages=1,
                    num_pairformer_blocks=8,
                ),
                recycling_steps=4,
                denoiser=DenoiserConfig(
                    num_token_encoder_blocks=8,
                    num_atom_encoder_blocks=1,
                    num_atom_decoder_blocks=1,
                ),
                sampler=SamplerConfig(
                    num_sampling_steps=5,
                    churn_rate=0.0,
                    step_scale=1.0,
                ),
            )
        if variant in {
            "protenix_base_default_v1.0.0",
            "protenix_base_20250630_v1.0.0",
        }:
            return ProtenixTemplateModelConfig()
        if variant == "protenix-v2":
            return ProtenixTemplateModelConfig(
                pair_size=256,
                trunk=ProtenixTrunkConfig(
                    msa_size=128,
                    msa_attention_head_dim=8,
                    msa_triangle_multiplication_hidden_size=256,
                    num_msa_triangle_attention_heads=8,
                    pairformer_triangle_multiplication_hidden_size=256,
                    num_pairformer_triangle_attention_heads=8,
                ),
                template=ProtenixTemplateConfig(
                    triangle_multiplication_hidden_size=64,
                    num_triangle_attention_heads=2,
                ),
                confidence=ProtenixConfidenceConfig(
                    triangle_multiplication_hidden_size=256,
                    num_triangle_attention_heads=8,
                ),
            )
        raise ValueError(f"unknown Protenix checkpoint variant: {variant!r}")


class ProtenixStructureConditionConfig(ProtenixConfig):
    r"""
    Configure the released structure-conditioned Protenix mini graph.

    The graph adds a noisy-structure pair update on every recycling pass; it is
    not an ordinary template or constraint variant.

    Examples:
        >>> from ofoldx.models.protenix import ProtenixStructureConditionConfig
        >>> config = ProtenixStructureConditionConfig()
        >>> config.recycling_steps, config.trunk.num_pairformer_blocks
        (4, 16)
    """

    trunk: ProtenixTrunkConfig = ProtenixTrunkConfig(
        num_msa_stages=1,
        num_pairformer_blocks=16,
    )
    recycling_steps: int = 4
    denoiser: DenoiserConfig = DenoiserConfig(
        num_token_encoder_blocks=8,
        num_atom_encoder_blocks=1,
        num_atom_decoder_blocks=1,
    )
    sampler: SamplerConfig = SamplerConfig(
        num_sampling_steps=5,
        churn_rate=0.0,
        step_scale=1.0,
    )


class ProtenixConstraintConfig(ProtenixConfig):
    r"""
    Configure the four-branch Protenix constraint graph.

    Attributes:
        constraints: Whether to build the jointly trained four-branch constraint embedder; fixed to ``True`` here.

    Examples:
        >>> from ofoldx.models.protenix import ProtenixConstraintConfig
        >>> config = ProtenixConstraintConfig()
        >>> config.constraints
        True
    """

    constraints: bool = True


class ProtenixTemplateModelConfig(ProtenixConfig):
    r"""
    Configure the base Protenix graph with the template embedder enabled.

    This is the ``protenix_template`` architecture: the same graph as base Protenix but with the released
    template embedder built in by default (``template`` on), so a fresh build pairs correctly with the
    template-producing Processor instead of silently ignoring template features.

    Examples:
        >>> from ofoldx.models.protenix import ProtenixTemplateModelConfig
        >>> ProtenixTemplateModelConfig().template is not None
        True
    """

    template: ProtenixTemplateConfig | None = ProtenixTemplateConfig()


class ProtenixProcessorConfig(Config):
    r"""
    Configure Protenix input preprocessing.

    Attributes:
        max_msa_rows: Maximum MSA depth included in model inputs.
        augment_reference_positions: Whether to randomly rotate and translate reference coordinates.
    """

    max_msa_rows: int = 16384
    augment_reference_positions: bool = True
