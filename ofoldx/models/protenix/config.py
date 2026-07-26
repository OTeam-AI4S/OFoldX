r"""
Configuration for Protenix folding and constraint architectures.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config

from ..config import (
    ConfidenceConfig,
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
    TemplateConfig,
    TrunkConfig,
)

ProtenixVariant = Literal[
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_tiny_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
]


class ProtenixTrunkConfig(TrunkConfig):
    r"""
    Configure the exact MSA and Pairformer axes of a Protenix trunk.

    Args:
        msa_size: Width of the projected MSA representation.
        num_msa_blocks: Number of joint MSA/pair update blocks.
        msa_outer_product_hidden_size: Hidden width of the MSA outer-product mean.
        num_msa_attention_heads: Number of heads in MSA pair-weighted averaging.
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

    msa_size: int = 64
    num_msa_blocks: int = 4
    msa_outer_product_hidden_size: int = 32
    num_msa_attention_heads: int = 8
    msa_triangle_multiplication_hidden_size: int = 128
    msa_triangle_attention_hidden_size: int = 32
    num_msa_triangle_attention_heads: int = 4
    msa_expansion: int = 4
    pairformer_triangle_multiplication_hidden_size: int = 128
    pairformer_triangle_attention_hidden_size: int = 32
    num_pairformer_triangle_attention_heads: int = 4
    num_pair_attention_heads: int = 16
    pairformer_expansion: int = 4


class ProtenixTemplateConfig(TemplateConfig):
    r"""
    Configure the released Protenix template graph used by base v1 and v2.

    Args:
        triangle_multiplication_hidden_size: Hidden width of template triangle multiplication.
    """

    num_blocks: int = 2
    triangle_multiplication_hidden_size: int = 128


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


class ProtenixConfig(Config):
    r"""
    Configure one released Protenix graph.

    Args:
        token_feature_size: Width of raw token features.
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
    """

    token_feature_size: int = 449
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
    confidence: ConfidenceConfig = ConfidenceConfig()

    @staticmethod
    def from_variant(variant: ProtenixVariant) -> ProtenixConfig:
        r"""
        Construct the exact persistent graph configuration for one released checkpoint variant.

        The two v1 releases return the same graph. ESM and ISM return the same
        Protenix graph and differ only in the companion weights selected explicitly
        by conversion.

        Args:
            variant: One of the nine released Protenix checkpoint names.

        Returns:
            The matching graph and inference configuration.

        Raises:
            ValueError: If ``variant`` is not a released checkpoint name.
        """

        if variant == "protenix_base_default_v0.5.0":
            return ProtenixConfig()
        if variant == "protenix_base_constraint_v0.5.0":
            return ProtenixConstraintConfig()
        if variant in {
            "protenix_mini_default_v0.5.0",
            "protenix_mini_esm_v0.5.0",
            "protenix_mini_ism_v0.5.0",
        }:
            return ProtenixConfig(
                trunk=ProtenixTrunkConfig(
                    num_msa_blocks=1,
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
                    num_msa_blocks=1,
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
            return ProtenixConfig(
                template=ProtenixTemplateConfig(),
            )
        if variant == "protenix-v2":
            return ProtenixConfig(
                pair_size=256,
                trunk=ProtenixTrunkConfig(
                    msa_size=128,
                    msa_triangle_multiplication_hidden_size=256,
                    num_msa_triangle_attention_heads=8,
                    pairformer_triangle_multiplication_hidden_size=256,
                    num_pairformer_triangle_attention_heads=8,
                ),
                template=ProtenixTemplateConfig(
                    triangle_multiplication_hidden_size=64,
                    num_triangle_attention_heads=2,
                ),
                confidence=ConfidenceConfig(
                    triangle_multiplication_hidden_size=256,
                    num_triangle_attention_heads=8,
                ),
            )
        raise ValueError(f"unknown Protenix checkpoint variant: {variant!r}")


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
