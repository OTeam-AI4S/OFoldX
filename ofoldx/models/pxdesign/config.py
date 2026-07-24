r"""
PxDesign architecture and sampler configuration.
"""

from __future__ import annotations

from chanfig import Config


class PxDesignConditionEmbeddingsConfig(Config):
    r"""
    Configure the design-condition embedder.

    Args:
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        token_hidden_size: Width of token condition features.
        vocab_size: Number of token types.
        plddt_size: Width of per-token pLDDT conditions.
        hotspot_size: Width of hotspot conditions.
        add_feature_size: Width of additional design conditions.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
    """

    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    token_hidden_size: int = 384
    vocab_size: int = 36
    plddt_size: int = 1
    hotspot_size: int = 1
    add_feature_size: int = 4
    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128


class PxDesignDenoiserConfig(Config):
    r"""
    Configure the coordinate denoiser.

    Args:
        sigma_data: Data noise scale used by EDM preconditioning.
        token_hidden_size: Width of denoiser token features.
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        num_token_encoder_blocks: Number of token encoder blocks.
        num_token_encoder_attention_heads: Number of token encoder attention heads.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        num_atom_decoder_blocks: Number of atom decoder blocks.
        num_atom_decoder_attention_heads: Number of atom decoder attention heads.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
    """

    sigma_data: float = 16.0
    token_hidden_size: int = 768
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_token_encoder_blocks: int = 16
    num_token_encoder_attention_heads: int = 16
    num_atom_encoder_blocks: int = 4
    num_atom_encoder_attention_heads: int = 4
    num_atom_decoder_blocks: int = 4
    num_atom_decoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128


class PxDesignSamplerConfig(Config):
    r"""
    Configure the PxDesign EDM inference schedule.

    Args:
        max_noise_level: Highest noise level in the schedule.
        min_noise_level: Lowest non-zero noise level in the schedule.
        rho: Exponent shaping noise-level spacing.
        churn_rate: Stochastic churn added during sampling.
        min_churn_noise_level: Noise threshold below which churn is disabled.
        noise_scale: Multiplier on churn-injected noise.
        step_scale_min: Minimum denoising-step multiplier.
        step_scale_max: Maximum denoising-step multiplier.
        step_scale_schedule: Schedule used to vary the denoising-step multiplier.
    """

    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
    churn_rate: float = 1.0
    min_churn_noise_level: float = 0.01
    noise_scale: float = 1.003
    step_scale_min: float = 1.0
    step_scale_max: float = 2.5
    step_scale_schedule: str = "piecewise_65"


class PxDesignConfig(Config):
    r"""
    Configure the PxDesign model.

    Args:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        token_feature_size: Width of raw token features.
        atom_feature_size: Width of raw atom features.
        num_template_bins: Number of template distance bins.
        condition_embeddings: Design-condition embedding architecture.
        denoiser: Coordinate denoiser architecture.
        sampler: EDM inference schedule.
    """

    single_size: int = 384
    pair_size: int = 128
    token_feature_size: int = 449
    atom_feature_size: int = 385
    num_template_bins: int = 64

    condition_embeddings: PxDesignConditionEmbeddingsConfig = PxDesignConditionEmbeddingsConfig()
    denoiser: PxDesignDenoiserConfig = PxDesignDenoiserConfig()
    sampler: PxDesignSamplerConfig = PxDesignSamplerConfig()


class PxDesignProcessorConfig(Config):
    r"""
    Configuration for the PxDesign processor.
    """
