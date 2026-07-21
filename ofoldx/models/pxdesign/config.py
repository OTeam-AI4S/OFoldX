r"""Configuration of the released PxDesign v0.1 graph and sampler."""

from __future__ import annotations

from chanfig import Config


class PxDesignConditionEmbedderConfig(Config):
    r"""Widths and block counts of the released design-condition embedder."""

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
    r"""Widths and block counts of the released coordinate denoiser."""

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
    r"""Released PxDesign EDM inference defaults."""

    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
    gamma_0: float = 1.0
    gamma_min: float = 0.01
    noise_scale: float = 1.003
    step_scale_min: float = 1.0
    step_scale_max: float = 2.5
    step_scale_schedule: str = "piecewise_65"


class PxDesignConfig(Config):
    r"""Exact configurable shape of the released PxDesign v0.1 model."""

    single_size: int = 384
    pair_size: int = 128
    token_feature_size: int = 449
    atom_feature_size: int = 385
    num_template_bins: int = 64

    condition_embedder: PxDesignConditionEmbedderConfig = PxDesignConditionEmbedderConfig()
    denoiser: PxDesignDenoiserConfig = PxDesignDenoiserConfig()
    sampler: PxDesignSamplerConfig = PxDesignSamplerConfig()


class PxDesignProcessorConfig(Config):
    r"""The exact empty persistent configuration for canonical PxDesign lowering."""
