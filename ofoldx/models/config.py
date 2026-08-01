r"""
Shared configuration blocks for diffusion structure models.
"""

from __future__ import annotations

from chanfig import Config


class DistogramConfig(Config):
    r"""Configure a binned inter-token distance head."""

    num_bins: int = 64


class DenoiserConfig(Config):
    r"""Configure the atom and token stacks of a coordinate denoiser."""

    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    sigma_data: float = 16.0
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    token_hidden_size: int = 768
    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    num_atom_decoder_blocks: int = 3
    num_atom_decoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128


class SamplerConfig(Config):
    r"""Configure the solver and noise schedule of EDM reverse diffusion."""

    solver: str = "euler"
    num_sampling_steps: int = 200
    churn_rate: float = 0.8
    min_churn_noise_level: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    center_random_augmentation: bool = True
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
