r"""
Shared configuration blocks for diffusion structure models.
"""

from __future__ import annotations

from chanfig import Config


class DistogramConfig(Config):
    r"""
    Configure a head that predicts binned inter-residue distances from pair representations.

    Attributes:
        num_bins: Number of distance bins the head predicts over.
    """

    num_bins: int = 64


class DenoiserConfig(Config):
    r"""
    Configure the atom and token stacks of the diffusion coordinate denoiser.

    Attributes:
        num_token_encoder_blocks: Number of token-level transformer blocks in the denoiser trunk.
        num_token_encoder_attention_heads: Number of attention heads in the token-level trunk.
        sigma_data: Data noise scale used by the EDM preconditioning.
        atom_hidden_size: Width of the per-atom single representation.
        atom_pair_hidden_size: Width of the per-atom pair representation.
        token_hidden_size: Width of the per-token representation inside the denoiser trunk.
        num_atom_encoder_blocks: Number of atom-encoder blocks.
        num_atom_encoder_attention_heads: Number of attention heads in the atom encoder.
        num_atom_decoder_blocks: Number of atom-decoder blocks.
        num_atom_decoder_attention_heads: Number of attention heads in the atom decoder.
        num_query_atoms: Local-attention query window size for the atom stacks.
        num_key_atoms: Local-attention key window size for the atom stacks.
    """

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
    r"""
    Configure the solver and noise schedule for EDM reverse diffusion.

    Attributes:
        solver: Name of the ODE/SDE solver driving the reverse diffusion (e.g. ``"euler"``).
        num_sampling_steps: Number of reverse-diffusion steps.
        churn_rate: Fraction of steps that re-inject noise (stochastic churn).
        min_churn_noise_level: Minimum noise level at which churn is applied.
        noise_scale: Multiplier applied to the re-injected churn noise.
        step_scale: Multiplier applied to each denoising step.
        center_random_augmentation: Whether to apply random rotation/translation augmentation per step.
        max_noise_level: Upper bound of the noise-level grid.
        min_noise_level: Lower bound of the noise-level grid.
        rho: Exponent shaping the noise-level grid spacing.
    """

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
