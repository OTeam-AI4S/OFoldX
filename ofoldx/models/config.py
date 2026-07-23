r"""Shared configuration blocks for diffusion structure models."""

from __future__ import annotations

from chanfig import Config


class DistogramConfig(Config):
    r"""Configure a head that predicts binned inter-residue distances from pair representations.

    Attributes:
        num_bins: Number of distance bins the head predicts over.
    """

    num_bins: int = 64


class ConfidenceConfig(Config):
    r"""Configure confidence metrics and the Pairformer trunk that predicts them.

    Attributes:
        num_plddt_bins: Number of pLDDT confidence bins.
        num_resolved_bins: Number of resolved-atom bins.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        distance_bin_start: Lower edge of the confidence distance grid, in angstroms.
        distance_bin_end: Upper edge of the confidence distance grid, in angstroms.
        distance_bin_step: Spacing between confidence distance bins, in angstroms.
        max_atoms_per_token: Maximum number of atoms represented for a single token.
        single_max_abs: Optional clamp on the absolute value of single-representation features; ``None`` disables it.
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


class TrunkConfig(Config):
    r"""Configure the Pairformer trunk that refines single, pair, and MSA representations.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
        pairformer_dropout: Dropout probability applied inside each Pairformer block.
        msa_dropout: Dropout probability applied along the MSA path.
        pair_dropout: Dropout probability applied along the pair path.
    """

    num_pairformer_blocks: int = 48
    pairformer_dropout: float = 0.25
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25


class TemplateConfig(Config):
    r"""Configure the optional stack that embeds structural templates into pair representations.

    Attributes:
        hidden_size: Width of the template pair embedding.
        num_blocks: Number of template Pairformer blocks; ``0`` disables the template path.
        vocab_size: Number of template token types.
        num_distogram_bins: Number of template distogram distance bins.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Feed-forward expansion factor inside the template transition.
        dropout: Dropout probability applied inside the template stack.
    """

    hidden_size: int = 64
    num_blocks: int = 0
    vocab_size: int = 32
    num_distogram_bins: int = 39
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 2
    dropout: float = 0.25


class DenoiserConfig(Config):
    r"""Configure the atom and token stacks of the diffusion coordinate denoiser.

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
    r"""Configure the solver and noise schedule for EDM reverse diffusion.

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
