r"""
Shared model configuration blocks.
"""

from __future__ import annotations

from chanfig import Config


class DistogramConfig(Config):
    r"""
    Distogram head configuration.

    The distogram head reads the pair representation and predicts, per residue pair, a categorical
    distribution over binned inter-residue distances.

    Attributes:
        num_bins: Number of distance bins the head predicts over.

    Examples:
        >>> config = DistogramConfig()
        >>> config.num_bins
        64
    """

    num_bins: int = 64


class ConfidenceConfig(Config):
    r"""
    Confidence head configuration.

    Groups every knob of the confidence trunk: the pLDDT / PAE / PDE / resolved bin counts, the
    distogram-derived distance grid, and the Pairformer stack that refines the single and pair
    representations before the per-metric projections.

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
        blocks_per_checkpoint: Blocks grouped into one gradient-checkpoint segment; ``None`` disables checkpointing.
        operator_backend: Operator backend selector (``"auto"`` prefers the OOps kernels when available).
        num_pair_attention_heads: Number of attention heads used for pair-biased attention.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication update.
        triangle_attention_hidden_size: Per-head hidden width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor inside the Pairformer transition.
        pairformer_dropout: Dropout probability applied inside the confidence Pairformer.

    Examples:
        >>> config = ConfidenceConfig()
        >>> config.num_pae_bins
        64
        >>> config.operator_backend
        'auto'
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
    blocks_per_checkpoint: int | None = 1
    operator_backend: str = "auto"
    num_pair_attention_heads: int = 16
    triangle_multiplication_hidden_size: int = 128
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    expansion: int = 4
    pairformer_dropout: float = 0.0


class TrunkConfig(Config):
    r"""
    Representation trunk configuration.

    Configures the main Pairformer trunk that jointly refines the single, pair, and MSA
    representations, including the per-axis dropout rates used during training.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
        blocks_per_checkpoint: Blocks grouped into one gradient-checkpoint segment; ``None`` disables checkpointing.
        operator_backend: Operator backend selector (``"auto"`` prefers the OOps kernels when available).
        pairformer_dropout: Dropout probability applied inside each Pairformer block.
        msa_dropout: Dropout probability applied along the MSA path.
        pair_dropout: Dropout probability applied along the pair path.

    Examples:
        >>> config = TrunkConfig()
        >>> config.num_pairformer_blocks
        48
    """

    num_pairformer_blocks: int = 48
    blocks_per_checkpoint: int | None = 1
    operator_backend: str = "auto"
    pairformer_dropout: float = 0.25
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25


class TemplateConfig(Config):
    r"""
    Template embedding configuration.

    Controls the optional template stack that embeds structural templates into the pair
    representation. A ``num_blocks`` of ``0`` disables template embedding entirely.

    Attributes:
        hidden_size: Width of the template pair embedding.
        num_blocks: Number of template Pairformer blocks; ``0`` disables the template path.
        vocab_size: Number of template token types.
        num_distogram_bins: Number of template distogram distance bins.
        triangle_attention_hidden_size: Per-head width of template triangle attention.
        num_triangle_attention_heads: Number of template triangle-attention heads.
        expansion: Feed-forward expansion factor inside the template transition.
        dropout: Dropout probability applied inside the template stack.

    Examples:
        >>> config = TemplateConfig()
        >>> config.hidden_size
        64
        >>> config.num_blocks
        0
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
    r"""
    Coordinate denoiser configuration.

    Configures the diffusion denoiser: the atom encoder / decoder that map between per-atom and
    per-token representations, the token-level transformer trunk, and the local-attention window
    sizes used by the atom stacks.

    Attributes:
        num_token_encoder_blocks: Number of token-level transformer blocks in the denoiser trunk.
        num_token_encoder_attention_heads: Number of attention heads in the token-level trunk.
        blocks_per_checkpoint: Blocks grouped into one gradient-checkpoint segment; ``None`` disables checkpointing.
        operator_backend: Operator backend selector (``"auto"`` prefers the OOps kernels when available).
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

    Examples:
        >>> config = DenoiserConfig()
        >>> config.token_hidden_size
        768
        >>> config.sigma_data
        16.0
    """

    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    blocks_per_checkpoint: int | None = 1
    operator_backend: str = "auto"
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
    Coordinate sampler configuration.

    Controls the EDM reverse-diffusion loop used at inference: the solver, the number of sampling
    steps, the churn (noise re-injection) schedule, and the log-uniform noise-level grid bounds.

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

    Examples:
        >>> config = SamplerConfig()
        >>> config.solver
        'euler'
        >>> config.num_sampling_steps
        200
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


class DiffusionTrainingConfig(Config):
    r"""
    Single-step diffusion-training configuration.

    Training denoises noised ground truth in one step, with independent log-normal noise per sample. A
    separate no-grad ``rollout_steps`` sampler produces coords for the confidence head; the full
    ``sampler.num_sampling_steps`` loop is used only for eval/inference.

    Attributes:
        num_samples: Number of independently noised diffusion samples drawn per training example.
        noise_log_mean: Mean of the log-normal training noise-level distribution.
        noise_log_std: Standard deviation of the log-normal training noise-level distribution.
        rollout_steps: Steps of the no-grad mini-rollout sampler that feeds the confidence head.
        rollout_num_samples: Number of samples produced by the mini-rollout sampler.

    Examples:
        >>> config = DiffusionTrainingConfig()
        >>> config.num_samples
        48
        >>> config.rollout_steps
        20
    """

    num_samples: int = 48
    noise_log_mean: float = -1.2
    noise_log_std: float = 1.5
    rollout_steps: int = 20
    rollout_num_samples: int = 1


class StructurePredictionConfig(Config):
    r"""
    Shared configuration for structure-prediction models with token, pair, template, diffusion, and confidence paths.

    Concrete AlphaFold3-family architectures subclass this and set their own ``architecture`` key; the base
    class collects the representation widths and the nested sub-configs (distogram, confidence, trunk,
    template, denoiser, sampler, diffusion-training) shared across the family.

    Attributes:
        token_feature_size: Width of the per-token input feature vector.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_feature_size: Width of the per-atom input feature vector.
        relative_position_size: Number of relative-position encoding channels.
        initializer_range: Standard deviation of the truncated-normal weight initialization.
        distogram: Distogram head sub-configuration.
        confidence: Confidence head sub-configuration.
        trunk: Representation trunk sub-configuration.
        template: Template embedding sub-configuration.
        denoiser: Coordinate denoiser sub-configuration.
        sampler: Coordinate sampler sub-configuration.
        diffusion_training: Single-step diffusion-training sub-configuration.
        recycling_steps: Number of trunk recycling iterations.

    Examples:
        >>> config = StructurePredictionConfig()
        >>> config.single_size
        384
        >>> config.pair_size
        128
        >>> config.trunk.num_pairformer_blocks
        48
    """

    token_feature_size: int = 449
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    initializer_range: float = 0.02

    distogram: DistogramConfig = DistogramConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    trunk: TrunkConfig = TrunkConfig()
    template: TemplateConfig = TemplateConfig()
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    diffusion_training: DiffusionTrainingConfig = DiffusionTrainingConfig()

    recycling_steps: int = 4
