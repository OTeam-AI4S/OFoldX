r"""
Configuration hierarchy for the native PxDesign diffusion generator and its sub-modules.

Groups the top-level [`PxDesignConfig`][ofoldx.models.pxdesign.config.PxDesignConfig] and its per-module
sub-configs (condition embedder, conditional template, denoiser, EDM sampler, distogram head) that expose the
tunable knobs of the PxDesign model family.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


class PxDesignConditionEmbedderConfig(Config):
    r"""
    Configuration of the PxDesign condition-embedding module.

    The condition embedder fuses per-atom reference features (through an
    atom-to-token attention encoder) with the design token features into the
    single/token conditioning consumed by the diffusion denoiser.

    Attributes:
        atom_hidden_size: Width of the per-atom hidden channel in the atom encoder.
        atom_pair_hidden_size: Width of the per-atom-pair channel in the atom encoder.
        token_hidden_size: Width of the aggregated per-token hidden channel.
        token_feature_size: Width of the token feature vector after the input projection.
        vocab_size: Number of restype ids, including the PxDesign design placeholders.
        plddt_size: Number of per-token pLDDT conditioning channels.
        hotspot_size: Number of per-token hotspot conditioning channels.
        add_feature_size: Width of each auxiliary one-hot design feature block.
        num_atom_encoder_blocks: Number of atom-encoder transformer blocks.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_query_atoms: Local-attention query-atom window size.
        num_key_atoms: Local-attention key-atom window size.
        operator_backend: Operator backend selector for the atom encoder (`"auto"`
            picks the OOps kernels when available, otherwise the public path).

    Examples:
        >>> config = PxDesignConditionEmbedderConfig()
        >>> config.token_hidden_size
        384
        >>> config.operator_backend
        'auto'
    """

    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    token_hidden_size: int = 384
    token_feature_size: int = 449
    vocab_size: int = 36
    plddt_size: int = 1
    hotspot_size: int = 1
    add_feature_size: int = 4
    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    operator_backend: str = "auto"


class PxDesignTemplateConfig(Config):
    r"""
    Configuration of the PxDesign conditional-template pair embedding.

    Sizes the embedding table that turns condition-condition distance bins into the initial pair representation
    fed to the diffusion module.

    Attributes:
        num_template_bins: Number of condition-condition distance bins; the
            embedding reserves one extra id for unconditioned pairs.
        pair_size: Width of the pair representation produced by the template embedder.

    Examples:
        >>> config = PxDesignTemplateConfig()
        >>> config.num_template_bins
        64
        >>> config.pair_size
        128
    """

    num_template_bins: int = 64
    pair_size: int = 128


class PxDesignDenoiserConfig(Config):
    r"""
    Configuration of the PxDesign AF3-style coordinate denoiser.

    Sets the widths, block counts, and local-attention windows of the token/atom encoder-decoder stack that
    predicts denoised coordinates at each EDM sampling step.

    Attributes:
        sigma_data: EDM data-scale used to precondition the denoiser inputs/outputs.
        token_hidden_size: Width of the token hidden channel in the token encoder.
        atom_hidden_size: Width of the per-atom hidden channel in the atom encoder/decoder.
        atom_pair_hidden_size: Width of the per-atom-pair channel in the atom encoder/decoder.
        num_token_encoder_blocks: Number of token-encoder transformer blocks.
        num_token_encoder_attention_heads: Number of attention heads per token-encoder block.
        num_atom_encoder_blocks: Number of atom-encoder transformer blocks.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_atom_decoder_blocks: Number of atom-decoder transformer blocks.
        num_atom_decoder_attention_heads: Number of attention heads per atom-decoder block.
        num_query_atoms: Local-attention query-atom window size.
        num_key_atoms: Local-attention key-atom window size.
        blocks_per_checkpoint: Number of token-encoder blocks per gradient-checkpoint
            segment, or ``None`` to disable checkpointing.
        operator_backend: Operator backend selector for the denoiser modules.

    Examples:
        >>> config = PxDesignDenoiserConfig()
        >>> config.num_token_encoder_blocks
        16
        >>> config.sigma_data
        16.0
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
    blocks_per_checkpoint: int | None = 1
    operator_backend: str = "auto"


class PxDesignSamplerConfig(Config):
    r"""
    Configuration of the PxDesign EDM coordinate sampler.

    Controls the EDM noise schedule, churn, per-step step-scale schedule, and how many coordinate samples are
    drawn (and chunked) per design input.

    Attributes:
        num_sampling_steps: Number of EDM denoising steps in the sampling schedule.
        num_samples: Number of design coordinate samples drawn per input.
        sample_chunk_size: Number of samples generated per forward chunk, or ``None``
            to generate all samples in one pass.
        max_noise_level: Highest noise level (sigma) at the start of the schedule.
        min_noise_level: Lowest noise level (sigma) at the end of the schedule.
        rho: EDM schedule curvature exponent.
        sigma_data: EDM data-scale shared with the denoiser preconditioning.
        gamma0: Churn factor added to each step's noise level when churn is active.
        gamma_min: Noise-level floor below which churn is disabled.
        noise_scale: Multiplier on the injected churn noise.
        step_scale_min: Lower bound of the per-step update scale.
        step_scale_max: Upper bound of the per-step update scale.
        step_scale_schedule: Named step-scale schedule over the step index (e.g.
            `"piecewise_65"`, `"linear"`, `"cos"`).
        center_random_augmentation: Whether to apply centered random augmentation
            before each sampling step.

    Examples:
        >>> config = PxDesignSamplerConfig()
        >>> config.num_sampling_steps
        400
        >>> config.step_scale_schedule
        'piecewise_65'
    """

    num_sampling_steps: int = 400
    num_samples: int = 100
    sample_chunk_size: int | None = 10
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
    sigma_data: float = 16.0
    gamma0: float = 1.0
    gamma_min: float = 0.01
    noise_scale: float = 1.003
    step_scale_min: float = 1.0
    step_scale_max: float = 2.5
    step_scale_schedule: str = "piecewise_65"
    center_random_augmentation: bool = True


class PxDesignDistogramConfig(Config):
    r"""
    Configuration of the PxDesign distogram head.

    The public PxDesign v0.1.0 inference model leaves its design distogram heads
    disabled. OFoldX keeps this lightweight head available for native runs, and
    the checkpoint converter reports whether it was loaded or freshly initialized.

    Attributes:
        num_distogram_bins: Number of distance bins predicted by the distogram head.
        symmetrize_before_projection: Whether to symmetrize the pair representation
            before the final bin projection.

    Examples:
        >>> config = PxDesignDistogramConfig()
        >>> config.num_distogram_bins
        64
        >>> config.symmetrize_before_projection
        True
    """

    num_distogram_bins: int = 64
    symmetrize_before_projection: bool = True


@MODEL_CONFIGS.register("pxdesign")
class PxDesignConfig(Config):
    r"""
    Native PxDesign diffusion-generator configuration.

    Defaults track the public PxDesign v0.1.0 generator where it can be represented by
    OFoldX-native modules: PX condition embedding, conditional-template pair
    conditioning, and the AF3-style coordinate diffusion module. Registered under the
    `"pxdesign"` key in [`MODEL_CONFIGS`][ofoldx.models.registry.MODEL_CONFIGS].

    Attributes:
        architecture: Registry key selecting this architecture (`"pxdesign"`).
        model_name: Human-readable model/release name.
        checkpoint_variant: Official checkpoint variant this config targets.
        initializer_range: Standard deviation of the truncated-normal weight init.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        token_feature_size: Width of the per-token feature vector fed to the denoiser.
        atom_feature_size: Width of the per-atom reference feature vector.
        relative_position_size: Width of the relative-position feature encoding.
        condition_embedder: Sub-configuration of the condition-embedding module.
        conditional_template: Sub-configuration of the conditional-template pair embedding.
        denoiser: Sub-configuration of the coordinate denoiser.
        sampler: Sub-configuration of the EDM coordinate sampler.
        distogram: Sub-configuration of the distogram head.

    Examples:
        >>> config = PxDesignConfig()
        >>> config.architecture
        'pxdesign'
        >>> config.single_size
        384
        >>> config.condition_embedder.vocab_size
        36
        >>> from ofoldx.models.pxdesign.model import PxDesignDesignGenerator
        >>> model = PxDesignDesignGenerator(config)  # doctest: +SKIP
    """

    architecture: str = "pxdesign"
    model_name: str = "pxdesign_v0.1.0"
    checkpoint_variant: str = "pxdesign_v0.1.0"
    initializer_range: float = 0.02

    single_size: int = 384
    pair_size: int = 128
    token_feature_size: int = 449
    atom_feature_size: int = 385
    relative_position_size: int = 139

    condition_embedder: PxDesignConditionEmbedderConfig = PxDesignConditionEmbedderConfig()
    conditional_template: PxDesignTemplateConfig = PxDesignTemplateConfig()
    denoiser: PxDesignDenoiserConfig = PxDesignDenoiserConfig()
    sampler: PxDesignSamplerConfig = PxDesignSamplerConfig()
    distogram: PxDesignDistogramConfig = PxDesignDistogramConfig()
