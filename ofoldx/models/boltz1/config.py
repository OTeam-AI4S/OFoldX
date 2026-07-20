r"""
Boltz-1 configuration.

    Defaults follow the released ``boltz1_conf.ckpt`` inference entry point: 48 Pairformer blocks,
    a 4-block MSA module, 8-head diffusion token transformer, and the Boltz-1 EDM sampling schedule.
"""

from __future__ import annotations

from chanfig import Config


class Boltz1TrunkConfig(Config):
    r"""
    Representation-trunk knobs for Boltz-1: the Pairformer stack and the optional MSA module.

    The two head counts are distinct: ``num_msa_attention_heads`` sizes the MSA pair-weighted-averaging,
    while ``num_triangle_attention_heads`` governs the triangle attention shared by the trunk Pairformer and
    the MSA pairformer-no-seq layer.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
        num_pair_attention_heads: Head count for the single-track pair-biased attention.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_feature_size: Width of the raw per-MSA feature vector (one-hot + deletion + pairing flags).
        msa_head_dim: Per-head dimension inside the MSA attention.
        num_msa_attention_heads: Head count for the MSA pair-weighted-averaging.
        outer_product_hidden_size: Hidden width of the MSA outer-product-mean projection.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Head count for the triangle attention.
        use_paired_feature: Whether the MSA carries the cross-chain pairing feature.
        dropout: Dropout applied to the Pairformer pair track.
        msa_dropout: Dropout applied inside the MSA module.

    Examples:
        >>> config = Boltz1TrunkConfig()
        >>> config.num_pairformer_blocks
        48
    """

    num_pairformer_blocks: int = 48
    num_pair_attention_heads: int = 16
    num_msa_blocks: int = 4
    msa_size: int = 64
    msa_feature_size: int = 35
    msa_head_dim: int = 32
    num_msa_attention_heads: int = 8
    outer_product_hidden_size: int = 32
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    use_paired_feature: bool = False
    dropout: float = 0.0
    msa_dropout: float = 0.0


class Boltz1InputEmbedderConfig(Config):
    r"""
    Input-embedder knobs for Boltz-1: the atom-attention encoder and its token-level feature widths.

    Attributes:
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size feeding the one-hot residue feature.
        pocket_feature_size: Width of the token-level pocket one-hot conditioning feature.
        no_atom_encoder: When ``True``, skip the atom encoder and emit zeroed token-from-atom states.

    Examples:
        >>> config = Boltz1InputEmbedderConfig()
        >>> config.num_atom_encoder_blocks
        3
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = 33
    pocket_feature_size: int = 4
    no_atom_encoder: bool = False


class Boltz1DiffusionConfig(Config):
    r"""
    EDM denoising-module knobs for Boltz-1: the token transformer, atom encoder/decoder, and score-model widths.

    Attributes:
        num_token_transformer_blocks: Number of conditioned token-transformer blocks in the score model.
        num_token_transformer_heads: Head count for the token transformer.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks in the denoiser.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks in the denoiser.
        num_atom_attention_heads: Head count shared by the denoiser atom encoder and decoder.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_conditioning_transitions: Number of transition blocks in each diffusion-conditioning stack.
        sigma_data: EDM data-scale hyperparameter used by the preconditioner and noise schedule.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step to the denoised coords.

    Examples:
        >>> config = Boltz1DiffusionConfig()
        >>> config.num_token_transformer_blocks
        24
    """

    num_token_transformer_blocks: int = 24
    num_token_transformer_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_decoder_blocks: int = 3
    num_atom_attention_heads: int = 4
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    num_conditioning_transitions: int = 2
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True


class Boltz1SamplerConfig(Config):
    r"""
    Reverse-diffusion sampler parameters for the Boltz-1 EDM schedule.

    Attributes:
        num_sampling_steps: Number of reverse-diffusion steps.
        rho: Karras schedule power controlling the sigma spacing.
        max_noise_level: Highest noise level (start of the schedule), in ``sigma_data`` units.
        min_noise_level: Lowest non-zero noise level (end of the schedule), in ``sigma_data`` units.
        churn_rate: Stochastic-churn factor added above ``min_churn_noise_level``.
        min_churn_noise_level: Noise level below which churn is disabled.
        noise_scale: Multiplier on the churn-injected noise.
        step_scale: Multiplier on the Euler update step.

    Examples:
        >>> config = Boltz1SamplerConfig()
        >>> config.num_sampling_steps
        200
    """

    num_sampling_steps: int = 200
    rho: float = 7.0
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    churn_rate: float = 0.8
    min_churn_noise_level: float = 1.0
    noise_scale: float = 1.0
    step_scale: float = 1.0


class Boltz1ConfidenceConfig(Config):
    r"""
    Boltz-1 confidence head and ranking configuration.

    The released ``boltz1_conf.ckpt`` uses the trunk-imitating confidence module, computes PAE/pTM metrics
    (``alpha_pae > 0`` in upstream Boltz), and conditions on the final diffusion token representation.

    Attributes:
        enabled: Whether to build and run the confidence/ranking head at all.
        imitate_trunk: Whether the confidence module re-runs a full trunk-like stack (vs. reusing trunk outputs).
        compute_pae: Whether to emit PAE logits and pTM/ipTM ranking metrics.
        num_pairformer_blocks: Number of Pairformer blocks inside the confidence module.
        num_pair_attention_heads: Head count for the confidence-module pair-biased attention.
        num_distogram_bins: Number of bins for the confidence distance embedding.
        max_distance: Upper distance bound (Angstrom) for the confidence distogram binning.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
        add_single_to_pair_product: Whether to add the single-input outer product into the pair track.
        add_single_input_to_single: Whether to inject the single-input representation into the single track.
        add_pair_input_to_pair: Whether to inject relative-position and bond features into the pair track.
        use_single_diffusion: Whether to condition on the accumulated final diffusion token representation.
        ranking_plddt_weight: Weight on complex pLDDT when forming the combined ranking score.
        ranking_ptm_weight: Weight on pTM/ipTM when forming the combined ranking score.

    Examples:
        >>> config = Boltz1ConfidenceConfig()
        >>> config.enabled
        True
    """

    enabled: bool = True
    imitate_trunk: bool = True
    compute_pae: bool = True
    num_pairformer_blocks: int = 48
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    add_single_to_pair_product: bool = True
    add_single_input_to_single: bool = True
    add_pair_input_to_pair: bool = True
    use_single_diffusion: bool = True
    ranking_plddt_weight: float = 4.0
    ranking_ptm_weight: float = 1.0


class Boltz1Config(Config):
    r"""
    Top-level configuration for the Boltz-1 structure-prediction model.

    Instantiating with the defaults reproduces the public Boltz-1 inference entry point. The nested sub-configs
    hold the trunk, input-embedder, diffusion, sampler, and confidence values; the top-level fields
    below set the shared representation widths and the recycling/distogram parameters.

    Attributes:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        atom_feature_size: Expected width of the raw per-atom feature vector.
        num_distogram_bins: Number of bins for the trunk distogram head.
        fourier_embedding_size: Width of the frozen Fourier noise embedding.
        relative_position_size: Width of the relative-position feature fed to the position embedder.
        recycling_steps: Total number of trunk passes.
        no_msa: When ``True``, build the model without the MSA module.
        input_embedder: Nested input-embedder configuration.
        trunk: Nested representation-trunk configuration.
        diffusion: Nested EDM denoising-module configuration.
        sampler: Nested reverse-diffusion sampler configuration.
        confidence: Nested confidence/ranking configuration.

    Examples:
        >>> config = Boltz1Config()
        >>> from ofoldx.models.boltz1.model import Boltz1Model
        >>> model = Boltz1Model(config)  # doctest: +SKIP
    """

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 389
    num_distogram_bins: int = 64
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    recycling_steps: int = 4
    no_msa: bool = False

    input_embedder: Boltz1InputEmbedderConfig = Boltz1InputEmbedderConfig()
    trunk: Boltz1TrunkConfig = Boltz1TrunkConfig()
    diffusion: Boltz1DiffusionConfig = Boltz1DiffusionConfig()
    sampler: Boltz1SamplerConfig = Boltz1SamplerConfig()
    confidence: Boltz1ConfidenceConfig = Boltz1ConfidenceConfig()
