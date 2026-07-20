r"""
Boltz-2 configuration.

Architecture values are taken from the public ``boltz2_conf.ckpt`` ``hyper_parameters`` and verified against its
``state_dict`` (pairformer 64 blocks, MSA 4 blocks, diffusion score model 24 token-transformer blocks
with 3+3 atom encoder/decoder). Sampler defaults follow the public Boltz-2 inference command, which overrides
three older values serialized in the checkpoint.
"""

from __future__ import annotations

from chanfig import Config


class Boltz2TrunkConfig(Config):
    r"""
    Boltz-2 representation trunk (pairformer + MSA).

    Note the two distinct head counts: ``num_msa_attention_heads`` is the
    PairWeightedAveraging head count (hardcoded 8 in Boltz ``MSALayer``), whereas
    ``num_triangle_attention_heads`` (``pairwise_num_heads`` = 4) governs the triangle
    attention inside both the trunk pairformer and the MSA pairformer-no-seq layer.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk stack.
        num_pair_attention_heads: Head count for the trunk pair-biased attention.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_feature_size: Width of the per-column MSA input features
            (one_hot(33) + has_deletion + deletion_value + is_paired).
        msa_head_dim: Per-head dimension of the MSA attention.
        num_msa_attention_heads: PairWeightedAveraging head count (hardcoded 8 in Boltz ``MSALayer``).
        outer_product_hidden_size: Hidden width of the outer-product-mean block.
        triangle_attention_hidden_size: Per-head width of the triangle attention (``pairwise_head_width``).
        num_triangle_attention_heads: Triangle-attention head count (``pairwise_num_heads``).
        use_paired_feature: Whether the paired-MSA feature is consumed.
        dropout: Pair-track dropout for the Pairformer (``pair_dropout``).
        msa_dropout: Dropout applied inside the MSA module.

    Examples:
        >>> config = Boltz2TrunkConfig()
        >>> config.num_pairformer_blocks
        64
        >>> config.num_msa_attention_heads
        8
    """

    num_pairformer_blocks: int = 64
    num_pair_attention_heads: int = 16
    num_msa_blocks: int = 4
    msa_size: int = 64
    msa_feature_size: int = 36  # one_hot(33) + has_deletion(1) + deletion_value(1) + is_paired(1)
    msa_head_dim: int = 32
    num_msa_attention_heads: int = 8  # MSALayer hardcodes PairWeightedAveraging num_heads=8
    outer_product_hidden_size: int = 32
    triangle_attention_hidden_size: int = 32  # pairwise_head_width
    num_triangle_attention_heads: int = 4  # pairwise_num_heads
    use_paired_feature: bool = True
    dropout: float = 0.25  # pair_dropout for the pairformer pair track
    msa_dropout: float = 0.15


class Boltz2InputEmbedderConfig(Config):
    r"""
    Boltz-2 input embedder (atom-attention encoder + token-level conditioning).

    The atom-attention encoder is a SEPARATE instance from the diffusion atom encoder (its own
    weights, ``structure_prediction=False`` so no ``s_to_c``/``z_to_p``/``r_to_q`` projections).
    The four ``add_*`` conditioning terms are all enabled in ``boltz2_conf.ckpt``.

    Attributes:
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size (``res_type_encoding``: ``Linear(33 -> single_size)``).
        msa_profile_size: Width of the MSA-profile feature (profile(33) + deletion_mean(1)).
        num_method_types: Number of experimental-method ids for method conditioning.
        num_molecule_types: Number of molecule types for molecule-type conditioning.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.

    Examples:
        >>> config = Boltz2InputEmbedderConfig()
        >>> config.num_atom_encoder_blocks
        3
        >>> config.add_method_conditioning
        True
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = 33  # res_type_encoding: Linear(33 -> single_size)
    msa_profile_size: int = 34  # msa_profile_encoding: Linear(profile(33) + deletion_mean(1))
    num_method_types: int = 12  # method_conditioning_init: Embedding(12, single_size)
    num_molecule_types: int = 4  # mol_type_conditioning_init: Embedding(4, single_size)
    add_method_conditioning: bool = True
    add_modified_flag: bool = True
    add_cyclic_flag: bool = True
    add_mol_type_feat: bool = True


class Boltz2TemplateConfig(Config):
    r"""
    Boltz-2 (V2) template embedding (on-the-fly distogram / unit-vector).

    ``a_proj`` input width = 2*vocab_size(33) + num_distogram_bins(38) + 5 = 109, where the
    +5 is cb_mask(1) + unit_vector(3) + frame_mask(1). The on-the-fly distogram buckets cdist
    into ``num_distogram_bins`` bins over (``min_distance``, ``max_distance``).

    Attributes:
        hidden_size: Width of the template pair hidden channel.
        num_blocks: Number of pair-only triangle blocks in the template stack.
        num_distogram_bins: Number of on-the-fly distogram bins.
        vocab_size: Residue-type vocabulary size feeding the template projection.
        expansion: Transition expansion factor (``transition_z`` hidden = ``hidden_size * 4``).
        triangle_attention_hidden_size: Per-head width of the template triangle attention.
        num_triangle_attention_heads: Template triangle-attention head count.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.

    Examples:
        >>> config = Boltz2TemplateConfig()
        >>> config.hidden_size
        64
        >>> config.num_distogram_bins
        38
    """

    hidden_size: int = 64
    num_blocks: int = 2
    num_distogram_bins: int = 38
    vocab_size: int = 33
    expansion: int = 4  # transition_z hidden = hidden_size*4 = 256
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75


class Boltz2DiffusionConfig(Config):
    r"""
    Boltz-2 diffusion / structure module (EDM score network).

    Attributes:
        num_token_transformer_blocks: Number of conditioned token-transformer blocks in the score model.
        num_token_transformer_heads: Head count for the token transformer.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks in the score model.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks in the score model.
        num_atom_attention_heads: Head count shared by the atom encoder and decoder.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_conditioning_transitions: Number of transition blocks in each diffusion conditioner.
        sigma_data: EDM data standard deviation used for preconditioning.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step.

    Examples:
        >>> config = Boltz2DiffusionConfig()
        >>> config.num_token_transformer_blocks
        24
        >>> config.sigma_data
        16.0
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


class Boltz2SamplerConfig(Config):
    r"""
    Boltz-2 EDM sampler (Euler, in-loop weighted rigid alignment).

    Attributes:
        num_sampling_steps: Number of EDM sampling steps.
        rho: Exponent of the Karras noise schedule.
        max_noise_level: Highest sigma at the start of the schedule.
        min_noise_level: Lowest non-zero sigma before the trailing zero.
        step_scale: Scale on the Euler update step used by the released public inference command.
        noise_scale: Multiplier on the injected churn noise used by the released public inference command.
        churn_rate: Stochastic churn added to each step above the churn threshold.

    Examples:
        >>> config = Boltz2SamplerConfig()
        >>> config.num_sampling_steps
        200
    """

    num_sampling_steps: int = 200
    rho: float = 7.0
    max_noise_level: float = 160.0
    # The released CLI replaces the older values serialized in the checkpoint
    # (4e-4, 1.0, 1.0) with Boltz2DiffusionParams for every Boltz-2 prediction.
    min_noise_level: float = 1e-4
    step_scale: float = 1.5
    noise_scale: float = 1.003
    churn_rate: float = 0.8


class Boltz2ConfidenceConfig(Config):
    r"""
    Boltz-2 confidence head (token-level, separate intra/inter PAE/PDE).

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the confidence head
            (``confidence_model_args.pairformer_args.num_blocks``).
        num_pair_attention_heads: Head count for the confidence pair-biased attention.
        num_distogram_bins: Number of distogram bins used to bin representative-atom distances.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.

    Examples:
        >>> config = Boltz2ConfidenceConfig()
        >>> config.num_pairformer_blocks
        8
    """

    num_pairformer_blocks: int = 8  # confidence_model_args.pairformer_args.num_blocks
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64


class Boltz2AffinityHeadConfig(Config):
    r"""
    Configure one Boltz-2 affinity head and its pair-only Pairformer stack.

    The public affinity checkpoint is a full Boltz-2 model with an extra masked pair head. These defaults follow
    the artifact shape: distogram-conditioned pair features, a pair-only Pairformer stack, then two scalar affinity
    branches.

    Attributes:
        num_pairformer_blocks: Number of pair-only Pairformer blocks in the affinity head.
        num_distogram_bins: Number of distogram bins used to condition the pair features.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_conditioning_transitions: Number of transition blocks in the pairwise conditioner.
        expansion: Transition expansion factor in the pairwise conditioner.
        triangle_attention_hidden_size: Per-head width of the affinity triangle attention.
        num_triangle_attention_heads: Affinity triangle-attention head count.
        dropout: Dropout applied in the affinity Pairformer stack.

    Examples:
        >>> config = Boltz2AffinityHeadConfig()
        >>> config.num_pairformer_blocks
        4
    """

    num_pairformer_blocks: int = 4
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_conditioning_transitions: int = 2
    expansion: int = 2
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    dropout: float = 0.25


class Boltz2Config(Config):
    r"""
    Boltz-2-shaped model configuration (folding + confidence).

    Aggregates the per-subsystem configs (input embedder, trunk, template, diffusion, sampler, confidence)
    with the top-level token/pair widths and architecture flags verified against ``boltz2_conf.ckpt``.

    Attributes:
        single_size: Width of the token (single) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single channel in the input embedder.
        atom_pair_hidden_size: Width of the atom pair channel in the input embedder.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_bins: Number of distogram / B-factor output bins.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        relative_position_size: Width of the relative-position feature vector.
        num_bond_types: Bond-type vocabulary size (``token_bonds_type``: ``Embedding(7, pair_size)``).
        recycling_steps: Number of trunk recycling passes.
        use_templates: Whether template embedding is enabled.
        predict_bfactor: Whether the B-factor head is instantiated.
        conditioning_cutoff_min: Lower distance cutoff for contact conditioning, in angstroms.
        conditioning_cutoff_max: Upper distance cutoff for contact conditioning, in angstroms.
        input_embedder: Input-embedder sub-config.
        trunk: Representation-trunk sub-config.
        template: Template-embedder sub-config.
        diffusion: Diffusion / structure-module sub-config.
        sampler: EDM-sampler sub-config.
        confidence: Confidence-head sub-config.

    Examples:
        >>> config = Boltz2Config()
        >>> config.single_size
        384
        >>> config.trunk.num_pairformer_blocks
        64
        >>> from ofoldx.models.boltz2 import Boltz2Model
        >>> model = Boltz2Model(config)  # doctest: +SKIP
    """

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    num_bins: int = 64
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_bond_types: int = 7  # token_bonds_type: Embedding(len(bond_types) + 1 = 7, pair_size)
    recycling_steps: int = 4

    use_templates: bool = True
    predict_bfactor: bool = True

    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    input_embedder: Boltz2InputEmbedderConfig = Boltz2InputEmbedderConfig()
    trunk: Boltz2TrunkConfig = Boltz2TrunkConfig()
    template: Boltz2TemplateConfig = Boltz2TemplateConfig()
    diffusion: Boltz2DiffusionConfig = Boltz2DiffusionConfig()
    sampler: Boltz2SamplerConfig = Boltz2SamplerConfig()
    confidence: Boltz2ConfidenceConfig = Boltz2ConfidenceConfig()


class Boltz2AffinityConfig(Boltz2Config):
    r"""
    Boltz-2 affinity-prediction configuration.

    The released affinity checkpoint carries two affinity modules: module 1 has 8 pair-only Pairformer layers,
    and module 2 has 4. They are one fixed architecture, not an optional ensemble mode.

    Attributes:
        recycling_steps: Released affinity-pass recycle count.
        affinity1: First ensemble affinity head (8 pair-only Pairformer blocks).
        affinity2: Second ensemble affinity head (4 pair-only Pairformer blocks).

    Examples:
        >>> config = Boltz2AffinityConfig()
        >>> config.affinity1.num_pairformer_blocks
        8
        >>> from ofoldx.models.boltz2 import Boltz2AffinityModel
        >>> model = Boltz2AffinityModel(config)  # doctest: +SKIP
    """

    use_templates: bool = False
    predict_bfactor: bool = False
    recycling_steps: int = 6

    affinity1: Boltz2AffinityHeadConfig = Boltz2AffinityHeadConfig({"num_pairformer_blocks": 8})
    affinity2: Boltz2AffinityHeadConfig = Boltz2AffinityHeadConfig({"num_pairformer_blocks": 4})
