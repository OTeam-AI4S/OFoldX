r"""
Boltz-2 configuration.

Values are taken from the public ``boltz2_conf.ckpt`` ``hyper_parameters`` and verified against its
``state_dict`` (pairformer 64 blocks, MSA 4 blocks, diffusion score model 24 token-transformer blocks
with 3+3 atom encoder/decoder).
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


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
        operator_backend: Selector for the fused-operator backend (``"auto"`` picks OOps when available).
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
    operator_backend: str = "auto"
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
        operator_backend: Selector for the fused-operator backend.

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
    operator_backend: str = "auto"


class Boltz2SamplerConfig(Config):
    r"""
    Boltz-2 EDM sampler (Euler, in-loop weighted rigid alignment).

    Attributes:
        solver: Reverse-diffusion solver name.
        num_sampling_steps: Number of EDM sampling steps.
        rho: Exponent of the Karras noise schedule.
        max_noise_level: Highest sigma at the start of the schedule.
        min_noise_level: Lowest non-zero sigma before the trailing zero.
        step_scale: Scale on the Euler update step (from ``boltz2_conf.ckpt``).
        noise_scale: Multiplier on the injected churn noise (from ``boltz2_conf.ckpt``).
        churn_rate: Stochastic churn added to each step above the churn threshold.

    Examples:
        >>> config = Boltz2SamplerConfig()
        >>> config.solver
        'euler'
        >>> config.num_sampling_steps
        200
    """

    solver: str = "euler"
    num_sampling_steps: int = 200
    rho: float = 7.0
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    step_scale: float = 1.0
    noise_scale: float = 1.0
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
        num_bfactor_bins: Number of B-factor output bins.
        add_s_to_z_prod: Whether to add the single-to-pair outer product to the pair features.
        add_s_input_to_s: Whether to add the input single representation to the single features.
        add_z_input_to_z: Whether to add the input pair representation to the pair features.
        use_separate_heads: Whether intra- and inter-chain PAE/PDE use separate projections.
        token_level: Whether the head predicts token-level (rather than atom-level) confidence.

    Examples:
        >>> config = Boltz2ConfidenceConfig()
        >>> config.num_pairformer_blocks
        8
        >>> config.token_level
        True
    """

    num_pairformer_blocks: int = 8  # confidence_model_args.pairformer_args.num_blocks
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    num_bfactor_bins: int = 64
    add_s_to_z_prod: bool = True
    add_s_input_to_s: bool = True
    add_z_input_to_z: bool = True
    use_separate_heads: bool = True
    token_level: bool = True


class Boltz2AffinityHeadConfig(Config):
    r"""
    Configure one Boltz-2 affinity head: its pair-only Pairformer stack and molecular-weight correction.

    The public affinity checkpoint is a full Boltz-2 model with an extra masked pair head. These defaults follow
    the artifact shape: distogram-conditioned pair features, a pair-only Pairformer stack, then two scalar affinity
    branches.

    Attributes:
        num_pairformer_blocks: Number of pair-only Pairformer blocks in the affinity head.
        num_pair_attention_heads: Head count for the affinity pair-biased attention.
        num_distogram_bins: Number of distogram bins used to condition the pair features.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_conditioning_transitions: Number of transition blocks in the pairwise conditioner.
        expansion: Transition expansion factor in the pairwise conditioner.
        triangle_attention_hidden_size: Per-head width of the affinity triangle attention.
        num_triangle_attention_heads: Affinity triangle-attention head count.
        dropout: Dropout applied in the affinity Pairformer stack.
        molecular_weight_correction: Whether to apply the molecular-weight correction to the predicted value.
        molecular_weight_exponent: Exponent applied to the molecular weight in the correction term.
        molecular_weight_model_coefficient: Multiplier on the model prediction in the correction.
        molecular_weight_coefficient: Coefficient on the molecular-weight power term.
        molecular_weight_bias: Additive bias in the molecular-weight correction.

    Examples:
        >>> config = Boltz2AffinityHeadConfig()
        >>> config.num_pairformer_blocks
        4
        >>> config.molecular_weight_correction
        False
    """

    num_pairformer_blocks: int = 4
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_conditioning_transitions: int = 2
    expansion: int = 2
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    dropout: float = 0.0
    molecular_weight_correction: bool = False
    molecular_weight_exponent: float = 0.3
    molecular_weight_model_coefficient: float = 1.03525938
    molecular_weight_coefficient: float = -0.59992683
    molecular_weight_bias: float = 2.83288489


@MODEL_CONFIGS.register("boltz2")
class Boltz2Config(Config):
    r"""
    Boltz-2-shaped model configuration (folding + confidence).

    Aggregates the per-subsystem configs (input embedder, trunk, template, diffusion, sampler, confidence)
    with the top-level token/pair widths and architecture flags verified against ``boltz2_conf.ckpt``.

    Attributes:
        architecture: Registry key selecting this architecture (``"boltz2"``).
        single_size: Width of the token (single) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single channel in the input embedder.
        atom_pair_hidden_size: Width of the atom pair channel in the input embedder.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_bins: Number of distogram / B-factor output bins.
        num_distograms: Number of distogram heads.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        relative_position_size: Width of the relative-position feature vector.
        num_bond_types: Bond-type vocabulary size (``token_bonds_type``: ``Embedding(7, pair_size)``).
        initializer_range: Standard deviation for weight initialization.
        recycling_steps: Number of trunk recycling passes.
        recycling_detach: Whether to detach gradients across recycling passes during training.
        fix_sym_check: Whether the symmetry-check fix is enabled.
        cyclic_pos_enc: Whether cyclic position encoding is enabled.
        use_templates: Whether template embedding is enabled.
        use_templates_v2: Whether the V2 (on-the-fly) template embedder is used.
        bond_type_feature: Whether the bond-type pair feature is added.
        token_level_confidence: Whether confidence is predicted at token level.
        predict_bfactor: Whether the B-factor head is instantiated.
        add_s_to_z_prod: Whether to add the single-to-pair outer product to the pair features.
        add_s_input_to_s: Whether to add the input single representation to the single features.
        add_z_input_to_z: Whether to add the input pair representation to the pair features.
        aggregate_distogram: Whether the distogram is aggregated across heads.
        conditioning_cutoff_min: Lower distance cutoff for contact conditioning, in angstroms.
        conditioning_cutoff_max: Upper distance cutoff for contact conditioning, in angstroms.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        input_embedder: Input-embedder sub-config.
        trunk: Representation-trunk sub-config.
        template: Template-embedder sub-config.
        diffusion: Diffusion / structure-module sub-config.
        sampler: EDM-sampler sub-config.
        confidence: Confidence-head sub-config.

    Examples:
        >>> config = Boltz2Config()
        >>> config.architecture
        'boltz2'
        >>> config.single_size
        384
        >>> config.trunk.num_pairformer_blocks
        64
        >>> from ofoldx.models.boltz2 import Boltz2Model
        >>> model = Boltz2Model(config)  # doctest: +SKIP
    """

    architecture: str = "boltz2"

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    num_bins: int = 64
    num_distograms: int = 1
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_bond_types: int = 7  # token_bonds_type: Embedding(len(bond_types) + 1 = 7, pair_size)
    initializer_range: float = 0.02

    recycling_steps: int = 3
    recycling_detach: bool = True

    # Architecture flags (verified from the checkpoint ``hyper_parameters``).
    fix_sym_check: bool = True
    cyclic_pos_enc: bool = True
    use_templates: bool = True
    use_templates_v2: bool = True
    bond_type_feature: bool = True
    token_level_confidence: bool = True
    predict_bfactor: bool = True
    add_s_to_z_prod: bool = True
    add_s_input_to_s: bool = True
    add_z_input_to_z: bool = True
    aggregate_distogram: bool = True

    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    min_distance: float = 2.0
    max_distance: float = 22.0

    input_embedder: Boltz2InputEmbedderConfig = Boltz2InputEmbedderConfig()
    trunk: Boltz2TrunkConfig = Boltz2TrunkConfig()
    template: Boltz2TemplateConfig = Boltz2TemplateConfig()
    diffusion: Boltz2DiffusionConfig = Boltz2DiffusionConfig()
    sampler: Boltz2SamplerConfig = Boltz2SamplerConfig()
    confidence: Boltz2ConfidenceConfig = Boltz2ConfidenceConfig()


@MODEL_CONFIGS.register("boltz2_affinity")
class Boltz2AffinityConfig(Boltz2Config):
    r"""
    Boltz-2 affinity-prediction configuration.

    The public BoltzGen affinity checkpoint carries two affinity modules: module 1 has 8 pair-only Pairformer
    layers, and module 2 has 4. This config models that ensemble by default. The single ``affinity`` block remains
    available for non-ensemble experiments and owns the shared molecular-weight correction coefficients.

    Attributes:
        architecture: Registry key selecting this architecture (``"boltz2_affinity"``).
        affinity_ensemble: Whether to run the two-module affinity ensemble (module 1 + module 2).
        affinity: Shared affinity-head sub-config; owns the molecular-weight correction coefficients.
        affinity1: First ensemble affinity head (8 pair-only Pairformer blocks).
        affinity2: Second ensemble affinity head (4 pair-only Pairformer blocks).

    Examples:
        >>> config = Boltz2AffinityConfig()
        >>> config.architecture
        'boltz2_affinity'
        >>> config.affinity_ensemble
        True
        >>> from ofoldx.models.boltz2 import Boltz2AffinityModel
        >>> model = Boltz2AffinityModel(config)  # doctest: +SKIP
    """

    architecture: str = "boltz2_affinity"
    use_templates: bool = False
    use_templates_v2: bool = False
    predict_bfactor: bool = False

    affinity_ensemble: bool = True
    affinity: Boltz2AffinityHeadConfig = Boltz2AffinityHeadConfig()
    affinity1: Boltz2AffinityHeadConfig = Boltz2AffinityHeadConfig({"num_pairformer_blocks": 8})
    affinity2: Boltz2AffinityHeadConfig = Boltz2AffinityHeadConfig({"num_pairformer_blocks": 4})
