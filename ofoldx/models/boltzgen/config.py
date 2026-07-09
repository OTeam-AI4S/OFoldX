r"""
BoltzGen configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


class BoltzGenTrunkConfig(Config):
    r"""
    BoltzGen representation-trunk configuration.

    Groups the settings for the shared Pairformer / MSA trunk that refines the single and pair representations
    before diffusion. The ``use_miniformer`` / ``msa_miniformer_blocks`` toggles select the compact Miniformer
    variants used by the structure-trained-small artifact.

    Attributes:
        num_pairformer_blocks: Number of Pairformer (or Miniformer) blocks in the trunk stack.
        num_pair_attention_heads: Number of heads in the pair-biased single-attention of each trunk block.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_feature_size: Width of the raw per-column MSA input features.
        msa_head_dim: Per-head dimension of the MSA pair-weighted-averaging attention.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted-averaging attention.
        outer_product_hidden_size: Hidden width of the MSA outer-product-mean projection.
        triangle_attention_hidden_size: Per-head width of the trunk triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads in the trunk.
        use_paired_feature: Whether the MSA module consumes the paired-MSA indicator feature.
        use_miniformer: Whether the trunk stack uses compact Miniformer blocks instead of Pairformer blocks.
        msa_miniformer_blocks: Whether the MSA module uses MiniformerNoSeq pair updates.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
        dropout: Row-wise dropout probability on the pair representation.
        msa_dropout: Row-wise dropout probability on the MSA representation.

    Examples:
        >>> config = BoltzGenTrunkConfig()
        >>> config.num_pairformer_blocks
        64
        >>> config.operator_backend
        'auto'
    """

    num_pairformer_blocks: int = 64
    num_pair_attention_heads: int = 16
    num_msa_blocks: int = 4
    msa_size: int = 64
    msa_feature_size: int = 36
    msa_head_dim: int = 32
    num_msa_attention_heads: int = 8
    outer_product_hidden_size: int = 32
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    use_paired_feature: bool = True
    use_miniformer: bool = False
    msa_miniformer_blocks: bool = False
    operator_backend: str = "auto"
    dropout: float = 0.25
    msa_dropout: float = 0.15


class BoltzGenInputEmbedderConfig(Config):
    r"""
    BoltzGen input-embedder configuration.

    Controls the atom-conditioned input embedder that composes atom-derived token states with token-level
    encodings and the optional design-conditioning embeddings. The ``add_*`` flags gate individual conditioning
    channels; ``add_binding_specification`` / ``add_ss_specification`` are the structure-design-only channels.

    Attributes:
        num_atom_encoder_blocks: Number of blocks in the windowed atom-attention encoder.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        vocab_size: Number of residue-type tokens in the model vocabulary.
        msa_profile_size: Width of the single-sequence MSA-profile input feature.
        num_method_types: Number of experimental-method conditioning ids.
        num_molecule_types: Number of molecule-type (chain-type) conditioning ids.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.
        add_binding_specification: Whether to add the binding-specification / design-mask conditioning.
        add_ss_specification: Whether to add the secondary-structure conditioning.

    Examples:
        >>> config = BoltzGenInputEmbedderConfig()
        >>> config.vocab_size
        33
        >>> config.add_binding_specification
        False
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = 33
    msa_profile_size: int = 34
    num_method_types: int = 12
    num_molecule_types: int = 4
    add_method_conditioning: bool = True
    add_modified_flag: bool = True
    add_cyclic_flag: bool = True
    add_mol_type_feat: bool = True
    add_binding_specification: bool = False
    add_ss_specification: bool = False


class BoltzGenTemplateConfig(Config):
    r"""
    BoltzGen template-embedding configuration.

    Controls the v1 template module that turns on-the-fly template geometry (CB distogram, per-frame CA unit
    vectors, and masks) into an additive pair update. ``min_distance`` / ``max_distance`` set the distogram range.

    Attributes:
        hidden_size: Width of the template hidden channel.
        num_blocks: Number of pair-only pairformer blocks in the template stack.
        num_distogram_bins: Number of template CB-distance distogram bins.
        vocab_size: Number of residue-type tokens used by the template residue-type projection.
        expansion: Feed-forward expansion factor inside the template blocks.
        triangle_attention_hidden_size: Per-head width of the template triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads in the template stack.
        min_distance: Lower edge (angstrom) of the template distogram binning.
        max_distance: Upper edge (angstrom) of the template distogram binning.
        miniformer_blocks: Whether the template stack uses compact Miniformer pair updates.

    Examples:
        >>> config = BoltzGenTemplateConfig()
        >>> config.num_blocks
        2
        >>> config.num_distogram_bins
        38
    """

    hidden_size: int = 64
    num_blocks: int = 2
    num_distogram_bins: int = 38
    vocab_size: int = 33
    expansion: int = 4
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75
    miniformer_blocks: bool = False


class BoltzGenDiffusionConfig(Config):
    r"""
    BoltzGen diffusion-module configuration.

    Controls the EDM score network that maps noisy atom positions and a noise level to a per-atom position update:
    the atom encoder / decoder, the conditioned token transformer, and the diffusion conditioning.

    Attributes:
        token_hidden_size: Width of the token channel inside the diffusion token transformer.
        num_token_transformer_blocks: Number of conditioned token-transformer blocks.
        num_token_transformer_heads: Number of attention heads in the token transformer.
        num_atom_encoder_blocks: Number of blocks in the diffusion atom encoder.
        num_atom_decoder_blocks: Number of blocks in the diffusion atom decoder.
        num_atom_attention_heads: Number of attention heads per atom encoder/decoder block.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair hidden channel.
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        conditioning_transition_layers: Number of transition layers in the diffusion conditioning.
        sigma_data: EDM data-scale hyperparameter used by the preconditioner.
        alignment_reverse_diff: Whether to rigidly re-align the noisy coords onto the denoised coords each step.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).

    Examples:
        >>> config = BoltzGenDiffusionConfig()
        >>> config.num_token_transformer_blocks
        24
        >>> config.sigma_data
        16.0
    """

    token_hidden_size: int = 768
    num_token_transformer_blocks: int = 24
    num_token_transformer_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_decoder_blocks: int = 3
    num_atom_attention_heads: int = 4
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    conditioning_transition_layers: int = 2
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True
    operator_backend: str = "auto"


class BoltzGenSamplerConfig(Config):
    r"""
    BoltzGen EDM-sampler configuration.

    Controls the reverse-diffusion sampling schedule and stochastic-churn hyperparameters of the EDM sampler.

    Attributes:
        solver: ODE solver name for the reverse-diffusion integration.
        num_sampling_steps: Number of denoising steps in the sampling schedule.
        sampling_schedule: Noise schedule variant, either ``"af3"`` or ``"dilated"``.
        rho: Karras schedule ``rho`` exponent controlling the noise-level spacing.
        max_noise_level: Highest noise level of the schedule (before ``sigma_data`` scaling).
        min_noise_level: Lowest non-zero noise level of the schedule.
        step_scale: Multiplier on the deterministic step (over-/under-shoot control).
        noise_scale: Multiplier on the injected stochastic-churn noise.
        churn_rate: Fraction of extra noise (``gamma``) injected per step above the churn floor.
        time_dilation: Dilation factor for the ``"dilated"`` schedule.
        time_dilation_start: Start of the normalized schedule interval to dilate.
        time_dilation_end: End of the normalized schedule interval to dilate.

    Examples:
        >>> config = BoltzGenSamplerConfig()
        >>> config.num_sampling_steps
        200
        >>> config.solver
        'euler'
    """

    solver: str = "euler"
    num_sampling_steps: int = 200
    sampling_schedule: str = "dilated"
    rho: float = 7.0
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    step_scale: float = 1.8
    noise_scale: float = 0.95
    churn_rate: float = 0.8
    time_dilation: float = 2.667
    time_dilation_start: float = 0.6
    time_dilation_end: float = 0.8


class BoltzGenConfidenceConfig(Config):
    r"""
    BoltzGen confidence-head configuration.

    Controls the opt-in Boltz-2-shaped confidence head that scores sampled coordinates into pLDDT / pAE / pDE /
    resolved logits. Only instantiated when ``BoltzGenConfig.use_confidence_head`` is set.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the confidence head.
        num_pair_attention_heads: Number of pair-biased single-attention heads per block.
        num_distogram_bins: Number of distance-distogram bins built from the sampled coords.
        max_distance: Upper edge (angstrom) of the confidence distogram binning.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        num_bfactor_bins: Number of B-factor output bins.
        add_s_to_z_prod: Whether to add the single-to-pair outer-product term.
        add_s_input_to_s: Whether to add the input-single projection back into the single representation.
        add_z_input_to_z: Whether to add the input-pair projection back into the pair representation.
        use_separate_heads: Whether intra- and inter-chain errors use separate projection heads.
        token_level: Whether confidence is scored at the token level (vs atom level).

    Examples:
        >>> config = BoltzGenConfidenceConfig()
        >>> config.num_pairformer_blocks
        8
        >>> config.num_plddt_bins
        50
    """

    num_pairformer_blocks: int = 8
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


class BoltzGenAffinityConfig(Config):
    r"""
    BoltzGen / Boltz-2 affinity-head configuration.

    Mirrors the public ``boltz2_aff`` artifact: a distogram-conditioned pair-only Pairformer stack followed by
    two scalar affinity branches, optionally in a two-module ensemble with a molecular-weight correction. Only
    instantiated when ``BoltzGenConfig.use_affinity_head`` is set.

    Attributes:
        ensemble: Whether to build a two-module ensemble (averaging value and binder probability).
        num_pairformer_blocks: Pairformer depth for the single (non-ensemble) affinity module.
        num_pairformer_blocks1: Pairformer depth for the first ensemble module.
        num_pairformer_blocks2: Pairformer depth for the second ensemble module.
        num_pair_attention_heads: Number of pair-biased attention heads per block.
        num_distogram_bins: Number of distance-distogram bins built from the sampled coords.
        max_distance: Upper edge (angstrom) of the affinity distogram binning.
        num_conditioning_transitions: Number of transition layers in the pairwise conditioner.
        expansion: Feed-forward / conditioning transition expansion factor.
        triangle_attention_hidden_size: Per-head width of the affinity triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads in the affinity stack.
        dropout: Dropout probability inside the affinity Pairformer.
        molecular_weight_correction: Whether to apply the molecular-weight correction to the predicted value.
        molecular_weight_exponent: Exponent ``e`` in the ``a*value + b*MW**e + c`` correction.
        molecular_weight_model_coefficient: Coefficient ``a`` multiplying the raw predicted value.
        molecular_weight_coefficient: Coefficient ``b`` multiplying ``MW**e``.
        molecular_weight_bias: Additive bias ``c`` in the correction.

    Examples:
        >>> config = BoltzGenAffinityConfig()
        >>> config.ensemble
        True
        >>> config.num_pairformer_blocks1
        8
    """

    ensemble: bool = True
    num_pairformer_blocks: int = 4
    num_pairformer_blocks1: int = 8
    num_pairformer_blocks2: int = 4
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_conditioning_transitions: int = 2
    expansion: int = 2
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    dropout: float = 0.0
    molecular_weight_correction: bool = True
    molecular_weight_exponent: float = 0.3
    molecular_weight_model_coefficient: float = 1.03525938
    molecular_weight_coefficient: float = -0.59992683
    molecular_weight_bias: float = 2.83288489


class BoltzGenStructureDesignConfig(Config):
    r"""
    BoltzGen structure-design-module configuration.

    Controls the design-specific trunk additions: the v1 template module and the optional token-distance
    conditioning module. ``use_token_distance`` is disabled by the structure-trained-small artifact.

    Attributes:
        template_size: Hidden width of the v1 template module.
        template_blocks: Number of pair-only pairformer blocks in the template module.
        token_distance_size: Hidden width of the token-distance conditioning module.
        token_distance_blocks: Number of pair-only pairformer blocks in the token-distance module.
        distance_gaussian_dim: Number of Gaussian basis functions in the token-distance encoder.
        use_token_distance: Whether the token-distance conditioning module is instantiated and applied.

    Examples:
        >>> config = BoltzGenStructureDesignConfig()
        >>> config.template_blocks
        2
        >>> config.use_token_distance
        True
    """

    template_size: int = 64
    template_blocks: int = 2
    token_distance_size: int = 64
    token_distance_blocks: int = 2
    distance_gaussian_dim: int = 32
    use_token_distance: bool = True


class BoltzGenIFoldModuleConfig(Config):
    r"""
    BoltzGen inverse-folding sequence-design-module configuration.

    Controls the kNN-graph encoder / decoder that designs residue tokens from a conditioned structure, plus the
    constraint-relaxation controls consumed by the decoder ``sample`` path.

    Attributes:
        node_size: Width of the GNN node channel.
        edge_size: Width of the GNN edge channel.
        hidden_size: Width of the GNN attention hidden channel.
        num_encoder_layers: Number of graph-attention encoder layers.
        num_decoder_layers: Number of graph-attention decoder layers.
        num_attention_heads: Number of attention heads per graph-attention layer.
        k_neighbors: Number of nearest neighbours per token in the kNN graph.
        num_atom_encoder_blocks: Number of blocks in the inverse-folding atom encoder.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
        inverse_fold_restriction: Canonical amino acids (three-letter CCD codes) globally blocked during decoding.
        tie_symmetric_sequences: Whether to tie homomer copies during autoregressive decoding.
        constraint_inf: Additive logit penalty magnitude applied to a blocked amino acid.

    Examples:
        >>> config = BoltzGenIFoldModuleConfig()
        >>> config.num_encoder_layers
        6
        >>> config.tie_symmetric_sequences
        False
    """

    node_size: int = 128
    edge_size: int = 128
    hidden_size: int = 128
    num_encoder_layers: int = 6
    num_decoder_layers: int = 3
    num_attention_heads: int = 4
    k_neighbors: int = 30
    num_atom_encoder_blocks: int = 1
    num_atom_encoder_attention_heads: int = 4
    operator_backend: str = "auto"
    # Constraint-relaxation controls consumed by the decoder ``sample`` path (upstream
    # ``InverseFoldingDecoder``). ``inverse_fold_restriction`` globally blocks the listed canonical amino acids
    # (three-letter CCD codes, e.g. ``["CYS"]``); ``tie_symmetric_sequences`` ties homomer copies during
    # autoregressive decoding; ``constraint_inf`` is the additive logit penalty for a blocked amino acid.
    inverse_fold_restriction: list = []  # noqa: RUF012
    tie_symmetric_sequences: bool = False
    constraint_inf: float = 1e6


@MODEL_CONFIGS.register("boltzgen")
class BoltzGenConfig(Config):
    r"""
    BoltzGen structure-generating design-model configuration.

    Top-level configuration for [`BoltzGenDesigner`][ofoldx.models.boltzgen.model.BoltzGenDesigner].
    Carries the shared representation widths and trunk/diffusion knobs plus the design-specific submodule configs
    (input embedder, template, token distance) and opt-in scoring heads. The registered subclasses select the
    published BoltzGen-1 artifacts.

    Attributes:
        architecture: Registry key selecting this architecture (``"boltzgen"``).
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair hidden channel.
        atom_feature_size: Width of the per-atom reference input features.
        num_distogram_bins: Number of trunk distogram output bins.
        num_distograms: Number of distogram heads.
        fourier_embedding_size: Width of the diffusion noise-level Fourier embedding.
        relative_position_size: Width of the relative-position input features.
        num_bond_types: Number of token bond-type ids.
        initializer_range: Standard deviation of the weight initializer.
        recycling_steps: Number of trunk recycling passes.
        recycling_detach: Whether to detach gradients across recycling iterations during training.
        use_confidence_head: Whether to instantiate the opt-in Boltz-2 confidence head.
        use_affinity_head: Whether to instantiate the opt-in Boltz-2 affinity head.
        fix_sym_check: Whether to apply the symmetry-check fix in relative-position features.
        cyclic_pos_enc: Whether to enable cyclic positional encoding.
        use_templates: Whether the trunk consumes template features.
        use_templates_v2: Whether templates use the v2 featurization.
        bond_type_feature: Whether to add the token bond-type embedding.
        token_level_confidence: Whether confidence is scored at the token level.
        predict_bfactor: Whether to predict per-token B-factor logits.
        add_s_to_z_prod: Whether to add the single-to-pair outer-product term.
        add_s_input_to_s: Whether to add the input-single projection back into the single representation.
        add_z_input_to_z: Whether to add the input-pair projection back into the pair representation.
        aggregate_distogram: Whether to aggregate the distogram across distogram heads.
        conditioning_cutoff_min: Lower distance cutoff (angstrom) for contact conditioning.
        conditioning_cutoff_max: Upper distance cutoff (angstrom) for contact conditioning.
        min_distance: Lower edge (angstrom) of the trunk distogram binning.
        max_distance: Upper edge (angstrom) of the trunk distogram binning.
        input_embedder: Input-embedder sub-configuration.
        trunk: Representation-trunk sub-configuration.
        template: Template-embedding sub-configuration.
        diffusion: Diffusion-module sub-configuration.
        sampler: EDM-sampler sub-configuration.
        confidence: Confidence-head sub-configuration.
        affinity: Affinity-head sub-configuration.
        design: Structure-design-module sub-configuration.
        ifold: Inverse-folding-module sub-configuration.

    Examples:
        >>> config = BoltzGenConfig()
        >>> config.architecture
        'boltzgen'
        >>> config.single_size
        384
        >>> model = BoltzGenDesigner(config)  # doctest: +SKIP
    """

    architecture: str = "boltzgen"

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    num_distogram_bins: int = 64
    num_distograms: int = 1
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_bond_types: int = 7
    initializer_range: float = 0.02

    recycling_steps: int = 3
    recycling_detach: bool = True

    # Opt-in scoring heads. The design-only generator omits both (base parameter count 482,175,298); enabling
    # either adds the corresponding Boltz-2-shaped head so the ``boltz2_conf`` / ``boltz2_aff`` weights can be
    # loaded and pLDDT/pAE/pDE/pTM/ipTM or affinity scores produced.
    use_confidence_head: bool = False
    use_affinity_head: bool = False

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

    input_embedder: BoltzGenInputEmbedderConfig = BoltzGenInputEmbedderConfig(
        {"add_binding_specification": True, "add_ss_specification": True}
    )
    trunk: BoltzGenTrunkConfig = BoltzGenTrunkConfig()
    template: BoltzGenTemplateConfig = BoltzGenTemplateConfig()
    diffusion: BoltzGenDiffusionConfig = BoltzGenDiffusionConfig()
    sampler: BoltzGenSamplerConfig = BoltzGenSamplerConfig()
    confidence: BoltzGenConfidenceConfig = BoltzGenConfidenceConfig()
    affinity: BoltzGenAffinityConfig = BoltzGenAffinityConfig()
    design: BoltzGenStructureDesignConfig = BoltzGenStructureDesignConfig()
    ifold: BoltzGenIFoldModuleConfig = BoltzGenIFoldModuleConfig()


@MODEL_CONFIGS.register("boltzgen_diverse")
class BoltzGenDiverseConfig(BoltzGenConfig):
    r"""
    BoltzGen-1 diverse structure-generation artifact configuration.

    Inherits every field of [`BoltzGenConfig`][ofoldx.models.boltzgen.config.BoltzGenConfig]; only the
    ``architecture`` registry key changes, selecting the published ``boltzgen1_diverse`` weights.

    Attributes:
        architecture: Registry key selecting this artifact (``"boltzgen_diverse"``).

    Examples:
        >>> config = BoltzGenDiverseConfig()
        >>> config.architecture
        'boltzgen_diverse'
    """

    architecture: str = "boltzgen_diverse"


@MODEL_CONFIGS.register("boltzgen_adherence")
class BoltzGenAdherenceConfig(BoltzGenConfig):
    r"""
    BoltzGen-1 adherence structure-generation artifact configuration.

    Inherits every field of [`BoltzGenConfig`][ofoldx.models.boltzgen.config.BoltzGenConfig]; only the
    ``architecture`` registry key changes, selecting the published ``boltzgen1_adherence`` weights.

    Attributes:
        architecture: Registry key selecting this artifact (``"boltzgen_adherence"``).

    Examples:
        >>> config = BoltzGenAdherenceConfig()
        >>> config.architecture
        'boltzgen_adherence'
    """

    architecture: str = "boltzgen_adherence"


@MODEL_CONFIGS.register("boltzgen_structuretrained_small")
class BoltzGenStructureTrainedSmallConfig(BoltzGenConfig):
    r"""
    BoltzGen-1 structure-trained-small structure-generation artifact configuration.

    A smaller published artifact: overrides the input-embedder, trunk, template, diffusion, and design
    sub-configs to the Miniformer variants (shallower stacks, no design conditioning, no token distance).

    Attributes:
        architecture: Registry key selecting this artifact (``"boltzgen_structuretrained_small"``).
        input_embedder: Input embedder without design conditioning.
        trunk: Shallower Miniformer trunk with Miniformer MSA blocks.
        template: Template module using Miniformer pair updates.
        diffusion: Diffusion module with fewer token-transformer blocks.
        design: Structure-design module with token-distance conditioning disabled.

    Examples:
        >>> config = BoltzGenStructureTrainedSmallConfig()
        >>> config.architecture
        'boltzgen_structuretrained_small'
        >>> config.trunk.use_miniformer
        True
    """

    architecture: str = "boltzgen_structuretrained_small"
    input_embedder: BoltzGenInputEmbedderConfig = BoltzGenInputEmbedderConfig(
        {"add_binding_specification": False, "add_ss_specification": False}
    )
    trunk: BoltzGenTrunkConfig = BoltzGenTrunkConfig(
        {"num_pairformer_blocks": 12, "num_msa_blocks": 3, "use_miniformer": True, "msa_miniformer_blocks": True}
    )
    template: BoltzGenTemplateConfig = BoltzGenTemplateConfig({"miniformer_blocks": True})
    diffusion: BoltzGenDiffusionConfig = BoltzGenDiffusionConfig({"num_token_transformer_blocks": 8})
    design: BoltzGenStructureDesignConfig = BoltzGenStructureDesignConfig({"use_token_distance": False})


@MODEL_CONFIGS.register("boltzgen_ifold")
class BoltzGenIFoldConfig(BoltzGenConfig):
    r"""
    BoltzGen inverse-folding design-generator configuration.

    Configuration for [`BoltzGenInverseFoldDesigner`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldDesigner].
    Inherits every field of [`BoltzGenConfig`][ofoldx.models.boltzgen.config.BoltzGenConfig] (the ``ifold`` sub-config
    carries the encoder/decoder settings) and resets the input embedder to the base variant without design
    conditioning.

    Attributes:
        architecture: Registry key selecting this architecture (``"boltzgen_ifold"``).
        input_embedder: Base input embedder without design conditioning.

    Examples:
        >>> config = BoltzGenIFoldConfig()
        >>> config.architecture
        'boltzgen_ifold'
        >>> config.ifold.num_encoder_layers
        6
    """

    architecture: str = "boltzgen_ifold"
    input_embedder: BoltzGenInputEmbedderConfig = BoltzGenInputEmbedderConfig()
