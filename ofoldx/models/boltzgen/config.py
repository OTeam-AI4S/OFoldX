r"""
Configuration values for the BoltzGen structure, confidence, and inverse-folding architectures.
"""

from __future__ import annotations

from chanfig import Config


class BoltzGenInputEmbeddingsConfig(Config):
    r"""
    Configure BoltzGen atom and token input embeddings.

    Args:
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        vocab_size: Number of token types.
        msa_profile_size: Width of MSA profile features.
        num_method_types: Number of experimental-method categories.
        num_molecule_types: Number of molecule-type categories.
        add_method_conditioning: Whether to embed experimental-method categories.
        add_modified_flag: Whether to embed modified-residue flags.
        add_cyclic_flag: Whether to project cyclic-position features.
        add_molecule_type_feature: Whether to embed molecule-type categories.
        add_binding_specification: Whether to embed binding-site specifications.
        add_design_mask: Whether to embed structure-design masks.
        add_ss_specification: Whether to embed secondary-structure specifications.
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
    add_molecule_type_feature: bool = True
    add_binding_specification: bool = False
    add_design_mask: bool = False
    add_ss_specification: bool = False


class BoltzGenTrunkConfig(Config):
    r"""
    Configure the Pairformer and MSA trunk shared by BoltzGen structure-design variants.

    Args:
        num_trunk_blocks: Number of pair trunk blocks.
        num_pair_attention_heads: Number of pair-biased attention heads.
        num_msa_blocks: Number of MSA blocks.
        msa_size: Width of MSA representations.
        msa_head_dim: Per-head width of MSA attention.
        num_msa_attention_heads: Number of MSA attention heads.
        outer_product_hidden_size: Hidden width of the outer-product-mean projection.
        msa_triangle_attention_hidden_size: Per-head width of MSA pair-update
            triangle attention.
        num_msa_triangle_attention_heads: Number of MSA pair-update triangle
            attention heads.
        triangle_attention_hidden_size: Per-head width of main-trunk triangle
            attention.
        num_triangle_attention_heads: Number of main-trunk triangle-attention heads.
        use_paired_feature: Whether to consume paired-MSA features.
        use_miniformer: Whether to replace the full pair trunk with Miniformer blocks.
        msa_miniformer_blocks: Whether the MSA trunk uses Miniformer pair updates.
        pair_dropout: Dropout probability along the pair trunk.
        msa_pair_dropout: Dropout probability along MSA pair updates.
        msa_dropout: Dropout probability inside the MSA trunk.
        post_layer_norm: Whether each pair-trunk block normalizes its updated single track.
        use_single_to_pair: Whether Miniformer adds projected single states to its pair track.
    """

    num_trunk_blocks: int = 64
    num_pair_attention_heads: int = 16
    num_msa_blocks: int = 4
    msa_size: int = 64
    msa_head_dim: int = 32
    num_msa_attention_heads: int = 8
    outer_product_hidden_size: int = 32
    msa_triangle_attention_hidden_size: int = 32
    num_msa_triangle_attention_heads: int = 4
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    use_paired_feature: bool = True
    use_miniformer: bool = False
    msa_miniformer_blocks: bool = False
    pair_dropout: float = 0.25
    msa_pair_dropout: float = 0.25
    msa_dropout: float = 0.15
    post_layer_norm: bool = False
    use_single_to_pair: bool = False


class BoltzGenTemplateConfig(Config):
    r"""
    Configure geometry shared by template and token-distance conditioning.

    Args:
        num_distogram_bins: Number of distance bins.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        min_distance: Lower distance bound, in angstroms.
        max_distance: Upper distance bound, in angstroms.
    """

    num_distogram_bins: int = 38
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75


class BoltzGenDenoiserConfig(Config):
    r"""
    Configure the BoltzGen EDM score network.

    Args:
        num_token_encoder_blocks: Number of token-encoder blocks.
        num_token_encoder_attention_heads: Number of token-encoder attention heads.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_decoder_blocks: Number of atom decoder blocks.
        num_atom_attention_heads: Number of atom attention heads.
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
        num_conditioning_transitions: Number of transitions in each diffusion conditioner.
        transformer_size: Width of the diffusion Transformer. ``None`` uses twice
            the trunk single width.
        post_layer_norm: Whether denoiser atom-encoder blocks apply post-layer normalization.
        use_qk_norm: Whether denoiser attentions normalize projected query/key heads.
        gaussian_random_3d_encoding_dim: Width of the frozen random 3D coordinate encoding.
        predict_residue_type: Whether the atom decoder also predicts per-token residue identities.
        sigma_data: Data noise scale used by EDM preconditioning.
        alignment_reverse_diff: Whether to rigid-align each reverse-diffusion step.
    """

    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_decoder_blocks: int = 3
    num_atom_attention_heads: int = 4
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    num_conditioning_transitions: int = 2
    transformer_size: int | None = None
    post_layer_norm: bool = False
    use_qk_norm: bool = False
    gaussian_random_3d_encoding_dim: int = 0
    predict_residue_type: bool = False
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True


class BoltzGenSamplerConfig(Config):
    r"""
    Configure the BoltzGen reverse-diffusion schedule.

    Args:
        num_sampling_steps: Number of reverse-diffusion steps.
        sampling_schedule: Name of the noise schedule.
        rho: Exponent shaping noise-level spacing.
        max_noise_level: Highest noise level in the schedule.
        min_noise_level: Lowest non-zero noise level in the schedule.
        step_scale: Multiplier on each denoising step.
        noise_scale: Multiplier on churn-injected noise.
        churn_rate: Stochastic churn added during sampling.
        time_dilation: Dilation factor applied inside the configured schedule interval.
        time_dilation_start: Start of the time-dilation interval.
        time_dilation_end: End of the time-dilation interval.
    """

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


class BoltzGenStructureDesignConfig(Config):
    r"""
    Configure structure-design template and token-distance modules.

    Args:
        template_size: Width of structure-design template features.
        num_template_blocks: Number of structure-design template blocks.
        template_dropout: Dropout probability in template pair updates.
        template_post_layer_norm: Whether Miniformer template blocks apply post-layer normalization.
        template_miniformer_blocks: Whether templates use compact pair-only Miniformer blocks.
        token_distance_size: Width of token-distance features.
        num_token_distance_blocks: Number of token-distance blocks.
        token_distance_dropout: Dropout probability in token-distance pair updates.
        token_distance_post_layer_norm: Whether Miniformer token-distance blocks apply post-layer normalization.
        token_distance_miniformer_blocks: Whether token distances use compact pair-only Miniformer blocks.
        distance_gaussian_dim: Number of Gaussian distance features.
        use_token_distance: Whether to build token-distance conditioning.
        use_token_distance_features: Whether to add per-backbone-atom distance features to the distogram.
    """

    template_size: int = 64
    num_template_blocks: int = 2
    template_dropout: float = 0.25
    template_post_layer_norm: bool = False
    template_miniformer_blocks: bool = False
    token_distance_size: int = 64
    num_token_distance_blocks: int = 2
    token_distance_dropout: float = 0.25
    token_distance_post_layer_norm: bool = False
    token_distance_miniformer_blocks: bool = False
    distance_gaussian_dim: int = 32
    use_token_distance: bool = True
    use_token_distance_features: bool = True


class BoltzGenConfidenceHeadConfig(Config):
    r"""
    Configure the distinct BoltzGen confidence equation.

    Args:
        num_pairformer_blocks: Number of full Pairformer blocks.
        num_pair_attention_heads: Number of pair-biased single-attention heads.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        pair_dropout: Dropout probability along Pairformer pair updates.
        post_layer_norm: Whether Pairformer blocks normalize their updated single track.
        num_distogram_bins: Number of representative-atom distance-conditioning bins.
        max_distance: Upper distance-conditioning boundary, in angstroms.
        token_level_confidence: Whether pLDDT and resolved logits are token-level.
            ``False`` uses the released 23 atom-slot projections and realizes them
            on the input atom axis.
        add_single_input: Whether to add projected input singles to trunk singles.
        add_single_to_pair_product: Whether to add the multiplicative single-to-pair
            projection.
        use_separate_heads: Whether PAE and PDE use separate intra- and inter-chain
            projections.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        return_latent_features: Whether to retain confidence single and pair features
            in the typed output.
    """

    num_pairformer_blocks: int = 8
    num_pair_attention_heads: int = 16
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    pair_dropout: float = 0.25
    post_layer_norm: bool = False
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    token_level_confidence: bool = True
    add_single_input: bool = False
    add_single_to_pair_product: bool = False
    use_separate_heads: bool = False
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    return_latent_features: bool = False


class BoltzGenInverseFoldingModuleConfig(Config):
    r"""
    Configure the BoltzGen inverse-folding atom encoder and residue graph.

    Args:
        node_size: Width of projected node features.
        edge_size: Width of projected edge features.
        hidden_size: Width of graph hidden states.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of sequence decoder blocks.
        num_attention_heads: Number of graph attention heads.
        num_neighbors: Number of nearest neighbors in each residue graph.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
        dropout: Dropout probability in graph blocks.
        softmax_dropout: Probability of masking attention logits before destination-wise softmax.
        coordinate_noise: Training-time coordinate-noise standard deviation.
    """

    node_size: int = 128
    edge_size: int = 128
    hidden_size: int = 128
    num_encoder_blocks: int = 6
    num_decoder_blocks: int = 3
    num_attention_heads: int = 4
    num_neighbors: int = 30
    num_atom_encoder_blocks: int = 1
    num_atom_encoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    dropout: float = 0.1
    softmax_dropout: float = 0.2
    coordinate_noise: float = 0.2


class BoltzGenConfig(Config):
    r"""
    Configure the parameterized BoltzGen structure-design architecture.

    Args:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of atom single features in the input embedder.
        atom_pair_hidden_size: Width of atom pair features in the input embedder.
        atom_feature_size: Width of raw atom features.
        num_distogram_bins: Number of distogram output bins.
        fourier_embedding_size: Width of diffusion-noise Fourier features.
        relative_position_size: Width of relative-position features.
        num_bond_types: Number of token bond types.
        recycling_steps: Number of complete trunk passes.
        conditioning_cutoff_min: Lower contact-conditioning cutoff, in angstroms.
        conditioning_cutoff_max: Upper contact-conditioning cutoff, in angstroms.
        input_embeddings: Input embedder with binding and secondary-structure conditioning enabled.
        trunk: Pairformer and MSA trunk configuration.
        template: Shared template and token-distance geometry configuration.
        denoiser: EDM score-network configuration.
        sampler: Reverse-diffusion schedule configuration.
        design: Structure-design modules, including token-distance conditioning.

    Examples:
        >>> config = BoltzGenConfig(recycling_steps=2)
        >>> config.recycling_steps
        2
    """

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    num_distogram_bins: int = 64
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_bond_types: int = 7
    recycling_steps: int = 4
    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0

    input_embeddings: BoltzGenInputEmbeddingsConfig = BoltzGenInputEmbeddingsConfig(
        {
            "add_binding_specification": True,
            "add_design_mask": True,
            "add_ss_specification": True,
        }
    )
    trunk: BoltzGenTrunkConfig = BoltzGenTrunkConfig()
    template: BoltzGenTemplateConfig = BoltzGenTemplateConfig()
    denoiser: BoltzGenDenoiserConfig = BoltzGenDenoiserConfig()
    sampler: BoltzGenSamplerConfig = BoltzGenSamplerConfig()
    design: BoltzGenStructureDesignConfig = BoltzGenStructureDesignConfig()

    @classmethod
    def structuretrained_small(cls) -> BoltzGenConfig:
        r"""
        Return the reduced BoltzGen structure-design configuration.

        This variant preserves the single, pair, atom, and EDM dimensions while using a shallower graph:

        - Miniformer trunk (compact triangular pair updates) instead of the full Pairformer/MSA stack.
        - ``12`` Miniformer trunk blocks (vs ``64``) and ``3`` MSA blocks (vs ``4``).
        - Miniformer template pair updates.
        - ``8`` diffusion token-encoder blocks (vs ``24``).
        - No token-distance conditioning module.
        - No binding-specification / design-mask / secondary-structure input conditioning.

        Every difference is represented by an existing configuration field, so both variants use the same
        ``BoltzGenModel`` graph.

        Returns:
            Reduced structure-design configuration.
        """

        config = cls()
        config.input_embeddings.add_binding_specification = False
        config.input_embeddings.add_design_mask = False
        config.input_embeddings.add_ss_specification = False
        config.trunk.use_miniformer = True
        config.trunk.msa_miniformer_blocks = True
        config.trunk.num_trunk_blocks = 12
        config.trunk.num_msa_blocks = 3
        config.design.template_miniformer_blocks = True
        config.denoiser.num_token_encoder_blocks = 8
        config.design.use_token_distance = False
        return config


class BoltzGenConfidenceConfig(BoltzGenConfig):
    r"""
    Configure the distinct BoltzGen structure-and-confidence architecture.

    The inherited fields construct the exact BoltzGen structure trunk and sampler.
    ``confidence`` owns only the independently live confidence-head axes.

    Examples:
        >>> config = BoltzGenConfidenceConfig()
        >>> config.confidence.num_pairformer_blocks
        8
    """

    confidence: BoltzGenConfidenceHeadConfig = BoltzGenConfidenceHeadConfig()


class BoltzGenInverseFoldingConfig(Config):
    r"""
    Configure the distinct BoltzGen inverse-folding architecture.

    Args:
        single_size: Width of the token representation.
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        atom_feature_size: Width of raw atom features.
        input_embeddings: Atom and token input embedding configuration.
        inverse_folding: Inverse-folding atom encoder and residue graph configuration.

    Examples:
        >>> config = BoltzGenInverseFoldingConfig(single_size=256)
        >>> config.single_size
        256
    """

    single_size: int = 384
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    input_embeddings: BoltzGenInputEmbeddingsConfig = BoltzGenInputEmbeddingsConfig()
    inverse_folding: BoltzGenInverseFoldingModuleConfig = BoltzGenInverseFoldingModuleConfig()
