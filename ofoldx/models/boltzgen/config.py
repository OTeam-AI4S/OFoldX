r"""
Configuration values for the two BoltzGen architectures.
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
        add_binding_specification: Whether to embed binding-site specifications.
        add_ss_specification: Whether to embed secondary-structure specifications.
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = 33
    msa_profile_size: int = 34
    num_method_types: int = 12
    num_molecule_types: int = 4
    add_binding_specification: bool = False
    add_ss_specification: bool = False


class BoltzGenTrunkConfig(Config):
    r"""
    Configure the Pairformer and MSA trunk shared by BoltzGen structure-design variants.

    Args:
        num_trunk_blocks: Number of pair trunk blocks.
        num_pair_attention_heads: Number of pair-biased attention heads.
        num_msa_blocks: Number of MSA blocks.
        msa_size: Width of MSA representations.
        msa_feature_size: Width of raw MSA features.
        msa_head_dim: Per-head width of MSA attention.
        num_msa_attention_heads: Number of MSA attention heads.
        outer_product_hidden_size: Hidden width of the outer-product-mean projection.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        use_paired_feature: Whether to consume paired-MSA features.
        use_miniformer: Whether to replace the full pair trunk with Miniformer blocks.
        msa_miniformer_blocks: Whether the MSA trunk uses Miniformer pair updates.
        pair_dropout: Dropout probability along the pair trunk.
        msa_dropout: Dropout probability inside the MSA trunk.
    """

    num_trunk_blocks: int = 64
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
    pair_dropout: float = 0.25
    msa_dropout: float = 0.15


class BoltzGenTemplateConfig(Config):
    r"""
    Configure geometry shared by template and token-distance conditioning.

    Args:
        num_distogram_bins: Number of distance bins.
        triangle_attention_hidden_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        min_distance: Lower distance bound, in angstroms.
        max_distance: Upper distance bound, in angstroms.
        miniformer_blocks: Whether to use Miniformer template pair updates.
    """

    num_distogram_bins: int = 38
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75
    miniformer_blocks: bool = False


class BoltzGenDenoiserConfig(Config):
    r"""
    Configure the BoltzGen EDM score network.

    Args:
        num_token_encoder_blocks: Number of token transformer blocks.
        num_token_encoder_attention_heads: Number of token transformer attention heads.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_decoder_blocks: Number of atom decoder blocks.
        num_atom_attention_heads: Number of atom attention heads.
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
        num_conditioning_transitions: Number of transitions in each diffusion conditioner.
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
        token_distance_size: Width of token-distance features.
        num_token_distance_blocks: Number of token-distance blocks.
        distance_gaussian_dim: Number of Gaussian distance features.
        use_token_distance: Whether to build token-distance conditioning.
    """

    template_size: int = 64
    num_template_blocks: int = 2
    token_distance_size: int = 64
    num_token_distance_blocks: int = 2
    distance_gaussian_dim: int = 32
    use_token_distance: bool = True


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
        recycling_steps: Number of trunk recycling passes.
        conditioning_cutoff_min: Lower contact-conditioning cutoff, in angstroms.
        conditioning_cutoff_max: Upper contact-conditioning cutoff, in angstroms.
        input_embeddings: Input embedder with binding and secondary-structure conditioning enabled.
        trunk: Pairformer and MSA trunk configuration.
        template: Shared template and token-distance geometry configuration.
        denoiser: EDM score-network configuration.
        sampler: Reverse-diffusion schedule configuration.
        design: Structure-design modules, including token-distance conditioning.
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
    recycling_steps: int = 3
    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0

    input_embeddings: BoltzGenInputEmbeddingsConfig = BoltzGenInputEmbeddingsConfig(
        {"add_binding_specification": True, "add_ss_specification": True}
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
        - ``12`` Miniformer pairformer blocks (vs ``64``) and ``3`` MSA blocks (vs ``4``).
        - Miniformer template pair updates.
        - ``8`` diffusion token-transformer blocks (vs ``24``).
        - No token-distance conditioning module.
        - No binding-specification / design-mask / secondary-structure input conditioning.

        Every difference is represented by an existing configuration field, so both variants use the same
        ``BoltzGenModel`` graph.

        Returns:
            Reduced structure-design configuration.
        """

        config = cls()
        config.input_embeddings.add_binding_specification = False
        config.input_embeddings.add_ss_specification = False
        config.trunk.use_miniformer = True
        config.trunk.msa_miniformer_blocks = True
        config.trunk.num_trunk_blocks = 12
        config.trunk.num_msa_blocks = 3
        config.template.miniformer_blocks = True
        config.denoiser.num_token_encoder_blocks = 8
        config.design.use_token_distance = False
        return config


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
    """

    single_size: int = 384
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    input_embeddings: BoltzGenInputEmbeddingsConfig = BoltzGenInputEmbeddingsConfig()
    inverse_folding: BoltzGenInverseFoldingModuleConfig = BoltzGenInverseFoldingModuleConfig()
