r"""Configuration values for the two BoltzGen architectures."""

from __future__ import annotations

from chanfig import Config


class BoltzGenTrunkConfig(Config):
    r"""Pairformer/MSA graph parameters shared by structure-design variants."""

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
    dropout: float = 0.25
    msa_dropout: float = 0.15


class BoltzGenInputEmbedderConfig(Config):
    r"""Atom/token input-embedder graph parameters."""

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = 33
    msa_profile_size: int = 34
    num_method_types: int = 12
    num_molecule_types: int = 4
    add_binding_specification: bool = False
    add_ss_specification: bool = False


class BoltzGenTemplateConfig(Config):
    r"""Geometry and block parameters shared by template and token-distance conditioning."""

    num_distogram_bins: int = 38
    triangle_attention_hidden_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75
    miniformer_blocks: bool = False


class BoltzGenDiffusionConfig(Config):
    r"""EDM score-network graph parameters."""

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


class BoltzGenSamplerConfig(Config):
    r"""Released reverse-diffusion schedule and solver parameters."""

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
    r"""Structure-design-only template and token-distance graph parameters."""

    template_size: int = 64
    template_blocks: int = 2
    token_distance_size: int = 64
    token_distance_blocks: int = 2
    distance_gaussian_dim: int = 32
    use_token_distance: bool = True


class BoltzGenIFoldModuleConfig(Config):
    r"""Inverse-folding atom encoder, graph encoder, and decoder parameters."""

    node_size: int = 128
    edge_size: int = 128
    hidden_size: int = 128
    num_encoder_layers: int = 6
    num_decoder_layers: int = 3
    num_attention_heads: int = 4
    num_neighbors: int = 30
    num_atom_encoder_blocks: int = 1
    num_atom_encoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    dropout: float = 0.1
    softmax_dropout: float = 0.2
    inverse_fold_noise: float = 0.2


class BoltzGenConfig(Config):
    r"""Model config for the parameterized BoltzGen structure-design architecture."""

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

    input_embedder: BoltzGenInputEmbedderConfig = BoltzGenInputEmbedderConfig(
        {"add_binding_specification": True, "add_ss_specification": True}
    )
    trunk: BoltzGenTrunkConfig = BoltzGenTrunkConfig()
    template: BoltzGenTemplateConfig = BoltzGenTemplateConfig()
    diffusion: BoltzGenDiffusionConfig = BoltzGenDiffusionConfig()
    sampler: BoltzGenSamplerConfig = BoltzGenSamplerConfig()
    design: BoltzGenStructureDesignConfig = BoltzGenStructureDesignConfig()

    @classmethod
    def structuretrained_small(cls) -> BoltzGenConfig:
        r"""Config for the reduced ``boltzgen1_structuretrained_small`` structure-design checkpoint.

        The released small artifact is the same dim-for-dim BoltzGen structure-design contract as
        ``boltzgen1_diverse`` (single/pair/atom widths, atom features, EDM schedule all identical) but a
        *shallower* realization, and it predates the design-conditioning + token-distance additions:

        - Miniformer trunk (compact triangular pair updates) instead of the full Pairformer/MSA stack.
        - ``12`` Miniformer pairformer blocks (vs ``64``) and ``3`` MSA blocks (vs ``4``).
        - Miniformer template pair updates.
        - ``8`` diffusion token-transformer blocks (vs ``24``).
        - No token-distance conditioning module.
        - No binding-specification / design-mask / secondary-structure input conditioning.

        Every one of these is an existing config knob, so the small model loads into the unchanged
        ``BoltzGenModel`` graph; no contract change is required.
        """

        config = cls()
        config.input_embedder.add_binding_specification = False
        config.input_embedder.add_ss_specification = False
        config.trunk.use_miniformer = True
        config.trunk.msa_miniformer_blocks = True
        config.trunk.num_pairformer_blocks = 12
        config.trunk.num_msa_blocks = 3
        config.template.miniformer_blocks = True
        config.diffusion.num_token_transformer_blocks = 8
        config.design.use_token_distance = False
        return config


class BoltzGenIFoldConfig(Config):
    r"""Model config for the distinct BoltzGen inverse-folding architecture."""

    single_size: int = 384
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    input_embedder: BoltzGenInputEmbedderConfig = BoltzGenInputEmbedderConfig()
    ifold: BoltzGenIFoldModuleConfig = BoltzGenIFoldModuleConfig()
