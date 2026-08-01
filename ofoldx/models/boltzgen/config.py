# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Configuration values for the BoltzGen structure, confidence, and inverse-folding architectures.
"""

from __future__ import annotations

from chanfig import Config

from ...data.tokenization import TOKENS


class BoltzGenInputEmbeddingsConfig(Config):
    r"""
    Configure BoltzGen atom and token input embeddings.

    Attributes:
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        vocab_size: Number of token types.
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
    vocab_size: int = len(TOKENS)
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

    Attributes:
        num_trunk_blocks: Number of pair trunk blocks.
        num_pair_attention_heads: Number of pair-biased attention heads.
        num_msa_blocks: Number of MSA blocks.
        msa_size: Width of MSA representations.
        msa_attention_head_size: Per-head width of MSA attention.
        num_msa_attention_heads: Number of MSA attention heads.
        outer_product_hidden_size: Hidden width of the outer-product-mean projection.
        msa_triangle_attention_head_size: Per-head width of MSA pair-update
            triangle attention.
        num_msa_triangle_attention_heads: Number of MSA pair-update triangle
            attention heads.
        triangle_attention_head_size: Per-head width of main-trunk triangle
            attention.
        num_triangle_attention_heads: Number of main-trunk triangle-attention heads.
        use_paired_feature: Whether to consume paired-MSA features.
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
    msa_attention_head_size: int = 32
    num_msa_attention_heads: int = 8
    outer_product_hidden_size: int = 32
    msa_triangle_attention_head_size: int = 32
    num_msa_triangle_attention_heads: int = 4
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    use_paired_feature: bool = True
    pair_dropout: float = 0.25
    msa_pair_dropout: float = 0.25
    msa_dropout: float = 0.15
    post_layer_norm: bool = False
    use_single_to_pair: bool = False


class BoltzGenTemplateConfig(Config):
    r"""
    Configure geometry shared by template and token-distance conditioning.

    Attributes:
        num_distogram_bins: Number of distance bins.
        triangle_attention_head_size: Per-head width of triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        min_distance: Lower distance bound, in angstroms.
        max_distance: Upper distance bound, in angstroms.
    """

    num_distogram_bins: int = 38
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    min_distance: float = 3.25
    max_distance: float = 50.75


class BoltzGenDenoiserConfig(Config):
    r"""
    Configure the BoltzGen EDM score network.

    Attributes:
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
        num_coordinate_gaussians: Width of the frozen random 3D coordinate encoding.
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
    num_coordinate_gaussians: int = 0
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True


class BoltzGenSamplerConfig(Config):
    r"""
    Configure the BoltzGen reverse-diffusion schedule.

    Attributes:
        num_sampling_steps: Number of reverse-diffusion steps.
        sampling_schedule: Noise-schedule shape, either ``"karras"`` for the plain rho-spaced ramp
            or ``"dilated"`` for that ramp with the configured interval dilated.
        rho: Exponent shaping noise-level spacing.
        max_noise_level: Highest noise level in the schedule.
        min_noise_level: Lowest non-zero noise level in the schedule.
        step_scale: Multiplier on each denoising step.
        noise_scale: Multiplier on churn-injected noise.
        churn_rate: Stochastic churn added during sampling.
        min_churn_noise_level: Noise threshold for stochastic churn.
        step_scale_function: Constant or beta-CDF step-scale schedule.
        noise_scale_function: Constant or beta-CDF noise-scale schedule.
        min_step_scale: Minimum beta step scale.
        max_step_scale: Maximum beta step scale.
        step_scale_alpha: First step-scale beta parameter.
        step_scale_beta: Second step-scale beta parameter.
        min_noise_scale: Minimum beta noise scale.
        max_noise_scale: Maximum beta noise scale.
        noise_scale_alpha: First physical noise-scale beta parameter.
        noise_scale_beta: Second physical noise-scale beta parameter.
        step_scale_random: Step-scale choices for training-time sampling.
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
    min_churn_noise_level: float = 1.0
    step_scale_function: str = "constant"
    noise_scale_function: str = "constant"
    min_step_scale: float = 1.0
    max_step_scale: float = 1.0
    step_scale_alpha: float = 1.0
    step_scale_beta: float = 1.0
    min_noise_scale: float = 1.0
    max_noise_scale: float = 1.0
    noise_scale_alpha: float = 1.0
    noise_scale_beta: float = 1.0
    step_scale_random: tuple[float, ...] | None = None
    time_dilation: float = 2.667
    time_dilation_start: float = 0.6
    time_dilation_end: float = 0.8


class BoltzGenStructureDesignConfig(Config):
    r"""
    Configure structure-design template and token-distance modules.

    Attributes:
        template_size: Width of structure-design template features.
        num_template_blocks: Number of structure-design template blocks.
        template_dropout: Dropout probability in template pair updates.
        template_post_layer_norm: Whether Miniformer template blocks apply post-layer normalization.
        token_distance_size: Width of token-distance features.
        num_token_distance_blocks: Number of token-distance blocks.
        token_distance_dropout: Dropout probability in token-distance pair updates.
        token_distance_post_layer_norm: Whether Miniformer token-distance blocks apply post-layer normalization.
        num_distance_gaussians: Number of Gaussian distance features.
        use_token_distance: Whether to build token-distance conditioning.
        use_token_distance_features: Whether to add per-backbone-atom distance features to the distogram.
    """

    template_size: int = 64
    num_template_blocks: int = 2
    template_dropout: float = 0.25
    template_post_layer_norm: bool = False
    token_distance_size: int = 64
    num_token_distance_blocks: int = 2
    token_distance_dropout: float = 0.25
    token_distance_post_layer_norm: bool = False
    num_distance_gaussians: int = 32
    use_token_distance: bool = True
    use_token_distance_features: bool = True


class BoltzGenConfidenceHeadConfig(Config):
    r"""
    Configure the distinct BoltzGen confidence equation.

    Attributes:
        num_pairformer_blocks: Number of full Pairformer blocks.
        num_pair_attention_heads: Number of pair-biased single-attention heads.
        triangle_attention_head_size: Per-head width of triangle attention.
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
    triangle_attention_head_size: int = 32
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

    Attributes:
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

    Attributes:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of atom single features in the input embedder.
        atom_pair_hidden_size: Width of atom pair features in the input embedder.
        atom_feature_size: Width of raw atom features.
        num_distogram_bins: Number of distogram output bins.
        fourier_embedding_size: Width of diffusion-noise Fourier features.
        relative_position_size: Width of relative-position features.
        num_bond_types: Number of token bond types.
        num_recycles: Number of complete trunk passes.
        min_conditioning_cutoff: Lower contact-conditioning cutoff, in angstroms.
        max_conditioning_cutoff: Upper contact-conditioning cutoff, in angstroms.
        input_embeddings: Input embedder with binding and secondary-structure conditioning enabled.
        trunk: Pairformer and MSA trunk configuration.
        template: Shared template and token-distance geometry configuration.
        denoiser: EDM score-network configuration.
        sampler: Reverse-diffusion schedule configuration.
        design: Structure-design modules, including token-distance conditioning.

    Examples:
        >>> config = BoltzGenConfig(num_recycles=2)
        >>> config.num_recycles
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
    num_recycles: int = 4
    min_conditioning_cutoff: float = 4.0
    max_conditioning_cutoff: float = 20.0

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

    @staticmethod
    def structure_trained_small() -> BoltzGenMiniformerConfig:
        r"""
        Return the reduced BoltzGen structure-design configuration.

        This variant preserves the single, pair, atom, and EDM dimensions while using a shallower graph:

        - Miniformer trunk (compact triangular pair updates) instead of the full Pairformer/MSA stack.
        - ``12`` Miniformer trunk blocks (vs ``64``) and ``3`` MSA blocks (vs ``4``).
        - Miniformer template pair updates.
        - ``8`` diffusion token-encoder blocks (vs ``24``).
        - No token-distance conditioning module.
        - No binding-specification / design-mask / secondary-structure input conditioning.

        The returned configuration has its own concrete model type because the
        Miniformer and Pairformer variants have different parameter graphs.

        Returns:
            Reduced structure-design configuration.
        """

        return BoltzGenMiniformerConfig()


class BoltzGenResidueTypeConfig(BoltzGenConfig):
    r"""
    Configure the Pairformer structure model with residue-type prediction.

    Examples:
        >>> config = BoltzGenResidueTypeConfig(num_recycles=2)
        >>> config.num_recycles
        2
    """


class BoltzGenMiniformerConfig(BoltzGenConfig):
    r"""
    Configure the reduced structure-trained Miniformer architecture.

    This is a distinct parameter graph, not a depth preset for
    [`BoltzGenConfig`][ofoldx.models.boltzgen.config.BoltzGenConfig].

    Examples:
        >>> config = BoltzGenMiniformerConfig(num_recycles=2)
        >>> config.trunk.num_trunk_blocks
        12
    """

    input_embeddings: BoltzGenInputEmbeddingsConfig = BoltzGenInputEmbeddingsConfig(
        {
            "add_binding_specification": False,
            "add_design_mask": False,
            "add_ss_specification": False,
        }
    )
    trunk: BoltzGenTrunkConfig = BoltzGenTrunkConfig(
        {
            "num_trunk_blocks": 12,
            "num_msa_blocks": 3,
        }
    )
    denoiser: BoltzGenDenoiserConfig = BoltzGenDenoiserConfig(
        {
            "num_token_encoder_blocks": 8,
        }
    )
    design: BoltzGenStructureDesignConfig = BoltzGenStructureDesignConfig(
        {
            "use_token_distance": False,
        }
    )


class BoltzGenMiniformerResidueTypeConfig(BoltzGenMiniformerConfig, BoltzGenResidueTypeConfig):
    r"""
    Configure the reduced Miniformer architecture with residue-type prediction.

    Examples:
        >>> config = BoltzGenMiniformerResidueTypeConfig(num_recycles=2)
        >>> config.denoiser.num_token_encoder_blocks
        8
    """


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

    Attributes:
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


class BoltzGenProcessorConfig(Config):
    r"""Configure BoltzGen structure and inverse-folding input processing.

    Attributes:
        seed: Seed used to embed missing reference conformers.
    """

    seed: int = 42
