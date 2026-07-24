r"""
Boltz-2 architecture and inference configuration.
"""

from __future__ import annotations

from chanfig import Config


class Boltz2InputEmbeddingsConfig(Config):
    r"""
    Configure the independent Boltz-2 atom input encoder and token conditioning.

    Args:
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size.
        msa_profile_size: Width of the MSA profile and mean-deletion feature.
        num_method_types: Number of experimental-method ids for method conditioning.
        num_molecule_types: Number of molecule types for molecule-type conditioning.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.
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


class Boltz2TrunkConfig(Config):
    r"""
    Configure the Boltz-2 Pairformer and MSA representation trunk.

    Args:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk stack.
        num_pair_attention_heads: Head count for the trunk pair-biased attention.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_feature_size: Width of the per-column MSA input features
            (one_hot(33) + has_deletion + deletion_value + is_paired).
        msa_head_dim: Per-head dimension of the MSA attention.
        num_msa_attention_heads: Pair-weighted-averaging head count.
        outer_product_hidden_size: Hidden width of the outer-product-mean block.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Triangle-attention head count.
        use_paired_feature: Whether the paired-MSA feature is consumed.
        pair_dropout: Pair-track dropout for the Pairformer.
        msa_dropout: Dropout applied inside the MSA trunk.
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
    pair_dropout: float = 0.25
    msa_dropout: float = 0.15


class Boltz2TemplateConfig(Config):
    r"""
    Configure Boltz-2 on-the-fly template distance and unit-vector embedding.

    Args:
        hidden_size: Width of the template pair hidden channel.
        num_blocks: Number of pair-only triangle blocks in the template stack.
        num_distogram_bins: Number of on-the-fly distogram bins.
        vocab_size: Residue-type vocabulary size feeding the template projection.
        expansion: Transition expansion factor.
        triangle_attention_hidden_size: Per-head width of the template triangle attention.
        num_triangle_attention_heads: Template triangle-attention head count.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.
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


class Boltz2DenoiserConfig(Config):
    r"""
    Configure the Boltz-2 learned coordinate denoiser.

    Args:
        num_token_encoder_blocks: Number of conditioned token-transformer blocks in the score model.
        num_token_encoder_attention_heads: Head count for the token transformer.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks in the score model.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks in the score model.
        num_atom_attention_heads: Head count shared by the atom encoder and decoder.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_conditioning_transitions: Number of transitions in each diffusion conditioner.
        sigma_data: EDM data standard deviation used for preconditioning.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step.
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


class Boltz2SamplerConfig(Config):
    r"""
    Configure Boltz-2 Euler sampling with in-loop weighted rigid alignment.

    Args:
        num_sampling_steps: Number of EDM sampling steps.
        rho: Exponent of the Karras noise schedule.
        max_noise_level: Highest sigma at the start of the schedule.
        min_noise_level: Lowest non-zero sigma before the trailing zero.
        step_scale: Scale on the Euler update step.
        noise_scale: Multiplier on the injected churn noise.
        churn_rate: Stochastic churn added to each step above the churn threshold.
    """

    num_sampling_steps: int = 200
    rho: float = 7.0
    max_noise_level: float = 160.0
    min_noise_level: float = 1e-4
    step_scale: float = 1.5
    noise_scale: float = 1.003
    churn_rate: float = 0.8


class Boltz2ConfidenceConfig(Config):
    r"""
    Configure token-level Boltz-2 confidence with separate intra/inter PAE and PDE.

    Args:
        num_pairformer_blocks: Number of Pairformer blocks in the confidence head.
        num_pair_attention_heads: Head count for the confidence pair-biased attention.
        num_distogram_bins: Number of distogram bins used to bin representative-atom distances.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
    """

    num_pairformer_blocks: int = 8
    num_pair_attention_heads: int = 16
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64


class Boltz2AffinityHeadConfig(Config):
    r"""
    Configure one Boltz-2 masked-pair affinity head and its pair-only Pairformer.

    Args:
        num_pairformer_blocks: Number of pair-only Pairformer blocks in the affinity head.
        num_distogram_bins: Number of distogram bins used to condition the pair features.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_conditioning_transitions: Number of transitions in the pairwise conditioner.
        expansion: Transition expansion factor in the pairwise conditioner.
        triangle_attention_hidden_size: Per-head width of the affinity triangle attention.
        num_triangle_attention_heads: Affinity triangle-attention head count.
        dropout: Dropout applied in the affinity Pairformer stack.
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
    Configure the Boltz-2 folding and confidence architecture.

    Args:
        single_size: Width of the token (single) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single channel in the input embedder.
        atom_pair_hidden_size: Width of the atom pair channel in the input embedder.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_distogram_bins: Number of distogram / B-factor output bins.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        relative_position_size: Width of the relative-position feature vector.
        num_bond_types: Bond-type vocabulary size.
        recycling_steps: Number of trunk recycling passes.
        use_templates: Whether template embedding is enabled.
        predict_bfactor: Whether the B-factor head is instantiated.
        conditioning_cutoff_min: Lower distance cutoff for contact conditioning, in angstroms.
        conditioning_cutoff_max: Upper distance cutoff for contact conditioning, in angstroms.
        input_embeddings: Input-embeddings sub-config.
        trunk: Representation-trunk sub-config.
        template: Template-embedder sub-config.
        denoiser: Learned coordinate-denoiser sub-config.
        sampler: EDM-sampler sub-config.
        confidence: Pairformer confidence-module sub-config.
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

    use_templates: bool = True
    predict_bfactor: bool = True

    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    input_embeddings: Boltz2InputEmbeddingsConfig = Boltz2InputEmbeddingsConfig()
    trunk: Boltz2TrunkConfig = Boltz2TrunkConfig()
    template: Boltz2TemplateConfig = Boltz2TemplateConfig()
    denoiser: Boltz2DenoiserConfig = Boltz2DenoiserConfig()
    sampler: Boltz2SamplerConfig = Boltz2SamplerConfig()
    confidence: Boltz2ConfidenceConfig = Boltz2ConfidenceConfig()


class Boltz2AffinityConfig(Boltz2Config):
    r"""
    Configure affinity prediction by overriding the inherited Boltz-2 graph and adding two fixed heads.

    Args:
        use_templates: Whether to run the inherited template embedder; fixed to ``False`` for affinity.
        predict_bfactor: Whether to build the inherited B-factor head; fixed to ``False`` for affinity.
        recycling_steps: Number of trunk recycling passes; overridden to six for affinity.
        affinity_heads: Two affinity-head configs in ensemble order.
    """

    use_templates: bool = False
    predict_bfactor: bool = False
    recycling_steps: int = 6

    affinity_heads: tuple[Boltz2AffinityHeadConfig, Boltz2AffinityHeadConfig] = (
        Boltz2AffinityHeadConfig({"num_pairformer_blocks": 8}),
        Boltz2AffinityHeadConfig({"num_pairformer_blocks": 4}),
    )
