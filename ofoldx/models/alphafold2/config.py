r"""
Configuration objects for AlphaFold 2 monomer and multimer models.
"""

from __future__ import annotations

from chanfig import Config


class AlphaFold2InputEmbeddingsConfig(Config):
    r"""
    Configure AlphaFold 2 monomer input feature embeddings.

    Args:
        token_feature_size: Width of raw target-token features.
        msa_feature_size: Width of clustered MSA features.
        extra_msa_feature_size: Width of extra-MSA features.
        max_relative_feature: Maximum residue offset encoded in relative-position features.
    """

    token_feature_size: int = 22
    msa_feature_size: int = 49
    extra_msa_feature_size: int = 25
    max_relative_feature: int = 32


class AlphaFold2MultimerInputEmbeddingsConfig(Config):
    r"""
    Configure chain-relative AlphaFold-Multimer input embeddings.

    Args:
        token_feature_size: Width of raw target-token features.
        msa_feature_size: Width of clustered MSA features.
        extra_msa_feature_size: Width of extra-MSA features.
        max_relative_index: Maximum within-chain residue offset encoded in relative-position features.
        max_relative_chain: Maximum cross-chain symmetry-copy offset.
        use_chain_relative: Whether to encode chain-relative offsets and same-entity identity.
    """

    token_feature_size: int = 21
    msa_feature_size: int = 49
    extra_msa_feature_size: int = 25
    max_relative_index: int = 32
    max_relative_chain: int = 2
    use_chain_relative: bool = True


class AlphaFold2PreviousPositionsConfig(Config):
    r"""
    Configure distance bins for recycling previous atom positions.

    Args:
        min_distance: Lower recycled-distance bound, in angstroms.
        max_distance: Upper recycled-distance bound, in angstroms.
        num_distogram_bins: Number of recycled-distance bins.
    """

    min_distance: float = 3.25
    max_distance: float = 20.75
    num_distogram_bins: int = 15


class AlphaFold2RecyclingConfig(Config):
    r"""
    Configure AlphaFold 2 representation and position recycling.

    Args:
        recycle_features: Whether to recycle previous MSA and pair representations.
        recycle_positions: Whether to recycle previous atom positions as a distogram.
        previous_positions: Distance-bin configuration for recycled atom positions.
    """

    recycle_features: bool = True
    recycle_positions: bool = True
    previous_positions: AlphaFold2PreviousPositionsConfig = AlphaFold2PreviousPositionsConfig()


class AlphaFold2MsaRowAttentionWithPairBiasConfig(Config):
    r"""
    Configure MSA row attention with pair bias.

    Args:
        num_heads: Number of MSA row-attention heads.
        dropout: Dropout probability along the MSA row path.
    """

    num_heads: int = 8
    dropout: float = 0.15


class AlphaFold2TriangleAttentionConfig(Config):
    r"""
    Configure triangle attention over pair representations.

    Args:
        num_heads: Number of triangle-attention heads.
        head_dim: Per-head value width.
        dropout: Dropout probability along the pair path.
    """

    num_heads: int = 4
    head_dim: int = 32
    dropout: float = 0.25


class AlphaFold2TransitionConfig(Config):
    r"""
    Configure a transition feed-forward network.

    Args:
        expansion: Hidden-width expansion factor.
    """

    expansion: int = 4


class AlphaFold2OuterProductMeanConfig(Config):
    r"""
    Configure the Evoformer outer-product-mean update.

    Args:
        first: Whether to apply the update before the other Evoformer operations.
        hidden_size: Width of projected outer-product features.
    """

    first: bool = False
    hidden_size: int = 32


class AlphaFold2TriangleMultiplicationConfig(Config):
    r"""
    Configure a triangle multiplicative update.

    Args:
        hidden_size: Width of projected triangle features.
    """

    hidden_size: int = 128


class AlphaFold2TemplatePairStackConfig(Config):
    r"""
    Configure the AlphaFold 2 template pair stack.

    Args:
        num_blocks: Number of template pair blocks.
        pair_size: Width of template pair representations.
        triangle_attention_starting_node: Triangle attention with a 16-wide per-head value projection.
        triangle_multiplication_outgoing: Outgoing triangle multiplication with 64 hidden features.
        transition: Template pair transition with a twofold expansion.
    """

    num_blocks: int = 2
    pair_size: int = 64
    triangle_attention_starting_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig(
        {"head_dim": 16}
    )
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationConfig = AlphaFold2TriangleMultiplicationConfig(
        {"hidden_size": 64}
    )
    transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig({"expansion": 2})


class AlphaFold2TemplatePointwiseAttentionConfig(Config):
    r"""
    Configure pointwise attention across templates.

    Args:
        num_heads: Number of template pointwise-attention heads.
    """

    num_heads: int = 4


class AlphaFold2TemplateConfig(Config):
    r"""
    Configure monomer template embedding.

    Args:
        enabled: Whether to embed template evidence.
        embed_torsion_angles: Whether to append template torsion features to the MSA representation.
        min_distance: Lower template distogram bound, in angstroms.
        max_distance: Upper template distogram bound, in angstroms.
        num_distogram_bins: Number of template distance bins.
        vocab_size: Number of template residue types.
        use_template_unit_vector: Whether to embed backbone unit vectors.
        template_pair_stack: Template pair refinement stack.
        attention: Pointwise attention from templates into the query pair representation.
    """

    enabled: bool = True
    embed_torsion_angles: bool = True
    min_distance: float = 3.25
    max_distance: float = 50.75
    num_distogram_bins: int = 39
    vocab_size: int = 22
    use_template_unit_vector: bool = False
    template_pair_stack: AlphaFold2TemplatePairStackConfig = AlphaFold2TemplatePairStackConfig()
    attention: AlphaFold2TemplatePointwiseAttentionConfig = AlphaFold2TemplatePointwiseAttentionConfig()


class AlphaFold2MultimerTemplateConfig(Config):
    r"""
    Configure AlphaFold-Multimer template embedding.

    Args:
        enabled: Whether to embed template evidence.
        embed_torsion_angles: Whether to append template torsions; fixed to ``False`` for multimer.
        min_distance: Lower template distogram bound, in angstroms.
        max_distance: Upper template distogram bound, in angstroms.
        num_distogram_bins: Number of template distance bins.
        vocab_size: Number of template residue types.
        use_template_unit_vector: Whether to embed backbone unit vectors; fixed to ``True`` for multimer.
        template_pair_stack: Template pair refinement stack.
        attention: Pointwise attention from templates into the query pair representation.
    """

    enabled: bool = True
    embed_torsion_angles: bool = False
    min_distance: float = 3.25
    max_distance: float = 50.75
    num_distogram_bins: int = 39
    vocab_size: int = 22
    use_template_unit_vector: bool = True
    template_pair_stack: AlphaFold2TemplatePairStackConfig = AlphaFold2TemplatePairStackConfig()
    attention: AlphaFold2TemplatePointwiseAttentionConfig = AlphaFold2TemplatePointwiseAttentionConfig()


class AlphaFold2EvoformerBlockConfig(Config):
    r"""
    Configure the operations in one AlphaFold 2 Evoformer block.

    Args:
        msa_row_attention_with_pair_bias: MSA attention heads and row dropout.
        msa_transition: MSA transition expansion.
        outer_product_mean: Outer-product-mean ordering and hidden width.
        triangle_multiplication_outgoing: Hidden width shared by triangle multiplication orientations.
        triangle_attention_starting_node: Heads, value width, and dropout shared by triangle orientations.
        pair_transition: Pair transition expansion.
    """

    msa_row_attention_with_pair_bias: AlphaFold2MsaRowAttentionWithPairBiasConfig = (
        AlphaFold2MsaRowAttentionWithPairBiasConfig()
    )
    msa_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig()
    outer_product_mean: AlphaFold2OuterProductMeanConfig = AlphaFold2OuterProductMeanConfig()
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationConfig = AlphaFold2TriangleMultiplicationConfig()
    triangle_attention_starting_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig()
    pair_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig()


class AlphaFold2EvoformerConfig(Config):
    r"""
    Configure the AlphaFold 2 Evoformer stack.

    Args:
        num_blocks: Number of main Evoformer blocks.
        num_extra_msa_blocks: Number of extra-MSA Evoformer blocks.
        msa_size: Width of clustered MSA representations.
        extra_msa_size: Width of extra-MSA representations.
        pair_size: Width of pair representations.
        single_size: Width of emitted single representations.
        block: Shared operation configuration for each Evoformer block.
    """

    num_blocks: int = 48
    num_extra_msa_blocks: int = 4
    msa_size: int = 256
    extra_msa_size: int = 64
    pair_size: int = 128
    single_size: int = 384
    block: AlphaFold2EvoformerBlockConfig = AlphaFold2EvoformerBlockConfig()


class AlphaFold2SidechainConfig(Config):
    r"""
    Configure the sidechain network in the AlphaFold 2 structure module.

    Args:
        hidden_size: Width of sidechain hidden states.
        num_blocks: Number of sidechain residual blocks.
    """

    hidden_size: int = 128
    num_blocks: int = 2


class AlphaFold2StructureModuleConfig(Config):
    r"""
    Configure the AlphaFold 2 invariant-point-attention structure module.

    Args:
        dropout: Dropout probability in structure-module blocks.
        hidden_size: Width of structure-module single representations.
        num_heads: Number of invariant-point-attention heads.
        num_blocks: Number of iterative structure-module blocks.
        num_transition_projections: Number of projections in each structure transition.
        num_point_qk: Number of query/key points per attention head.
        num_point_v: Number of value points per attention head.
        num_scalar_qk: Number of scalar query/key channels per attention head.
        num_scalar_v: Number of scalar value channels per attention head.
        position_scale: Scale converting backbone translations to model coordinates.
        sidechain: Sidechain prediction network configuration.
    """

    dropout: float = 0.1
    hidden_size: int = 384
    num_heads: int = 12
    num_blocks: int = 8
    num_transition_projections: int = 3
    num_point_qk: int = 4
    num_point_v: int = 8
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    position_scale: float = 10.0
    sidechain: AlphaFold2SidechainConfig = AlphaFold2SidechainConfig()


# AlphaFold-Multimer configuration.


class AlphaFold2MultimerStructureModuleConfig(AlphaFold2StructureModuleConfig):
    r"""
    Configure the AlphaFold-Multimer structure-module override.

    Args:
        position_scale: Backbone translation scale, overridden from ``10`` to ``20`` for multimer.
    """

    position_scale: float = 20.0


class AlphaFold2DistogramHeadConfig(Config):
    r"""
    Configure the AlphaFold 2 distogram head.

    Args:
        first_break: Center of the first distance bin, in angstroms.
        last_break: Center of the last bounded distance bin, in angstroms.
        num_bins: Number of distogram output bins.
    """

    first_break: float = 2.3125
    last_break: float = 21.6875
    num_bins: int = 64


class AlphaFold2PredictedAlignedErrorHeadConfig(Config):
    r"""
    Configure the monomer predicted-aligned-error and pTM head.

    Args:
        enabled: Whether to build and run the predicted-aligned-error head.
        max_error_bin: Upper predicted-aligned-error bin center, in angstroms.
        num_bins: Number of predicted-aligned-error bins.
    """

    enabled: bool = True
    max_error_bin: float = 31.0
    num_bins: int = 64


class AlphaFold2MultimerPredictedAlignedErrorHeadConfig(Config):
    r"""
    Configure AlphaFold-Multimer PAE, pTM, ipTM, and ranking outputs.

    Args:
        enabled: Whether to build and run the predicted-aligned-error head.
        max_error_bin: Upper predicted-aligned-error bin center, in angstroms.
        num_bins: Number of predicted-aligned-error bins.
        interface: Whether to compute interface pTM.
        ptm_weight: pTM contribution to the combined ranking score.
        iptm_weight: ipTM contribution to the combined ranking score.
    """

    enabled: bool = True
    max_error_bin: float = 31.0
    num_bins: int = 64
    interface: bool = True
    ptm_weight: float = 0.2
    iptm_weight: float = 0.8


class AlphaFold2PredictedLddtHeadConfig(Config):
    r"""
    Configure the AlphaFold 2 predicted-lDDT head.

    Args:
        num_bins: Number of predicted-lDDT bins.
        hidden_size: Width of head hidden states.
    """

    num_bins: int = 50
    hidden_size: int = 128


class AlphaFold2ExperimentallyResolvedHeadConfig(Config):
    r"""
    Configure the experimentally-resolved atom head.

    Args:
        num_atom_types: Number of atom37 output types.
    """

    num_atom_types: int = 37


class AlphaFold2MaskedMsaHeadConfig(Config):
    r"""
    Configure the monomer masked-MSA reconstruction head.

    Args:
        num_output: Number of reconstructed MSA token classes.
    """

    num_output: int = 23


class AlphaFold2MultimerMaskedMsaHeadConfig(Config):
    r"""
    Configure the 22-class AlphaFold-Multimer masked-MSA head.

    Args:
        num_output: Number of reconstructed MSA token classes.
    """

    num_output: int = 22


class AlphaFold2MultimerChainCenterOfMassHeadConfig(Config):
    r"""
    Configure the parameterized but inference-inert multimer chain-center head.

    Args:
        enabled: Whether to instantiate the chain center-of-mass head.
    """

    enabled: bool = True


class AlphaFold2HeadsConfig(Config):
    r"""
    Configure AlphaFold 2 monomer prediction heads.

    Args:
        distogram: Inter-residue distogram head.
        masked_msa: Masked-MSA reconstruction head.
        predicted_aligned_error: Predicted-aligned-error and pTM head.
        predicted_lddt: Per-residue predicted-lDDT head.
        experimentally_resolved: Atom37 experimentally-resolved head.
    """

    distogram: AlphaFold2DistogramHeadConfig = AlphaFold2DistogramHeadConfig()
    masked_msa: AlphaFold2MaskedMsaHeadConfig = AlphaFold2MaskedMsaHeadConfig()
    predicted_aligned_error: AlphaFold2PredictedAlignedErrorHeadConfig = AlphaFold2PredictedAlignedErrorHeadConfig()
    predicted_lddt: AlphaFold2PredictedLddtHeadConfig = AlphaFold2PredictedLddtHeadConfig()
    experimentally_resolved: AlphaFold2ExperimentallyResolvedHeadConfig = AlphaFold2ExperimentallyResolvedHeadConfig()


class AlphaFold2MultimerHeadsConfig(Config):
    r"""
    Configure AlphaFold-Multimer prediction heads.

    Args:
        distogram: Inter-residue distogram head.
        masked_msa: 22-class masked-MSA reconstruction head.
        predicted_aligned_error: Predicted-aligned-error, pTM, ipTM, and ranking head.
        predicted_lddt: Per-residue predicted-lDDT head.
        experimentally_resolved: Atom37 experimentally-resolved head.
        chain_center_of_mass: Parameterized but inference-inert chain-center head.
    """

    distogram: AlphaFold2DistogramHeadConfig = AlphaFold2DistogramHeadConfig()
    masked_msa: AlphaFold2MultimerMaskedMsaHeadConfig = AlphaFold2MultimerMaskedMsaHeadConfig()
    predicted_aligned_error: AlphaFold2MultimerPredictedAlignedErrorHeadConfig = (
        AlphaFold2MultimerPredictedAlignedErrorHeadConfig()
    )
    predicted_lddt: AlphaFold2PredictedLddtHeadConfig = AlphaFold2PredictedLddtHeadConfig()
    experimentally_resolved: AlphaFold2ExperimentallyResolvedHeadConfig = AlphaFold2ExperimentallyResolvedHeadConfig()
    chain_center_of_mass: AlphaFold2MultimerChainCenterOfMassHeadConfig = (
        AlphaFold2MultimerChainCenterOfMassHeadConfig()
    )


class AlphaFold2MsaConfig(Config):
    r"""
    Configure stochastic monomer MSA sampling, masking, clustering, and cropping.

    Args:
        max_msa_clusters: Maximum number of clustered MSA rows.
        max_extra_msa: Maximum number of extra-MSA rows.
        msa_cluster_features: Whether to append cluster profiles and deletion means.
        resample_msa_in_recycling: Whether to redraw MSA features for each recycling pass.
        gap_agreement_weight: Weight assigned to gap agreement during nearest-neighbor clustering.
        masked_msa_enabled: Whether to apply masked-MSA corruption.
        masked_msa_replace_fraction: Fraction of MSA tokens selected for corruption.
        masked_msa_uniform_prob: Probability of uniformly random replacement for a selected token.
        masked_msa_profile_prob: Probability of profile-sampled replacement for a selected token.
        masked_msa_same_prob: Probability of preserving a selected token unchanged.
    """

    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    msa_cluster_features: bool = True
    resample_msa_in_recycling: bool = True
    gap_agreement_weight: float = 0.0
    masked_msa_enabled: bool = True
    masked_msa_replace_fraction: float = 0.15
    masked_msa_uniform_prob: float = 0.10
    masked_msa_profile_prob: float = 0.10
    masked_msa_same_prob: float = 0.10


class AlphaFold2MultimerMsaConfig(Config):
    r"""
    Configure stochastic multimer MSA sampling, masking, clustering, and cropping.

    Args:
        max_msa_clusters: Maximum number of clustered MSA rows.
        max_extra_msa: Maximum number of extra-MSA rows cropped during sampling.
        msa_cluster_features: Whether to append cluster profiles and deletion means.
        resample_msa_in_recycling: Whether to redraw MSA features for each recycling pass.
        gap_agreement_weight: Weight assigned to gap agreement during nearest-neighbor clustering.
        masked_msa_enabled: Whether to apply 22-class masked-MSA corruption before clustering.
        masked_msa_replace_fraction: Fraction of MSA tokens selected for corruption.
        masked_msa_uniform_prob: Probability of uniformly random replacement for a selected token.
        masked_msa_profile_prob: Probability of profile-sampled replacement for a selected token.
        masked_msa_same_prob: Probability of preserving a selected token unchanged.
    """

    max_msa_clusters: int = 508
    max_extra_msa: int = 2048
    msa_cluster_features: bool = True
    resample_msa_in_recycling: bool = True
    gap_agreement_weight: float = 0.0
    masked_msa_enabled: bool = True
    masked_msa_replace_fraction: float = 0.15
    masked_msa_uniform_prob: float = 0.10
    masked_msa_profile_prob: float = 0.10
    masked_msa_same_prob: float = 0.10


class AlphaFold2Config(Config):
    r"""
    Configure the complete [`AlphaFold2Model`][ofoldx.models.alphafold2.AlphaFold2Model] graph.

    Args:
        input_embeddings: Monomer token, MSA, extra-MSA, and relative-position embeddings.
        recycling: Previous representation and position recycling configuration.
        template: Monomer template path with torsion embedding enabled and unit vectors disabled.
        evoformer: Main and extra-MSA Evoformer stack configuration.
        structure_module: Invariant-point-attention structure and sidechain module.
        heads: Monomer distogram, masked-MSA, PAE/pTM, pLDDT, and resolved heads.
        msa: Model-owned stochastic MSA realization configuration.
        recycling_steps: Number of complete recycling passes.
        num_ensembles: Number of independently embedded and trunk-processed representations to average.
    """

    input_embeddings: AlphaFold2InputEmbeddingsConfig = AlphaFold2InputEmbeddingsConfig()
    recycling: AlphaFold2RecyclingConfig = AlphaFold2RecyclingConfig()
    template: AlphaFold2TemplateConfig = AlphaFold2TemplateConfig()
    evoformer: AlphaFold2EvoformerConfig = AlphaFold2EvoformerConfig()
    structure_module: AlphaFold2StructureModuleConfig = AlphaFold2StructureModuleConfig()
    heads: AlphaFold2HeadsConfig = AlphaFold2HeadsConfig()
    msa: AlphaFold2MsaConfig = AlphaFold2MsaConfig()

    recycling_steps: int = 4
    num_ensembles: int = 1


class AlphaFold2MultimerConfig(Config):
    r"""
    Configure the complete AlphaFold-Multimer graph.

    Args:
        input_embeddings: Multimer token, MSA, extra-MSA, and chain-relative embeddings.
        recycling: Previous representation and position recycling configuration.
        evoformer: Evoformer stack with outer-product mean applied first.
        template: Multimer template path with torsions disabled and unit vectors enabled.
        structure_module: Structure module with the multimer translation scale.
        heads: Multimer heads including ipTM ranking and chain-center parameters.
        msa: Model-owned stochastic multimer MSA realization configuration.
        recycling_steps: Number of complete recycling passes.
        num_ensembles: Number of independently embedded and trunk-processed representations to average.
    """

    input_embeddings: AlphaFold2MultimerInputEmbeddingsConfig = AlphaFold2MultimerInputEmbeddingsConfig()
    recycling: AlphaFold2RecyclingConfig = AlphaFold2RecyclingConfig()
    # AF-Multimer applies outer-product mean before the other Evoformer updates;
    # the monomer configuration applies it last.
    evoformer: AlphaFold2EvoformerConfig = AlphaFold2EvoformerConfig({"block": {"outer_product_mean": {"first": True}}})
    template: AlphaFold2MultimerTemplateConfig = AlphaFold2MultimerTemplateConfig()
    structure_module: AlphaFold2MultimerStructureModuleConfig = AlphaFold2MultimerStructureModuleConfig()
    heads: AlphaFold2MultimerHeadsConfig = AlphaFold2MultimerHeadsConfig()
    msa: AlphaFold2MultimerMsaConfig = AlphaFold2MultimerMsaConfig()

    recycling_steps: int = 21
    num_ensembles: int = 1


class AlphaFold2ProcessorConfig(Config):
    r"""
    Configure monomer template preprocessing.

    Args:
        max_templates: Maximum number of templates included in model inputs.
    """

    max_templates: int = 4


class AlphaFold2MultimerProcessorConfig(Config):
    r"""
    Configure multimer MSA and template preprocessing.

    Args:
        max_msa_clusters: Maximum clustered MSA depth.
        max_extra_msa: Maximum extra-MSA depth.
        max_templates: Maximum number of templates included in model inputs.
    """

    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    max_templates: int = 4
