r"""
AlphaFold 2 configuration.

The public surface contains only values that construct the released computation.
Runtime placement, operator dispatch, checkpointing, loss weights, and source
checkpoint metadata are deliberately absent.
"""

from __future__ import annotations

from chanfig import Config


class AlphaFold2PrevPosConfig(Config):
    r"""
    Previous-position distogram recycling bins.
    """

    min_distance: float = 3.25
    max_distance: float = 20.75
    num_distogram_bins: int = 15


class AlphaFold2MSARowAttentionWithPairBiasConfig(Config):
    num_heads: int = 8
    dropout: float = 0.15


class AlphaFold2TriangleAttentionConfig(Config):
    num_heads: int = 4
    value_dim: int = 32
    dropout: float = 0.25


class AlphaFold2TransitionConfig(Config):
    r"""
    Transition feed-forward expansion.
    """

    expansion: int = 4


class AlphaFold2OuterProductMeanConfig(Config):
    r"""
    Outer-product-mean projection config.
    """

    first: bool = False
    num_outer_channel: int = 32


class AlphaFold2TriangleMultiplicationConfig(Config):
    r"""
    Triangle multiplicative update config.
    """

    num_intermediate_channel: int = 128


class AlphaFold2EvoformerIterationConfig(Config):
    r"""
    One official Evoformer iteration's module-level knobs.

    These fields all alter a constructed module or its released dropout behavior.
    The released graph uses one shared width/head/dropout specification for the
    two triangle orientations and one MSA head count for row and column attention.
    """

    msa_row_attention_with_pair_bias: AlphaFold2MSARowAttentionWithPairBiasConfig = (
        AlphaFold2MSARowAttentionWithPairBiasConfig()
    )
    msa_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig()
    outer_product_mean: AlphaFold2OuterProductMeanConfig = AlphaFold2OuterProductMeanConfig()
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationConfig = AlphaFold2TriangleMultiplicationConfig()
    triangle_attention_starting_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig()
    pair_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig()


class AlphaFold2TemplatePairStackConfig(Config):
    r"""
    Template pair stack config.
    """

    num_block: int = 2
    num_channels: int = 64
    triangle_attention_starting_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig(
        {"value_dim": 16}
    )
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationConfig = AlphaFold2TriangleMultiplicationConfig(
        {"num_intermediate_channel": 64}
    )
    pair_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig({"expansion": 2})


class AlphaFold2TemplatePointwiseAttentionConfig(Config):
    r"""
    Template pointwise attention config.
    """

    num_heads: int = 4


class AlphaFold2TemplateConfig(Config):
    r"""
    Template embedding config with official names.
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


class AlphaFold2InputEmbeddingConfig(Config):
    r"""
    Input feature embedding config.
    """

    target_feature_size: int = 22
    msa_feature_size: int = 49
    extra_msa_feature_size: int = 25
    max_relative_feature: int = 32


class AlphaFold2RecyclingConfig(Config):
    r"""
    Recycling feature config.
    """

    recycle_features: bool = True
    recycle_pos: bool = True
    prev_pos: AlphaFold2PrevPosConfig = AlphaFold2PrevPosConfig()


class AlphaFold2EvoformerConfig(Config):
    r"""
    Evoformer stack config.
    """

    num_blocks: int = 48
    extra_msa_num_blocks: int = 4
    msa_size: int = 256
    extra_msa_size: int = 64
    pair_size: int = 128
    single_size: int = 384
    iteration: AlphaFold2EvoformerIterationConfig = AlphaFold2EvoformerIterationConfig()


class AlphaFold2DistogramHeadConfig(Config):
    r"""
    Distogram head width and the published distance-bin boundaries in angstroms.
    """

    first_break: float = 2.3125
    last_break: float = 21.6875
    num_bins: int = 64


class AlphaFold2PredictedAlignedErrorHeadConfig(Config):
    r"""
    Predicted aligned error / pTM head config.
    """

    enabled: bool = True
    max_error_bin: float = 31.0
    num_bins: int = 64


class AlphaFold2PredictedLDDTHeadConfig(Config):
    r"""
    Predicted lDDT confidence head config.
    """

    num_bins: int = 50
    num_channels: int = 128


class AlphaFold2ExperimentallyResolvedHeadConfig(Config):
    r"""
    Experimentally-resolved atom37 head config.
    """

    num_atom_types: int = 37


class AlphaFold2MaskedMSAHeadConfig(Config):
    r"""
    Masked-MSA reconstruction head config.
    """

    num_output: int = 23


class AlphaFold2SidechainConfig(Config):
    r"""
    Sidechain architecture used by the official structure module.
    """

    num_channel: int = 128
    num_residual_block: int = 2


class AlphaFold2StructureModuleHeadConfig(Config):
    r"""
    Official StructureModule config surface.

    The model consumes every field to construct the native PyTorch structure
    module.
    """

    dropout: float = 0.1
    num_channel: int = 384
    num_heads: int = 12
    num_layer: int = 8
    num_layer_in_transition: int = 3
    num_point_qk: int = 4
    num_point_v: int = 8
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    position_scale: float = 10.0
    sidechain: AlphaFold2SidechainConfig = AlphaFold2SidechainConfig()


class AlphaFold2HeadsConfig(Config):
    r"""
    Official ``model.heads`` config group.
    """

    distogram: AlphaFold2DistogramHeadConfig = AlphaFold2DistogramHeadConfig()
    masked_msa: AlphaFold2MaskedMSAHeadConfig = AlphaFold2MaskedMSAHeadConfig()
    predicted_aligned_error: AlphaFold2PredictedAlignedErrorHeadConfig = AlphaFold2PredictedAlignedErrorHeadConfig()
    predicted_lddt: AlphaFold2PredictedLDDTHeadConfig = AlphaFold2PredictedLDDTHeadConfig()
    structure_module: AlphaFold2StructureModuleHeadConfig = AlphaFold2StructureModuleHeadConfig()
    experimentally_resolved: AlphaFold2ExperimentallyResolvedHeadConfig = AlphaFold2ExperimentallyResolvedHeadConfig()


class AlphaFold2MSAConfig(Config):
    r"""Released predict-mode stochastic MSA realization constants.

    Under the frozen stochastic-fold contract the *Model* (not the Processor)
    realizes MSA features once per recycle from the raw alignment pool carried on
    :class:`AlphaFold2Input`: ``sample_msa`` -> ``make_masked_msa`` ->
    ``nearest_neighbor_clusters`` / ``summarize_clusters`` -> ``crop_extra_msa`` ->
    ``make_msa_feat``. These fields therefore live on the model config beside the
    computation that consumes them; every value is a DeepMind/OpenFold ``predict``
    constant (OpenFold ``config.py`` predict preset + ``input_pipeline.py``).
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
    num_ensemble: int = 1


class AlphaFold2Config(Config):
    r"""
    Configuration for [`AlphaFold2Model`][ofoldx.models.alphafold2.AlphaFold2Model].

    The nested fields expose AF2's executable architectural boundaries.
    """

    input_embedding: AlphaFold2InputEmbeddingConfig = AlphaFold2InputEmbeddingConfig()
    recycling: AlphaFold2RecyclingConfig = AlphaFold2RecyclingConfig()
    evoformer: AlphaFold2EvoformerConfig = AlphaFold2EvoformerConfig()
    template: AlphaFold2TemplateConfig = AlphaFold2TemplateConfig()
    heads: AlphaFold2HeadsConfig = AlphaFold2HeadsConfig()
    msa: AlphaFold2MSAConfig = AlphaFold2MSAConfig()

    recycling_steps: int = 4


class AlphaFold2ProcessorConfig(Config):
    r"""Deterministic limits applied while lowering template evidence.

    The Processor lowers the *raw* alignment pool (all rows, query-projected) and
    the full-MSA HHblits profile; the stochastic sample/mask/cluster/crop limits
    now live on :class:`AlphaFold2MSAConfig` because the Model owns that draw.
    """

    max_templates: int = 4


# ===========================================================================
# AlphaFold-Multimer configuration.
#
# Sibling config classes to the monomer surface above. The registry forbids two
# Model classes sharing one Model config class (``registry.py:76-78``), so the
# multimer Model needs its own ``config_class``; it reuses the shared recycling and
# Evoformer groups defined above and locks the multimer-specific values drawn from
# OpenFold's ``multimer_config_update`` (``openfold/config.py:800``).
#
# The public surface follows the same exclusions as the monomer config: no runtime
# placement, operator dispatch, checkpointing, loss weights, or source-checkpoint
# metadata.
# ===========================================================================


class AlphaFold2MultimerStructureModuleHeadConfig(AlphaFold2StructureModuleHeadConfig):
    r"""AF-Multimer structure module surface.

    Identical to the monomer surface except the backbone translation scale, which the
    released multimer graph sets to ``20`` (``openfold/config.py`` multimer
    ``trans_scale_factor``) versus the monomer ``10``.
    """

    position_scale: float = 20.0


class AlphaFold2MultimerInputEmbeddingConfig(Config):
    r"""
    Chain-relative input feature embedding config.

    ``max_relative_idx`` bounds the within-chain relative position one-hot and
    ``max_relative_chain`` bounds the cross-chain (asym) relative index. When
    ``use_chain_relative`` is set the positional encoding also carries a
    same-entity flag, matching OpenFold ``InputEmbedderMultimer``.
    """

    target_feature_size: int = 21
    msa_feature_size: int = 49
    extra_msa_feature_size: int = 25
    max_relative_idx: int = 32
    max_relative_chain: int = 2
    use_chain_relative: bool = True


class AlphaFold2MultimerTemplateConfig(Config):
    r"""
    Multimer template embedding config.

    The multimer template embedder consumes atom37 template coordinates and emits
    the pseudo-beta distogram plus a backbone unit-vector; it carries no template
    torsion-angle MSA row (``embed_torsion_angles`` is therefore ``False``) and
    always uses the template unit-vector (``use_template_unit_vector`` is ``True``).
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


class AlphaFold2MultimerMaskedMSAHeadConfig(Config):
    r"""
    Masked-MSA reconstruction head config.

    The multimer masked-MSA head emits ``22`` classes, not the monomer ``23``
    (``openfold/config.py:963``).
    """

    num_output: int = 22


class AlphaFold2MultimerPredictedAlignedErrorHeadConfig(Config):
    r"""
    Predicted-aligned-error / weighted-pTM head config.

    The multimer TM head is always enabled and additionally produces an interface
    ipTM branch. ``ranking_score`` is the released ``weighted_ptm_score``
    ``iptm_weight * iptm + ptm_weight * ptm`` (``openfold/model/heads.py:82-88``).
    """

    enabled: bool = True
    max_error_bin: float = 31.0
    num_bins: int = 64
    interface: bool = True
    ptm_weight: float = 0.2
    iptm_weight: float = 0.8


class AlphaFold2MultimerChainCenterOfMassHeadConfig(Config):
    r"""
    Chain center-of-mass head config.

    Part of the released module set (``openfold/config.py:974``); inference-inert
    but present so a strict checkpoint load sees its parameters.
    """

    enabled: bool = True


class AlphaFold2MultimerHeadsConfig(Config):
    r"""
    Multimer ``model.heads`` config group.
    """

    distogram: AlphaFold2DistogramHeadConfig = AlphaFold2DistogramHeadConfig()
    masked_msa: AlphaFold2MultimerMaskedMSAHeadConfig = AlphaFold2MultimerMaskedMSAHeadConfig()
    predicted_aligned_error: AlphaFold2MultimerPredictedAlignedErrorHeadConfig = (
        AlphaFold2MultimerPredictedAlignedErrorHeadConfig()
    )
    predicted_lddt: AlphaFold2PredictedLDDTHeadConfig = AlphaFold2PredictedLDDTHeadConfig()
    structure_module: AlphaFold2MultimerStructureModuleHeadConfig = AlphaFold2MultimerStructureModuleHeadConfig()
    experimentally_resolved: AlphaFold2ExperimentallyResolvedHeadConfig = AlphaFold2ExperimentallyResolvedHeadConfig()
    chain_center_of_mass: AlphaFold2MultimerChainCenterOfMassHeadConfig = (
        AlphaFold2MultimerChainCenterOfMassHeadConfig()
    )


class AlphaFold2MultimerMSAConfig(Config):
    r"""Released AlphaFold-Multimer stochastic MSA realization constants.

    Under the frozen stochastic-fold contract the *Model* (not the Processor)
    realizes MSA features once per recycle from the raw alignment pool carried on
    :class:`AlphaFold2MultimerInput`, mirroring OpenFold's multimer
    ``ensembled_transform_fns``: ``sample_msa`` (Gumbel argsort, row-0 preserved via
    ``cluster_bias_mask``) -> ``make_masked_msa`` (22-class Gumbel-max BERT, *before*
    clustering) -> ``nearest_neighbor_clusters`` (softmax soft assignment) ->
    ``create_msa_feat``. Every value is a DeepMind/OpenFold multimer ``predict``
    constant (``multimer_config_update`` + base ``masked_msa`` block). The extra-MSA
    axis is cropped inside ``sample_msa`` itself, so there is no separate crop step.

    ``max_msa_clusters`` / ``max_extra_msa`` are the ``model_[1-3]_multimer_v3``
    predict values (508 / 2048); the earlier ``_v2`` presets used 252 / 1152.
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
    num_ensemble: int = 1


class AlphaFold2MultimerConfig(Config):
    r"""
    Configuration for
    [`AlphaFold2MultimerModel`][ofoldx.models.alphafold2.AlphaFold2MultimerModel].

    The nested groups reuse the shared recycling and Evoformer trunk unchanged and
    lock the multimer-specific input-embedding, template, and head values.
    """

    input_embedding: AlphaFold2MultimerInputEmbeddingConfig = AlphaFold2MultimerInputEmbeddingConfig()
    recycling: AlphaFold2RecyclingConfig = AlphaFold2RecyclingConfig()
    # AF-Multimer applies the outer-product-mean at the START of every Evoformer /
    # extra-MSA iteration (OpenFold ``opm_first=True``); the monomer default is opm-last.
    evoformer: AlphaFold2EvoformerConfig = AlphaFold2EvoformerConfig(
        {"iteration": {"outer_product_mean": {"first": True}}}
    )
    template: AlphaFold2MultimerTemplateConfig = AlphaFold2MultimerTemplateConfig()
    heads: AlphaFold2MultimerHeadsConfig = AlphaFold2MultimerHeadsConfig()
    msa: AlphaFold2MultimerMSAConfig = AlphaFold2MultimerMSAConfig()

    recycling_steps: int = 4
    num_ensembles: int = 1


class AlphaFold2MultimerProcessorConfig(Config):
    r"""Deterministic limits applied while lowering paired MSA and template evidence."""

    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    max_templates: int = 4
