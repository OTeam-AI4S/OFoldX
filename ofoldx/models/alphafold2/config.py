r"""
AlphaFold 2 configuration.

The public surface mirrors the official DeepMind AlphaFold 2 model config names
where they affect parameter layout while keeping OFoldX module boundaries
explicit: input embedding, recycling, template embedding, Evoformer, structure
module, heads, and ``recycling_steps``. Top-level feature-size fields expose the
processor and artifact metadata needed by OFoldX without changing AF2's nested
architecture groups.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


class AlphaFold2GlobalConfig(Config):
    r"""
    Global AlphaFold 2 execution knobs.

    These names follow the official config and are read by future converters even
    when the current PyTorch implementation only needs a subset.
    """

    deterministic: bool = False
    subbatch_size: int = 4
    use_remat: bool = False
    zero_init: bool = True
    eval_dropout: bool = False


class AlphaFold2PrevPosConfig(Config):
    r"""
    Previous-position distogram recycling bins.
    """

    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15


class AlphaFold2AttentionConfig(Config):
    r"""
    Official-style attention block config.
    """

    num_head: int = 8
    value_dim: int = 32
    gating: bool = True
    orientation: str = "per_row"
    dropout_rate: float = 0.0
    shared_dropout: bool = False


class AlphaFold2MsaRowAttentionWithPairBiasConfig(AlphaFold2AttentionConfig):
    orientation: str = "per_row"
    dropout_rate: float = 0.15
    shared_dropout: bool = True


class AlphaFold2MsaColumnAttentionConfig(AlphaFold2AttentionConfig):
    orientation: str = "per_column"


class AlphaFold2TriangleAttentionConfig(AlphaFold2AttentionConfig):
    num_head: int = 4
    value_dim: int = 32
    dropout_rate: float = 0.25
    shared_dropout: bool = True


class AlphaFold2TransitionConfig(Config):
    r"""
    Transition feed-forward expansion.
    """

    num_intermediate_factor: int = 4


class AlphaFold2OuterProductMeanConfig(Config):
    r"""
    Outer-product-mean projection config.
    """

    first: bool = False
    chunk_size: int = 128
    num_outer_channel: int = 32
    dropout_rate: float = 0.0
    orientation: str = "per_row"
    shared_dropout: bool = True


class AlphaFold2TriangleMultiplicationConfig(Config):
    r"""
    Triangle multiplicative update config.
    """

    equation: str = "ikc,jkc->ijc"
    num_intermediate_channel: int = 128
    dropout_rate: float = 0.25
    orientation: str = "per_row"
    shared_dropout: bool = True


class AlphaFold2TriangleMultiplicationOutgoingConfig(AlphaFold2TriangleMultiplicationConfig):
    equation: str = "ikc,jkc->ijc"


class AlphaFold2TriangleMultiplicationIncomingConfig(AlphaFold2TriangleMultiplicationConfig):
    equation: str = "kjc,kic->ijc"


class AlphaFold2EvoformerIterationConfig(Config):
    r"""
    One official Evoformer iteration's module-level knobs.

    The current implementation maps these values onto OFoldX's MSA/pair update
    blocks. The field names intentionally match DeepMind's config keys so
    released parameter names can be routed module-by-module later.
    """

    msa_row_attention_with_pair_bias: AlphaFold2MsaRowAttentionWithPairBiasConfig = (
        AlphaFold2MsaRowAttentionWithPairBiasConfig()
    )
    msa_column_attention: AlphaFold2MsaColumnAttentionConfig = AlphaFold2MsaColumnAttentionConfig()
    msa_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig()
    outer_product_mean: AlphaFold2OuterProductMeanConfig = AlphaFold2OuterProductMeanConfig()
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationOutgoingConfig = (
        AlphaFold2TriangleMultiplicationOutgoingConfig()
    )
    triangle_multiplication_incoming: AlphaFold2TriangleMultiplicationIncomingConfig = (
        AlphaFold2TriangleMultiplicationIncomingConfig()
    )
    triangle_attention_starting_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig()
    triangle_attention_ending_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig()
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
    triangle_attention_ending_node: AlphaFold2TriangleAttentionConfig = AlphaFold2TriangleAttentionConfig(
        {"value_dim": 16}
    )
    triangle_multiplication_outgoing: AlphaFold2TriangleMultiplicationOutgoingConfig = (
        AlphaFold2TriangleMultiplicationOutgoingConfig({"num_intermediate_channel": 64})
    )
    triangle_multiplication_incoming: AlphaFold2TriangleMultiplicationIncomingConfig = (
        AlphaFold2TriangleMultiplicationIncomingConfig({"num_intermediate_channel": 64})
    )
    pair_transition: AlphaFold2TransitionConfig = AlphaFold2TransitionConfig({"num_intermediate_factor": 2})


class AlphaFold2TemplatePointwiseAttentionConfig(Config):
    r"""
    Template pointwise attention config.
    """

    num_head: int = 4


class AlphaFold2TemplateConfig(Config):
    r"""
    Template embedding config with official names.
    """

    enabled: bool = True
    embed_torsion_angles: bool = True
    max_templates: int = 4
    min_bin: float = 3.25
    max_bin: float = 50.75
    num_distogram_bins: int = 39
    vocab_size: int = 22
    use_template_unit_vector: bool = True
    template_pair_stack: AlphaFold2TemplatePairStackConfig = AlphaFold2TemplatePairStackConfig()
    attention: AlphaFold2TemplatePointwiseAttentionConfig = AlphaFold2TemplatePointwiseAttentionConfig()


class AlphaFold2InputEmbeddingConfig(Config):
    r"""
    Input feature embedding config.
    """

    target_feature_size: int = 22
    msa_feature_size: int = 49
    extra_msa_feature_size: int = 25
    extra_msa_vocab_size: int = 23
    ofoldx_msa_feature_size: int = 34
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
    operator_backend: str = "auto"
    blocks_per_checkpoint: int | None = 1
    iteration: AlphaFold2EvoformerIterationConfig = AlphaFold2EvoformerIterationConfig()


class AlphaFold2DistogramHeadConfig(Config):
    r"""
    Distogram head config.
    """

    first_break: float = 2.3125
    last_break: float = 21.6875
    num_bins: int = 64
    weight: float = 0.3


class AlphaFold2PredictedAlignedErrorHeadConfig(Config):
    r"""
    Predicted aligned error / pTM head config.
    """

    enabled: bool = True
    max_error_bin: float = 31.0
    num_bins: int = 64
    num_channels: int = 128
    weight: float = 0.1


class AlphaFold2PredictedLDDTHeadConfig(Config):
    r"""
    Predicted lDDT confidence head config.
    """

    num_bins: int = 50
    num_channels: int = 128
    weight: float = 0.01


class AlphaFold2ExperimentallyResolvedHeadConfig(Config):
    r"""
    Experimentally-resolved atom37 head config.
    """

    filter_by_resolution: bool = True
    max_resolution: float = 3.0
    min_resolution: float = 0.1
    num_atom_types: int = 37
    weight: float = 0.0


class AlphaFold2MaskedMSAHeadConfig(Config):
    r"""
    Masked-MSA reconstruction head config.
    """

    num_output: int = 23
    weight: float = 2.0


class AlphaFold2FapeConfig(Config):
    r"""
    Structure-module FAPE loss config names.
    """

    clamp_distance: float = 10.0
    loss_unit_distance: float = 10.0


class AlphaFold2SidechainConfig(Config):
    r"""
    Sidechain loss/config names used by the official structure module.
    """

    atom_clamp_distance: float = 10.0
    length_scale: float = 10.0
    num_channel: int = 128
    num_residual_block: int = 2
    weight_frac: float = 0.5


class AlphaFold2StructureModuleHeadConfig(Config):
    r"""
    Official StructureModule config surface.

    The model consumes the channel/layer, IPA, and sidechain fields for the
    native PyTorch structure module. FAPE and violation fields are kept for the
    training/loss parity slice.
    """

    compute_in_graph_metrics: bool = True
    dropout: float = 0.1
    num_channel: int = 384
    num_head: int = 12
    num_layer: int = 8
    num_layer_in_transition: int = 3
    num_point_qk: int = 4
    num_point_v: int = 8
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    position_scale: float = 10.0
    structural_violation_loss_weight: float = 1.0
    weight: float = 1.0

    fape: AlphaFold2FapeConfig = AlphaFold2FapeConfig()
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


@MODEL_CONFIGS.register("alphafold2")
class AlphaFold2Config(Config):
    r"""
    Configuration for [`AlphaFold2Model`][ofoldx.models.alphafold2.AlphaFold2Model].

    The top-level nested fields expose AF2's architectural boundaries. Feature
    widths such as ``single_size``, ``pair_size``, ``token_feature_size``, and
    ``atom_feature_size`` are mirrored at the top level for processor and
    artifact metadata.
    """

    architecture: str = "alphafold2"
    initializer_range: float = 0.02

    global_config: AlphaFold2GlobalConfig = AlphaFold2GlobalConfig()
    input_embedding: AlphaFold2InputEmbeddingConfig = AlphaFold2InputEmbeddingConfig()
    recycling: AlphaFold2RecyclingConfig = AlphaFold2RecyclingConfig()
    evoformer: AlphaFold2EvoformerConfig = AlphaFold2EvoformerConfig()
    template: AlphaFold2TemplateConfig = AlphaFold2TemplateConfig()
    heads: AlphaFold2HeadsConfig = AlphaFold2HeadsConfig()

    recycling_steps: int = 3
    resample_msa_in_recycling: bool = True
    recycle_early_stop_tolerance: float = 0.0

    # OFoldX processor/artifact metadata.
    token_feature_size: int = 449
    atom_feature_size: int = 385
    single_size: int = 384
    pair_size: int = 128
    msa_size: int = 256
    atom37_count: int = 37
