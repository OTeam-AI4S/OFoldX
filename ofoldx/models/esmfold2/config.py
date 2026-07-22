r"""Configuration for the released ESMFold2 architecture."""

from __future__ import annotations

from chanfig import Config


class EsmFold2AtomEncoderConfig(Config):
    r"""Sliding-window atom attention with three-dimensional rotary features."""

    atom_hidden_size: int = 128
    token_hidden_size: int = 768
    num_blocks: int = 3
    num_attention_heads: int = 4
    sliding_window_size: int = 128
    expansion: int = 2
    spatial_rope_base_frequency: float = 20.0
    num_spatial_rope_pairs_per_axis: int = 2
    num_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0


class EsmFold2InputEmbeddingsConfig(Config):
    r"""Token and atom input embedding."""

    input_size: int = 451
    atom_encoder: EsmFold2AtomEncoderConfig = EsmFold2AtomEncoderConfig()


class EsmFold2EsmCConfig(Config):
    r"""Architecture of the MultiMolecule ESMC child module."""

    vocab_size: int = 37
    hidden_size: int = 2560
    num_hidden_layers: int = 80
    num_attention_heads: int = 40
    intermediate_size: int = 6912
    hidden_act: str = "swiglu"
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1.0e-5
    attention_bias: bool = False
    feedforward_bias: bool = False
    attention_layer_norm_bias: bool = True
    qk_layer_norm: bool = True
    qk_layer_norm_bias: bool = False
    final_layer_norm_bias: bool = False
    residue_scaling_factor: float | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    mask_token_id: int = 4
    null_token_id: int = 5


class EsmFold2LmEncoderConfig(Config):
    r"""Pair encoder for projected ESMC representations."""

    enabled: bool = True
    num_blocks: int = 4
    dropout: float = 0.25
    per_loop_dropout: bool = True


class EsmFold2MsaEncoderConfig(Config):
    r"""Single-sequence/MSA encoder in the released folding head."""

    enabled: bool = True
    msa_size: int = 128
    hidden_size: int = 32
    num_blocks: int = 4
    num_msa_attention_heads: int = 8
    msa_head_dim: int = 16


class EsmFold2FoldingTrunkConfig(Config):
    r"""Pair-only folding trunk."""

    num_blocks: int = 48


class EsmFold2ParcaeConfig(Config):
    r"""Linear-recurrent trunk coda."""

    num_coda_blocks: int = 2


class EsmFold2DiffusionModuleConfig(Config):
    r"""Diffusion score network."""

    sigma_data: float = 16.0
    atom_hidden_size: int = 128
    token_hidden_size: int = 768
    pair_size: int = 256
    input_size: int = 451
    fourier_embedding_size: int = 256
    num_atom_blocks: int = 3
    num_atom_attention_heads: int = 4
    num_token_blocks: int = 12
    num_token_attention_heads: int = 16
    expansion: int = 2


class EsmFold2DiffusionStructureHeadConfig(Config):
    r"""Diffusion structure head and sampler defaults."""

    diffusion_module: EsmFold2DiffusionModuleConfig = EsmFold2DiffusionModuleConfig()
    num_distogram_bins: int = 64
    churn_rate: float = 0.8
    min_churn_noise_level: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
    num_sampling_steps: int = 14


class EsmFold2ConfidenceHeadConfig(Config):
    r"""Per-site and pairwise confidence heads."""

    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    min_distance: float = 3.25
    max_distance: float = 50.75
    num_distogram_bins: int = 39
    folding_trunk: EsmFold2FoldingTrunkConfig = EsmFold2FoldingTrunkConfig({"num_blocks": 4})


class EsmFold2Config(Config):
    r"""Complete graph construction for ESMC plus the released ESMFold2 head."""

    single_size: int = 384
    pair_size: int = 256
    num_relative_residue_bins: int = 32
    num_relative_chain_bins: int = 2
    recycling_steps: int = 4
    language_model_mask_probability: float = 0.0
    msa_encoder_overwrite: bool = True

    input_embeddings: EsmFold2InputEmbeddingsConfig = EsmFold2InputEmbeddingsConfig()
    esmc: EsmFold2EsmCConfig = EsmFold2EsmCConfig()
    lm_encoder: EsmFold2LmEncoderConfig = EsmFold2LmEncoderConfig()
    msa_encoder: EsmFold2MsaEncoderConfig = EsmFold2MsaEncoderConfig()
    folding_trunk: EsmFold2FoldingTrunkConfig = EsmFold2FoldingTrunkConfig()
    parcae: EsmFold2ParcaeConfig = EsmFold2ParcaeConfig()
    diffusion_structure_head: EsmFold2DiffusionStructureHeadConfig = EsmFold2DiffusionStructureHeadConfig()
    confidence_head: EsmFold2ConfidenceHeadConfig = EsmFold2ConfidenceHeadConfig()


class EsmFold2ProcessorConfig(Config):
    r"""Input contract paired with one released ESMFold2 checkpoint."""

    msa_conditioning: bool = True


class EsmFold2ExperimentalConfidenceHeadConfig(Config):
    r"""Confidence head of the experimental recurrent ESMFold2 graph.

    A narrower confidence contract than the released head: pLDDT and PAE only
    (no PDE head, no resolved head) and a 128-bin coarse-distogram pair embedding
    over a 2.0-52.0 angstrom range. Implemented by
    [`EsmFold2ExperimentalConfidenceHead`][ofoldx.models.esmfold2.model.EsmFold2ExperimentalConfidenceHead].

    ``enabled`` reflects whether the checkpoint ships a confidence head. The design/inversion
    and hero-critic checkpoints train it (``True``); the scaling-study critics are distogram-
    only and disable it (``confidence_head.enabled == false`` in their config.json), so the
    module is absent from their state dict and confidence outputs are unavailable.
    """

    enabled: bool = True
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    min_distance: float = 2.0
    max_distance: float = 52.0
    num_distogram_bins: int = 128
    folding_trunk: EsmFold2FoldingTrunkConfig = EsmFold2FoldingTrunkConfig({"num_blocks": 4})


class EsmFold2ExperimentalConfig(Config):
    r"""Graph construction for the experimental recurrent ESMFold2 binder-design head.

    A different architecture from [`EsmFold2Config`][ofoldx.models.esmfold2.EsmFold2Config]:
    the older recurrent trunk re-injects the previous pair representation through a
    zero-initialised ``pair_loop_proj`` each loop (no ``parcae`` linear-recurrent coda,
    no per-loop ``lm_encoder`` trunk, no ``msa_encoder_overwrite``), integrates the ESMC
    projection into ``z_init`` once, and carries the narrower experimental confidence
    contract with a 128-bin distogram head. ``disable_msa_features`` mirrors the released
    ``ESMFold2-Experimental-Fast`` single-sequence inversion checkpoint.

    Defaults reflect ``biohub/ESMFold2-Experimental-Fast`` (the primary design inversion
    model: 24 trunk blocks, MSA disabled). The MSA-conditioned ``biohub/ESMFold2-Experimental``
    critic checkpoint sets ``folding_trunk.num_blocks = 48``, ``disable_msa_features = False``
    and ``msa_encoder.enabled = True``; construct those from the checkpoint ``config.json``.
    """

    single_size: int = 384
    pair_size: int = 256
    num_relative_residue_bins: int = 32
    num_relative_chain_bins: int = 2
    recycling_steps: int = 4
    disable_msa_features: bool = True
    language_model_dropout: float = 0.0
    force_language_model_dropout: bool = False

    # Algorithm 14 ESMC pseudo-perplexity sequence-regulariser gradient weight
    # (binder_design.py L948-L950): the fixed released hyperparameter blending the masked-LM
    # sequence prior into the structure objective during binder design. The released default
    # 0.15 is the general de-novo-binder weight; construct an antibody-framework design (fixed
    # framework, mutable CDR loops) with 0.05 so the structure objective dominates the short
    # CDR loops. ``EsmFold2DesignPipeline.design`` reads this value and never selects it at runtime.
    language_model_weight: float = 0.15

    input_embeddings: EsmFold2InputEmbeddingsConfig = EsmFold2InputEmbeddingsConfig()
    esmc: EsmFold2EsmCConfig = EsmFold2EsmCConfig()
    msa_encoder: EsmFold2MsaEncoderConfig = EsmFold2MsaEncoderConfig({"enabled": False, "msa_head_dim": 32})
    folding_trunk: EsmFold2FoldingTrunkConfig = EsmFold2FoldingTrunkConfig({"num_blocks": 24})
    diffusion_structure_head: EsmFold2DiffusionStructureHeadConfig = EsmFold2DiffusionStructureHeadConfig(
        {"num_distogram_bins": 128}
    )
    confidence_head: EsmFold2ExperimentalConfidenceHeadConfig = EsmFold2ExperimentalConfidenceHeadConfig()


class EsmFold2ExperimentalProcessorConfig(Config):
    r"""Input contract paired with an experimental ESMFold2 checkpoint."""

    msa_conditioning: bool = False
