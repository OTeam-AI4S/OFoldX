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
ESMFold2 architecture configuration.
"""

from __future__ import annotations

from chanfig import Config

from ...data.tokenization import TOKENS

#: Raw token feature width: the atom encoder's 384 token features, the residue and MSA-profile planes over the 72
#: canonical classes, and the deletion mean.
_INPUT_SIZE = 384 + 2 * len(TOKENS) + 1


class EsmFold2AtomEncoderConfig(Config):
    r"""
    Configure sliding-window atom attention with three-dimensional rotary features.

    Attributes:
        atom_hidden_size: Width of atom single features.
        token_hidden_size: Token projection width; the atom encoder emits half this width.
        num_blocks: Number of atom attention blocks.
        num_attention_heads: Number of atom attention heads.
        sliding_window_size: Width of the local atom attention window.
        expansion: Feed-forward expansion factor.
        spatial_rope_base_frequency: Base frequency for Cartesian rotary features.
        num_spatial_rope_pairs_per_axis: Number of rotary feature pairs per Cartesian axis.
        num_reference_space_rope_pairs: Number of rotary pairs for reference-space identifiers.
        reference_space_rope_base_frequency: Base frequency for reference-space rotary features.
    """

    atom_hidden_size: int = 128
    token_hidden_size: int = 768
    num_blocks: int = 3
    num_attention_heads: int = 4
    sliding_window_size: int = 128
    expansion: int = 2
    spatial_rope_base_frequency: float = 20.0
    num_spatial_rope_pairs_per_axis: int = 2
    num_reference_space_rope_pairs: int = 10
    reference_space_rope_base_frequency: float = 10000.0


class EsmFold2InputEmbeddingsConfig(Config):
    r"""
    Configure ESMFold2 token and atom input embeddings.

    Attributes:
        input_size: Width of raw token and atom input features.
        atom_encoder: Sliding-window atom encoder configuration.
    """

    input_size: int = _INPUT_SIZE
    atom_encoder: EsmFold2AtomEncoderConfig = EsmFold2AtomEncoderConfig()


class EsmFold2EsmCConfig(Config):
    r"""
    Configure the ESMC child module.

    Attributes:
        vocab_size: Number of ESMC token types.
        hidden_size: Width of transformer hidden states.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Number of self-attention heads.
        intermediate_size: Width of feed-forward intermediate states.
        hidden_act: Feed-forward activation function.
        hidden_dropout: Dropout probability on hidden states.
        attention_dropout: Dropout probability on attention weights.
        max_position_embeddings: Maximum supported sequence length.
        initializer_range: Standard deviation used for parameter initialization.
        layer_norm_eps: Epsilon used by layer normalization.
        attention_bias: Whether attention projections include biases.
        feedforward_bias: Whether feed-forward projections include biases.
        attention_layer_norm_bias: Whether attention layer normalization includes a bias.
        qk_layer_norm: Whether to normalize query and key projections.
        qk_layer_norm_bias: Whether query/key layer normalization includes a bias.
        final_layer_norm_bias: Whether final layer normalization includes a bias.
        residue_scaling_factor: Optional residual scaling factor; ``None`` disables scaling.
        pad_token_id: Padding token identifier.
        bos_token_id: Beginning-of-sequence token identifier.
        eos_token_id: End-of-sequence token identifier.
        unk_token_id: Unknown-residue token identifier.
        mask_token_id: Mask token identifier.
        null_token_id: Null token identifier.
    """

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


class EsmFold2LanguageModelEncoderConfig(Config):
    r"""
    Configure the pair encoder for projected ESMC representations.

    Attributes:
        enabled: Whether to run the language-model pair encoder.
        num_blocks: Number of pair encoder blocks.
        dropout: Dropout probability in pair encoder blocks.
        per_loop_dropout: Whether to redraw language-model dropout for each recycling loop.
    """

    enabled: bool = True
    num_blocks: int = 4
    dropout: float = 0.25
    per_loop_dropout: bool = True


class EsmFold2MsaEncoderConfig(Config):
    r"""
    Configure the single-sequence or MSA encoder in the folding head.

    Attributes:
        enabled: Whether to run the MSA encoder.
        msa_size: Width of MSA representations.
        hidden_size: Width of projected MSA features.
        num_stages: Number of MSA encoder stages, including the distinct final
            pair-only stage in the standard model.
        num_msa_attention_heads: Number of MSA attention heads.
        msa_attention_head_size: Per-head width of MSA attention.
    """

    enabled: bool = True
    msa_size: int = 128
    hidden_size: int = 32
    num_stages: int = 4
    num_msa_attention_heads: int = 8
    msa_attention_head_size: int = 16


class EsmFold2FoldingStackConfig(Config):
    r"""
    Configure a homogeneous stack of pair-only folding blocks.

    Attributes:
        num_blocks: Number of folding blocks.
    """

    num_blocks: int = 48


class EsmFold2ParcaeConfig(Config):
    r"""
    Configure the linear-recurrent trunk coda.

    Attributes:
        num_coda_blocks: Number of coda blocks applied after recurrent folding.
    """

    num_coda_blocks: int = 2


class EsmFold2DenoiserConfig(Config):
    r"""
    Configure the ESMFold2 diffusion score network.

    Attributes:
        sigma_data: Data noise scale used by EDM preconditioning.
        atom_hidden_size: Width of atom single features.
        token_hidden_size: Width of denoiser token features.
        pair_size: Width of denoiser pair features.
        input_size: Width of raw atom and token features.
        fourier_embedding_size: Width of diffusion-noise Fourier features.
        num_atom_blocks: Number of atom attention blocks.
        num_atom_attention_heads: Number of atom attention heads.
        num_token_encoder_blocks: Number of token-encoder blocks.
        num_token_encoder_attention_heads: Number of token-encoder attention heads.
        expansion: Feed-forward expansion factor.
    """

    sigma_data: float = 16.0
    atom_hidden_size: int = 128
    token_hidden_size: int = 768
    pair_size: int = 256
    input_size: int = _INPUT_SIZE
    fourier_embedding_size: int = 256
    num_atom_blocks: int = 3
    num_atom_attention_heads: int = 4
    num_token_encoder_blocks: int = 12
    num_token_encoder_attention_heads: int = 16
    expansion: int = 2


class EsmFold2DiffusionStructureHeadConfig(Config):
    r"""
    Configure the diffusion structure head and sampler defaults.

    Attributes:
        denoiser: Diffusion score-network configuration.
        num_distogram_bins: Number of structure-head distogram bins.
        churn_rate: Stochastic churn added during sampling.
        min_churn_noise_level: Noise threshold below which churn is disabled.
        noise_scale: Multiplier on churn-injected noise.
        step_scale: Multiplier on each denoising step.
        max_noise_level: Highest noise level in the schedule.
        min_noise_level: Lowest non-zero noise level in the schedule.
        rho: Exponent shaping noise-level spacing.
        num_sampling_steps: Number of reverse-diffusion steps.
        max_inference_sigma: Absolute sampling cap. ``None`` keeps the full schedule.
    """

    denoiser: EsmFold2DenoiserConfig = EsmFold2DenoiserConfig()
    num_distogram_bins: int = 64
    churn_rate: float = 0.8
    min_churn_noise_level: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    rho: float = 7.0
    num_sampling_steps: int = 14
    max_inference_sigma: float | None = 256.0


class EsmFold2ConfidenceHeadConfig(Config):
    r"""
    Configure per-site and pairwise confidence heads.

    Attributes:
        num_plddt_bins: Number of pLDDT output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        min_distance: Lower confidence distogram bound, in angstroms.
        max_distance: Upper confidence distogram bound, in angstroms.
        num_distogram_bins: Number of confidence distance bins.
        folding_stack: Four-block pair stack used by the confidence head.
    """

    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    min_distance: float = 3.25
    max_distance: float = 50.75
    num_distogram_bins: int = 39
    folding_stack: EsmFold2FoldingStackConfig = EsmFold2FoldingStackConfig({"num_blocks": 4})


class EsmFold2Config(Config):
    r"""
    Configure the complete ESMC and ESMFold2 folding graph.

    Examples:
        >>> config = EsmFold2Config({"num_recycles": 2})
        >>> config.num_recycles
        2

    Attributes:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        num_relative_residue_bins: Number of within-chain relative-position bins.
        num_relative_chain_bins: Number of relative-chain bins.
        num_recycles: Number of folding trunk recycling passes.
        language_model_mask_probability: Default probability of masking non-special ESMC tokens.
        msa_encoder_overwrite: Whether MSA pair features replace rather than augment language-model pair features.
        input_embeddings: Token and sliding-window atom input embedding configuration.
        esmc: MultiMolecule ESMC language-model configuration.
        esmc_repository: Optional MM repository supplying the frozen ESMC weights.
        esmc_revision: Revision of the referenced MM repository.
        language_model_encoder: Per-loop language-model pair encoder configuration.
        msa_encoder: MSA encoder configuration.
        folding_stack: Forty-eight-block pair folding stack.
        parcae: Linear-recurrent trunk coda configuration.
        diffusion_structure_head: Diffusion score network and 14-step sampler configuration.
        confidence: pLDDT, PDE, and PAE heads with a four-block confidence trunk.
    """

    single_size: int = 384
    pair_size: int = 256
    num_relative_residue_bins: int = 32
    num_relative_chain_bins: int = 2
    num_recycles: int = 4
    language_model_mask_probability: float = 0.0
    msa_encoder_overwrite: bool = True

    input_embeddings: EsmFold2InputEmbeddingsConfig = EsmFold2InputEmbeddingsConfig()
    esmc: EsmFold2EsmCConfig = EsmFold2EsmCConfig()
    esmc_repository: str | None = None
    esmc_revision: str | None = None
    language_model_encoder: EsmFold2LanguageModelEncoderConfig = EsmFold2LanguageModelEncoderConfig()
    msa_encoder: EsmFold2MsaEncoderConfig = EsmFold2MsaEncoderConfig()
    folding_stack: EsmFold2FoldingStackConfig = EsmFold2FoldingStackConfig()
    parcae: EsmFold2ParcaeConfig = EsmFold2ParcaeConfig()
    diffusion_structure_head: EsmFold2DiffusionStructureHeadConfig = EsmFold2DiffusionStructureHeadConfig()
    confidence: EsmFold2ConfidenceHeadConfig = EsmFold2ConfidenceHeadConfig()


class EsmFold2ProcessorConfig(Config):
    r"""
    Configure ESMFold2 input processing.

    Attributes:
        msa_conditioning: Whether the processor accepts MSA evidence.
        seed: Seed used to embed missing reference conformers.
    """

    msa_conditioning: bool = True
    seed: int = 42


class EsmFold2ExperimentalConfidenceHeadConfig(Config):
    r"""
    Configure the narrower pLDDT/PAE head of recurrent experimental ESMFold2 graphs.

    Attributes:
        enabled: Whether to build the confidence head and emit confidence outputs.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        min_distance: Lower confidence distogram bound, in angstroms.
        max_distance: Upper confidence distogram bound, in angstroms.
        num_distogram_bins: Number of coarse confidence distance bins.
        folding_stack: Four-block pair stack used by the confidence head.
    """

    enabled: bool = True
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    min_distance: float = 2.0
    max_distance: float = 52.0
    num_distogram_bins: int = 128
    folding_stack: EsmFold2FoldingStackConfig = EsmFold2FoldingStackConfig({"num_blocks": 4})


class EsmFold2ExperimentalConfig(Config):
    r"""
    Configure the recurrent ``ESMFold2-Experimental-Fast`` binder-design graph.

    Examples:
        >>> config = EsmFold2ExperimentalConfig({"use_msa_features": True})
        >>> config.use_msa_features
        True

    Attributes:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        num_relative_residue_bins: Number of within-chain relative-position bins.
        num_relative_chain_bins: Number of relative-chain bins.
        num_recycles: Number of recurrent folding passes.
        use_msa_features: Whether to use MSA features during sequence inversion.
        language_model_dropout: Dropout probability on language-model representations.
        force_language_model_dropout: Whether to apply language-model dropout outside training mode.
        input_embeddings: Token and sliding-window atom input embedding configuration.
        esmc: MultiMolecule ESMC language-model configuration.
        esmc_repository: Optional MM repository supplying the frozen ESMC and masked-language-model head.
        esmc_revision: Revision of the referenced MM repository.
        esmc_dtype: Floating dtype of a referenced ESMC trunk, recorded when saving.
        esmc_lm_head_dtype: Floating dtype of the referenced masked-language-model head, recorded when saving.
        msa_encoder: MSA encoder, disabled by default with a 32-wide attention head.
        folding_stack: Recurrent 24-block pair stack.
        diffusion_structure_head: Diffusion head overridden to 128 distogram bins.
        confidence: Experimental pLDDT/PAE confidence-head configuration.
    """

    single_size: int = 384
    pair_size: int = 256
    num_relative_residue_bins: int = 32
    num_relative_chain_bins: int = 2
    num_recycles: int = 4
    use_msa_features: bool = False
    language_model_dropout: float = 0.0
    force_language_model_dropout: bool = False

    input_embeddings: EsmFold2InputEmbeddingsConfig = EsmFold2InputEmbeddingsConfig()
    esmc: EsmFold2EsmCConfig = EsmFold2EsmCConfig()
    esmc_repository: str | None = None
    esmc_revision: str | None = None
    esmc_dtype: str | None = None
    esmc_lm_head_dtype: str | None = None
    msa_encoder: EsmFold2MsaEncoderConfig = EsmFold2MsaEncoderConfig({"enabled": False, "msa_attention_head_size": 32})
    folding_stack: EsmFold2FoldingStackConfig = EsmFold2FoldingStackConfig({"num_blocks": 24})
    diffusion_structure_head: EsmFold2DiffusionStructureHeadConfig = EsmFold2DiffusionStructureHeadConfig(
        {"num_distogram_bins": 128}
    )
    confidence: EsmFold2ExperimentalConfidenceHeadConfig = EsmFold2ExperimentalConfidenceHeadConfig()


class EsmFold2ExperimentalProcessorConfig(EsmFold2ProcessorConfig):
    r"""
    Configure input processing for recurrent ESMFold2 binder design.

    Attributes:
        msa_conditioning: Whether the processor accepts MSA evidence.
    """

    msa_conditioning: bool = False
