r"""
ESMFold2 configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from chanfig import Config

from ..registry import MODEL_CONFIGS

ConfigT = TypeVar("ConfigT", bound=Config)


def _coerce_config(value: Config | Mapping[str, Any], config_class: type[ConfigT]) -> ConfigT:
    r"""Return ``value`` as ``config_class`` while preserving already-typed configs."""

    if isinstance(value, config_class):
        return cast(ConfigT, value)
    return config_class(value)


class ESMFold2MSAEncoderConfig(Config):
    r"""Optional MSA encoder used by the full ESMFold2 checkpoint."""

    enabled: bool = True
    msa_size: int = 128
    hidden_size: int = 32
    num_blocks: int = 4
    num_msa_attention_heads: int = 8
    msa_head_dim: int = 16


class ESMFold2ParcaeConfig(Config):
    r"""Linear-recurrent trunk ("parcae") config."""

    enabled: bool = True
    num_coda_blocks: int = 2


class ESMFold2LMEncoderConfig(Config):
    r"""LM-side pair encoder config."""

    enabled: bool = True
    num_blocks: int = 4
    dropout: float = 0.25
    per_loop_dropout: bool = True


class ESMFold2AtomAttentionConfig(Config):
    r"""Sliding-window atom attention with 3D rotary features."""

    atom_size: int = 128
    token_size: int = 768
    num_blocks: int = 3
    num_attention_heads: int = 4
    sliding_window_size: int = 128
    expansion_ratio: int = 2
    spatial_rope_base_frequency: float = 20.0
    num_spatial_rope_pairs_per_axis: int = 2
    num_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0


class ESMFold2FoldingTrunkConfig(Config):
    r"""Pair-only folding trunk config."""

    num_blocks: int = 48
    num_attention_heads: int = 8
    dropout: float = 0.25


class ESMFold2InputsEmbedderConfig(Config):
    r"""Input embedder config."""

    input_size: int = 451
    atom_encoder: ESMFold2AtomAttentionConfig = ESMFold2AtomAttentionConfig()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.atom_encoder = _coerce_config(self.atom_encoder, ESMFold2AtomAttentionConfig)


class ESMFold2DiffusionModuleConfig(Config):
    r"""Diffusion score network config."""

    sigma_data: float = 16.0
    atom_size: int = 128
    token_size: int = 768
    pair_size: int = 256
    input_size: int = 451
    fourier_dim: int = 256
    relpos_r_max: int = 32
    relpos_s_max: int = 2
    num_atom_blocks: int = 3
    num_atom_attention_heads: int = 4
    num_token_blocks: int = 12
    num_token_attention_heads: int = 16
    transition_multiplier: int = 2


class ESMFold2StructureHeadConfig(Config):
    r"""Diffusion structure head config."""

    diffusion_module: ESMFold2DiffusionModuleConfig = ESMFold2DiffusionModuleConfig()
    distogram_bins: int = 64
    train_noise_log_mean: float = -1.2
    train_noise_log_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    inference_s_max: float = 160.0
    inference_s_min: float = 4e-4
    inference_p: float = 7.0
    num_sampling_steps: int = 14

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion_module = _coerce_config(self.diffusion_module, ESMFold2DiffusionModuleConfig)


class ESMFold2ConfidenceHeadConfig(Config):
    r"""ESMFold2 confidence head config."""

    enabled: bool = True
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    min_dist: float = 3.25
    max_dist: float = 50.75
    distogram_bins: int = 39
    folding_trunk: ESMFold2FoldingTrunkConfig = ESMFold2FoldingTrunkConfig(
        {"num_blocks": 4, "num_attention_heads": 8, "dropout": 0.25}
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.folding_trunk = _coerce_config(self.folding_trunk, ESMFold2FoldingTrunkConfig)


@MODEL_CONFIGS.register("esmfold2")
class ESMFold2Config(Config):
    r"""
    Released ESMFold2 model configuration.

    Defaults match the public ``biohub/ESMFold2`` Hub artifact. The ESMC language
    model is intentionally referenced by id rather than embedded in this config:
    the folding head can consume caller-provided ``lm_hidden_states`` and can also
    run without them for lightweight graph checks.
    """

    architecture: str = "esmfold2"
    model_type: str = "esmfold2"
    type: str = "release"
    dtype: str = "float32"
    initializer_range: float = 0.02

    single_size: int = 384
    pair_size: int = 256
    num_relative_residue_bins: int = 32
    num_relative_chain_bins: int = 2
    num_loops: int = 3
    num_diffusion_samples: int = 32
    disable_msa_features: bool = False
    language_model_dropout: float = 0.0
    language_model_mask_probability: float = 0.0
    force_language_model_dropout_during_inference: bool = False
    language_model_hidden_size: int = 2560
    num_language_model_layers: int = 80
    esmc_id: str = "multimolecule/esmc-6b"
    msa_encoder_overwrite: bool = True

    inputs: ESMFold2InputsEmbedderConfig = ESMFold2InputsEmbedderConfig()
    folding_trunk: ESMFold2FoldingTrunkConfig = ESMFold2FoldingTrunkConfig()
    structure_head: ESMFold2StructureHeadConfig = ESMFold2StructureHeadConfig()
    confidence_head: ESMFold2ConfidenceHeadConfig = ESMFold2ConfidenceHeadConfig()
    msa_encoder: ESMFold2MSAEncoderConfig = ESMFold2MSAEncoderConfig()
    parcae: ESMFold2ParcaeConfig = ESMFold2ParcaeConfig()
    lm_encoder: ESMFold2LMEncoderConfig = ESMFold2LMEncoderConfig()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inputs = _coerce_config(self.inputs, ESMFold2InputsEmbedderConfig)
        self.folding_trunk = _coerce_config(self.folding_trunk, ESMFold2FoldingTrunkConfig)
        self.structure_head = _coerce_config(self.structure_head, ESMFold2StructureHeadConfig)
        self.confidence_head = _coerce_config(self.confidence_head, ESMFold2ConfidenceHeadConfig)
        self.msa_encoder = _coerce_config(self.msa_encoder, ESMFold2MSAEncoderConfig)
        self.parcae = _coerce_config(self.parcae, ESMFold2ParcaeConfig)
        self.lm_encoder = _coerce_config(self.lm_encoder, ESMFold2LMEncoderConfig)
        if self.type != "release":
            raise ValueError(f"OFoldX currently supports only ESMFold2 release configs, got type={self.type!r}")
