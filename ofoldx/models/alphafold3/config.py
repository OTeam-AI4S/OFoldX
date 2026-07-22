r"""
AlphaFold 3 configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..config import (
    ConfidenceConfig,
    DenoiserConfig,
    DistogramConfig,
    SamplerConfig,
    TemplateConfig,
    TrunkConfig,
)


class AlphaFold3Config(Config):
    r"""
    Configuration for [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model].

    Instantiating with the defaults yields the AlphaFold 3 architecture. The config owns the model's
    token/pair widths, trunk, template, denoiser, sampler, confidence, and recycling settings directly.

    Attributes:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        recycling_steps: Number of trunk recycling passes.
        rank_metric_eps: Denominator epsilon for the contact-weighted PAE ranking metric.

    Examples:
        >>> config = AlphaFold3Config()
        >>> config.single_size
        384
        >>> config.pair_size
        128
    """

    token_feature_size: int = 447
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    template: TemplateConfig = TemplateConfig(
        num_blocks=2,
        vocab_size=31,
        triangle_attention_hidden_size=16,
        num_triangle_attention_heads=4,
    )
    trunk: TrunkConfig = TrunkConfig()
    recycling_steps: int = 4
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    distogram: DistogramConfig = DistogramConfig()
    # DeepMind AlphaFold 3's confidence head consumes the trunk ``single`` representation
    # unclamped (``confidence_head.py`` casts ``embeddings['single']`` straight into the
    # pairformer with no ``clip``); the AF3 trunk single legitimately spans |x| up to ~4e4, so
    # the shared ``single_max_abs=512`` clamp would corrupt ~20% of its entries and drive the
    # pLDDT / resolved logits far off (cosine ~0.36).  Disable the clamp for the AF3 identity.
    confidence: ConfidenceConfig = ConfidenceConfig(single_max_abs=None)
    rank_metric_eps: float = 1e-6


class AlphaFold3ProcessorConfig(Config):
    r"""Select the input contract paired with a released AlphaFold 3 checkpoint.

    ``max_msa_rows`` bounds the assembled MSA depth the Processor lowers into the
    ``msa_features`` / ``msa_mask`` tensors; the released data pipeline caps the
    unpaired-plus-paired stack at 16384 rows.

    ``ccd_revision`` is the exact wwPDB snapshot date that pins the shared
    ``ofoldx/ccd`` Hub dataset revision passed to
    :func:`~ofoldx.data.ccd.resolve_ccd_component` when the Processor lowers a
    bare CCD-code ligand / ion / modified residue / glycan monomer into embedded
    chemistry. It is a provenance pin on the resolution *source*, not a runtime
    mode: two Processors with different ``ccd_revision`` values may lower the same
    code to different (reproducible, auditable) chemistry. The released default is
    the checkpoint's training CCD snapshot.
    """

    max_msa_rows: int = 16384
    ccd_revision: str = "2022-09-28"
