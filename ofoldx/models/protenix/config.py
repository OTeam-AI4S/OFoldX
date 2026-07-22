r"""Configuration for the two released Protenix v0.5 base graphs."""

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


class ProtenixConfig(Config):
    r"""Configuration of the released Protenix base graph.

    ``constraints=False`` selects the default v0.5.0 graph. Setting it to
    ``True`` selects the v0.5.3 graph containing all four released constraint
    embedders. The four branches are one checkpoint variant, not independently
    configurable features.
    """

    token_feature_size: int = 449
    single_size: int = 384
    pair_size: int = 128
    atom_feature_size: int = 385
    relative_position_size: int = 139
    distogram: DistogramConfig = DistogramConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    trunk: TrunkConfig = TrunkConfig()
    template: TemplateConfig = TemplateConfig()
    denoiser: DenoiserConfig = DenoiserConfig()
    sampler: SamplerConfig = SamplerConfig()
    recycling_steps: int = 10
    constraints: bool = False


class ProtenixConstraintConfig(ProtenixConfig):
    r"""Configuration of the released ``protenix_base_constraint_v0.5.0`` graph.

    Flips ``constraints`` to ``True`` so the four-branch constraint pair-bias
    embedder is built. Every other field is inherited unchanged from the
    base graph; the constraint checkpoint always carries all four
    embedders (they were trained jointly), so this is one checkpoint-variant
    selector, not four independently toggleable features.
    """

    constraints: bool = True


class ProtenixProcessorConfig(Config):
    r"""Input contract of the default v0.5.0 Processor.

    ``constraints`` is a record-only marker of which checkpoint contract this
    config pairs with; it does NOT gate keyword acceptance. The accept/reject
    split between the default and constraint Processors is carried entirely by the
    two distinct ``__call__`` signatures: the default
    :class:`~ofoldx.models.protenix.processor.ProtenixProcessor` signature omits the
    constraint keywords, so passing them raises ``TypeError`` at the call boundary.
    """

    constraints: bool = False
    max_msa_rows: int = 16384
    augment_reference_positions: bool = True


class ProtenixConstraintProcessorConfig(ProtenixProcessorConfig):
    r"""Input contract of the v0.5.3 constraint Processor.

    Only the paired
    :class:`~ofoldx.models.protenix.processor.ProtenixConstraintProcessor`
    signature declares the atom-contact, token-contact, and pocket keywords; the
    ``constraints`` marker below merely records that this config selects the
    constraint checkpoint contract.
    """

    constraints: bool = True
