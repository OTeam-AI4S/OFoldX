r"""
Configuration classes for Protenix and its four constraint pair-bias sub-embedders.
"""

from __future__ import annotations

from chanfig import Config

from ..config import StructurePredictionConfig
from ..registry import MODEL_CONFIGS


class ProtenixConstraintPocketConfig(Config):
    r"""
    Configuration for the pocket-constraint pair-bias projection.

    Controls the sub-embedder that turns pocket (binder/receptor proximity) constraints into an
    additive pair-representation bias. Mirrors the ``pocket_embedder`` block of Protenix v0.5.

    Attributes:
        enable: Whether the pocket pair-bias projection is instantiated and applied.
        single_input_size: Width of the reference single-channel pocket feature. Kept only to mirror the
            Protenix v0.5 config; this pair-bias embedder does not read it.
        pair_input_size: Number of pocket channels per token pair projected into the pair representation.

    Examples:
        >>> config = ProtenixConstraintPocketConfig()
        >>> config.enable
        True
        >>> config.pair_input_size
        1
    """

    enable: bool = True
    single_input_size: int = 3  # reference-mirror only (Protenix v0.5); unused by this pair-bias embedder
    pair_input_size: int = 1


class ProtenixConstraintContactConfig(Config):
    r"""
    Configuration for the token-contact-constraint pair-bias projection.

    Controls the sub-embedder that turns token-level contact constraints (a min/max distance band
    between two tokens) into an additive pair-representation bias. Mirrors the ``contact_embedder``
    block of Protenix v0.5.

    Attributes:
        enable: Whether the token-contact pair-bias projection is instantiated and applied.
        pair_input_size: Number of contact channels per token pair (min and max distance) projected
            into the pair representation.

    Examples:
        >>> config = ProtenixConstraintContactConfig()
        >>> config.pair_input_size
        2
    """

    enable: bool = True
    pair_input_size: int = 2


class ProtenixConstraintContactAtomConfig(Config):
    r"""
    Configuration for the atom-contact-constraint pair-bias projection.

    Controls the sub-embedder that turns atom-level contact constraints into an additive
    pair-representation bias. Atom-contact constraints operate on tokens produced by expanding
    the constrained standard-residue tokens into atom-level tokens. Mirrors the
    ``contact_atom_embedder`` block of Protenix v0.5.

    Attributes:
        enable: Whether the atom-contact pair-bias projection is instantiated and applied.
        pair_input_size: Number of atom-contact channels per token pair (min and max distance)
            projected into the pair representation.

    Examples:
        >>> config = ProtenixConstraintContactAtomConfig()
        >>> config.pair_input_size
        2
    """

    enable: bool = True
    pair_input_size: int = 2


class ProtenixConstraintSubstructureConfig(Config):
    r"""
    Configuration for the substructure-distance-constraint pair-bias projection.

    Controls the sub-embedder that turns a binned pairwise substructure-distance feature into an
    additive pair-representation bias, either via an MLP or a small transformer. Mirrors the
    ``substructure_embedder`` block of Protenix v0.5.

    Attributes:
        enable: Whether the substructure pair-bias projection is instantiated and applied.
        n_classes: Number of distance-bin classes in the one-hot substructure feature.
        architecture: Sub-embedder architecture, either ``"transformer"`` or ``"mlp"``.
        hidden_size: Width of the sub-embedder hidden channel.
        n_layers: Number of MLP/transformer layers.
        dropout: Dropout probability inside the sub-embedder.

    Examples:
        >>> config = ProtenixConstraintSubstructureConfig()
        >>> config.architecture
        'transformer'
        >>> config.n_classes
        4
    """

    enable: bool = True
    n_classes: int = 4
    architecture: str = "transformer"
    hidden_size: int = 128
    n_layers: int = 1
    dropout: float = 0.1


class ProtenixConstraintEmbedderConfig(Config):
    r"""
    Configuration for the aggregate constraint pair-bias embedder.

    Bundles the four constraint sub-embedders (pocket, token contact, atom contact, substructure) and
    the shared projection width. Mirrors the Protenix v0.5 ``model.constraint_embedder`` block; each
    enabled sub-embedder contributes an additive update to the initial pair representation.

    Attributes:
        pocket_embedder: Pocket-constraint sub-embedder config.
        contact_embedder: Token-contact-constraint sub-embedder config.
        substructure_embedder: Substructure-distance-constraint sub-embedder config.
        contact_atom_embedder: Atom-contact-constraint sub-embedder config.
        constraint_pair_size: Width of the pair channel the sub-embedders project into; must equal the
            model ``pair_size``.
        constraint_single_size: Width of the reference single-channel constraint feature. Kept only to
            mirror the Protenix v0.5 config; unread by this embedder.
        c_constraint_atom_pair: Width of the reference atom-pair constraint channel. Kept only to mirror
            the Protenix v0.5 config; unread by this embedder.
        initialize_method: Weight initialization for the constraint projections, one of ``"zero"``,
            ``"kaiming"``, or ``"default"``.

    Examples:
        >>> config = ProtenixConstraintEmbedderConfig()
        >>> config.constraint_pair_size
        128
        >>> config.initialize_method
        'zero'
    """

    pocket_embedder: ProtenixConstraintPocketConfig = ProtenixConstraintPocketConfig()
    contact_embedder: ProtenixConstraintContactConfig = ProtenixConstraintContactConfig()
    substructure_embedder: ProtenixConstraintSubstructureConfig = ProtenixConstraintSubstructureConfig()
    contact_atom_embedder: ProtenixConstraintContactAtomConfig = ProtenixConstraintContactAtomConfig()
    constraint_pair_size: int = 128
    constraint_single_size: int = 449  # reference-mirror only (Protenix v0.5); unread by this embedder
    c_constraint_atom_pair: int = 16  # reference-mirror only (Protenix v0.5); unread by this embedder
    initialize_method: str = "zero"


@MODEL_CONFIGS.register("protenix")
class ProtenixConfig(StructurePredictionConfig):
    r"""
    Configuration for the Protenix base structure-prediction model.

    Extends [`StructurePredictionConfig`][ofoldx.config.StructurePredictionConfig] with the
    optional constraint pair-bias embedder and the Protenix recycling depth. Defaults target
    ``protenix_base_default_v0.5.0``: base AF3-shaped trunk and sampler, ten recycling cycles, no
    template stack, and zero-initialized pocket/contact/contact-atom/substructure pair-bias embedders.

    Attributes:
        architecture: Registry key selecting this architecture (`"protenix"`).
        recycling_steps: Number of trunk recycling iterations run before the diffusion sampler.
        constraint_embedder: Configuration of the aggregate constraint pair-bias embedder.

    Examples:
        >>> config = ProtenixConfig()
        >>> config.architecture
        'protenix'
        >>> config.recycling_steps
        10
        >>> from ofoldx import AutoModel  # doctest: +SKIP
        >>> model = AutoModel.from_config(config)  # doctest: +SKIP
    """

    architecture: str = "protenix"
    recycling_steps: int = 10
    constraint_embedder: ProtenixConstraintEmbedderConfig = ProtenixConstraintEmbedderConfig()
