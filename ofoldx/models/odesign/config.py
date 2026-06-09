r"""
ODesign model configuration classes.

Defines the ODesign structure-prediction config and its design-generator variant, together with the two
ODesign-specific sub-config blocks (pairwise design heads and the constraint-distogram embedder). The two
top-level configs are registered under their architecture keys in [`MODEL_CONFIGS`][ofoldx.models.registry].
"""

from __future__ import annotations

from chanfig import Config

from ..config import StructurePredictionConfig
from ..registry import MODEL_CONFIGS


class ODesignPairwiseConfig(Config):
    r"""
    Configuration for ODesign's pairwise (token-pair) design heads.

    Controls the token bond-type reconstruction head that predicts per-token-pair covalent bond categories
    from the pair representation.

    Attributes:
        bond_reconstruction: Whether to build the token bond-type head. When ``False`` no pairwise design head
            is instantiated (default ``True``).
        num_bond_types: Number of bond-type classes the head predicts (default ``2``).
        hidden_size: Width of the head's hidden classifier layer (default ``128``).

    Examples:
        >>> config = ODesignPairwiseConfig()
        >>> config.bond_reconstruction
        True
        >>> config.num_bond_types
        2
    """

    bond_reconstruction: bool = True
    num_bond_types: int = 2
    hidden_size: int = 128


class ODesignConstraintConfig(Config):
    r"""
    Configuration for ODesign's constraint-distogram embedder.

    Mirrors upstream ``constraint_distogram_embedder`` (Algorithm 16 template path). When ``enabled`` the
    model builds a [`ConstraintTemplateEmbedder`][ofoldx.models.odesign.model.ConstraintTemplateEmbedder]
    and folds any ``constraint_distogram`` passed on the input into the pair representation each recycling
    cycle. Disabled by default so the base parameter count is unchanged; set ``enabled=True`` to condition
    generation on distance constraints.

    Attributes:
        enabled: Whether to build the constraint embedder and consume ``constraint_distogram`` (default
            ``False``, leaving the base model unchanged).
        num_blocks: Number of pair-only Pairformer blocks in the embedder (default ``2``).
        hidden_size: Width of the embedder's internal pair channel (default ``64``).
        num_bins: Number of distance bins the raw constraint distances are one-hot encoded into (default
            ``39``).
        min_distance: Lower edge of the binned distance range, in angstroms (default ``3.25``).
        max_distance: Upper edge of the binned distance range, in angstroms (default ``50.75``).
        dropout: Dropout probability inside the embedder's Pairformer stack (default ``0.25``).

    Examples:
        >>> config = ODesignConstraintConfig()
        >>> config.enabled
        False
        >>> config.num_bins
        39
    """

    enabled: bool = False
    num_blocks: int = 2
    hidden_size: int = 64
    num_bins: int = 39
    min_distance: float = 3.25
    max_distance: float = 50.75
    dropout: float = 0.25


@MODEL_CONFIGS.register("odesign")
class ODesignConfig(StructurePredictionConfig):
    r"""
    Configuration for the ODesign structure-prediction model.

    ODesign keeps the structure-prediction trunk and diffusion geometry, but uses the ODesign token schema:
    restype(35) + profile(32) + deletion_mean(1) + is_hotspot_residue(1), giving 453 token channels. It
    inherits every shared trunk/diffusion/confidence knob from
    [`StructurePredictionConfig`][ofoldx.models.config.StructurePredictionConfig] and overrides only the
    ODesign-specific token/atom feature widths, recycling depth, and design heads.

    Attributes:
        architecture: Registry key selecting this architecture (`"odesign"`).
        token_feature_size: Width of the ODesign per-token input feature vector (default ``453``).
        atom_feature_size: Width of the per-atom input feature vector (default ``386``).
        recycling_steps: Number of recycling passes through the trunk (default ``10``).
        num_model_seeds: Number of independent model seeds ensembled at inference (default ``1``).
        pairwise: Configuration of the pairwise design heads.
        constraint: Configuration of the constraint-distogram embedder.

    Examples:
        >>> config = ODesignConfig()
        >>> config.architecture
        'odesign'
        >>> config.token_feature_size
        453
        >>> from ofoldx.models.odesign.model import ODesignModel
        >>> model = ODesignModel(config)  # doctest: +SKIP
    """

    architecture: str = "odesign"
    token_feature_size: int = 453
    atom_feature_size: int = 386
    recycling_steps: int = 10
    num_model_seeds: int = 1
    pairwise: ODesignPairwiseConfig = ODesignPairwiseConfig()
    constraint: ODesignConstraintConfig = ODesignConstraintConfig()


@MODEL_CONFIGS.register("odesign_design")
class ODesignDesignConfig(ODesignConfig):
    r"""
    Configuration for the ODesign design-generator variant.

    This design entry point currently shares the ODesign structure trunk and heads, returning generated
    structure predictions through the design-generator contract. Sequence generation can be added behind this
    distinct architecture without changing the structure-prediction ``odesign`` registration. It inherits
    every field from [`ODesignConfig`][ofoldx.models.odesign.config.ODesignConfig] and only overrides the
    registry key.

    Attributes:
        architecture: Registry key selecting this architecture (`"odesign_design"`).

    Examples:
        >>> config = ODesignDesignConfig()
        >>> config.architecture
        'odesign_design'
        >>> config.token_feature_size
        453
        >>> from ofoldx.models.odesign.model import ODesignDesignGenerator
        >>> model = ODesignDesignGenerator(config)  # doctest: +SKIP
    """

    architecture: str = "odesign_design"
