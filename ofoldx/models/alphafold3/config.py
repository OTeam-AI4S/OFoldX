r"""
AlphaFold 3 configuration.
"""

from __future__ import annotations

from ..config import StructurePredictionConfig, TemplateConfig
from ..registry import MODEL_CONFIGS


@MODEL_CONFIGS.register("alphafold3")
class AlphaFold3Config(StructurePredictionConfig):
    r"""
    Configuration for [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model].

    Instantiating with the defaults yields the AlphaFold 3 architecture. Inherits every structure-prediction
    knob from [`StructurePredictionConfig`][ofoldx.models.config.StructurePredictionConfig] (token/pair
    widths, trunk, template, denoiser, sampler, confidence, and recycling blocks) and only pins the
    architecture registry key, so the sub-configs it exposes (``trunk``, ``denoiser``, ``sampler``,
    ``confidence``, …) are the primary tuning surface.

    Attributes:
        architecture: Registry key selecting this architecture (``"alphafold3"``).
        single_size: Width of the single (token) representation, inherited default ``384``.
        pair_size: Width of the pair representation, inherited default ``128``.
        recycling_steps: Number of trunk recycling passes, inherited default ``4``.

    Examples:
        >>> config = AlphaFold3Config()
        >>> config.architecture
        'alphafold3'
        >>> config.single_size
        384
        >>> config.pair_size
        128
        >>> from ofoldx import AutoModel  # doctest: +SKIP
        >>> model = AutoModel.from_config(config)  # doctest: +SKIP
    """

    architecture: str = "alphafold3"
    token_feature_size: int = 447
    template: TemplateConfig = TemplateConfig(
        num_blocks=2,
        vocab_size=31,
        triangle_attention_hidden_size=16,
        num_triangle_attention_heads=4,
    )
