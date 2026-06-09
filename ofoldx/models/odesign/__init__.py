r"""
ODesign model package.

Re-exports the public ODesign surface: the structure-prediction model and its design-generator wrapper,
their configs, the typed input dataclass, the structure processor, and the constraint-distogram embedder.
Everything listed in ``__all__`` is what downstream code and the model registry import from here.
"""

from __future__ import annotations

from .config import ODesignConfig, ODesignConstraintConfig, ODesignDesignConfig, ODesignPairwiseConfig
from .model import ConstraintTemplateEmbedder, ODesignDesignGenerator, ODesignInput, ODesignModel
from .processor import ODesignProcessor

__all__ = [
    "ConstraintTemplateEmbedder",
    "ODesignConfig",
    "ODesignConstraintConfig",
    "ODesignDesignConfig",
    "ODesignDesignGenerator",
    "ODesignInput",
    "ODesignModel",
    "ODesignPairwiseConfig",
    "ODesignProcessor",
]
