r"""OFoldX implementation of the released ODesign core architecture."""

from __future__ import annotations

from ..registry import register_model as _register_model
from .config import (
    ODesignConfig,
    ODesignConstraintConfig,
    ODesignDenoiserConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ConstraintDistogramEmbedder,
    InputEmbedder,
    ODesignDesignCandidate,
    ODesignDesignResult,
    ODesignHotspot,
    ODesignInput,
    ODesignLayout,
    ODesignModel,
    ODesignOutput,
    TokenBondTypeHead,
)
from .pipeline import ODesignPipeline
from .processor import ODesignProcessor

# §4.11.9 registration: ODesign stays gated (``standard_tasks=()`` and no registered Pipeline)
# while de-novo ligand/NA generation remains a quality blocker (README). When the gate opens,
# flip ``standard_tasks`` to ``("design",)`` and register the concrete ``ODesignPipeline`` for
# the ``design`` verb. The two released protein checkpoints publish as TWO repositories over
# ONE architecture (§4.11.1) — ``ofoldx/odesign-base-prot-flex`` and
# ``ofoldx/odesign-base-prot-rigid``, whose ``model.json`` both name ``ofoldx.odesign`` and
# whose ``processor.json`` differ only in ``receptor_mode`` — NOT two registered arch pairs.
_register_model(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
    standard_tasks=(),
)
