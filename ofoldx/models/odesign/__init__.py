r"""OFoldX implementation of the released ODesign core architecture."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
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

# Publication surface (Principle 13). ``standard_tasks=()`` is the correct TERMINAL state, not a
# gate: ODesign is a coordinate-*producing* design family whose ``design`` returns the
# family-owned ``ODesignDesignResult`` (§5.3), which the generic standard ``DesignPipeline`` (the
# coordinate-free shared ``DesignResult``) cannot losslessly carry — exactly as rfdiffusion /
# pxdesign / boltzgen keep ``standard_tasks=()``.
#
# The working protein deliverable IS published, by the concrete ``ODesignPipeline`` (§4.11.6): it
# generates backbones (S seeds × M diffusion samples) and assigns sequences by composing the
# already-registered ``ofoldx/proteinmpnn`` / ``ofoldx/ligandmpnn`` Pipelines read-only, yielding
# the M×S×K ``ODesignDesignResult``. Protein binder design and existing ligand / nucleic-context
# design all work today. It is a DESIGN §7.4 multi-pair Pipeline: the ODesign backbone pair owns the
# runnable root and the two read-only sequence designers live at the ``protein_designer/`` /
# ``ligand_designer/`` sub-directories (declared in its ``_PAIR_LAYOUT``). It is registered below,
# so a marked repository's ``design`` routes through the factory's multi-pair loader
# (``from_repository``) instead of the one-pair ``(model, processor)`` call.
#
# Honest remaining CAPABILITY gap: de-novo *ligand / NA* generation. A designed non-core-polymer
# run (odesign ``{"chain_type": "ligand", "sequence": "N-N"}`` = ``num_atoms``) is GAP-1 in
# ``model.from_design_mask`` — it has no ``PlacementValue`` and is rejected, not silently dropped.
#
# The two released protein checkpoints publish as TWO repositories over ONE architecture
# (§4.11.1) — ``ofoldx/odesign-base-prot-flex`` and ``ofoldx/odesign-base-prot-rigid``, whose
# ``model.json`` both name ``ofoldx.odesign`` and whose ``processor.json`` differ only in
# ``receptor_mode`` — NOT two registered arch pairs.
MODELS.register(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.odesign",
    pipeline=ODesignPipeline,
)
