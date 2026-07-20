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

# Publication surface (Principle 13). ``standard_tasks=()`` is the correct TERMINAL state, not a
# gate: ODesign is a coordinate-*producing* design family whose ``design`` returns the
# family-owned ``ODesignDesignResult`` (§5.3), which the generic standard ``DesignPipeline`` (the
# coordinate-free shared ``DesignResult``) cannot losslessly carry — exactly as rfdiffusion /
# pxdesign / boltzgen keep ``standard_tasks=()``.
#
# The working protein deliverable IS published, by export of the concrete ``ODesignPipeline``
# (§4.11.6): it generates backbones (S seeds × M diffusion samples) and assigns sequences by
# composing the already-registered ``ofoldx/proteinmpnn`` / ``ofoldx/ligandmpnn`` Pipelines
# read-only, yielding the M×S×K ``ODesignDesignResult``. Protein binder design and existing
# ligand / nucleic-context design all work today. Publication is by EXPORT (importable +
# manually constructible) rather than routed ``register_pipeline`` — mirroring the composition
# families ``BoltzGenPipeline`` / ``PxDesignSuitePipeline`` / ``ProteinMpnnDesignAndPackPipeline``:
# ODesignPipeline is a multi-child composition (a 4-arg
# ``(model, processor, protein_designer, ligand_designer)`` ctor), and the ``api.tasks.pipeline``
# factory builds registered pipelines with only ``(model, processor)`` — so routing it would
# break, exactly as the ``boltz2_affinity`` reservation note forbids for a multi-arg ctor. The
# routed concrete-Pipeline registration is therefore honestly deferred, NOT gated.
#
# Honest remaining CAPABILITY gap: de-novo *ligand / NA* generation. A designed non-core-polymer
# run (odesign ``{"chain_type": "ligand", "sequence": "N-N"}`` = ``num_atoms``) is GAP-1 in
# ``model.from_design_mask`` — it has no ``PlacementValue`` and is rejected, not silently dropped.
#
# The two released protein checkpoints publish as TWO repositories over ONE architecture
# (§4.11.1) — ``ofoldx/odesign-base-prot-flex`` and ``ofoldx/odesign-base-prot-rigid``, whose
# ``model.json`` both name ``ofoldx.odesign`` and whose ``processor.json`` differ only in
# ``receptor_mode`` — NOT two registered arch pairs.
_register_model(
    "ofoldx.odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
    standard_tasks=(),
)
