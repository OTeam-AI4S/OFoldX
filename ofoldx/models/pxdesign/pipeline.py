r"""PXDesign's biological operation boundary.

:class:`PxDesignPipeline` binds the native PXDesign-d generator Model+Processor and owns
the ``design`` verb over PXDesign's OWN topology arg (``PlacementSpec``) and per-residue target
label (``Hotspot``), returning the family-owned :class:`PxDesignResult`. It mirrors
:class:`~ofoldx.models.proteinmpnn.pipeline.ProteinMpnnPipeline` and diverges from the shared
coordinate-free :class:`~ofoldx.pipelines.design.DesignPipeline` (which lowers
``AllowedResidueIdentities`` / ``TiedResidueIdentities`` conditions and returns the frozen
coordinate-free ``DesignResult``) because PXDesign is coordinate-producing.

The released *suite composition* pipeline (backbone -> ProteinMPNN sequence -> AF2-IG /
Protenix evaluation -> ``pxdbench`` ranking) is NOT published here: it is dependency-gated on
the AF2 initial-guess (AF2-IG) fold entry point, which does not yet exist in the OFoldX
AlphaFold2 family, and on the un-vendored ``pxdbench`` (PXDesignBench) ranking / geometry /
RMSD metrics. Per Principle 13 (public reexport IS a publication surface) it stays absent from
this module until those dependencies land, rather than shipping a placeholder that only raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data import PlacementSpec
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import Hotspot, PxDesignModel, PxDesignResult
from .processor import PxDesignProcessor


class PxDesignPipeline(PairedPipeline):
    r"""Bind one PXDesign-d Model--Processor pair and own the ``design`` verb."""

    model: PxDesignModel
    processor: PxDesignProcessor

    def __init__(
        self,
        *,
        model: PxDesignModel,
        processor: PxDesignProcessor,
    ) -> None:
        super().__init__(model=model, processor=processor)

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        conditions: Sequence[PlacementSpec | Hotspot] = (),
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignResult:
        r"""Generate ranked target--binder backbone candidates for one released target.

        The Processor lowers ``input`` plus PXDesign's own ``PlacementSpec`` topology and optional
        ``Hotspot`` label to a :class:`~ofoldx.models.pxdesign.model.PxDesignInput`; the
        Model runs the released EDM sampler and decodes each coordinate sample. The returned
        :class:`PxDesignResult` carries raw generator backbones whose ``scores`` are empty
        (the released sampler emits no per-sample confidence); the suite Pipeline populates
        the evaluation Scores and ranking.
        """

        batch = self.processor(input, conditions=conditions)
        generator = resolve_generator(seed, generator, device=batch.reference_positions.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
