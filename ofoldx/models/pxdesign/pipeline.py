r"""PXDesign's biological operation boundary.

:class:`PxDesignPipeline` binds the native PXDesign-d generator Model+Processor and owns
the ``design`` verb over the PXDesign topology argument (``PlacementSpec``) and
per-residue target label (``Hotspot``), returning :class:`PxDesignResult`. It mirrors
:class:`~ofoldx.models.proteinmpnn.pipeline.ProteinMpnnPipeline` and diverges from the shared
coordinate-free :class:`~ofoldx.pipelines.design.DesignPipeline` (which lowers
``AllowedResidueIdentities`` / ``TiedResidueIdentities`` conditions and returns the
coordinate-free ``DesignResult``) because PXDesign is coordinate-producing.

The suite composition (backbone -> ProteinMPNN sequence -> AF2-IG / Protenix evaluation ->
``pxdbench`` ranking) is not implemented here. Its AF2 initial-guess entry point and
``pxdbench`` ranking, geometry, and RMSD metrics are not yet available in OFoldX.
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


class PxDesignPipeline(PairedPipeline[PxDesignModel, PxDesignProcessor]):
    r"""Bind one PXDesign-d Model--Processor pair and own the ``design`` verb."""

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
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

        batch = self.processor(input, placement=placement, conditions=conditions)
        generator = resolve_generator(seed, generator, device=batch.reference_positions.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
