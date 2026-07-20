r"""Protenix's biological operation boundary for constraint-conditioned folding.

The default Protenix v0.5 checkpoint routes through the generic
:class:`~ofoldx.pipelines.fold.FoldPipeline`: its ``fold`` verb takes only
``msa``/``templates`` and the standard sampling controls, which the generic
Pipeline forwards faithfully. The constraint checkpoint (``ofoldx.protenix_constraint``)
adds the released v0.5.3 constraint channels — ``atom_contacts`` / ``token_contacts`` /
``pocket`` — which live on :meth:`ProtenixConstraintProcessor.__call__`. The generic
``FoldPipeline.fold`` signature cannot accept or forward those keywords, so the whole
reason the constraint arch id exists would be unreachable through ``ofoldx.fold``.

:class:`ProtenixConstraintPipeline` binds the constraint Model+Processor pair and owns a
``fold`` verb whose signature ADDS the three constraint keywords and forwards them to the
Processor, mirroring the generic ``FoldPipeline.fold`` body for the msa/templates/generator
handling. Marker id ``"ofoldx.protenix_constraint"``: a checkpoint shipping
``pipeline.json = {"pipeline": "ofoldx.protenix_constraint"}`` routes ``fold`` here.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, inputs_device, resolve_generator
from ...results import FoldResult
from .model import ProtenixAtomContact, ProtenixConstraintModel, ProtenixPocket, ProtenixTokenContact
from .processor import ProtenixConstraintProcessor


class ProtenixConstraintPipeline(PairedPipeline):
    r"""Bind the Protenix constraint Model--Processor pair and own the constraint ``fold`` verb."""

    model: ProtenixConstraintModel
    processor: ProtenixConstraintProcessor

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> FoldResult:
        r"""Lower one input plus its v0.5.3 constraints and run released Protenix constraint folding.

        Forwards the released ``atom_contacts`` / ``token_contacts`` / ``pocket`` constraint channels
        to :meth:`ProtenixConstraintProcessor.__call__`, then runs
        :meth:`ProtenixConstraintModel.fold`. ``msa``/``templates``/``seed``/``generator`` are handled
        exactly as the generic :meth:`~ofoldx.pipelines.fold.FoldPipeline.fold`.
        """

        processor_arguments: dict[str, object] = {
            "atom_contacts": atom_contacts,
            "token_contacts": token_contacts,
            "pocket": pocket,
        }
        if msa is not None:
            processor_arguments["msa"] = msa
        if templates:
            processor_arguments["templates"] = templates
        inputs = self.processor(input, **processor_arguments)  # type: ignore[operator]
        generator = resolve_generator(seed, generator, device=inputs_device(inputs))
        return self.model.fold(  # type: ignore[attr-defined]
            inputs,
            num_samples=num_samples,
            generator=generator,
        )
