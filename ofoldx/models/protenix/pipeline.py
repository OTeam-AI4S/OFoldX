r"""Protenix's biological operation boundary for constraint-conditioned folding.

The default and constraint checkpoints each publish a concrete Pipeline so recycling and
diffusion sampling controls are not erased by the common fold surface. The constraint
Pipeline additionally forwards the released atom-contact, token-contact, and pocket inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, processor_device, resolve_generator
from ...results import FoldResult
from .model import (
    ProtenixAtomContact,
    ProtenixConstraintModel,
    ProtenixModel,
    ProtenixPocket,
    ProtenixTokenContact,
)
from .processor import ProtenixConstraintProcessor, ProtenixProcessor


class ProtenixPipeline(PairedPipeline[ProtenixModel, ProtenixProcessor]):
    r"""Bind the default Protenix v0.5 Model and Processor."""

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""Lower one input and run released Protenix folding."""

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(
            input,
            msa=msa,
            templates=templates,
            generator=generator,
        )
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )


class ProtenixConstraintPipeline(PairedPipeline[ProtenixConstraintModel, ProtenixConstraintProcessor]):
    r"""Bind the Protenix constraint Model--Processor pair and own the constraint ``fold`` verb."""

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""Lower one input plus its v0.5.3 constraints and run released Protenix constraint folding.

        Forwards the released ``atom_contacts`` / ``token_contacts`` / ``pocket`` constraint channels
        to :meth:`ProtenixConstraintProcessor.__call__`, then runs
        :meth:`ProtenixConstraintModel.fold`. ``seed`` / ``generator`` resolve one stream which is
        passed directly through the stochastic Processor and Model; ``msa`` / ``templates`` are forwarded.
        """

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(
            input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            msa=msa,
            templates=templates,
            generator=generator,
        )
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
