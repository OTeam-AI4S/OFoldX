r"""Protenix folding Pipelines.

The default and constraint variants expose separate task boundaries. The constraint
Pipeline additionally accepts atom contacts, token contacts, and pocket inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
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
    r"""Bind the default Protenix Model and Processor."""

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
        r"""Lower one assembly and run Protenix folding.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            templates: Reserved template evidence; this architecture requires an empty sequence.
            num_samples: Number of structure predictions. ``None`` uses the Model configuration.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            A ``FoldResult`` whose predictions follow generation order.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(
            input,
            msa=msa,
            templates=templates,
            generator=generator,
        )
        return self.model.fold(
            model_input,
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
        r"""Lower one assembly and run constraint-conditioned Protenix folding.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            msa: Optional MSA value, collection, or alignment path.
            templates: Reserved template evidence; this architecture requires an empty sequence.
            num_samples: Number of structure predictions. ``None`` uses the Model configuration.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional RNG shared by preprocessing and folding; mutually exclusive with
                ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            A ``FoldResult`` whose predictions follow generation order.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(
            input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            msa=msa,
            templates=templates,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
