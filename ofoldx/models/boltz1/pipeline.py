r"""
Boltz-1 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_msa, read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import FoldResult
from .model import Boltz1Model, Boltz1Pocket
from .processor import Boltz1Processor


class Boltz1Pipeline(PairedPipeline[Boltz1Model, Boltz1Processor]):
    r"""
    Structure prediction pipeline for Boltz 1.
    """

    model_architecture = "boltz1"

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        pocket: Boltz1Pocket | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular system with Boltz 1.

        Args:
            input: Molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            pocket: Optional binder and pocket-residue specification.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-conformer augmentation and
                coordinate sampling. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and coordinate
                sampling. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            msa_max_depth: Optional rows independently sampled for each MSA-module
                invocation. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading an MSA path.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            use_potentials: Whether to apply configured physical-potential guidance.
            steering_args: Optional potential-guidance controls.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, and PDE values
            are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if isinstance(msa, str | PathLike):
            system = input.system if isinstance(input, Structure) else input
            proteins = [
                entity_id
                for entity_id, entity in system.entities.items()
                if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
            ]
            if len(proteins) != 1:
                raise ValueError("an aligned FASTA is unambiguous only for one protein entity; " "pass MSACollection")
            msa = read_msa(msa, entity_id=proteins[0], max_rows=max_msa_rows)
        batch = self.processor(input, msa=msa, pocket=pocket, generator=generator)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
