r"""
Protenix structure-prediction pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import (
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_polymer_input
from ...results import FoldResult
from .model import (
    ProtenixAtomContact,
    ProtenixConstraintModel,
    ProtenixModel,
    ProtenixPocket,
    ProtenixStructureConditionModel,
    ProtenixTemplateModel,
    ProtenixTfgGuidance,
    ProtenixTokenContact,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
    molecular_system_from_named_sequences,
)


class ProtenixPipeline(PairedPipeline[ProtenixModel, ProtenixProcessor]):
    r"""
    Structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with Protenix.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, PDE, and gPDE
            values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


def _canonical_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""Apply Protenix chemistry after shared input classification."""

    resolved = resolve_polymer_input(input)
    return (
        resolved
        if isinstance(resolved, MolecularSystem | Structure)
        else molecular_system_from_named_sequences(resolved)
    )


class ProtenixTemplatePipeline(PairedPipeline[ProtenixTemplateModel, ProtenixTemplateProcessor]):
    r"""
    Template- and RNA-MSA-capable structure prediction pipeline for Protenix v1 and v2.
    """

    model_architecture = "protenix_template"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures with the released Protenix v1/v2 template and RNA-MSA path.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional protein/RNA MSA value, collection, or alignment path.
            templates: Protein templates aligned to target chains.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and diffusion.
                Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations.
            num_sampling_steps: Number of diffusion steps.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN, ENTITY_RNA}),
            templates=templates,
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


class ProtenixStructureConditionPipeline(
    PairedPipeline[ProtenixStructureConditionModel, ProtenixStructureConditionProcessor]
):
    r"""
    Structure-conditioned prediction pipeline for Protenix mini template.
    """

    model_architecture = "protenix_structure_condition"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        structure_condition: Structure | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures using an explicit in-memory coordinate condition.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence
                collection, molecular system, or structure topology.
            structure_condition: Optional in-memory Structure aligned by exact
                atom occurrence to the input assembly.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations.
            num_sampling_steps: Number of diffusion transitions.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Optional training-free chemical guidance.

        Returns:
            Generation-ordered structure predictions.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            structure_condition=structure_condition,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


class ProtenixConstraintPipeline(PairedPipeline[ProtenixConstraintModel, ProtenixConstraintProcessor]):
    r"""
    Constrained structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix_constraint"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict constrained structures for one molecular assembly with Protenix.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            substructure: Optional resolved token-center coordinates used for the
                released four-channel substructure condition.
            substructure_coordinate_noise_scale: Standard deviation of Gaussian
                noise added to selected center coordinates before distance binning.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by constraint preprocessing and
                folding. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, PDE, and gPDE
            values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
