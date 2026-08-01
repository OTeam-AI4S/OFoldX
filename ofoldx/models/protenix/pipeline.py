r"""
Protenix structure-prediction pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import cast

import torch

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import (
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import (
    resolve_msa_input,
    resolve_polymer_input,
    resolve_structure_input,
    resolve_template_input,
)
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .model import (
    ProtenixAtomContact,
    ProtenixConstraintModel,
    ProtenixInput,
    ProtenixModel,
    ProtenixPocket,
    ProtenixSamplingOutput,
    ProtenixStructureConditionModel,
    ProtenixTemplateModel,
    ProtenixTfgGuidance,
    ProtenixTokenContact,
    protenix_confidence_summary,
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

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Prepare one molecular assembly for the Protenix sampling graph."""

        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )

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
        model_input = self.prepare_fold(
            input,
            msa=msa,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        output = self.model.sample(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
        return self.decode_fold(model_input, output)

    def decode_fold(self, input: ProtenixInput, output: ProtenixSamplingOutput) -> FoldResult:
        r"""Decode raw Protenix tensors without executing the Model."""

        return _decode_fold_result(input, output)


class ProtenixStructureConditionPipeline(
    PairedPipeline[ProtenixStructureConditionModel, ProtenixStructureConditionProcessor]
):
    r"""
    Structure-conditioned prediction pipeline for Protenix mini template.
    """

    model_architecture = "protenix_structure_condition"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        structure_condition: Structure | str | PathLike[str] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Prepare one structure-conditioned Protenix sampling input."""

        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            structure_condition=cast(Structure | None, resolve_structure_input(structure_condition)),
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        structure_condition: Structure | str | PathLike[str] | None = None,
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
        model_input = self.prepare_fold(
            input,
            structure_condition=structure_condition,
            msa=msa,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        output = self.model.sample(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
        return self.decode_fold(model_input, output)

    def decode_fold(self, input: ProtenixInput, output: ProtenixSamplingOutput) -> FoldResult:
        r"""Decode raw structure-conditioned Protenix tensors."""

        return _decode_fold_result(input, output)


class ProtenixConstraintPipeline(PairedPipeline[ProtenixConstraintModel, ProtenixConstraintProcessor]):
    r"""
    Constrained structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix_constraint"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | str | PathLike[str] | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Prepare one constrained Protenix sampling input."""

        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=cast(Structure | None, resolve_structure_input(substructure)),
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN}),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | str | PathLike[str] | None = None,
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
        model_input = self.prepare_fold(
            input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            msa=msa,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        output = self.model.sample(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
        return self.decode_fold(model_input, output)

    def decode_fold(self, input: ProtenixInput, output: ProtenixSamplingOutput) -> FoldResult:
        r"""Decode raw constrained Protenix tensors."""

        return _decode_fold_result(input, output)


class ProtenixTemplatePipeline(PairedPipeline[ProtenixTemplateModel, ProtenixTemplateProcessor]):
    r"""
    Template- and RNA-MSA-capable structure prediction pipeline for Protenix v1 and v2.
    """

    model_architecture = "protenix_template"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Prepare one template-aware Protenix sampling input."""

        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN, ENTITY_RNA}),
            templates=resolve_template_input(templates),
            prepare_tfg_features=tfg_guidance is not None,
            generator=generator,
        )

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
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
        model_input = self.prepare_fold(
            input,
            msa=msa,
            templates=templates,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        output = self.model.sample(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )
        return self.decode_fold(model_input, output)

    def decode_fold(self, input: ProtenixInput, output: ProtenixSamplingOutput) -> FoldResult:
        r"""Decode raw template-aware Protenix tensors."""

        return _decode_fold_result(input, output)


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


def _decode_fold_result(input: ProtenixInput, output: ProtenixSamplingOutput) -> FoldResult:
    r"""Map raw Protenix axes and confidence tensors onto public result objects."""

    if input.token_mask.shape[0] != 1:
        raise ValueError("Protenix Pipeline decoding accepts one molecular assembly per call")
    num_samples = output.coordinates.shape[-3]
    atom_count = int(input.atom_mask[0].sum())
    token_count = int(input.token_mask[0].sum())
    canonical_atom_index = input.atom_index[0, :atom_count].detach().cpu().long()
    canonical_token_component = input.token_component_index[0, :token_count].detach().cpu().long()
    canonical_token_atom = input.token_atom_index[0, :token_count].detach().cpu().long()
    result_system = input.system.cpu()  # type: ignore[attr-defined]
    atom_axis = tuple(result_system.atom_site(int(index)) for index in canonical_atom_index)
    token_axis_values: list[ComponentReference | AtomSite] = []
    for atom_index, component_index in zip(
        canonical_token_atom.tolist(),
        canonical_token_component.tolist(),
    ):
        if atom_index >= 0:
            token_axis_values.append(result_system.atom_site(atom_index))
        else:
            token_axis_values.append(result_system.component_reference(component_index))
    token_axis = tuple(token_axis_values)

    coordinates = (
        output.coordinates[0, :, :atom_count]
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )
    confidence = protenix_confidence_summary(output, input)
    atom_plddt = confidence["plddt"]
    pae = confidence["pae"]
    pde = confidence["pde"]
    resolved_probability = confidence["resolved_probability"]
    contact_probability = confidence["contact_probability"]
    ptm = confidence["ptm"]
    iptm = confidence["iptm"]
    chain_ptm = confidence["chain_ptm"]
    chain_iptm = confidence["chain_iptm"]
    chain_pair_iptm = confidence["chain_pair_iptm"]
    chain_pair_iptm_global = confidence["chain_pair_iptm_global"]
    gpde = confidence["gpde"]
    chain_gpde = confidence["chain_gpde"]
    chain_pair_gpde = confidence["chain_pair_gpde"]
    chain_plddt = confidence["chain_plddt"]
    chain_pair_plddt = confidence["chain_pair_plddt"]
    has_clash = confidence["has_clash"]
    disorder = confidence["disorder"]
    ranking_score = confidence["ranking_score"]

    component_instance = torch.repeat_interleave(
        torch.arange(
            input.system.instance_symbol.numel(),
            device=input.system.instance_symbol.device,
        ),
        input.system.instance_component_ptr[1:] - input.system.instance_component_ptr[:-1],
    )
    contact_numpy = contact_probability.detach().float().cpu().numpy()
    component_instance_cpu = component_instance.detach().cpu()
    token_chain_cpu = input.chain_index[0, :token_count].detach().cpu().long()
    chain_ids: list[str] = []
    for chain in torch.unique(token_chain_cpu, sorted=True):
        first_token = int((token_chain_cpu == chain).nonzero(as_tuple=False)[0, 0])
        component_index = int(canonical_token_component[first_token])
        instance_index = int(component_instance_cpu[component_index])
        chain_ids.append(result_system.instance_id(instance_index))
    chain_axis = tuple(chain_ids)

    complete_atom_count = result_system.atom_symbol.numel()
    complete_component_count = result_system.component_symbol.numel()
    predictions: list[FoldPrediction] = []
    for sample_index in range(num_samples):
        complete_coordinates = torch.zeros(complete_atom_count, 3, dtype=torch.float32)
        coordinate_mask = torch.zeros(complete_atom_count, dtype=torch.bool)
        complete_coordinates[canonical_atom_index] = coordinates[sample_index]
        coordinate_mask[canonical_atom_index] = True
        structure = Structure(  # type: ignore[call-arg]
            system=result_system,
            coordinates=complete_coordinates,
            coordinate_mask=coordinate_mask,
            occupancy=torch.zeros(complete_atom_count),
            occupancy_mask=torch.zeros(complete_atom_count, dtype=torch.bool),
            b_factor=torch.zeros(complete_atom_count),
            b_factor_mask=torch.zeros(complete_atom_count, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((complete_component_count,), -1, dtype=torch.int64),
            insertion_code=torch.full((complete_component_count,), -1, dtype=torch.int64),
            batch_size=[],
        )
        sample_plddt = atom_plddt[sample_index].detach().float().cpu().numpy()
        sample_pae = pae[sample_index].detach().float().cpu().numpy()
        sample_pde = pde[sample_index].detach().float().cpu().numpy()
        sample_resolved_probability = resolved_probability[sample_index].detach().float().cpu().numpy()
        predictions.append(
            FoldPrediction(
                structure=structure,
                scores={
                    "mean_plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
                    "gpde": Score(float(gpde[sample_index]), unit="angstrom", preferred="lower"),
                    "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                    "iptm": Score(float(iptm[sample_index]), preferred="higher"),
                    "has_clash": Score(float(has_clash[sample_index]), preferred="lower"),
                    "disorder": Score(float(disorder[sample_index]), preferred="higher"),
                    "ranking_score": Score(float(ranking_score[sample_index]), preferred="higher"),
                },
                metrics={
                    "plddt": Metric(values=sample_plddt, axis=atom_axis),
                    "resolved_probability": Metric(values=sample_resolved_probability, axis=atom_axis),
                    "chain_plddt": Metric(
                        values=chain_plddt[sample_index].detach().float().cpu().numpy(),
                        axis=chain_axis,
                    ),
                    "chain_ptm": Metric(
                        values=chain_ptm[sample_index].detach().float().cpu().numpy(),
                        axis=chain_axis,
                    ),
                    "chain_iptm": Metric(
                        values=chain_iptm[sample_index].detach().float().cpu().numpy(),
                        axis=chain_axis,
                    ),
                    "chain_gpde": Metric(
                        values=chain_gpde[sample_index].detach().float().cpu().numpy(),
                        axis=chain_axis,
                        unit="angstrom",
                    ),
                },
                pair_metrics={
                    "pae": PairMetric(
                        values=sample_pae,
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                    "pde": PairMetric(
                        values=sample_pde,
                        row_axis=token_axis,
                        column_axis=token_axis,
                        unit="angstrom",
                    ),
                    "contact_probs": PairMetric(
                        values=contact_numpy,
                        row_axis=token_axis,
                        column_axis=token_axis,
                    ),
                    "chain_pair_iptm": PairMetric(
                        values=chain_pair_iptm[sample_index].detach().float().cpu().numpy(),
                        row_axis=chain_axis,
                        column_axis=chain_axis,
                    ),
                    "chain_pair_iptm_global": PairMetric(
                        values=chain_pair_iptm_global[sample_index].detach().float().cpu().numpy(),
                        row_axis=chain_axis,
                        column_axis=chain_axis,
                    ),
                    "chain_pair_gpde": PairMetric(
                        values=chain_pair_gpde[sample_index].detach().float().cpu().numpy(),
                        row_axis=chain_axis,
                        column_axis=chain_axis,
                        unit="angstrom",
                    ),
                    "chain_pair_plddt": PairMetric(
                        values=chain_pair_plddt[sample_index].detach().float().cpu().numpy(),
                        row_axis=chain_axis,
                        column_axis=chain_axis,
                    ),
                },
            )
        )
    return FoldResult(predictions=tuple(predictions))
