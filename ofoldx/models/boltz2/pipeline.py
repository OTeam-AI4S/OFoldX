r"""
Pipelines for Boltz-2 folding and two-stage affinity scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import cast

import torch

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import ENTITY_PROTEIN, AtomSite, ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_structure_input, resolve_template_input
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, ScoreResult
from .model import (
    Boltz2AffinityModel,
    Boltz2AffinityScoreOutput,
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Pocket,
    Boltz2Template,
)
from .processor import Boltz2AffinityProcessor, Boltz2Processor


class Boltz2Pipeline(PairedPipeline[Boltz2Model, Boltz2Processor]):
    r"""
    Structure prediction pipeline for Boltz 2.
    """

    model_architecture = "boltz2"

    def prepare_fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | Boltz2Template] | str | PathLike[str] = (),
        design: Sequence[ComponentReference] = (),
        self_template: Sequence[ComponentReference] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        generator: torch.Generator | None = None,
        max_msa_rows: int | None = 8192,
    ) -> Boltz2Input:
        r"""
        Prepare one molecular system and its evidence for Boltz-2 folding.

        ``generator`` is advanced by stochastic reference-conformer preparation.
        Staged execution can pass the same generator to
        ``model.sample_with_confidence`` and retain the convenience Pipeline's
        random-number order.
        """

        structure_input = cast(MolecularSystem | Structure, resolve_structure_input(input))
        system = structure_input.system if isinstance(structure_input, Structure) else structure_input
        prepared_msa = resolve_msa_input(
            msa,
            system,
            entity_types={ENTITY_PROTEIN},
            max_rows=max_msa_rows,
        )
        return self.processor(
            structure_input,
            msa=prepared_msa,
            templates=resolve_template_input(templates) if isinstance(templates, str | PathLike) else templates,
            design=design,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | Boltz2Template] | str | PathLike[str] = (),
        design: Sequence[ComponentReference] = (),
        self_template: Sequence[ComponentReference] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular system with Boltz 2.

        Args:
            input: Molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            templates: Structural protein templates with canonical residue alignments.
            design: Component occurrences designed by the calling protocol.
            self_template: Component occurrences whose input coordinates form one
                direct self-template.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-conformer augmentation and
                coordinate sampling. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and coordinate
                sampling. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            msa_max_depth: Optional rows independently sampled for each MSA-module
                invocation. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading a text alignment path.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            max_parallel_samples: Maximum samples evaluated by one denoiser call.
            use_potentials: Enable the released Feynman--Kac and physical-guidance
                preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            contact_guidance: Whether forced contact, pocket, and template
                conditions apply their potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, and PDE values
                are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        features = self.prepare_fold(
            input,
            msa=msa,
            templates=templates,
            design=design,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
            max_msa_rows=max_msa_rows,
        )
        with torch.inference_mode():
            output = self.model.sample_with_confidence(
                features,
                num_samples=num_samples,
                generator=generator,
                recycling_steps=recycling_steps,
                msa_max_depth=msa_max_depth,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                max_parallel_samples=max_parallel_samples,
                use_potentials=use_potentials,
                feynman_kac=feynman_kac,
                physical_guidance=physical_guidance,
                contact_guidance=contact_guidance,
                num_particles=num_particles,
                feynman_kac_lambda=feynman_kac_lambda,
                feynman_kac_resampling_interval=feynman_kac_resampling_interval,
                num_guidance_steps=num_guidance_steps,
            )
            return self.decode_fold(features, output)

    def decode_fold(
        self,
        input: Boltz2Input,
        output: Boltz2ConfidenceOutput,
    ) -> FoldResult:
        r"""
        Decode one raw Boltz-2 folding output without executing the model.

        Model atom and token axes are mapped back onto the complete canonical
        system, and confidence tensors are lifted onto public symbolic axes.
        """

        if input.token_mask.shape[0] != 1 or input.system.system_entity_ptr.numel() != 2:
            raise ValueError("Boltz2Pipeline.decode_fold accepts one molecular system per call")

        token_valid = input.token_mask[0].detach().cpu().bool()
        token_selection = token_valid.nonzero(as_tuple=False).flatten()
        result_system = input.system.cpu()  # type: ignore[attr-defined]
        token_component_index = input.token_component_index[0].detach().cpu().long()[token_valid]
        token_atom_index = input.token_atom_index[0].detach().cpu().long()[token_valid]
        token_axis_values: list[ComponentReference | AtomSite] = []
        for atom_index, component_index in zip(
            token_atom_index.tolist(),
            token_component_index.tolist(),
        ):
            if atom_index >= 0:
                token_axis_values.append(result_system.atom_site(atom_index))
            else:
                token_axis_values.append(result_system.component_reference(component_index))
        token_axis = tuple(token_axis_values)
        chain_axis = tuple(
            result_system.instance_id(instance_index) for instance_index in range(result_system.instance_symbol.numel())
        )
        num_samples = output.coordinates.shape[1]

        output_token_selection = token_selection.to(output.plddt.device)
        plddt = (
            output.plddt[0, :, output_token_selection]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )
        resolved_probability = (
            output.resolved_probability[
                0,
                :,
                output_token_selection,
            ]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        pae = (
            output.pae[0, :, output_token_selection][
                :,
                :,
                output_token_selection,
            ]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        pde = (
            output.pde[0, :, output_token_selection][
                :,
                :,
                output_token_selection,
            ]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )

        ptm = output.ptm[0].detach().float().cpu()
        iptm = output.iptm[0].detach().float().cpu()
        ligand_iptm = output.ligand_iptm[0].detach().float().cpu()
        protein_iptm = output.protein_iptm[0].detach().float().cpu()
        chain_pair_iptm = output.pair_chains_iptm[0].detach().float().cpu()
        complex_plddt = output.complex_plddt[0].detach().float().cpu()
        complex_iplddt = output.complex_iplddt[0].detach().float().cpu()
        complex_pde = output.complex_pde[0].detach().float().cpu()
        complex_ipde = output.complex_ipde[0].detach().float().cpu()
        interface = torch.where(torch.isclose(iptm, torch.zeros_like(iptm)), ptm, iptm)
        confidence_score = (4.0 * (complex_plddt / 100.0) + interface) / 5.0
        has_design = bool(
            input.design_mask[
                0,
                token_selection.to(input.design_mask.device),
            ].any()
        )
        design_scores = {
            "design_to_target_iptm": output.design_to_target_iptm[0].detach().float().cpu(),
            "design_residue_iptm": output.design_residue_iptm[0].detach().float().cpu(),
            "design_iptm": output.design_iptm[0].detach().float().cpu(),
            "design_iiptm": output.design_iiptm[0].detach().float().cpu(),
            "target_ptm": output.target_ptm[0].detach().float().cpu(),
            "design_ptm": output.design_ptm[0].detach().float().cpu(),
            "interaction_pae": output.interaction_pae[0].detach().float().cpu(),
            "min_interaction_pae": output.min_interaction_pae[0].detach().float().cpu(),
            "min_design_to_target_pae": output.min_design_to_target_pae[0].detach().float().cpu(),
            "design_ipsae_min": output.design_ipsae_min[0].detach().float().cpu(),
            "design_to_target_ipsae": output.design_to_target_ipsae[0].detach().float().cpu(),
            "target_to_design_ipsae": output.target_to_design_ipsae[0].detach().float().cpu(),
        }

        model_atom_index = input.atom_index[0].detach().cpu().long()
        atom_valid = input.atom_mask[0].detach().cpu().bool() & (model_atom_index >= 0)
        canonical_atom_index = model_atom_index[atom_valid]
        complete_atom_count = result_system.atom_symbol.numel()
        complete_component_count = result_system.component_symbol.numel()

        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            complete_coordinates = torch.zeros(
                complete_atom_count,
                3,
                dtype=torch.float32,
            )
            coordinate_mask = torch.zeros(
                complete_atom_count,
                dtype=torch.bool,
            )
            model_coordinates = output.coordinates[0, sample_index]
            complete_coordinates[canonical_atom_index] = (
                model_coordinates[atom_valid.to(model_coordinates.device)]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            coordinate_mask[canonical_atom_index] = True
            structure = Structure(  # type: ignore[call-arg]
                system=result_system,
                coordinates=complete_coordinates,
                coordinate_mask=coordinate_mask,
                occupancy=torch.zeros(complete_atom_count),
                occupancy_mask=torch.zeros(
                    complete_atom_count,
                    dtype=torch.bool,
                ),
                b_factor=torch.zeros(complete_atom_count),
                b_factor_mask=torch.zeros(
                    complete_atom_count,
                    dtype=torch.bool,
                ),
                source_symbols=(),
                residue_id=torch.full(
                    (complete_component_count,),
                    -1,
                    dtype=torch.int64,
                ),
                insertion_code=torch.full(
                    (complete_component_count,),
                    -1,
                    dtype=torch.int64,
                ),
                batch_size=[],
            )
            scores = {
                "plddt": Score(
                    float(complex_plddt[sample_index]),
                    preferred="higher",
                ),
                "ptm": Score(
                    float(ptm[sample_index]),
                    preferred="higher",
                ),
                "iptm": Score(
                    float(iptm[sample_index]),
                    preferred="higher",
                ),
                "ligand_iptm": Score(
                    float(ligand_iptm[sample_index]),
                    preferred="higher",
                ),
                "protein_iptm": Score(
                    float(protein_iptm[sample_index]),
                    preferred="higher",
                ),
                "confidence_score": Score(
                    float(confidence_score[sample_index]),
                    preferred="higher",
                ),
                "complex_iplddt": Score(
                    float(complex_iplddt[sample_index]),
                    preferred="higher",
                ),
                "complex_pde": Score(
                    float(complex_pde[sample_index]),
                    unit="angstrom",
                    preferred="lower",
                ),
                "complex_ipde": Score(
                    float(complex_ipde[sample_index]),
                    unit="angstrom",
                    preferred="lower",
                ),
            }
            if has_design:
                for name, values in design_scores.items():
                    if name in {
                        "interaction_pae",
                        "min_interaction_pae",
                        "min_design_to_target_pae",
                    }:
                        scores[name] = Score(
                            float(values[sample_index]),
                            unit="angstrom",
                            preferred="lower",
                        )
                    else:
                        scores[name] = Score(
                            float(values[sample_index]),
                            preferred="higher",
                        )
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores=scores,
                    metrics={
                        "plddt": Metric(
                            values=plddt[sample_index].numpy(),
                            axis=token_axis,
                        ),
                        "resolved_probability": Metric(
                            values=resolved_probability[sample_index].numpy(),
                            axis=token_axis,
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=pae[sample_index].numpy(),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=pde[sample_index].numpy(),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_pair_iptm[sample_index].numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))


class Boltz2AffinityScorePipeline(PairedPipeline[Boltz2AffinityModel, Boltz2AffinityProcessor]):
    r"""
    Binding-affinity scoring pipeline for one folded protein-ligand complex.
    """

    model_architecture = "boltz2_affinity"

    def prepare_score(
        self,
        input: Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        max_msa_rows: int | None = 8192,
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""Prepare one folded protein-ligand complex for affinity scoring."""

        structure = cast(Structure, resolve_structure_input(input))
        prepared_msa = resolve_msa_input(
            msa,
            structure.system,
            entity_types={ENTITY_PROTEIN},
            max_rows=max_msa_rows,
        )
        return self.processor(
            structure,
            ligand=ligand,
            msa=prepared_msa,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        num_samples: int = 5,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score affinity for one already folded protein-ligand complex.

        Args:
            input: Folded Structure or local structure path.
            ligand: Instance ID of the molecule to score. May be omitted when the
                system contains exactly one molecule.
            msa: Optional MSA value, collection, or alignment path.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            num_samples: Number of affinity-pass coordinate samples.
            recycling_steps: Number of affinity trunk recycling iterations.
            msa_max_depth: Optional rows independently sampled for every MSA-module
                invocation. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading a text alignment path.
            num_sampling_steps: Number of affinity-pass diffusion transitions.
            step_scale: Optional reverse-diffusion Euler-step scale.
            molecular_weight_correction: Whether to apply molecular-weight calibration.
            seed: Optional seed shared by preprocessing and scoring; mutually exclusive
                with ``generator``.
            generator: Optional generator advanced by preprocessing and scoring;
                mutually exclusive with ``seed``.

        Returns:
            The affinity pass's selected structure and scalar affinity scores.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        features = self.prepare_score(
            input,
            ligand=ligand,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method=method,
            max_msa_rows=max_msa_rows,
            generator=generator,
        )
        output = self.model.score(
            features,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        return self.decode_score(features, output)

    def decode_score(self, input: Boltz2Input, output: Boltz2AffinityScoreOutput) -> ScoreResult:
        r"""Decode raw Boltz-2 affinity tensors without executing the Model."""

        selected_index = int(output.selected_sample_index[0].detach().cpu())
        structure = _decode_affinity_structure(input, output.structure.coordinates[0, selected_index])
        affinity = output.affinity.detach().cpu()
        scope = frozenset({_affinity_instance_id(input)})
        return ScoreResult(
            target=structure,
            scores={
                "affinity_pred_value": Score(
                    float(affinity.affinity_pred_value[0, 0, 0]),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary": Score(
                    float(affinity.affinity_probability_binary[0, 0, 0]),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_pred_value1": Score(
                    float(affinity.affinity_pred_value1[0, 0, 0]),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_pred_value2": Score(
                    float(affinity.affinity_pred_value2[0, 0, 0]),
                    unit="log10(IC50/µM)",
                    preferred="lower",
                    scope=scope,
                ),
                "affinity_probability_binary1": Score(
                    float(affinity.affinity_probability_binary1[0, 0, 0]),
                    preferred="higher",
                    scope=scope,
                ),
                "affinity_probability_binary2": Score(
                    float(affinity.affinity_probability_binary2[0, 0, 0]),
                    preferred="higher",
                    scope=scope,
                ),
            },
        )


class Boltz2AffinityPipeline(CompositePipeline):
    r"""
    Two-pass structure prediction and binding-affinity pipeline for Boltz 2.

    Args:
        affinity: Independently deployable affinity-scoring stage.
        fold: Independently deployable structure-prediction stage.
    """

    # The affinity pass owns the repository root; the structure/confidence pair is stored in
    # the fixed ``structure/`` subdirectory.
    pair_layout = (
        CompositePair(
            "affinity",
            "",
            Boltz2AffinityScorePipeline,
        ),
        CompositePair("fold", "structure", Boltz2Pipeline),
    )

    def __init__(
        self,
        *,
        affinity: Boltz2AffinityScorePipeline,
        fold: Boltz2Pipeline,
    ) -> None:
        super().__init__({"affinity": affinity, "fold": fold})

    @property
    def affinity(self) -> Boltz2AffinityScorePipeline:
        return cast(Boltz2AffinityScorePipeline, self.components["affinity"])

    @property
    def fold(self) -> Boltz2Pipeline:
        return cast(Boltz2Pipeline, self.components["fold"])

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | Boltz2Template] | str | PathLike[str] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        num_samples: int = 1,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        affinity_num_samples: int = 5,
        affinity_recycling_steps: int | None = None,
        affinity_num_sampling_steps: int | None = None,
        affinity_step_scale: float | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityResult:
        r"""
        Predict candidate structures and score protein-ligand affinity.

        Args:
            input: Molecular system, structure, or local structure path.
            ligand: Instance ID of the molecule to score. May be omitted when the
                system contains exactly one molecule.
            msa: Optional MSA value, collection, or alignment path used by both passes.
            templates: Structural protein templates used by the first structure pass.
            contacts: Learned pairwise contact conditions used by both passes.
            pockets: Learned binder-to-pocket conditions used by both passes.
            method: Experimental method used by the first structure pass.
            num_samples: Number of first-pass structure samples.
            recycling_steps: Number of first-pass trunk recycling iterations.
            num_sampling_steps: Number of first-pass diffusion transitions.
            step_scale: Optional first-pass reverse-diffusion Euler-step scale.
            max_parallel_samples: Maximum first-pass samples evaluated by one
                denoiser call.
            use_potentials: Enable the released first-pass Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether the first pass uses particle resampling.
            physical_guidance: Whether the first pass applies physical-potential
                gradients.
            contact_guidance: Whether forced first-pass contact, pocket, and
                template conditions apply their potential gradients.
            num_particles: Number of first-pass particles per requested sample.
            feynman_kac_lambda: First-pass Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: First-pass reverse-diffusion step
                interval between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                first-pass reverse-diffusion step.
            affinity_num_samples: Number of second-pass candidate samples.
            affinity_recycling_steps: Number of second-pass trunk recycling
                iterations.
            affinity_num_sampling_steps: Number of second-pass diffusion transitions.
            affinity_step_scale: Optional second-pass reverse-diffusion Euler-step scale.
            msa_max_depth: Optional rows independently sampled for every MSA-module
                invocation in both passes. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading a text alignment path.
            molecular_weight_correction: Whether to apply molecular-weight calibration.
            seed: Optional seed shared by preprocessing and both model passes; mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by preprocessing and both model
                passes; mutually exclusive with ``seed``.

        Returns:
            First-pass predictions sorted by descending confidence, the selected
                structure, and affinity scores. Affinity values use
                ``log10(IC50 / micromolar)`` units; binder probabilities lie in
                ``[0, 1]``.
        """

        # Reference augmentation and both model passes advance one RNG stream.
        generator = resolve_generator(seed, generator, device=self.fold.processor.device)

        structure_input = cast(MolecularSystem | Structure, resolve_structure_input(input))
        msa = resolve_msa_input(
            msa,
            structure_input.system if isinstance(structure_input, Structure) else structure_input,
            entity_types={ENTITY_PROTEIN},
            max_rows=max_msa_rows,
        )

        # The structure and affinity passes use independent sampling counts.
        fold_result = self.fold.fold(
            structure_input,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            msa_max_depth=msa_max_depth,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            max_msa_rows=max_msa_rows,
            generator=generator,
        )

        # Rank first-stage predictions by descending confidence_score; Python's stable sort keeps generation
        # order among ties (earliest-tie stable), so structure_predictions is confidence-descending, not
        # generation-ordered.
        structure_predictions = tuple(
            sorted(
                fold_result.predictions,
                key=lambda prediction: prediction.scores["confidence_score"].value,
                reverse=True,
            )
        )
        best_structure = structure_predictions[0].structure

        score_result = self.affinity.score(
            best_structure,
            ligand=ligand,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method="other",
            num_samples=affinity_num_samples,
            recycling_steps=affinity_recycling_steps,
            msa_max_depth=msa_max_depth,
            max_msa_rows=max_msa_rows,
            num_sampling_steps=affinity_num_sampling_steps,
            step_scale=affinity_step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        return Boltz2AffinityResult(
            target=score_result.target,
            structure_predictions=structure_predictions,
            scores=score_result.scores,
            metrics=score_result.metrics,
        )


@dataclass(eq=False)
class Boltz2AffinityResult:
    r"""
    One two-stage Boltz-2 affinity result.

    Contains the selected second-pass ``target``, confidence-sorted first-pass
    ``structure_predictions``, and affinity scores and metrics.

    Attributes:
        target: Structure selected for affinity scoring.
        structure_predictions: First-stage predictions in descending confidence order.
        scores: Scalar affinity scores keyed by stable scientific names.
        metrics: Affinity metrics keyed by stable scientific names.
    """

    target: Structure
    structure_predictions: tuple[FoldPrediction, ...]
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


def _decode_affinity_structure(input: Boltz2Input, model_coordinates: torch.Tensor) -> Structure:
    r"""Scatter selected affinity-pass coordinates onto the canonical atom axis."""

    result_system = input.system.cpu()  # type: ignore[attr-defined]
    atom_index = input.atom_index[0].detach().cpu().long()
    valid = input.atom_mask[0].detach().cpu().bool() & (atom_index >= 0)
    canonical_atom_index = atom_index[valid]
    complete_atom_count = result_system.atom_symbol.numel()
    complete_component_count = result_system.component_symbol.numel()
    coordinates = torch.zeros(complete_atom_count, 3, dtype=torch.float32)
    coordinate_mask = torch.zeros(complete_atom_count, dtype=torch.bool)
    coordinates[canonical_atom_index] = (
        model_coordinates[valid.to(model_coordinates.device)]
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )
    coordinate_mask[canonical_atom_index] = True
    return Structure(  # type: ignore[call-arg]
        system=result_system,
        coordinates=coordinates,
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


def _affinity_instance_id(input: Boltz2Input) -> str:
    affinity_mask = cast(torch.Tensor, input.affinity_token_mask)[0].detach().cpu().bool()
    affinity_mask &= input.token_mask[0].detach().cpu().bool()
    component_index = input.token_component_index[0].detach().cpu().long()[affinity_mask]
    component_instance = torch.repeat_interleave(
        torch.arange(input.system.instance_symbol.numel()),
        (input.system.instance_component_ptr[1:] - input.system.instance_component_ptr[:-1]).detach().cpu(),
    )
    instance_index = torch.unique(component_instance[component_index])
    if instance_index.numel() != 1:
        raise ValueError("Boltz-2 affinity tokens must identify one ligand instance")
    return input.system.instance_id(int(instance_index[0]))
