r"""
Boltz-1 folding Pipeline.
"""

from __future__ import annotations

from os import PathLike
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...data.evidence import MSA, MsaCollection
from ...data.molecular import ENTITY_PROTEIN, AtomSite, ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_structure_input
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .model import Boltz1Input, Boltz1Model, Boltz1Pocket, Boltz1SamplingWithConfidenceOutput
from .processor import Boltz1Processor


class Boltz1Pipeline(PairedPipeline[Boltz1Model, Boltz1Processor]):
    r"""
    Structure prediction pipeline for Boltz 1.
    """

    model_architecture = "boltz1"

    def prepare_fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        pocket: Boltz1Pocket | None = None,
        generator: torch.Generator | None = None,
        max_msa_rows: int | None = 8192,
    ) -> Boltz1Input:
        r"""
        Prepare one molecular system and its evidence for Boltz-1 folding.

        ``generator`` is advanced by stochastic reference-conformer preparation.
        Staged execution can pass the same generator to
        ``model.sample_with_confidence`` and retain the convenience Pipeline's
        random-number order.
        """

        prepared_input = cast(MolecularSystem | Structure, resolve_structure_input(input))
        system = prepared_input.system if isinstance(prepared_input, Structure) else prepared_input
        prepared_msa = resolve_msa_input(
            msa,
            system,
            entity_types=(ENTITY_PROTEIN,),
            max_rows=max_msa_rows,
        )
        return self.processor(
            prepared_input,
            msa=prepared_msa,
            pocket=pocket,
            generator=generator,
        )

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
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
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
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
            max_msa_rows: Optional cap applied only while reading a text alignment path.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            use_potentials: Enable the released Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle
                resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, and PDE values
                are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch = self.prepare_fold(
            input,
            msa=msa,
            pocket=pocket,
            generator=generator,
            max_msa_rows=max_msa_rows,
        )
        with torch.inference_mode():
            output = self.model.sample_with_confidence(
                batch,
                num_samples=num_samples,
                generator=generator,
                recycling_steps=recycling_steps,
                msa_max_depth=msa_max_depth,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                use_potentials=use_potentials,
                feynman_kac=feynman_kac,
                physical_guidance=physical_guidance,
                num_particles=num_particles,
                feynman_kac_lambda=feynman_kac_lambda,
                feynman_kac_resampling_interval=feynman_kac_resampling_interval,
                num_guidance_steps=num_guidance_steps,
            )
            return self.decode_fold(batch, output)

    def decode_fold(
        self,
        input: Boltz1Input,
        output: Boltz1SamplingWithConfidenceOutput,
    ) -> FoldResult:
        r"""
        Decode one raw Boltz-1 model output without executing the model.

        Model atom order is mapped back onto the complete canonical atom axis;
        confidence values are lifted onto public symbolic axes at this boundary.
        """

        if input.token_mask.shape[0] != 1:
            raise ValueError("Boltz1Pipeline.decode_fold accepts one molecular system per call")
        token_count = int(input.token_mask[0].sum())
        atom_count = int(input.atom_mask[0].sum())
        canonical_atom_index = input.atom_index[0, :atom_count].detach().cpu().long()
        canonical_token_component = input.token_component_index[0, :token_count].detach().cpu().long()
        canonical_token_atom = input.token_atom_index[0, :token_count].detach().cpu().long()
        result_system = input.system.cpu()  # type: ignore[attr-defined]
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

        coordinates = output.coordinates[0, :, :atom_count].to(
            device="cpu",
            dtype=torch.float32,
        )
        num_samples = coordinates.shape[0]
        confidence = output.confidence

        def per_sample(value: Tensor | None) -> np.ndarray | None:
            return None if value is None else value[0].detach().to(device="cpu", dtype=torch.float32).numpy()

        plddt = per_sample(None if confidence is None else confidence.plddt)
        pae = per_sample(None if confidence is None else confidence.pae)
        pde = per_sample(None if confidence is None else confidence.pde)
        complex_plddt = per_sample(None if confidence is None else confidence.complex_plddt)
        complex_iplddt = per_sample(None if confidence is None else confidence.complex_iplddt)
        complex_pde = per_sample(None if confidence is None else confidence.complex_pde)
        complex_ipde = per_sample(None if confidence is None else confidence.complex_ipde)
        confidence_score = per_sample(None if confidence is None else confidence.confidence_score)
        ptm = per_sample(None if confidence is None else confidence.ptm)
        iptm = per_sample(None if confidence is None else confidence.iptm)
        ligand_iptm = per_sample(None if confidence is None else confidence.ligand_iptm)
        protein_iptm = per_sample(None if confidence is None else confidence.protein_iptm)
        chain_pair_iptm = per_sample(None if confidence is None else confidence.chain_pair_iptm)
        chain_axis = tuple(
            result_system.instance_id(instance_index) for instance_index in range(result_system.instance_symbol.numel())
        )

        resolved_probability: np.ndarray | None = None
        if confidence is not None:
            resolved_probability = (
                confidence.resolved_logits[0, :, :token_count].detach().float().softmax(dim=-1)[..., 1].cpu().numpy()
            )

        predictions: list[FoldPrediction] = []
        complete_atom_count = result_system.atom_symbol.numel()
        complete_component_count = result_system.component_symbol.numel()
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
            complete_coordinates[canonical_atom_index] = coordinates[sample_index]
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

            sample_scores: dict[str, Score] = {}
            if complex_plddt is not None:
                sample_scores["plddt"] = Score(float(complex_plddt[sample_index]), preferred="higher")
            if complex_iplddt is not None:
                sample_scores["iplddt"] = Score(float(complex_iplddt[sample_index]), preferred="higher")
            if complex_pde is not None:
                sample_scores["pde"] = Score(
                    float(complex_pde[sample_index]),
                    unit="angstrom",
                    preferred="lower",
                )
            if complex_ipde is not None:
                sample_scores["ipde"] = Score(
                    float(complex_ipde[sample_index]),
                    unit="angstrom",
                    preferred="lower",
                )
            if ptm is not None:
                sample_scores["ptm"] = Score(float(ptm[sample_index]), preferred="higher")
            if iptm is not None:
                sample_scores["iptm"] = Score(float(iptm[sample_index]), preferred="higher")
            if ligand_iptm is not None:
                sample_scores["ligand_iptm"] = Score(
                    float(ligand_iptm[sample_index]),
                    preferred="higher",
                )
            if protein_iptm is not None:
                sample_scores["protein_iptm"] = Score(
                    float(protein_iptm[sample_index]),
                    preferred="higher",
                )
            if confidence_score is not None:
                sample_scores["confidence_score"] = Score(
                    float(confidence_score[sample_index]),
                    preferred="higher",
                )

            sample_metrics: dict[str, Metric] = {}
            if plddt is not None:
                sample_metrics["plddt"] = Metric(values=plddt[sample_index], axis=token_axis)
            if resolved_probability is not None:
                sample_metrics["resolved_probability"] = Metric(
                    values=resolved_probability[sample_index],
                    axis=token_axis,
                )

            sample_pair_metrics: dict[str, PairMetric] = {}
            if pae is not None:
                sample_pair_metrics["pae"] = PairMetric(
                    values=pae[sample_index],
                    row_axis=token_axis,
                    column_axis=token_axis,
                    unit="angstrom",
                )
            if pde is not None:
                sample_pair_metrics["pde"] = PairMetric(
                    values=pde[sample_index],
                    row_axis=token_axis,
                    column_axis=token_axis,
                    unit="angstrom",
                )
            if chain_pair_iptm is not None:
                sample_pair_metrics["chain_pair_iptm"] = PairMetric(
                    values=chain_pair_iptm[sample_index],
                    row_axis=chain_axis,
                    column_axis=chain_axis,
                )

            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores=sample_scores,
                    metrics=sample_metrics,
                    pair_metrics=sample_pair_metrics,
                )
            )

        return FoldResult(predictions=tuple(predictions))
