r"""
Concrete AlphaFold 2 monomer and multimer folding Pipelines.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import ENTITY_PROTEIN, MolecularSystem, PolymerSequence
from ...data.structure import STRUCTURE_ARTIFACT_SUFFIX, Structure
from ...io import read_fasta_records
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_structure_input, resolve_template_input
from ...references import classify_reference
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .config import AlphaFold2Config, AlphaFold2MultimerConfig
from .model import (
    AlphaFold2InitialGuessInput,
    AlphaFold2InitialGuessModel,
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import (
    AlphaFold2InitialGuessProcessor,
    AlphaFold2Processor,
    monomer_system_from_sequences,
)
from .relax import AlphaFold2AmberRelaxation, AlphaFold2RelaxationResult

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".seq"))


class AlphaFold2Pipeline(PairedPipeline[AlphaFold2Model, AlphaFold2Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 2.
    """

    model_architecture = "alphafold2"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
    ) -> AlphaFold2Input:
        r"""Prepare one monomer and its evidence for AlphaFold 2 folding."""

        canonical_input = _monomer_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types=(ENTITY_PROTEIN,)),
            templates=resolve_template_input(templates),
        )

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict the structure of one protein with AlphaFold 2.

        Args:
            input: Inline sequence, FASTA path, protein sequence, molecular system, or structure
                topology containing exactly one protein instance.
            msa: Optional monomer MSA value or alignment path.
            templates: Template structures with explicit target alignments, or a
                native template-artifact path.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually
                exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction.
                Mutually exclusive with ``seed``.
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses the Model configuration.

        Returns:
            Generation-ordered structure predictions. Coordinates and PAE values
                are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.prepare_fold(
            input,
            msa=msa,
            templates=templates,
        )
        output = self.sample_fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )
        return self.decode_fold(model_input, output)

    @torch.inference_mode()
    def sample_fold(
        self,
        input: AlphaFold2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent AlphaFold 2 forwards and return raw tensor outputs."""

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.residue_type_ids.shape[0] != 1:
            raise ValueError("AlphaFold2Pipeline.sample_fold accepts one canonical system per call")
        device = input.residue_type_ids.device
        return tuple(
            self.model(
                input,
                recycling_steps=recycling_steps,
                generator=_sample_generator(generator, device),
            )
            for _ in range(num_samples)
        )

    def decode_fold(
        self,
        input: AlphaFold2Input,
        output: Sequence[AlphaFold2Output],
    ) -> FoldResult:
        r"""Decode raw AlphaFold 2 outputs without executing the model."""

        return _decode_monomer_outputs(input, output, self.model.config)


class AlphaFold2InitialGuessPipeline(PairedPipeline[AlphaFold2InitialGuessModel, AlphaFold2InitialGuessProcessor]):
    r"""
    Initial-guess target--binder evaluation with the monomer/pTM graph.
    """

    model_architecture = "alphafold2_initial_guess"

    def prepare_fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        binder_instance_id: str,
    ) -> AlphaFold2InitialGuessInput:
        r"""Prepare one target--binder structure for initial-guess folding."""

        structure = resolve_structure_input(input)
        if not isinstance(structure, Structure):
            raise TypeError("AlphaFold 2 initial guess requires a Structure")
        return self.processor(structure, binder_instance_id=binder_instance_id)

    def fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        binder_instance_id: str,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int = 4,
    ) -> FoldResult:
        r"""
        Predict a complex from its generated atom37 recycling state.

        Args:
            input: Sequence-assigned generated target--binder complex.
            binder_instance_id: Exact protein instance treated as the binder.
            num_samples: Number of predictions.
            seed: Optional MSA sampling seed, mutually exclusive with ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with
                ``seed``.
            recycling_steps: Total network passes, including the final pass.

        Returns:
            Complex predictions with binder pLDDT, pTM, iPTM, and raw and
                normalized interface PAE scores.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.prepare_fold(input, binder_instance_id=binder_instance_id)
        output = self.sample_fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )
        return self.decode_fold(model_input, output)

    @torch.inference_mode()
    def sample_fold(
        self,
        input: AlphaFold2InitialGuessInput,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent initial-guess forwards and return raw outputs."""

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.residue_type_ids.shape[0] != 1:
            raise ValueError("AlphaFold2InitialGuessPipeline.sample_fold accepts one canonical system per call")
        device = input.residue_type_ids.device
        return tuple(
            self.model(
                input,
                recycling_steps=recycling_steps,
                generator=_sample_generator(generator, device),
            )
            for _ in range(num_samples)
        )

    def decode_fold(
        self,
        input: AlphaFold2InitialGuessInput,
        output: Sequence[AlphaFold2Output],
    ) -> FoldResult:
        r"""Decode raw initial-guess outputs and add binder/interface scores."""

        result = _decode_monomer_outputs(input, output, self.model.config)
        binder_mask = input.binder_mask[0].bool()
        target_mask = input.sequence_mask[0].bool() & ~binder_mask
        interface_mask = binder_mask[:, None] & target_mask[None, :]
        chain_index = input.chain_index[0]
        binder_scope = frozenset(
            residue
            for residue, selected in zip(
                _component_axis(input.system, input.component_index[0]),
                binder_mask.cpu().numpy(),
                strict=True,
            )
            if selected
        )
        for sample, prediction in zip(output, result.predictions, strict=True):
            if sample.pae_logits is None:
                continue
            pae = prediction.pair_metrics["pae"].values
            symmetric_pae = (pae + pae.T) / 2.0
            raw_interface_pae = float(symmetric_pae[interface_mask.cpu().numpy()].mean())
            binder_plddt = prediction.metrics["plddt"].values[binder_mask.cpu().numpy()]
            iptm = alphafold2_predicted_interface_tm_score(
                sample.pae_logits[0],
                chain_index,
                input.sequence_mask[0],
                self.model.config,
            )
            prediction.scores["binder_mean_plddt"] = Score(
                float(binder_plddt.mean()),
                preferred="higher",
                scope=binder_scope,
            )
            prediction.scores["iptm"] = Score(iptm, preferred="higher")
            prediction.scores["interface_pae"] = Score(
                raw_interface_pae,
                unit="angstrom",
                preferred="lower",
            )
            prediction.scores["normalized_interface_pae"] = Score(
                raw_interface_pae / 31.0,
                preferred="lower",
            )
        return result


class AlphaFold2MultimerPipeline(PairedPipeline[AlphaFold2MultimerModel, AlphaFold2MultimerProcessor]):
    r"""
    Multimer structure prediction pipeline for AlphaFold-Multimer.
    """

    model_architecture = "alphafold2_multimer"

    def prepare_fold(
        self,
        input: (
            MolecularSystem
            | Structure
            | str
            | PathLike[str]
            | Sequence[PolymerSequence]
            | Mapping[str, PolymerSequence]
        ),
        *,
        msa: MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
    ) -> AlphaFold2MultimerInput:
        r"""Prepare one protein assembly and its evidence for AlphaFold-Multimer."""

        canonical_input: MolecularSystem | Sequence[PolymerSequence] | Mapping[str, PolymerSequence]
        if isinstance(input, str | PathLike | Structure):
            resolved_input = resolve_structure_input(input)
            if not isinstance(resolved_input, MolecularSystem | Structure):
                raise TypeError("AlphaFold-Multimer requires a molecular structure or polymer sequences")
            canonical_input = resolved_input.system if isinstance(resolved_input, Structure) else resolved_input
        else:
            canonical_input = input
        system = canonical_input if isinstance(canonical_input, MolecularSystem) else None
        resolved_msa = resolve_msa_input(
            msa,
            system,
            entity_types=(ENTITY_PROTEIN,),
        )
        if isinstance(resolved_msa, MSA):
            raise ValueError("AlphaFold-Multimer requires MsaCollection for supplied MSA evidence")
        return self.processor(
            canonical_input,
            msa=resolved_msa,
            templates=resolve_template_input(templates),
        )

    def fold(
        self,
        input: (
            MolecularSystem
            | Structure
            | str
            | PathLike[str]
            | Sequence[PolymerSequence]
            | Mapping[str, PolymerSequence]
        ),
        *,
        msa: MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict the structure of one protein assembly with AlphaFold-Multimer.

        Args:
            input: Molecular system or named/ordered protein sequences forming one assembly.
            msa: Optional multimer MSA collection or native MSA-artifact path.
            templates: Template structures with explicit target alignments, or a
                native template-artifact path.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually
                exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction.
                Mutually exclusive with ``seed``.
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses the Model configuration.

        Returns:
            Generation-ordered structure predictions. Coordinates and PAE values
                are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.prepare_fold(input, msa=msa, templates=templates)
        output = self.sample_fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )
        return self.decode_fold(model_input, output)

    @torch.inference_mode()
    def sample_fold(
        self,
        input: AlphaFold2MultimerInput,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent AlphaFold-Multimer forwards and return raw outputs."""

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if input.residue_type_ids.shape[0] != 1:
            raise ValueError("AlphaFold2MultimerPipeline.sample_fold accepts one canonical system per call")
        device = input.residue_type_ids.device
        return tuple(
            self.model(
                input,
                recycling_steps=recycling_steps,
                generator=_sample_generator(generator, device),
            )
            for _ in range(num_samples)
        )

    def decode_fold(
        self,
        input: AlphaFold2MultimerInput,
        output: Sequence[AlphaFold2Output],
    ) -> FoldResult:
        r"""Decode raw AlphaFold-Multimer outputs without executing the model."""

        residue_axis = _component_axis(input.system, input.component_index[0])
        predictions: list[FoldPrediction] = []
        for sample in output:
            # There is no target assembly against which to permute symmetric copies.
            # Residue labels therefore retain the Processor's concatenation order.
            structure = _decode_atom37_structure(
                sample.final_atom_positions[0],
                sample.final_atom_mask[0],
                input.system,
                input.atom37_index[0],
            )
            plddt_logits = sample.plddt_logits[0].detach().float()
            plddt_bins = (
                torch.arange(
                    plddt_logits.shape[-1],
                    device=plddt_logits.device,
                    dtype=plddt_logits.dtype,
                )
                + 0.5
            ) / plddt_logits.shape[-1]
            plddt = (100.0 * (plddt_logits.softmax(dim=-1) * plddt_bins).sum(dim=-1)).cpu().numpy()

            pae_logits = sample.pae_logits
            assert pae_logits is not None  # the multimer TM head is always constructed
            pae = _predicted_aligned_error(pae_logits[0], self.model.config)
            ptm = alphafold2_predicted_tm_score(pae_logits[0], input.sequence_mask[0], self.model.config)
            iptm = alphafold2_predicted_interface_tm_score(
                pae_logits[0],
                input.chain_index[0],
                input.sequence_mask[0],
                self.model.config,
            )
            weights = self.model.config.heads.predicted_aligned_error
            ranking_score = weights.iptm_weight * iptm + weights.ptm_weight * ptm

            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "mean_plddt": Score(float(plddt.mean()), preferred="higher"),
                        "ptm": Score(ptm, preferred="higher"),
                        "iptm": Score(iptm, preferred="higher"),
                        "ranking_score": Score(ranking_score, preferred="higher"),
                    },
                    metrics={"plddt": Metric(values=plddt, axis=residue_axis)},
                    pair_metrics={
                        "pae": PairMetric(
                            values=pae,
                            row_axis=residue_axis,
                            column_axis=residue_axis,
                            unit="angstrom",
                        )
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))


@dataclass(eq=False)
class AlphaFold2Casp14Prediction:
    r"""
    One ranked prediction from the fixed five-model CASP14 ensemble.

    Attributes:
        model_name: Official checkpoint name, from ``"model_1"`` through
            ``"model_5"``.
        sample_index: Zero-based sample index within that checkpoint.
        prediction: Decoded AlphaFold 2 prediction.
        relaxation: Amber result when this ranked prediction was selected for
            relaxation.
    """

    model_name: str
    sample_index: int
    prediction: FoldPrediction
    relaxation: AlphaFold2RelaxationResult | None = None


@dataclass(eq=False)
class AlphaFold2Casp14Result:
    r"""
    Complete stable ranking from the fixed five-model CASP14 ensemble.

    Attributes:
        ranked: Predictions in descending mean-pLDDT order, with checkpoint and
            sample generation order breaking ties.
    """

    ranked: tuple[AlphaFold2Casp14Prediction, ...]

    @property
    def best(self) -> AlphaFold2Casp14Prediction:
        r"""
        Return the highest-ranked prediction.
        """

        return self.ranked[0]


class AlphaFold2Casp14Pipeline(CompositePipeline):
    r"""
    Fixed five-checkpoint AlphaFold 2 CASP14 inference and relaxation.

    Args:
        model_1: Ordinary eight-ensemble ``model_1`` monomer pipeline.
        model_2: Ordinary eight-ensemble ``model_2`` monomer pipeline.
        model_3: Ordinary eight-ensemble ``model_3`` monomer pipeline.
        model_4: Ordinary eight-ensemble ``model_4`` monomer pipeline.
        model_5: Ordinary eight-ensemble ``model_5`` monomer pipeline.
    """

    relaxation: AlphaFold2AmberRelaxation

    pair_layout = (
        CompositePair("model_1", "", AlphaFold2Pipeline),
        CompositePair("model_2", "model_2", AlphaFold2Pipeline),
        CompositePair("model_3", "model_3", AlphaFold2Pipeline),
        CompositePair("model_4", "model_4", AlphaFold2Pipeline),
        CompositePair("model_5", "model_5", AlphaFold2Pipeline),
    )

    def __init__(
        self,
        *,
        model_1: AlphaFold2Pipeline,
        model_2: AlphaFold2Pipeline,
        model_3: AlphaFold2Pipeline,
        model_4: AlphaFold2Pipeline,
        model_5: AlphaFold2Pipeline,
    ) -> None:
        super().__init__(
            {
                "model_1": model_1,
                "model_2": model_2,
                "model_3": model_3,
                "model_4": model_4,
                "model_5": model_5,
            }
        )
        self.relaxation = AlphaFold2AmberRelaxation()
        for model_name, pipeline in self._models_in_generation_order():
            if pipeline.model.config.num_ensembles != 8:
                raise ValueError(f"{model_name} must use the CASP14 eight-ensemble configuration")

    @property
    def model_1(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["model_1"])

    @property
    def model_2(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["model_2"])

    @property
    def model_3(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["model_3"])

    @property
    def model_4(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["model_4"])

    @property
    def model_5(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["model_5"])

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        models_to_relax: Literal["none", "best", "all"] = "best",
        platform: str = "CPU",
        platform_properties: Mapping[str, str] | None = None,
    ) -> AlphaFold2Casp14Result:
        r"""
        Predict, rank, and optionally relax the fixed CASP14 five-model ensemble.

        The biological input is lowered once through ``model_1``'s Processor.
        Every checkpoint consumes that same tensor input sequentially.

        Args:
            input: Inline sequence, FASTA path, protein sequence, molecular
                system, or structure topology containing one protein instance.
            msa: Optional monomer MSA value or alignment path.
            templates: Template structures with explicit target alignments, or a
                native template-artifact path.
            num_samples: Number of predictions generated by each checkpoint.
            seed: Optional MSA sampling seed, mutually exclusive with
                ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with
                ``seed``.
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses each Model configuration.
            models_to_relax: Whether Amber relaxation is applied to no
                prediction, only the best prediction, or every prediction.
            platform: Exact OpenMM platform name passed to Amber relaxation.
            platform_properties: Properties passed directly to the selected
                OpenMM platform.

        Returns:
            Every prediction in stable descending mean-pLDDT order, including
                checkpoint identity, sample identity, and any relaxation result.
        """

        if models_to_relax not in ("none", "best", "all"):
            raise ValueError("models_to_relax must be 'none', 'best', or 'all'")
        generator = resolve_generator(seed, generator, device=self.model_1.processor.device)
        model_input = self.model_1.prepare_fold(
            input,
            msa=msa,
            templates=templates,
        )

        generated: list[AlphaFold2Casp14Prediction] = []
        for model_name, pipeline in self._models_in_generation_order():
            output = pipeline.sample_fold(
                model_input,
                num_samples=num_samples,
                generator=generator,
                recycling_steps=recycling_steps,
            )
            result = pipeline.decode_fold(model_input, output)
            generated.extend(
                AlphaFold2Casp14Prediction(
                    model_name=model_name,
                    sample_index=sample_index,
                    prediction=prediction,
                )
                for sample_index, prediction in enumerate(result.predictions)
            )
        ranked = tuple(
            sorted(
                generated,
                key=lambda item: item.prediction.scores["mean_plddt"].value,
                reverse=True,
            )
        )

        if models_to_relax == "none":
            relax_indices: tuple[int, ...] = ()
        elif models_to_relax == "best":
            relax_indices = (0,)
        else:
            relax_indices = tuple(range(len(ranked)))

        relaxed = list(ranked)
        for index in relax_indices:
            candidate = relaxed[index]
            relaxed[index] = AlphaFold2Casp14Prediction(
                model_name=candidate.model_name,
                sample_index=candidate.sample_index,
                prediction=candidate.prediction,
                relaxation=self.relaxation.relax(
                    candidate.prediction.structure,
                    platform=platform,
                    platform_properties=platform_properties,
                ),
            )
        return AlphaFold2Casp14Result(ranked=tuple(relaxed))

    def _models_in_generation_order(self) -> tuple[tuple[str, AlphaFold2Pipeline], ...]:
        return (
            ("model_1", self.model_1),
            ("model_2", self.model_2),
            ("model_3", self.model_3),
            ("model_4", self.model_4),
            ("model_5", self.model_5),
        )


def _monomer_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""
    Resolve AlphaFold 2's inline-sequence and FASTA conveniences.
    """

    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            if classified.name.endswith(STRUCTURE_ARTIFACT_SUFFIX):
                structure = resolve_structure_input(classified)
                if not isinstance(structure, Structure):
                    raise TypeError("native structure artifacts must contain a Structure")
                return structure
            if not any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
                raise ValueError(
                    "AlphaFold 2 monomer cannot consume a coordinate-bearing structure path; use a FASTA path, "
                    "input.system, or templates="
                )
            named = [
                (record.id or "protein", PolymerSequence.from_one_letter(record.sequence.strip().upper()))
                for record in read_fasta_records(classified)
            ]
            if len(named) != 1:
                raise ValueError("AlphaFold 2 monomer requires exactly one PolymerSequence")
            entity_id, sequence = named[0]
            return MolecularSystem.from_entities(
                entities={entity_id: sequence},
                instances={entity_id: entity_id},
            )
        return monomer_system_from_sequences(PolymerSequence.from_one_letter(classified.strip().upper()))
    if isinstance(input, Structure):
        return input
    if isinstance(input, MolecularSystem):
        return input
    return monomer_system_from_sequences(input)


def _sample_generator(root: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    r"""Draw one independent per-sample generator from the root stream."""

    if root is None:
        return None
    sub_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=root, device=root.device).item())
    return torch.Generator(device=device).manual_seed(sub_seed)


def _component_axis(
    system: MolecularSystem,
    component_index: Tensor,
) -> tuple[tuple[str, int], ...]:
    return tuple(system.component_reference(component) for component in component_index.detach().cpu().tolist())


def _decode_atom37_structure(
    atom37_positions: Tensor,
    atom37_mask: Tensor,
    system: MolecularSystem,
    atom37_index: Tensor,
) -> Structure:
    system = system.cpu()  # type: ignore[attr-defined]
    atom37_index = atom37_index.detach().cpu()
    positions = atom37_positions.detach().to(device="cpu", dtype=torch.float32)
    mask = atom37_mask.detach().to(device="cpu", dtype=torch.bool) & (atom37_index >= 0)
    num_atoms = system.atom_symbol.numel()
    coordinates = torch.zeros(num_atoms, 3, dtype=torch.float32)
    coordinate_mask = torch.zeros(num_atoms, dtype=torch.bool)
    coordinates[atom37_index[mask]] = positions[mask]
    coordinate_mask[atom37_index[mask]] = True
    num_components = system.component_symbol.numel()
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=torch.zeros(num_atoms, dtype=torch.float32),
        occupancy_mask=torch.zeros(num_atoms, dtype=torch.bool),
        b_factor=torch.zeros(num_atoms, dtype=torch.float32),
        b_factor_mask=torch.zeros(num_atoms, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((num_components,), -1, dtype=torch.long),
        insertion_code=torch.full((num_components,), -1, dtype=torch.long),
        batch_size=[],
    )


def _predicted_aligned_error(
    logits: Tensor,
    config: AlphaFold2Config | AlphaFold2MultimerConfig,
) -> np.ndarray:
    r"""Return expected aligned error in angstrom for one AlphaFold 2 example."""

    logits = logits.detach()
    max_error = config.heads.predicted_aligned_error.max_error_bin
    breaks = torch.linspace(0.0, max_error, logits.shape[-1] - 1, device=logits.device)
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    return (logits.float().softmax(dim=-1) * centers).sum(dim=-1).cpu().numpy()


def _decode_monomer_outputs(
    input: AlphaFold2Input,
    output: Sequence[AlphaFold2Output],
    config: AlphaFold2Config,
) -> FoldResult:
    r"""Decode monomer-family raw outputs into generation-ordered predictions."""

    residue_axis = _component_axis(input.system, input.component_index[0])
    predictions: list[FoldPrediction] = []
    for sample in output:
        structure = _decode_atom37_structure(
            sample.final_atom_positions[0],
            sample.final_atom_mask[0],
            input.system,
            input.atom37_index[0],
        )
        plddt_logits = sample.plddt_logits[0].detach().float()
        plddt_bins = (
            torch.arange(
                plddt_logits.shape[-1],
                device=plddt_logits.device,
                dtype=plddt_logits.dtype,
            )
            + 0.5
        ) / plddt_logits.shape[-1]
        plddt = (100.0 * (plddt_logits.softmax(dim=-1) * plddt_bins).sum(dim=-1)).cpu().numpy()
        pae = None if sample.pae_logits is None else _predicted_aligned_error(sample.pae_logits[0], config)
        scores = {"mean_plddt": Score(float(plddt.mean()), preferred="higher")}
        metrics = {"plddt": Metric(values=plddt, axis=residue_axis)}
        pair_metrics: dict[str, PairMetric] = {}
        if sample.pae_logits is not None and pae is not None:
            scores["ptm"] = Score(
                alphafold2_predicted_tm_score(
                    sample.pae_logits[0],
                    input.sequence_mask[0],
                    config,
                ),
                preferred="higher",
            )
            pair_metrics["pae"] = PairMetric(
                values=pae,
                row_axis=residue_axis,
                column_axis=residue_axis,
                unit="angstrom",
            )
        predictions.append(
            FoldPrediction(
                structure=structure,
                scores=scores,
                metrics=metrics,
                pair_metrics=pair_metrics,
            )
        )
    return FoldResult(predictions=tuple(predictions))
