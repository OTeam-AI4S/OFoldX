# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
AlphaFold 2 monomer, initial-guess, multimer and CASP14 folding Pipelines, written against the shared preparation
contract: ``prepare`` returns ``PreparedInput``; ``postprocess`` decodes every sample through
``decode_coordinates`` (evidence PREDICTED, scope every atom, new frame) and reports confidence on the prepared
residue axis.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

import numpy as np
import torch
from torch import Tensor

from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...metrics import categorical_expectation, plddt_from_logits
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, floating_numpy
from ...templates import Template
from ..preparation import PreparedInput, decode_coordinates
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
from .processor import AlphaFold2InitialGuessProcessor, AlphaFold2MultimerProcessor, AlphaFold2Processor
from .relax import AlphaFold2AmberRelaxation, AlphaFold2RelaxationResult

InputT = TypeVar("InputT")


def sample_alphafold2(
    input: InputT,
    model: Callable[[InputT, int | None, torch.Generator | None], AlphaFold2Output],
    *,
    device: torch.device,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
    num_recycles: int | None = None,
) -> tuple[AlphaFold2Output, ...]:
    r"""Execute independent AlphaFold 2 samples with one child stream per Model call."""

    num_samples = operator.index(num_samples)
    if num_samples < 1:
        fail("value_invalid", RequestRef("num_samples", str(num_samples)))
    return tuple(model(input, num_recycles, _sample_generator(generator, device)) for _ in range(num_samples))


class AlphaFold2Pipeline(PairedPipeline[AlphaFold2Model, AlphaFold2Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 2.
    """

    task = "fold"

    model_architecture = "alphafold2"

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
    ) -> PreparedInput[AlphaFold2Input]:
        r"""
        Prepare one protein chain and its evidence for AlphaFold 2 folding.

        Notes:
            The atom37/atom14 representation and pseudo-beta convention follow the model's residue vocabulary.
            Template observations and initial model coordinates have separate availability masks. Physical backbone
            frames
            require available, noncollinear defining atoms.
        """
        return self.processor.prepare_input(structure, msa=msa, templates=templates)

    def fold(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> FoldResult:
        r"""
        Predict the structure of one protein chain with AlphaFold 2.

        Args:
            structure: The target (read or constructed) whose residues form one protein chain.
            msa: Optional MSA pools by instance, or an exact plan.
            templates: Templates with explicit target alignments.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction. Mutually exclusive with ``seed``.
            num_recycles: Total network passes, including the final pass. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions decoded onto the target. Coordinates and PAE values are in angstroms;
            pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(structure, msa=msa, templates=templates)
        output = self.forward(
            prepared.features, num_samples=num_samples, generator=generator, num_recycles=num_recycles
        )
        return self.postprocess(prepared, output)

    @torch.inference_mode()
    def forward(
        self,
        input: AlphaFold2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent AlphaFold 2 forwards and return raw tensor outputs."""

        return sample_alphafold2(
            input,
            lambda model_input, recycles, rng: self.model(model_input, num_recycles=recycles, generator=rng),
            device=input.residue_type_ids.device,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
        )

    def postprocess(self, prepared: PreparedInput[AlphaFold2Input], output: Sequence[AlphaFold2Output]) -> FoldResult:
        r"""
        Decode raw AlphaFold 2 outputs without executing the model.

        Notes:
            Coordinates are placed on the prepared target using its atom mapping. Atoms outside the predicted
            representation, including hydrogens, ligands and terminal OXT, have unavailable coordinates in the new
            frame.
            Relaxation and linear-neighbor diagnostics require an actual closed standard linear protein graph; residue
            labels
            alone do not establish compatibility. Clash calculations exempt an admitted peptide C-N pair, with no
            general
            exemption for atom slot 5 or putative disulfides.
        """
        return _decode_monomer_outputs(prepared, output, self.model.config, backend=self.model.backend)

    __call__ = task_call(fold)


class AlphaFold2InitialGuessPipeline(PairedPipeline[AlphaFold2InitialGuessModel, AlphaFold2InitialGuessProcessor]):
    r"""
    Initial-guess target--binder evaluation with the monomer/pTM graph.
    """

    task = "fold"

    model_architecture = "alphafold2_initial_guess"

    def prepare(self, structure: Structure, *, binder_instance_id: str) -> PreparedInput[AlphaFold2InitialGuessInput]:
        r"""Prepare one target--binder structure for initial-guess folding."""
        return self.processor.prepare_input(structure, binder_instance_id=binder_instance_id)

    def fold(
        self,
        structure: Structure,
        *,
        binder_instance_id: str,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int = 4,
    ) -> FoldResult:
        r"""
        Predict a complex from its generated atom37 recycling state.

        Args:
            structure: Sequence-assigned generated target--binder complex.
            binder_instance_id: The protein instance treated as the binder.
            num_samples: Number of predictions.
            seed: Optional MSA sampling seed, mutually exclusive with ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with ``seed``.
            num_recycles: Total network passes, including the final pass.

        Returns:
            Complex predictions with binder pLDDT, pTM, iPTM, and raw and normalized interface PAE scores.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(structure, binder_instance_id=binder_instance_id)
        output = self.forward(
            prepared.features, num_samples=num_samples, generator=generator, num_recycles=num_recycles
        )
        return self.postprocess(prepared, output)

    @torch.inference_mode()
    def forward(
        self,
        input: AlphaFold2InitialGuessInput,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent initial-guess forwards and return raw outputs."""

        return sample_alphafold2(
            input,
            lambda model_input, recycles, rng: self.model(model_input, num_recycles=recycles, generator=rng),
            device=input.residue_type_ids.device,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
        )

    def postprocess(
        self, prepared: PreparedInput[AlphaFold2InitialGuessInput], output: Sequence[AlphaFold2Output]
    ) -> FoldResult:
        r"""Decode raw initial-guess outputs and add binder/interface scores."""

        input = prepared.features
        result = _decode_monomer_outputs(prepared, output, self.model.config, backend=self.model.backend)
        binder_mask = input.binder_mask[0].bool()
        target_mask = input.residue_mask[0].bool() & ~binder_mask
        interface_mask = (binder_mask[:, None] & target_mask[None, :]).cpu().numpy()
        binder = binder_mask.cpu().numpy()
        binder_scope = frozenset(ref for ref, selected in zip(_residue_axis(prepared), binder) if selected)
        for sample, prediction in zip(output, result.predictions, strict=True):
            binder_plddt = prediction.metrics["plddt"].values[binder]
            prediction.scores["binder_mean_plddt"] = Score(
                float(binder_plddt.mean()), preferred="higher", scope=binder_scope
            )
            if sample.pae_logits is None:
                continue
            pae = prediction.pair_metrics["pae"].values
            raw_interface_pae = float(((pae + pae.T) / 2.0)[interface_mask].mean())
            iptm = alphafold2_predicted_interface_tm_score(
                sample.pae_logits[0],
                input.chain_index[0],
                input.residue_mask[0],
                self.model.config,
                backend=self.model.backend,
            )
            prediction.scores["iptm"] = Score(iptm, preferred="higher")
            prediction.scores["interface_pae"] = Score(raw_interface_pae, unit="angstrom", preferred="lower")
            prediction.scores["normalized_interface_pae"] = Score(raw_interface_pae / 31.0, preferred="lower")
        return result

    __call__ = task_call(fold)


class AlphaFold2MultimerPipeline(PairedPipeline[AlphaFold2MultimerModel, AlphaFold2MultimerProcessor]):
    r"""
    Multimer structure prediction pipeline for AlphaFold-Multimer.
    """

    task = "fold"

    model_architecture = "alphafold2_multimer"

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
    ) -> PreparedInput[AlphaFold2MultimerInput]:
        r"""Prepare one protein assembly and its evidence for AlphaFold-Multimer."""
        return self.processor.prepare_input(structure, msa=msa, templates=templates)

    def fold(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> FoldResult:
        r"""
        Predict the structure of one protein assembly with AlphaFold-Multimer.

        Args:
            structure: The target (read or constructed); its protein chains are folded together.
            msa: Optional MSA pools by instance (paired by the one OFoldX rule), or an exact plan.
            templates: Templates with explicit target alignments.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction. Mutually exclusive with ``seed``.
            num_recycles: Total network passes, including the final pass. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions decoded onto the target. Coordinates and PAE values are in angstroms;
            pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(structure, msa=msa, templates=templates)
        output = self.forward(
            prepared.features, num_samples=num_samples, generator=generator, num_recycles=num_recycles
        )
        return self.postprocess(prepared, output)

    @torch.inference_mode()
    def forward(
        self,
        input: AlphaFold2MultimerInput,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> tuple[AlphaFold2Output, ...]:
        r"""Execute independent AlphaFold-Multimer forwards and return raw outputs."""

        return sample_alphafold2(
            input,
            lambda model_input, recycles, rng: self.model(model_input, num_recycles=recycles, generator=rng),
            device=input.residue_type_ids.device,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
        )

    def postprocess(
        self, prepared: PreparedInput[AlphaFold2MultimerInput], output: Sequence[AlphaFold2Output]
    ) -> FoldResult:
        r"""
        Decode raw AlphaFold-Multimer outputs without executing the model. Residue labels keep the prepared axis
        order (there is no target assembly against which to permute symmetric copies).
        """

        input = prepared.features
        config, backend = self.model.config, self.model.backend
        axis = _residue_axis(prepared)
        predictions = []
        for sample in output:
            plddt = floating_numpy(plddt_from_logits(sample.plddt_logits[0].detach()))
            pae_logits = sample.pae_logits
            assert pae_logits is not None  # the multimer TM head is always constructed
            pae = _predicted_aligned_error(pae_logits[0], config)
            ptm = alphafold2_predicted_tm_score(pae_logits[0], input.residue_mask[0], config, backend=backend)
            iptm = alphafold2_predicted_interface_tm_score(
                pae_logits[0], input.chain_index[0], input.residue_mask[0], config, backend=backend
            )
            weights = config.heads.predicted_aligned_error
            predictions.append(
                FoldPrediction(
                    structure=_decode_sample(prepared, sample),
                    scores={
                        "mean_plddt": Score(float(plddt.mean()), preferred="higher"),
                        "ptm": Score(ptm, preferred="higher"),
                        "iptm": Score(iptm, preferred="higher"),
                        "ranking_score": Score(
                            weights.iptm_weight * iptm + weights.ptm_weight * ptm, preferred="higher"
                        ),
                    },
                    metrics={"plddt": Metric(values=plddt, axis=axis)},
                    pair_metrics={"pae": PairMetric(values=pae, row_axis=axis, column_axis=axis, unit="angstrom")},
                )
            )
        return FoldResult(predictions=tuple(predictions))

    __call__ = task_call(fold)


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

    task = "fold"

    relaxation: AlphaFold2AmberRelaxation

    component_layout = (
        ComponentSpec("model_1", "model_1", AlphaFold2Pipeline),
        ComponentSpec("model_2", "model_2", AlphaFold2Pipeline),
        ComponentSpec("model_3", "model_3", AlphaFold2Pipeline),
        ComponentSpec("model_4", "model_4", AlphaFold2Pipeline),
        ComponentSpec("model_5", "model_5", AlphaFold2Pipeline),
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
                fail("model_capacity", RequestRef(model_name, "num_ensembles"), num_ensembles=8)

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
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        models_to_relax: Literal["none", "best", "all"] = "best",
        platform: str = "CPU",
        platform_properties: Mapping[str, str] | None = None,
    ) -> AlphaFold2Casp14Result:
        r"""
        Predict, rank, and optionally relax the fixed CASP14 five-model ensemble.

        The structure is prepared once through ``model_1``'s Processor; every checkpoint consumes that same
        prepared input sequentially.

        Args:
            structure: The target whose residues form one protein chain.
            msa: Optional MSA pools by instance, or an exact plan.
            templates: Templates with explicit target alignments.
            num_samples: Number of predictions generated by each checkpoint.
            seed: Optional MSA sampling seed, mutually exclusive with
                ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with
                ``seed``.
            num_recycles: Total network passes, including the final pass.
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
            fail("value_invalid", RequestRef("models_to_relax", str(models_to_relax)))
        generator = resolve_generator(seed, generator, device=self.model_1.processor.device)
        prepared = self.prepare(structure, msa=msa, templates=templates)
        output = self.forward(
            prepared,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
        )
        return self.postprocess(
            prepared,
            output,
            models_to_relax=models_to_relax,
            platform=platform,
            platform_properties=platform_properties,
        )

    __call__ = task_call(fold)

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
    ) -> PreparedInput[AlphaFold2Input]:
        r"""Prepare the common molecular input and evidence for all five checkpoints."""

        return self.model_1.prepare(structure, msa=msa, templates=templates)

    def forward(
        self,
        prepared: PreparedInput[AlphaFold2Input],
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
    ) -> tuple[AlphaFold2Casp14Prediction, ...]:
        r"""Generate checkpoint-labelled predictions in model and sample order."""

        generated: list[AlphaFold2Casp14Prediction] = []
        for model_name, pipeline in self._models_in_generation_order():
            output = pipeline.forward(
                prepared.features,
                num_samples=num_samples,
                generator=generator,
                num_recycles=num_recycles,
            )
            result = pipeline.postprocess(prepared, output)
            generated.extend(
                AlphaFold2Casp14Prediction(
                    model_name=model_name,
                    sample_index=sample_index,
                    prediction=prediction,
                )
                for sample_index, prediction in enumerate(result.predictions)
            )
        return tuple(generated)

    def postprocess(
        self,
        prepared: PreparedInput[AlphaFold2Input],
        output: Sequence[AlphaFold2Casp14Prediction],
        *,
        models_to_relax: Literal["none", "best", "all"] = "best",
        platform: str = "CPU",
        platform_properties: Mapping[str, str] | None = None,
    ) -> AlphaFold2Casp14Result:
        r"""Rank ensemble predictions and optionally relax their structures."""

        if models_to_relax not in ("none", "best", "all"):
            fail("value_invalid", RequestRef("models_to_relax", str(models_to_relax)))
        ranked = tuple(
            sorted(
                output,
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


def _sample_generator(root: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    r"""Draw one independent per-sample generator from the root stream."""

    if root is None:
        return None
    sub_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=root, device=root.device).item())
    return torch.Generator(device=device).manual_seed(sub_seed)


def _residue_axis(prepared: PreparedInput) -> tuple[ComponentRef, ...]:
    r"""The prepared residue axis: each token's component."""
    return prepared.context.structure.system.component_refs(prepared.context.slots.layout.token_component)


def _decode_sample(prepared: PreparedInput, sample: AlphaFold2Output) -> Structure:
    r"""One sample's atom37 coordinates decoded onto the target: the slots are each residue's atom37 names, valid
    where the predicted residue type has the atom."""
    coordinates = sample.final_atom_positions[0].detach().cpu().double().numpy().reshape(-1, 3)
    valid = sample.final_atom_mask[0].detach().cpu().numpy().reshape(-1) > 0
    return decode_coordinates(prepared, coordinates, valid=valid)[0]


def _predicted_aligned_error(logits: Tensor, config: AlphaFold2Config | AlphaFold2MultimerConfig) -> np.ndarray:
    r"""Return expected aligned error in angstrom for one AlphaFold 2 example."""

    logits = logits.detach()
    max_error = config.heads.predicted_aligned_error.max_error_bin
    breaks = torch.linspace(0.0, max_error, logits.shape[-1] - 1, device=logits.device, dtype=logits.dtype)
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    return floating_numpy(categorical_expectation(logits.softmax(dim=-1), centers))


def _decode_monomer_outputs(
    prepared: PreparedInput[AlphaFold2Input] | PreparedInput[AlphaFold2InitialGuessInput],
    output: Sequence[AlphaFold2Output],
    config: AlphaFold2Config,
    *,
    backend: str = "auto",
) -> FoldResult:
    r"""Decode monomer-family raw outputs into generation-ordered predictions."""

    residue_mask = prepared.features.residue_mask[0]
    axis = _residue_axis(prepared)
    predictions = []
    for sample in output:
        plddt = floating_numpy(plddt_from_logits(sample.plddt_logits[0].detach()))
        scores = {"mean_plddt": Score(float(plddt.mean()), preferred="higher")}
        pair_metrics: dict[str, PairMetric] = {}
        if sample.pae_logits is not None:
            ptm = alphafold2_predicted_tm_score(sample.pae_logits[0], residue_mask, config, backend=backend)
            scores["ptm"] = Score(ptm, preferred="higher")
            pae = _predicted_aligned_error(sample.pae_logits[0], config)
            pair_metrics["pae"] = PairMetric(values=pae, row_axis=axis, column_axis=axis, unit="angstrom")
        predictions.append(
            FoldPrediction(
                structure=_decode_sample(prepared, sample),
                scores=scores,
                metrics={"plddt": Metric(values=plddt, axis=axis)},
                pair_metrics=pair_metrics,
            )
        )
    return FoldResult(predictions=tuple(predictions))
