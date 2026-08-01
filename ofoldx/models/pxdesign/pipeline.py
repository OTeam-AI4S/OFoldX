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
PXDesign backbone generation and the released Preview/Extended composition.

The backbone Pipeline generates the placement's binder coordinates and decodes them onto the prepared problem (the
generated components in the target's own frame, as GENERATED). The composition assigns one ProteinMPNN sequence per
backbone, evaluates the complex and the unbound binder with AF2, routes Extended evaluation through Protenix, and
filters and ranks every candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ...chemistry.definitions import ComponentLibrary
from ...data.molecular import AtomSite, ComponentRef, EntityType, MolecularSystem, MonomerType
from ...data.projection import project_instances
from ...data.structure import CoordinateEvidence, Structure
from ...data.tokenization import CLASS_MONOMER, token_classes
from ...errors import RequestRef, StructureRef, fail
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...provenance import known
from ...results import FoldPrediction, Score
from ..alphafold2 import AlphaFold2InitialGuessPipeline, AlphaFold2Pipeline
from ..conditions import Hotspot, PlacementSpec
from ..design import PositionSelection
from ..preparation import PreparedInput, decode_coordinates
from ..proteinmpnn import ProteinMpnnPipeline
from ..protenix import ProtenixPipeline, ProtenixStructureConditionPipeline
from .model import PxDesignInput, PxDesignModel, PxDesignSamplingOutput
from .processor import PxDesignProcessor, assemble, to_binder


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def generated_components(prepared: PreparedInput[PxDesignInput]) -> NDArray[np.int64]:
    r"""The components of the prepared problem whose coordinates the model generates."""
    tokens = prepared.features.design_features[0, :, 0].detach().cpu().numpy() > 0
    return np.unique(prepared.context.slots.layout.token_component[tokens])


def _instances(structure: Structure, instance_ids: Sequence[str]) -> Structure:
    r"""The named instances of ``structure``, whole."""
    system = structure.system
    found = system.instance_index(list(instance_ids))
    if (found < 0).any():
        fail("request_invalid_positions", RequestRef("instance_ids", instance_ids[int(np.argmin(found))]))
    keep = np.zeros(len(system.instance_symbol), dtype=bool)
    keep[found] = True
    return project_instances(structure, keep)[0]


def _instance_of(system: MolecularSystem, component: int) -> str:
    r"""The instance symbol of one component."""
    instance = _owner(system.instance_component_ptr)[component]
    return str(system.text("instance", system.instance_symbol[instance : instance + 1])[0])


@dataclass(eq=False)
class PxDesignBackboneResult:
    r"""
    Generated backbones in generation order.

    Attributes:
        candidates: The decoded problems, one per generated sample.
        unwritten: Per candidate, the scope atoms the model did not place (a generated side chain).
    """

    candidates: tuple[Structure, ...]
    unwritten: tuple[NDArray[np.int64], ...]


class PxDesignBackbonePipeline(PairedPipeline[PxDesignModel, PxDesignProcessor]):
    r"""
    Independently callable PXDesign-d backbone generator.
    """

    task = "design"

    model_architecture = "pxdesign"

    def prepare(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        library: ComponentLibrary | None = None,
    ) -> PreparedInput[PxDesignInput]:
        r"""
        Prepare one target and placement for PXDesign-d backbone sampling.

        Notes:
            Component grouping follows its resolved biological class and declared representative atom. Conditioning
            retains every admitted heavy atom, including Se/SE names and elements. Observation masks control geometry,
            not
            chemical membership. No MSE-to-MET substitution is applied, which can change checkpoint feature
            distributions.

            Generated N/CA/C/O backbones have unknown side-chain identity, electronic state and hydrogen inventory.
            Prepared
            atom/component maps identify the exact physical copies used by reconstruction. Coordinate and condition
            inputs
            use explicit availability masks: excluded coordinates have zero derivatives; available values must be finite
            and representable in the model dtype. Inputs without such masks require complete finite fields.
            Normalization
            retains the input carrier.
        """
        return self.processor.prepare_input(structure, placement=placement, conditions=conditions, library=library)

    @torch.inference_mode()
    def forward(
        self,
        input: PxDesignInput,
        *,
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        generator: torch.Generator | None = None,
    ) -> PxDesignSamplingOutput:
        r"""Sample backbone coordinates for the prepared problem."""
        return self.model.sample(
            input,
            num_samples=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )

    def postprocess(
        self, prepared: PreparedInput[PxDesignInput], output: PxDesignSamplingOutput
    ) -> PxDesignBackboneResult:
        r"""
        Decode every generated sample onto the prepared problem without executing the model.

        The model denoises every slot, the target's own atoms among them; only the generated components are the task's
        range, so only their slots are written, as GENERATED, and the target keeps its own coordinates and provenance.

        Notes:
            Reconstruction reuses the prepared atom/component maps. After sequence design, evaluation consumes the
            inverse-folding candidate's final ``target`` directly. Matching labels or atom names alone cannot recover
            physical
            identity after a sequence edit.
        """

        samples = output.coordinates[0].detach().cpu().double().numpy()
        atom = prepared.context.slots.atom
        generated = (atom >= 0) & prepared.context.scope.atoms[np.maximum(atom, 0)]
        decoded = [
            decode_coordinates(prepared, sample, valid=generated, evidence=CoordinateEvidence.GENERATED)
            for sample in samples
        ]
        return PxDesignBackboneResult(
            candidates=tuple(structure for structure, _ in decoded),
            unwritten=tuple(report.invalidated_unwritten for _, report in decoded),
        )

    def design(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignBackboneResult:
        r"""
        Generate target--binder backbone candidates.

        Args:
            structure: The target the placement's spans are taken from; ``None`` generates without a target.
            placement: Target spans followed by one fixed-length generated protein chain.
            conditions: Target hotspot labels.
            library: The definitions the generated residues and their backbone template are read from (default the
                configured artifact).
            num_candidates: Number of backbone candidates.
            num_sampling_steps: Number of EDM sampling transitions.
            sample_chunk_size: Maximum number of candidates denoised together. ``None`` denoises every candidate
                together.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            The decoded problems in sampling order.
        """

        prepared = self.prepare(structure, placement=placement, conditions=conditions, library=library)
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=resolve_generator(seed, generator, device=self.processor.device),
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)


@dataclass(frozen=True)
class PxDesignFilterResult:
    r"""
    Exact released filter decisions for one PXDesign candidate.

    Attributes:
        af2_easy: Whether every AF2 easy threshold passes.
        af2_strict: Whether every AF2 strict threshold passes.
        complex_rmsd: Whether AF2 complex-to-design CA RMSD is below 4 angstrom.
        protenix_basic: Whether every Extended Protenix basic threshold passes.
        protenix_strict: Whether every Extended Protenix strict threshold passes.
    """

    af2_easy: bool
    af2_strict: bool
    complex_rmsd: bool
    protenix_basic: bool | None = None
    protenix_strict: bool | None = None


@dataclass(eq=False)
class PxDesignCandidate:
    r"""
    One fully evaluated PXDesign target--binder candidate.

    Attributes:
        original_backbone: Generated PXDesign-d backbone before sequence assignment.
        sequence_structure: The target--binder problem after ProteinMPNN design.
        mpnn_scores: ProteinMPNN candidate scores.
        af2_complex: AF2-IG target--binder prediction.
        af2_binder: Ordinary AF2 unbound-binder prediction.
        protenix_screening: Two-step Extended Protenix prediction, when evaluated.
        protenix_final: Twenty-step Extended Protenix prediction, when selected.
        scores: Stable scalar identities used by filters and ranking.
        screening_filters: Exact Extended filter booleans that determined whether the 20-step rerun was executed.
        filters: Exact released filter booleans.
        bucket: One-based released selection bucket.
        rank: One-based stable rank over every generated candidate.
        chosen_source: Source of ``chosen_structure``.
        generation_index: Zero-based backbone generation order.
    """

    original_backbone: Structure
    sequence_structure: Structure
    mpnn_scores: dict[str, Score]
    af2_complex: FoldPrediction
    af2_binder: FoldPrediction
    protenix_screening: FoldPrediction | None
    protenix_final: FoldPrediction | None
    scores: dict[str, Score]
    screening_filters: PxDesignFilterResult | None
    filters: PxDesignFilterResult
    bucket: int
    rank: int
    chosen_source: Literal["original", "af2", "protenix"]
    generation_index: int

    @property
    def chosen_structure(self) -> Structure:
        r"""Final Protenix, AF2 complex, or original sequence structure."""
        if self.chosen_source == "original":
            return self.sequence_structure
        if self.chosen_source == "af2":
            return self.af2_complex.structure
        return cast(FoldPrediction, self.protenix_final).structure


@dataclass(eq=False)
class PxDesignResult:
    r"""
    Complete Preview or Extended PXDesign operation result.

    Attributes:
        mode: Released ``"preview"`` or ``"extended"`` operation.
        target: Full uncropped target, retained independently of the generation spans.
        candidates: Every candidate in generation order.
        ranked_indices: Generation indices in stable released rank order.
        target_prediction: Extended full-target Protenix prediction.
        target_rmsd: Extended predicted-to-input full-target CA RMSD.
    """

    mode: Literal["preview", "extended"]
    target: Structure
    candidates: tuple[PxDesignCandidate, ...]
    ranked_indices: tuple[int, ...]
    target_prediction: FoldPrediction | None = None
    target_rmsd: Score | None = None

    @property
    def ranked_candidates(self) -> tuple[PxDesignCandidate, ...]:
        r"""Every candidate in released rank order."""
        return tuple(self.candidates[index] for index in self.ranked_indices)

    @property
    def best(self) -> PxDesignCandidate:
        r"""The first candidate under the released canonical ranking."""
        return self.ranked_candidates[0]


class PxDesignPipelineInput(NamedTuple):
    r"""The prepared generation problem, the full target, its MSA and the operation."""

    backbone: PreparedInput[PxDesignInput]
    target: Structure
    target_instance_ids: tuple[str, ...]
    msa: Mapping[str, MSA] | MsaCollection | None
    mode: Literal["preview", "extended"]
    num_candidates: int
    library: ComponentLibrary | None
    generator: torch.Generator | None


class PxDesignPipelineOutput(NamedTuple):
    r"""Evaluated candidates and the full-target prediction before final ranking."""

    candidates: tuple[PxDesignCandidate, ...]
    target_prediction: FoldPrediction | None
    target_rmsd: Score | None


class PxDesignPipeline(CompositePipeline):
    r"""
    Released six-pair PXDesign Preview and Extended composition.

    The repository owns fixed PXDesign-d, ProteinMPNN ``v_48_020``, AF2-IG monomer/pTM, ordinary AF2 monomer,
    Protenix base v0.5, and structure-conditioned Protenix mini pairs.

    The six child paths are ``backbone/``, ``proteinmpnn/``, ``af2_complex/``,
    ``af2_binder/``, ``protenix/`` and ``protenix_structure/``, respectively.
    Preview executes the first four children; Extended additionally uses both
    Protenix children. Complete repositories retain all six components.
    """

    task = "design"

    component_layout = (
        ComponentSpec("backbone", "backbone", PxDesignBackbonePipeline),
        ComponentSpec("inverse_folder", "proteinmpnn", ProteinMpnnPipeline),
        ComponentSpec("complex_folder", "af2_complex", AlphaFold2InitialGuessPipeline),
        ComponentSpec("binder_folder", "af2_binder", AlphaFold2Pipeline),
        ComponentSpec("protenix_folder", "protenix", ProtenixPipeline),
        ComponentSpec("structure_conditioned_folder", "protenix_structure", ProtenixStructureConditionPipeline),
    )

    def __init__(
        self,
        *,
        backbone: PxDesignBackbonePipeline,
        inverse_folder: ProteinMpnnPipeline,
        complex_folder: AlphaFold2InitialGuessPipeline,
        binder_folder: AlphaFold2Pipeline,
        protenix_folder: ProtenixPipeline,
        structure_conditioned_folder: ProtenixStructureConditionPipeline,
    ) -> None:
        super().__init__(
            {
                "backbone": backbone,
                "inverse_folder": inverse_folder,
                "complex_folder": complex_folder,
                "binder_folder": binder_folder,
                "protenix_folder": protenix_folder,
                "structure_conditioned_folder": structure_conditioned_folder,
            }
        )

    @property
    def backbone(self) -> PxDesignBackbonePipeline:
        return cast(PxDesignBackbonePipeline, self.components["backbone"])

    @property
    def inverse_folder(self) -> ProteinMpnnPipeline:
        return cast(ProteinMpnnPipeline, self.components["inverse_folder"])

    @property
    def complex_folder(self) -> AlphaFold2InitialGuessPipeline:
        return cast(AlphaFold2InitialGuessPipeline, self.components["complex_folder"])

    @property
    def binder_folder(self) -> AlphaFold2Pipeline:
        return cast(AlphaFold2Pipeline, self.components["binder_folder"])

    @property
    def protenix_folder(self) -> ProtenixPipeline:
        return cast(ProtenixPipeline, self.components["protenix_folder"])

    @property
    def structure_conditioned_folder(self) -> ProtenixStructureConditionPipeline:
        return cast(ProtenixStructureConditionPipeline, self.components["structure_conditioned_folder"])

    def prepare(
        self,
        structure: Structure,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        mode: Literal["preview", "extended"] = "preview",
        num_candidates: int | None = None,
        library: ComponentLibrary | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignPipelineInput:
        r"""Prepare the generation problem, the full target and the target-bound MSA."""
        if mode not in ("preview", "extended"):
            fail("value_invalid", RequestRef("mode", str(mode)))
        if mode == "extended" and msa is None:
            fail("value_invalid", RequestRef("msa", "None"), mode=mode)
        _, targets = to_binder(placement)
        if not targets:
            fail("value_invalid", RequestRef("placement", "no target span"), mode=mode)
        instance_ids = tuple(instance_id for instance_id, _ in targets)
        return PxDesignPipelineInput(
            backbone=self.backbone.prepare(structure, placement=placement, conditions=conditions, library=library),
            target=_instances(structure, instance_ids),
            target_instance_ids=instance_ids,
            msa=msa,
            mode=mode,
            num_candidates=(100 if mode == "preview" else 500) if num_candidates is None else num_candidates,
            library=library,
            generator=resolve_generator(seed, generator, device=self.backbone.processor.device),
        )

    def forward(self, input: PxDesignPipelineInput, *, sample_chunk_size: int | None = 10) -> PxDesignPipelineOutput:
        r"""
        Generate and evaluate every candidate in the released draw order.

        Notes:
            Extended evaluation reruns every basic screening pass, including strict passes, for 20 diffusion
            transitions. Returned structures come from the rerun, not the two-step screen.

            Target routing and AF2 comparisons refit globally after chemically qualified chain assignment;
            Protenix retains anchored-assignment RMSD. Candidate and target must describe the same canonical
            atom-address domain. Complete chemical evidence must agree, including explicit negative relations,
            stereochemical ownership, and each independent inventory and connection closure. All canonical atoms,
            including hydrogens and atoms without coordinates, constrain an exchange. Unresolved chemistry stays
            at its original address, and external sites stay fixed. This conservative policy need not find all
            simultaneous permutations of crosslinked chains.

            Only CA coordinates available in both structures rank qualified assignments. Anchors require at least
            three noncollinear paired sites and a uniquely resolved proper float64 Kabsch rotation; planar rank-two
            anchors are allowed. Infeasible assignments are skipped. Missing orientation information and unsupported
            numerical ranges raise distinct typed errors. Returned structures retain their instance identities;
            numerical alignment capability does not establish chemical or physical identity.
        """
        generator = input.generator
        output = self.backbone.forward(
            input.backbone.features,
            num_candidates=input.num_candidates,
            num_sampling_steps=400,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        generated = self.backbone.postprocess(input.backbone, output)
        binder = generated_components(input.backbone)

        target_prediction: FoldPrediction | None = None
        target_rmsd: Score | None = None
        use_protenix_base = False
        if input.mode == "extended":
            target_prediction = self.protenix_folder.fold(
                input.target,
                msa=input.msa,
                num_samples=1,
                num_recycles=4,
                num_sampling_steps=2,
                churn_rate=0.0,
                step_scale=1.0,
                generator=generator,
            ).predictions[0]
            value = _refitted_complex_rmsd(target_prediction.structure, input.target)
            target_rmsd = Score(value, unit="angstrom", preferred="lower")
            use_protenix_base = value < 2.0

        candidates = [
            self._evaluate_candidate(
                backbone,
                binder=binder,
                input=input,
                use_protenix_base=use_protenix_base,
                generation_index=index,
            )
            for index, backbone in enumerate(generated.candidates)
        ]
        if input.mode == "extended":
            for candidate in candidates:
                self._rerun_protenix(candidate, input=input, use_protenix_base=use_protenix_base)
        return PxDesignPipelineOutput(tuple(candidates), target_prediction, target_rmsd)

    def postprocess(self, input: PxDesignPipelineInput, output: PxDesignPipelineOutput) -> PxDesignResult:
        r"""Bind final selection buckets, chosen structures and stable ranks."""
        candidates = list(output.candidates)
        if input.mode == "extended":
            for candidate in candidates:
                candidate.bucket = _extended_bucket(candidate.filters)
                if candidate.protenix_final is not None:
                    candidate.chosen_source = "protenix"
                elif candidate.bucket in (1, 2, 3, 4, 7, 8):
                    candidate.chosen_source = "af2"
                else:
                    candidate.chosen_source = "original"
        ranked = _rank_preview(candidates) if input.mode == "preview" else _rank_extended(candidates)
        for rank, index in enumerate(ranked, start=1):
            candidates[index].rank = rank
        return PxDesignResult(
            mode=input.mode,
            target=input.target,
            candidates=tuple(candidates),
            ranked_indices=ranked,
            target_prediction=output.target_prediction,
            target_rmsd=output.target_rmsd,
        )

    def design(
        self,
        structure: Structure,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        mode: Literal["preview", "extended"] = "preview",
        num_candidates: int | None = None,
        sample_chunk_size: int | None = 10,
        library: ComponentLibrary | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignResult:
        r"""
        Run the released PXDesign Preview or Extended operation in memory.

        Args:
            structure: The full target. The placement's spans affect backbone generation only.
            placement: Target spans followed by one fixed-length generated protein chain.
            conditions: Target hotspot conditions for PXDesign-d.
            msa: Full-target MSA pools or plan, used by Extended target evaluation and the Protenix-base branch.
            mode: ``"preview"`` evaluates with AF2; ``"extended"`` also applies the released Protenix target routing
                and evaluation.
            num_candidates: Number of generated candidates. ``None`` uses 100 for Preview and 500 for Extended;
                every generated candidate is returned.
            sample_chunk_size: Maximum PXDesign-d samples denoised together.
            library: The definitions generated residues and designed replacements are built from.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: Random generator advanced by all child operations.

        Returns:
            Every generated and evaluated candidate, plus stable rank indices.
        """

        prepared = self.prepare(
            structure,
            placement=placement,
            conditions=conditions,
            msa=msa,
            mode=mode,
            num_candidates=num_candidates,
            library=library,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, self.forward(prepared, sample_chunk_size=sample_chunk_size))

    __call__ = task_call(design)

    def _evaluate_candidate(
        self,
        original_backbone: Structure,
        *,
        binder: NDArray[np.int64],
        input: PxDesignPipelineInput,
        use_protenix_base: bool,
        generation_index: int,
    ) -> PxDesignCandidate:
        r"""One backbone's sequence, AF2 evaluation and, under Extended, its Protenix screening."""
        generator = input.generator
        binder_instance_id = _instance_of(original_backbone.system, int(binder[0]))
        sequence_result = self.inverse_folder.design(
            original_backbone,
            design=PositionSelection(indices=binder),
            num_candidates=1,
            temperature=0.0001,
            global_omit_tokens=("C", "X"),
            library=input.library,
            generator=generator,
        )
        sequence_candidate = sequence_result.candidates[0]
        sequence_structure = sequence_candidate.target
        residues = original_backbone.system.component_refs(binder)

        af2_complex = self.complex_folder.fold(
            sequence_structure,
            binder_instance_id=binder_instance_id,
            num_samples=1,
            num_recycles=4,
            generator=generator,
        ).predictions[0]
        af2_binder = self.binder_folder.fold(
            _instances(sequence_structure, (binder_instance_id,)),
            num_samples=1,
            num_recycles=4,
            generator=generator,
        ).predictions[0]

        complex_rmsd = _refitted_complex_rmsd(af2_complex.structure, sequence_structure)
        bound_unbound_rmsd = _ca_rmsd(af2_complex.structure, af2_binder.structure, residues)
        binder_design_rmsd = _ca_rmsd(af2_binder.structure, sequence_structure, residues)
        scores = {
            "af2_binder_mean_plddt": af2_complex.scores["binder_mean_plddt"],
            "af2_iptm": af2_complex.scores["iptm"],
            "af2_interface_pae": af2_complex.scores["interface_pae"],
            "af2_normalized_interface_pae": af2_complex.scores["normalized_interface_pae"],
            "af2_bound_unbound_binder_rmsd": Score(bound_unbound_rmsd, unit="angstrom", preferred="lower"),
            "af2_binder_predicted_design_rmsd": Score(binder_design_rmsd, unit="angstrom", preferred="lower"),
            "af2_complex_predicted_design_rmsd": Score(complex_rmsd, unit="angstrom", preferred="lower"),
        }
        plddt = scores["af2_binder_mean_plddt"].value / 100.0
        iptm = scores["af2_iptm"].value
        interface_pae = scores["af2_interface_pae"].value
        normalized_interface_pae = scores["af2_normalized_interface_pae"].value
        filters = PxDesignFilterResult(
            af2_easy=(plddt > 0.8 and iptm > 0.5 and normalized_interface_pae < 0.35 and bound_unbound_rmsd < 3.5),
            af2_strict=(plddt > 0.9 and interface_pae < 7.0 and binder_design_rmsd < 1.5),
            complex_rmsd=complex_rmsd < 4.0,
        )

        protenix_screening = None
        if input.mode == "extended":
            evaluation = _complex(input.target, _instances(sequence_structure, (binder_instance_id,)))
            protenix_screening = self._protenix_fold(
                evaluation,
                input=input,
                use_protenix_base=use_protenix_base,
                num_sampling_steps=2,
            )
            protenix_scores, basic, strict = _protenix_decision(protenix_screening, evaluation, binder_instance_id)
            scores.update(
                {
                    **protenix_scores,
                    **{
                        name.replace("protenix_", "protenix_screening_", 1): score
                        for name, score in protenix_scores.items()
                    },
                }
            )
            filters = PxDesignFilterResult(
                af2_easy=filters.af2_easy,
                af2_strict=filters.af2_strict,
                complex_rmsd=filters.complex_rmsd,
                protenix_basic=basic,
                protenix_strict=strict,
            )

        bucket = _preview_bucket(filters) if input.mode == "preview" else _extended_bucket(filters)
        pass_af2 = bucket < 5 if input.mode == "preview" else bucket in (1, 2, 3, 4, 7, 8)
        return PxDesignCandidate(
            original_backbone=original_backbone,
            sequence_structure=sequence_structure,
            mpnn_scores=dict(sequence_candidate.scores),
            af2_complex=af2_complex,
            af2_binder=af2_binder,
            protenix_screening=protenix_screening,
            protenix_final=None,
            scores=scores,
            screening_filters=filters if input.mode == "extended" else None,
            filters=filters,
            bucket=bucket,
            rank=0,
            chosen_source="af2" if pass_af2 else "original",
            generation_index=generation_index,
        )

    def _rerun_protenix(
        self, candidate: PxDesignCandidate, *, input: PxDesignPipelineInput, use_protenix_base: bool
    ) -> None:
        r"""The Extended 20-step rerun of a candidate that passed basic screening, and its final decision."""
        screening = cast(PxDesignFilterResult, candidate.screening_filters)
        if not screening.protenix_basic:
            return
        binder_instance_id = _instance_of(
            candidate.sequence_structure.system, int(generated_components(input.backbone)[0])
        )
        evaluation = _complex(input.target, _instances(candidate.sequence_structure, (binder_instance_id,)))
        final = self._protenix_fold(evaluation, input=input, use_protenix_base=use_protenix_base, num_sampling_steps=20)
        candidate.protenix_final = final
        scores, basic, strict = _protenix_decision(final, evaluation, binder_instance_id)
        candidate.scores.update(
            {**scores, **{name.replace("protenix_", "protenix_final_", 1): score for name, score in scores.items()}}
        )
        candidate.filters = PxDesignFilterResult(
            af2_easy=candidate.filters.af2_easy,
            af2_strict=candidate.filters.af2_strict,
            complex_rmsd=candidate.filters.complex_rmsd,
            protenix_basic=basic,
            protenix_strict=strict,
        )

    def _protenix_fold(
        self,
        evaluation: Structure,
        *,
        input: PxDesignPipelineInput,
        use_protenix_base: bool,
        num_sampling_steps: int,
    ) -> FoldPrediction:
        r"""The routed Extended Protenix evaluation: the base model with the MSA, else the conditioned mini model."""
        if use_protenix_base:
            return self.protenix_folder.fold(
                evaluation,
                msa=input.msa,
                num_samples=1,
                num_recycles=4,
                num_sampling_steps=num_sampling_steps,
                churn_rate=0.0,
                step_scale=1.0,
                generator=input.generator,
            ).predictions[0]
        return self.structure_conditioned_folder.fold(
            evaluation,
            structure_condition=_instances(evaluation, input.target_instance_ids),
            msa=None,
            num_samples=1,
            num_recycles=4,
            num_sampling_steps=num_sampling_steps,
            churn_rate=0.0,
            step_scale=1.0,
            generator=input.generator,
        ).predictions[0]


def _complex(target: Structure, binder: Structure) -> Structure:
    r"""The evaluation complex of the full target and one designed binder (``assemble``)."""
    return assemble((target, binder))[0]


def _preview_bucket(filters: PxDesignFilterResult) -> int:
    if filters.af2_strict and filters.af2_easy:
        return 1
    if filters.af2_strict:
        return 2
    if filters.af2_easy and filters.complex_rmsd:
        return 3
    if filters.af2_easy:
        return 4
    if filters.complex_rmsd:
        return 5
    return 6


def _extended_bucket(filters: PxDesignFilterResult) -> int:
    if filters.af2_strict and filters.protenix_strict:
        return 1
    if filters.af2_strict and filters.protenix_basic:
        return 2
    if filters.af2_easy and filters.protenix_strict:
        return 3
    if filters.af2_easy and filters.protenix_basic:
        return 4
    if filters.protenix_strict:
        return 5
    if filters.protenix_basic:
        return 6
    if filters.af2_strict:
        return 7
    if filters.af2_easy:
        return 8
    return 9


def _rank_preview(candidates: list[PxDesignCandidate]) -> tuple[int, ...]:
    r"""Buckets first, then the AF2 interface PAE, then generation order."""
    return tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].bucket,
                candidates[index].scores["af2_interface_pae"].value,
                index,
            ),
        )
    )


def _rank_extended(candidates: list[PxDesignCandidate]) -> tuple[int, ...]:
    r"""Successful buckets first by their own score, then the failures by the mean of both score ranks."""
    successful = [index for index, candidate in enumerate(candidates) if candidate.bucket < 9]
    successful.sort(
        key=lambda index: (
            candidates[index].bucket,
            (
                -candidates[index].scores["protenix_global_iptm"].value
                if candidates[index].bucket <= 6
                else candidates[index].scores["af2_interface_pae"].value
            ),
            index,
        )
    )
    failed = [index for index, candidate in enumerate(candidates) if candidate.bucket == 9]
    af2_order = sorted(failed, key=lambda index: (candidates[index].scores["af2_interface_pae"].value, index))
    protenix_order = sorted(failed, key=lambda index: (-candidates[index].scores["protenix_global_iptm"].value, index))
    af2_rank = {index: rank for rank, index in enumerate(af2_order, start=1)}
    protenix_rank = {index: rank for rank, index in enumerate(protenix_order, start=1)}
    failed.sort(
        key=lambda index: (
            0.5 * af2_rank[index] + 0.5 * protenix_rank[index],
            protenix_rank[index],
            af2_rank[index],
            index,
        )
    )
    return (*successful, *failed)


def _chain_metric(prediction: FoldPrediction, name: str, instance_id: str) -> float:
    metric = prediction.metrics[name]
    return float(metric.values[metric.axis.index(instance_id)])


def _protenix_decision(
    prediction: FoldPrediction, design: Structure, binder_instance_id: str
) -> tuple[dict[str, Score], bool, bool]:
    r"""The released Protenix scores of one evaluation and its basic and strict decisions."""
    binder_iptm = _chain_metric(prediction, "chain_mean_partner_iptm", binder_instance_id)
    binder_ptm = _chain_metric(prediction, "chain_ptm", binder_instance_id)
    complex_rmsd = _anchored_complex_rmsd(prediction.structure, design)
    scores = {
        "protenix_binder_iptm": Score(binder_iptm, preferred="higher"),
        "protenix_binder_ptm": Score(binder_ptm, preferred="higher"),
        "protenix_global_iptm": prediction.scores["iptm"],
        "protenix_complex_predicted_design_rmsd": Score(complex_rmsd, unit="angstrom", preferred="lower"),
    }
    basic = binder_iptm > 0.8 and binder_ptm > 0.8 and complex_rmsd < 2.5
    strict = binder_iptm > 0.85 and binder_ptm > 0.88 and complex_rmsd < 2.5
    return scores, basic, strict


# ===========================================================================
# Chain-permuted CA alignment (scipy for non-graph numerics).
# ===========================================================================


@dataclass(frozen=True)
class _Chain:
    r"""One protein chain's entity and its observed CA sites, by component ordinal within the instance."""

    entity: str
    position: NDArray[np.int64]
    coordinates: NDArray[np.float64]


def _chains(structure: Structure) -> tuple[_Chain, ...]:
    r"""
    Per instance holding an observed protein CA, its entity symbol and those CA sites in component order.

    The atom is read at protein polymer components only: a calcium ion and a free amino acid also carry an atom named
    ``CA``.
    """

    system = structure.system
    s = system.arrays()
    component = _owner(s["component_atom_ptr"])
    instance = _owner(s["instance_component_ptr"])
    names = system.text("atom", s["atom_symbol"])
    entity = s["instance_entity"]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    protein = (CLASS_MONOMER[token_classes(system).token] == MonomerType.PROTEIN) & polymer[instance]
    rows = np.flatnonzero((names == "CA") & structure.coordinate_mask & protein[component])
    owner = instance[component[rows]]
    entities = system.text("entity", s["entity_symbol"][s["instance_entity"]])
    chains = []
    for value in np.unique(owner).tolist():
        atoms = rows[owner == value]
        position = component[atoms] - s["instance_component_ptr"][value]
        chains.append(_Chain(str(entities[value]), position, structure.coordinates[atoms]))
    return tuple(chains)


def _paired(first: _Chain, second: _Chain) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The two chains' CA coordinates at the component positions both observe."""
    shared = np.intersect1d(first.position, second.position)
    return (
        first.coordinates[np.searchsorted(first.position, shared)],
        second.coordinates[np.searchsorted(second.position, shared)],
    )


def _rigid_fit(
    prediction: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    r"""
    The proper Kabsch rotation (row-vector convention) and both centroids, or ``None`` when the points fix no
    rotation: fewer than three pairs, or collinear points at float64 precision.
    """

    if len(prediction) < 3:
        return None
    centre, reference_centre = prediction.mean(axis=0), reference.mean(axis=0)
    left, singular, right = np.linalg.svd((prediction - centre).T @ (reference - reference_centre))
    if singular[1] <= 3 * np.finfo(np.float64).eps * singular[0]:
        return None
    correction = np.diag((1.0, 1.0, -1.0 if np.linalg.det(left @ right) < 0 else 1.0))
    return left @ correction @ right, centre, reference_centre


def _fitted_rmsd(
    prediction: NDArray[np.float64],
    reference: NDArray[np.float64],
    fit: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> float:
    rotation, centre, reference_centre = fit
    difference = (prediction - centre) @ rotation + reference_centre - reference
    return float(np.sqrt(np.square(difference).sum() / len(prediction)))


def _chain_assignment(prediction: Structure, reference: Structure) -> tuple[tuple[int, ...], float]:
    r"""
    The chain correspondence of equal entity with the lowest anchored CA RMSD, and that RMSD.

    Two chains may correspond when they are instances of one entity (the chain equivalence of the shared contract);
    the observed CA sites of a candidate anchor pair fix the rigid transform the other chains are scored under, and
    one assignment is solved per anchor (``scipy.optimize.linear_sum_assignment``).

    Raises:
        InvalidRequestError: ``value_invalid`` when the two values do not hold the same protein chains by entity.
        InsufficientInformationError: ``alignment_unresolved`` when no anchor resolves a transform under which every
            chain pairs.
    """

    predicted, referenced = _chains(prediction), _chains(reference)
    if sorted(chain.entity for chain in predicted) != sorted(chain.entity for chain in referenced):
        fail("value_invalid", RequestRef("reference", "a different protein chain domain"))
    count = len(predicted)
    qualified = np.array(
        [[first.entity == second.entity for second in referenced] for first in predicted], dtype=bool
    ).reshape(count, count)
    best: tuple[tuple[int, ...], float] | None = None
    for anchor in range(count):
        for reference_anchor in np.flatnonzero(qualified[anchor]).tolist():
            fit = _rigid_fit(*_paired(predicted[anchor], referenced[reference_anchor]))
            if fit is None:
                continue
            cost = np.full((count, count), np.inf)
            for row in range(count):
                for column in np.flatnonzero(qualified[row]).tolist():
                    pairs = _paired(predicted[row], referenced[column])
                    if len(pairs[0]):
                        cost[row, column] = _fitted_rmsd(*pairs, fit)
            cost[anchor, :], cost[:, reference_anchor] = np.inf, np.inf
            cost[anchor, reference_anchor] = 0.0
            if not np.isfinite(cost).any(axis=0).all() or not np.isfinite(cost).any(axis=1).all():
                continue
            rows, columns = linear_sum_assignment(cost)
            if not np.isfinite(cost[rows, columns]).all():
                continue
            squared, points, assignment = 0.0, 0, [0] * count
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                pairs = _paired(predicted[row], referenced[column])
                squared += _fitted_rmsd(*pairs, fit) ** 2 * len(pairs[0])
                points += len(pairs[0])
                assignment[row] = column
            rmsd = float(np.sqrt(squared / points))
            if best is None or rmsd < best[1]:
                best = (tuple(assignment), rmsd)
    if best is None:
        fail("alignment_unresolved", StructureRef(item="CA"), chains=count)
    return best


def _refitted_complex_rmsd(prediction: Structure, reference: Structure) -> float:
    r"""The CA RMSD of the best chain correspondence, refitted globally over its paired sites."""
    assignment, _ = _chain_assignment(prediction, reference)
    predicted, referenced = _chains(prediction), _chains(reference)
    pairs = [_paired(predicted[row], referenced[column]) for row, column in enumerate(assignment)]
    first = np.concatenate([pair[0] for pair in pairs], axis=0)
    second = np.concatenate([pair[1] for pair in pairs], axis=0)
    fit = _rigid_fit(first, second)
    if fit is None:
        fail("alignment_unresolved", StructureRef(item="CA"), points=len(first))
    return _fitted_rmsd(first, second, fit)


def _anchored_complex_rmsd(prediction: Structure, reference: Structure) -> float:
    r"""The anchored CA RMSD the released Protenix evaluator keeps (no global refit)."""
    return _chain_assignment(prediction, reference)[1]


def _ca_rmsd(prediction: Structure, reference: Structure, residues: Sequence[ComponentRef]) -> float:
    r"""The CA RMSD of the named residues, over the sites both values observe, after one global fit."""
    sites = [AtomSite(residue.instance, residue.component, "CA") for residue in residues]
    first, second = prediction.system.atom_index(sites), reference.system.atom_index(sites)
    shared = (first >= 0) & (second >= 0)
    shared[shared] = prediction.coordinate_mask[first[shared]] & reference.coordinate_mask[second[shared]]
    predicted, referenced = prediction.coordinates[first[shared]], reference.coordinates[second[shared]]
    fit = _rigid_fit(predicted, referenced)
    if fit is None:
        fail("alignment_unresolved", StructureRef(item="CA"), points=int(shared.sum()))
    return _fitted_rmsd(predicted, referenced, fit)
