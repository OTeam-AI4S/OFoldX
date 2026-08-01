r"""
Released PXDesign backbone generation and target--binder design composition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal, cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ...data import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
)
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MsaCollection
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_PROTEIN,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    disjoint_union,
    instance_subgraph,
)
from ...data.structure import Structure
from ...io import read_msa, read_structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...results import FoldPrediction, Score
from ..alphafold2 import (
    AlphaFold2InitialGuessPipeline,
    AlphaFold2Pipeline,
)
from ..proteinmpnn import ProteinMpnnPipeline
from ..protenix import ProtenixPipeline, ProtenixStructureConditionPipeline
from .model import PxDesignInput, PxDesignModel
from .processor import PxDesignProcessor, to_binder


@dataclass(eq=False)
class PxDesignBackboneResult:
    r"""
    A generation-ordered collection of PxDesign candidates.

    Attributes:
        candidates: Decoded target-binder structures in generation order.
        placement_map: Realized output occurrences for every declared placement.
    """

    candidates: tuple[Structure, ...]
    placement_map: PlacementMap


class PxDesignBackbonePipeline(PairedPipeline[PxDesignModel, PxDesignProcessor]):
    r"""
    Independently callable PXDesign-d backbone generator.
    """

    model_architecture = "pxdesign"

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignBackboneResult:
        r"""
        Generate target--binder backbone candidates.

        Args:
            input: Optional target system, observed structure, or local structure path.
            placement: Topology containing target spans and one fixed-length
                generated protein chain.
            conditions: Target hotspot labels.
            num_candidates: Number of backbone candidates.
            num_sampling_steps: Number of EDM sampling transitions.
            sample_chunk_size: Maximum number of candidates denoised together.
                ``None`` processes every candidate together.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            Coordinate-bearing structures in sampling order.
        """

        if isinstance(input, str | PathLike):
            input = read_structure(input)
        model_input = self.processor(input, placement=placement, conditions=conditions)
        generator = resolve_generator(seed, generator, device=model_input.reference_positions.device)
        output = self.model.sample(
            model_input,
            num_samples=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
        source_system = input.system if isinstance(input, Structure) else input
        placement_map, source_component_index = _placement_projection(
            placement,
            model_input.system,
            source_system,
        )
        source_structure = input if isinstance(input, Structure) else None
        return PxDesignBackboneResult(
            candidates=_decode_backbones(
                output.coordinates,
                model_input,
                source_structure,
                source_component_index,
            ),
            placement_map=placement_map,
        )


@dataclass(frozen=True)
class PxDesignFilterResult:
    r"""
    Exact released filter decisions for one PXDesign candidate.

    Attributes:
        af2_easy: Whether every AF2 easy threshold passes.
        af2_strict: Whether every AF2 strict threshold passes.
        complex_rmsd: Whether AF2 complex-to-design CA RMSD is below 4 Å.
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
        sequence_structure: Cropped target--binder structure after ProteinMPNN grafting.
        mpnn_scores: ProteinMPNN candidate scores.
        af2_complex: AF2-IG target--binder prediction.
        af2_binder: Ordinary AF2 unbound-binder prediction.
        protenix_screening: Two-step Extended Protenix prediction, when evaluated.
        protenix_final: Twenty-step Extended Protenix prediction, when selected.
        scores: Stable scalar identities used by filters and ranking.
        screening_filters: Exact Extended filter booleans that determined
            whether the 20-step rerun was executed.
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
        r"""
        Final Protenix, AF2 complex, or original sequence structure.
        """

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
        target: Full uncropped target retained independently of generation crops.
        candidates: Every candidate in generation order.
        ranked_indices: Generation indices in stable released rank order.
        target_prediction: Extended full-target Protenix prediction.
        target_rmsd: Extended predicted-to-input full-target CA RMSD.
        placement_map: Realized target and binder occurrences.
    """

    mode: Literal["preview", "extended"]
    target: Structure
    candidates: tuple[PxDesignCandidate, ...]
    ranked_indices: tuple[int, ...]
    placement_map: PlacementMap
    target_prediction: FoldPrediction | None = None
    target_rmsd: Score | None = None

    @property
    def ranked_candidates(self) -> tuple[PxDesignCandidate, ...]:
        r"""
        Return every candidate in released rank order.
        """

        return tuple(self.candidates[index] for index in self.ranked_indices)

    @property
    def best(self) -> PxDesignCandidate:
        r"""
        Return the first candidate under the released canonical ranking.
        """

        return self.ranked_candidates[0]


class PxDesignPipeline(CompositePipeline):
    r"""
    Released six-pair PXDesign Preview and Extended composition.

    The repository owns fixed PXDesign-d, ProteinMPNN ``v_48_020``, AF2-IG
    monomer/pTM, ordinary AF2 monomer, Protenix base v0.5, and
    structure-conditioned Protenix mini pairs.
    """

    pair_layout = (
        CompositePair("", "pxdesign", "backbone", None, wrap=PxDesignBackbonePipeline),
        CompositePair(
            "proteinmpnn",
            "proteinmpnn",
            "proteinmpnn",
            None,
            wrap=ProteinMpnnPipeline,
        ),
        CompositePair(
            "af2_complex",
            "alphafold2_initial_guess",
            "af2_complex",
            None,
            wrap=AlphaFold2InitialGuessPipeline,
        ),
        CompositePair(
            "af2_binder",
            "alphafold2",
            "af2_binder",
            None,
            wrap=AlphaFold2Pipeline,
        ),
        CompositePair(
            "protenix",
            "protenix",
            "protenix",
            None,
            wrap=ProtenixPipeline,
        ),
        CompositePair(
            "protenix_structure",
            "protenix_structure_condition",
            "protenix_structure",
            None,
            wrap=ProtenixStructureConditionPipeline,
        ),
    )

    def __init__(
        self,
        *,
        backbone: PxDesignBackbonePipeline,
        proteinmpnn: ProteinMpnnPipeline,
        af2_complex: AlphaFold2InitialGuessPipeline,
        af2_binder: AlphaFold2Pipeline,
        protenix: ProtenixPipeline,
        protenix_structure: ProtenixStructureConditionPipeline,
    ) -> None:
        self.backbone = backbone
        self.proteinmpnn = proteinmpnn
        self.af2_complex = af2_complex
        self.af2_binder = af2_binder
        self.protenix = protenix
        self.protenix_structure = protenix_structure
        self._validate_components()

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        mode: Literal["preview", "extended"] = "preview",
        num_candidates: int | None = None,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignResult:
        r"""
        Run the released PXDesign Preview or Extended operation in memory.

        Args:
            input: Full target Structure or local structure path. Placement
                crops affect backbone generation only.
            placement: Target crops followed by one fixed-length protein binder.
            conditions: Target hotspot conditions for PXDesign-d.
            msa: Full-target MSA value, collection, or alignment path used by
                Extended target evaluation and the Protenix-base branch.
            mode: ``"preview"`` evaluates with AF2; ``"extended"`` also applies
                released Protenix target routing and evaluation.
            num_candidates: Number of generated candidates. ``None`` uses 100 for
                Preview and 500 for Extended; every generated candidate is returned.
            sample_chunk_size: Maximum PXDesign-d samples denoised together.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: Random generator advanced by all child operations.

        Returns:
            Every generated and evaluated candidate, plus stable rank indices.
        """

        if mode not in ("preview", "extended"):
            raise ValueError(f"unknown PXDesign mode: {mode!r}")
        if mode == "extended" and msa is None:
            raise ValueError("PXDesign Extended requires the full-target MSA")
        if num_candidates is None:
            num_candidates = 100 if mode == "preview" else 500
        generator = resolve_generator(seed, generator, device=self.backbone.processor.device)
        if isinstance(input, str | PathLike):
            input = read_structure(input)

        _, targets = to_binder(placement)
        if not targets:
            raise ValueError("PXDesign Preview and Extended require a protein target")
        target_instance_ids = tuple(instance_id for instance_id, _ in targets)
        full_target = _substructure(input, target_instance_ids)
        if isinstance(msa, str | PathLike):
            protein_entities = [
                full_target.system.entity_id(entity)
                for entity in range(full_target.system.entity_symbol.numel())
                if int(full_target.system.entity_type[entity].item()) == ENTITY_PROTEIN
            ]
            if len(protein_entities) != 1:
                raise ValueError("an aligned FASTA MSA is unambiguous only for one protein target entity")
            msa = read_msa(msa, entity_id=protein_entities[0])
        generated = self.backbone.design(
            input,
            placement=placement,
            conditions=conditions,
            num_candidates=num_candidates,
            num_sampling_steps=400,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )

        target_prediction: FoldPrediction | None = None
        target_rmsd: Score | None = None
        use_protenix_base = False
        if mode == "extended":
            target_prediction = self.protenix.fold(
                full_target,
                msa=msa,
                num_samples=1,
                recycling_steps=4,
                num_sampling_steps=2,
                churn_rate=0.0,
                step_scale=1.0,
                generator=generator,
            ).predictions[0]
            target_rmsd_value = _refitted_permuted_complex_rmsd(
                target_prediction.structure,
                full_target,
            )
            target_rmsd = Score(
                target_rmsd_value,
                unit="angstrom",
                preferred="lower",
            )
            use_protenix_base = target_rmsd_value < 2.0

        candidates = [
            self._evaluate_candidate(
                generated_candidate,
                full_target=full_target,
                target_instance_ids=target_instance_ids,
                msa=msa,
                mode=mode,
                use_protenix_base=use_protenix_base,
                generator=generator,
                generation_index=generation_index,
            )
            for generation_index, generated_candidate in enumerate(generated.candidates)
        ]
        if mode == "extended":
            for candidate in candidates:
                screening_filters = cast(
                    PxDesignFilterResult,
                    candidate.screening_filters,
                )
                if not screening_filters.protenix_basic:
                    continue
                evaluation_structure = _full_complex(
                    full_target,
                    candidate.sequence_structure,
                )
                final = self._protenix_fold(
                    evaluation_structure,
                    full_target=full_target,
                    msa=msa,
                    use_protenix_base=use_protenix_base,
                    num_sampling_steps=20,
                    generator=generator,
                )
                candidate.protenix_final = final
                binder_instance_id = _binder_instance_id(
                    candidate.sequence_structure.system,
                )
                final_scores, protenix_basic, protenix_strict = _protenix_decision(
                    final,
                    evaluation_structure,
                    binder_instance_id,
                )
                candidate.scores.update(
                    {
                        **final_scores,
                        **{
                            name.replace("protenix_", "protenix_final_", 1): score
                            for name, score in final_scores.items()
                        },
                    }
                )
                candidate.filters = PxDesignFilterResult(
                    af2_easy=candidate.filters.af2_easy,
                    af2_strict=candidate.filters.af2_strict,
                    complex_rmsd=candidate.filters.complex_rmsd,
                    protenix_basic=protenix_basic,
                    protenix_strict=protenix_strict,
                )

            for candidate in candidates:
                candidate.bucket = _extended_bucket(candidate.filters)
                pass_af2 = candidate.bucket in (1, 2, 3, 4, 7, 8)
                if candidate.protenix_final is not None:
                    candidate.chosen_source = "protenix"
                elif pass_af2:
                    candidate.chosen_source = "af2"
                else:
                    candidate.chosen_source = "original"

        ranked_indices = _rank_preview(candidates) if mode == "preview" else _rank_extended(candidates)
        for rank, candidate_index in enumerate(ranked_indices, start=1):
            candidates[candidate_index].rank = rank

        return PxDesignResult(
            mode=mode,
            target=full_target,
            candidates=tuple(candidates),
            ranked_indices=ranked_indices,
            placement_map=generated.placement_map,
            target_prediction=target_prediction,
            target_rmsd=target_rmsd,
        )

    def _evaluate_candidate(
        self,
        original_backbone: Structure,
        *,
        full_target: Structure,
        target_instance_ids: tuple[str, ...],
        msa: MSA | MsaCollection | None,
        mode: Literal["preview", "extended"],
        use_protenix_base: bool,
        generator: torch.Generator | None,
        generation_index: int,
    ) -> PxDesignCandidate:
        binder_instance_id = _binder_instance_id(original_backbone.system)
        binder_references = _protein_references(
            original_backbone.system,
            (binder_instance_id,),
        )
        sequence_result = self.proteinmpnn.design(
            original_backbone,
            positions=binder_references,
            num_candidates=1,
            temperature=0.0001,
            global_omit_tokens=("C", "X"),
            generator=generator,
        )
        sequence_candidate = sequence_result.candidates[0]
        sequence_structure = _structure_with_system(
            original_backbone,
            sequence_candidate.system,
        )

        af2_complex = self.af2_complex.fold(
            sequence_structure,
            binder_instance_id=binder_instance_id,
            num_samples=1,
            recycling_steps=4,
            generator=generator,
        ).predictions[0]
        binder_structure = _substructure(sequence_structure, (binder_instance_id,))
        af2_binder = self.af2_binder.fold(
            binder_structure,
            num_samples=1,
            recycling_steps=4,
            generator=generator,
        ).predictions[0]

        af2_complex_rmsd = _refitted_permuted_complex_rmsd(
            af2_complex.structure,
            sequence_structure,
        )
        bound_unbound_rmsd = _ca_rmsd(
            af2_complex.structure,
            af2_binder.structure,
            binder_references,
        )
        binder_design_rmsd = _ca_rmsd(
            af2_binder.structure,
            sequence_structure,
            binder_references,
        )
        scores = {
            "af2_binder_mean_plddt": af2_complex.scores["binder_mean_plddt"],
            "af2_iptm": af2_complex.scores["iptm"],
            "af2_interface_pae": af2_complex.scores["interface_pae"],
            "af2_normalized_interface_pae": af2_complex.scores["normalized_interface_pae"],
            "af2_bound_unbound_binder_rmsd": Score(
                bound_unbound_rmsd,
                unit="angstrom",
                preferred="lower",
            ),
            "af2_binder_predicted_design_rmsd": Score(
                binder_design_rmsd,
                unit="angstrom",
                preferred="lower",
            ),
            "af2_complex_predicted_design_rmsd": Score(
                af2_complex_rmsd,
                unit="angstrom",
                preferred="lower",
            ),
        }
        plddt = scores["af2_binder_mean_plddt"].value / 100.0
        iptm = scores["af2_iptm"].value
        interface_pae = scores["af2_interface_pae"].value
        normalized_interface_pae = scores["af2_normalized_interface_pae"].value
        filters = PxDesignFilterResult(
            af2_easy=(plddt > 0.8 and iptm > 0.5 and normalized_interface_pae < 0.35 and bound_unbound_rmsd < 3.5),
            af2_strict=(plddt > 0.9 and interface_pae < 7.0 and binder_design_rmsd < 1.5),
            complex_rmsd=af2_complex_rmsd < 4.0,
        )

        protenix_screening = None
        if mode == "extended":
            evaluation_structure = _full_complex(
                full_target,
                sequence_structure,
            )
            protenix_screening = self._protenix_fold(
                evaluation_structure,
                full_target=full_target,
                msa=msa,
                use_protenix_base=use_protenix_base,
                num_sampling_steps=2,
                generator=generator,
            )
            protenix_scores, protenix_basic, protenix_strict = _protenix_decision(
                protenix_screening,
                evaluation_structure,
                binder_instance_id,
            )
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
                protenix_basic=protenix_basic,
                protenix_strict=protenix_strict,
            )

        bucket = _preview_bucket(filters) if mode == "preview" else _extended_bucket(filters)
        pass_af2 = bucket < 5 if mode == "preview" else bucket in (1, 2, 3, 4, 7, 8)
        chosen_source: Literal["original", "af2", "protenix"] = "af2" if pass_af2 else "original"
        return PxDesignCandidate(
            original_backbone=original_backbone,
            sequence_structure=sequence_structure,
            mpnn_scores=dict(sequence_candidate.scores),
            af2_complex=af2_complex,
            af2_binder=af2_binder,
            protenix_screening=protenix_screening,
            protenix_final=None,
            scores=scores,
            screening_filters=(
                PxDesignFilterResult(
                    af2_easy=filters.af2_easy,
                    af2_strict=filters.af2_strict,
                    complex_rmsd=filters.complex_rmsd,
                    protenix_basic=filters.protenix_basic,
                    protenix_strict=filters.protenix_strict,
                )
                if mode == "extended"
                else None
            ),
            filters=filters,
            bucket=bucket,
            rank=0,
            chosen_source=chosen_source,
            generation_index=generation_index,
        )

    def _protenix_fold(
        self,
        input: Structure,
        *,
        full_target: Structure,
        msa: MSA | MsaCollection | None,
        use_protenix_base: bool,
        num_sampling_steps: int,
        generator: torch.Generator | None,
    ) -> FoldPrediction:
        if use_protenix_base:
            return self.protenix.fold(
                input,
                msa=msa,
                num_samples=1,
                recycling_steps=4,
                num_sampling_steps=num_sampling_steps,
                churn_rate=0.0,
                step_scale=1.0,
                generator=generator,
            ).predictions[0]
        condition = _substructure(
            input,
            tuple(
                full_target.system.instance_id(instance)
                for instance in range(full_target.system.instance_symbol.numel())
            ),
        )
        return self.protenix_structure.fold(
            input,
            structure_condition=condition,
            msa=None,
            num_samples=1,
            recycling_steps=4,
            num_sampling_steps=num_sampling_steps,
            churn_rate=0.0,
            step_scale=1.0,
            generator=generator,
        ).predictions[0]


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
    af2_order = sorted(
        failed,
        key=lambda index: (
            candidates[index].scores["af2_interface_pae"].value,
            index,
        ),
    )
    protenix_order = sorted(
        failed,
        key=lambda index: (
            -candidates[index].scores["protenix_global_iptm"].value,
            index,
        ),
    )
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


def _binder_instance_id(
    system: MolecularSystem,
) -> str:
    binder = system.instance_symbol.numel() - 1
    entity = int(system.instance_entity[binder].item())
    if int(system.entity_type[entity].item()) != ENTITY_PROTEIN:
        raise ValueError("PXDesign's final instance must be its protein binder")
    return system.instance_id(binder)


def _protein_references(
    system: MolecularSystem,
    instance_ids: tuple[str, ...],
) -> tuple[ComponentReference, ...]:
    references: list[ComponentReference] = []
    for instance_id in instance_ids:
        instance = system.instance_index(instance_id)
        entity = int(system.instance_entity[instance].item())
        if int(system.entity_type[entity].item()) != ENTITY_PROTEIN:
            continue
        start = int(system.instance_component_ptr[instance].item())
        stop = int(system.instance_component_ptr[instance + 1].item())
        references.extend((instance_id, component - start) for component in range(start, stop))
    return tuple(references)


def _placement_projection(
    placement: PlacementSpec,
    system: MolecularSystem,
    source: MolecularSystem | None,
) -> tuple[PlacementMap, Tensor]:
    r"""
    Present the canonical graph through the declared placement identities.
    """

    placement_map: PlacementMap = {}
    source_component_index = torch.full(
        (system.component_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=system.component_symbol.device,
    )
    groups = tuple(placement.items())
    if len(groups) != system.instance_symbol.numel():
        raise ValueError("PXDesign placement groups must match output instances")

    for output_instance, (group, values) in enumerate(groups):
        instance_id = system.instance_id(output_instance)
        output_start = int(system.instance_component_ptr[output_instance].item())
        realized: dict[str, tuple[ComponentReference, ...]] = {}
        if output_instance == len(groups) - 1:
            placement_id, value = next(iter(values.items()))
            if not isinstance(value, GeneratedSegment):
                raise ValueError("PXDesign's final placement must be its binder")
            realized[placement_id] = tuple((instance_id, component) for component in range(value.max_length))
        else:
            assert source is not None
            spans = cast(dict[str, ExistingSpan], values)
            selected = sorted({component for span in spans.values() for component in range(span.start, span.stop)})
            local = {source_component: output_component for output_component, source_component in enumerate(selected)}
            source_instance = source.instance_index(next(iter(spans.values())).instance_id)
            source_start = int(source.instance_component_ptr[source_instance].item())
            for output_component, source_component in enumerate(selected):
                source_component_index[output_start + output_component] = source_start + source_component
            for placement_id, span in spans.items():
                realized[placement_id] = tuple(
                    (instance_id, local[component]) for component in range(span.start, span.stop)
                )
        placement_map[group] = realized
    return placement_map, source_component_index


def _decode_backbones(
    coordinates: Tensor,
    input: PxDesignInput,
    source: Structure | None,
    source_component_index: Tensor,
) -> tuple[Structure, ...]:
    r"""
    Scatter model atoms onto the complete canonical atom axis.
    """

    system = input.system.cpu()  # type: ignore[attr-defined]
    atom_index = input.atom_index[0].detach().cpu()
    source_component_index = source_component_index.cpu()
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()

    source_symbols: tuple[str, ...] = ()
    residue_id = torch.full(
        (component_count,),
        -1,
        dtype=torch.int64,
    )
    insertion_code = residue_id.clone()
    standard_component_id = residue_id.clone()
    if source is not None:
        source = source.cpu()  # type: ignore[attr-defined]
        source_symbols = source.source_symbols
        selected = source_component_index >= 0
        residue_id[selected] = source.residue_id[source_component_index[selected]]
        insertion_code[selected] = source.insertion_code[source_component_index[selected]]
        for output_index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
            source_index = int(source_component_index[output_index])
            source_id = source.system.symbols[int(source.system.component_symbol[source_index])]
            result_id = system.symbols[int(system.component_symbol[output_index])]
            if source_id == result_id:
                standard_component_id[output_index] = source.standard_component_id[source_index]

    candidates = []
    for sample in coordinates[0]:
        values = torch.zeros(atom_count, 3, dtype=torch.float32)
        values[atom_index] = sample.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
        coordinate_mask[atom_index] = True
        candidates.append(
            Structure(  # type: ignore[call-arg]
                system=system,
                coordinates=values,
                coordinate_mask=coordinate_mask,
                occupancy=torch.zeros(atom_count),
                occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
                b_factor=torch.zeros(atom_count),
                b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
                source_symbols=source_symbols,
                residue_id=residue_id,
                insertion_code=insertion_code,
                standard_component_id=standard_component_id,
                batch_size=[],
            )
        )
    return tuple(candidates)


def _substructure(
    structure: Structure,
    instance_ids: tuple[str, ...],
) -> Structure:
    instance_mask = torch.zeros(
        structure.system.instance_symbol.numel(),
        dtype=torch.bool,
        device=structure.system.instance_symbol.device,
    )
    for instance_id in instance_ids:
        instance_mask[structure.system.instance_index(instance_id)] = True
    system, component_to_new, atom_to_new = instance_subgraph(
        structure.system,
        instance_mask,
    )
    component_index = torch.nonzero(
        component_to_new >= 0,
        as_tuple=True,
    )[0]
    atom_index = torch.nonzero(atom_to_new >= 0, as_tuple=True)[0]
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=structure.coordinates[atom_index],
        coordinate_mask=structure.coordinate_mask[atom_index],
        occupancy=structure.occupancy[atom_index],
        occupancy_mask=structure.occupancy_mask[atom_index],
        b_factor=structure.b_factor[atom_index],
        b_factor_mask=structure.b_factor_mask[atom_index],
        source_symbols=structure.source_symbols,
        residue_id=structure.residue_id[component_index],
        insertion_code=structure.insertion_code[component_index],
        standard_component_id=structure.standard_component_id[component_index],
        batch_size=[],
    )


def _structure_with_system(
    structure: Structure,
    system: MolecularSystem,
) -> Structure:
    if system.component_symbol.numel() != structure.system.component_symbol.numel():
        raise ValueError("ProteinMPNN sequence realization must preserve the component axis")
    coordinates = torch.zeros(
        system.atom_symbol.numel(),
        3,
        dtype=structure.coordinates.dtype,
        device=structure.coordinates.device,
    )
    coordinate_mask = torch.zeros(
        system.atom_symbol.numel(),
        dtype=torch.bool,
        device=structure.coordinate_mask.device,
    )
    occupancy = torch.zeros(
        system.atom_symbol.numel(),
        dtype=structure.occupancy.dtype,
        device=structure.occupancy.device,
    )
    occupancy_mask = torch.zeros_like(coordinate_mask)
    b_factor = torch.zeros(
        system.atom_symbol.numel(),
        dtype=structure.b_factor.dtype,
        device=structure.b_factor.device,
    )
    b_factor_mask = torch.zeros_like(coordinate_mask)
    for output_atom in range(system.atom_symbol.numel()):
        site = system.atom_site(output_atom)
        try:
            source_atom = structure.system.atom_index(site)
        except KeyError:
            continue
        coordinates[output_atom] = structure.coordinates[source_atom]
        coordinate_mask[output_atom] = structure.coordinate_mask[source_atom]
        occupancy[output_atom] = structure.occupancy[source_atom]
        occupancy_mask[output_atom] = structure.occupancy_mask[source_atom]
        b_factor[output_atom] = structure.b_factor[source_atom]
        b_factor_mask[output_atom] = structure.b_factor_mask[source_atom]
    standard_component_id = structure.standard_component_id.clone()
    for component in range(system.component_symbol.numel()):
        source_id = structure.system.symbols[int(structure.system.component_symbol[component])]
        result_id = system.symbols[int(system.component_symbol[component])]
        if source_id != result_id:
            standard_component_id[component] = -1
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=structure.source_symbols,
        residue_id=structure.residue_id,
        insertion_code=structure.insertion_code,
        standard_component_id=standard_component_id,
        batch_size=[],
    )


def _full_complex(
    full_target: Structure,
    design: Structure,
) -> Structure:
    binder = _substructure(
        design,
        (_binder_instance_id(design.system),),
    )
    system, _, _ = disjoint_union((full_target.system, binder.system))
    source_symbols = (
        *full_target.source_symbols,
        *binder.source_symbols,
    )
    source_offset = len(full_target.source_symbols)
    binder_residue_id = torch.where(
        binder.residue_id < 0,
        binder.residue_id,
        binder.residue_id + source_offset,
    )
    binder_insertion_code = torch.where(
        binder.insertion_code < 0,
        binder.insertion_code,
        binder.insertion_code + source_offset,
    )
    binder_standard_component_id = torch.where(
        binder.standard_component_id < 0,
        binder.standard_component_id,
        binder.standard_component_id + source_offset,
    )
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=torch.cat((full_target.coordinates, binder.coordinates)),
        coordinate_mask=torch.cat((full_target.coordinate_mask, binder.coordinate_mask)),
        occupancy=torch.cat((full_target.occupancy, binder.occupancy)),
        occupancy_mask=torch.cat((full_target.occupancy_mask, binder.occupancy_mask)),
        b_factor=torch.cat((full_target.b_factor, binder.b_factor)),
        b_factor_mask=torch.cat((full_target.b_factor_mask, binder.b_factor_mask)),
        source_symbols=source_symbols,
        residue_id=torch.cat((full_target.residue_id, binder_residue_id)),
        insertion_code=torch.cat((full_target.insertion_code, binder_insertion_code)),
        standard_component_id=torch.cat((full_target.standard_component_id, binder_standard_component_id)),
        batch_size=[],
    )


def _refitted_permuted_complex_rmsd(
    prediction: Structure,
    reference: Structure,
) -> float:
    r"""
    Refit global Kabsch RMSD after the released identical-sequence assignment.
    """

    assignment, _ = _identical_sequence_chain_assignment(
        prediction,
        reference,
    )
    _, prediction_coordinates = _protein_chain_data(prediction)
    _, reference_coordinates = _protein_chain_data(reference)
    prediction_pairs: list[np.ndarray] = []
    reference_pairs: list[np.ndarray] = []
    for prediction_index, reference_index in enumerate(assignment):
        prediction_pair, reference_pair = _common_ca_coordinates(
            prediction_coordinates[prediction_index],
            reference_coordinates[reference_index],
        )
        prediction_pairs.append(prediction_pair)
        reference_pairs.append(reference_pair)
    return _coordinate_rmsd(
        np.concatenate(prediction_pairs, axis=0),
        np.concatenate(reference_pairs, axis=0),
    )


def _anchored_permuted_complex_rmsd(
    prediction: Structure,
    reference: Structure,
) -> float:
    r"""
    Return the anchored RMSD retained by the released Protenix evaluator.
    """

    _, rmsd = _identical_sequence_chain_assignment(
        prediction,
        reference,
    )
    return rmsd


def _identical_sequence_chain_assignment(
    prediction: Structure,
    reference: Structure,
) -> tuple[tuple[int, ...], float]:
    r"""
    Select the released anchored/Hungarian identical-sequence chain assignment.

    One generated representative from each sequence group is anchored in turn
    to every matching reference chain. The resulting rigid transform fixes a
    same-sequence Hungarian assignment over the complete complex. Only scalar
    metrics consume this assignment; both Structures retain canonical instance
    identity.
    """

    prediction_sequences, prediction_coordinates = _protein_chain_data(prediction)
    reference_sequences, reference_coordinates = _protein_chain_data(reference)

    prediction_groups: dict[str, list[int]] = defaultdict(list)
    reference_groups: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(prediction_sequences):
        prediction_groups[sequence].append(index)
    for index, sequence in enumerate(reference_sequences):
        reference_groups[sequence].append(index)

    prediction_group_sizes = {sequence: len(indices) for sequence, indices in prediction_groups.items()}
    reference_group_sizes = {sequence: len(indices) for sequence, indices in reference_groups.items()}
    if prediction_group_sizes != reference_group_sizes:
        raise ValueError("PXDesign complex RMSD requires matching protein sequence groups")

    best_rmsd = float("inf")
    best_assignment: tuple[int, ...] | None = None
    for sequence in sorted(prediction_groups):
        prediction_anchor = prediction_groups[sequence][0]
        for reference_anchor in reference_groups[sequence]:
            prediction_pair, reference_pair = _common_ca_coordinates(
                prediction_coordinates[prediction_anchor],
                reference_coordinates[reference_anchor],
            )
            if len(prediction_pair) < 3:
                continue
            rotation, prediction_center, reference_center = _fit_ca_transform(
                prediction_pair,
                reference_pair,
            )

            count = len(prediction_sequences)
            cost = np.full((count, count), 1e9, dtype=np.float64)
            for prediction_index, coordinates in enumerate(prediction_coordinates):
                for reference_index in reference_groups[prediction_sequences[prediction_index]]:
                    prediction_pair, reference_pair = _common_ca_coordinates(
                        coordinates,
                        reference_coordinates[reference_index],
                    )
                    if len(prediction_pair) == 0:
                        continue
                    transformed = (prediction_pair - prediction_center) @ rotation + reference_center
                    difference = transformed - reference_pair
                    cost[prediction_index, reference_index] = float(
                        np.sqrt(np.square(difference).sum() / len(difference))
                    )

            cost[prediction_anchor, :] = 1e9
            cost[:, reference_anchor] = 1e9
            cost[prediction_anchor, reference_anchor] = 0.0
            prediction_assignment, reference_assignment = linear_sum_assignment(cost)

            sum_squared = 0.0
            num_coordinates = 0
            for prediction_index, reference_index in zip(
                prediction_assignment,
                reference_assignment,
                strict=True,
            ):
                prediction_pair, reference_pair = _common_ca_coordinates(
                    prediction_coordinates[prediction_index],
                    reference_coordinates[reference_index],
                )
                if len(prediction_pair) == 0:
                    continue
                transformed = (prediction_pair - prediction_center) @ rotation + reference_center
                difference = transformed - reference_pair
                sum_squared += float(np.square(difference).sum())
                num_coordinates += len(difference)
            rmsd = float(np.sqrt(sum_squared / num_coordinates))
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                assignment = [0] * count
                for prediction_index, reference_index in zip(
                    prediction_assignment,
                    reference_assignment,
                    strict=True,
                ):
                    assignment[int(prediction_index)] = int(reference_index)
                best_assignment = tuple(assignment)
    if best_assignment is None:
        raise RuntimeError("PXDesign found no valid identical-sequence chain assignment")
    return best_assignment, best_rmsd


def _protein_chain_data(
    structure: Structure,
) -> tuple[list[str], list[dict[int, np.ndarray]]]:
    sequences: list[str] = []
    coordinates: list[dict[int, np.ndarray]] = []
    for instance in range(structure.system.instance_symbol.numel()):
        entity = int(structure.system.instance_entity[instance].item())
        if int(structure.system.entity_type[entity].item()) != ENTITY_PROTEIN:
            continue
        instance_id = structure.system.instance_id(instance)
        component_start = int(structure.system.instance_component_ptr[instance].item())
        component_stop = int(structure.system.instance_component_ptr[instance + 1].item())
        sequences.append(
            "".join(
                AMINO_ACID_THREE_TO_ONE.get(
                    structure.system.symbols[int(structure.system.component_symbol[component].item())],
                    "X",
                )
                for component in range(component_start, component_stop)
            )
        )
        chain_coordinates: dict[int, np.ndarray] = {}
        for component_index in range(component_stop - component_start):
            try:
                atom = structure.system.atom_index(AtomSite(instance_id, component_index, "CA"))
            except KeyError:
                continue
            if not bool(structure.coordinate_mask[atom]):
                continue
            chain_coordinates[component_index] = (
                structure.coordinates[atom].detach().cpu().numpy().astype(np.float64, copy=False)
            )
        coordinates.append(chain_coordinates)
    return sequences, coordinates


def _common_ca_coordinates(
    prediction: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(prediction.keys() & reference.keys())
    return (
        np.asarray([prediction[index] for index in common], dtype=np.float64).reshape(-1, 3),
        np.asarray([reference[index] for index in common], dtype=np.float64).reshape(-1, 3),
    )


def _ca_rmsd(
    prediction: Structure,
    reference: Structure,
    residues: tuple[ComponentReference, ...],
) -> float:
    prediction_coordinates = []
    reference_coordinates = []
    for instance_id, component_index in residues:
        site = AtomSite(instance_id, component_index, "CA")
        try:
            prediction_atom = prediction.system.atom_index(site)
            reference_atom = reference.system.atom_index(site)
        except KeyError:
            continue
        if not bool(prediction.coordinate_mask[prediction_atom]):
            continue
        if not bool(reference.coordinate_mask[reference_atom]):
            continue
        prediction_coordinates.append(prediction.coordinates[prediction_atom].detach().cpu().numpy())
        reference_coordinates.append(reference.coordinates[reference_atom].detach().cpu().numpy())
    return _coordinate_rmsd(
        np.asarray(prediction_coordinates, dtype=np.float64).reshape(-1, 3),
        np.asarray(reference_coordinates, dtype=np.float64).reshape(-1, 3),
    )


def _coordinate_rmsd(
    prediction_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
) -> float:
    rotation, prediction_center, reference_center = _fit_ca_transform(
        prediction_coordinates,
        reference_coordinates,
    )
    difference = (prediction_coordinates - prediction_center) @ rotation + reference_center - reference_coordinates
    return float(np.sqrt(np.square(difference).sum(axis=-1).mean()))


def _fit_ca_transform(
    prediction_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction_coordinates.shape != reference_coordinates.shape:
        raise ValueError("PXDesign RMSD requires paired CA coordinate arrays")
    if len(prediction_coordinates) < 3:
        raise ValueError("PXDesign RMSD requires at least three paired CA coordinates")
    prediction_center = prediction_coordinates.mean(axis=0)
    reference_center = reference_coordinates.mean(axis=0)
    prediction_centered = prediction_coordinates - prediction_center
    reference_centered = reference_coordinates - reference_center
    left, _, right = np.linalg.svd(prediction_centered.T @ reference_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    return rotation, prediction_center, reference_center


def _chain_metric(
    prediction: FoldPrediction,
    name: str,
    instance_id: str,
) -> float:
    metric = prediction.metrics[name]
    return float(metric.values[metric.axis.index(instance_id)])


def _protenix_decision(
    prediction: FoldPrediction,
    design: Structure,
    binder_instance_id: str,
) -> tuple[dict[str, Score], bool, bool]:
    binder_iptm = _chain_metric(
        prediction,
        "chain_iptm",
        binder_instance_id,
    )
    binder_ptm = _chain_metric(
        prediction,
        "chain_ptm",
        binder_instance_id,
    )
    complex_rmsd = _anchored_permuted_complex_rmsd(
        prediction.structure,
        design,
    )
    scores = {
        "protenix_binder_iptm": Score(
            binder_iptm,
            preferred="higher",
        ),
        "protenix_binder_ptm": Score(
            binder_ptm,
            preferred="higher",
        ),
        "protenix_global_iptm": prediction.scores["iptm"],
        "protenix_complex_predicted_design_rmsd": Score(
            complex_rmsd,
            unit="angstrom",
            preferred="lower",
        ),
    }
    basic = binder_iptm > 0.8 and binder_ptm > 0.8 and complex_rmsd < 2.5
    strict = binder_iptm > 0.85 and binder_ptm > 0.88 and complex_rmsd < 2.5
    return scores, basic, strict
