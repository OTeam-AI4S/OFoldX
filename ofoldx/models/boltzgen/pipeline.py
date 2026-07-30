r"""
Released BoltzGen structure generation, inverse folding, analysis, and selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import torch

from ...data.components import get_component
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.placement import SymbolicAtomSite, SymbolicResidueSite, resolve
from ...data.structure import Structure
from ...io.msa import read_msa
from ...io.structure import read_structure
from ...pipelines import CompositePair, CompositePipeline, resolve_generator
from ...results import FoldPrediction, ScoreResult
from ..boltz2 import Boltz2AffinityModel, Boltz2AffinityProcessor, Boltz2Model, Boltz2Processor
from .analysis import (
    AnalysisValue,
    BoltzGenDiversityMetrics,
    BoltzGenDiversitySample,
    CandidateFilter,
    amino_acid_fractions,
    assign_secondary_structure,
    binding_site_adherence_curve,
    candidate_filter_results,
    count_noncovalent_contacts,
    deduplicate_sequences,
    greedy_diversity_selection,
    largest_hydrophobic_patch_area,
    liability_metrics,
    rank_candidates,
    secondary_structure_fractions,
    sequence_hydrophobicity,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
from .model import (
    BoltzGenBinding,
    BoltzGenInput,
    BoltzGenInverseFoldingDesignOutput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingLayout,
    BoltzGenInverseFoldingModel,
    BoltzGenLayout,
    BoltzGenModel,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
    BoltzGenSymmetry,
)
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor

BoltzGenProtocol: TypeAlias = Literal[
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
]
BoltzGenDesignCheckpoint: TypeAlias = Literal["diverse", "adherence"]

_PROTOCOLS: tuple[BoltzGenProtocol, ...] = (
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
)


@dataclass(frozen=True)
class BoltzGenFilterOutcome:
    r"""
    Value and pass state for one candidate filter.
    """

    filter: CandidateFilter
    value: float
    passed: bool


@dataclass(frozen=True)
class BoltzGenDesignedChainAnalysis:
    r"""
    Sequence analysis for one protein instance containing designed residues.

    ``designed_sequence`` retains only exact designed occurrences;
    ``full_sequence`` retains the complete chain.  Instance identity keeps
    distinct chains from acquiring artificial sequence adjacency.
    """

    instance_id: str
    designed_sequence: str
    full_sequence: str
    hydrophobicity: float
    liabilities: dict[str, AnalysisValue]


@dataclass(frozen=True)
class BoltzGenCandidateAnalysis:
    r"""
    Named in-memory analysis values for one coherent selected refold.

    The RMSD fields compare the sequence-assigned original design with the
    selected whole-complex refold. ``design_folding_*`` compares the extracted
    design component with its independently selected design-only refold.
    ``first_designed_chain_*`` matches the released scalar aggregate; every
    designed protein instance remains available separately in ``designed_chains``.
    """

    designed_sequence: str
    first_designed_chain_sequence: str
    designed_chains: tuple[BoltzGenDesignedChainAnalysis, ...]
    num_design: int
    rmsd: float
    rmsd_design: float
    rmsd_target: float
    rmsd_design_target: float
    target_aligned_rmsd_design: float
    bb_rmsd: float
    bb_rmsd_design: float
    bb_rmsd_target: float
    bb_rmsd_design_target: float
    bb_target_aligned_rmsd_design: float
    design_folding_rmsd: float | None
    design_folding_bb_rmsd: float | None
    binding_site_under_3angstrom: float | None
    binding_site_under_4angstrom: float | None
    binding_site_under_5angstrom: float | None
    binding_site_under_6angstrom: float | None
    binding_site_under_7angstrom: float | None
    binding_site_under_8angstrom: float | None
    binding_site_under_9angstrom: float | None
    amino_acid_fractions: dict[str, float]
    secondary_structure_fractions: dict[str, float]
    design_hydrophobicity: float
    first_designed_chain_hydrophobicity: float
    first_designed_chain_liabilities: dict[str, AnalysisValue]
    original_hydrogen_bonds: int
    original_charged_pairs: int
    refolded_hydrogen_bonds: int
    refolded_charged_pairs: int
    target_delta_sasa_original: float
    target_sasa_unbound_original: float
    target_sasa_bound_original: float
    target_delta_sasa: float
    target_sasa_unbound: float
    target_sasa_bound: float
    largest_hydrophobic_patch: float | None


@dataclass(eq=False)
class BoltzGenDesignCandidate:
    r"""
    One sequence-assigned design and every retained evaluation artifact.

    Args:
        structure: Sequence-assigned original generated structure.
        source_checkpoint: Structure-design checkpoint that generated the backbone.
        design: Exact designed component occurrences.
        chain_design: Complete covalently propagated design component.
        binding_sites: Resolved positive binding-site residue or atom occurrences.
        sequence_symmetry: Inverse-folding sequence-identity groups.
        coordinate_trajectory: Current-coordinate sampler history, when requested.
        denoised_coordinate_trajectory: Denoised ``x0`` sampler history, when requested.
        whole_fold_predictions: Five whole-complex Boltz-2 predictions.
        whole_fold_selected_index: Coherent design-focused prediction index.
        design_fold_predictions: Optional five design-only predictions.
        design_fold_selected_index: Selected design-only prediction index.
        affinity_result: Optional complete Boltz-2 affinity result.
        analysis: Named family analysis values.
        filters: Individual filter outcomes.
    """

    structure: Structure
    source_checkpoint: BoltzGenDesignCheckpoint | None
    design: tuple[ComponentReference, ...]
    chain_design: tuple[ComponentReference, ...]
    binding_sites: tuple[ComponentReference | AtomSite, ...] = ()
    sequence_symmetry: tuple[tuple[ComponentReference, ...], ...] = ()
    coordinate_trajectory: torch.Tensor | None = None
    denoised_coordinate_trajectory: torch.Tensor | None = None
    whole_fold_predictions: tuple[FoldPrediction, ...] = ()
    whole_fold_selected_index: int | None = None
    design_fold_predictions: tuple[FoldPrediction, ...] = ()
    design_fold_selected_index: int | None = None
    affinity_result: ScoreResult | None = None
    analysis: BoltzGenCandidateAnalysis | None = None
    filters: tuple[BoltzGenFilterOutcome, ...] = ()

    @property
    def whole_refold(self) -> FoldPrediction | None:
        r"""
        Return the selected coherent whole-complex refold.
        """

        if self.whole_fold_selected_index is None:
            return None
        return self.whole_fold_predictions[self.whole_fold_selected_index]

    @property
    def design_refold(self) -> FoldPrediction | None:
        r"""
        Return the selected design-only refold.
        """

        if self.design_fold_selected_index is None:
            return None
        return self.design_fold_predictions[self.design_fold_selected_index]

    def diversity_sample(
        self,
        target_id: str,
        *,
        compute_lddt: bool = True,
    ) -> BoltzGenDiversitySample:
        r"""
        Return the named in-memory values used by diversity aggregation.

        ``compute_lddt=False`` omits the quadratic optional lDDT conditions
        without substituting another metric.
        """

        selected = self.whole_refold
        analysis = self.analysis
        if selected is None or analysis is None:
            raise ValueError("BoltzGen diversity requires completed folding and analysis")
        return BoltzGenDiversitySample(
            target_id=target_id,
            sequence=analysis.designed_sequence,
            original_coordinates=_design_ca_coordinates(
                self.structure,
                self.design,
            ),
            refolded_coordinates=_design_ca_coordinates(
                selected.structure,
                self.design,
            ),
            metrics=BoltzGenDiversityMetrics(
                rmsd=analysis.rmsd,
                rmsd_design=analysis.rmsd_design,
                target_aligned_rmsd_design=analysis.target_aligned_rmsd_design,
                lddt_intra_design=(
                    structure_design_lddt(
                        self.structure,
                        selected.structure,
                        design=self.design,
                    )
                    if compute_lddt
                    else None
                ),
                bb_rmsd=analysis.bb_rmsd,
                bb_rmsd_design=analysis.bb_rmsd_design,
                bb_target_aligned_rmsd_design=(analysis.bb_target_aligned_rmsd_design),
                bb_lddt_intra_design=(
                    structure_design_lddt(
                        self.structure,
                        selected.structure,
                        design=self.design,
                        backbone_only=True,
                    )
                    if compute_lddt
                    else None
                ),
                design_iptm=selected.scores["design_iptm"].value,
                min_interaction_pae=selected.scores["min_interaction_pae"].value,
            ),
        )


@dataclass(eq=False)
class BoltzGenDesignResult:
    r"""
    Complete analyzed candidates and stable selection views.

    ``ranked_indices``, ``top_indices``, and ``diversity_selected_indices`` refer
    to ``candidates``. Duplicate sequences remain in ``candidates`` for
    provenance, but only their earliest occurrence enters these selections.
    ``rank`` and ``quality`` also align to ``candidates``; excluded duplicates
    carry rank zero and NaN quality.
    """

    candidates: tuple[BoltzGenDesignCandidate, ...]
    ranked_indices: tuple[int, ...] = ()
    top_indices: tuple[int, ...] = ()
    diversity_selected_indices: tuple[int, ...] = ()
    rank: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    quality: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @property
    def ranked_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.ranked_indices)

    @property
    def top_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.top_indices)

    @property
    def diversity_selected_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.diversity_selected_indices)


class BoltzGenPipeline(CompositePipeline):
    r"""
    Released five-checkpoint BoltzGen design pipeline.

    The repository topology is fixed: diverse design at the root and adherence
    design, inverse folding, Boltz-2 folding, and Boltz-2 affinity under their
    named subdirectories.
    """

    pair_layout = (
        CompositePair("", "boltzgen", "diverse_model", "diverse_processor"),
        CompositePair("adherence", "boltzgen", "adherence_model", "adherence_processor"),
        CompositePair(
            "ifold",
            "boltzgen_ifold",
            "inverse_folding_model",
            "inverse_folding_processor",
        ),
        CompositePair("fold", "boltz2", "fold_model", "fold_processor"),
        CompositePair("affinity", "boltz2_affinity", "affinity_model", "affinity_processor"),
    )

    def __init__(
        self,
        *,
        diverse_model: BoltzGenModel,
        diverse_processor: BoltzGenProcessor,
        adherence_model: BoltzGenModel,
        adherence_processor: BoltzGenProcessor,
        inverse_folding_model: BoltzGenInverseFoldingModel,
        inverse_folding_processor: BoltzGenInverseFoldingProcessor,
        fold_model: Boltz2Model,
        fold_processor: Boltz2Processor,
        affinity_model: Boltz2AffinityModel,
        affinity_processor: Boltz2AffinityProcessor,
    ) -> None:
        self.diverse_model = diverse_model
        self.diverse_processor = diverse_processor
        self.adherence_model = adherence_model
        self.adherence_processor = adherence_processor
        self.inverse_folding_model = inverse_folding_model
        self.inverse_folding_processor = inverse_folding_processor
        self.fold_model = fold_model
        self.fold_processor = fold_processor
        self.affinity_model = affinity_model
        self.affinity_processor = affinity_processor
        self._validate_components()

    def design_backbones(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        checkpoint: BoltzGenDesignCheckpoint = "diverse",
        placement: BoltzGenPlacements | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        recycling_steps: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Generate backbones with one explicitly selected released design model.
        """

        model, processor = self._design_pair(checkpoint)
        generator = resolve_generator(seed, generator, device=processor.device)
        batch_size = _diffusion_batch_size(num_candidates, diffusion_batch_size)
        candidates = self._generate_backbones(
            input,
            sources=((checkpoint, model, processor, num_candidates),),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            total_candidates=num_candidates,
            diffusion_batch_size=batch_size,
            num_sampling_steps=num_sampling_steps,
            recycling_steps=recycling_steps,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            generator=generator,
        )
        return BoltzGenDesignResult(candidates=candidates)

    def inverse_fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Assign sequences to a fixed backbone without running evaluation stages.
        """

        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        generator = resolve_generator(
            seed,
            generator,
            device=self.inverse_folding_processor.device,
        )
        structure = cast(Structure, _resolve_structure_reference(input))
        design_references = tuple(design)
        chain_design = _chain_design_references(structure.system, design_references)
        candidate = BoltzGenDesignCandidate(
            structure=structure,
            source_checkpoint=None,
            design=design_references,
            chain_design=chain_design,
            sequence_symmetry=tuple(tuple(group) for group in symmetry),
        )
        canonical_msa = _resolve_msa_reference(msa, structure.system)
        candidates = self._inverse_fold_candidates(
            (candidate,),
            num_sequences_per_backbone=num_sequences,
            msa=canonical_msa,
            temperature=temperature,
            global_omit_tokens=global_omit_tokens,
            aa_constraints=aa_constraints,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )
        return BoltzGenDesignResult(candidates=candidates)

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        protocol: BoltzGenProtocol = "protein-anything",
        placement: BoltzGenPlacements | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_candidates: int = 10_000,
        num_sequences_per_backbone: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int = 500,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        return_trajectory: bool = False,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] | None = None,
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        tie_symmetric_sequences: bool = True,
        budget: int = 30,
        top_budget: int = 10,
        alpha: float | None = None,
        refolding_rmsd_threshold: float | None = None,
        filter_cysteine: bool | None = None,
        filter_binding_site: bool = False,
        filter_target_alignment: bool = False,
        filter_biased: bool = True,
        affinity_ligand: str | None = None,
        metric_weights: Mapping[str, float | None] | None = None,
        additional_filters: Sequence[CandidateFilter] = (),
        size_buckets: Sequence[tuple[int, int, int]] = (),
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Run one of the six released end-to-end BoltzGen protocols.

        All backbones finish inverse folding before any Boltz-2 fold call, and
        every fold call finishes before affinity scoring begins.

        Args:
            input: Optional target/context system, structure, or local structure
                path. ``None`` requests a fully generated assembly.
            protocol: Released scientific protocol.
            placement: Existing/generated segment placement.
            total_len: Total token budget used to realize generated segments.
            conditions: BoltzGen conditioning records.
            msa: Optional MSA evidence or alignment path.
            templates: Optional structural templates.
            num_candidates: Total backbones split between diverse and adherence.
            num_sequences_per_backbone: Inverse-folded sequences per backbone.
            diffusion_batch_size: Designs sharing one Processor/trunk realization.
            num_sampling_steps: Structure-design diffusion transitions.
            step_scale: Optional fixed structure-design Euler-step scale.
            noise_scale: Optional fixed structure-design noise scale.
            return_trajectory: Retain both structure-design sampler histories.
            temperature: Inverse-folding sampling temperature.
            global_omit_tokens: Explicit amino-acid omissions. ``None`` selects
                the protocol default.
            aa_constraints: Per-residue inverse-folding constraints.
            tie_symmetric_sequences: Tie declared symmetric sequence positions.
            budget: Diversity-selected candidate count.
            top_budget: Best-ranked candidate count.
            alpha: Optional quality/diversity trade-off.
            refolding_rmsd_threshold: Optional protocol RMSD-filter override.
            filter_cysteine: Optional protocol cysteine-filter override.
            filter_binding_site: Apply the released binding-site filter.
            filter_target_alignment: Apply the released target-alignment filter.
            filter_biased: Apply released A/G/E/L/V composition filters.
            affinity_ligand: Molecule instance scored by
                ``protein-small_molecule``. May be omitted for one unambiguous
                Molecule instance.
            metric_weights: Direct ranking-weight overrides; ``None`` removes a
                protocol metric.
            additional_filters: Additional explicit candidate filters.
            size_buckets: Optional ``(minimum, maximum, count)`` diversity
                budgets.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with
                ``seed``.

        Returns:
            Every analyzed candidate plus stable ranked, top, and
            diversity-selected indices.
        """

        if protocol not in _PROTOCOLS:
            raise ValueError(f"unknown BoltzGen protocol: {protocol!r}")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if num_sequences_per_backbone <= 0:
            raise ValueError("num_sequences_per_backbone must be positive")
        if budget <= 0 or top_budget <= 0:
            raise ValueError("selection budgets must be positive")

        generator = resolve_generator(
            seed,
            generator,
            device=self.diverse_processor.device,
        )
        batch_size = _diffusion_batch_size(num_candidates, diffusion_batch_size)
        diverse_count = (num_candidates + 1) // 2
        adherence_count = num_candidates // 2
        backbones = self._generate_backbones(
            input,
            sources=(
                ("diverse", self.diverse_model, self.diverse_processor, diverse_count),
                ("adherence", self.adherence_model, self.adherence_processor, adherence_count),
            ),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            total_candidates=num_candidates,
            diffusion_batch_size=batch_size,
            num_sampling_steps=num_sampling_steps,
            recycling_steps=4,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            generator=generator,
        )

        omit_tokens = (
            ("CYS",)
            if global_omit_tokens is None and protocol in {"peptide-anything", "nanobody-anything", "antibody-anything"}
            else (() if global_omit_tokens is None else tuple(global_omit_tokens))
        )
        candidates = self._inverse_fold_candidates(
            backbones,
            num_sequences_per_backbone=num_sequences_per_backbone,
            msa=msa,
            temperature=temperature,
            global_omit_tokens=omit_tokens,
            aa_constraints=aa_constraints,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )

        design_folding = protocol in {"protein-anything", "protein-small_molecule"}
        candidates = tuple(
            self._fold_candidate(
                candidate,
                protocol=protocol,
                design_folding=design_folding,
                generator=generator,
            )
            for candidate in candidates
        )

        if protocol == "protein-small_molecule":
            for candidate in candidates:
                affinity_input = self.affinity_processor(
                    candidate.structure,
                    ligand=affinity_ligand,
                    generator=generator,
                )
                candidate.affinity_result = self.affinity_model.score(
                    affinity_input,
                    num_samples=5,
                    recycling_steps=4,
                    num_sampling_steps=200,
                    molecular_weight_correction=True,
                    generator=generator,
                )

        analyze_hydrophobic_patch = protocol not in {
            "peptide-anything",
            "nanobody-anything",
            "antibody-anything",
        }
        for candidate in candidates:
            candidate.analysis = self._analyze_candidate(
                candidate,
                protocol=protocol,
                largest_hydrophobic_patch=analyze_hydrophobic_patch,
            )

        threshold = (
            2.0
            if refolding_rmsd_threshold is None and protocol == "peptide-anything"
            else (2.5 if refolding_rmsd_threshold is None else refolding_rmsd_threshold)
        )
        cysteine_filter = (
            protocol in {"peptide-anything", "nanobody-anything", "antibody-anything"}
            if filter_cysteine is None
            else filter_cysteine
        )
        released_filters = _released_filters(
            design_folding=design_folding,
            filter_binding_site=filter_binding_site,
            filter_target_alignment=filter_target_alignment,
            filter_cysteine=cysteine_filter,
            filter_biased=filter_biased,
            rmsd_threshold=threshold,
        )
        all_filters = (*released_filters, *tuple(additional_filters))
        for candidate in candidates:
            values = _candidate_values(candidate)
            filter_results = candidate_filter_results(values, all_filters)
            candidate.filters = tuple(
                BoltzGenFilterOutcome(
                    filter=candidate_filter,
                    value=float(values[candidate_filter.feature]),
                    passed=filter_results[f"pass_{candidate_filter.feature}_filter"],
                )
                for candidate_filter in all_filters
            )

        weights = _ranking_metric_weights(protocol)
        if metric_weights is not None:
            for name, weight in metric_weights.items():
                if weight is None:
                    weights.pop(name, None)
                else:
                    weights[name] = weight
        return _select_candidates(
            candidates,
            metric_weights=weights,
            tie_breaker=("design_iptm" if protocol == "protein-redesign" else "design_to_target_iptm"),
            budget=budget,
            top_budget=top_budget,
            alpha=(0.01 if alpha is None and protocol == "peptide-anything" else (0.001 if alpha is None else alpha)),
            size_buckets=size_buckets,
        )

    def _design_pair(
        self,
        checkpoint: BoltzGenDesignCheckpoint,
    ) -> tuple[BoltzGenModel, BoltzGenProcessor]:
        if checkpoint == "diverse":
            return self.diverse_model, self.diverse_processor
        if checkpoint == "adherence":
            return self.adherence_model, self.adherence_processor
        raise ValueError(f"unknown BoltzGen design checkpoint: {checkpoint!r}")

    def _generate_backbones(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        sources: Sequence[
            tuple[
                BoltzGenDesignCheckpoint,
                BoltzGenModel,
                BoltzGenProcessor,
                int,
            ]
        ],
        placement: BoltzGenPlacements | None,
        total_len: int | None,
        conditions: Sequence[BoltzGenSchema],
        msa: MSA | MSACollection | str | PathLike[str] | None,
        templates: Sequence[Template],
        total_candidates: int,
        diffusion_batch_size: int,
        num_sampling_steps: int | None,
        recycling_steps: int | None,
        step_scale: float | None,
        noise_scale: float | None,
        return_trajectory: bool,
        generator: torch.Generator | None,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""
        Generate checkpoint-contiguous batches and rerun Processor/trunk per batch.
        """

        if total_candidates <= 0:
            raise ValueError("total_candidates must be positive")
        canonical_input = _resolve_structure_reference(input)
        condition_values = tuple(conditions)
        generated: list[BoltzGenDesignCandidate] = []
        global_start = 0
        for source, model, processor, source_count in sources:
            remaining = source_count
            while remaining:
                current_batch_size = min(diffusion_batch_size, remaining)
                plan = processor._plan(
                    canonical_input,
                    placement=placement,
                    total_len=total_len,
                    conditions=condition_values,
                )
                canonical_msa = _resolve_msa_reference(msa, plan.system)
                design_input, _ = processor._lower(
                    plan,
                    canonical_msa,
                    templates=templates,
                    generator=generator,
                )
                step_scale_value, noise_scale_value = _sampler_scales(
                    global_start,
                    current_batch_size,
                    total_candidates,
                    device=design_input.atom_mask.device,
                    step_scale=step_scale,
                    noise_scale=noise_scale,
                )
                sample = model.sample(
                    design_input,
                    num_samples=current_batch_size,
                    num_sampling_steps=num_sampling_steps,
                    recycling_steps=recycling_steps,
                    generator=generator,
                    step_scale=step_scale_value,
                    noise_scale=noise_scale_value,
                    return_trajectory=return_trajectory,
                )
                layout = cast(BoltzGenLayout, design_input.layout)
                system = layout.system
                if system is None:
                    raise ValueError("a design layout must carry a molecular system")
                design_references = _design_references(design_input)
                chain_design = _chain_design_references(system, design_references)
                binding_sites = _binding_references(condition_values, layout)
                sequence_symmetry = _protein_sequence_symmetry(
                    system,
                    condition_values,
                )
                generated.extend(
                    BoltzGenDesignCandidate(
                        structure=_decode_structure(
                            design_input,
                            sample.coordinates[0, sample_index],
                        ),
                        source_checkpoint=source,
                        design=design_references,
                        chain_design=chain_design,
                        binding_sites=binding_sites,
                        sequence_symmetry=sequence_symmetry,
                        coordinate_trajectory=(
                            sample.coordinate_trajectory[0, sample_index]
                            if sample.coordinate_trajectory is not None
                            else None
                        ),
                        denoised_coordinate_trajectory=(
                            sample.denoised_coordinate_trajectory[0, sample_index]
                            if sample.denoised_coordinate_trajectory is not None
                            else None
                        ),
                    )
                    for sample_index in range(current_batch_size)
                )
                global_start += current_batch_size
                remaining -= current_batch_size
        return tuple(generated)

    def _inverse_fold_candidates(
        self,
        backbones: Sequence[BoltzGenDesignCandidate],
        *,
        num_sequences_per_backbone: int,
        msa: MSA | MSACollection | str | PathLike[str] | None,
        temperature: float | None,
        global_omit_tokens: Sequence[str],
        aa_constraints: Sequence[BoltzGenResidueConstraint],
        tie_symmetric_sequences: bool,
        generator: torch.Generator | None,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""
        Complete every inverse-folding sample before downstream evaluation.
        """

        samples: list[
            tuple[
                BoltzGenDesignCandidate,
                BoltzGenInverseFoldingInput,
                BoltzGenInverseFoldingDesignOutput,
            ]
        ] = []
        for backbone in backbones:
            canonical_msa = _resolve_msa_reference(msa, backbone.structure.system)
            for _ in range(num_sequences_per_backbone):
                inverse_folding_input = self.inverse_folding_processor(
                    backbone.structure,
                    design=backbone.design,
                    global_omit_tokens=global_omit_tokens,
                    aa_constraints=aa_constraints,
                    symmetry=backbone.sequence_symmetry,
                    msa=canonical_msa,
                    generator=generator,
                )
                inverse_folding_output = self.inverse_folding_model.design(
                    inverse_folding_input,
                    temperature=temperature,
                    tie_symmetric_sequences=tie_symmetric_sequences,
                    generator=generator,
                )
                samples.append(
                    (
                        backbone,
                        inverse_folding_input,
                        inverse_folding_output,
                    )
                )
        return tuple(
            _decode_candidate(
                backbone,
                inverse_folding_input,
                inverse_folding_output,
            )
            for backbone, inverse_folding_input, inverse_folding_output in samples
        )

    def _fold_candidate(
        self,
        candidate: BoltzGenDesignCandidate,
        *,
        protocol: BoltzGenProtocol,
        design_folding: bool,
        generator: torch.Generator | None,
    ) -> BoltzGenDesignCandidate:
        r"""
        Retain all released whole-complex and optional design-only predictions.
        """

        target = _target_references(
            candidate.structure.system,
            design=(candidate.design if protocol == "protein-redesign" else candidate.chain_design),
        )
        whole_input = self.fold_processor(
            candidate.structure,
            design=candidate.design,
            self_template=target,
            generator=generator,
        )
        whole_result = self.fold_model.fold(
            whole_input,
            num_samples=5,
            recycling_steps=4,
            num_sampling_steps=200,
            generator=generator,
        )
        candidate.whole_fold_predictions = whole_result.predictions
        candidate.whole_fold_selected_index = _select_fold_prediction(
            candidate.whole_fold_predictions,
        )

        if design_folding:
            design_structure = _extract_instances(
                candidate.structure,
                {reference[0] for reference in candidate.chain_design},
            )
            design_references = tuple(
                reference for reference in candidate.design if reference[0] in design_structure.system.instances
            )
            design_input = self.fold_processor(
                design_structure,
                design=design_references,
                generator=generator,
            )
            design_result = self.fold_model.fold(
                design_input,
                num_samples=5,
                recycling_steps=4,
                num_sampling_steps=200,
                generator=generator,
            )
            candidate.design_fold_predictions = design_result.predictions
            candidate.design_fold_selected_index = _select_fold_prediction(
                candidate.design_fold_predictions,
            )
        return candidate

    def _analyze_candidate(
        self,
        candidate: BoltzGenDesignCandidate,
        *,
        protocol: BoltzGenProtocol,
        largest_hydrophobic_patch: bool,
    ) -> BoltzGenCandidateAnalysis:
        r"""
        Analyze exactly the selected whole refold used by every derived metric.
        """

        selected = candidate.whole_refold
        if selected is None:
            raise ValueError("BoltzGen candidate analysis requires whole-complex folding")
        target = _target_references(
            candidate.structure.system,
            design=(candidate.design if protocol == "protein-redesign" else candidate.chain_design),
        )
        all_atom_rmsd = structure_rmsd_metrics(
            candidate.structure,
            selected.structure,
            design=candidate.design,
            target=target,
        )
        backbone_rmsd = structure_rmsd_metrics(
            candidate.structure,
            selected.structure,
            design=candidate.design,
            target=target,
            backbone_only=True,
        )

        design_folding_rmsd: float | None = None
        design_folding_bb_rmsd: float | None = None
        design_refold = candidate.design_refold
        if design_refold is not None:
            design_structure = _extract_instances(
                candidate.structure,
                {reference[0] for reference in candidate.chain_design},
            )
            design_references = tuple(
                reference for reference in candidate.design if reference[0] in design_structure.system.instances
            )
            design_only_rmsd = structure_rmsd_metrics(
                design_structure,
                design_refold.structure,
                design=design_references,
                target=(),
            )
            design_only_bb_rmsd = structure_rmsd_metrics(
                design_structure,
                design_refold.structure,
                design=design_references,
                target=(),
                backbone_only=True,
            )
            design_folding_rmsd = design_only_rmsd["rmsd"]
            design_folding_bb_rmsd = design_only_bb_rmsd["rmsd"]

        designed_sequence = _designed_sequence(
            candidate.structure.system,
            candidate.design,
        )
        designed_chains = _designed_chain_analyses(
            candidate.structure.system,
            candidate.design,
        )
        first_designed_chain = designed_chains[0]
        secondary_structure = secondary_structure_fractions(
            assign_secondary_structure(
                candidate.structure,
                components=candidate.design,
            )
        )
        original_noncovalents = count_noncovalent_contacts(
            candidate.structure,
            design=candidate.design,
            chain_design=candidate.chain_design,
        )
        refolded_noncovalents = count_noncovalent_contacts(
            selected.structure,
            design=candidate.design,
            chain_design=candidate.chain_design,
        )
        original_sasa = target_delta_sasa(
            candidate.structure,
            target=target,
            design=candidate.design,
        )
        refolded_sasa = target_delta_sasa(
            selected.structure,
            target=target,
            design=candidate.design,
        )
        binding_site_adherence = _binding_site_adherence(
            candidate.structure,
            candidate.binding_sites,
            candidate.design,
        )
        return BoltzGenCandidateAnalysis(
            designed_sequence=designed_sequence,
            first_designed_chain_sequence=first_designed_chain.full_sequence,
            designed_chains=designed_chains,
            num_design=len(candidate.design),
            rmsd=all_atom_rmsd["rmsd"],
            rmsd_design=all_atom_rmsd["rmsd_design"],
            rmsd_target=all_atom_rmsd["rmsd_target"],
            rmsd_design_target=all_atom_rmsd["rmsd_design_target"],
            target_aligned_rmsd_design=all_atom_rmsd["target_aligned_rmsd_design"],
            bb_rmsd=backbone_rmsd["rmsd"],
            bb_rmsd_design=backbone_rmsd["rmsd_design"],
            bb_rmsd_target=backbone_rmsd["rmsd_target"],
            bb_rmsd_design_target=backbone_rmsd["rmsd_design_target"],
            bb_target_aligned_rmsd_design=backbone_rmsd["target_aligned_rmsd_design"],
            design_folding_rmsd=design_folding_rmsd,
            design_folding_bb_rmsd=design_folding_bb_rmsd,
            binding_site_under_3angstrom=binding_site_adherence.get(3.0),
            binding_site_under_4angstrom=binding_site_adherence.get(4.0),
            binding_site_under_5angstrom=binding_site_adherence.get(5.0),
            binding_site_under_6angstrom=binding_site_adherence.get(6.0),
            binding_site_under_7angstrom=binding_site_adherence.get(7.0),
            binding_site_under_8angstrom=binding_site_adherence.get(8.0),
            binding_site_under_9angstrom=binding_site_adherence.get(9.0),
            amino_acid_fractions=amino_acid_fractions(designed_sequence),
            secondary_structure_fractions=secondary_structure,
            design_hydrophobicity=sequence_hydrophobicity(designed_sequence),
            first_designed_chain_hydrophobicity=(first_designed_chain.hydrophobicity),
            first_designed_chain_liabilities=first_designed_chain.liabilities,
            original_hydrogen_bonds=original_noncovalents.hydrogen_bonds,
            original_charged_pairs=original_noncovalents.charged_pairs,
            refolded_hydrogen_bonds=refolded_noncovalents.hydrogen_bonds,
            refolded_charged_pairs=refolded_noncovalents.charged_pairs,
            target_delta_sasa_original=original_sasa.delta,
            target_sasa_unbound_original=original_sasa.target_unbound,
            target_sasa_bound_original=original_sasa.target_bound,
            target_delta_sasa=refolded_sasa.delta,
            target_sasa_unbound=refolded_sasa.target_unbound,
            target_sasa_bound=refolded_sasa.target_bound,
            largest_hydrophobic_patch=(
                largest_hydrophobic_patch_area(
                    selected.structure,
                    components=candidate.chain_design,
                )
                if largest_hydrophobic_patch
                else None
            ),
        )


def _diffusion_batch_size(num_candidates: int, value: int | None) -> int:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    batch_size = 1 if value is None and num_candidates < 100 else (10 if value is None else value)
    if batch_size <= 0:
        raise ValueError("diffusion_batch_size must be positive")
    return batch_size


def _sampler_scales(
    global_start: int,
    batch_size: int,
    total_candidates: int,
    *,
    device: torch.device,
    step_scale: float | None,
    noise_scale: float | None,
) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    r"""
    Select the released quarter schedule by global candidate order.
    """

    global_index = torch.arange(
        global_start,
        global_start + batch_size,
        device=device,
    )
    quarter = global_index * 4 // total_candidates
    step_scale_value: float | torch.Tensor
    noise_scale_value: float | torch.Tensor
    if step_scale is None:
        step_scale_value = torch.tensor(
            (1.8, 2.0, 1.8, 2.0),
            device=device,
            dtype=torch.float32,
        )[quarter]
    else:
        step_scale_value = step_scale
    if noise_scale is None:
        noise_scale_value = torch.tensor(
            (0.95, 0.88, 0.95, 0.88),
            device=device,
            dtype=torch.float32,
        )[quarter]
    else:
        noise_scale_value = noise_scale
    return step_scale_value, noise_scale_value


def _released_filters(
    *,
    design_folding: bool,
    filter_binding_site: bool,
    filter_target_alignment: bool,
    filter_cysteine: bool,
    filter_biased: bool,
    rmsd_threshold: float,
) -> tuple[CandidateFilter, ...]:
    filters = [
        CandidateFilter("has_x", 0.0, True),
        CandidateFilter("filter_rmsd", rmsd_threshold, True),
        CandidateFilter("filter_rmsd_design", rmsd_threshold, True),
    ]
    if design_folding:
        filters.append(
            CandidateFilter(
                "design_folding_filter_rmsd",
                rmsd_threshold,
                True,
            )
        )
    if filter_binding_site:
        filters.append(CandidateFilter("binding_site_under_8angstrom", 0.0001, False))
    if filter_target_alignment:
        filters.append(
            CandidateFilter(
                "bb_target_aligned_rmsd_design",
                2.5,
                True,
            )
        )
    if filter_cysteine:
        filters.append(CandidateFilter("CYS_fraction", 0.0, True))
    if filter_biased:
        filters.extend(
            CandidateFilter(name, 0.3, True)
            for name in (
                "ALA_fraction",
                "GLY_fraction",
                "GLU_fraction",
                "LEU_fraction",
                "VAL_fraction",
            )
        )
    return tuple(filters)


def _ranking_metric_weights(protocol: BoltzGenProtocol) -> dict[str, float]:
    if protocol == "protein-redesign":
        return {
            "design_residue_iptm": 1.0,
            "iptm": 2.0,
            "ptm": 3.0,
            "neg_filter_rmsd_design": 4.0,
        }
    values = {
        "design_to_target_iptm": 1.0,
        "design_ptm": 1.0,
        "neg_min_design_to_target_pae": 1.0,
        "hydrogen_bonds_refolded": 2.0,
        "charged_pairs_refolded": 2.0,
        "target_delta_sasa_refolded": 2.0,
    }
    if protocol == "protein-small_molecule":
        values["design_to_target_iptm"] = 1.1
        values["design_ptm"] = 1.1
        values["neg_min_design_to_target_pae"] = 1.1
        values["affinity_probability_binary1"] = 1.0
    return values


def _candidate_values(candidate: BoltzGenDesignCandidate) -> dict[str, float]:
    analysis = candidate.analysis
    selected = candidate.whole_refold
    if analysis is None or selected is None:
        raise ValueError("candidate values require completed folding and analysis")
    values = {name: score.value for name, score in selected.scores.items()}
    if candidate.affinity_result is not None:
        values.update({name: score.value for name, score in candidate.affinity_result.scores.items()})
    values.update(analysis.amino_acid_fractions)
    values.update(analysis.secondary_structure_fractions)
    values.update(
        {
            name: float(value)
            for name, value in analysis.first_designed_chain_liabilities.items()
            if isinstance(value, int | float | bool)
        }
    )
    values.update(
        {
            "has_x": float("X" in analysis.designed_sequence),
            "num_design": float(analysis.num_design),
            "filter_rmsd": analysis.bb_rmsd,
            "filter_rmsd_design": analysis.bb_rmsd_design,
            "bb_target_aligned_rmsd_design": analysis.bb_target_aligned_rmsd_design,
            "hydrogen_bonds_original": float(analysis.original_hydrogen_bonds),
            "charged_pairs_original": float(analysis.original_charged_pairs),
            "hydrogen_bonds_refolded": float(analysis.refolded_hydrogen_bonds),
            "charged_pairs_refolded": float(analysis.refolded_charged_pairs),
            "target_delta_sasa_original": analysis.target_delta_sasa_original,
            "target_delta_sasa_refolded": analysis.target_delta_sasa,
            "target_sasa_unbound_refolded": analysis.target_sasa_unbound,
            "target_sasa_bound_refolded": analysis.target_sasa_bound,
            "design_hydrophobicity": analysis.design_hydrophobicity,
            "design_chain_hydrophobicity": (analysis.first_designed_chain_hydrophobicity),
            "neg_design_hydrophobicity": -analysis.design_hydrophobicity,
            "neg_filter_rmsd": -analysis.bb_rmsd,
            "neg_filter_rmsd_design": -analysis.bb_rmsd_design,
        }
    )
    if analysis.design_folding_bb_rmsd is not None:
        values["design_folding_filter_rmsd"] = analysis.design_folding_bb_rmsd
    values.update(
        {
            name: value
            for name, value in (
                (
                    "binding_site_under_3angstrom",
                    analysis.binding_site_under_3angstrom,
                ),
                (
                    "binding_site_under_4angstrom",
                    analysis.binding_site_under_4angstrom,
                ),
                (
                    "binding_site_under_5angstrom",
                    analysis.binding_site_under_5angstrom,
                ),
                (
                    "binding_site_under_6angstrom",
                    analysis.binding_site_under_6angstrom,
                ),
                (
                    "binding_site_under_7angstrom",
                    analysis.binding_site_under_7angstrom,
                ),
                (
                    "binding_site_under_8angstrom",
                    analysis.binding_site_under_8angstrom,
                ),
                (
                    "binding_site_under_9angstrom",
                    analysis.binding_site_under_9angstrom,
                ),
            )
            if value is not None
        }
    )
    if analysis.largest_hydrophobic_patch is not None:
        values["design_largest_hydrophobic_patch_refolded"] = analysis.largest_hydrophobic_patch
        values["neg_design_largest_hydrophobic_patch_refolded"] = -analysis.largest_hydrophobic_patch
    if "min_design_to_target_pae" in values:
        values["neg_min_design_to_target_pae"] = -values["min_design_to_target_pae"]
    if "min_interaction_pae" in values:
        values["neg_min_interaction_pae"] = -values["min_interaction_pae"]
    return values


def _select_candidates(
    candidates: tuple[BoltzGenDesignCandidate, ...],
    *,
    metric_weights: Mapping[str, float],
    tie_breaker: str,
    budget: int,
    top_budget: int,
    alpha: float,
    size_buckets: Sequence[tuple[int, int, int]],
) -> BoltzGenDesignResult:
    r"""
    Deduplicate, rank, and diversity-select while retaining every analyzed row.
    """

    sequences = tuple(cast(BoltzGenCandidateAnalysis, candidate.analysis).designed_sequence for candidate in candidates)
    unique_indices = deduplicate_sequences(sequences)
    unique_candidates = tuple(candidates[index] for index in unique_indices)
    rows = tuple(_candidate_values(candidate) for candidate in unique_candidates)
    filters = tuple(outcome.filter for outcome in unique_candidates[0].filters) if unique_candidates else ()
    ranking = rank_candidates(
        rows,
        filters=filters,
        metric_weights=metric_weights,
        tie_breaker=tie_breaker,
    )
    ranked_indices = tuple(unique_indices[index] for index in ranking.order)
    top_indices = ranked_indices[:top_budget]
    ranked_sequences = tuple(sequences[index] for index in ranked_indices)
    ranked_quality = tuple(float(ranking.quality[index]) for index in ranking.order)
    diverse_rank_positions = greedy_diversity_selection(
        ranked_sequences,
        ranked_quality,
        budget=budget,
        alpha=alpha,
        size_buckets=size_buckets,
    )
    diversity_selected_indices = tuple(ranked_indices[index] for index in diverse_rank_positions)

    rank = np.zeros(len(candidates), dtype=np.int64)
    quality = np.full(len(candidates), np.nan, dtype=np.float64)
    for unique_index, candidate_index in enumerate(unique_indices):
        rank[candidate_index] = ranking.rank[unique_index]
        quality[candidate_index] = ranking.quality[unique_index]
    return BoltzGenDesignResult(
        candidates=candidates,
        ranked_indices=ranked_indices,
        top_indices=top_indices,
        diversity_selected_indices=diversity_selected_indices,
        rank=rank,
        quality=quality,
    )


def _decode_candidate(
    backbone: BoltzGenDesignCandidate,
    inverse_folding_input: BoltzGenInverseFoldingInput,
    inverse_folding_output: BoltzGenInverseFoldingDesignOutput,
) -> BoltzGenDesignCandidate:
    return BoltzGenDesignCandidate(
        structure=_remap_designed_sequence(
            backbone.structure,
            inverse_folding_input,
            inverse_folding_output,
        ),
        source_checkpoint=backbone.source_checkpoint,
        design=backbone.design,
        chain_design=backbone.chain_design,
        binding_sites=backbone.binding_sites,
        sequence_symmetry=backbone.sequence_symmetry,
        coordinate_trajectory=backbone.coordinate_trajectory,
        denoised_coordinate_trajectory=backbone.denoised_coordinate_trajectory,
    )


def _select_fold_prediction(predictions: Sequence[FoldPrediction]) -> int:
    r"""
    Select one design-focused sample; generation order breaks ties.
    """

    return max(
        range(len(predictions)),
        key=lambda index: (
            0.8 * predictions[index].scores["design_to_target_iptm"].value
            + 0.2 * predictions[index].scores["design_ptm"].value
        ),
    )


def _all_component_references(system: MolecularSystem) -> tuple[ComponentReference, ...]:
    return tuple(
        (instance_id, component_index)
        for instance_id, entity_id in system.instances.items()
        for component_index in range(len(system.entities[entity_id].components))
    )


def _chain_design_references(
    system: MolecularSystem,
    design: Sequence[ComponentReference],
) -> tuple[ComponentReference, ...]:
    r"""
    Propagate design membership over whole instances and covalently linked instances.
    """

    selected = {reference[0] for reference in design}
    connections: dict[str, set[str]] = {instance_id: set() for instance_id in system.instances}
    for bond in system.iter_bonds():
        source = bond.site_a.instance_id
        target = bond.site_b.instance_id
        if source != target:
            connections[source].add(target)
            connections[target].add(source)
    pending = list(selected)
    while pending:
        instance_id = pending.pop()
        for connected in connections[instance_id] - selected:
            selected.add(connected)
            pending.append(connected)
    return tuple(reference for reference in _all_component_references(system) if reference[0] in selected)


def _target_references(
    system: MolecularSystem,
    *,
    design: Sequence[ComponentReference],
) -> tuple[ComponentReference, ...]:
    design_set = frozenset(design)
    return tuple(reference for reference in _all_component_references(system) if reference not in design_set)


def _binding_references(
    conditions: Sequence[BoltzGenSchema],
    layout: BoltzGenLayout,
) -> tuple[ComponentReference | AtomSite, ...]:
    system = cast(MolecularSystem, layout.system)

    def token_center_reference(
        site: AtomSite,
    ) -> ComponentReference | AtomSite:
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence):
            return (site.instance_id, site.component_index)
        return site

    references: list[ComponentReference | AtomSite] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenBinding):
            continue
        for endpoint in condition.binding.residues:
            if isinstance(endpoint, str):
                references.extend(layout.placement_map[layout.instance_of[endpoint]][endpoint])
            elif isinstance(endpoint, SymbolicResidueSite):
                references.append(resolve(endpoint, layout.placement_map, layout.instance_of))
            elif isinstance(endpoint, SymbolicAtomSite):
                reference = resolve(
                    endpoint.residue,
                    layout.placement_map,
                    layout.instance_of,
                )
                references.append(
                    token_center_reference(
                        AtomSite(
                            reference[0],
                            reference[1],
                            endpoint.atom_id,
                        )
                    )
                )
            elif isinstance(endpoint, AtomSite):
                references.append(token_center_reference(endpoint))
            else:
                references.append(cast(ComponentReference, endpoint))
    return tuple(dict.fromkeys(references))


def _component_center_coordinates(
    structure: Structure,
    references: Sequence[ComponentReference | AtomSite],
) -> np.ndarray:
    coordinate_index = {site: index for index, site in enumerate(structure.sites) if structure.coordinate_mask[index]}
    centers: list[AtomSite] = []
    for reference in references:
        if isinstance(reference, AtomSite):
            centers.append(reference)
            continue
        instance_id, component_index = reference
        sites = tuple(
            site
            for site in coordinate_index
            if site.instance_id == instance_id and site.component_index == component_index
        )
        entity = structure.system.entities[structure.system.instances[instance_id]]
        if isinstance(entity, PolymerSequence):
            by_name = {site.atom_id: site for site in sites}
            centers.append(by_name.get("CA") or by_name.get("C1'") or sites[0])
        else:
            centers.extend(sites)
    return np.asarray([structure.coordinates[coordinate_index[site]] for site in dict.fromkeys(centers)])


def _binding_site_adherence(
    structure: Structure,
    binding_sites: Sequence[ComponentReference | AtomSite],
    design: Sequence[ComponentReference],
) -> dict[float, float]:
    if not binding_sites:
        return {}
    return binding_site_adherence_curve(
        _component_center_coordinates(structure, binding_sites),
        _component_center_coordinates(structure, design),
    )


def _designed_sequence(
    system: MolecularSystem,
    design: Sequence[ComponentReference],
) -> str:
    return "".join(AMINO_ACID_THREE_TO_ONE.get(system.component(reference).component_id, "X") for reference in design)


def _designed_chain_analyses(
    system: MolecularSystem,
    design: Sequence[ComponentReference],
) -> tuple[BoltzGenDesignedChainAnalysis, ...]:
    instance_ids = tuple(
        dict.fromkeys(reference[0] for reference in design if _is_protein_reference(system, reference))
    )
    analyses = []
    for instance_id in instance_ids:
        entity = cast(
            PolymerSequence,
            system.entities[system.instances[instance_id]],
        )
        designed_sequence = _designed_sequence(
            system,
            tuple(reference for reference in design if reference[0] == instance_id),
        )
        full_sequence = "".join(
            AMINO_ACID_THREE_TO_ONE.get(component.component_id, "X") for component in entity.components
        )
        analyses.append(
            BoltzGenDesignedChainAnalysis(
                instance_id=instance_id,
                designed_sequence=designed_sequence,
                full_sequence=full_sequence,
                hydrophobicity=sequence_hydrophobicity(full_sequence),
                liabilities=liability_metrics(
                    full_sequence,
                    modality="peptide",
                ),
            )
        )
    return tuple(analyses)


def _design_ca_coordinates(
    structure: Structure,
    design: Sequence[ComponentReference],
) -> np.ndarray:
    coordinate_index = {site: index for index, site in enumerate(structure.sites) if structure.coordinate_mask[index]}
    return np.stack(
        [
            structure.coordinates[coordinate_index[AtomSite(instance_id, component_index, "CA")]]
            for instance_id, component_index in design
        ]
    )


def _extract_instances(
    structure: Structure,
    selected: set[str],
) -> Structure:
    instances = {
        instance_id: entity_id
        for instance_id, entity_id in structure.system.instances.items()
        if instance_id in selected
    }
    entities = {entity_id: structure.system.entities[entity_id] for entity_id in dict.fromkeys(instances.values())}
    bonds = frozenset(
        bond
        for bond in structure.system.bonds
        if bond.site_a.instance_id in selected
        and bond.site_b.instance_id in selected
        and (bond.stereo_sites is None or all(site.instance_id in selected for site in bond.stereo_sites))
    )
    system = MolecularSystem(entities, instances, bonds)
    indices = [index for index, site in enumerate(structure.sites) if site.instance_id in selected]
    return Structure(
        system=system,
        sites=tuple(structure.sites[index] for index in indices),
        coordinates=structure.coordinates[indices].copy(),
        coordinate_mask=structure.coordinate_mask[indices].copy(),
        occupancy=(structure.occupancy[indices].copy() if structure.occupancy is not None else None),
        b_factor=(structure.b_factor[indices].copy() if structure.b_factor is not None else None),
        residue_numbers=(
            {reference: number for reference, number in structure.residue_numbers.items() if reference[0] in selected}
            if structure.residue_numbers is not None
            else None
        ),
    )


def _decode_structure(input: BoltzGenInput, coordinates: torch.Tensor) -> Structure:
    r"""
    Map one design sample's model atoms back onto its canonical heavy-atom sites.
    """

    layout = input.layout
    if layout is None or layout.system is None:
        raise ValueError("BoltzGen decoding requires a Processor-populated layout with a system")
    system = layout.system
    complete_sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H")
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    output_coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
    if coordinates.shape[0] < len(layout.atom_sites):
        raise ValueError(
            f"BoltzGen returned {coordinates.shape[0]} atom coordinates for a " f"{len(layout.atom_sites)}-atom layout"
        )
    model_coordinates = coordinates[: len(layout.atom_sites)].float().cpu().numpy()
    for site, values in zip(layout.atom_sites, model_coordinates, strict=True):
        if site is None:
            continue
        index = complete_index[site]
        output_coordinates[index] = values
        coordinate_mask[index] = True
    return Structure(
        system=system,
        sites=complete_sites,
        coordinates=output_coordinates,
        coordinate_mask=coordinate_mask,
    )


def _resolve_structure_reference(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    if isinstance(input, str | PathLike):
        return read_structure(Path(input).expanduser())
    return input


def _resolve_msa_reference(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if not isinstance(msa, str | PathLike):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; " "pass MSACollection")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])


# Model token id -> canonical component id in the BoltzGen residue vocabulary.
_DESIGN_TOKEN_TO_COMPONENT: dict[int, str] = dict(
    enumerate(
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
            "A",
            "G",
            "C",
            "U",
            "N",
            "DA",
            "DG",
            "DC",
            "DT",
            "DN",
        ),
        start=2,
    )
)


def _design_references(
    design_input: BoltzGenInput,
) -> tuple[ComponentReference, ...]:
    r"""
    Return the concrete designed protein residue occurrences.
    """

    layout = cast(BoltzGenLayout, design_input.layout)
    mask = cast(torch.Tensor, design_input.design_mask)
    flags = mask[0].bool().tolist()
    molecule_types = design_input.molecule_type[0].tolist()
    return tuple(
        reference
        for reference, flag, molecule_type in zip(
            layout.output_residues,
            flags,
            molecule_types,
            strict=True,
        )
        if flag and molecule_type == 0 and isinstance(reference, tuple)
    )


def _protein_sequence_symmetry(
    system: MolecularSystem,
    conditions: Sequence[BoltzGenSchema],
) -> tuple[tuple[ComponentReference, ...], ...]:
    r"""
    Transpose structure-copy units into protein-only sequence-identity groups.
    """

    groups: list[tuple[ComponentReference, ...]] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenSymmetry):
            continue
        for group in zip(*condition.units, strict=True):
            protein_group = tuple(reference for reference in group if _is_protein_reference(system, reference))
            if len(protein_group) > 1:
                groups.append(protein_group)
    return tuple(groups)


def _is_protein_reference(
    system: MolecularSystem,
    reference: ComponentReference,
) -> bool:
    entity = system.entities[system.instances[reference[0]]]
    return isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"


def _remap_designed_sequence(
    backbone: Structure,
    inverse_folding_input: BoltzGenInverseFoldingInput,
    inverse_folding_output: BoltzGenInverseFoldingDesignOutput,
) -> Structure:
    r"""
    Write inverse-folding residue identities into one generated backbone.
    """

    layout = cast(BoltzGenInverseFoldingLayout, inverse_folding_input.layout)
    token_ids = inverse_folding_output.tokens[0].long().tolist()
    flags = cast(torch.Tensor, inverse_folding_input.design_mask)[0].bool().tolist()
    designed: dict[ComponentReference, str] = {}
    for token_index, reference in enumerate(layout.output_residues):
        if not flags[token_index]:
            continue
        designed[cast(ComponentReference, reference)] = _DESIGN_TOKEN_TO_COMPONENT[token_ids[token_index]]
    if not designed:
        return backbone
    return _write_designed_structure(backbone, designed)


def _write_designed_structure(
    backbone: Structure,
    designed: Mapping[ComponentReference, str],
) -> Structure:
    r"""
    Rebuild a backbone with designed identities and unresolved side chains.
    """

    system = backbone.system
    design_positions = frozenset(designed)
    new_entities: dict[str, PolymerSequence | Molecule] = {}
    new_instances: dict[str, str] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            components = list(entity.components)
            changed = False
            for index in range(len(components)):
                component_id = designed.get((instance_id, index))
                if component_id is not None:
                    components[index] = get_component(component_id)
                    changed = True
            if changed:
                new_entity_id = f"{entity_id}__designed_{instance_id}"
                new_entities[new_entity_id] = PolymerSequence(
                    entity.polymer_type,
                    tuple(components),
                )
                new_instances[instance_id] = new_entity_id
                continue
        new_entities.setdefault(entity_id, entity)
        new_instances[instance_id] = entity_id

    new_system = MolecularSystem(new_entities, new_instances, system.bonds)
    old_index = {site: index for index, site in enumerate(backbone.sites)}
    backbone_atoms = {"N", "CA", "C", "O"}
    sites: list[AtomSite] = []
    coordinates: list[np.ndarray] = []
    coordinate_mask: list[bool] = []
    occupancy: list[float] | None = None if backbone.occupancy is None else []
    b_factor: list[float] | None = None if backbone.b_factor is None else []
    for site, atom in new_system.iter_atoms():
        if atom.element == "H":
            continue
        reference = (site.instance_id, site.component_index)
        if reference in design_positions and site.atom_id not in backbone_atoms:
            coordinate = np.zeros(3, dtype=backbone.coordinates.dtype)
            present = False
            source_index = None
        else:
            source_index = old_index.get(site)
            coordinate = (
                backbone.coordinates[source_index]
                if source_index is not None
                else np.zeros(3, dtype=backbone.coordinates.dtype)
            )
            present = bool(backbone.coordinate_mask[source_index]) if source_index is not None else False
        sites.append(site)
        coordinates.append(np.asarray(coordinate))
        coordinate_mask.append(present)
        if occupancy is not None:
            occupancy.append(
                float(backbone.occupancy[source_index])
                if source_index is not None and backbone.occupancy is not None
                else float("nan")
            )
        if b_factor is not None:
            b_factor.append(
                float(backbone.b_factor[source_index])
                if source_index is not None and backbone.b_factor is not None
                else float("nan")
            )

    return Structure(
        system=new_system,
        sites=tuple(sites),
        coordinates=np.asarray(
            coordinates,
            dtype=backbone.coordinates.dtype,
        ).reshape(
            len(sites),
            3,
        ),
        coordinate_mask=np.asarray(coordinate_mask, dtype=np.bool_),
        occupancy=(None if occupancy is None else np.asarray(occupancy)),
        b_factor=(None if b_factor is None else np.asarray(b_factor)),
        residue_numbers=backbone.residue_numbers,
    )
