r"""
Released BoltzGen structure generation, inverse folding, analysis, and selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ...data import PlacementSpec
from ...data.components import get_component
from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    instance_subgraph,
    repartition_entities_by_instance_sequence,
    replace_components,
)
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.input import (
    resolve_msa_input,
    resolve_structure_input,
    resolve_template_input,
)
from ...results import FoldPrediction, ScoreResult
from ..boltz2 import Boltz2AffinityScorePipeline, Boltz2Pipeline
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
    BoltzGenInverseFoldingModel,
    BoltzGenMiniformerModel,
    BoltzGenModel,
    BoltzGenResidueConstraint,
    BoltzGenSamplingOutput,
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

    Attributes:
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
    source_checkpoint: BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None
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
class BoltzGenCandidateResult:
    r"""Candidates retained in generation order before analysis and ranking."""

    candidates: tuple[BoltzGenDesignCandidate, ...]


@dataclass(eq=False)
class BoltzGenDesignResult(BoltzGenCandidateResult):
    r"""
    Complete analyzed candidates and stable selection views.

    ``ranked_indices``, ``top_indices``, and ``diversity_selected_indices`` refer
    to ``candidates``. Duplicate sequences remain in ``candidates`` for
    provenance, but only their earliest occurrence enters these selections.
    ``rank`` and ``quality`` also align to ``candidates``; excluded duplicates
    carry rank zero and NaN quality.
    """

    ranked_indices: tuple[int, ...]
    top_indices: tuple[int, ...]
    diversity_selected_indices: tuple[int, ...]
    rank: np.ndarray
    quality: np.ndarray

    @property
    def best(self) -> BoltzGenDesignCandidate:
        return self.candidates[self.ranked_indices[0]]

    @property
    def ranked_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.ranked_indices)

    @property
    def top_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.top_indices)

    @property
    def diversity_selected_candidates(self) -> tuple[BoltzGenDesignCandidate, ...]:
        return tuple(self.candidates[index] for index in self.diversity_selected_indices)


class BoltzGenBackbonePipeline(PairedPipeline[BoltzGenModel, BoltzGenProcessor]):
    r"""
    Independently callable backbone-design pipeline for a full BoltzGen checkpoint.
    """

    model_architecture = "boltzgen"

    def prepare_design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, tuple[BoltzGenSchema, ...]]:
        r"""Resolve one symbolic design plan and lower its model input."""

        structure, resolved_conditions, generated_references = self.processor.plan(
            _resolve_structure_reference(input),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
        )
        model_input, _ = self.processor.lower(
            structure,
            resolved_conditions,
            generated_references,
            _resolve_msa_reference(msa, structure.system),
            templates=resolve_template_input(templates),
            generator=generator,
        )
        return model_input, resolved_conditions

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        recycling_steps: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        return_trajectory: bool = False,
        candidate_offset: int = 0,
        total_candidates: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate backbones with one full BoltzGen checkpoint.

        ``candidate_offset`` and ``total_candidates`` define the released global
        quarter schedule only. Separately executed ranges must retain complete
        ``diffusion_batch_size`` groups and use independently derived seeds or
        generators; these values do not reproduce a monolithic call's RNG consumption.
        """

        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        total_candidates = num_candidates if total_candidates is None else total_candidates
        generator = resolve_generator(seed, generator, device=self.processor.device)
        candidates = _generate_backbones(
            input,
            sources=((None, self, num_candidates),),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            candidate_offset=candidate_offset,
            total_candidates=total_candidates,
            diffusion_batch_size=_diffusion_batch_size(total_candidates, diffusion_batch_size),
            num_sampling_steps=num_sampling_steps,
            recycling_steps=recycling_steps,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            generator=generator,
        )
        return BoltzGenCandidateResult(candidates=candidates)

    def decode_design(
        self,
        input: BoltzGenInput,
        output: BoltzGenSamplingOutput,
        conditions: Sequence[BoltzGenSchema],
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Decode raw backbone samples without executing the Model."""

        return _decode_backbone_candidates(input, output, conditions, source=None)


class BoltzGenMiniformerPipeline(PairedPipeline[BoltzGenMiniformerModel, BoltzGenProcessor]):
    r"""
    Independent structure-design pipeline for the released structure-trained small model.
    """

    model_architecture = "boltzgen_miniformer"

    def prepare_design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, tuple[BoltzGenSchema, ...]]:
        r"""Resolve one miniformer design plan and lower its model input."""

        structure, resolved_conditions, generated_references = self.processor.plan(
            _resolve_structure_reference(input),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
        )
        model_input, _ = self.processor.lower(
            structure,
            resolved_conditions,
            generated_references,
            _resolve_msa_reference(msa, structure.system),
            templates=resolve_template_input(templates),
            generator=generator,
        )
        return model_input, resolved_conditions

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        recycling_steps: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate backbones with the structure-trained small checkpoint.

        This operation owns one complete candidate range. It supports internal
        diffusion batching but not independently executed candidate shards.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        candidates = _generate_backbones(
            input,
            sources=(("structuretrained_small", self, num_candidates),),
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            total_candidates=num_candidates,
            diffusion_batch_size=_diffusion_batch_size(num_candidates, diffusion_batch_size),
            num_sampling_steps=num_sampling_steps,
            recycling_steps=recycling_steps,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            generator=generator,
        )
        return BoltzGenCandidateResult(candidates=candidates)

    def decode_design(
        self,
        input: BoltzGenInput,
        output: BoltzGenSamplingOutput,
        conditions: Sequence[BoltzGenSchema],
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Decode raw miniformer backbone samples without executing the Model."""

        return _decode_backbone_candidates(input, output, conditions, source="structuretrained_small")


class BoltzGenInverseFoldingPipeline(PairedPipeline[BoltzGenInverseFoldingModel, BoltzGenInverseFoldingProcessor]):
    r"""
    Independently callable BoltzGen inverse-folding pipeline.
    """

    model_architecture = "boltzgen_ifold"

    def prepare_design(
        self,
        input: BoltzGenDesignCandidate | Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] | None = None,
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        global_omit_tokens: Sequence[str] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInverseFoldingInput, BoltzGenDesignCandidate]:
        r"""Prepare one backbone and one stochastic inverse-folding input."""

        if isinstance(input, BoltzGenDesignCandidate):
            structure = input.structure
            design_references = input.design if design is None else tuple(design)
            candidate = BoltzGenDesignCandidate(
                structure=structure,
                source_checkpoint=input.source_checkpoint,
                design=design_references,
                chain_design=(
                    input.chain_design
                    if design is None
                    else _chain_design_references(structure.system, design_references)
                ),
                binding_sites=input.binding_sites,
                sequence_symmetry=(
                    input.sequence_symmetry if symmetry is None else tuple(tuple(group) for group in symmetry)
                ),
                coordinate_trajectory=input.coordinate_trajectory,
                denoised_coordinate_trajectory=input.denoised_coordinate_trajectory,
            )
        else:
            structure = cast(Structure, _resolve_structure_reference(input))
            design_references = () if design is None else tuple(design)
            candidate = BoltzGenDesignCandidate(
                structure=structure,
                source_checkpoint=None,
                design=design_references,
                chain_design=_chain_design_references(structure.system, design_references),
                sequence_symmetry=(() if symmetry is None else tuple(tuple(group) for group in symmetry)),
            )
        model_input = self.processor(
            structure,
            design=candidate.design,
            global_omit_tokens=global_omit_tokens,
            aa_constraints=aa_constraints,
            symmetry=candidate.sequence_symmetry,
            msa=_resolve_msa_reference(msa, structure.system),
            generator=generator,
        )
        return model_input, candidate

    def design(
        self,
        input: BoltzGenDesignCandidate | Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] | None = None,
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Assign sequences to a fixed backbone without evaluation stages.

        A candidate input retains its checkpoint, binding, symmetry, and
        trajectory metadata. Explicit ``design`` or ``symmetry`` values replace
        the corresponding candidate fields. Results intentionally clear any
        downstream folding, affinity, analysis, and filtering state because the
        sequence has changed.
        """

        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input: BoltzGenDesignCandidate | Structure
        if isinstance(input, BoltzGenDesignCandidate | Structure):
            canonical_input = input
        else:
            canonical_input = cast(Structure, _resolve_structure_reference(input))
        structure = (
            canonical_input.structure if isinstance(canonical_input, BoltzGenDesignCandidate) else canonical_input
        )
        canonical_msa = _resolve_msa_reference(msa, structure.system)
        candidates = []
        for _ in range(num_sequences):
            model_input, candidate = self.prepare_design(
                canonical_input,
                design=design,
                aa_constraints=aa_constraints,
                symmetry=symmetry,
                msa=canonical_msa,
                global_omit_tokens=global_omit_tokens,
                generator=generator,
            )
            output = self.model.design(
                model_input,
                temperature=temperature,
                tie_symmetric_sequences=tie_symmetric_sequences,
                generator=generator,
            )
            candidates.append(self.decode_design(model_input, output, candidate))
        return BoltzGenCandidateResult(candidates=tuple(candidates))

    def decode_design(
        self,
        input: BoltzGenInverseFoldingInput,
        output: BoltzGenInverseFoldingDesignOutput,
        candidate: BoltzGenDesignCandidate,
    ) -> BoltzGenDesignCandidate:
        r"""Decode one raw inverse-folding output without executing the Model."""

        return _decode_candidate(candidate, input, output)


class BoltzGenPipeline(CompositePipeline):
    r"""
    Released five-checkpoint BoltzGen design pipeline.

    The repository topology is fixed: diverse design at the root and adherence
    design, inverse folding, Boltz-2 folding, and Boltz-2 affinity under their
    named subdirectories.
    """

    pair_layout = (
        CompositePair(
            "design_diverse",
            "",
            BoltzGenBackbonePipeline,
        ),
        CompositePair(
            "design_adherence",
            "adherence",
            BoltzGenBackbonePipeline,
        ),
        CompositePair(
            "inverse_fold",
            "ifold",
            BoltzGenInverseFoldingPipeline,
        ),
        CompositePair(
            "fold",
            "fold",
            Boltz2Pipeline,
        ),
        CompositePair(
            "affinity",
            "affinity",
            Boltz2AffinityScorePipeline,
        ),
    )

    def __init__(
        self,
        *,
        design_diverse: BoltzGenBackbonePipeline,
        design_adherence: BoltzGenBackbonePipeline,
        inverse_fold: BoltzGenInverseFoldingPipeline,
        fold: Boltz2Pipeline,
        affinity: Boltz2AffinityScorePipeline,
    ) -> None:
        super().__init__(
            {
                "design_diverse": design_diverse,
                "design_adherence": design_adherence,
                "inverse_fold": inverse_fold,
                "fold": fold,
                "affinity": affinity,
            }
        )

    @property
    def design_diverse(self) -> BoltzGenBackbonePipeline:
        return cast(BoltzGenBackbonePipeline, self.components["design_diverse"])

    @property
    def design_adherence(self) -> BoltzGenBackbonePipeline:
        return cast(BoltzGenBackbonePipeline, self.components["design_adherence"])

    @property
    def inverse_folding(self) -> BoltzGenInverseFoldingPipeline:
        return cast(BoltzGenInverseFoldingPipeline, self.components["inverse_fold"])

    @property
    def fold(self) -> Boltz2Pipeline:
        return cast(Boltz2Pipeline, self.components["fold"])

    @property
    def affinity(self) -> Boltz2AffinityScorePipeline:
        return cast(Boltz2AffinityScorePipeline, self.components["affinity"])

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        protocol: BoltzGenProtocol = "protein-anything",
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
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
            device=self.design_diverse.processor.device,
        )
        batch_size = _diffusion_batch_size(num_candidates, diffusion_batch_size)
        diverse_count = (num_candidates + 1) // 2
        adherence_count = num_candidates // 2
        backbones = _generate_backbones(
            input,
            sources=(
                (
                    "diverse",
                    self.design_diverse,
                    diverse_count,
                ),
                (
                    "adherence",
                    self.design_adherence,
                    adherence_count,
                ),
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
        candidates = tuple(
            candidate
            for backbone in backbones
            for candidate in self.inverse_folding.design(
                backbone,
                aa_constraints=aa_constraints,
                msa=msa,
                temperature=temperature,
                global_omit_tokens=omit_tokens,
                tie_symmetric_sequences=tie_symmetric_sequences,
                num_sequences=num_sequences_per_backbone,
                generator=generator,
            ).candidates
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
                candidate.affinity_result = self.affinity.score(
                    candidate.structure,
                    ligand=affinity_ligand,
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

    def design_backbones(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        checkpoint: BoltzGenDesignCheckpoint = "diverse",
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        recycling_steps: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate backbones with one explicitly selected released design model.
        """

        result = self._design_pipeline(checkpoint).design(
            input,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            num_candidates=num_candidates,
            diffusion_batch_size=diffusion_batch_size,
            num_sampling_steps=num_sampling_steps,
            recycling_steps=recycling_steps,
            step_scale=step_scale,
            noise_scale=noise_scale,
            return_trajectory=return_trajectory,
            seed=seed,
            generator=generator,
        )
        for candidate in result.candidates:
            candidate.source_checkpoint = checkpoint
        return result

    def inverse_fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Assign sequences to a fixed backbone without running evaluation stages.
        """

        return self.inverse_folding.design(
            input,
            design=design,
            aa_constraints=aa_constraints,
            symmetry=symmetry,
            msa=msa,
            temperature=temperature,
            global_omit_tokens=global_omit_tokens,
            tie_symmetric_sequences=tie_symmetric_sequences,
            num_sequences=num_sequences,
            seed=seed,
            generator=generator,
        )

    def _design_pipeline(
        self,
        checkpoint: BoltzGenDesignCheckpoint,
    ) -> BoltzGenBackbonePipeline:
        if checkpoint == "diverse":
            return self.design_diverse
        if checkpoint == "adherence":
            return self.design_adherence
        raise ValueError(f"unknown BoltzGen design checkpoint: {checkpoint!r}")

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
        whole_result = self.fold.fold(
            candidate.structure,
            design=candidate.design,
            self_template=target,
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
                reference for reference in candidate.design if reference[0] in _instance_ids(design_structure.system)
            )
            design_result = self.fold.fold(
                design_structure,
                design=design_references,
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
                reference for reference in candidate.design if reference[0] in _instance_ids(design_structure.system)
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


def _decode_backbone_candidates(
    input: BoltzGenInput,
    output: BoltzGenSamplingOutput,
    conditions: Sequence[BoltzGenSchema],
    *,
    source: BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None,
) -> tuple[BoltzGenDesignCandidate, ...]:
    r"""Decode one raw BoltzGen sampling batch into canonical candidates."""

    system = input.system
    design_references = _design_references(input)
    chain_design = _chain_design_references(system, design_references)
    binding_sites = _binding_references(conditions, system)
    sequence_symmetry = _protein_sequence_symmetry(system, conditions)
    return tuple(
        BoltzGenDesignCandidate(
            structure=_decode_structure(input, output.coordinates[0, sample_index]),
            source_checkpoint=source,
            design=design_references,
            chain_design=chain_design,
            binding_sites=binding_sites,
            sequence_symmetry=sequence_symmetry,
            coordinate_trajectory=(
                output.coordinate_trajectory[0, sample_index] if output.coordinate_trajectory is not None else None
            ),
            denoised_coordinate_trajectory=(
                output.denoised_coordinate_trajectory[0, sample_index]
                if output.denoised_coordinate_trajectory is not None
                else None
            ),
        )
        for sample_index in range(output.coordinates.shape[1])
    )


def _generate_backbones(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
    *,
    sources: Sequence[
        tuple[
            BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None,
            BoltzGenBackbonePipeline | BoltzGenMiniformerPipeline,
            int,
        ]
    ],
    placement: PlacementSpec | None,
    total_len: int | None,
    conditions: Sequence[BoltzGenSchema],
    msa: MSA | MsaCollection | str | PathLike[str] | None,
    templates: Sequence[Template] | str | PathLike[str],
    total_candidates: int,
    candidate_offset: int = 0,
    diffusion_batch_size: int,
    num_sampling_steps: int | None,
    recycling_steps: int | None,
    step_scale: float | None,
    noise_scale: float | None,
    return_trajectory: bool,
    generator: torch.Generator | None,
) -> tuple[BoltzGenDesignCandidate, ...]:
    r"""Generate checkpoint-contiguous batches and rerun Processor/trunk per batch."""

    if total_candidates <= 0:
        raise ValueError("total_candidates must be positive")
    generated_count = sum(source_count for _, _, source_count in sources)
    if candidate_offset < 0 or candidate_offset + generated_count > total_candidates:
        raise ValueError("candidate range must lie within total_candidates")
    canonical_input = _resolve_structure_reference(input)
    templates = resolve_template_input(templates)
    condition_values = tuple(conditions)
    generated: list[BoltzGenDesignCandidate] = []
    global_start = candidate_offset
    for source, pipeline, source_count in sources:
        remaining = source_count
        while remaining:
            current_batch_size = min(diffusion_batch_size, remaining)
            design_input, resolved_conditions = pipeline.prepare_design(
                canonical_input,
                placement=placement,
                total_len=total_len,
                conditions=condition_values,
                msa=msa,
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
            sample = pipeline.model.sample(
                design_input,
                num_samples=current_batch_size,
                num_sampling_steps=num_sampling_steps,
                recycling_steps=recycling_steps,
                generator=generator,
                step_scale=step_scale_value,
                noise_scale=noise_scale_value,
                return_trajectory=return_trajectory,
            )
            generated.extend(_decode_backbone_candidates(design_input, sample, resolved_conditions, source=source))
            global_start += current_batch_size
            remaining -= current_batch_size
    return tuple(generated)


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

    rank: NDArray[np.int64] = np.zeros(len(candidates), dtype=np.int64)
    quality: NDArray[np.float64] = np.full(len(candidates), np.nan, dtype=np.float64)
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
        system.component_reference(component_index) for component_index in range(system.component_symbol.numel())
    )


def _instance_ids(system: MolecularSystem) -> frozenset[str]:
    return frozenset(system.instance_id(index) for index in range(system.instance_symbol.numel()))


def _atom_component_index(system: MolecularSystem) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _component_instance_index(system: MolecularSystem) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )


def _chain_design_references(
    system: MolecularSystem,
    design: Sequence[ComponentReference],
) -> tuple[ComponentReference, ...]:
    r"""
    Propagate design membership over whole instances and covalently linked instances.
    """

    selected = {system.instance_index(reference[0]) for reference in design}
    instance_count = system.instance_symbol.numel()
    connections: dict[int, set[int]] = {instance_index: set() for instance_index in range(instance_count)}
    atom_component = _atom_component_index(system)
    component_instance = _component_instance_index(system)
    edge_instance = component_instance[atom_component[system.edge_index]]
    for source, target in edge_instance.detach().cpu().T.tolist():
        if source != target:
            connections[source].add(target)
            connections[target].add(source)
    pending = list(selected)
    while pending:
        instance_index = pending.pop()
        for connected in connections[instance_index] - selected:
            selected.add(connected)
            pending.append(connected)
    return tuple(
        reference for reference in _all_component_references(system) if system.instance_index(reference[0]) in selected
    )


def _target_references(
    system: MolecularSystem,
    *,
    design: Sequence[ComponentReference],
) -> tuple[ComponentReference, ...]:
    design_set = frozenset(design)
    return tuple(reference for reference in _all_component_references(system) if reference not in design_set)


def _binding_references(
    conditions: Sequence[BoltzGenSchema],
    system: MolecularSystem,
) -> tuple[ComponentReference | AtomSite, ...]:
    def token_center_reference(
        site: AtomSite,
    ) -> ComponentReference | AtomSite:
        instance_index = system.instance_index(site.instance_id)
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
            return (site.instance_id, site.component_index)
        return site

    references: list[ComponentReference | AtomSite] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenBinding):
            continue
        for endpoint in condition.binding.residues:
            if isinstance(endpoint, AtomSite):
                references.append(token_center_reference(endpoint))
            else:
                references.append(cast(ComponentReference, endpoint))
    return tuple(dict.fromkeys(references))


def _component_center_coordinates(
    structure: Structure,
    references: Sequence[ComponentReference | AtomSite],
) -> np.ndarray:
    system = structure.system
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    component_instance = _component_instance_index(system).detach().cpu().tolist()
    centers: list[int] = []
    for reference in references:
        if isinstance(reference, AtomSite):
            atom_index = system.atom_index(reference)
            if bool(structure.coordinate_mask[atom_index]):
                centers.append(atom_index)
            continue
        component_index = system.component_index(reference)
        atoms = tuple(
            atom_index
            for atom_index in range(
                int(system.component_atom_ptr[component_index]),
                int(system.component_atom_ptr[component_index + 1]),
            )
            if bool(structure.coordinate_mask[atom_index])
        )
        if not atoms:
            continue
        instance_index = component_instance[component_index]
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
            by_name = {system.symbols[atom_symbols[atom_index]]: atom_index for atom_index in atoms}
            centers.append(by_name["CA"] if "CA" in by_name else by_name["C1'"] if "C1'" in by_name else atoms[0])
        else:
            centers.extend(atoms)
    unique = tuple(dict.fromkeys(centers))
    return (
        structure.coordinates[
            torch.tensor(
                unique,
                dtype=torch.long,
                device=structure.coordinates.device,
            )
        ]
        .detach()
        .cpu()
        .numpy()
    )


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
    return "".join(
        AMINO_ACID_THREE_TO_ONE.get(
            system.symbols[int(system.component_symbol[system.component_index(reference)])],
            "X",
        )
        for reference in design
    )


def _designed_chain_analyses(
    system: MolecularSystem,
    design: Sequence[ComponentReference],
) -> tuple[BoltzGenDesignedChainAnalysis, ...]:
    instance_ids = tuple(
        dict.fromkeys(reference[0] for reference in design if _is_protein_reference(system, reference))
    )
    analyses = []
    for instance_id in instance_ids:
        instance_index = system.instance_index(instance_id)
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        designed_sequence = _designed_sequence(
            system,
            tuple(reference for reference in design if reference[0] == instance_id),
        )
        full_sequence = "".join(
            AMINO_ACID_THREE_TO_ONE.get(
                system.symbols[int(system.component_symbol[component_index])],
                "X",
            )
            for component_index in range(component_start, component_stop)
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
    indices = [
        structure.system.atom_index(AtomSite(instance_id, component_index, "CA"))
        for instance_id, component_index in design
    ]
    return structure.coordinates[torch.tensor(indices, device=structure.coordinates.device)].detach().cpu().numpy()


def _extract_instances(
    structure: Structure,
    selected: set[str],
) -> Structure:
    instance_mask = torch.tensor(
        [structure.system.instance_id(index) in selected for index in range(structure.system.instance_symbol.numel())],
        dtype=torch.bool,
        device=structure.system.instance_symbol.device,
    )
    system, component_to_new, atom_to_new = instance_subgraph(
        structure.system,
        instance_mask,
    )
    retained_atoms = atom_to_new >= 0
    retained_components = component_to_new >= 0
    component_parent_ptr, parent_component_id = structure.select_component_parents(
        torch.nonzero(retained_components, as_tuple=True)[0]
    )
    return structure.replace(  # type: ignore[attr-defined]
        system=system,
        coordinates=structure.coordinates[retained_atoms],
        coordinate_mask=structure.coordinate_mask[retained_atoms],
        occupancy=structure.occupancy[retained_atoms],
        occupancy_mask=structure.occupancy_mask[retained_atoms],
        b_factor=structure.b_factor[retained_atoms],
        b_factor_mask=structure.b_factor_mask[retained_atoms],
        source_formal_charge_mask=structure.source_formal_charge_mask[retained_atoms],
        residue_id=structure.residue_id[retained_components],
        insertion_code=structure.insertion_code[retained_components],
        component_parent_ptr=component_parent_ptr,
        parent_component_id=parent_component_id,
    )


def _decode_structure(input: BoltzGenInput, coordinates: torch.Tensor) -> Structure:
    r"""
    Scatter one design sample's real model rows onto the canonical atom axis.
    """

    system = input.system
    atom_index = input.atom_index[0].detach().cpu()
    atom_mask = input.atom_mask[0].detach().cpu().bool()
    valid = atom_mask & (atom_index >= 0)
    output_coordinates = (
        coordinates.detach()
        .cpu()
        .new_zeros(
            system.atom_symbol.numel(),
            3,
        )
    )
    coordinate_mask = torch.zeros(
        system.atom_symbol.numel(),
        dtype=torch.bool,
    )
    output_coordinates[atom_index[valid]] = coordinates.detach().cpu()[valid]
    coordinate_mask[atom_index[valid]] = True
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=output_coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=output_coordinates.new_zeros(system.atom_symbol.numel()),
        occupancy_mask=coordinate_mask.new_zeros(system.atom_symbol.numel()),
        b_factor=output_coordinates.new_zeros(system.atom_symbol.numel()),
        b_factor_mask=coordinate_mask.new_zeros(system.atom_symbol.numel()),
        source_symbols=(),
        residue_id=torch.full(
            (system.component_symbol.numel(),),
            -1,
            dtype=torch.long,
        ),
        insertion_code=torch.full(
            (system.component_symbol.numel(),),
            -1,
            dtype=torch.long,
        ),
        batch_size=[],
    )


def _resolve_structure_reference(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    return resolve_structure_input(input)


def _resolve_msa_reference(
    msa: MSA | MsaCollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MsaCollection | None:
    return resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN})


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

    mask = cast(torch.Tensor, design_input.design_mask)
    flags = mask[0].bool().tolist()
    molecule_types = design_input.molecule_type[0].tolist()
    return tuple(
        design_input.system.component_reference(component_index)
        for component_index, flag, molecule_type in zip(
            design_input.token_component_index[0].detach().cpu().tolist(),
            flags,
            molecule_types,
            strict=True,
        )
        if flag and molecule_type == 0
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
    instance_index = system.instance_index(reference[0])
    entity_index = int(system.instance_entity[instance_index])
    return int(system.entity_type[entity_index]) == ENTITY_PROTEIN


def _remap_designed_sequence(
    backbone: Structure,
    inverse_folding_input: BoltzGenInverseFoldingInput,
    inverse_folding_output: BoltzGenInverseFoldingDesignOutput,
) -> Structure:
    r"""
    Write inverse-folding residue identities into one generated backbone.
    """

    token_ids = inverse_folding_output.tokens[0].long().tolist()
    flags = cast(torch.Tensor, inverse_folding_input.design_mask)[0].bool().tolist()
    designed: dict[ComponentReference, str] = {}
    for token_index, component_index in enumerate(
        inverse_folding_input.token_component_index[0].detach().cpu().tolist()
    ):
        if not flags[token_index]:
            continue
        designed[inverse_folding_input.system.component_reference(component_index)] = _DESIGN_TOKEN_TO_COMPONENT[
            token_ids[token_index]
        ]
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
    component_index = torch.tensor(
        [system.component_index(reference) for reference in designed],
        dtype=torch.long,
        device=system.component_symbol.device,
    )
    new_system, old_to_new, _ = replace_components(
        system,
        component_index,
        [get_component(component_id) for component_id in designed.values()],
    )
    new_system = repartition_entities_by_instance_sequence(system, new_system)

    backbone_atoms = {"N", "CA", "C", "O"}
    selected = torch.zeros(
        system.component_symbol.numel(),
        dtype=torch.bool,
        device=system.component_symbol.device,
    )
    selected[component_index] = True
    atom_component = _atom_component_index(system)
    atom_names = [system.symbols[index] for index in system.atom_symbol.detach().cpu().tolist()]
    retain = (old_to_new >= 0) & (
        ~selected[atom_component]
        | torch.tensor(
            [name in backbone_atoms for name in atom_names],
            dtype=torch.bool,
            device=old_to_new.device,
        )
    )
    atom_count = new_system.atom_symbol.numel()

    def transfer(value: torch.Tensor) -> torch.Tensor:
        output = value.new_zeros((atom_count, *value.shape[1:]))
        output[old_to_new[retain]] = value[retain]
        return output

    parent_source_index = list(range(system.component_symbol.numel()))
    for reference in designed:
        component = system.component_index(reference)
        source_id = system.symbols[int(system.component_symbol[component])]
        result_id = new_system.symbols[int(new_system.component_symbol[component])]
        if source_id != result_id:
            parent_source_index[component] = -1
    component_parent_ptr, parent_component_id = backbone.select_component_parents(parent_source_index)
    return Structure(  # type: ignore[call-arg]
        system=new_system,
        coordinates=transfer(backbone.coordinates),
        coordinate_mask=transfer(backbone.coordinate_mask),
        occupancy=transfer(backbone.occupancy),
        occupancy_mask=transfer(backbone.occupancy_mask),
        b_factor=transfer(backbone.b_factor),
        b_factor_mask=transfer(backbone.b_factor_mask),
        source_symbols=backbone.source_symbols,
        residue_id=backbone.residue_id,
        insertion_code=backbone.insertion_code,
        component_parent_ptr=component_parent_ptr,
        parent_component_id=parent_component_id,
        batch_size=[],
    )
