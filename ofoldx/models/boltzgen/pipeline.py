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
BoltzGen structure-generation, inverse-folding, analysis and selection Pipelines.

A generation Pipeline materialises the design topology once (``BoltzGenProcessor.plan``) and prepares that one
structure per diffusion batch; each sample is decoded by ``decode_coordinates`` onto the designed components in the
source frame. Inverse folding samples canonical classes and ``produce_design`` builds each candidate's final target;
the SolubleMPNN alternative runs the ProteinMPNN Pipeline on the same positions. The composite
protocol then refolds, scores, analyses, filters, ranks and diversity-selects the candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from typing import Literal, NamedTuple, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ...analysis import (
    AnalysisUnavailableError,
    AnalysisValue,
    CandidateFilter,
    DiversityMetrics,
    DiversitySample,
    NoncovalentContacts,
    PresentAtomContacts,
    PresentAtomHydrophobicPatch,
    PresentAtomSasa,
    PresentAtomUnavailableError,
    TargetDeltaSasa,
    assign_secondary_structure,
    candidate_filter_results,
    count_noncovalent_contacts,
    deduplicate_sequences,
    greedy_diversity_selection,
    largest_hydrophobic_patch_area,
    liability_metrics,
    present_atom_largest_hydrophobic_patch_area,
    present_atom_noncovalent_contacts,
    present_atom_target_delta_sasa,
    rank_candidates,
    structure_coordinates,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.editing import EditReport
from ...data.molecular import AtomSite, ComponentRef, MolecularSystem, MonomerType
from ...data.projection import project_instances
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, token_classes
from ...errors import RequestRef, fail
from ...metrics import (
    amino_acid_fractions,
    binding_site_adherence_curve,
    directional_ipsae,
    secondary_structure_fractions,
    sequence_hydrophobicity,
)
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...pipelines.input import resolve_msa_input, resolve_template_input
from ...results import FoldPrediction, Metric, PairMetric, Score, ScoreResult
from ...templates import Template
from ..boltz2 import Boltz2AffinityScorePipeline, Boltz2Pipeline
from ..conditions import AllowedResidueIdentities, PlacementSpec, TiedResidueIdentities
from ..design import DesignPolicy, PositionSelection, produce_design, resolve_positions
from ..esmfold2.pipeline import EsmFold2Pipeline
from ..preparation import PreparationContext, PreparedInput, decode_coordinates
from ..proteinmpnn.pipeline import ProteinMpnnPipeline
from .model import (
    BoltzGenBinding,
    BoltzGenInput,
    BoltzGenInverseFoldingDesignOutput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingModel,
    BoltzGenMiniformerModel,
    BoltzGenModel,
    BoltzGenSamplingOutput,
    BoltzGenSchema,
    BoltzGenSymmetry,
)
from .processor import (
    BOLTZGEN_LAYOUT,
    CANONICAL_LETTER,
    BoltzGenInverseFoldingProcessor,
    BoltzGenProcessor,
    boltzgen_atom14_names,
    decode_boltzgen_atom14,
    design_instances,
)

BoltzGenProtocol: TypeAlias = Literal[
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
]
BoltzGenDesignCheckpoint: TypeAlias = Literal["diverse", "adherence"]
BoltzGenInverseFoldModel: TypeAlias = Literal["boltzif", "solublempnn"]
BoltzGenInverseFoldSelection: TypeAlias = Literal["auto", "boltzif", "solublempnn"] | None

_PROTOCOLS: tuple[BoltzGenProtocol, ...] = (
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
)
#: The identities the released peptide, nanobody and antibody protocols omit everywhere (canonical letters).
_OMIT_CYSTEINE = ("C",)


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
class BoltzGenAtomicAnalysis:
    r"""Present-atom screening values and independently qualified complete-chemistry values.

    Present-atom contacts reproduce the released approximations, including Hydride's inferred hydrogen geometry;
    their coverage identifies the input atoms and omissions. Strict values are ``None`` when the complete chemistry or
    coordinates are unavailable, with the corresponding reason retained.
    """

    contacts: PresentAtomContacts | None
    contacts_unavailable: str | None
    sasa: PresentAtomSasa | None
    sasa_unavailable: str | None
    strict_contacts: NoncovalentContacts | None
    strict_contacts_unavailable: str | None
    strict_sasa: TargetDeltaSasa | None
    strict_sasa_unavailable: str | None


@dataclass(frozen=True)
class BoltzGenCandidateAnalysis:
    r"""
    Named in-memory analysis values for one coherent selected refold.

    The RMSD fields compare the sequence-assigned original design with the
    selected whole-complex refold using only mutually present, matched atom sites;
    ``rmsd_atom_sites`` records that range. Present-atom screening values use the released approximations and are not
    complete-chemistry metrics. ``original_atomic`` is None when inverse folding is applied, matching the released
    default; ``refolded_atomic`` describes the selected refold independently. ``design_folding_*`` compares the
    extracted design component with its independently selected design-only refold.
    Every designed protein instance remains available separately in ``designed_chains``.

    ``design_rmsd`` fits and measures the exact designed positions;
    ``complex_aligned_design_rmsd`` measures them after fitting the whole complex,
    and ``target_aligned_design_rmsd`` fits the target first. The ``bb_`` fields
    use backbone atoms. Binding-site adherence retains the released 3--9 angstrom
    cutoffs. Original, refolded and target-specific diversity values are available
    through each candidate's ``diversity_sample`` method.
    """

    designed_sequence: str
    designed_chains: tuple[BoltzGenDesignedChainAnalysis, ...]
    num_design: int
    rmsd: float
    design_rmsd: float
    target_rmsd: float
    complex_aligned_design_rmsd: float
    target_aligned_design_rmsd: float
    bb_rmsd: float
    bb_design_rmsd: float
    bb_target_rmsd: float
    bb_complex_aligned_design_rmsd: float
    bb_target_aligned_design_rmsd: float
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
    design_chain_hydrophobicity: float
    liabilities: dict[str, AnalysisValue]
    rmsd_atom_sites: tuple[AtomSite, ...]
    original_atomic: BoltzGenAtomicAnalysis | None
    refolded_atomic: BoltzGenAtomicAnalysis
    hydrophobic_patch: PresentAtomHydrophobicPatch | None
    hydrophobic_patch_unavailable: str | None
    strict_largest_hydrophobic_patch: float | None
    strict_largest_hydrophobic_patch_unavailable: str | None


@dataclass(eq=False)
class BoltzGenDesignCandidate:
    r"""
    One sequence-assigned design and every retained evaluation artifact.

    Attributes:
        structure: The active design target, initially the decoded codesign and, after inverse folding, the
            sequence-redesigned structure.
        codesign_structure: The initial decoded sequence and structure before inverse-folding redesign, when generated
            by an atom14 structure-design Pipeline.
        atom14_coordinates: Raw 14-slot coordinates for each designed position, in ``designed_positions`` order.
        atom14_residue_types: Decoded three-letter identities in that same order.
        atom14_recognized: Whether each identity had an upstream-recognized marker signature. A recognized UNK and
            an invalid signature both retain unknown identity, but remain distinguishable here.
        source_checkpoint: Structure-design checkpoint that generated the codesign.
        designed_positions: Exact designed component occurrences.
        connected_designed_positions: The designed occurrences and every occurrence of their instances and of the
            instances a KNOWN covalent edge joins to them.
        binding_sites: Resolved positive binding-site residue or atom occurrences.
        sequence_symmetry: Inverse-folding sequence-identity groups.
        coordinate_trajectory: Current-coordinate sampler history, when requested.
        denoised_coordinate_trajectory: Denoised ``x0`` sampler history, when requested.
        whole_fold_predictions: Five whole-complex Boltz-2 predictions.
        whole_fold_selected_index: Coherent design-focused prediction index.
        design_fold_predictions: Optional five design-only predictions.
        design_fold_selected_index: Selected design-only prediction index.
        esmfold2_predictions: Independent ESMFold2 predictions in sampling order.
        esmfold2_selected_index: Earliest sample maximizing the ESMFold2 scoring metric.
        esmfold2_score_metric: iPSAE or monomer pTM used for ESMFold2 selection.
        affinity_result: Optional complete Boltz-2 affinity result.
        analysis: Named family analysis values.
        filters: Individual filter outcomes.
    """

    structure: Structure
    source_checkpoint: BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None
    designed_positions: tuple[ComponentRef, ...]
    connected_designed_positions: tuple[ComponentRef, ...]
    codesign_structure: Structure | None = None
    atom14_coordinates: torch.Tensor | None = None
    atom14_residue_types: tuple[str, ...] = ()
    atom14_recognized: tuple[bool, ...] = ()
    binding_sites: tuple[ComponentRef | AtomSite, ...] = ()
    sequence_symmetry: tuple[tuple[ComponentRef, ...], ...] = ()
    coordinate_trajectory: torch.Tensor | None = None
    denoised_coordinate_trajectory: torch.Tensor | None = None
    whole_fold_predictions: tuple[FoldPrediction, ...] = ()
    whole_fold_selected_index: int | None = None
    design_fold_predictions: tuple[FoldPrediction, ...] = ()
    design_fold_selected_index: int | None = None
    esmfold2_predictions: tuple[FoldPrediction, ...] = ()
    esmfold2_selected_index: int | None = None
    esmfold2_score_metric: Literal["ipsae", "ptm"] | None = None
    affinity_result: ScoreResult | None = None
    analysis: BoltzGenCandidateAnalysis | None = None
    filters: tuple[BoltzGenFilterOutcome, ...] = ()

    def copied(self, **values) -> BoltzGenDesignCandidate:
        r"""This candidate with ``values`` replaced (``dataclasses.replace`` is reserved for canonical values)."""
        return BoltzGenDesignCandidate(**({field.name: getattr(self, field.name) for field in fields(self)} | values))

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

    @property
    def esmfold2_refold(self) -> FoldPrediction | None:
        r"""Return the coherent ESMFold2 prediction selected by its scientific score."""

        if self.esmfold2_selected_index is None:
            return None
        return self.esmfold2_predictions[self.esmfold2_selected_index]

    def diversity_sample(self, target_id: str, *, compute_lddt: bool = True) -> DiversitySample:
        r"""
        Return the named in-memory values used by diversity aggregation.

        ``compute_lddt=False`` omits the quadratic optional lDDT conditions
        without substituting another metric.
        """

        selected = self.whole_refold
        analysis = self.analysis
        if selected is None or analysis is None:
            fail("value_invalid", RequestRef("candidate", "without folding and analysis"))
        return DiversitySample(
            target_id=target_id,
            sequence=analysis.designed_sequence,
            original_coordinates=_design_ca_coordinates(self.structure, self.designed_positions),
            refolded_coordinates=_design_ca_coordinates(selected.structure, self.designed_positions),
            metrics=DiversityMetrics(
                rmsd=analysis.rmsd,
                design_rmsd=analysis.design_rmsd,
                target_aligned_design_rmsd=analysis.target_aligned_design_rmsd,
                lddt_intra_design=(
                    structure_design_lddt(
                        self.structure, selected.structure, designed_positions=self.designed_positions
                    )
                    if compute_lddt
                    else None
                ),
                bb_rmsd=analysis.bb_rmsd,
                bb_design_rmsd=analysis.bb_design_rmsd,
                bb_target_aligned_design_rmsd=analysis.bb_target_aligned_design_rmsd,
                bb_lddt_intra_design=(
                    structure_design_lddt(
                        self.structure,
                        selected.structure,
                        designed_positions=self.designed_positions,
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


class BoltzGenBackbonePipelineInput(NamedTuple):
    r"""One materialised design topology, its resolved conditions, its bound evidence and the report of its link
    edit."""

    structure: Structure
    conditions: tuple[BoltzGenSchema, ...]
    generated: tuple[ComponentRef, ...]
    msa: Mapping[str, MSA] | MsaCollection | None
    templates: tuple[Template, ...]
    edit: EditReport
    library: ComponentLibrary | None


def plan_boltzgen_design(
    processor: BoltzGenProcessor,
    structure: Structure | None = None,
    *,
    placement: PlacementSpec | None = None,
    total_len: int | None = None,
    conditions: Sequence[BoltzGenSchema] = (),
    msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
    templates: Sequence[Template] | str | PathLike[str] = (),
    library: ComponentLibrary | None = None,
) -> BoltzGenBackbonePipelineInput:
    r"""Materialise the design topology (``BoltzGenProcessor.plan``) and bind the MSA and template evidence to it."""

    planned, resolved, generated, edit = processor.plan(
        structure, placement=placement, total_len=total_len, conditions=conditions, library=library
    )
    return BoltzGenBackbonePipelineInput(
        structure=planned,
        conditions=resolved,
        generated=generated,
        msa=resolve_msa_input(msa, planned.system, monomer_types=(MonomerType.PROTEIN,)),
        templates=resolve_template_input(templates),
        edit=edit,
        library=library,
    )


class BoltzGenBackbonePipeline(PairedPipeline[BoltzGenModel, BoltzGenProcessor]):
    r"""Independently callable sequence-and-structure design Pipeline for a full BoltzGen checkpoint."""

    task = "design"

    model_architecture = "boltzgen"

    def prepare(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
    ) -> BoltzGenBackbonePipelineInput:
        r"""
        Materialize the design topology and bind its evidence.

        Notes:
            Before sampling, generated protein positions are represented as unknown residues. RNA and DNA are fixed
            context only. ``ExistingSpan`` retains source rows during an extension without creating a new copy.

            Explicit covalent conditions use the shared partial final-bond construction. Requested edges and departing
            atoms
            retain intrinsic endpoint identity; charge, compressed hydrogen, aromaticity and atom stereochemistry become
            unknown where the construction does not determine them. Source charge provenance survives only when the
            unchanged
            final charge is still supported; coordinate observations are transported independently. Requested toolkit
            reference
            calculations require their own chemical prerequisites.

            The atom14 representation separates canonical physical atom identity from coordinate-training source. Valid
            synthetic entries may have coordinate truth and canonical index -1, but create no chemical atoms in the
            result.
            Designed positions use the declared amino-acid representation and require its actual named atoms. A
            biological
            annotation alone does not establish chemical identity.

            After sampling, the fourteen predicted slots identify a residue by the upstream marker rule. Recognized
            canonical residues receive the predicted coordinates of their physical atoms; marker slots remain available
            separately and never become chemical atoms.
        """

        return plan_boltzgen_design(
            self.processor,
            structure,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
        )

    def design(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        num_recycles: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        candidate_offset: int = 0,
        num_total_candidates: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate sequence-and-structure candidates with one full BoltzGen checkpoint.

        ``step_scale_schedule`` and ``noise_scale_schedule`` contain shared
        per-transition coefficients and take precedence over constant overrides.

        ``candidate_offset`` and ``num_total_candidates`` define the released global
        quarter schedule only. Separately executed ranges must retain complete
        ``diffusion_batch_size`` groups and use independently derived seeds or
        generators; these values do not reproduce a monolithic call's RNG consumption.
        """

        if num_candidates <= 0:
            fail("value_invalid", RequestRef("num_candidates", str(num_candidates)))
        num_total_candidates = num_candidates if num_total_candidates is None else num_total_candidates
        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch_size = _diffusion_batch_size(num_total_candidates, diffusion_batch_size)
        if candidate_offset < 0 or candidate_offset + num_candidates > num_total_candidates:
            fail("value_invalid", RequestRef("candidate_offset", str(candidate_offset)), total=num_total_candidates)
        prepared = self.prepare(
            structure,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
        )
        output = self.forward(
            prepared,
            num_candidates=num_candidates,
            diffusion_batch_size=batch_size,
            num_sampling_steps=num_sampling_steps,
            num_recycles=num_recycles,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            candidate_offset=candidate_offset,
            num_total_candidates=num_total_candidates,
            generator=generator,
            rng=None if seed is None else np.random.default_rng(seed),
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: BoltzGenBackbonePipelineInput,
        *,
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        num_recycles: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        candidate_offset: int = 0,
        num_total_candidates: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Prepare, sample and immediately decode each independent diffusion batch."""

        if num_candidates <= 0:
            fail("value_invalid", RequestRef("num_candidates", str(num_candidates)))
        num_total_candidates = num_candidates if num_total_candidates is None else num_total_candidates
        generator = resolve_generator(seed, generator, device=self.processor.device)
        return _generate_backbones(
            input,
            sources=((None, self, num_candidates),),
            candidate_offset=candidate_offset,
            num_total_candidates=num_total_candidates,
            diffusion_batch_size=_diffusion_batch_size(num_total_candidates, diffusion_batch_size),
            num_sampling_steps=num_sampling_steps,
            num_recycles=num_recycles,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            generator=generator,
            rng=rng,
        )

    def postprocess(
        self, input: BoltzGenBackbonePipelineInput, output: tuple[BoltzGenDesignCandidate, ...]
    ) -> BoltzGenCandidateResult:
        r"""Collect decoded sequence-and-structure candidates in generation order."""

        return BoltzGenCandidateResult(candidates=output)

    __call__ = task_call(design)


class BoltzGenMiniformerPipeline(PairedPipeline[BoltzGenMiniformerModel, BoltzGenProcessor]):
    r"""
    Independent structure-design pipeline for the released structure-trained small model.
    """

    task = "design"

    model_architecture = "boltzgen_miniformer"

    def prepare(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
    ) -> BoltzGenBackbonePipelineInput:
        r"""Materialize the miniformer topology and bind its evidence."""

        return plan_boltzgen_design(
            self.processor,
            structure,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
        )

    def design(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        num_recycles: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate sequence-and-structure candidates with the structure-trained small checkpoint.

        ``step_scale_schedule`` and ``noise_scale_schedule`` contain shared
        per-transition coefficients and take precedence over constant overrides.

        This operation owns one complete candidate range. It supports internal
        diffusion batching but not independently executed candidate shards.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch_size = _diffusion_batch_size(num_candidates, diffusion_batch_size)
        prepared = self.prepare(
            structure,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
        )
        output = self.forward(
            prepared,
            num_candidates=num_candidates,
            diffusion_batch_size=batch_size,
            num_sampling_steps=num_sampling_steps,
            num_recycles=num_recycles,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            generator=generator,
            rng=None if seed is None else np.random.default_rng(seed),
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: BoltzGenBackbonePipelineInput,
        *,
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        num_recycles: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Prepare, sample and immediately decode each independent diffusion batch."""

        generator = resolve_generator(seed, generator, device=self.processor.device)
        return _generate_backbones(
            input,
            sources=(("structuretrained_small", self, num_candidates),),
            num_total_candidates=num_candidates,
            diffusion_batch_size=_diffusion_batch_size(num_candidates, diffusion_batch_size),
            num_sampling_steps=num_sampling_steps,
            num_recycles=num_recycles,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            generator=generator,
            rng=rng,
        )

    def postprocess(
        self, input: BoltzGenBackbonePipelineInput, output: tuple[BoltzGenDesignCandidate, ...]
    ) -> BoltzGenCandidateResult:
        r"""Collect decoded sequence-and-structure candidates in generation order."""

        return BoltzGenCandidateResult(candidates=output)

    __call__ = task_call(design)


class BoltzGenInverseFoldingPipelineInput(NamedTuple):
    r"""One backbone candidate, its sequence-design scope and the conditions of its identities."""

    candidate: BoltzGenDesignCandidate
    msa: Mapping[str, MSA] | MsaCollection | None
    conditions: tuple[AllowedResidueIdentities | TiedResidueIdentities, ...]
    global_omit_tokens: tuple[str, ...]


class BoltzGenInverseFoldingPipeline(PairedPipeline[BoltzGenInverseFoldingModel, BoltzGenInverseFoldingProcessor]):
    r"""
    Independently callable BoltzGen inverse-folding pipeline.
    """

    task = "design"

    model_architecture = "boltzgen_ifold"

    def prepare(
        self,
        structure: BoltzGenDesignCandidate | Structure,
        *,
        design: PositionSelection | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry: Sequence[Sequence[ComponentRef]] | None = None,
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
    ) -> BoltzGenInverseFoldingPipelineInput:
        r"""Resolve the backbone, its design scope, its symmetry groups and its MSA."""

        candidate = _inverse_folding_candidate(structure, design=design, symmetry=symmetry)
        tied = candidate.sequence_symmetry if tie_symmetric_sequences else ()
        return BoltzGenInverseFoldingPipelineInput(
            candidate=candidate,
            msa=resolve_msa_input(msa, candidate.structure.system, monomer_types=(MonomerType.PROTEIN,)),
            conditions=(*conditions, *((TiedResidueIdentities(tuple(tied)),) if tied else ())),
            global_omit_tokens=tuple(global_omit_tokens),
        )

    def design(
        self,
        structure: BoltzGenDesignCandidate | Structure,
        *,
        design: PositionSelection | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry: Sequence[Sequence[ComponentRef]] | None = None,
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Assign sequences to a fixed backbone without evaluation stages.

        A candidate input retains its checkpoint, binding, symmetry and
        trajectory metadata. An explicit ``design`` selection or ``symmetry`` replaces
        the corresponding candidate fields. Results intentionally clear any
        downstream folding, affinity, analysis and filtering state because the
        sequence has changed.
        """

        if num_sequences <= 0:
            fail("value_invalid", RequestRef("num_sequences", str(num_sequences)))
        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            symmetry=symmetry,
            msa=msa,
            global_omit_tokens=global_omit_tokens,
            tie_symmetric_sequences=tie_symmetric_sequences,
        )
        output = self.forward(
            prepared,
            temperature=temperature,
            num_sequences=num_sequences,
            generator=generator,
            rng=None if seed is None else np.random.default_rng(seed),
            library=library,
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: BoltzGenInverseFoldingPipelineInput,
        *,
        temperature: float | None = 0.1,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        rng: np.random.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Prepare, sample and immediately produce the design of each independent sequence."""

        if num_sequences <= 0:
            fail("value_invalid", RequestRef("num_sequences", str(num_sequences)))
        generator = resolve_generator(seed, generator, device=self.processor.device)
        candidate = input.candidate
        library = configured_library() if library is None else library
        designed = []
        for _ in range(num_sequences):
            prepared = self.processor.prepare_input(
                candidate.structure,
                design=PositionSelection(components=candidate.designed_positions),
                conditions=input.conditions,
                global_omit_tokens=input.global_omit_tokens,
                msa=input.msa,
                generator=rng,
            )
            output = self.model.design(prepared.features, temperature=temperature, generator=generator)
            designed.append(_produce_sequence(candidate, prepared, output, library=library))
        return tuple(designed)

    def postprocess(
        self, input: BoltzGenInverseFoldingPipelineInput, output: tuple[BoltzGenDesignCandidate, ...]
    ) -> BoltzGenCandidateResult:
        r"""Collect the sequence-assigned candidates in sampling order."""

        return BoltzGenCandidateResult(candidates=output)

    __call__ = task_call(design)


class BoltzGenPipelineInput(NamedTuple):
    r"""Prepared child generation plans, released protocol, counts and shared RNG.

    Diverse and adherence inputs retain their materialized structures, canonical
    conditions, generated component references, MSA and templates. Candidate and
    selection counts remain attached to this preparation through postprocessing.
    The generator advances through all children of the complete workflow.
    """

    diverse_input: BoltzGenBackbonePipelineInput
    adherence_input: BoltzGenBackbonePipelineInput
    protocol: BoltzGenProtocol
    inverse_model: BoltzGenInverseFoldModel | None
    num_candidates: int
    num_sequences_per_backbone: int
    diffusion_batch_size: int
    budget: int
    top_budget: int
    generator: torch.Generator | None
    rng: np.random.Generator | None


class BoltzGenPipeline(CompositePipeline):
    r"""
    BoltzGen design with soluble inverse folding and independent folding evaluations.

    The repository topology contains diverse and adherence design, both
    inverse-folding models, Boltz-2 folding/affinity and ESMFold2 under their
    named subdirectories.

    The seven child paths are ``design_diverse/``, ``adherence/``, ``ifold/``,
    ``soluble/``, ``fold/``, ``affinity/`` and ``esmfold2/``. They contain the
    diverse generator, adherence generator, BoltzIF, SolubleMPNN, Boltz-2 folder,
    Boltz-2 affinity scorer and ESMFold2 folder, respectively.
    """

    task = "design"

    component_layout = (
        ComponentSpec("diverse_generator", "design_diverse", BoltzGenBackbonePipeline),
        ComponentSpec("adherence_generator", "adherence", BoltzGenBackbonePipeline),
        ComponentSpec("inverse_folder", "ifold", BoltzGenInverseFoldingPipeline),
        ComponentSpec("soluble_designer", "soluble", ProteinMpnnPipeline),
        ComponentSpec("folder", "fold", Boltz2Pipeline),
        ComponentSpec("affinity_scorer", "affinity", Boltz2AffinityScorePipeline),
        ComponentSpec("esmfold2", "esmfold2", EsmFold2Pipeline),
    )

    def __init__(
        self,
        *,
        diverse_generator: BoltzGenBackbonePipeline,
        adherence_generator: BoltzGenBackbonePipeline,
        inverse_folder: BoltzGenInverseFoldingPipeline,
        soluble_designer: ProteinMpnnPipeline,
        folder: Boltz2Pipeline,
        affinity_scorer: Boltz2AffinityScorePipeline,
        esmfold2: EsmFold2Pipeline,
    ) -> None:
        super().__init__(
            {
                "diverse_generator": diverse_generator,
                "adherence_generator": adherence_generator,
                "inverse_folder": inverse_folder,
                "soluble_designer": soluble_designer,
                "folder": folder,
                "affinity_scorer": affinity_scorer,
                "esmfold2": esmfold2,
            }
        )

    @property
    def diverse_generator(self) -> BoltzGenBackbonePipeline:
        return cast(BoltzGenBackbonePipeline, self.components["diverse_generator"])

    @property
    def adherence_generator(self) -> BoltzGenBackbonePipeline:
        return cast(BoltzGenBackbonePipeline, self.components["adherence_generator"])

    @property
    def inverse_folder(self) -> BoltzGenInverseFoldingPipeline:
        return cast(BoltzGenInverseFoldingPipeline, self.components["inverse_folder"])

    @property
    def soluble_designer(self) -> ProteinMpnnPipeline:
        return cast(ProteinMpnnPipeline, self.components["soluble_designer"])

    @property
    def esmfold2(self) -> EsmFold2Pipeline:
        return cast(EsmFold2Pipeline, self.components["esmfold2"])

    @property
    def folder(self) -> Boltz2Pipeline:
        return cast(Boltz2Pipeline, self.components["folder"])

    @property
    def affinity_scorer(self) -> Boltz2AffinityScorePipeline:
        return cast(Boltz2AffinityScorePipeline, self.components["affinity_scorer"])

    def prepare(
        self,
        structure: Structure | None = None,
        *,
        protocol: BoltzGenProtocol = "protein-anything",
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 10_000,
        num_sequences_per_backbone: int = 1,
        diffusion_batch_size: int | None = None,
        inverse_fold_model: BoltzGenInverseFoldSelection = "auto",
        budget: int = 30,
        top_budget: int = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenPipelineInput:
        r"""
        Resolve the released protocol and the generation plan of both design checkpoints.

        The plans materialize source and generated occurrences once. Existing
        spans keep their original positions rather than creating copies. Each
        diffusion batch reuses its child's canonical plan and correspondence.
        Preparation retains the selection budgets for ``postprocess``; explicit
        postprocessing overrides replace those retained values.

        ``inverse_fold_model="auto"`` selects the protocol's inverse folder.
        ``None`` retains the original codesign, with one sequence per backbone.
        """

        if protocol not in _PROTOCOLS:
            fail("value_invalid", RequestRef("protocol", str(protocol)))
        if num_candidates <= 0:
            fail("value_invalid", RequestRef("num_candidates", str(num_candidates)))
        if num_sequences_per_backbone <= 0:
            fail("value_invalid", RequestRef("num_sequences_per_backbone", str(num_sequences_per_backbone)))
        if inverse_fold_model is None and num_sequences_per_backbone != 1:
            fail("value_invalid", RequestRef("num_sequences_per_backbone", str(num_sequences_per_backbone)))
        if budget <= 0 or top_budget <= 0:
            fail("value_invalid", RequestRef("budget", f"{budget}, {top_budget}"))
        inverse_model = (
            None
            if inverse_fold_model is None
            else _inverse_fold_model(protocol, None if inverse_fold_model == "auto" else inverse_fold_model)
        )
        generator = resolve_generator(seed, generator, device=self.diverse_generator.processor.device)
        batch_size = _diffusion_batch_size(num_candidates, diffusion_batch_size)
        return BoltzGenPipelineInput(
            diverse_input=self.diverse_generator.prepare(
                structure,
                placement=placement,
                total_len=total_len,
                conditions=conditions,
                msa=msa,
                templates=templates,
                library=library,
            ),
            adherence_input=self.adherence_generator.prepare(
                structure,
                placement=placement,
                total_len=total_len,
                conditions=conditions,
                msa=msa,
                templates=templates,
                library=library,
            ),
            protocol=protocol,
            inverse_model=inverse_model,
            num_candidates=num_candidates,
            num_sequences_per_backbone=num_sequences_per_backbone,
            diffusion_batch_size=batch_size,
            budget=budget,
            top_budget=top_budget,
            generator=generator,
            rng=None if seed is None else np.random.default_rng(seed),
        )

    def forward(
        self,
        input: BoltzGenPipelineInput,
        *,
        num_sampling_steps: int = 500,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] | None = None,
        conditions: Sequence[AllowedResidueIdentities] = (),
        tie_symmetric_sequences: bool = True,
        affinity_ligand: str | None = None,
        scoring_target_instances: Sequence[str] | None = None,
        esmfold2_num_samples: int = 5,
        esmfold2_num_recycles: int = 21,
        esmfold2_num_sampling_steps: int = 200,
        esmfold2_language_model_dropout: float = 0.3,
    ) -> tuple[BoltzGenDesignCandidate, ...]:
        r"""Generate, optionally inverse-fold and evaluate candidates in released stage order.

        Diverse candidates precede adherence candidates. Each backbone batch is
        prepared, sampled and decoded before inverse folding with its associated
        MSA; sequence preparation retains the design positions and symmetry.
        Every backbone finishes inverse folding before any Boltz-2 folding call,
        and folding finishes before affinity scoring.

        When inverse folding is disabled, sequence constraints cannot be supplied and a decoded unknown identity
        refuses the complete batch before evaluation. The independently callable design Pipeline retains raw unknown
        candidates for inspection.
        """

        protocol = input.protocol
        num_candidates = input.num_candidates
        generator, rng = input.generator, input.rng
        if input.inverse_model is None and (conditions or global_omit_tokens):
            fail("value_invalid", RequestRef("inverse_fold_model", "sequence constraints without inverse folding"))
        diverse_count = (num_candidates + 1) // 2
        backbones: list[BoltzGenDesignCandidate] = []
        sources = (
            ("diverse", self.diverse_generator, input.diverse_input, diverse_count, 0),
            (
                "adherence",
                self.adherence_generator,
                input.adherence_input,
                num_candidates - diverse_count,
                diverse_count,
            ),
        )
        for source, pipeline, prepared, source_count, candidate_offset in sources:
            if not source_count:
                continue
            output = pipeline.forward(
                prepared,
                num_candidates=source_count,
                candidate_offset=candidate_offset,
                num_total_candidates=num_candidates,
                diffusion_batch_size=input.diffusion_batch_size,
                num_sampling_steps=num_sampling_steps,
                num_recycles=4,
                step_scale=step_scale,
                noise_scale=noise_scale,
                step_scale_schedule=step_scale_schedule,
                noise_scale_schedule=noise_scale_schedule,
                return_trajectory=return_trajectory,
                generator=generator,
                rng=rng,
            )
            for candidate in pipeline.postprocess(prepared, output).candidates:
                candidate.source_checkpoint = cast(BoltzGenDesignCheckpoint, source)
                backbones.append(candidate)

        omit_tokens = (
            _OMIT_CYSTEINE
            if global_omit_tokens is None and protocol in {"peptide-anything", "nanobody-anything", "antibody-anything"}
            else (() if global_omit_tokens is None else tuple(global_omit_tokens))
        )
        msa = input.diverse_input.msa
        if input.inverse_model is None:
            if any("UNK" in candidate.atom14_residue_types for candidate in backbones):
                fail("value_invalid", RequestRef("atom14", "unknown identity requires inverse folding"))
            candidates = tuple(backbones)
        else:
            candidates = tuple(
                candidate
                for backbone in backbones
                for candidate in self.inverse_fold(
                    backbone,
                    protocol=protocol,
                    inverse_fold_model=input.inverse_model,
                    conditions=conditions,
                    msa=msa,
                    temperature=temperature,
                    global_omit_tokens=omit_tokens,
                    tie_symmetric_sequences=tie_symmetric_sequences,
                    num_sequences=input.num_sequences_per_backbone,
                    generator=generator,
                ).candidates
            )

        design_folding = protocol in {"protein-anything", "protein-small_molecule"}
        candidates = tuple(
            self._fold_candidate(candidate, protocol=protocol, design_folding=design_folding, generator=generator)
            for candidate in candidates
        )

        if protocol == "protein-small_molecule":
            for candidate in candidates:
                candidate.affinity_result = self.affinity_scorer.score(
                    candidate.structure,
                    ligand=affinity_ligand,
                    num_samples=5,
                    num_recycles=4,
                    num_sampling_steps=200,
                    molecular_weight_correction=True,
                    generator=generator,
                )
        else:
            for candidate in candidates:
                self.score_esmfold2(
                    candidate,
                    scoring_mode="redesign" if protocol == "protein-redesign" else "binder",
                    target_instances=scoring_target_instances,
                    num_samples=esmfold2_num_samples,
                    num_recycles=esmfold2_num_recycles,
                    num_sampling_steps=esmfold2_num_sampling_steps,
                    language_model_dropout=esmfold2_language_model_dropout,
                    generator=generator,
                )

        return candidates

    def postprocess(
        self,
        input: BoltzGenPipelineInput,
        output: tuple[BoltzGenDesignCandidate, ...],
        *,
        budget: int | None = None,
        top_budget: int | None = None,
        alpha: float | None = None,
        refolding_rmsd_threshold: float | None = None,
        filter_cysteine: bool | None = None,
        filter_binding_site: bool = False,
        filter_target_alignment: bool = False,
        filter_biased: bool = True,
        metric_weights: Mapping[str, float | None] | None = None,
        additional_filters: Sequence[CandidateFilter] = (),
        size_buckets: Sequence[tuple[int, int, int]] = (),
        liability_modality: Literal["antibody", "peptide"] | None = None,
        liability_peptide_type: Literal["linear", "cyclic"] = "linear",
    ) -> BoltzGenDesignResult:
        r"""
        Analyze and filter evaluated candidates, then rank and select them.

        Non-affinity protocols retain all ESMFold2 samples and rank by the
        selected sample's iPSAE; monomer redesign uses pTM. Antibody and nanobody
        protocols default to antibody liability analysis. ``liability_modality``
        and ``liability_peptide_type`` explicitly select a different analysis.

        Notes:
            Sequence assignment retains admitted source chemistry when its biological class is sampled. A different
            class
            creates a replacement with backbone guide coordinates and unavailable new side chains.

            Reference conformers and observations use separate availability. Reference augmentation uses conformer
            availability; source observations, template centers, frames and placement compatibility retain their own
            masks.
            Frames require available, nondegenerate defining points. Missing coordinate payload is masked before
            geometry and
            dtype conversion; active nonfinite values and finite values outside model range raise errors.

            Regeneration clears occupancy/B observations and old hidden-atom coordinates within regenerated components
            while
            retaining chemical vertices and supported source charge provenance. Unrepresented atoms remain chemically
            present
            with unavailable coordinates. Open connectivity and explicit negative relations can pass through coordinate
            features; positive relation features do not establish known absence. Missing chemistry and known unsupported
            representations raise distinct errors. Checkpoint feature conventions do not establish scientific identity
            or
            coordinate availability.
        """

        budget = input.budget if budget is None else budget
        top_budget = input.top_budget if top_budget is None else top_budget
        if budget <= 0 or top_budget <= 0:
            fail("value_invalid", RequestRef("budget", f"{budget}, {top_budget}"))
        protocol = input.protocol
        candidates = output
        design_folding = protocol in {"protein-anything", "protein-small_molecule"}
        analyze_hydrophobic_patch = protocol not in {"peptide-anything", "nanobody-anything", "antibody-anything"}
        for candidate in candidates:
            candidate.analysis = self._analyze_candidate(
                candidate,
                protocol=protocol,
                from_inverse_folding=input.inverse_model is not None,
                library=input.diverse_input.library,
                largest_hydrophobic_patch=analyze_hydrophobic_patch,
                liability_modality=(
                    "antibody"
                    if liability_modality is None and protocol in {"antibody-anything", "nanobody-anything"}
                    else ("peptide" if liability_modality is None else liability_modality)
                ),
                liability_peptide_type=liability_peptide_type,
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
        weights = _ranking_metric_weights(protocol, from_inverse_folding=input.inverse_model is not None)
        if metric_weights is not None:
            for name, weight in metric_weights.items():
                if weight is None:
                    weights.pop(name, None)
                else:
                    weights[name] = weight
        if not weights or any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
            fail("value_invalid", RequestRef("metric_weights", str(weights)), reason="invalid_ranking_weights")
        tie_breaker = (
            "design_to_target_iptm" if protocol in {"protein-small_molecule", "protein-redesign"} else "esmfold2_score"
        )
        for candidate in candidates:
            values = _candidate_values(candidate)
            _require_candidate_values(values, tuple(item.feature for item in all_filters), "filter", candidate)
            _require_candidate_values(values, (*weights, tie_breaker), "ranking", candidate)
            filter_results = candidate_filter_results(values, all_filters)
            candidate.filters = tuple(
                BoltzGenFilterOutcome(
                    filter=candidate_filter,
                    value=float(values[candidate_filter.feature]),
                    passed=filter_results[f"pass_{candidate_filter.feature}_filter"],
                )
                for candidate_filter in all_filters
            )

        return _select_candidates(
            candidates,
            metric_weights=weights,
            tie_breaker=tie_breaker,
            budget=budget,
            top_budget=top_budget,
            alpha=(0.01 if alpha is None and protocol == "peptide-anything" else (0.001 if alpha is None else alpha)),
            size_buckets=size_buckets,
        )

    def design(
        self,
        structure: Structure | None = None,
        *,
        protocol: BoltzGenProtocol = "protein-anything",
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 10_000,
        num_sequences_per_backbone: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int = 500,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        temperature: float | None = 0.1,
        inverse_fold_model: BoltzGenInverseFoldSelection = "auto",
        global_omit_tokens: Sequence[str] | None = None,
        identity_conditions: Sequence[AllowedResidueIdentities] = (),
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
        scoring_target_instances: Sequence[str] | None = None,
        esmfold2_num_samples: int = 5,
        esmfold2_num_recycles: int = 21,
        esmfold2_num_sampling_steps: int = 200,
        esmfold2_language_model_dropout: float = 0.3,
        liability_modality: Literal["antibody", "peptide"] | None = None,
        liability_peptide_type: Literal["linear", "cyclic"] = "linear",
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Run one of the six released end-to-end BoltzGen protocols.

        All backbones finish inverse folding before any Boltz-2 fold call, and
        every fold call finishes before affinity scoring begins.

        Args:
            structure: The target or context; ``None`` requests a fully generated assembly.
            protocol: Released scientific protocol.
            placement: Existing/generated segment placement.
            total_len: Total token budget used to realize generated segments.
            conditions: BoltzGen conditioning records.
            msa: Optional MSA pools by instance, an exact plan, or an alignment path.
            templates: Optional structural templates or a native template artifact.
            library: The definitions generated and designed components are built from.
            num_candidates: Total backbones split between diverse and adherence.
            num_sequences_per_backbone: Inverse-folded sequences per backbone; must be one when inverse folding is off.
            diffusion_batch_size: Designs sharing one Processor/trunk realization.
            num_sampling_steps: Structure-design diffusion transitions.
            step_scale: Optional fixed structure-design Euler-step scale.
            noise_scale: Optional fixed structure-design noise scale.
            step_scale_schedule: Shared per-transition Euler-step coefficients.
            noise_scale_schedule: Shared per-transition churn-noise coefficients.
            return_trajectory: Retain both structure-design sampler histories.
            temperature: Inverse-folding sampling temperature.
            inverse_fold_model: ``"auto"`` chooses the protocol default, ``"boltzif"`` or ``"solublempnn"``
                selects an inverse folder, and ``None`` keeps the original codesign sequence and coordinates.
            global_omit_tokens: Canonical letters excluded at every designed position. ``None`` selects
                the protocol default.
            identity_conditions: Allowed-identity conditions of the inverse folder.
            tie_symmetric_sequences: Tie declared symmetric sequence positions.
            budget: Diversity-selected candidate count.
            top_budget: Best-ranked candidate count.
            alpha: Optional quality/diversity trade-off.
            refolding_rmsd_threshold: Optional protocol RMSD-filter override.
            filter_cysteine: Optional protocol cysteine-filter override.
            filter_binding_site: Apply the released binding-site filter.
            filter_target_alignment: Apply the released target-alignment filter.
            filter_biased: Apply released A/G/E/L/V composition filters.
            affinity_ligand: Molecule instance scored by ``protein-small_molecule``. May be omitted for one
                unambiguous non-polymer instance.
            metric_weights: Direct ranking-weight overrides; ``None`` removes a protocol metric.
            additional_filters: Additional explicit candidate filters.
            size_buckets: Optional ``(minimum, maximum, count)`` diversity budgets.
            scoring_target_instances: Target polymer chains used for ESMFold2 binder scoring.
            esmfold2_num_samples: Independent ESMFold2 samples per candidate.
            esmfold2_num_recycles: ESMFold2 recurrent trunk passes.
            esmfold2_num_sampling_steps: ESMFold2 diffusion transitions.
            esmfold2_language_model_dropout: ESMFold2 language-model pair-feature dropout.
            liability_modality: Antibody or peptide analysis; omission uses the protocol default.
            liability_peptide_type: Linear or cyclic peptide liability analysis.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            Every analyzed candidate plus stable ranked, top, and diversity-selected indices.
        """

        prepared = self.prepare(
            structure,
            protocol=protocol,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
            num_candidates=num_candidates,
            num_sequences_per_backbone=num_sequences_per_backbone,
            diffusion_batch_size=diffusion_batch_size,
            inverse_fold_model=inverse_fold_model,
            budget=budget,
            top_budget=top_budget,
            seed=seed,
            generator=generator,
        )
        output = self.forward(
            prepared,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            temperature=temperature,
            global_omit_tokens=global_omit_tokens,
            conditions=identity_conditions,
            tie_symmetric_sequences=tie_symmetric_sequences,
            affinity_ligand=affinity_ligand,
            scoring_target_instances=scoring_target_instances,
            esmfold2_num_samples=esmfold2_num_samples,
            esmfold2_num_recycles=esmfold2_num_recycles,
            esmfold2_num_sampling_steps=esmfold2_num_sampling_steps,
            esmfold2_language_model_dropout=esmfold2_language_model_dropout,
        )
        return self.postprocess(
            prepared,
            output,
            budget=budget,
            top_budget=top_budget,
            alpha=alpha,
            refolding_rmsd_threshold=refolding_rmsd_threshold,
            filter_cysteine=filter_cysteine,
            filter_binding_site=filter_binding_site,
            filter_target_alignment=filter_target_alignment,
            filter_biased=filter_biased,
            metric_weights=metric_weights,
            additional_filters=additional_filters,
            size_buckets=size_buckets,
            liability_modality=liability_modality,
            liability_peptide_type=liability_peptide_type,
        )

    __call__ = task_call(design)

    def design_backbones(
        self,
        structure: Structure | None = None,
        *,
        checkpoint: BoltzGenDesignCheckpoint = "diverse",
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        library: ComponentLibrary | None = None,
        num_candidates: int = 1,
        diffusion_batch_size: int | None = None,
        num_sampling_steps: int | None = 500,
        num_recycles: int | None = 4,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        step_scale_schedule: torch.Tensor | None = None,
        noise_scale_schedule: torch.Tensor | None = None,
        return_trajectory: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Generate sequence-and-structure candidates with one explicitly selected released design model.
        Per-transition scale tensors are shared across its candidate groups.
        """

        result = self._design_pipeline(checkpoint).design(
            structure,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
            msa=msa,
            templates=templates,
            library=library,
            num_candidates=num_candidates,
            diffusion_batch_size=diffusion_batch_size,
            num_sampling_steps=num_sampling_steps,
            num_recycles=num_recycles,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            seed=seed,
            generator=generator,
        )
        for candidate in result.candidates:
            candidate.source_checkpoint = checkpoint
        return result

    def inverse_fold(
        self,
        structure: BoltzGenDesignCandidate | Structure,
        *,
        protocol: BoltzGenProtocol = "protein-anything",
        inverse_fold_model: BoltzGenInverseFoldModel | None = None,
        design: PositionSelection | None = None,
        conditions: Sequence[AllowedResidueIdentities] = (),
        symmetry: Sequence[Sequence[ComponentRef]] | None = None,
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        temperature: float | None = 0.1,
        global_omit_tokens: Sequence[str] = (),
        tie_symmetric_sequences: bool = True,
        num_sequences: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> BoltzGenCandidateResult:
        r"""
        Assign sequences to a fixed backbone without running evaluation stages.

        BoltzIF designs with the family's own inverse folder; SolubleMPNN designs the same positions with the
        ProteinMPNN Pipeline, whose candidates are the targets of its own design producer.
        """

        if _inverse_fold_model(protocol, inverse_fold_model) == "boltzif":
            return self.inverse_folder.design(
                structure,
                design=design,
                conditions=conditions,
                symmetry=symmetry,
                msa=msa,
                temperature=temperature,
                global_omit_tokens=global_omit_tokens,
                tie_symmetric_sequences=tie_symmetric_sequences,
                num_sequences=num_sequences,
                seed=seed,
                generator=generator,
                library=library,
            )
        if num_sequences <= 0:
            fail("value_invalid", RequestRef("num_sequences", str(num_sequences)))
        candidate = _inverse_folding_candidate(structure, design=design, symmetry=symmetry)
        tied = candidate.sequence_symmetry if tie_symmetric_sequences else ()
        result = self.soluble_designer.design(
            candidate.structure,
            design=PositionSelection(components=candidate.designed_positions),
            conditions=(*conditions, *((TiedResidueIdentities(tuple(tied)),) if tied else ())),
            num_candidates=num_sequences,
            temperature=0.1 if temperature is None else temperature,
            global_omit_tokens=tuple(global_omit_tokens),
            seed=seed,
            generator=generator,
            library=library,
        )
        return BoltzGenCandidateResult(
            candidates=tuple(candidate.copied(structure=designed.target) for designed in result.candidates)
        )

    def score_esmfold2(
        self,
        candidate: BoltzGenDesignCandidate,
        *,
        scoring_mode: Literal["binder", "redesign"] = "binder",
        target_instances: Sequence[str] | None = None,
        num_samples: int = 5,
        num_recycles: int = 21,
        num_sampling_steps: int = 200,
        language_model_dropout: float = 0.3,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignCandidate:
        r"""Refold a design independently with ESMFold2 and retain its interaction scores.

        Binder scoring selects separate target polymer chains; redesign scores
        every retained polymer chain against the rest, or native pTM for one chain.
        ``target_instances`` restricts those target chains; the complete design
        workflow exposes the corresponding choice as ``scoring_target_instances``.
        """

        result = self.esmfold2.fold(
            candidate.structure,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=0.0,
            language_model_dropout=language_model_dropout,
            seed=seed,
            generator=generator,
        )
        designed_instances = {reference.instance for reference in candidate.designed_positions}
        for prediction in result.predictions:
            _esmfold2_candidate_scores(
                prediction,
                designed_instances=designed_instances,
                target_instances=target_instances,
                scoring_mode=scoring_mode,
                backend=self.esmfold2.model.backend,
            )
        candidate.esmfold2_predictions = result.predictions
        candidate.esmfold2_selected_index = max(
            range(len(result.predictions)), key=lambda index: result.predictions[index].scores["esmfold2_score"].value
        )
        candidate.esmfold2_score_metric = (
            "ptm"
            if scoring_mode == "redesign" and len(_polymer_representatives(result.predictions[0])) == 1
            else "ipsae"
        )
        return candidate

    def _design_pipeline(self, checkpoint: BoltzGenDesignCheckpoint) -> BoltzGenBackbonePipeline:
        if checkpoint == "diverse":
            return self.diverse_generator
        if checkpoint == "adherence":
            return self.adherence_generator
        fail("value_invalid", RequestRef("checkpoint", str(checkpoint)))

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
            designed_positions=(
                candidate.designed_positions
                if protocol == "protein-redesign"
                else candidate.connected_designed_positions
            ),
        )
        whole_result = self.folder.fold(
            candidate.structure,
            designed_positions=candidate.designed_positions,
            self_template=target,
            num_samples=5,
            num_recycles=4,
            num_sampling_steps=200,
            generator=generator,
        )
        candidate.whole_fold_predictions = whole_result.predictions
        candidate.whole_fold_selected_index = _select_fold_prediction(candidate.whole_fold_predictions)

        if design_folding:
            design_structure, design_references = _design_only(candidate)
            design_result = self.folder.fold(
                design_structure,
                designed_positions=design_references,
                num_samples=5,
                num_recycles=4,
                num_sampling_steps=200,
                generator=generator,
            )
            candidate.design_fold_predictions = design_result.predictions
            candidate.design_fold_selected_index = _select_fold_prediction(candidate.design_fold_predictions)
        return candidate

    def _analyze_candidate(
        self,
        candidate: BoltzGenDesignCandidate,
        *,
        protocol: BoltzGenProtocol,
        from_inverse_folding: bool,
        library: ComponentLibrary | None,
        largest_hydrophobic_patch: bool,
        liability_modality: Literal["antibody", "peptide"],
        liability_peptide_type: Literal["linear", "cyclic"],
    ) -> BoltzGenCandidateAnalysis:
        r"""
        Analyze exactly the selected whole refold used by every derived metric.
        """

        selected = candidate.whole_refold
        if selected is None:
            fail("value_invalid", RequestRef("candidate", "without whole-complex folding"))
        system = candidate.structure.system
        target = _target_references(
            system,
            designed_positions=(
                candidate.designed_positions
                if protocol == "protein-redesign"
                else candidate.connected_designed_positions
            ),
        )
        all_atom_rmsd = structure_rmsd_metrics(
            candidate.structure,
            selected.structure,
            designed_positions=candidate.designed_positions,
            target=target,
        )
        backbone_rmsd = structure_rmsd_metrics(
            candidate.structure,
            selected.structure,
            designed_positions=candidate.designed_positions,
            target=target,
            backbone_only=True,
        )

        design_folding_rmsd: float | None = None
        design_folding_bb_rmsd: float | None = None
        design_refold = candidate.design_refold
        if design_refold is not None:
            design_structure, design_references = _design_only(candidate)
            design_folding_rmsd = structure_rmsd_metrics(
                design_structure, design_refold.structure, designed_positions=design_references, target=()
            )["rmsd"]
            design_folding_bb_rmsd = structure_rmsd_metrics(
                design_structure,
                design_refold.structure,
                designed_positions=design_references,
                target=(),
                backbone_only=True,
            )["rmsd"]

        designed_chains = _designed_chain_analyses(
            system,
            candidate.designed_positions,
            liability_modality=liability_modality,
            liability_peptide_type=liability_peptide_type,
        )
        designed_sequence = ":".join(chain.designed_sequence for chain in designed_chains)
        designed_residues = "".join(chain.designed_sequence for chain in designed_chains)
        secondary_structure = secondary_structure_fractions(
            assign_secondary_structure(candidate.structure, components=candidate.designed_positions)
        )
        original_atomic = (
            None
            if from_inverse_folding
            else _atomic_analysis(
                candidate.structure,
                candidate.designed_positions,
                candidate.connected_designed_positions,
                target,
                library,
            )
        )
        refolded_atomic = _atomic_analysis(
            selected.structure, candidate.designed_positions, candidate.connected_designed_positions, target, library
        )
        hydrophobic_patch, hydrophobic_patch_error = (
            _present_metric(
                present_atom_largest_hydrophobic_patch_area,
                selected.structure,
                components=candidate.connected_designed_positions,
            )
            if largest_hydrophobic_patch
            else (None, None)
        )
        strict_patch, strict_patch_error = (
            _strict_metric(
                largest_hydrophobic_patch_area, selected.structure, components=candidate.connected_designed_positions
            )
            if largest_hydrophobic_patch
            else (None, None)
        )
        if strict_patch is not None and not np.isfinite(strict_patch):
            strict_patch, strict_patch_error = None, "no finite complete-chemistry hydrophobic patch"
        _, _, rmsd_atom_sites = structure_coordinates(candidate.structure, selected.structure)
        binding_site_adherence = _binding_site_adherence(
            candidate.structure, candidate.binding_sites, candidate.designed_positions
        )
        return BoltzGenCandidateAnalysis(
            designed_sequence=designed_sequence,
            designed_chains=designed_chains,
            num_design=len(candidate.designed_positions),
            rmsd=all_atom_rmsd["rmsd"],
            design_rmsd=all_atom_rmsd["design_rmsd"],
            target_rmsd=all_atom_rmsd["target_rmsd"],
            complex_aligned_design_rmsd=all_atom_rmsd["complex_aligned_design_rmsd"],
            target_aligned_design_rmsd=all_atom_rmsd["target_aligned_design_rmsd"],
            bb_rmsd=backbone_rmsd["rmsd"],
            bb_design_rmsd=backbone_rmsd["design_rmsd"],
            bb_target_rmsd=backbone_rmsd["target_rmsd"],
            bb_complex_aligned_design_rmsd=backbone_rmsd["complex_aligned_design_rmsd"],
            bb_target_aligned_design_rmsd=backbone_rmsd["target_aligned_design_rmsd"],
            design_folding_rmsd=design_folding_rmsd,
            design_folding_bb_rmsd=design_folding_bb_rmsd,
            binding_site_under_3angstrom=binding_site_adherence.get(3.0),
            binding_site_under_4angstrom=binding_site_adherence.get(4.0),
            binding_site_under_5angstrom=binding_site_adherence.get(5.0),
            binding_site_under_6angstrom=binding_site_adherence.get(6.0),
            binding_site_under_7angstrom=binding_site_adherence.get(7.0),
            binding_site_under_8angstrom=binding_site_adherence.get(8.0),
            binding_site_under_9angstrom=binding_site_adherence.get(9.0),
            amino_acid_fractions=amino_acid_fractions(designed_residues),
            secondary_structure_fractions=secondary_structure,
            design_hydrophobicity=(
                sum(
                    len(chain.designed_sequence) * sequence_hydrophobicity(chain.designed_sequence)
                    for chain in designed_chains
                )
                / sum(len(chain.designed_sequence) for chain in designed_chains)
            ),
            design_chain_hydrophobicity=(
                sum(len(chain.full_sequence) * chain.hydrophobicity for chain in designed_chains)
                / sum(len(chain.full_sequence) for chain in designed_chains)
            ),
            liabilities=_chain_liabilities(designed_chains),
            rmsd_atom_sites=rmsd_atom_sites,
            original_atomic=original_atomic,
            refolded_atomic=refolded_atomic,
            hydrophobic_patch=hydrophobic_patch,
            hydrophobic_patch_unavailable=hydrophobic_patch_error,
            strict_largest_hydrophobic_patch=strict_patch,
            strict_largest_hydrophobic_patch_unavailable=strict_patch_error,
        )


# ===========================================================================
# Decoding, generation and sequence production.
# ===========================================================================


def _designed_references(prepared: PreparedInput[BoltzGenInput]) -> tuple[ComponentRef, ...]:
    r"""The designed protein residue occurrences of one prepared plan."""
    layout = prepared.context.slots.layout
    designed = cast(torch.Tensor, prepared.features.design_mask)[0].cpu().numpy().astype(bool)
    protein = prepared.features.molecule_type[0].cpu().numpy() == BOLTZGEN_LAYOUT.monomer_types[MonomerType.PROTEIN]
    return prepared.context.structure.system.component_refs(np.unique(layout.token_component[designed & protein]))


def _connected_references(prepared: PreparedInput[BoltzGenInput]) -> tuple[ComponentRef, ...]:
    r"""The occurrences of every instance the design reaches (the model's own ``chain_design_mask``)."""
    layout = prepared.context.slots.layout
    chains = cast(torch.Tensor, prepared.features.chain_design_mask)[0].cpu().numpy().astype(bool)
    return prepared.context.structure.system.component_refs(np.unique(layout.token_component[chains]))


def _binding_references(
    system: MolecularSystem, conditions: Sequence[BoltzGenSchema]
) -> tuple[ComponentRef | AtomSite, ...]:
    r"""The positive binding-site occurrences the conditions name: an atom site of a class-less component stays an
    atom, every other endpoint is its component."""
    monomer = CLASS_MONOMER[token_classes(system).token]
    references: list[ComponentRef | AtomSite] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenBinding):
            continue
        for endpoint in condition.binding.residues:
            if not isinstance(endpoint, AtomSite):
                references.append(cast(ComponentRef, endpoint))
                continue
            component = int(system.component_index([ComponentRef(endpoint.instance, endpoint.component)])[0])
            polymer = component >= 0 and monomer[component] != MonomerType.NONE
            references.append(ComponentRef(endpoint.instance, endpoint.component) if polymer else endpoint)
    return tuple(dict.fromkeys(references))


def _protein_sequence_symmetry(
    system: MolecularSystem, conditions: Sequence[BoltzGenSchema]
) -> tuple[tuple[ComponentRef, ...], ...]:
    r"""Transpose structure-copy units into protein-only sequence-identity groups."""
    monomer = CLASS_MONOMER[token_classes(system).token]
    groups: list[tuple[ComponentRef, ...]] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenSymmetry):
            continue
        for group in zip(*condition.units, strict=True):
            components = system.component_index(list(group))
            protein = tuple(
                reference
                for reference, component in zip(group, components.tolist())
                if component >= 0 and monomer[component] == MonomerType.PROTEIN
            )
            if len(protein) > 1:
                groups.append(protein)
    return tuple(groups)


def _decode_backbone_candidates(
    prepared: PreparedInput[BoltzGenInput],
    output: BoltzGenSamplingOutput,
    conditions: Sequence[BoltzGenSchema],
    *,
    source: BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None,
    library: ComponentLibrary | None,
) -> tuple[BoltzGenDesignCandidate, ...]:
    r"""Decode every sample's atom14 identities and physical coordinates in the source frame."""

    system = prepared.context.structure.system
    designed = _designed_references(prepared)
    connected = _connected_references(prepared)
    binding_sites = _binding_references(system, conditions)
    symmetry = _protein_sequence_symmetry(system, conditions)
    atom = prepared.context.slots.atom
    inside = (atom >= 0) & prepared.context.scope.atoms[atom]
    slots = prepared.context.slots
    components = system.component_index(list(designed))
    token_design = cast(torch.Tensor, prepared.features.design_mask)[0].detach().cpu().numpy().astype(bool)
    token_protein = (
        prepared.features.molecule_type[0].detach().cpu().numpy() == BOLTZGEN_LAYOUT.monomer_types[MonomerType.PROTEIN]
    )
    rows = []
    for component in components:
        token = np.flatnonzero((slots.layout.token_component == component) & token_design & token_protein)
        if len(token) != 1:
            fail("value_invalid", RequestRef("atom14", "designed protein must have one token"))
        selected = np.flatnonzero(slots.token == token[0])
        if len(selected) != 14:
            fail("value_invalid", RequestRef("atom14", str(len(selected))), expected=14)
        rows.append(selected)
    row_index = np.stack(rows) if rows else np.empty((0, 14), dtype=np.int64)
    # A planned UNK can contain a CB atom, but its fifth raw slot is not a
    # physical atom until a canonical identity is recognized.
    inside[row_index[:, 4:].reshape(-1)] = False
    samples = output.coordinates[0].detach()
    candidates = []
    for index, sample in enumerate(samples):
        raw = sample[torch.as_tensor(row_index, device=sample.device)].clone()
        identities, recognized = decode_boltzgen_atom14(raw)
        intermediate, _ = decode_coordinates(
            prepared, sample.double().cpu().numpy(), valid=inside, evidence=CoordinateEvidence.GENERATED
        )
        valid = [i for i, name in enumerate(identities) if recognized[i] and name != "UNK"]
        target = intermediate
        if identities:
            component_names = system.text("component", system.component_symbol)
            atom_names = system.text("atom", system.atom_symbol)
            for component, name in zip(components.tolist(), identities):
                if component_names[component] == name:
                    continue
                atoms = np.arange(system.component_atom_ptr[component], system.component_atom_ptr[component + 1])
                protected = atoms[~prepared.context.scope.atoms[atoms]]
                if (
                    prepared.context.scope.rule not in {"all", "components"}
                    or not prepared.context.scope.atoms[atoms].any()
                    or not np.isin(atom_names[protected], ("N", "CA", "C", "O")).all()
                ):
                    fail("scope_violation", RequestRef("atom14", "identity change would replace protected atoms"))
            context = prepared.context
            translated = PreparedInput(
                prepared.features,
                PreparationContext(intermediate, context.slots, context.scope, context.record, context.spec),
            )
            target, _ = produce_design(
                translated,
                np.array(
                    [TOKEN_INDEX[CANONICAL_LETTER[name]] if name != "UNK" else TOKEN_INDEX["X"] for name in identities],
                    dtype=np.int64,
                ),
                positions=components,
                table=BOLTZGEN_LAYOUT,
                library=configured_library() if library is None else library,
                policy=DesignPolicy(allow_unknown=True),
            )
        if valid:
            sites: list[AtomSite] = []
            coordinates: list[NDArray[np.float64]] = []
            for i in valid:
                names = boltzgen_atom14_names(identities[i])
                proposed = [AtomSite(designed[i].instance, designed[i].component, name) for name in names]
                existing = system.atom_index(proposed)
                component = int(components[i])
                atoms = np.arange(system.component_atom_ptr[component], system.component_atom_ptr[component + 1])
                may_add = prepared.context.scope.rule in {"all", "components"} and bool(
                    prepared.context.scope.atoms[atoms].any()
                )
                for site, old_atom, position in zip(proposed, existing, raw[i, : len(names)], strict=True):
                    if (old_atom >= 0 and prepared.context.scope.atoms[old_atom]) or (old_atom < 0 and may_add):
                        sites.append(site)
                        coordinates.append(position.double().cpu().numpy())
            if sites:
                indexes = target.system.atom_index(sites)
                if (indexes < 0).any():
                    fail("value_invalid", RequestRef("atom14", "decoded atom absent from component"))
                scope = np.zeros(len(target.system.atom_symbol), dtype=bool)
                scope[indexes] = True
                target, _ = replace_coordinates(
                    target,
                    indexes,
                    np.asarray(coordinates, dtype=np.float64),
                    scope=scope,
                    evidence=CoordinateEvidence.GENERATED,
                    frame="source",
                )
        candidates.append(
            BoltzGenDesignCandidate(
                structure=target,
                codesign_structure=target,
                atom14_coordinates=raw.cpu(),
                atom14_residue_types=identities,
                atom14_recognized=recognized,
                source_checkpoint=source,
                designed_positions=designed,
                connected_designed_positions=connected,
                binding_sites=binding_sites,
                sequence_symmetry=symmetry,
                coordinate_trajectory=(
                    output.coordinate_trajectory[0, index] if output.coordinate_trajectory is not None else None
                ),
                denoised_coordinate_trajectory=(
                    output.denoised_coordinate_trajectory[0, index]
                    if output.denoised_coordinate_trajectory is not None
                    else None
                ),
            )
        )
    return tuple(candidates)


def _generate_backbones(
    input: BoltzGenBackbonePipelineInput,
    *,
    sources: Sequence[
        tuple[
            BoltzGenDesignCheckpoint | Literal["structuretrained_small"] | None,
            BoltzGenBackbonePipeline | BoltzGenMiniformerPipeline,
            int,
        ]
    ],
    num_total_candidates: int,
    candidate_offset: int = 0,
    diffusion_batch_size: int,
    num_sampling_steps: int | None,
    num_recycles: int | None,
    step_scale: float | None,
    noise_scale: float | None,
    step_scale_schedule: torch.Tensor | None,
    noise_scale_schedule: torch.Tensor | None,
    return_trajectory: bool,
    generator: torch.Generator | None,
    rng: np.random.Generator | None,
) -> tuple[BoltzGenDesignCandidate, ...]:
    r"""Generate checkpoint-contiguous batches, preparing the plan once per batch."""

    if num_total_candidates <= 0:
        fail("value_invalid", RequestRef("num_total_candidates", str(num_total_candidates)))
    num_requested_candidates = sum(source_count for _, _, source_count in sources)
    if candidate_offset < 0 or candidate_offset + num_requested_candidates > num_total_candidates:
        fail("value_invalid", RequestRef("candidate_offset", str(candidate_offset)), total=num_total_candidates)
    generated: list[BoltzGenDesignCandidate] = []
    global_start = candidate_offset
    for source, pipeline, source_count in sources:
        remaining = source_count
        while remaining:
            current_batch_size = min(diffusion_batch_size, remaining)
            prepared = pipeline.processor.prepare_input(
                input.structure,
                conditions=input.conditions,
                msa=input.msa,
                templates=input.templates,
                generator=rng,
            )
            step_scale_value, noise_scale_value = _sampler_scales(
                global_start,
                current_batch_size,
                num_total_candidates,
                device=prepared.features.atom_mask.device,
                dtype=pipeline.model.single_init_projection.weight.dtype,
                step_scale=step_scale,
                noise_scale=noise_scale,
            )
            sampler = pipeline.model.coordinate_diffusion
            sample = pipeline.model.sample(
                prepared.features,
                num_samples=current_batch_size,
                num_sampling_steps=num_sampling_steps,
                num_recycles=num_recycles,
                generator=generator,
                step_scale=(
                    None
                    if step_scale is None
                    and (
                        sampler.step_scale_function == "beta"
                        or (sampler.training and sampler.step_scale_random is not None)
                    )
                    else step_scale_value
                ),
                noise_scale=(
                    None if noise_scale is None and sampler.noise_scale_function == "beta" else noise_scale_value
                ),
                step_scale_schedule=step_scale_schedule,
                noise_scale_schedule=noise_scale_schedule,
                return_trajectory=return_trajectory,
            )
            generated.extend(
                _decode_backbone_candidates(prepared, sample, input.conditions, source=source, library=input.library)
            )
            global_start += current_batch_size
            remaining -= current_batch_size
    return tuple(generated)


def _produce_sequence(
    backbone: BoltzGenDesignCandidate,
    prepared: PreparedInput[BoltzGenInverseFoldingInput],
    output: BoltzGenInverseFoldingDesignOutput,
    *,
    library: ComponentLibrary,
) -> BoltzGenDesignCandidate:
    r"""One sampled sequence as a candidate: the designed positions' sampled classes through ``produce_design``."""
    layout = prepared.context.slots.layout
    designed = cast(torch.Tensor, prepared.features.design_mask)[0].cpu().numpy().astype(bool)
    classes = output.sequence_tokens[0].detach().cpu().numpy()[designed]
    target, _ = produce_design(
        prepared,
        classes,
        positions=layout.token_component[designed],
        table=BOLTZGEN_LAYOUT,
        library=library,
    )
    return backbone.copied(structure=target)


def _inverse_folding_candidate(
    input: BoltzGenDesignCandidate | Structure,
    *,
    design: PositionSelection | None,
    symmetry: Sequence[Sequence[ComponentRef]] | None,
) -> BoltzGenDesignCandidate:
    r"""
    The candidate one inverse-folding request designs: a backbone candidate with its scope, or a bare structure.

    A backbone candidate keeps its own design scope unless ``design`` replaces it; a bare structure with no selection
    designs every protein residue (``PositionSelection()``).
    """

    if not isinstance(input, BoltzGenDesignCandidate):
        candidate = BoltzGenDesignCandidate(
            structure=input, source_checkpoint=None, designed_positions=(), connected_designed_positions=()
        )
        design = PositionSelection() if design is None else design
    else:
        candidate = input
        if design is None and symmetry is None:
            return candidate
    structure = candidate.structure
    designed = (
        candidate.designed_positions
        if design is None
        else structure.system.component_refs(
            resolve_positions(structure, design, allowed=_protein_components(structure.system))
        )
    )
    return candidate.copied(
        designed_positions=designed,
        connected_designed_positions=(
            candidate.connected_designed_positions
            if design is None
            else _connected_positions(structure.system, designed)
        ),
        sequence_symmetry=(
            candidate.sequence_symmetry if symmetry is None else tuple(tuple(group) for group in symmetry)
        ),
    )


# ===========================================================================
# Released schedules, filters and selection.
# ===========================================================================


def _diffusion_batch_size(num_candidates: int, value: int | None) -> int:
    if num_candidates <= 0:
        fail("value_invalid", RequestRef("num_candidates", str(num_candidates)))
    batch_size = 1 if value is None and num_candidates < 100 else (10 if value is None else value)
    if batch_size <= 0:
        fail("value_invalid", RequestRef("diffusion_batch_size", str(value)))
    return batch_size


def _sampler_scales(
    global_start: int,
    batch_size: int,
    num_total_candidates: int,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    step_scale: float | None,
    noise_scale: float | None,
) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    r"""
    Select the released quarter schedule by global candidate order.
    """

    global_index = torch.arange(global_start, global_start + batch_size, device=device)
    quarter = global_index * 4 // num_total_candidates
    step_scale_value: float | torch.Tensor
    noise_scale_value: float | torch.Tensor
    if step_scale is None:
        step_scale_value = torch.tensor((1.8, 2.0, 1.8, 2.0), device=device, dtype=dtype)[quarter]
    else:
        step_scale_value = step_scale
    if noise_scale is None:
        noise_scale_value = torch.tensor((0.95, 0.88, 0.95, 0.88), device=device, dtype=dtype)[quarter]
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
        CandidateFilter("filter_design_rmsd", rmsd_threshold, True),
    ]
    if design_folding:
        filters.append(CandidateFilter("design_folding_filter_rmsd", rmsd_threshold, True))
    if filter_binding_site:
        filters.append(CandidateFilter("binding_site_under_8angstrom", 0.0001, False))
    if filter_target_alignment:
        filters.append(CandidateFilter("bb_target_aligned_design_rmsd", 2.5, True))
    if filter_cysteine:
        filters.append(CandidateFilter("CYS_fraction", 0.0, True))
    if filter_biased:
        filters.extend(
            CandidateFilter(name, 0.3, True)
            for name in ("ALA_fraction", "GLY_fraction", "GLU_fraction", "LEU_fraction", "VAL_fraction")
        )
    return tuple(filters)


def _strict_metric(operation, structure: Structure, **options):
    r"""Keep the reason when a complete-chemistry metric cannot be established."""
    try:
        return operation(structure, **options), None
    except AnalysisUnavailableError as error:
        return None, str(error)


def _present_metric(operation, structure: Structure, **options):
    r"""Keep known present-atom availability failures without suppressing unrelated errors."""
    try:
        return operation(structure, **options), None
    except PresentAtomUnavailableError as error:
        return None, str(error)


def _atomic_analysis(
    structure: Structure,
    designed_positions: tuple[ComponentRef, ...],
    connected_designed_positions: tuple[ComponentRef, ...],
    target: tuple[ComponentRef, ...],
    library: ComponentLibrary | None,
) -> BoltzGenAtomicAnalysis:
    r"""Evaluate screening on present atoms and complete chemistry independently."""
    contact_options = {
        "designed_positions": designed_positions,
        "connected_designed_positions": connected_designed_positions,
    }
    sasa_options = {"target": target, "designed_positions": designed_positions}
    if not target:
        reason = "no target components"
        return BoltzGenAtomicAnalysis(None, reason, None, reason, None, reason, None, reason)
    system = structure.system
    component = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
    target_components = system.component_index(list(target))
    if np.any(target_components < 0):
        reason = "target components are absent from the structure"
        return BoltzGenAtomicAnalysis(None, reason, None, reason, None, reason, None, reason)
    target_atoms = np.isin(component, target_components) & (system.atomic_number != 1)
    if not np.any(target_atoms & structure.coordinate_mask):
        reason = "no present target heavy atoms"
        return BoltzGenAtomicAnalysis(None, reason, None, reason, None, reason, None, reason)
    design_components = system.component_index(list(designed_positions))
    if np.any(design_components < 0):
        reason = "designed components are absent from the structure"
        return BoltzGenAtomicAnalysis(None, reason, None, reason, None, reason, None, reason)
    design_atoms = np.isin(component, design_components) & (system.atomic_number != 1)
    if not np.any(design_atoms & structure.coordinate_mask):
        reason = "no present design heavy atoms"
        return BoltzGenAtomicAnalysis(None, reason, None, reason, None, reason, None, reason)
    contacts, present_contacts_error = _present_metric(
        present_atom_noncovalent_contacts, structure, library=library, **contact_options
    )
    sasa, present_sasa_error = _present_metric(present_atom_target_delta_sasa, structure, **sasa_options)
    connected_components = system.component_index(list(connected_designed_positions))
    contact_target_atoms = ~np.isin(component, connected_components) & (system.atomic_number != 1)
    if np.any(connected_components < 0):
        strict_contacts, contacts_error = None, "connected design components are absent from the structure"
    elif not np.any(contact_target_atoms & structure.coordinate_mask):
        strict_contacts, contacts_error = None, "no present contact target heavy atoms"
    else:
        strict_contacts, contacts_error = _strict_metric(count_noncovalent_contacts, structure, **contact_options)
    strict_sasa, sasa_error = _strict_metric(target_delta_sasa, structure, **sasa_options)
    if (
        strict_sasa is not None
        and not np.isfinite((strict_sasa.delta, strict_sasa.target_unbound, strict_sasa.target_bound)).all()
    ):
        strict_sasa, sasa_error = None, "no finite complete-chemistry target SASA"
    return BoltzGenAtomicAnalysis(
        contacts,
        present_contacts_error,
        sasa,
        present_sasa_error,
        strict_contacts,
        contacts_error,
        strict_sasa,
        sasa_error,
    )


def _require_candidate_values(
    values: Mapping[str, float], names: Sequence[str], purpose: str, candidate: BoltzGenDesignCandidate | None = None
) -> None:
    r"""Refuse an unavailable or nonfinite metric rather than ranking or filtering by a substitute."""
    invalid = [name for name in names if name not in values or not np.isfinite(values[name])]
    if invalid:
        details: dict[str, str] = {}
        if candidate is not None and candidate.analysis is not None:
            for phase, atomic in (
                ("original", candidate.analysis.original_atomic),
                ("refolded", candidate.analysis.refolded_atomic),
            ):
                for name in invalid:
                    if phase not in name:
                        continue
                    if atomic is None:
                        details[name] = "original atomic analysis is not computed after inverse folding"
                    elif "hydrophobic_patch" in name and not name.startswith("strict_"):
                        if candidate.analysis.hydrophobic_patch_unavailable is not None:
                            details[name] = candidate.analysis.hydrophobic_patch_unavailable
                    elif "sasa" in name and atomic.sasa is None and not name.startswith("strict_"):
                        if atomic.sasa_unavailable is not None:
                            details[name] = atomic.sasa_unavailable
                    elif ("hydrogen_bonds" in name or "charged_pairs" in name) and not name.startswith("strict_"):
                        if atomic.contacts_unavailable is not None:
                            details[name] = atomic.contacts_unavailable
                    elif name.startswith("strict_"):
                        reason = (
                            candidate.analysis.strict_largest_hydrophobic_patch_unavailable
                            if "hydrophobic_patch" in name
                            else (
                                atomic.strict_sasa_unavailable if "sasa" in name else atomic.strict_contacts_unavailable
                            )
                        )
                        if reason is not None:
                            details[name] = reason
        fail("value_invalid", RequestRef(purpose, ", ".join(invalid)), reason="metric_unavailable", details=details)


def _ranking_metric_weights(protocol: BoltzGenProtocol, *, from_inverse_folding: bool) -> dict[str, float]:
    if protocol == "protein-redesign":
        return {"design_residue_iptm": 1.0, "iptm": 2.0, "ptm": 3.0, "neg_filter_design_rmsd": 4.0}
    values = {
        "design_to_target_iptm": 1.0,
        "design_ptm": 1.0,
        "neg_min_design_to_target_pae": 1.0,
        f"num_{'refolded' if from_inverse_folding else 'original'}_hydrogen_bonds": 2.0,
        f"num_{'refolded' if from_inverse_folding else 'original'}_charged_pairs": 2.0,
        f"target_delta_sasa_{'refolded' if from_inverse_folding else 'original'}": 2.0,
    }
    if protocol == "protein-small_molecule":
        values["design_to_target_iptm"] = 1.1
        values["design_ptm"] = 1.1
        values["neg_min_design_to_target_pae"] = 1.1
        values["affinity_probability_binary1"] = 1.0
    else:
        del values["design_to_target_iptm"]
        del values["neg_min_design_to_target_pae"]
        values["esmfold2_score"] = 1.0
    return values


def _candidate_values(candidate: BoltzGenDesignCandidate) -> dict[str, float]:
    analysis = candidate.analysis
    selected = candidate.whole_refold
    if analysis is None or selected is None:
        fail("value_invalid", RequestRef("candidate", "without folding and analysis"))
    values = {name: score.value for name, score in selected.scores.items()}
    if candidate.affinity_result is not None:
        values.update({name: score.value for name, score in candidate.affinity_result.scores.items()})
    if candidate.esmfold2_refold is not None:
        values["esmfold2_score"] = candidate.esmfold2_refold.scores["esmfold2_score"].value
        values["esmfold2_ptm"] = candidate.esmfold2_refold.scores["ptm"].value
        values.update(
            {
                f"esmfold2_{name}": candidate.esmfold2_refold.scores[name].value
                for name in ("design_to_target_ipsae", "target_to_design_ipsae", "design_ipsae_min")
                if name in candidate.esmfold2_refold.scores
            }
        )
    values.update(analysis.amino_acid_fractions)
    values.update(analysis.secondary_structure_fractions)
    values.update(
        {name: float(value) for name, value in analysis.liabilities.items() if isinstance(value, int | float | bool)}
    )
    values.update(
        {
            "has_x": float("X" in analysis.designed_sequence),
            "num_design": float(analysis.num_design),
            "filter_rmsd": analysis.bb_rmsd,
            "filter_design_rmsd": analysis.bb_design_rmsd,
            "bb_target_aligned_design_rmsd": analysis.bb_target_aligned_design_rmsd,
            "design_hydrophobicity": analysis.design_hydrophobicity,
            "design_chain_hydrophobicity": analysis.design_chain_hydrophobicity,
            "neg_design_hydrophobicity": -analysis.design_hydrophobicity,
            "neg_filter_rmsd": -analysis.bb_rmsd,
            "neg_filter_design_rmsd": -analysis.bb_design_rmsd,
        }
    )
    optional: dict[str, float | int | None] = {}
    for phase, atomic in (("original", analysis.original_atomic), ("refolded", analysis.refolded_atomic)):
        if atomic is not None:
            if atomic.contacts is not None:
                optional[f"num_{phase}_hydrogen_bonds"] = atomic.contacts.contacts.num_hydrogen_bonds
                optional[f"num_{phase}_charged_pairs"] = atomic.contacts.contacts.num_charged_pairs
            if atomic.sasa is not None:
                optional[f"target_delta_sasa_{phase}"] = atomic.sasa.sasa.delta
                optional[f"target_sasa_unbound_{phase}"] = atomic.sasa.sasa.target_unbound
                optional[f"target_sasa_bound_{phase}"] = atomic.sasa.sasa.target_bound
            if atomic.strict_contacts is not None:
                optional[f"strict_num_{phase}_hydrogen_bonds"] = atomic.strict_contacts.num_hydrogen_bonds
                optional[f"strict_num_{phase}_charged_pairs"] = atomic.strict_contacts.num_charged_pairs
            if atomic.strict_sasa is not None:
                optional[f"strict_target_delta_sasa_{phase}"] = atomic.strict_sasa.delta
                optional[f"strict_target_sasa_unbound_{phase}"] = atomic.strict_sasa.target_unbound
                optional[f"strict_target_sasa_bound_{phase}"] = atomic.strict_sasa.target_bound
    values.update({name: float(value) for name, value in optional.items() if value is not None})
    if analysis.design_folding_bb_rmsd is not None:
        values["design_folding_filter_rmsd"] = analysis.design_folding_bb_rmsd
    values.update(
        {
            f"binding_site_under_{threshold}angstrom": value
            for threshold, value in (
                (3, analysis.binding_site_under_3angstrom),
                (4, analysis.binding_site_under_4angstrom),
                (5, analysis.binding_site_under_5angstrom),
                (6, analysis.binding_site_under_6angstrom),
                (7, analysis.binding_site_under_7angstrom),
                (8, analysis.binding_site_under_8angstrom),
                (9, analysis.binding_site_under_9angstrom),
            )
            if value is not None
        }
    )
    if analysis.hydrophobic_patch is not None:
        values["design_largest_hydrophobic_patch_refolded"] = analysis.hydrophobic_patch.area
        values["neg_design_largest_hydrophobic_patch_refolded"] = -analysis.hydrophobic_patch.area
    if analysis.strict_largest_hydrophobic_patch is not None:
        values["strict_design_largest_hydrophobic_patch_refolded"] = analysis.strict_largest_hydrophobic_patch
        values["strict_neg_design_largest_hydrophobic_patch_refolded"] = -analysis.strict_largest_hydrophobic_patch
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
    metrics = {
        candidate.esmfold2_score_metric for candidate in candidates if candidate.esmfold2_score_metric is not None
    }
    if len(metrics) > 1:
        fail("value_invalid", RequestRef("candidates", "mixed pTM and iPSAE scores"))
    unique_indices = deduplicate_sequences(sequences)
    unique_candidates = tuple(candidates[index] for index in unique_indices)
    rows = tuple(_candidate_values(candidate) for candidate in unique_candidates)
    for values in rows:
        _require_candidate_values(values, (*metric_weights, tie_breaker), "ranking")
    filters = tuple(outcome.filter for outcome in unique_candidates[0].filters) if unique_candidates else ()
    ranking = rank_candidates(rows, filters=filters, metric_weights=metric_weights, tie_breaker=tie_breaker)
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
        size_values=tuple(
            cast(BoltzGenCandidateAnalysis, candidates[index].analysis).num_design for index in ranked_indices
        ),
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


def _inverse_fold_model(protocol: BoltzGenProtocol, model: BoltzGenInverseFoldModel | None) -> BoltzGenInverseFoldModel:
    selected = ("boltzif" if protocol == "protein-small_molecule" else "solublempnn") if model is None else model
    if selected not in {"boltzif", "solublempnn"}:
        fail("value_invalid", RequestRef("inverse_fold_model", str(selected)))
    if protocol == "protein-small_molecule" and selected != "boltzif":
        fail("value_invalid", RequestRef("inverse_fold_model", str(selected)), protocol=protocol)
    return selected


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


# ===========================================================================
# Candidate scopes, sequences and coordinates.
# ===========================================================================


def _protein_components(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""Per component, whether it is a protein residue."""
    return CLASS_MONOMER[token_classes(system).token] == MonomerType.PROTEIN


def _connected_positions(system: MolecularSystem, designed: Sequence[ComponentRef]) -> tuple[ComponentRef, ...]:
    r"""Every occurrence of the designed instances and of the instances a KNOWN covalent edge joins to them."""
    components = system.component_index(list(designed))
    selected = design_instances(system, components[components >= 0])
    pointer = system.instance_component_ptr
    owner = np.repeat(np.arange(len(system.instance_symbol)), np.diff(pointer))
    return system.component_refs(np.flatnonzero(selected[owner]))


def _target_references(
    system: MolecularSystem, *, designed_positions: Sequence[ComponentRef]
) -> tuple[ComponentRef, ...]:
    r"""Every occurrence the design does not cover."""
    design = frozenset(designed_positions)
    every = system.component_refs(np.arange(len(system.component_symbol)))
    return tuple(reference for reference in every if reference not in design)


def _designed_sequence(system: MolecularSystem, positions: Sequence[ComponentRef]) -> str:
    r"""The canonical one-letter class of each named protein occurrence."""
    classes = token_classes(system).token[system.component_index(list(positions))]
    return "".join(TOKENS[int(token)] for token in classes)


def _designed_chain_analyses(
    system: MolecularSystem,
    designed_positions: Sequence[ComponentRef],
    *,
    liability_modality: Literal["antibody", "peptide"],
    liability_peptide_type: Literal["linear", "cyclic"],
) -> tuple[BoltzGenDesignedChainAnalysis, ...]:
    protein = _protein_components(system)
    components = system.component_index(list(designed_positions))
    designed = [
        reference
        for reference, component in zip(designed_positions, components.tolist())
        if component >= 0 and protein[component]
    ]
    pointer = system.instance_component_ptr
    analyses = []
    for instance_id in dict.fromkeys(reference.instance for reference in designed):
        instance = int(system.instance_index([instance_id])[0])
        length = int(pointer[instance + 1] - pointer[instance])
        designed_sequence = _designed_sequence(
            system, [reference for reference in designed if reference.instance == instance_id]
        )
        full_sequence = _designed_sequence(system, [ComponentRef(instance_id, index) for index in range(length)])
        analyses.append(
            BoltzGenDesignedChainAnalysis(
                instance_id=instance_id,
                designed_sequence=designed_sequence,
                full_sequence=full_sequence,
                hydrophobicity=sequence_hydrophobicity(full_sequence),
                liabilities=liability_metrics(
                    full_sequence, modality=liability_modality, peptide_type=liability_peptide_type
                ),
            )
        )
    return tuple(analyses)


def _chain_liabilities(chains: Sequence[BoltzGenDesignedChainAnalysis]) -> dict[str, AnalysisValue]:
    if len(chains) == 1:
        return chains[0].liabilities
    totals: dict[str, AnalysisValue] = {
        key: sum(cast(int, chain.liabilities[key]) for chain in chains)
        for key in (
            "liability_score",
            "liability_num_violations",
            "liability_high_severity_violations",
            "liability_medium_severity_violations",
            "liability_low_severity_violations",
        )
    }
    for key in sorted(
        {metric_key for chain in chains for metric_key in chain.liabilities if metric_key.endswith("_count")}
    ):
        prefix = key.removesuffix("_count")
        count = sum(cast(int, chain.liabilities.get(key, 0)) for chain in chains)
        severity = max(
            max(
                cast(float, chain.liabilities.get(f"{prefix}_severity", 0)),
                cast(float, chain.liabilities.get(f"{prefix}_avg_severity", 0)),
            )
            for chain in chains
        )
        totals.update(
            {
                key: count,
                f"{prefix}_position": -1,
                f"{prefix}_num_residues": 0,
                f"{prefix}_severity": int(severity) if count else 0,
                f"{prefix}_num_positions": sum(
                    cast(int, chain.liabilities.get(f"{prefix}_num_positions", 0))
                    + int(cast(int, chain.liabilities.get(f"{prefix}_position", -1)) >= 0)
                    for chain in chains
                ),
                f"{prefix}_positions": "",
                f"{prefix}_details": "; ".join(
                    f"{chain.instance_id}: {chain.liabilities.get(key, 0)} "
                    f"{chain.liabilities.get(f'{prefix}_positions') or chain.liabilities.get(f'{prefix}_position', -1)}"
                    for chain in chains
                    if chain.liabilities.get(key, 0)
                ),
                f"{prefix}_global_details": "",
                f"{prefix}_avg_severity": float(severity) if count else 0.0,
            }
        )
    totals["liability_violations_summary"] = "; ".join(
        f"{chain.instance_id}: {chain.liabilities['liability_violations_summary']}" for chain in chains
    )
    return totals


def _design_ca_coordinates(structure: Structure, designed_positions: Sequence[ComponentRef]) -> NDArray[np.float64]:
    r"""The alpha carbon of every designed occurrence."""
    sites = [AtomSite(reference.instance, reference.component, "CA") for reference in designed_positions]
    atoms = structure.system.atom_index(sites)
    if len(atoms) and (atoms < 0).any():
        fail("request_invalid_positions", RequestRef("designed_positions", str(sites[int(np.argmin(atoms))])))
    return structure.coordinates[atoms]


def _component_centres(structure: Structure, references: Sequence[ComponentRef | AtomSite]) -> NDArray[np.float64]:
    r"""The observed coordinates the named sites contribute: an atom its own, a polymer residue its centre atom, a
    class-less component every observed atom of it."""
    system = structure.system
    monomer = CLASS_MONOMER[token_classes(system).token]
    names = system.text("atom", system.atom_symbol)
    pointer = system.component_atom_ptr
    observed = structure.coordinate_mask
    rows: list[int] = []
    for reference in references:
        if isinstance(reference, AtomSite):
            atom = int(system.atom_index([reference])[0])
            if atom >= 0 and observed[atom]:
                rows.append(atom)
            continue
        component = int(system.component_index([reference])[0])
        if component < 0:
            continue
        atoms = np.arange(pointer[component], pointer[component + 1])
        atoms = atoms[observed[atoms]]
        if not len(atoms):
            continue
        centre = BOLTZGEN_LAYOUT.centre_atom.get(MonomerType(int(monomer[component])))
        if centre is None:
            rows += atoms.tolist()
            continue
        named = atoms[names[atoms] == centre]
        rows.append(int(named[0]) if len(named) else int(atoms[0]))
    return structure.coordinates[np.array(tuple(dict.fromkeys(rows)), dtype=np.int64)]


def _binding_site_adherence(
    structure: Structure,
    binding_sites: Sequence[ComponentRef | AtomSite],
    designed_positions: Sequence[ComponentRef],
) -> dict[float, float]:
    if not binding_sites:
        return {}
    return binding_site_adherence_curve(
        _component_centres(structure, binding_sites), _component_centres(structure, designed_positions)
    )


def _design_only(candidate: BoltzGenDesignCandidate) -> tuple[Structure, tuple[ComponentRef, ...]]:
    r"""The design's own instances as one structure, and the designed occurrences inside it."""
    system = candidate.structure.system
    selected = {reference.instance for reference in candidate.connected_designed_positions}
    instances = np.isin(system.text("instance", system.instance_symbol), list(selected))
    structure, _ = project_instances(candidate.structure, instances)
    kept = frozenset(structure.system.text("instance", structure.system.instance_symbol).tolist())
    return structure, tuple(reference for reference in candidate.designed_positions if reference.instance in kept)


# ===========================================================================
# ESMFold2 interaction scores.
# ===========================================================================


def _polymer_representatives(prediction: FoldPrediction) -> dict[str, list[int]]:
    r"""Per polymer instance, the metric rows of its residues (one representative each)."""
    system = prediction.structure.system
    monomer = CLASS_MONOMER[token_classes(system).token]
    rows: dict[ComponentRef, list[int]] = {}
    for row, reference in enumerate(prediction.pair_metrics["pae"].row_axis):
        component = (
            ComponentRef(reference.instance, reference.component) if isinstance(reference, AtomSite) else reference
        )
        if not isinstance(component, ComponentRef):
            continue
        index = int(system.component_index([component])[0])
        kind = MonomerType(int(monomer[index])) if index >= 0 else MonomerType.NONE
        if kind is MonomerType.NONE:
            continue
        matches = rows.setdefault(component, [])
        if not isinstance(reference, AtomSite) or reference.atom == ("CA" if kind is MonomerType.PROTEIN else "C1'"):
            matches.append(row)
    representatives: dict[str, list[int]] = {}
    for component, matches in rows.items():
        if len(matches) != 1:
            fail("value_invalid", RequestRef("prediction", "without one representative per polymer residue"))
        representatives.setdefault(component.instance, []).append(matches[0])
    return representatives


def _esmfold2_candidate_scores(
    prediction: FoldPrediction,
    *,
    designed_instances: set[str],
    target_instances: Sequence[str] | None,
    scoring_mode: Literal["binder", "redesign"],
    backend: str = "auto",
) -> None:
    representatives = _polymer_representatives(prediction)
    system = prediction.structure.system
    monomer = CLASS_MONOMER[token_classes(system).token]
    pointer = system.instance_component_ptr
    kinds = {
        chain: MonomerType(int(monomer[int(pointer[int(system.instance_index([chain])[0])])]))
        for chain in representatives
    }
    nucleic = {MonomerType.RNA, MonomerType.DNA}
    if not designed_instances.intersection(representatives):
        fail("value_invalid", RequestRef("designed_instances", "without a polymer chain"))
    pae = torch.as_tensor(prediction.pair_metrics["pae"].values)
    valid = torch.ones(pae.shape[-1], dtype=torch.bool, device=pae.device)

    def directed(source: Sequence[int], target: Sequence[int], floor: float) -> float:
        source_mask = torch.zeros_like(valid)
        target_mask = torch.zeros_like(valid)
        source_mask[list(source)] = True
        target_mask[list(target)] = True
        return float(
            directional_ipsae(
                pae,
                source_mask=source_mask,
                target_mask=target_mask,
                frame_mask=valid,
                token_mask=valid,
                backend=backend,
                cutoff=10.0,
                d0_min=floor,
            )
        )

    chains = tuple(representatives)
    chain_pairs = np.zeros((len(chains), len(chains)), dtype=prediction.pair_metrics["pae"].values.dtype)
    for row, source in enumerate(chains):
        for column, target in enumerate(chains):
            if source == target:
                continue
            floor = 2.0 if kinds[source] in nucleic or kinds[target] in nucleic else 1.0
            chain_pairs[row, column] = directed(representatives[source], representatives[target], floor)
    prediction.pair_metrics["chain_pair_ipsae"] = PairMetric(chain_pairs, row_axis=chains, column_axis=chains)
    prediction.pair_metrics["chain_pair_ipsae_min"] = PairMetric(
        np.minimum(chain_pairs, chain_pairs.T), row_axis=chains, column_axis=chains
    )

    if scoring_mode == "redesign":
        if target_instances is not None:
            fail("value_invalid", RequestRef("target_instances", "with redesign scoring"))
        if len(chains) == 1:
            prediction.scores["esmfold2_score"] = prediction.scores["ptm"]
            return
        floor = 2.0 if any(kind in nucleic for kind in kinds.values()) else 1.0
        chain_scores = []
        for chain in chains:
            rest = [index for other in chains if other != chain for index in representatives[other]]
            chain_scores.append(
                min(directed(representatives[chain], rest, floor), directed(rest, representatives[chain], floor))
            )
        prediction.metrics["chain_vs_rest_ipsae"] = Metric(np.asarray(chain_scores), axis=chains)
        prediction.scores["esmfold2_score"] = Score(min(chain_scores), preferred="higher")
        return

    targets = (
        tuple(chain for chain in chains if chain not in designed_instances)
        if target_instances is None
        else tuple(target_instances)
    )
    if not targets or designed_instances.intersection(targets):
        fail("value_invalid", RequestRef("target_instances", "without a separate target chain"))
    if len({kinds[chain] is MonomerType.PROTEIN for chain in targets}) > 1:
        fail("value_invalid", RequestRef("target_instances", "mixing protein and nucleic-acid chains"))
    design_rows = [index for chain in chains if chain in designed_instances for index in representatives[chain]]
    target_rows = [index for chain in targets for index in representatives[chain]]
    participating = (*targets, *(chain for chain in chains if chain in designed_instances))
    floor = 2.0 if any(kinds[chain] in nucleic for chain in participating) else 1.0
    forward = directed(design_rows, target_rows, floor)
    reverse = directed(target_rows, design_rows, floor)
    prediction.scores.update(
        {
            "design_to_target_ipsae": Score(forward, preferred="higher"),
            "target_to_design_ipsae": Score(reverse, preferred="higher"),
            "design_ipsae_min": Score(min(forward, reverse), preferred="higher"),
            "esmfold2_score": Score(min(forward, reverse), preferred="higher"),
        }
    )
