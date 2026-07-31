r"""
In-memory scientific analysis for BoltzGen designs.

This module contains the geometry, sequence, interaction, filtering, ranking, and
diversity operations used by the released BoltzGen design protocols.  It does not
own protocol selection, model execution, filesystem staging, or report generation.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure.info import vdw_radius_protor, vdw_radius_single
from numpy.typing import NDArray

from ...data import AtomSite, ComponentReference, Structure
from ...data.molecular import (
    AMINO_ACID_ONE_TO_THREE,
    ENTITY_DNA,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
)

AnalysisValue: TypeAlias = float | int | bool | str
LiabilityModality: TypeAlias = Literal["antibody", "peptide"]
PeptideType: TypeAlias = Literal["linear", "cyclic"]


@dataclass(frozen=True)
class NoncovalentContacts:
    r"""
    Released BoltzGen design-to-target noncovalent contact counts.

    Args:
        hydrogen_bonds: Hydride/Biotite donor--acceptor hydrogen bonds.
        charged_pairs: Oppositely charged atom pairs at ``0.5 < d < 5.5`` Å.
    """

    hydrogen_bonds: int
    charged_pairs: int


@dataclass(frozen=True)
class TargetDeltaSasa:
    r"""
    Target burial caused by the designed component.

    Args:
        delta: Target SASA in isolation minus target SASA in the bound complex.
        target_unbound: Target SASA in isolation.
        target_bound: Target SASA in the bound target--design complex.
    """

    delta: float
    target_unbound: float
    target_bound: float


@dataclass(frozen=True)
class Liability:
    r"""
    One released sequence-liability occurrence.

    Args:
        motif: Liability name.
        position: One-based sequence position, or ``None`` for a global property.
        length: Number of residues in the matched motif.
        severity: Released integer severity.
        details: Optional human-readable value for a global property.
    """

    motif: str
    position: int | None
    length: int
    severity: int
    details: str = ""


@dataclass(frozen=True)
class FoldseekMatch:
    r"""
    One Foldseek reference match used by the released novelty axis.

    Args:
        query_index: Zero-based index into the query structure sequence.
        query_tm_score: TM score normalized by query length.
        target_tm_score: TM score normalized by reference length.
    """

    query_index: int
    query_tm_score: float
    target_tm_score: float


FoldseekNoveltyOperation: TypeAlias = Callable[[Sequence[Structure]], Sequence[FoldseekMatch]]
FoldseekClusteringOperation: TypeAlias = Callable[[Sequence[Structure], float], Sequence[str]]


@dataclass(frozen=True)
class CandidateFilter:
    r"""
    One explicit BoltzGen candidate filter.

    Args:
        feature: Named scalar candidate score.
        threshold: Inclusive threshold.
        lower_is_better: Whether values at or below the threshold pass.
    """

    feature: str
    threshold: float
    lower_is_better: bool


@dataclass(frozen=True)
class CandidateRanking:
    r"""
    Stable BoltzGen ranking values aligned to input candidates.

    Args:
        order: Zero-based input indices in best-first order.
        rank: One-based final rank for every input row.
        quality: Released rank-derived quality for every input row.
        filters_passed: Number of individual filters passed by every row.
        passes_all_filters: Whether every filter passed for every row.
    """

    order: tuple[int, ...]
    rank: NDArray[np.int64]
    quality: NDArray[np.float64]
    filters_passed: NDArray[np.int64]
    passes_all_filters: NDArray[np.bool_]


@dataclass(frozen=True)
class BoltzGenDiversityMetrics:
    r"""
    Candidate metrics that condition the released structural Vendi scores.

    All-atom and backbone RMSD/lDDT values are kept distinct.  Design iPTM and
    minimum interaction PAE are confidence values shared by both metric scopes.
    """

    rmsd: float
    rmsd_design: float
    target_aligned_rmsd_design: float
    lddt_intra_design: float | None
    bb_rmsd: float
    bb_rmsd_design: float
    bb_target_aligned_rmsd_design: float
    bb_lddt_intra_design: float | None
    design_iptm: float
    min_interaction_pae: float


@dataclass(frozen=True, eq=False)
class BoltzGenDiversitySample:
    r"""
    One target-labelled design used by the released diversity reductions.
    """

    target_id: str
    sequence: str
    original_coordinates: NDArray
    refolded_coordinates: NDArray
    metrics: BoltzGenDiversityMetrics


@dataclass(frozen=True)
class BoltzGenConditionedVendiScores:
    r"""
    Structural Vendi scores after the released candidate-quality conditions.
    """

    rmsd_below_2_5: float
    design_rmsd_below_2: float
    design_rmsd_below_4: float
    target_aligned_design_rmsd_below_2_5: float
    design_lddt_at_least_0_60: float | None
    design_lddt_at_least_0_65: float | None
    design_lddt_at_least_0_70: float | None
    design_lddt_at_least_0_75: float | None
    design_lddt_at_least_0_80: float | None
    design_lddt_at_least_0_85: float | None
    design_lddt_at_least_0_90: float | None
    design_iptm_at_least_0_80: float
    design_iptm_at_least_0_70: float
    design_iptm_at_least_0_60: float
    design_iptm_at_least_0_50: float
    min_interaction_pae_at_most_1_5: float
    min_interaction_pae_at_most_2: float
    min_interaction_pae_at_most_2_5: float
    min_interaction_pae_at_most_3: float
    min_interaction_pae_at_most_4: float
    min_interaction_pae_at_most_5: float


@dataclass(frozen=True)
class BoltzGenStructureVendiScores:
    r"""
    Released base and conditionally filtered structural Vendi scores.
    """

    vendi_tm_fixed: float
    vendi_tm_align: float
    all_atom: BoltzGenConditionedVendiScores | None = None
    backbone: BoltzGenConditionedVendiScores | None = None


@dataclass(frozen=True)
class BoltzGenDiversityAggregate:
    r"""
    Structure and optional sequence diversity for one candidate collection.
    """

    structure: BoltzGenStructureVendiScores
    vendi_seq_sim: float | None
    num_structures: int
    num_sequences: int


@dataclass(frozen=True)
class BoltzGenDiversityResult:
    r"""
    Released original, refolded, and per-target diversity aggregates.
    """

    original: BoltzGenDiversityAggregate
    refolded: BoltzGenDiversityAggregate
    per_target_original: dict[str, BoltzGenDiversityAggregate]
    per_target_refolded: dict[str, BoltzGenDiversityAggregate]


_PROTEIN_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
_DNA_BACKBONE_ATOMS = frozenset({"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"})
_RNA_BACKBONE_ATOMS = _DNA_BACKBONE_ATOMS | {"O2'"}
_HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"})

_HYDROPHOBICITY = {
    "W": {"Rc": 12.25, "Rc1": 11.1, "Rc2": 11.8, "Rn": 12.25, "Rn1": 12.1},
    "F": {"Rc": 10.90, "Rc1": 7.5, "Rc2": 9.5, "Rn": 10.90, "Rn1": 10.3},
    "L": {"Rc": 9.30, "Rc1": 5.55, "Rc2": 7.4, "Rn": 9.30, "Rn1": 9.3},
    "I": {"Rc": 8.00, "Rc1": 5.2, "Rc2": 6.6, "Rn": 8.00, "Rn1": 7.7},
    "M": {"Rc": 6.20, "Rc1": 4.4, "Rc2": 5.7, "Rn": 6.20, "Rn1": 6.0},
    "V": {"Rc": 5.00, "Rc1": 2.9, "Rc2": 3.4, "Rn": 5.00, "Rn1": 4.2},
    "Y": {"Rc": 4.85, "Rc1": 3.7, "Rc2": 4.5, "Rn": 4.85, "Rn1": 4.4},
    "C": {"Rc": 0.45, "Rc1": 0.9, "Rc2": 0.2, "Rn": 0.45, "Rn1": -0.5},
    "P": {"Rc": 2.10, "Rc1": 2.1, "Rc2": 2.1, "Rn": 2.10, "Rn1": 2.1},
    "A": {"Rc": 1.10, "Rc1": 0.35, "Rc2": 0.5, "Rn": 1.10, "Rn1": -0.1},
    "E": {"Rc": 0.95, "Rc1": 1.0, "Rc2": 0.0, "Rn": 0.95, "Rn1": -0.1},
    "T": {"Rc": 0.65, "Rc1": 0.8, "Rc2": 0.6, "Rn": 0.65, "Rn1": 0.0},
    "D": {"Rc": 0.15, "Rc1": 0.5, "Rc2": 0.4, "Rn": 0.15, "Rn1": -0.5},
    "Q": {"Rc": -0.40, "Rc1": -0.7, "Rc2": -0.2, "Rn": -0.40, "Rn1": -1.1},
    "S": {"Rc": -0.15, "Rc1": 0.8, "Rc2": -0.1, "Rn": -0.15, "Rn1": -1.2},
    "G": {"Rc": -0.35, "Rc1": 0.2, "Rc2": 0.15, "Rn": -0.35, "Rn1": -0.7},
    "R": {"Rc": -1.40, "Rc1": 0.5, "Rc2": -1.1, "Rn": -1.30, "Rn1": -1.1},
    "N": {"Rc": -0.85, "Rc1": 0.2, "Rc2": -0.2, "Rn": -0.85, "Rn1": -1.1},
    "H": {"Rc": -1.45, "Rc1": -0.1, "Rc2": -0.2, "Rn": -1.45, "Rn1": -1.7},
    "K": {"Rc": -2.05, "Rc1": -0.6, "Rc2": -1.5, "Rn": -1.90, "Rn1": -1.45},
}
_NEIGHBOR_PENALTY = {"W": 0.15, "F": 0.10, "L": 0.30, "I": 0.15, "V": 0.20, "Y": 0.05}
_LIABILITY_SEVERITY = {
    "UnpairedCys": 10,
    "DeAmdH": 10,
    "FragH": 10,
    "Isom": 10,
    "TrpOx": 10,
    "IntBind": 10,
    "DeAmdM": 5,
    "FragM": 5,
    "DeAmdL": 1,
    "AspBridge": 10,
    "AspCleave": 10,
    "ProtTryp": 10,
    "DPP4": 5,
    "LowHydrophilic": 7,
    "ConsecIdentical": 7,
    "LongHydrophobic": 7,
}
_DEFAULT_LIABILITY_SEVERITY = 5


def aligned_rmsd(
    reference: NDArray,
    predicted: NDArray,
    *,
    align_mask: NDArray[np.bool_] | None = None,
    measure_mask: NDArray[np.bool_] | None = None,
    weights: NDArray | None = None,
    measure_weights: NDArray | None = None,
) -> float:
    r"""
    Return RMSD after weighted rigid alignment of ``reference`` to ``predicted``.

    ``align_mask`` selects the atoms that determine the transform and
    ``measure_mask`` selects the atoms whose displacement is measured.  This is
    the released BoltzGen subset-RMSD operation used for design-, target-, and
    target-aligned RMSDs.
    """

    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    count = reference.shape[0]
    if align_mask is None:
        align_mask = np.ones(count, dtype=np.bool_)
    if measure_mask is None:
        measure_mask = align_mask
    if weights is None:
        weights = np.ones(count, dtype=np.float64)
    if measure_weights is None:
        measure_weights = weights

    align_weight = np.asarray(weights, dtype=np.float64) * np.asarray(align_mask, dtype=np.bool_)
    aligned = _aligned_coordinates(reference, predicted, align_weight)
    measured_weight = np.asarray(measure_weights, dtype=np.float64) * np.asarray(measure_mask, dtype=np.bool_)
    squared_distance = np.sum(np.square(predicted - aligned), axis=-1)
    return float(np.sqrt(np.sum(squared_distance * measured_weight) / np.sum(measured_weight)))


def lddt(
    reference: NDArray,
    predicted: NDArray,
    *,
    pair_mask: NDArray[np.bool_] | None = None,
    cutoff: float | NDArray = 15.0,
) -> float:
    r"""
    Return the released hard-threshold lDDT score.

    Pairs whose reference distance is below ``cutoff`` are scored at the 0.5,
    1, 2, and 4 Å thresholds.  The identity is always excluded.
    """

    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    reference_distance = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    predicted_distance = np.linalg.norm(predicted[:, None] - predicted[None, :], axis=-1)

    if pair_mask is None:
        pair_mask = np.ones(reference_distance.shape, dtype=np.bool_)
    pair_mask = np.asarray(pair_mask, dtype=np.bool_).copy()
    np.fill_diagonal(pair_mask, False)
    scored = pair_mask & (reference_distance < cutoff)
    distance_error = np.abs(reference_distance - predicted_distance)
    score = 0.25 * (
        (distance_error < 0.5).astype(np.float64)
        + (distance_error < 1.0).astype(np.float64)
        + (distance_error < 2.0).astype(np.float64)
        + (distance_error < 4.0).astype(np.float64)
    )
    return float((1e-10 + np.sum(score * scored)) / (1e-10 + np.sum(scored)))


def _component_instance_index(structure: Structure) -> torch.Tensor:
    system = structure.system
    return torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )


def _atom_component_index(structure: Structure) -> torch.Tensor:
    system = structure.system
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _canonical_atom_sites(structure: Structure) -> tuple[AtomSite, ...]:
    system = structure.system
    atom_component = _atom_component_index(structure).detach().cpu().tolist()
    component_instance = _component_instance_index(structure).detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    component_start = system.instance_component_ptr.detach().cpu().tolist()
    return tuple(
        AtomSite(
            system.instance_id(component_instance[component_index]),
            component_index - component_start[component_instance[component_index]],
            system.symbols[atom_symbols[atom_index]],
        )
        for atom_index, component_index in enumerate(atom_component)
    )


def structure_coordinates(
    reference: Structure,
    predicted: Structure,
    *,
    components: Collection[ComponentReference] | None = None,
    backbone_only: bool = False,
) -> tuple[NDArray, NDArray, tuple[AtomSite, ...]]:
    r"""
    Return observed coordinates for exact atom sites shared by two structures.

    The output order follows the reference canonical atom axis. ``components``
    selects exact component occurrences; ``backbone_only`` uses the released
    BoltzGen protein and nucleic-acid backbone atom definitions.
    """

    predicted_index = {
        site: index
        for index, site in enumerate(_canonical_atom_sites(predicted))
        if bool(predicted.coordinate_mask[index])
    }
    component_set = None if components is None else frozenset(components)
    reference_coordinates = []
    predicted_coordinates = []
    sites = []
    for index, site in enumerate(_canonical_atom_sites(reference)):
        component = (site.instance_id, site.component_index)
        if not bool(reference.coordinate_mask[index]) or site not in predicted_index:
            continue
        if component_set is not None and component not in component_set:
            continue
        if backbone_only and not _is_backbone_site(reference, site):
            continue
        reference_coordinates.append(reference.coordinates[index].detach().cpu().numpy())
        predicted_coordinates.append(predicted.coordinates[predicted_index[site]].detach().cpu().numpy())
        sites.append(site)
    return np.asarray(reference_coordinates), np.asarray(predicted_coordinates), tuple(sites)


def structure_rmsd_metrics(
    original: Structure,
    refolded: Structure,
    *,
    design: Collection[ComponentReference],
    target: Collection[ComponentReference],
    backbone_only: bool = False,
) -> dict[str, float]:
    r"""
    Return the released BoltzGen all/design/target subset RMSDs.

    Exact atom sites observed in both structures are matched first.  Whole,
    design, and target alignments use the released relative atom weights:
    protein ``1``, nucleic acid ``6``, and non-polymer ``11``.  RMSDs measured
    on the design after whole- or target-only alignment are unweighted, matching
    the released implementation.
    """

    reference, predicted, sites = structure_coordinates(original, refolded, backbone_only=backbone_only)
    design = frozenset(design)
    target = frozenset(target)
    design_mask = np.array([(site.instance_id, site.component_index) in design for site in sites], dtype=np.bool_)
    target_mask = np.array([(site.instance_id, site.component_index) in target for site in sites], dtype=np.bool_)
    all_mask = np.ones(len(sites), dtype=np.bool_)
    weights = np.array([_atom_rmsd_weight(original, site) for site in sites], dtype=np.float64)
    unweighted = np.ones(len(sites), dtype=np.float64)
    target_alignment_supported = (
        any(
            int(
                original.system.entity_type[
                    int(original.system.instance_entity[original.system.instance_index(instance_id)])
                ]
            )
            != ENTITY_MOLECULE
            for instance_id, _ in target
        )
        or int(np.sum(target_mask)) > 3
    )

    def subset(align_mask: NDArray[np.bool_], measure_mask: NDArray[np.bool_], *, weighted: bool = True) -> float:
        if not align_mask.any() or not measure_mask.any():
            return float("nan")
        return aligned_rmsd(
            reference,
            predicted,
            align_mask=align_mask,
            measure_mask=measure_mask,
            weights=weights,
            measure_weights=weights if weighted else unweighted,
        )

    metrics = {
        "rmsd": subset(all_mask, all_mask),
        "rmsd_design": subset(design_mask, design_mask),
        "rmsd_target": subset(target_mask, target_mask),
        "rmsd_design_target": float("nan"),
        "target_aligned_rmsd_design": float("nan"),
    }
    if target_alignment_supported:
        metrics["rmsd_design_target"] = subset(all_mask, design_mask, weighted=False)
        metrics["target_aligned_rmsd_design"] = subset(target_mask, design_mask, weighted=False)
    return metrics


def structure_design_lddt(
    original: Structure,
    refolded: Structure,
    *,
    design: Collection[ComponentReference],
    backbone_only: bool = False,
) -> float:
    r"""
    Return released same-chain intra-design lDDT for one coherent refold.

    Exact atom sites observed in both structures are paired.  Only pairs of
    designed atoms from the same molecular instance contribute.
    """

    reference, predicted, sites = structure_coordinates(
        original,
        refolded,
        backbone_only=backbone_only,
    )
    design = frozenset(design)
    is_design = np.array(
        [(site.instance_id, site.component_index) in design for site in sites],
        dtype=np.bool_,
    )
    instance_ids = np.array([site.instance_id for site in sites], dtype=object)
    pair_mask = is_design[:, None] & is_design[None, :] & (instance_ids[:, None] == instance_ids[None, :])
    return lddt(reference, predicted, pair_mask=pair_mask)


def assign_secondary_structure(
    structure: Structure,
    components: Collection[ComponentReference] | None = None,
) -> dict[ComponentReference, str]:
    r"""
    Assign released three-state DSSP classes to selected protein residues.

    The return values are ``"L"``, ``"H"``, and ``"E"``.  As in the released
    analysis, fewer than six selected residues produce no DSSP assignments.
    """

    import pydssp

    selected = None if components is None else frozenset(components)
    system = structure.system
    component_instances = _component_instance_index(structure).detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    references = []
    backbone = []
    for component_index, instance_index in enumerate(component_instances):
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
            continue
        reference = system.component_reference(component_index)
        if selected is not None and reference not in selected:
            continue
        by_name = {
            system.symbols[atom_symbols[atom_index]]: atom_index
            for atom_index in range(
                int(system.component_atom_ptr[component_index]),
                int(system.component_atom_ptr[component_index + 1]),
            )
            if bool(structure.coordinate_mask[atom_index])
        }
        names = ("N", "CA", "C", "O")
        if not all(name in by_name for name in names):
            raise ValueError(f"protein component {reference!r} does not have a complete " "observed N/CA/C/O backbone")
        references.append(reference)
        backbone.append(torch.stack([structure.coordinates[by_name[name]] for name in names]).detach().cpu().numpy())

    if len(backbone) <= 5:
        return {}
    assignment = np.asarray(pydssp.assign(np.stack(backbone), out_type="index"))
    dssp_states = ("L", "H", "E")
    return {reference: dssp_states[int(value)] for reference, value in zip(references, assignment, strict=True)}


def secondary_structure_fractions(assignments: Mapping[ComponentReference, str] | Sequence[str]) -> dict[str, float]:
    r"""
    Return loop, helix, and sheet fractions for three-state DSSP assignments.
    """

    values = tuple(assignments.values()) if isinstance(assignments, Mapping) else tuple(assignments)
    if not values:
        return {"loop": float("nan"), "helix": float("nan"), "sheet": float("nan")}
    return {
        "loop": values.count("L") / len(values),
        "helix": values.count("H") / len(values),
        "sheet": values.count("E") / len(values),
    }


def secondary_structure_agreement(
    predicted: Sequence[str],
    conditioned: Sequence[str | int | None],
) -> dict[str, float]:
    r"""
    Return released per-class precision, recall, Jaccard accuracy, and overall accuracy.

    ``None`` conditioning entries are ignored.
    """

    predicted_array = np.asarray(predicted)
    normalized_conditioned: list[str | None] = []
    for value in conditioned:
        if value is None or value == 0:
            normalized_conditioned.append(None)
        elif value == 1:
            normalized_conditioned.append("L")
        elif value == 2:
            normalized_conditioned.append("H")
        elif value == 3:
            normalized_conditioned.append("E")
        else:
            normalized_conditioned.append(str(value))
    conditioned_array = np.asarray(normalized_conditioned, dtype=object)
    conditioned_mask = np.fromiter((value is not None for value in conditioned_array), dtype=np.bool_)
    names = ("loop", "helix", "sheet")
    labels = ("L", "H", "E")
    metrics: dict[str, float] = {}
    true_positive_total = 0
    for label, name in zip(labels, names, strict=True):
        true_positive = int(np.sum((predicted_array == label) & (conditioned_array == label)))
        false_positive = int(np.sum((predicted_array == label) & (conditioned_array != label) & conditioned_mask))
        false_negative = int(np.sum((predicted_array != label) & (conditioned_array == label)))
        metrics[f"precision_{name}"] = (
            true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
        )
        metrics[f"recall_{name}"] = (
            true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
        )
        metrics[f"accuracy_{name}"] = (
            true_positive / (true_positive + false_positive + false_negative)
            if true_positive + false_positive + false_negative
            else float("nan")
        )
        true_positive_total += true_positive
    metrics["accuracy_overall"] = (
        true_positive_total / int(np.sum(conditioned_mask)) if np.any(conditioned_mask) else float("nan")
    )
    return metrics


def amino_acid_fractions(sequence: str) -> dict[str, float]:
    r"""
    Return all 20 released three-letter amino-acid fraction scores.

    ``sequence`` uses the one-letter designed sequence retained for BoltzGen
    deduplication and diversity selection.
    """

    return {
        f"{three_letter}_fraction": sequence.count(one_letter) / len(sequence)
        for one_letter, three_letter in AMINO_ACID_ONE_TO_THREE.items()
    }


def sequence_recovery(designed: str, native: str) -> float:
    r"""
    Return positional sequence recovery against a native designed sequence.
    """

    return sum(
        designed_residue == native_residue for designed_residue, native_residue in zip(designed, native, strict=True)
    ) / len(designed)


def sequence_hydrophobicity(sequence: str) -> float:
    r"""
    Return the released Thermo Fisher-like sequence hydrophobicity score.
    """

    sequence = (sequence or "").strip().upper()
    if not sequence or "X" in sequence:
        return float("nan")

    score = 0.0
    for index, residue in enumerate(sequence):
        if residue not in _HYDROPHOBICITY:
            raise ValueError(f"unknown residue {residue!r} in {sequence!r}")
        if index == 0:
            coefficient = "Rc1"
        elif index == 1:
            coefficient = "Rc2"
        elif index == len(sequence) - 1:
            coefficient = "Rn"
        elif index == len(sequence) - 2:
            coefficient = "Rn1"
        else:
            coefficient = "Rc"
        score += _HYDROPHOBICITY[residue][coefficient]

    for index, residue in enumerate(sequence):
        if residue in "HRK":
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(sequence):
                    score -= _NEIGHBOR_PENALTY.get(sequence[neighbor], 0.0)

    index = 0
    while index < len(sequence):
        if sequence[index] != "P":
            index += 1
            continue
        end = index
        while end < len(sequence) and sequence[end] == "P":
            end += 1
        run = end - index
        if run >= 4:
            score -= 5.0
        elif run == 3:
            score -= 3.5
        elif run == 2:
            score -= 1.2
        index = end

    if len(sequence) < 8:
        score *= 1.0 - 0.055 * (8 - len(sequence))
    elif len(sequence) > 20:
        score *= 1.0 / (1.0 + 0.027 * (len(sequence) - 20))

    if score <= 20:
        adjusted = score
    elif score <= 30:
        adjusted = score - 0.27 * (score - 18.0)
    elif score <= 40:
        adjusted = score - 0.33 * (score - 18.0)
    elif score <= 50:
        adjusted = score - 0.38 * (score - 18.0)
    else:
        adjusted = score - 0.447 * (score - 18.0)
    return round(adjusted, 4)


def sequence_liabilities(
    sequence: str,
    *,
    modality: LiabilityModality = "antibody",
    peptide_type: PeptideType = "linear",
) -> tuple[Liability, ...]:
    r"""
    Return every released antibody or peptide sequence liability.
    """

    violations: list[Liability] = []
    for name, pattern in _liability_motifs(modality, peptide_type).items():
        for match in pattern.finditer(sequence):
            violations.append(
                Liability(
                    name,
                    match.start() + 1,
                    len(match.group()),
                    _LIABILITY_SEVERITY.get(name, _DEFAULT_LIABILITY_SEVERITY),
                )
            )

    cysteines = [index for index, residue in enumerate(sequence) if residue == "C"]
    if modality == "antibody":
        paired: set[int] = set()
        for index in range(len(cysteines) - 1):
            if cysteines[index + 1] - cysteines[index] in (1, 2):
                paired.update((cysteines[index], cysteines[index + 1]))
        violations.extend(
            Liability("UnpairedCys", index + 1, 1, _LIABILITY_SEVERITY["UnpairedCys"])
            for index in cysteines
            if index not in paired
        )
        charge = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
        if charge > 1:
            violations.append(Liability("HighNetCharge", None, 0, _DEFAULT_LIABILITY_SEVERITY))
    elif peptide_type == "linear":
        if len(cysteines) % 2:
            violations.extend(
                Liability("UnpairedCys", index + 1, 1, _LIABILITY_SEVERITY["UnpairedCys"]) for index in cysteines
            )
    else:
        internal_cysteines = [index for index in cysteines if index not in (0, len(sequence) - 1)]
        if len(internal_cysteines) % 2:
            violations.extend(
                Liability("UnpairedCys", index + 1, 1, _LIABILITY_SEVERITY["UnpairedCys"])
                for index in internal_cysteines
            )

        hydrophilic = sum(sequence.count(residue) for residue in "DEKRHNQST")
        hydrophilic_percentage = hydrophilic / len(sequence) * 100
        if hydrophilic_percentage < 40:
            violations.append(
                Liability(
                    "LowHydrophilic",
                    None,
                    0,
                    _LIABILITY_SEVERITY["LowHydrophilic"],
                    f"{hydrophilic_percentage:.1f}% hydrophilic",
                )
            )

        longest_identical = 1
        current_identical = 1
        for index in range(1, len(sequence)):
            if sequence[index] == sequence[index - 1]:
                current_identical += 1
                longest_identical = max(longest_identical, current_identical)
            else:
                current_identical = 1
        if longest_identical > 1:
            violations.append(
                Liability(
                    "ConsecIdentical",
                    None,
                    0,
                    _LIABILITY_SEVERITY["ConsecIdentical"],
                    f"{longest_identical} consecutive identical",
                )
            )

        longest_hydrophobic = 0
        current_hydrophobic = 0
        for residue in sequence:
            current_hydrophobic = current_hydrophobic + 1 if residue in "FILVWY" else 0
            longest_hydrophobic = max(longest_hydrophobic, current_hydrophobic)
        if longest_hydrophobic > 4:
            violations.append(
                Liability(
                    "LongHydrophobic",
                    None,
                    0,
                    _LIABILITY_SEVERITY["LongHydrophobic"],
                    f"{longest_hydrophobic} consecutive hydrophobic",
                )
            )
    return tuple(violations)


def liability_metrics(
    sequence: str,
    *,
    modality: LiabilityModality = "antibody",
    peptide_type: PeptideType = "linear",
) -> dict[str, AnalysisValue]:
    r"""
    Return released aggregate and per-motif liability values.
    """

    if not sequence:
        raise ValueError("sequence is empty")
    violations = sequence_liabilities(sequence, modality=modality, peptide_type=peptide_type)
    metrics: dict[str, AnalysisValue] = {
        "liability_score": sum(violation.severity for violation in violations),
        "liability_num_violations": len(violations),
        "liability_high_severity_violations": sum(violation.severity >= 10 for violation in violations),
        "liability_medium_severity_violations": sum(5 <= violation.severity < 10 for violation in violations),
        "liability_low_severity_violations": sum(violation.severity < 5 for violation in violations),
    }
    grouped: dict[str, list[Liability]] = {}
    for violation in violations:
        grouped.setdefault(violation.motif, []).append(violation)
    motif_names = set(_liability_motifs(modality, peptide_type))
    motif_names.update(grouped)
    for motif in sorted(motif_names):
        matches = grouped.get(motif, [])
        positions = [violation.position for violation in matches if violation.position is not None]
        details = [violation.details for violation in matches if violation.details]
        metrics[f"liability_{motif}_count"] = len(matches)
        metrics[f"liability_{motif}_position"] = -1
        metrics[f"liability_{motif}_length"] = 0
        metrics[f"liability_{motif}_severity"] = 0
        metrics[f"liability_{motif}_details"] = ""
        metrics[f"liability_{motif}_positions"] = ""
        metrics[f"liability_{motif}_num_positions"] = 0
        metrics[f"liability_{motif}_global_details"] = ""
        metrics[f"liability_{motif}_avg_severity"] = 0.0
        if len(matches) == 1:
            violation = matches[0]
            metrics[f"liability_{motif}_position"] = -1 if violation.position is None else violation.position
            metrics[f"liability_{motif}_length"] = violation.length
            metrics[f"liability_{motif}_severity"] = violation.severity
            metrics[f"liability_{motif}_details"] = violation.details
        elif matches:
            if positions:
                metrics[f"liability_{motif}_positions"] = f"{min(positions)}-{max(positions)}"
                metrics[f"liability_{motif}_num_positions"] = len(positions)
            metrics[f"liability_{motif}_global_details"] = "; ".join(details)
            metrics[f"liability_{motif}_avg_severity"] = round(
                sum(violation.severity for violation in matches) / len(matches),
                1,
            )

    summary: list[str] = []
    for motif, matches in grouped.items():
        positions = [violation.position for violation in matches if violation.position is not None]
        if len(matches) == 1:
            violation = matches[0]
            if violation.position is not None:
                summary.append(f"{motif}(pos{violation.position},sev{violation.severity})")
            elif violation.details:
                summary.append(f"{motif}({violation.details},sev{violation.severity})")
            else:
                summary.append(f"{motif}(sev{violation.severity})")
        elif positions:
            summary.append(f"{motif}x{len(matches)}(pos{min(positions)}-{max(positions)},sev{matches[0].severity})")
        else:
            summary.append(f"{motif}x{len(matches)}(sev{matches[0].severity})")
    metrics["liability_violations_summary"] = "; ".join(summary)
    return metrics


def count_noncovalent_contacts(
    structure: Structure,
    *,
    design: Collection[ComponentReference],
    chain_design: Collection[ComponentReference],
) -> NoncovalentContacts:
    r"""
    Count released design-to-target hydrogen bonds and charged atom pairs.

    Hydrogens are added with Hydride and hydrogen bonds are assigned with
    Biotite.  Charged pairs use the strict released distance interval
    ``0.5 < d < 5.5`` Å.
    """

    import hydride

    atom_array, _ = _biotite_atom_array(structure, design=design, chain_design=chain_design)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atom_array, _ = hydride.add_hydrogen(atom_array)
        hydrogen_bonds = struc.hbond(atom_array)

    donors = hydrogen_bonds[:, 0]
    acceptors = hydrogen_bonds[:, 2]
    charge = atom_array.get_annotation("charge")
    is_design = atom_array.get_annotation("is_design")
    is_chain_design = atom_array.get_annotation("is_chain_design")
    design_donors = is_design[donors] & ~is_chain_design[acceptors]
    design_acceptors = ~is_chain_design[donors] & is_design[acceptors]
    hydrogen_bond_count = int(np.sum(design_donors) + np.sum(design_acceptors))

    positive = np.flatnonzero(charge > 0)
    negative = np.flatnonzero(charge < 0)
    charged_pair_count = 0
    if len(positive) and len(negative):
        distances = np.linalg.norm(atom_array.coord[positive, None] - atom_array.coord[None, negative], axis=-1)
        positive_index, negative_index = np.where((distances > 0.5) & (distances < 5.5))
        positive_design = is_design[positive[positive_index]] & ~is_chain_design[negative[negative_index]]
        negative_design = ~is_chain_design[positive[positive_index]] & is_design[negative[negative_index]]
        charged_pair_count = int(np.sum(positive_design) + np.sum(negative_design))
    return NoncovalentContacts(hydrogen_bond_count, charged_pair_count)


def target_delta_sasa(
    structure: Structure,
    *,
    target: Collection[ComponentReference],
    design: Collection[ComponentReference],
) -> TargetDeltaSasa:
    r"""
    Return target delta-SASA with the released 1.4 Å/960-point/ProtOr operation.
    """

    target = frozenset(target)
    bound_array, bound_sites = _biotite_atom_array(structure, components=target | frozenset(design))
    bound_sasa = _atom_sasa(bound_array)
    target_in_bound = np.array(
        [(site.instance_id, site.component_index) in target for site in bound_sites],
        dtype=np.bool_,
    )
    target_bound = float(np.sum(bound_sasa[target_in_bound]))

    target_array, _ = _biotite_atom_array(structure, components=target)
    target_unbound = float(np.sum(_atom_sasa(target_array)))
    return TargetDeltaSasa(target_unbound - target_bound, target_unbound, target_bound)


def largest_hydrophobic_patch_area(
    structure: Structure,
    *,
    components: Collection[ComponentReference] | None = None,
    distance_cutoff: float = 6.0,
) -> float:
    r"""
    Return the largest connected exposed hydrophobic-carbon patch area.

    Exposed carbon atoms in the released hydrophobic residue set are connected
    transitively when their distance is at most ``distance_cutoff`` Å.
    """

    atom_array, _ = _biotite_atom_array(structure, components=components)
    atom_sasa = _atom_sasa(atom_array)
    mask = (
        np.isin(atom_array.res_name, tuple(_HYDROPHOBIC_RESIDUES))
        & np.char.startswith(atom_array.atom_name, "C")
        & (atom_sasa > 0)
    )
    coordinates = atom_array.coord[mask]
    areas = atom_sasa[mask]
    if not len(coordinates):
        return float("nan")

    adjacent = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1) <= distance_cutoff
    unseen = set(range(len(coordinates)))
    largest = 0.0
    while unseen:
        frontier = [unseen.pop()]
        component = []
        while frontier:
            index = frontier.pop()
            component.append(index)
            neighbors = {int(neighbor) for neighbor in np.flatnonzero(adjacent[index])} & unseen
            unseen.difference_update(neighbors)
            frontier.extend(neighbors)
        largest = max(largest, float(np.sum(areas[component])))
    return largest


def sequence_identity(first: str, second: str) -> float:
    r"""
    Return released pairwise-alignment score divided by the longer length.

    The released aligner uses global alignment with match ``1``, mismatch ``0``,
    and a ``-1`` score for each gap position.
    """

    previous = -np.arange(len(second) + 1, dtype=np.int64)
    for first_index, first_residue in enumerate(first, start=1):
        current = np.empty(len(second) + 1, dtype=np.int64)
        current[0] = -first_index
        for index, second_residue in enumerate(second, start=1):
            current[index] = max(
                previous[index - 1] + int(first_residue == second_residue),
                previous[index] - 1,
                current[index - 1] - 1,
            )
        previous = current
    return float(previous[-1] / max(len(first), len(second)))


def binding_site_adherence(
    binding_site_coordinates: NDArray,
    design_coordinates: NDArray,
    *,
    distance_threshold: float = 8.0,
) -> float:
    r"""
    Return the fraction of binding-site centers within a strict distance of a design center.

    This is the released ``bindsite_under_<threshold>rmsd`` operation; despite
    the upstream metric name, it is a nearest-center distance fraction rather
    than an RMSD.
    """

    return binding_site_adherence_curve(
        binding_site_coordinates,
        design_coordinates,
        thresholds=(distance_threshold,),
    )[distance_threshold]


def binding_site_adherence_curve(
    binding_site_coordinates: NDArray,
    design_coordinates: NDArray,
    *,
    thresholds: Sequence[float] = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
) -> dict[float, float]:
    r"""
    Return released binding-site adherence at each requested strict cutoff.
    """

    distances = np.linalg.norm(
        np.asarray(binding_site_coordinates)[:, None] - np.asarray(design_coordinates)[None, :],
        axis=-1,
    )
    minimum_distance = np.min(distances, axis=1)
    return {threshold: float(np.mean(minimum_distance < threshold)) for threshold in thresholds}


def deduplicate_sequences(sequences: Sequence[str]) -> tuple[int, ...]:
    r"""
    Return indices of the earliest occurrence of every designed sequence.
    """

    seen: set[str] = set()
    indices: list[int] = []
    for index, sequence in enumerate(sequences):
        if sequence not in seen:
            seen.add(sequence)
            indices.append(index)
    return tuple(indices)


def candidate_filter_results(
    row: Mapping[str, AnalysisValue],
    filters: Sequence[CandidateFilter],
) -> dict[str, bool]:
    r"""
    Return every released named inclusive filter outcome for one candidate.
    """

    return {
        f"pass_{candidate_filter.feature}_filter": (
            float(row[candidate_filter.feature]) <= candidate_filter.threshold
            if candidate_filter.lower_is_better
            else float(row[candidate_filter.feature]) >= candidate_filter.threshold
        )
        for candidate_filter in filters
    }


def rank_candidates(
    rows: Sequence[Mapping[str, AnalysisValue]],
    *,
    filters: Sequence[CandidateFilter],
    metric_weights: Mapping[str, float],
    tie_breaker: str,
) -> CandidateRanking:
    r"""
    Filter and rank BoltzGen candidates with the released worst-rank rule.

    Every metric is min-ranked descending on
    ``(number of individual filters passed, metric value)`` and divided by its
    inverse-importance weight.  The worst scaled metric rank determines a dense
    rank group; ``tie_breaker`` then produces the stable final order.
    """

    named_filter_results = [candidate_filter_results(row, filters) for row in rows]
    filter_passes = np.array(
        [
            [results[f"pass_{candidate_filter.feature}_filter"] for candidate_filter in filters]
            for results in named_filter_results
        ],
        dtype=np.bool_,
    )
    filters_passed = np.sum(filter_passes, axis=1, dtype=np.int64)
    passes_all_filters = np.all(filter_passes, axis=1)

    scaled_ranks: list[NDArray] = []
    for metric, inverse_importance in metric_weights.items():
        values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
        metric_rank = _minimum_descending_rank(filters_passed, values)
        scaled_ranks.append(metric_rank / inverse_importance)
    worst_rank = np.max(np.stack(scaled_ranks), axis=0)
    dense_rank = np.searchsorted(np.unique(np.sort(worst_rank)), worst_rank) + 1
    tie_values = np.array([float(row[tie_breaker]) for row in rows], dtype=np.float64)
    order = tuple(sorted(range(len(rows)), key=lambda index: (dense_rank[index], -tie_values[index], index)))

    final_rank = np.empty(len(rows), dtype=np.int64)
    for index, row_index in enumerate(order, start=1):
        final_rank[row_index] = index
    if len(rows) == 1:
        quality = np.ones(1, dtype=np.float64)
    else:
        quality = 1.0 - (final_rank - 1) / (len(rows) - 1)
    return CandidateRanking(order, final_rank, quality, filters_passed, passes_all_filters)


def greedy_diversity_selection(
    sequences: Sequence[str],
    quality: Sequence[float],
    *,
    budget: int = 30,
    alpha: float = 0.001,
    size_buckets: Sequence[tuple[int, int, int]] = (),
) -> tuple[int, ...]:
    r"""
    Select the released quality--sequence-diversity candidate set.

    Each size bucket is ``(minimum_inclusive, maximum_exclusive, budget)``.
    Returned indices are in their original stable order, matching the released
    selection output.
    """

    quality_array = np.asarray(quality, dtype=np.float64)
    if len(quality_array) <= budget:
        return tuple(range(len(quality_array)))

    selected = [int(np.argmax(quality_array))]
    bucket_counts = np.zeros(len(size_buckets), dtype=np.int64)

    def bucket_index(index: int) -> int | None:
        length = len(sequences[index])
        for position, (minimum, maximum, _) in enumerate(size_buckets):
            if minimum <= length < maximum:
                return position
        return None

    first_bucket = bucket_index(selected[0])
    if first_bucket is not None:
        bucket_counts[first_bucket] += 1

    remaining = set(range(len(quality_array))) - set(selected)
    while len(selected) < budget:
        best_index = None
        best_gain = -float("inf")
        for index in sorted(remaining):
            bucket = bucket_index(index)
            if bucket is not None and bucket_counts[bucket] >= size_buckets[bucket][2]:
                continue
            maximum_identity = max(sequence_identity(sequences[index], sequences[chosen]) for chosen in selected)
            gain = (1.0 - alpha) * quality_array[index] + alpha * (1.0 - maximum_identity)
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        selected_bucket = bucket_index(best_index)
        if selected_bucket is not None:
            bucket_counts[selected_bucket] += 1
    return tuple(sorted(selected))


def vendi_score(similarity: NDArray) -> float:
    r"""
    Return the Vendi diversity score of a symmetric similarity matrix.
    """

    similarity = np.asarray(similarity, dtype=np.float64).copy()
    np.fill_diagonal(similarity, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(similarity / len(similarity)), 0.0, None)
    nonzero = eigenvalues > 0
    return float(np.exp(-np.sum(eigenvalues[nonzero] * np.log(eigenvalues[nonzero]))))


def sequence_vendi_score(sequences: Sequence[str]) -> float:
    r"""
    Return the released Vendi score over pairwise sequence similarities.
    """

    similarity = np.eye(len(sequences), dtype=np.float64)
    for first in range(len(sequences)):
        for second in range(first + 1, len(sequences)):
            similarity[first, second] = similarity[second, first] = sequence_identity(
                sequences[first], sequences[second]
            )
    return vendi_score(similarity)


def structure_tm_scores(first: NDArray, second: NDArray) -> tuple[float, float]:
    r"""
    Return released RMSD-aligned and structural-homology-aligned TM scores.
    """

    first_atoms = _ca_atom_array(first)
    second_atoms = _ca_atom_array(second)
    homology_score = 0.0
    try:
        aligned, _, fixed_indices, mobile_indices = struc.superimpose_structural_homologs(
            first_atoms,
            second_atoms,
            max_iterations=25,
        )
        homology_score = float(struc.tm_score(first_atoms, aligned, fixed_indices, mobile_indices))
    except ValueError:
        pass

    rmsd_aligned_score = 0.0
    if len(first) == len(second):
        aligned_coordinates = _aligned_coordinates(np.asarray(first), np.asarray(second))
        aligned_atoms = _ca_atom_array(aligned_coordinates)
        try:
            _, _, fixed_indices, mobile_indices = struc.superimpose_structural_homologs(
                aligned_atoms,
                second_atoms,
                max_iterations=25,
            )
            rmsd_aligned_score = float(struc.tm_score(aligned_atoms, second_atoms, fixed_indices, mobile_indices))
        except ValueError:
            pass
    return rmsd_aligned_score, homology_score


def structure_vendi_scores(
    coordinates: Sequence[NDArray],
    metrics: Sequence[BoltzGenDiversityMetrics] | None = None,
    *,
    all_atom: bool = True,
    backbone: bool = False,
) -> BoltzGenStructureVendiScores:
    r"""
    Return released base and conditionally filtered structural Vendi scores.

    ``metrics`` enables the released RMSD, lDDT, design-iPTM, and minimum-PAE
    conditions.  The same structural similarity matrix is filtered separately
    by all-atom and backbone candidate metrics.
    """

    rmsd_aligned = np.eye(len(coordinates), dtype=np.float64)
    homology_aligned = np.eye(len(coordinates), dtype=np.float64)
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            rmsd_score, homology_score = structure_tm_scores(
                coordinates[first],
                coordinates[second],
            )
            rmsd_aligned[first, second] = rmsd_aligned[second, first] = rmsd_score
            homology_aligned[first, second] = homology_score
            homology_aligned[second, first] = homology_score
    return BoltzGenStructureVendiScores(
        vendi_tm_fixed=vendi_score(homology_aligned),
        vendi_tm_align=vendi_score(rmsd_aligned),
        all_atom=(
            _conditioned_vendi_scores(homology_aligned, metrics, backbone=False)
            if metrics is not None and all_atom
            else None
        ),
        backbone=(
            _conditioned_vendi_scores(homology_aligned, metrics, backbone=True)
            if metrics is not None and backbone
            else None
        ),
    )


def analyze_diversity(
    samples: Sequence[BoltzGenDiversitySample],
    *,
    all_atom: bool = True,
    backbone: bool = True,
) -> BoltzGenDiversityResult:
    r"""
    Aggregate released original, refolded, and per-target diversity values.

    Sequence diversity belongs to the original candidate collection; refolding
    changes coordinates but not sequences.  Global values use the first
    generation-ordered design for each target; per-target values use every
    design for that target.  Structures shorter than three design residues are
    excluded from structural Vendi values, matching the released reduction.
    """

    sample_values = tuple(samples)
    grouped: dict[str, list[BoltzGenDiversitySample]] = {}
    for sample in sample_values:
        grouped.setdefault(sample.target_id, []).append(sample)
    representative_samples = tuple(target_samples[0] for target_samples in grouped.values())
    return BoltzGenDiversityResult(
        original=_diversity_aggregate(
            representative_samples,
            refolded=False,
            include_sequence=True,
            all_atom=all_atom,
            backbone=backbone,
        ),
        refolded=_diversity_aggregate(
            representative_samples,
            refolded=True,
            include_sequence=False,
            all_atom=all_atom,
            backbone=backbone,
        ),
        per_target_original={
            target_id: _diversity_aggregate(
                target_samples,
                refolded=False,
                include_sequence=True,
                all_atom=all_atom,
                backbone=backbone,
            )
            for target_id, target_samples in grouped.items()
        },
        per_target_refolded={
            target_id: _diversity_aggregate(
                target_samples,
                refolded=True,
                include_sequence=False,
                all_atom=all_atom,
                backbone=backbone,
            )
            for target_id, target_samples in grouped.items()
        },
    )


def _diversity_aggregate(
    samples: Sequence[BoltzGenDiversitySample],
    *,
    refolded: bool,
    include_sequence: bool,
    all_atom: bool,
    backbone: bool,
) -> BoltzGenDiversityAggregate:
    structural_samples = tuple(
        sample
        for sample in samples
        if len(sample.refolded_coordinates if refolded else sample.original_coordinates) >= 3
    )
    coordinates = tuple(
        sample.refolded_coordinates if refolded else sample.original_coordinates for sample in structural_samples
    )
    return BoltzGenDiversityAggregate(
        structure=structure_vendi_scores(
            coordinates,
            tuple(sample.metrics for sample in structural_samples),
            all_atom=all_atom,
            backbone=backbone,
        ),
        vendi_seq_sim=(
            sequence_vendi_score(tuple(sample.sequence for sample in samples)) if include_sequence else None
        ),
        num_structures=len(structural_samples),
        num_sequences=len(samples) if include_sequence else 0,
    )


def _conditioned_vendi_scores(
    similarity: NDArray,
    metrics: Sequence[BoltzGenDiversityMetrics],
    *,
    backbone: bool,
) -> BoltzGenConditionedVendiScores:
    if backbone:
        rmsd = np.array([metric.bb_rmsd for metric in metrics])
        design_rmsd = np.array([metric.bb_rmsd_design for metric in metrics])
        target_aligned_rmsd = np.array([metric.bb_target_aligned_rmsd_design for metric in metrics])
        design_lddt = tuple(metric.bb_lddt_intra_design for metric in metrics)
    else:
        rmsd = np.array([metric.rmsd for metric in metrics])
        design_rmsd = np.array([metric.rmsd_design for metric in metrics])
        target_aligned_rmsd = np.array([metric.target_aligned_rmsd_design for metric in metrics])
        design_lddt = tuple(metric.lddt_intra_design for metric in metrics)
    design_iptm = np.array([metric.design_iptm for metric in metrics])
    min_interaction_pae = np.array([metric.min_interaction_pae for metric in metrics])

    lddt_scores: tuple[float, ...] | None = None
    if all(value is not None for value in design_lddt):
        lddt = np.array([float(value) for value in design_lddt if value is not None])
        lddt_scores = tuple(
            _subset_vendi_score(similarity, lddt >= threshold)
            for threshold in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
        )

    return BoltzGenConditionedVendiScores(
        rmsd_below_2_5=_subset_vendi_score(similarity, rmsd <= 2.5),
        design_rmsd_below_2=_subset_vendi_score(similarity, design_rmsd <= 2.0),
        design_rmsd_below_4=_subset_vendi_score(similarity, design_rmsd <= 4.0),
        target_aligned_design_rmsd_below_2_5=_subset_vendi_score(
            similarity,
            target_aligned_rmsd <= 2.5,
        ),
        design_lddt_at_least_0_60=None if lddt_scores is None else lddt_scores[0],
        design_lddt_at_least_0_65=None if lddt_scores is None else lddt_scores[1],
        design_lddt_at_least_0_70=None if lddt_scores is None else lddt_scores[2],
        design_lddt_at_least_0_75=None if lddt_scores is None else lddt_scores[3],
        design_lddt_at_least_0_80=None if lddt_scores is None else lddt_scores[4],
        design_lddt_at_least_0_85=None if lddt_scores is None else lddt_scores[5],
        design_lddt_at_least_0_90=None if lddt_scores is None else lddt_scores[6],
        design_iptm_at_least_0_80=_subset_vendi_score(
            similarity,
            design_iptm >= 0.8,
        ),
        design_iptm_at_least_0_70=_subset_vendi_score(
            similarity,
            design_iptm >= 0.7,
        ),
        design_iptm_at_least_0_60=_subset_vendi_score(
            similarity,
            design_iptm >= 0.6,
        ),
        design_iptm_at_least_0_50=_subset_vendi_score(
            similarity,
            design_iptm >= 0.5,
        ),
        min_interaction_pae_at_most_1_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 1.5,
        ),
        min_interaction_pae_at_most_2=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 2.0,
        ),
        min_interaction_pae_at_most_2_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 2.5,
        ),
        min_interaction_pae_at_most_3=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 3.0,
        ),
        min_interaction_pae_at_most_4=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 4.0,
        ),
        min_interaction_pae_at_most_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 5.0,
        ),
    )


def _subset_vendi_score(
    similarity: NDArray,
    mask: NDArray[np.bool_],
) -> float:
    return vendi_score(similarity[mask][:, mask]) if bool(np.any(mask)) else 0.0


def foldseek_novelty(
    structures: Sequence[Structure],
    operation: FoldseekNoveltyOperation,
) -> NDArray[np.float64]:
    r"""
    Run an explicit in-memory Foldseek boundary for released novelty values.

    The supplied operation owns any Foldseek repository, process, or transport
    details.  This function retains the released scientific reduction: average
    query- and target-normalized TM scores, then take the maximum per query,
    filling unmatched queries with zero.
    """

    novelty = np.zeros(len(structures), dtype=np.float64)
    for match in operation(structures):
        score = (match.query_tm_score + match.target_tm_score) / 2.0
        novelty[match.query_index] = max(novelty[match.query_index], score)
    return novelty


def foldseek_clusters(
    structures: Sequence[Structure],
    operation: FoldseekClusteringOperation,
    *,
    tmscore_threshold: float = 0.5,
) -> tuple[str, ...]:
    r"""
    Run an explicit in-memory Foldseek boundary for released cluster labels.
    """

    return tuple(operation(structures, tmscore_threshold))


def _liability_motifs(
    modality: LiabilityModality,
    peptide_type: PeptideType,
) -> dict[str, re.Pattern[str]]:
    if modality == "antibody":
        return {
            "DeAmdH": re.compile(r"N[GS]"),
            "DeAmdM": re.compile(r"N[AHNT]"),
            "DeAmdL": re.compile(r"[STK]N"),
            "Ngly": re.compile(r"N[^P][ST]"),
            "Isom": re.compile(r"D[DGHST]"),
            "Isomer": re.compile(r"DG|DS|DD"),
            "FragH": re.compile(r"DP"),
            "FragM": re.compile(r"TS"),
            "TrpOx": re.compile(r"W"),
            "MetOx": re.compile(r"M"),
            "Hydro": re.compile(r"NP"),
            "IntBind": re.compile(r"GPR|RGD|RYD|LDV|DGE|KGD|NGR"),
            "Polyreactive": re.compile(r"GGG|GG|RR|VG|VVV|WWW|YY|WxW"),
            "AggPatch": re.compile(r"FHW"),
            "ViscPatch": re.compile(r"HYF|HWH"),
        }
    motifs = {
        "AspBridge": re.compile(r"N[GSQA]"),
        "AspCleave": re.compile(r"D[PGS]"),
        "NTCycl": re.compile(r"^[QN]"),
        "ProtTryp": re.compile(r"[KR](?=.)"),
        "DPP4": re.compile(r"^[PX]?[AP]"),
        "MetOx": re.compile(r"M"),
        "TrpOx": re.compile(r"W"),
        "HydroPatch": re.compile(r"[FILVWY]{3,}"),
    }
    if peptide_type == "cyclic":
        motifs.pop("NTCycl")
        motifs.pop("DPP4")
    return motifs


def _is_backbone_site(structure: Structure, site: AtomSite) -> bool:
    system = structure.system
    instance_index = system.instance_index(site.instance_id)
    entity_index = int(system.instance_entity[instance_index])
    entity_type = int(system.entity_type[entity_index])
    if entity_type == ENTITY_MOLECULE:
        return False
    if entity_type == ENTITY_PROTEIN:
        return site.atom_id in _PROTEIN_BACKBONE_ATOMS
    if entity_type == ENTITY_DNA:
        return site.atom_id in _DNA_BACKBONE_ATOMS
    if entity_type == ENTITY_RNA:
        return site.atom_id in _RNA_BACKBONE_ATOMS
    return site.atom_id in _RNA_BACKBONE_ATOMS


def _atom_rmsd_weight(structure: Structure, site: AtomSite) -> float:
    system = structure.system
    instance_index = system.instance_index(site.instance_id)
    entity_index = int(system.instance_entity[instance_index])
    entity_type = int(system.entity_type[entity_index])
    if entity_type == ENTITY_MOLECULE:
        return 11.0
    return 1.0 if entity_type == ENTITY_PROTEIN else 6.0


def _biotite_atom_array(
    structure: Structure,
    *,
    components: Collection[ComponentReference] | None = None,
    design: Collection[ComponentReference] = (),
    chain_design: Collection[ComponentReference] = (),
) -> tuple[struc.AtomArray, tuple[AtomSite, ...]]:
    components = None if components is None else frozenset(components)
    design = frozenset(design)
    chain_design = frozenset(chain_design)
    system = structure.system
    all_sites = _canonical_atom_sites(structure)
    indices = [
        index
        for index, site in enumerate(all_sites)
        if bool(structure.coordinate_mask[index])
        and (components is None or (site.instance_id, site.component_index) in components)
    ]
    sites = tuple(all_sites[index] for index in indices)
    atom_array = struc.AtomArray(len(indices))
    canonical_index = torch.tensor(
        indices,
        dtype=torch.long,
        device=structure.coordinates.device,
    )
    atom_array.coord = structure.coordinates[canonical_index].detach().cpu().numpy()
    chain_ids = {system.instance_id(index): str(index) for index in range(system.instance_symbol.numel())}
    atom_array.chain_id = np.array([chain_ids[site.instance_id] for site in sites])
    atom_array.res_id = np.array([site.component_index + 1 for site in sites])
    atom_array.ins_code = np.full(len(sites), "")
    atom_array.atom_name = np.array([site.atom_id for site in sites])
    atom_array.res_name = np.array(
        [
            system.symbols[
                int(system.component_symbol[system.component_index((site.instance_id, site.component_index))])
            ]
            for site in sites
        ]
    )
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    atom_array.element = np.array(
        [periodic_table.GetElementSymbol(int(system.atomic_number[index])) for index in indices]
    )
    atom_array.hetero = np.array(
        [
            int(system.entity_type[int(system.instance_entity[system.instance_index(site.instance_id)])])
            == ENTITY_MOLECULE
            for site in sites
        ],
        dtype=np.bool_,
    )
    atom_array.set_annotation(
        "charge",
        np.array(
            [int(system.formal_charge[index]) if bool(system.formal_charge_known[index]) else 0 for index in indices],
            dtype=np.int64,
        ),
    )
    atom_array.set_annotation(
        "is_design",
        np.array(
            [(site.instance_id, site.component_index) in design for site in sites],
            dtype=np.bool_,
        ),
    )
    atom_array.set_annotation(
        "is_chain_design",
        np.array(
            [(site.instance_id, site.component_index) in chain_design for site in sites],
            dtype=np.bool_,
        ),
    )

    selected_index = {canonical_atom_index: output_index for output_index, canonical_atom_index in enumerate(indices)}
    bonds: list[tuple[int, int, int]] = []
    for edge_index, (first, second) in enumerate(system.edge_index.detach().cpu().T.tolist()):
        if first in selected_index and second in selected_index:
            bonds.append(
                (
                    selected_index[first],
                    selected_index[second],
                    int(
                        _biotite_bond_type(
                            float(system.bond_order[edge_index]) if bool(system.bond_order_known[edge_index]) else 0.0,
                            (
                                bool(system.bond_aromatic[edge_index])
                                if bool(system.bond_aromatic_known[edge_index])
                                else False
                            ),
                        )
                    ),
                )
            )
    atom_array.bonds = struc.BondList(len(atom_array), np.asarray(bonds, dtype=np.uint32).reshape(-1, 3))
    return atom_array, sites


def _biotite_bond_type(order: float, aromatic: bool) -> struc.BondType:
    if aromatic:
        return struc.BondType.AROMATIC
    return {
        1.0: struc.BondType.SINGLE,
        2.0: struc.BondType.DOUBLE,
        3.0: struc.BondType.TRIPLE,
        4.0: struc.BondType.QUADRUPLE,
    }.get(order, struc.BondType.ANY)


def _atom_sasa(atom_array: struc.AtomArray) -> NDArray:
    radii: list[float] = []
    for residue, atom_name, element in zip(
        atom_array.res_name,
        atom_array.atom_name,
        atom_array.element,
        strict=True,
    ):
        try:
            radius = vdw_radius_protor(str(residue).strip(), str(atom_name).strip())
        except KeyError:
            radius = None
        if radius is None:
            radius = vdw_radius_single(str(element))
        radii.append(1.8 if radius is None else radius)
    return struc.sasa(
        atom_array,
        probe_radius=1.4,
        point_number=960,
        vdw_radii=np.asarray(radii, dtype=np.float64),
    )


def _minimum_descending_rank(primary: NDArray, secondary: NDArray) -> NDArray[np.int64]:
    order = sorted(range(len(primary)), key=lambda index: (-primary[index], -secondary[index], index))
    rank = np.empty(len(primary), dtype=np.int64)
    previous: tuple[object, object] | None = None
    current_rank = 0
    for position, index in enumerate(order, start=1):
        key = (primary[index], secondary[index])
        if key != previous:
            current_rank = position
            previous = key
        rank[index] = current_rank
    return rank


def _aligned_coordinates(
    reference: NDArray,
    predicted: NDArray,
    weights: NDArray | None = None,
) -> NDArray:
    if weights is None:
        weights = np.ones(len(reference), dtype=np.float64)
    reference_centroid = np.sum(reference * weights[:, None], axis=0) / np.sum(weights)
    predicted_centroid = np.sum(predicted * weights[:, None], axis=0) / np.sum(weights)
    reference_centered = reference - reference_centroid
    predicted_centered = predicted - predicted_centroid
    covariance = (predicted_centered * weights[:, None]).T @ reference_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_transpose)
    return reference_centered @ (left @ correction @ right_transpose).T + predicted_centroid


def _ca_atom_array(coordinates: NDArray) -> struc.AtomArray:
    atom_array = struc.AtomArray(len(coordinates))
    atom_array.coord = np.asarray(coordinates)
    atom_array.element = np.full(len(coordinates), "C")
    atom_array.atom_name = np.full(len(coordinates), "CA")
    atom_array.res_name = np.full(len(coordinates), "ALA")
    atom_array.chain_id = np.full(len(coordinates), "A")
    atom_array.res_id = np.arange(1, len(coordinates) + 1)
    return atom_array
