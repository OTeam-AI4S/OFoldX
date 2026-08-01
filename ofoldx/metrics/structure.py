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
Coordinate RMSD, lDDT, and structural TM scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import biotite.structure as struc
import numpy as np
import torch
from numpy.typing import NDArray


def backbone_secondary_structure(
    backbone: NDArray,
    peptide_bond_index: NDArray,
    donor_mask: NDArray[np.bool_],
) -> NDArray:
    r"""Assign simplified DSSP ``L/H/E`` states in the supplied peptide graph.

    ``backbone[N, 4, 3]`` contains observed N/CA/C/O coordinates. Directed
    ``peptide_bond_index[2, E]`` connects each carbonyl residue to its successor;
    disconnected and circular chains retain their actual topology. ``donor_mask``
    excludes proline and any other residue without an amide hydrogen. Missing
    predecessors have no inferred donor H. All carbonyls remain acceptors.

    Uses the Kabsch--Sander electrostatic hydrogen-bond equation and PyDSSP's
    three/four/five-turn and bridge definitions, expressed through real peptide
    neighbors instead of array offsets. Beta bridges can cross chain boundaries.
    This is the simplified three-state operation, not a complete mkdssp program.
    """

    backbone = np.asarray(backbone)
    num_residues = len(backbone)
    if not num_residues:
        return np.empty(0, dtype="U1")
    source, target = np.asarray(peptide_bond_index)
    if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
        raise ValueError("secondary structure requires a nonbranching peptide backbone")
    previous = np.full(num_residues, -1, dtype=np.int64)
    following = np.full(num_residues, -1, dtype=np.int64)
    previous[target] = source
    following[source] = target
    nitrogen, alpha_carbon, carbon, oxygen = np.moveaxis(backbone, 1, 0)
    # Kabsch & Sander (1983), doi:10.1002/bip.360221211; PyDSSP (MIT,
    # Copyright (c) 2022 Shintaro Minami) provides the inspected turn conventions:
    # e251a43ff8622fe0a555313b1567edce45e789e8, pydssp_numpy.py.
    with np.errstate(divide="ignore", invalid="ignore"):
        cn = nitrogen - carbon[np.maximum(previous, 0)]
        ca_n = nitrogen - alpha_carbon
        nh = cn / np.linalg.norm(cn, axis=-1, keepdims=True) + ca_n / np.linalg.norm(ca_n, axis=-1, keepdims=True)
        hydrogen = nitrogen + 1.01 * nh / np.linalg.norm(nh, axis=-1, keepdims=True)
        on = np.linalg.norm(oxygen[:, None] - nitrogen[None, :], axis=-1)
        ch = np.linalg.norm(carbon[:, None] - hydrogen[None, :], axis=-1)
        oh = np.linalg.norm(oxygen[:, None] - hydrogen[None, :], axis=-1)
        cn_distance = np.linalg.norm(carbon[:, None] - nitrogen[None, :], axis=-1)
        energy = (0.084 * 332) * (1 / on + 1 / ch - 1 / oh - 1 / cn_distance)
    hydrogen_bond = (energy < -0.5) & ((previous >= 0) & np.asarray(donor_mask, dtype=np.bool_))[None, :]

    paths = np.full((num_residues, 7), -1, dtype=np.int64)
    paths[:, 0] = np.arange(num_residues)
    for step in range(1, 7):
        preceding = paths[:, step - 1]
        paths[:, step] = np.where(preceding >= 0, following[np.maximum(preceding, 0)], -1)
    # The carbonyl and its next two peptide successors do not form scored bonds.
    for step in range(3):
        valid = paths[:, step] >= 0
        hydrogen_bond[np.arange(num_residues)[valid], paths[valid, step]] = False

    def contacts(acceptor: NDArray, donor: NDArray) -> NDArray:
        return hydrogen_bond[np.maximum(acceptor, 0), np.maximum(donor, 0)] & (acceptor >= 0) & (donor >= 0)

    helix = np.zeros(num_residues, dtype=np.bool_)
    alpha_helix = np.zeros(num_residues, dtype=np.bool_)
    for turn_length in (4, 3, 5):
        turn = contacts(np.arange(num_residues), paths[:, turn_length])
        starts = turn & (following >= 0) & turn[np.maximum(following, 0)]
        if turn_length != 4:
            # Four-turn helices take priority over overlapping three/five turns.
            starts &= ~alpha_helix[np.maximum(paths[:, 1], 0)] & ~alpha_helix[np.maximum(paths[:, 2], 0)]
        selected = paths[starts, 1 : turn_length + 1]
        helix[selected[selected >= 0]] = True
        if turn_length == 4:
            alpha_helix = helix.copy()

    row, column = np.arange(num_residues)[:, None], np.arange(num_residues)[None, :]
    previous_row, previous_column = previous[:, None], previous[None, :]
    following_row, following_column = following[:, None], following[None, :]
    parallel = (contacts(previous_row, column) & contacts(column, following_row)) | (
        contacts(previous_column, row) & contacts(row, following_column)
    )
    antiparallel = (contacts(row, column) & contacts(column, row)) | (
        contacts(previous_row, following_column) & contacts(previous_column, following_row)
    )
    strand = (parallel | antiparallel).any(axis=-1)
    return np.where(helix, "H", np.where(strand, "E", "L"))


def secondary_structure_fractions(assignments: Mapping[Any, str] | Sequence[str]) -> dict[str, float]:
    r"""
    Return loop, helix, and sheet fractions for ``L/H/E`` assignments.
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
    Return shared per-class precision, recall, Jaccard intersection over union, and overall accuracy.

    ``None`` and integer zero conditioning entries are ignored. Integer
    conditioning labels one, two, and three mean ``L``, ``H``, and ``E``.
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
    num_true_positives_total = 0
    for label, name in zip(labels, names, strict=True):
        num_true_positives = int(np.sum((predicted_array == label) & (conditioned_array == label)))
        num_false_positives = int(np.sum((predicted_array == label) & (conditioned_array != label) & conditioned_mask))
        num_false_negatives = int(np.sum((predicted_array != label) & (conditioned_array == label)))
        metrics[f"precision_{name}"] = (
            num_true_positives / (num_true_positives + num_false_positives)
            if num_true_positives + num_false_positives
            else float("nan")
        )
        metrics[f"recall_{name}"] = (
            num_true_positives / (num_true_positives + num_false_negatives)
            if num_true_positives + num_false_negatives
            else float("nan")
        )
        metrics[f"jaccard_{name}"] = (
            num_true_positives / (num_true_positives + num_false_positives + num_false_negatives)
            if num_true_positives + num_false_positives + num_false_negatives
            else float("nan")
        )
        num_true_positives_total += num_true_positives
    metrics["accuracy_overall"] = (
        num_true_positives_total / int(np.sum(conditioned_mask)) if np.any(conditioned_mask) else float("nan")
    )
    return metrics


def binding_site_adherence(
    binding_site_coordinates: NDArray,
    design_coordinates: NDArray,
    *,
    distance_threshold: float = 8.0,
) -> float:
    r"""
    Return the fraction of binding-site centers within a strict distance of a design center.

    This is the shared ``bindsite_under_<threshold>rmsd`` operation; despite
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
    Return binding-site coverage at each requested strict distance cutoff.

    An empty design covers no binding sites; an empty binding-site observation
    domain has an undefined fraction (NaN).
    """

    distances = np.linalg.norm(
        np.asarray(binding_site_coordinates)[:, None] - np.asarray(design_coordinates)[None, :],
        axis=-1,
    )
    minimum_distance = np.min(distances, axis=1, initial=np.inf)
    return {
        threshold: float(np.mean(minimum_distance < threshold)) if len(minimum_distance) else float("nan")
        for threshold in thresholds
    }


def lddt_from_distances(
    predicted_distance: torch.Tensor,
    reference_distance: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    dim: int | tuple[int, ...] = (-2, -1),
) -> torch.Tensor:
    r"""Return hard-threshold lDDT over the supplied pair observation domain.

    Distances and the Boolean pair mask broadcast, including rectangular query
    and reference axes. Callers construct neighborhoods and exclude pairs that
    refer to the same atom before this reduction. Each selected pair contributes
    the mean of strict 0.5, 1, 2 and 4 angstrom distance-error comparisons. Empty
    domains report NaN; a caller may replace that undefined mean according to
    its scientific use of the score.
    """

    distance_error = (predicted_distance - reference_distance).abs()
    score = 0.25 * (
        (distance_error < 0.5).to(distance_error)
        + (distance_error < 1.0).to(distance_error)
        + (distance_error < 2.0).to(distance_error)
        + (distance_error < 4.0).to(distance_error)
    )
    selected = pair_mask.bool().expand_as(score)
    count = selected.sum(dim=dim)
    value = score.masked_fill(~selected, 0).sum(dim=dim) / count.clamp_min(1)
    return torch.where(count > 0, value, value.new_full((), float("nan")))


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
    the shared subset-RMSD operation used for design-, target-, and
    target-aligned RMSDs.
    """

    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    num_points = reference.shape[0]
    if align_mask is None:
        align_mask = np.ones(num_points, dtype=np.bool_)
    if measure_mask is None:
        measure_mask = align_mask
    if weights is None:
        weights = np.ones(num_points, dtype=np.float64)
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
    Return the shared hard-threshold lDDT score.

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
    score = lddt_from_distances(
        torch.as_tensor(predicted_distance, dtype=torch.float64),
        torch.as_tensor(reference_distance, dtype=torch.float64),
        torch.as_tensor(scored),
    )
    return float(torch.where(torch.as_tensor(scored).any(), score, 1.0))


def structure_tm_scores(first: NDArray, second: NDArray) -> tuple[float, float]:
    r"""
    Return shared RMSD-aligned and structural-homology-aligned TM scores.
    """

    first_atoms = _ca_atom_array(first)
    second_atoms = _ca_atom_array(second)
    homology_score = 0.0
    try:
        fixed: NDArray[np.integer[Any]]
        mobile: NDArray[np.integer[Any]]
        transform, fixed, mobile = struc.superimpose_structural_homologs(
            first_atoms,
            second_atoms,
            max_iterations=25,
        )
        homology_score = float(
            struc.tm_score(
                first_atoms,
                transform.apply(second_atoms),
                np.asarray(fixed, dtype=np.int64),
                np.asarray(mobile, dtype=np.int64),
            )
        )
    except ValueError:
        pass

    rmsd_aligned_score = 0.0
    if len(first) == len(second):
        aligned_coordinates = _aligned_coordinates(np.asarray(first), np.asarray(second))
        aligned_atoms = _ca_atom_array(aligned_coordinates)
        try:
            _, fixed, mobile = struc.superimpose_structural_homologs(
                aligned_atoms,
                second_atoms,
                max_iterations=25,
            )
            rmsd_aligned_score = float(
                struc.tm_score(
                    aligned_atoms,
                    second_atoms,
                    np.asarray(fixed, dtype=np.int64),
                    np.asarray(mobile, dtype=np.int64),
                )
            )
        except ValueError:
            pass
    return rmsd_aligned_score, homology_score


def _aligned_coordinates(
    reference: NDArray,
    predicted: NDArray,
    weights: NDArray | None = None,
) -> NDArray:
    from ..modules.geometry import weighted_rigid_align

    if weights is None:
        weights = np.ones(len(reference), dtype=np.float64)
    aligned = weighted_rigid_align(
        source=torch.as_tensor(reference, dtype=torch.float64),
        target=torch.as_tensor(predicted, dtype=torch.float64),
        weight=torch.as_tensor(weights, dtype=torch.float64),
    )
    return aligned.numpy()


def _ca_atom_array(coordinates: NDArray) -> struc.AtomArray:
    atom_array: struc.AtomArray = struc.AtomArray(len(coordinates))
    atom_array.coord = np.asarray(coordinates)
    atom_array.element = np.full(len(coordinates), "C")
    atom_array.atom_name = np.full(len(coordinates), "CA")
    atom_array.res_name = np.full(len(coordinates), "ALA")
    atom_array.chain_id = np.full(len(coordinates), "A")
    atom_array.res_id = np.arange(1, len(coordinates) + 1)
    return atom_array
