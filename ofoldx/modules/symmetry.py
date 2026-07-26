r"""
Finite point-group and discrete truth-symmetry operations.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


def point_group_rotations(kind: str, order: int | None = None) -> tuple[Tensor, ...]:
    r"""
    Return the canonically ordered rotation matrices of a finite three-dimensional point group.
    """

    if kind in ("cyclic", "dihedral"):
        angles = np.arange(order, dtype=np.float64) * (2 * np.pi / order)
        rotations = [
            np.array(
                (
                    (np.cos(angle), -np.sin(angle), 0.0),
                    (np.sin(angle), np.cos(angle), 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
            for angle in angles
        ]
        if kind == "dihedral":
            flip = np.diag((1.0, -1.0, -1.0))
            rotations = [matrix for rotation in rotations for matrix in (rotation, flip @ rotation)]
        matrices = np.stack(rotations)
    else:
        group = {"tetrahedral": "T", "octahedral": "O", "icosahedral": "I"}[kind]
        matrices = Rotation.create_group(group).as_matrix()
        if kind == "icosahedral":
            orientation = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
            matrices = np.einsum("ij,njk,lk->nil", orientation, matrices, orientation)
        rounded = np.round(matrices, decimals=12)
        canonical_order = sorted(range(len(matrices)), key=lambda index: tuple(rounded[index].reshape(-1)))
        matrices = matrices[np.asarray(canonical_order)]
        permutation = {
            "tetrahedral": (10, 7, 11, 6, 5, 9, 4, 8, 3, 2, 1, 0),
            "octahedral": (19, 18, 23, 21, 14, 15, 11, 10, 16, 22, 20, 17, 3, 2, 13, 7, 5, 8, 12, 6, 9, 1, 0, 4),
            "icosahedral": (
                59,
                58,
                55,
                53,
                57,
                51,
                56,
                50,
                54,
                52,
                47,
                45,
                49,
                43,
                48,
                42,
                46,
                44,
                41,
                35,
                39,
                37,
                38,
                36,
                40,
                34,
                31,
                29,
                33,
                26,
                28,
                30,
                32,
                27,
                0,
                1,
                9,
                3,
                7,
                5,
                6,
                4,
                8,
                2,
                17,
                11,
                15,
                13,
                14,
                12,
                16,
                10,
                23,
                21,
                25,
                19,
                24,
                18,
                22,
                20,
            ),
        }[kind]
        matrices = matrices[np.asarray(permutation)]
    return tuple(torch.as_tensor(matrix, dtype=torch.float32) for matrix in matrices)


@torch.no_grad()
def match_symmetric_truth(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    target_atom_index: Tensor,
    target_chain_span: Tensor,
    target_chain_equivalence_index: Tensor,
    target_symmetry_atom_index: Tensor,
    target_symmetry_permutation: Tensor,
    *,
    max_chain_assignments: int | None = 100,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Select chain and local-atom truth assignments by the released staged lDDT rule.

    Chain assignments are injective within each equivalence class. Complete
    enumeration returns the global optimum of the chain stage; a finite budget
    keeps identity and samples the remaining assignment ordinals uniformly
    without replacement. Local atom groups are then optimized greedily in their
    carried order.
    """

    if max_chain_assignments is not None:
        max_chain_assignments = operator.index(max_chain_assignments)
        if max_chain_assignments < 1:
            raise ValueError("max_chain_assignments must be positive or None")

    device = predicted_coordinates.device
    target_coordinates = target_coordinates.to(
        device=device,
        dtype=predicted_coordinates.dtype,
    )
    target_mask = target_mask.to(device=device, dtype=torch.bool)
    target_atom_index = target_atom_index.to(device=device, dtype=torch.long)
    target_chain_span = target_chain_span.to(device=device, dtype=torch.long)
    target_chain_equivalence_index = target_chain_equivalence_index.to(
        device=device,
        dtype=torch.long,
    )
    target_symmetry_atom_index = target_symmetry_atom_index.to(
        device=device,
        dtype=torch.long,
    )
    target_symmetry_permutation = target_symmetry_permutation.to(
        device=device,
        dtype=torch.long,
    )

    matched_coordinates = torch.empty_like(predicted_coordinates)
    matched_mask = torch.empty(
        predicted_coordinates.shape[:-1],
        device=device,
        dtype=torch.bool,
    )
    batch_size, num_samples = predicted_coordinates.shape[:2]
    for batch_index in range(batch_size):
        spans = target_chain_span[batch_index]
        valid_chain = spans[:, 1] > spans[:, 0]
        spans = spans[valid_chain]
        equivalence = target_chain_equivalence_index[batch_index][valid_chain]
        atom_index = target_atom_index[batch_index]
        assignment_groups = _chain_assignment_groups(
            spans,
            equivalence,
            atom_index,
        )
        assignment_counts = [math.perm(len(destinations), len(sources)) for sources, destinations in assignment_groups]
        total_assignments = math.prod(assignment_counts)
        identity_ordinal = _chain_assignment_ordinal(
            assignment_groups,
            assignment_counts,
        )
        for sample_index in range(num_samples):
            assignment_ordinals = _assignment_ordinals(
                total_assignments,
                identity_ordinal,
                max_chain_assignments,
                device=device,
                generator=generator,
            )
            predicted = predicted_coordinates[batch_index, sample_index]
            best_coordinates: Tensor | None = None
            best_mask: Tensor | None = None
            best_score: Tensor | None = None
            for ordinal in assignment_ordinals:
                assignment = _decode_chain_assignment(
                    ordinal,
                    assignment_groups,
                    assignment_counts,
                )
                candidate_atom_index = _assigned_atom_index(
                    atom_index,
                    spans,
                    assignment,
                )
                valid_atom = candidate_atom_index >= 0
                gather_index = candidate_atom_index.clamp_min(0)
                candidate_coordinates = target_coordinates[batch_index][gather_index]
                candidate_mask = target_mask[batch_index][gather_index] & valid_atom
                score = _global_lddt_score(
                    predicted,
                    candidate_coordinates,
                    candidate_mask,
                )
                if best_score is None or score > best_score:
                    best_coordinates = candidate_coordinates
                    best_mask = candidate_mask
                    best_score = score

            current_coordinates = best_coordinates
            current_mask = best_mask
            assert current_coordinates is not None and current_mask is not None
            for group_index in range(target_symmetry_atom_index.shape[1]):
                atom_group = target_symmetry_atom_index[
                    batch_index,
                    group_index,
                ]
                atom_group = atom_group[atom_group >= 0]
                if atom_group.numel() == 0:
                    continue
                group_score = _local_lddt_score(
                    predicted,
                    current_coordinates,
                    current_mask,
                    atom_group,
                )
                group_coordinates = current_coordinates
                group_mask = current_mask
                for permutation in target_symmetry_permutation[
                    batch_index,
                    group_index,
                ]:
                    permutation = permutation[permutation >= 0]
                    if permutation.numel() == 0:
                        continue
                    candidate_coordinates = current_coordinates.clone()
                    candidate_mask = current_mask.clone()
                    candidate_coordinates[atom_group] = current_coordinates[permutation]
                    candidate_mask[atom_group] = current_mask[permutation]
                    score = _local_lddt_score(
                        predicted,
                        candidate_coordinates,
                        candidate_mask,
                        atom_group,
                    )
                    if score > group_score:
                        group_coordinates = candidate_coordinates
                        group_mask = candidate_mask
                        group_score = score
                current_coordinates = group_coordinates
                current_mask = group_mask

            matched_coordinates[batch_index, sample_index] = current_coordinates
            matched_mask[batch_index, sample_index] = current_mask
    return matched_coordinates, matched_mask


def _chain_assignment_groups(
    chain_span: Tensor,
    chain_equivalence_index: Tensor,
    atom_index: Tensor,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    represented = []
    for chain_index, (start, end) in enumerate(chain_span.tolist()):
        if bool(((atom_index >= start) & (atom_index < end)).any()):
            represented.append(chain_index)

    groups = []
    for equivalence_index in torch.unique(chain_equivalence_index).tolist():
        destinations = tuple(
            index for index, value in enumerate(chain_equivalence_index.tolist()) if value == equivalence_index
        )
        sources = tuple(index for index in represented if index in destinations)
        if sources:
            groups.append((sources, destinations))
    return groups


def _chain_assignment_ordinal(
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]],
    counts: list[int],
) -> int:
    ordinal = 0
    for (sources, destinations), count in zip(groups, counts):
        available = list(destinations)
        group_ordinal = 0
        for position, source in enumerate(sources):
            block_size = math.perm(
                len(destinations) - position - 1,
                len(sources) - position - 1,
            )
            digit = available.index(source)
            group_ordinal += digit * block_size
            available.pop(digit)
        ordinal = ordinal * count + group_ordinal
    return ordinal


def _assignment_ordinals(
    total: int,
    identity: int,
    budget: int | None,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[int, ...]:
    if budget is None or total <= budget:
        return (identity, *(ordinal for ordinal in range(total) if ordinal != identity))

    population = total - 1
    sample_count = budget - 1
    remaining = population
    moved: dict[int, int] = {}
    sampled = []
    for _ in range(sample_count):
        draw = int(
            torch.randint(
                remaining,
                (),
                device=device,
                generator=generator,
            ).item()
        )
        value = moved.get(draw, draw)
        remaining -= 1
        moved[draw] = moved.get(remaining, remaining)
        sampled.append(value if value < identity else value + 1)
    return (identity, *sampled)


def _decode_chain_assignment(
    ordinal: int,
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]],
    counts: list[int],
) -> dict[int, int]:
    group_ordinals = [0] * len(groups)
    for index in range(len(groups) - 1, -1, -1):
        ordinal, group_ordinals[index] = divmod(ordinal, counts[index])

    assignment: dict[int, int] = {}
    for group_ordinal, (sources, destinations) in zip(group_ordinals, groups):
        available = list(destinations)
        for position, source in enumerate(sources):
            block_size = math.perm(
                len(destinations) - position - 1,
                len(sources) - position - 1,
            )
            digit, group_ordinal = divmod(group_ordinal, block_size)
            assignment[source] = available.pop(digit)
    return assignment


def _assigned_atom_index(
    atom_index: Tensor,
    chain_span: Tensor,
    assignment: dict[int, int],
) -> Tensor:
    assigned = atom_index.clone()
    for source, destination in assignment.items():
        source_start, source_end = chain_span[source]
        destination_start = chain_span[destination, 0]
        included = (atom_index >= source_start) & (atom_index < source_end)
        assigned[included] = destination_start + atom_index[included] - source_start
    return assigned


def _global_lddt_score(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
) -> Tensor:
    predicted_distance = torch.cdist(
        predicted_coordinates,
        predicted_coordinates,
    )
    target_distance = torch.cdist(target_coordinates, target_coordinates)
    num_atoms = target_coordinates.shape[0]
    pair_mask = target_mask.unsqueeze(-1) & target_mask.unsqueeze(-2)
    pair_mask = pair_mask & ~torch.eye(
        num_atoms,
        device=target_coordinates.device,
        dtype=torch.bool,
    )
    pair_mask = pair_mask & (target_distance < 15.0)
    return _lddt_score(predicted_distance, target_distance, pair_mask)


def _local_lddt_score(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    atom_group: Tensor,
) -> Tensor:
    predicted_distance = torch.cdist(
        predicted_coordinates,
        predicted_coordinates[atom_group],
    )
    target_distance = torch.cdist(
        target_coordinates,
        target_coordinates[atom_group],
    )
    row_index = torch.arange(
        target_coordinates.shape[0],
        device=target_coordinates.device,
    ).unsqueeze(-1)
    pair_mask = target_mask.unsqueeze(-1) & target_mask[atom_group].unsqueeze(0)
    pair_mask = pair_mask & (row_index != atom_group.unsqueeze(0))
    pair_mask = pair_mask & (target_distance < 15.0)
    return _lddt_score(predicted_distance, target_distance, pair_mask)


def _lddt_score(
    predicted_distance: Tensor,
    target_distance: Tensor,
    pair_mask: Tensor,
) -> Tensor:
    distance_error = (predicted_distance - target_distance).abs()
    score = 0.25 * (
        (distance_error < 0.5).to(distance_error)
        + (distance_error < 1.0).to(distance_error)
        + (distance_error < 2.0).to(distance_error)
        + (distance_error < 4.0).to(distance_error)
    )
    weight = pair_mask.to(score)
    return (score * weight).sum() / weight.sum().clamp_min(1)
