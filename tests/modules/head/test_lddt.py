from __future__ import annotations

import pytest
import torch

from ofoldx.modules.head.lddt import compute_per_atom_lddt, lddt_bin_labels


@pytest.fixture
def coords() -> torch.Tensor:
    # Six atoms spread within the 15 A inclusion radius of each other.
    torch.manual_seed(0)
    return torch.randn(1, 6, 3) * 3.0


def test_perfect_prediction_scores_one(coords: torch.Tensor) -> None:
    # pred == true => every kept pair clears all four thresholds, so the score equals the weight.
    per_atom_lddt, per_atom_weight = compute_per_atom_lddt(
        coords.unsqueeze(-3),
        coords,
        neighbor_mask=torch.ones(1, 6, dtype=torch.bool),
        is_nucleotide=torch.zeros(1, 6, dtype=torch.bool),
    )
    # All six atoms lie within 15 A of one another, so every off-diagonal pair is kept: weight = 5.
    torch.testing.assert_close(per_atom_weight, torch.full((1, 1, 6), 5.0))
    torch.testing.assert_close(per_atom_lddt, per_atom_weight.expand_as(per_atom_lddt))
    labels = lddt_bin_labels(per_atom_lddt, per_atom_weight, num_bins=50)
    assert torch.all(labels == 49)


def test_exact_score_for_known_geometry() -> None:
    # Collinear x-axis atoms; prediction shifts C from 3 to 3.5, giving pair errors AB=0, AC=BC=0.5.
    # Per-pair LDDT = (#thresholds in {0.5,1,2,4} strictly above the error): error 0.0 -> all four (1.00),
    # error 0.5 -> {1,2,4} (0.5<0.5 is false, 0.75). Summed over each atom's two neighbours (weight 2):
    #   A=1.00+0.75=1.75, B=1.00+0.75=1.75, C=0.75+0.75=1.50.
    true = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [3, 0, 0]]])
    pred = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [3.5, 0, 0]]]).unsqueeze(-3)
    per_atom_lddt, per_atom_weight = compute_per_atom_lddt(
        pred,
        true,
        neighbor_mask=torch.ones(1, 3, dtype=torch.bool),
        is_nucleotide=torch.zeros(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(per_atom_weight, torch.tensor([[[2.0, 2.0, 2.0]]]))
    torch.testing.assert_close(per_atom_lddt, torch.tensor([[[1.75, 1.75, 1.50]]]))

    # normalized = score/weight = [0.875, 0.875, 0.75]; bin = clamp(#boundaries below, 1, 50) - 1
    # over 51 evenly-spaced boundaries: 0.875 -> 43, 0.75 -> 37.
    labels = lddt_bin_labels(per_atom_lddt, per_atom_weight, num_bins=50)
    assert labels.flatten().tolist() == [43, 43, 37]


def test_inclusion_radius_uses_neighbor_nucleotide_flag() -> None:
    # D sits 20 A from A/B: outside the 15 A protein radius, inside the 30 A nucleotide radius. The radius
    # is keyed on the *neighbour* (column) atom, so D's nucleotide flag decides whether A/B count D.
    true = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [20, 0, 0]]])
    pred = true.clone().unsqueeze(-3)
    neighbor_mask = torch.ones(1, 3, dtype=torch.bool)

    # D not a nucleotide (15 A): A and B keep only each other, D keeps nobody.
    _, weight_protein = compute_per_atom_lddt(
        pred, true, neighbor_mask=neighbor_mask, is_nucleotide=torch.zeros(1, 3, dtype=torch.bool)
    )
    torch.testing.assert_close(weight_protein, torch.tensor([[[1.0, 1.0, 0.0]]]))

    # D a nucleotide (column radius 30 A): A and B now also keep D; D still keeps nobody (it uses the
    # protein radius on its own neighbours, which are 19-20 A away).
    _, weight_nuc = compute_per_atom_lddt(
        pred, true, neighbor_mask=neighbor_mask, is_nucleotide=torch.tensor([[False, False, True]])
    )
    torch.testing.assert_close(weight_nuc, torch.tensor([[[2.0, 2.0, 0.0]]]))


def test_excludes_non_neighbor_atoms(coords: torch.Tensor) -> None:
    is_nucleotide = torch.zeros(1, 6, dtype=torch.bool)
    full_mask = torch.ones(1, 6, dtype=torch.bool)
    # Drop the last atom from the eligible-neighbour set; it must not influence any score.
    restricted_mask = torch.tensor([[True, True, True, True, True, False]])

    full_lddt, _ = compute_per_atom_lddt(
        coords.unsqueeze(-3), coords, neighbor_mask=full_mask, is_nucleotide=is_nucleotide
    )
    restricted_lddt, _ = compute_per_atom_lddt(
        coords.unsqueeze(-3), coords, neighbor_mask=restricted_mask, is_nucleotide=is_nucleotide
    )

    # The dropped atom never counted itself as a neighbour, so its own score is unchanged.
    assert torch.equal(restricted_lddt[..., 5], full_lddt[..., 5])
