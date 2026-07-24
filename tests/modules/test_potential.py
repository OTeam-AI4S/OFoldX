from __future__ import annotations

import torch

from ofoldx.modules.potential import (
    interval_distance_penalty,
    smooth_coordination_potential,
    substrate_interaction_potential,
    upper_bounded_radius_of_gyration,
)


def test_geometric_potentials() -> None:
    distance = torch.tensor(3.0, requires_grad=True)
    potential = smooth_coordination_potential(distance, center=3.0, scale=1.0)
    potential = potential - interval_distance_penalty(distance, lower=2.0, upper=4.0)
    potential = potential + upper_bounded_radius_of_gyration(torch.tensor([[0.0, 0.0], [2.0, 0.0]]), 0.5)
    potential = potential + substrate_interaction_potential(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[2.0, 0.0]]),
        center=2.0,
        scale=1.0,
        attraction_strength=1.0,
        repulsion_slope=1.0,
        repulsion_cutoff=1.0,
    )
    potential.backward()

    assert torch.isfinite(potential)
    assert distance.grad is not None
