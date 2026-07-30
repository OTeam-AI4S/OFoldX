from __future__ import annotations

import torch

from ofoldx.modules.symmetry import match_symmetric_truth, point_group_rotations


def test_point_group_rotations() -> None:
    rotations = torch.stack(point_group_rotations("dihedral", 3))

    assert rotations.shape == (6, 3, 3)
    torch.testing.assert_close(
        rotations @ rotations.transpose(-1, -2),
        torch.eye(3).expand(6, -1, -1),
        atol=1e-6,
        rtol=0,
    )


def test_match_symmetric_truth_chain_and_atom_assignments() -> None:
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 6.0, 0.0],
            ]
        ]
    )
    prediction = target[:, [3, 2, 0, 1]].unsqueeze(1)

    matched, mask = match_symmetric_truth(
        prediction,
        target,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[[0, 2], [2, 4]]]),
        torch.tensor([[0, 0]]),
        torch.tensor([[[0, 1]]]),
        torch.tensor([[[[1, 0]]]]),
        max_chain_assignments=None,
    )

    torch.testing.assert_close(matched, prediction)
    assert torch.equal(mask, torch.ones(1, 1, 4, dtype=torch.bool))
