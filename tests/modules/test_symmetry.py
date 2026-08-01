from __future__ import annotations

import pytest
import torch

from ofoldx.modules.symmetry import match_symmetric_truth, point_group_rotations


@pytest.mark.parametrize(
    ("kind", "order", "size"),
    (
        ("cyclic", 3, 3),
        ("dihedral", 3, 6),
        ("t3", None, 4),
        ("tetrahedral", None, 12),
        ("octahedral", None, 24),
        ("icosahedral", None, 60),
    ),
)
def test_point_group_rotations(kind: str, order: int | None, size: int) -> None:
    rotations = torch.stack(point_group_rotations(kind, order, dtype=torch.float64, device="cpu"))
    assert rotations.shape == (size, 3, 3)
    assert rotations.dtype == torch.float64
    assert rotations.device.type == "cpu"
    torch.testing.assert_close(rotations[0], torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        rotations @ rotations.transpose(-1, -2),
        torch.eye(3, dtype=torch.float64).expand(size, -1, -1),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(size, dtype=torch.float64), atol=1e-12, rtol=0)


class TestMatchSymmetricTruth:
    def test_assignment(self) -> None:
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

    def test_budget(self) -> None:
        num_chains = 21
        target = torch.zeros(1, num_chains, 3)
        target[0, :, 0] = torch.arange(num_chains)
        chain_index = torch.arange(num_chains)

        matched, mask = match_symmetric_truth(
            target.unsqueeze(1),
            target,
            torch.ones(1, num_chains, dtype=torch.bool),
            chain_index.unsqueeze(0),
            torch.stack((chain_index, chain_index + 1), dim=-1).unsqueeze(0),
            torch.zeros(1, num_chains, dtype=torch.long),
            torch.empty(1, 0, 0, dtype=torch.long),
            torch.empty(1, 0, 0, 0, dtype=torch.long),
            max_chain_assignments=2,
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(matched, target.unsqueeze(1))
        assert torch.equal(mask, torch.ones(1, 1, num_chains, dtype=torch.bool))
