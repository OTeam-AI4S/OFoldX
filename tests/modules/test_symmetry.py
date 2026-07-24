from __future__ import annotations

import torch

from ofoldx.modules.symmetry import point_group_rotations


def test_point_group_rotations() -> None:
    rotations = torch.stack(point_group_rotations("dihedral", 3))

    assert rotations.shape == (6, 3, 3)
    torch.testing.assert_close(
        rotations @ rotations.transpose(-1, -2),
        torch.eye(3).expand(6, -1, -1),
        atol=1e-6,
        rtol=0,
    )
