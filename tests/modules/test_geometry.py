from __future__ import annotations

import torch

from ofoldx.modules.geometry import rigid_from_three_points


def test_rigid_from_three_points() -> None:
    rotation, translation = rigid_from_three_points(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 0.0]),
    )

    torch.testing.assert_close(rotation, torch.eye(3))
    torch.testing.assert_close(translation, torch.zeros(3))
