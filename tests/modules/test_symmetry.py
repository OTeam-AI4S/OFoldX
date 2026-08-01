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

from __future__ import annotations

from typing import Literal

import pytest
import torch

from ofoldx.modules.symmetry import point_group_rotations


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
def test_point_group_rotations(
    kind: Literal["cyclic", "dihedral", "t3", "tetrahedral", "octahedral", "icosahedral"],
    order: int | None,
    size: int,
) -> None:
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
