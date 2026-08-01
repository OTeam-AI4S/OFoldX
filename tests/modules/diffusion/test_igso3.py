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

import numpy as np
import pytest
import torch

from ofoldx.modules.diffusion.igso3 import (
    calculate_igso3,
    igso3_expansion,
    so3_exp_map,
    so3_log_map,
    so3_project_to_rotation,
)


@pytest.fixture(params=("identity", "repeated", "near_rotation", "reflection", "rank_two"))
def projection_case(request: pytest.FixtureRequest) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(3, dtype=torch.float64)
    left = so3_exp_map(torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64))
    right = so3_exp_map(torch.tensor([-0.4, 0.1, 0.2], dtype=torch.float64))
    if request.param == "identity":
        return identity, identity
    if request.param == "near_rotation":
        symmetric = torch.tensor([[1.03, 0.02, 0.01], [0.02, 0.97, -0.01], [0.01, -0.01, 1.01]], dtype=torch.float64)
        return left @ symmetric, left
    spectrum = {"repeated": (2.0, 2.0, 0.7), "reflection": (2.0, 1.0, -0.5), "rank_two": (2.0, 1.0, 0.0)}
    matrix = left @ torch.diag(torch.tensor(spectrum[request.param], dtype=torch.float64)) @ right.T
    return matrix, left @ right.T


class TestSo3:
    def test_round_trip(self) -> None:
        vector = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float64, requires_grad=True)

        rotation = so3_exp_map(vector)
        recovered = so3_log_map(rotation)
        recovered.sum().backward()

        torch.testing.assert_close(recovered, vector, atol=1e-10, rtol=1e-10)
        assert rotation.dtype == vector.dtype
        assert vector.grad is not None

    def test_identity(self) -> None:
        identity = torch.eye(3, dtype=torch.float64).requires_grad_()

        vector = so3_log_map(identity)
        vector.sum().backward()

        torch.testing.assert_close(vector, torch.zeros(3, dtype=torch.float64))
        assert identity.grad is not None
        assert torch.isfinite(identity.grad).all()

    def test_projection(self, projection_case: tuple[torch.Tensor, torch.Tensor]) -> None:
        matrix, expected = projection_case

        rotation = so3_project_to_rotation(matrix)

        torch.testing.assert_close(rotation, expected, atol=1e-10, rtol=1e-10)

    def test_projection_gradients(self, projection_case: tuple[torch.Tensor, torch.Tensor]) -> None:
        matrix = projection_case[0].requires_grad_()

        assert torch.autograd.gradcheck(so3_project_to_rotation, (matrix,))
        assert torch.autograd.gradgradcheck(so3_project_to_rotation, (matrix,))

    def test_projection_transforms(self, projection_case: tuple[torch.Tensor, torch.Tensor]) -> None:
        matrix = projection_case[0]
        matrices = torch.stack((matrix, matrix.T))
        tangent = torch.arange(18, dtype=matrix.dtype).reshape(2, 3, 3) / 17
        step = 1e-5
        expected = (
            so3_project_to_rotation(matrices + step * tangent) - so3_project_to_rotation(matrices - step * tangent)
        ) / (2 * step)

        rotations, jvp = torch.func.jvp(so3_project_to_rotation, (matrices,), (tangent,))
        _, pullback = torch.func.vjp(so3_project_to_rotation, matrices)
        vjp = pullback(tangent)[0]

        torch.testing.assert_close(torch.vmap(so3_project_to_rotation)(matrices), rotations)
        torch.testing.assert_close(jvp, expected, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(vjp, expected, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(
            torch.func.jacfwd(so3_project_to_rotation)(matrix), torch.func.jacrev(so3_project_to_rotation)(matrix)
        )


def test_calculate_igso3_table() -> None:
    table = calculate_igso3(num_sigma=2, num_omega=8, min_sigma=0.1, max_sigma=1.0, num_terms=64)

    assert set(table) == {"cdf", "score_norm", "exp_score_norms", "discrete_omega", "discrete_sigma"}
    assert table["cdf"].shape == table["score_norm"].shape == (2, 8)
    assert table["exp_score_norms"].shape == table["discrete_sigma"].shape == (2,)
    assert table["discrete_omega"].shape == (8,)
    assert all(value.dtype == np.float64 for value in table.values())
    assert np.isfinite(table["score_norm"]).all()
    assert np.count_nonzero(table["cdf"]) > 0


class TestIgso3Expansion:
    def test_tensor(self) -> None:
        omega = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)

        density = igso3_expansion(omega, 0.5, num_terms=64)
        density.sum().backward()

        assert density.dtype == omega.dtype
        assert density.device == omega.device
        assert omega.grad is not None

    def test_identity(self) -> None:
        omega = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        degree = torch.arange(64, dtype=torch.float64)
        expected = ((2 * degree + 1).square() * torch.exp(-degree * (degree + 1) * 0.5 / 2)).sum()

        density = igso3_expansion(omega, 0.5, num_terms=64)
        density.backward()

        torch.testing.assert_close(density, expected)
        torch.testing.assert_close(omega.grad, torch.zeros_like(omega))
