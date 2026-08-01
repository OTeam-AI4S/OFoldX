from __future__ import annotations

import numpy as np
import torch

from ofoldx.modules.diffusion.igso3 import (
    calculate_igso3,
    igso3_expansion,
    so3_exp_map,
    so3_log_map,
    so3_project_to_rotation,
)


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

    def test_projection(self) -> None:
        matrix = so3_exp_map(torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float64)) + 1e-5

        rotation = so3_project_to_rotation(matrix)

        torch.testing.assert_close(
            rotation @ rotation.transpose(-1, -2),
            torch.eye(3, dtype=torch.float64).unsqueeze(0),
            atol=1e-10,
            rtol=1e-10,
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
