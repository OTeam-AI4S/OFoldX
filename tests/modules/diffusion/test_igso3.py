from __future__ import annotations

import numpy as np
import torch

from ofoldx.modules.diffusion.igso3 import calculate_igso3, so3_exp_map


def test_igso3_math() -> None:
    rotation = so3_exp_map(torch.tensor([[0.0, 0.0, torch.pi / 2]]))[0]
    torch.testing.assert_close(
        rotation,
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
        rtol=0,
    )

    table = calculate_igso3(num_sigma=2, num_omega=4, min_sigma=0.1, max_sigma=1.0, num_terms=32)
    assert table["cdf"].shape == table["score_norm"].shape == (2, 4)
    assert np.isfinite(table["exp_score_norms"]).all()
