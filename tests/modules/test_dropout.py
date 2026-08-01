from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.dropout import StructuredDropout


class TestStructuredDropout:
    @pytest.mark.parametrize(
        ("broadcast_dims", "mask_shape"),
        [
            (-3, (2, 1, 4, 5)),
            ((-2, -1), (2, 3, 1, 1)),
            ((-3, -1), (2, 1, 4, 1)),
        ],
    )
    def test_training(
        self,
        broadcast_dims: int | tuple[int, ...],
        mask_shape: tuple[int, ...],
    ) -> None:
        hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)
        dropout = StructuredDropout(0.5, broadcast_dims=broadcast_dims)

        torch.manual_seed(0)
        actual = dropout.train()(hidden)
        torch.manual_seed(0)
        expected = hidden * F.dropout(hidden.new_ones(mask_shape), p=0.5, training=True)

        torch.testing.assert_close(actual, expected)

    def test_eval(self) -> None:
        hidden = torch.randn(2, 3, 4, 5)

        torch.testing.assert_close(StructuredDropout(0.5, broadcast_dims=-3).eval()(hidden), hidden)
