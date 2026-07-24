from __future__ import annotations

import torch

from ofoldx.modules.dropout import RowwiseDropout, StructuredDropout


class TestStructuredDropout:
    def test_broadcasts_one_expectation_preserving_mask(self) -> None:
        hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)
        dropout = StructuredDropout(0.5, broadcast_dim=2)

        torch.manual_seed(0)
        ratio = dropout.train()(hidden) / hidden

        torch.testing.assert_close(ratio[:, :, 0], ratio[:, :, 1])
        torch.testing.assert_close(ratio[:, :, 1], ratio[:, :, 2])
        torch.testing.assert_close(ratio[:, :, 2], ratio[:, :, 3])
        torch.testing.assert_close(dropout.eval()(hidden), hidden)


class TestRowwiseDropout:
    def test_shares_one_mask_across_rows(self) -> None:
        hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)
        dropout = RowwiseDropout(0.5)

        torch.manual_seed(0)
        ratio = dropout.train()(hidden) / hidden

        torch.testing.assert_close(ratio[:, 0], ratio[:, 1])
        torch.testing.assert_close(ratio[:, 1], ratio[:, 2])
        torch.testing.assert_close(dropout.eval()(hidden), hidden)
