from __future__ import annotations

import pytest
import torch

from ofoldx.modules.head import DistogramHead, PairSymmetrizedDistogramHead


@pytest.mark.parametrize("head_type", [DistogramHead, PairSymmetrizedDistogramHead])
def test_head_symmetric(head_type: type[DistogramHead] | type[PairSymmetrizedDistogramHead]) -> None:
    head = head_type(pair_size=4, num_bins=7)
    logits = head(torch.randn(2, 3, 3, 4))

    # Both projection orders symmetrise across the two token axes; channel-axis Linear preserves it.
    assert logits.shape == (2, 3, 3, 7)
    torch.testing.assert_close(logits, logits.transpose(-2, -3))
