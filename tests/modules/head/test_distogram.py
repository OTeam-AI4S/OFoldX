from __future__ import annotations

import pytest
import torch

from ofoldx.modules.head import DistogramHead


@pytest.mark.parametrize("symmetrize_before_projection", [False, True])
def test_head_symmetric(symmetrize_before_projection: bool) -> None:
    head = DistogramHead(pair_size=4, num_bins=7, symmetrize_before_projection=symmetrize_before_projection)
    logits = head(torch.randn(2, 3, 3, 4))

    # Both projection orders symmetrise across the two token axes; channel-axis Linear preserves it.
    assert logits.shape == (2, 3, 3, 7)
    torch.testing.assert_close(logits, logits.transpose(-2, -3))
