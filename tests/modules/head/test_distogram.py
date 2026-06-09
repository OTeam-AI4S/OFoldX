from __future__ import annotations

import pytest
import torch

from ofoldx.modules.head import DistogramHead
from ofoldx.modules.head.distogram import masked_cross_entropy

F = torch.nn.functional


@pytest.mark.parametrize("symmetrize_before_projection", [False, True])
def test_head_symmetric(symmetrize_before_projection: bool) -> None:
    head = DistogramHead(pair_size=4, num_bins=7, symmetrize_before_projection=symmetrize_before_projection)
    logits = head(torch.randn(2, 3, 3, 4))

    # Both projection orders symmetrise across the two token axes; channel-axis Linear preserves it.
    assert logits.shape == (2, 3, 3, 7)
    torch.testing.assert_close(logits, logits.transpose(-2, -3))


@pytest.mark.parametrize("shape", [(2, 3, 5), (2, 3, 3, 7)])
def test_masked_cross_entropy(shape: tuple[int, ...]) -> None:
    logits = torch.randn(*shape, requires_grad=True)
    labels = torch.randint(shape[-1], shape[:-1])
    mask = torch.rand(*shape[:-1]) > 0.25

    value = masked_cross_entropy(logits, labels, mask)

    losses = F.cross_entropy(logits.reshape(-1, shape[-1]), labels.reshape(-1), reduction="none")
    mask_flat = mask.reshape(-1).to(dtype=logits.dtype)
    expected = (losses * mask_flat).sum() / mask_flat.sum().clamp_min(1)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert logits.grad is not None
