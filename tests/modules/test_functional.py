from __future__ import annotations

import torch

from ofoldx.modules.functional import (
    pair_mask_from_msa_mask,
    pair_mask_from_token_mask,
)


def test_pair_mask_from_token_and_msa() -> None:
    token_mask = torch.tensor([[True, False, True]])
    expected = torch.tensor([[[True, False, True], [False, False, False], [True, False, True]]])
    assert torch.equal(pair_mask_from_token_mask(token_mask), expected)

    # an MSA mask collapses its rows (logical-or) before the outer mask, so two rows == token [F, T, T]
    msa_mask = torch.tensor([[[False, True, False], [False, False, True]]])
    pair_mask = pair_mask_from_msa_mask(msa_mask)
    assert pair_mask.shape == (1, 3, 3)
    assert torch.equal(pair_mask, pair_mask_from_token_mask(torch.tensor([[False, True, True]])))
