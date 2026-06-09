from __future__ import annotations

import pytest
import torch

from ofoldx.modules.functional import (
    checkpoint_blocks,
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


@pytest.mark.parametrize("blocks_per_checkpoint", [None, 1, 2])
def test_checkpoint_blocks_execute_sequentially(blocks_per_checkpoint: int | None) -> None:
    seed = torch.tensor(1.0, requires_grad=True)

    (output,) = checkpoint_blocks(
        [lambda x: x + 1, lambda x: x * 2], (seed,), blocks_per_checkpoint=blocks_per_checkpoint
    )

    assert output.item() == 4.0


@pytest.mark.parametrize("blocks_per_checkpoint", [0, 3])
def test_invalid_inputs(blocks_per_checkpoint: int) -> None:
    blocks = [lambda x: x + 1, lambda x: x * 2]
    with pytest.raises(ValueError, match="blocks_per_checkpoint must be between 1 and len"):
        checkpoint_blocks(blocks, (torch.zeros(1, requires_grad=True),), blocks_per_checkpoint)
