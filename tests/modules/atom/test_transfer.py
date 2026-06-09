from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from ofoldx.modules.atom import (
    TokenPairToAtomPairBroadcast,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from ofoldx.modules.atom.transfer import scatter_mean_to_tokens

if TYPE_CHECKING:
    from torch import Tensor


@pytest.mark.parametrize("count_eps", [0.0, 1e-6])
@pytest.mark.parametrize("extra_prefix", [(), (1,)])
def test_atom_token_transfer(count_eps: float, extra_prefix: tuple[int, ...]) -> None:
    # masked-out atom 1 drops from both sum and count, so token 0 = [1,2]/1, token 1 = ([5,10]+[7,14])/2.
    atom_hidden = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]]])
    atom_to_token = torch.tensor([[0, 0, 1, 1]])
    atom_mask = torch.tensor([[True, False, True, True]])

    # count_eps softens the denominator from clamp(count, 1) to count + eps: each mean scales by count/(count+eps).
    token0_scale = 1.0 / (1.0 + count_eps) if count_eps else 1.0
    token1_scale = 2.0 / (2.0 + count_eps) if count_eps else 1.0
    expected = torch.tensor(
        [[[1.0 * token0_scale, 2.0 * token0_scale], [6.0 * token1_scale, 12.0 * token1_scale], [0.0, 0.0]]]
    )

    token_hidden = scatter_mean_to_tokens(atom_hidden, atom_to_token, 3, atom_mask, count_eps=count_eps)
    # tight tol so the ~5e-7 count_eps shift is discriminated, not absorbed
    torch.testing.assert_close(token_hidden, expected, rtol=0.0, atol=1e-6)

    # token -> atom broadcast gathers each atom's token row, here through an arbitrary leading dim
    token_hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).reshape(1, *extra_prefix, 3, 2)
    atom_to_token = torch.tensor([[2, 0, 1, 1]])
    atom_hidden = TokenToAtomBroadcast()(token_hidden, atom_to_token)
    broadcast_expected = torch.tensor([[[5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]]).reshape(
        1, *extra_prefix, 4, 2
    )
    assert torch.equal(atom_hidden, broadcast_expected)


def test_token_pair_dense_broadcast() -> None:
    # Dense atom-pair broadcast gathers token_pair[row_token, col_token] for every atom pair.
    token_pair = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    atom_to_token = torch.tensor([[1, 0, 1]])

    atom_pair = TokenPairToAtomPairBroadcast()(token_pair, atom_to_token)

    expected = torch.tensor([[[[4.0], [3.0], [4.0]], [[2.0], [1.0], [2.0]], [[4.0], [3.0], [4.0]]]])
    assert torch.equal(atom_pair, expected)


@pytest.mark.parametrize(
    ("atom_to_token", "expected"),
    [
        # Shared-across-batch layout -> oops windowing kernel path. q=2,k=4 over 4 atoms gives 2 windows;
        # key_padding_left=(4-2)//2=1 so window w spans atoms [2w-1 .. 2w+2], F.pad backfilling token id 0.
        (
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            torch.tensor(
                [
                    [
                        [[0.0, 0.0, 1.0, 2.0], [10.0, 10.0, 11.0, 12.0]],
                        [[21.0, 22.0, 23.0, 20.0], [31.0, 32.0, 33.0, 30.0]],
                    ],
                    [
                        [[100.0, 100.0, 101.0, 102.0], [110.0, 110.0, 111.0, 112.0]],
                        [[121.0, 122.0, 123.0, 120.0], [131.0, 132.0, 133.0, 130.0]],
                    ],
                ]
            ),
        ),
        # Distinct per-batch layout -> torch fallback path (_shared_atom_to_token returns None); batch 1 reverses
        # its assignment to tokens [3, 2, 1, 0], same window geometry over different gathered ids.
        (
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            torch.tensor(
                [
                    [
                        [[0.0, 0.0, 1.0, 2.0], [10.0, 10.0, 11.0, 12.0]],
                        [[21.0, 22.0, 23.0, 20.0], [31.0, 32.0, 33.0, 30.0]],
                    ],
                    [
                        [[130.0, 133.0, 132.0, 131.0], [120.0, 123.0, 122.0, 121.0]],
                        [[112.0, 111.0, 110.0, 110.0], [102.0, 101.0, 100.0, 100.0]],
                    ],
                ]
            ),
        ),
    ],
    ids=["shared-layout", "per-batch-layout"],
)
def test_token_pair_local_window_broadcast(atom_to_token: Tensor, expected: Tensor) -> None:
    # token_pair[b, i, j] = 100*b + 10*i + j uniquely encodes each (row, col) token pair, so a gathered value
    # reads back directly as the (query_token, key_token) the window selected (expected traced in parametrize).
    num_tokens = 4
    token_pair = torch.zeros(2, num_tokens, num_tokens, 1)
    rows = torch.arange(num_tokens).view(num_tokens, 1)
    cols = torch.arange(num_tokens).view(1, num_tokens)
    token_pair[0, :, :, 0] = (10 * rows + cols).float()
    token_pair[1, :, :, 0] = (100 + 10 * rows + cols).float()

    atom_pair = TokenPairToLocalAtomPairBroadcast(num_query_atoms=2, num_key_atoms=4)(token_pair, atom_to_token)

    assert atom_pair.shape == (2, 2, 2, 4, 1)
    torch.testing.assert_close(atom_pair, expected.unsqueeze(-1))
