# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

import pytest
import torch

from ofoldx.modules.evoformer import (
    EvoformerBlock,
    EvoformerMsaBlock,
    EvoformerStack,
    ExtraMsaTrackBlock,
    OuterProductFirstEvoformerBlock,
)


@pytest.mark.parametrize(
    ("block_type", "msa_block_type"),
    [
        pytest.param(EvoformerBlock, EvoformerMsaBlock, id="evoformer"),
        pytest.param(EvoformerBlock, ExtraMsaTrackBlock, id="extra-msa"),
        pytest.param(OuterProductFirstEvoformerBlock, EvoformerMsaBlock, id="outer-product-first"),
        pytest.param(OuterProductFirstEvoformerBlock, ExtraMsaTrackBlock, id="outer-product-first-extra-msa"),
    ],
)
def test_evoformer_stack_shapes(
    block_type: type[EvoformerBlock],
    msa_block_type: type[torch.nn.Module],
) -> None:
    stack = EvoformerStack(
        block=block_type,
        msa_block=msa_block_type,
        num_blocks=1,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        num_msa_row_attention_heads=2,
        num_msa_column_attention_heads=4,
        triangle_multiplication_outgoing_hidden_size=3,
        triangle_multiplication_incoming_hidden_size=5,
        triangle_attention_head_size=2,
        num_triangle_attention_starting_node_heads=2,
        num_triangle_attention_ending_node_heads=3,
        msa_transition_expansion=2,
        pair_transition_expansion=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        backend="eager",
    ).eval()

    msa, pair = stack(
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 3, 3, 4),
        msa_mask=torch.ones(1, 2, 3),
        pair_mask=torch.ones(1, 3, 3, dtype=torch.bool),
    )

    assert msa.shape == (1, 2, 3, 8)
    assert pair.shape == (1, 3, 3, 4)


def test_evoformer_zero_blocks() -> None:
    stack = EvoformerStack(
        num_blocks=0,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        num_msa_row_attention_heads=2,
        num_msa_column_attention_heads=2,
        triangle_multiplication_outgoing_hidden_size=4,
        triangle_multiplication_incoming_hidden_size=4,
        triangle_attention_head_size=2,
        num_triangle_attention_starting_node_heads=2,
        num_triangle_attention_ending_node_heads=2,
        msa_transition_expansion=2,
        pair_transition_expansion=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        backend="eager",
    )
    msa = torch.randn(1, 2, 3, 8)
    pair = torch.randn(1, 3, 3, 4)

    actual_msa, actual_pair = stack(msa, pair)

    torch.testing.assert_close(actual_msa, msa)
    torch.testing.assert_close(actual_pair, pair)


@pytest.mark.parametrize(
    ("block_type", "expected_msa", "expected_pair", "expected_gradients"),
    [(EvoformerBlock, 5.0, 56.0, (21.0, 23.0)), (OuterProductFirstEvoformerBlock, 9.0, 14.0, (13.0, 3.0))],
)
def test_evoformer_ordered_updates(
    block_type: type[EvoformerBlock],
    expected_msa: float,
    expected_pair: float,
    expected_gradients: tuple[float, float],
) -> None:
    class MsaUpdate(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def forward(self, msa, pair, *, msa_mask, pair_mask):
            return msa + pair

    class OuterProductUpdate(torch.nn.Module):
        def forward(self, msa, msa_mask):
            return msa.square()

    class PairUpdate(torch.nn.Module):
        def forward(self, pair, *, pair_mask):
            return 2 * pair

    block = block_type(
        msa_size=1,
        pair_size=1,
        outer_product_hidden_size=1,
        num_msa_row_attention_heads=1,
        num_msa_column_attention_heads=1,
        triangle_multiplication_outgoing_hidden_size=1,
        triangle_multiplication_incoming_hidden_size=1,
        triangle_attention_head_size=1,
        num_triangle_attention_starting_node_heads=1,
        num_triangle_attention_ending_node_heads=1,
        msa_transition_expansion=1,
        pair_transition_expansion=1,
        msa_dropout=0.0,
        pair_dropout=0.0,
        backend="eager",
        msa_block=MsaUpdate,
    )
    block.outer_product_mean = OuterProductUpdate()
    block.pair_block = PairUpdate()
    msa = torch.full((1, 1, 1, 1), 2.0, requires_grad=True)
    pair = torch.full((1, 1, 1, 1), 3.0, requires_grad=True)
    result_msa, result_pair = block(
        msa,
        pair,
        msa_mask=torch.ones(1, 1, 1),
        pair_mask=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(result_msa, torch.full_like(msa, expected_msa))
    torch.testing.assert_close(result_pair, torch.full_like(pair, expected_pair))
    gradients = torch.autograd.grad(result_msa.sum() + result_pair.sum(), (msa, pair))
    for gradient, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, torch.full_like(gradient, expected))
