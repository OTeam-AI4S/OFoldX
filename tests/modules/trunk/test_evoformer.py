from __future__ import annotations

import pytest
import torch

from ofoldx.modules.trunk.evoformer import (
    EvoformerStack,
    ExtraMsaEvoformerStack,
    OuterProductFirstEvoformerStack,
    OuterProductFirstExtraMsaEvoformerStack,
)


@pytest.mark.parametrize(
    "stack_type",
    [
        pytest.param(EvoformerStack, id="evoformer"),
        pytest.param(ExtraMsaEvoformerStack, id="extra-msa"),
        pytest.param(OuterProductFirstEvoformerStack, id="outer-product-first"),
        pytest.param(OuterProductFirstExtraMsaEvoformerStack, id="outer-product-first-extra-msa"),
    ],
)
def test_evoformer_stack_shapes(
    stack_type: type[
        EvoformerStack
        | ExtraMsaEvoformerStack
        | OuterProductFirstEvoformerStack
        | OuterProductFirstExtraMsaEvoformerStack
    ],
) -> None:
    stack = stack_type(
        num_blocks=1,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        num_msa_row_attention_heads=2,
        num_msa_column_attention_heads=4,
        triangle_multiplication_outgoing_hidden_size=3,
        triangle_multiplication_incoming_hidden_size=5,
        triangle_attention_hidden_size=2,
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
        triangle_attention_hidden_size=2,
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
