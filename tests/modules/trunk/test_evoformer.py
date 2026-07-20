from __future__ import annotations

import torch

from ofoldx.modules.trunk import EvoformerStack


def test_evoformer_stack_refines_msa_and_pair_shapes() -> None:
    stack = EvoformerStack(
        num_blocks=1,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        outer_product_first=False,
        num_msa_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        msa_transition_expansion=2,
        pair_transition_expansion=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        is_extra_msa=False,
        operator_backend="sdpa",
    ).eval()

    msa, pair = stack(
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 3, 3, 4),
        msa_mask=torch.ones(1, 2, 3),
        pair_mask=torch.ones(1, 3, 3, dtype=torch.bool),
    )

    assert msa.shape == (1, 2, 3, 8)
    assert pair.shape == (1, 3, 3, 4)


def test_evoformer_stack_allows_zero_blocks() -> None:
    stack = EvoformerStack(
        num_blocks=0,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        outer_product_first=False,
        num_msa_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        msa_transition_expansion=2,
        pair_transition_expansion=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        is_extra_msa=False,
        operator_backend="sdpa",
    )
    msa = torch.randn(1, 2, 3, 8)
    pair = torch.randn(1, 3, 3, 4)

    actual_msa, actual_pair = stack(msa, pair)

    assert actual_msa is msa
    assert actual_pair is pair


def test_evoformer_stack_supports_extra_msa_global_attention() -> None:
    stack = EvoformerStack(
        num_blocks=1,
        msa_size=8,
        pair_size=4,
        outer_product_hidden_size=4,
        outer_product_first=False,
        num_msa_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        msa_transition_expansion=2,
        pair_transition_expansion=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        is_extra_msa=True,
        operator_backend="sdpa",
    ).eval()

    msa, pair = stack(
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 3, 3, 4),
        msa_mask=torch.ones(1, 2, 3),
        pair_mask=torch.ones(1, 3, 3, dtype=torch.bool),
    )

    assert msa.shape == (1, 2, 3, 8)
    assert pair.shape == (1, 3, 3, 4)
