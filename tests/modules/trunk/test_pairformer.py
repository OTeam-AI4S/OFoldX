from __future__ import annotations

import torch

from ofoldx.modules.trunk import PairformerBlock, PairformerStack


def test_pairformer_block_and_stack() -> None:
    kwargs = {
        "single_size": 8,
        "pair_size": 4,
        "triangle_multiplication_hidden_size": 4,
        "triangle_attention_hidden_size": 2,
        "num_triangle_attention_heads": 2,
        "num_pair_attention_heads": 2,
        "dropout": 0.0,
        "operator_backend": "eager",
    }
    single = torch.randn(1, 3, 8)
    pair = torch.randn(1, 3, 3, 4)

    block_single, block_pair = PairformerBlock(**kwargs)(single, pair)
    stack_single, stack_pair = PairformerStack(num_blocks=2, **kwargs)(single, pair)

    assert block_single.shape == stack_single.shape == single.shape
    assert block_pair.shape == stack_pair.shape == pair.shape
    assert not torch.equal(stack_single, single)
    assert not torch.equal(stack_pair, pair)
