from __future__ import annotations

from typing import Any

import pytest
import torch

from ofoldx.modules.trunk import PairformerBlock, PairformerStack


@pytest.fixture
def kwargs() -> dict[str, Any]:
    r"""
    Shared small Pairformer config using the family-wide PyTorch reference.
    """
    return {
        "single_size": 8,
        "pair_size": 4,
        "triangle_multiplication_hidden_size": 4,
        "triangle_attention_hidden_size": 2,
        "num_triangle_attention_heads": 2,
        "num_pair_attention_heads": 2,
        "dropout": 0.0,
        "operator_backend": "eager",
    }


@pytest.mark.parametrize("single_size", [8, 0])
def test_forward(kwargs: dict[str, Any], single_size: int) -> None:
    # single_size==0 drops the pair_attention/single_feed_forward branch: single stays None end-to-end.
    kwargs = {**kwargs, "single_size": single_size}
    torch.manual_seed(0)
    block = PairformerBlock(**kwargs).eval()  # eval() so dropout is identity and forward is deterministic
    torch.manual_seed(1)
    pair = torch.randn(1, 3, 3, 4)
    single = torch.randn(1, 3, single_size) if single_size else None
    full_mask = torch.ones(1, 3, 3, dtype=torch.bool)

    block_single, block_pair = block(single, pair, pair_mask=full_mask)
    assert block_pair.shape == (1, 3, 3, 4)
    assert (block_single is None) is (single_size == 0)

    assert not torch.allclose(block_pair, pair, atol=1e-6)  # residual update must change the input

    redo_single, redo_pair = block(single, pair, pair_mask=full_mask)
    torch.testing.assert_close(redo_pair, block_pair, rtol=0, atol=0)  # deterministic with dropout off

    # An all-true mask must be a no-op: every path treats it like pair_mask=None.
    none_single, none_pair = block(single, pair, pair_mask=None)
    torch.testing.assert_close(none_pair, block_pair)

    partial_mask = full_mask.clone()
    partial_mask[0, 2, 2] = False
    partial_mask[0, 0, 1] = False
    masked_single, masked_pair = block(single, pair, pair_mask=partial_mask)
    assert not torch.allclose(masked_pair, block_pair, atol=1e-6)  # masked keys must flow through

    if single_size:
        assert block_single is not None and single is not None
        assert block_single.shape == (1, 3, single_size)
        # The same invariants hold for the single representation it updates.
        assert not torch.allclose(block_single, single, atol=1e-6)  # residual did real work
        torch.testing.assert_close(redo_single, block_single, rtol=0, atol=0)  # deterministic
        torch.testing.assert_close(none_single, block_single)  # all-true mask is a no-op
        assert not torch.allclose(masked_single, block_single, atol=1e-6)  # partial mask flows through

    torch.manual_seed(2)
    stack = PairformerStack(num_blocks=2, **kwargs).eval()
    stack_single, stack_pair = stack(single, pair)
    assert stack_pair.shape == (1, 3, 3, 4)
    assert (stack_single is None) is (single_size == 0)
    assert not torch.allclose(stack_pair, pair, atol=1e-6)
    redo_stack_single, redo_stack_pair = stack(single, pair)
    torch.testing.assert_close(redo_stack_pair, stack_pair, rtol=0, atol=0)

    one_block_stack = PairformerStack(num_blocks=1, **kwargs).eval()
    one_block_stack.blocks[0].load_state_dict(stack.blocks[0].state_dict())
    _, one_pair = one_block_stack(single, pair)
    # 2-block output must diverge from the 1-block result: the second block actually runs.
    assert not torch.allclose(stack_pair, one_pair, atol=1e-6)
