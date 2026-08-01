from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.trunk import PairformerBlock, PairformerStack


def _kwargs() -> dict[str, object]:
    return {
        "single_size": 8,
        "pair_size": 4,
        "triangle_multiplication_hidden_size": 4,
        "triangle_attention_hidden_size": 2,
        "num_triangle_attention_heads": 2,
        "num_pair_attention_heads": 2,
        "dropout": 0.0,
        "backend": "eager",
    }


class TestPairformerBlock:
    def test_shapes(self) -> None:
        single = torch.randn(1, 3, 8)
        pair = torch.randn(1, 3, 3, 4)

        output_single, output_pair = PairformerBlock(**_kwargs())(single, pair)

        assert output_single.shape == single.shape
        assert output_pair.shape == pair.shape


class TestPairformerStack:
    def test_shapes(self) -> None:
        single = torch.randn(1, 3, 8)
        pair = torch.randn(1, 3, 3, 4)

        output_single, output_pair = PairformerStack(num_blocks=2, **_kwargs())(single, pair)

        assert output_single.shape == single.shape
        assert output_pair.shape == pair.shape

    def test_gradient_checkpointing(self) -> None:
        eager = PairformerStack(num_blocks=2, **_kwargs())
        checkpointed = PairformerStack(num_blocks=2, **_kwargs())
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_single = torch.randn(1, 3, 8, requires_grad=True)
        eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        checkpointed_single = eager_single.detach().clone().requires_grad_()
        checkpointed_pair = eager_pair.detach().clone().requires_grad_()

        eager_output = eager(eager_single, eager_pair)
        checkpointed_output = checkpointed(checkpointed_single, checkpointed_pair)
        sum(output.sum() for output in eager_output).backward()
        sum(output.sum() for output in checkpointed_output).backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_single.grad, eager_single.grad)
        torch.testing.assert_close(checkpointed_pair.grad, eager_pair.grad)
        for actual, expected in zip(checkpointed.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
