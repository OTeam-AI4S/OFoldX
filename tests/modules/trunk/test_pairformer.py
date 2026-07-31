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
        "backend": "eager",
    }
    single = torch.randn(1, 3, 8)
    pair = torch.randn(1, 3, 3, 4)

    block_single, block_pair = PairformerBlock(**kwargs)(single, pair)
    stack_single, stack_pair = PairformerStack(num_blocks=2, **kwargs)(single, pair)

    assert block_single.shape == stack_single.shape == single.shape
    assert block_pair.shape == stack_pair.shape == pair.shape
    assert not torch.equal(stack_single, single)
    assert not torch.equal(stack_pair, pair)


def test_pairformer_stack_gradient_checkpointing() -> None:
    kwargs = {
        "num_blocks": 2,
        "single_size": 8,
        "pair_size": 4,
        "triangle_multiplication_hidden_size": 4,
        "triangle_attention_hidden_size": 2,
        "num_triangle_attention_heads": 2,
        "num_pair_attention_heads": 2,
        "dropout": 0.0,
        "backend": "eager",
    }
    eager = PairformerStack(**kwargs)
    checkpointed = PairformerStack(**kwargs)
    checkpointed.load_state_dict(eager.state_dict())
    checkpointed.gradient_checkpointing = True
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
    for checkpointed_parameter, eager_parameter in zip(
        checkpointed.parameters(),
        eager.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(checkpointed_parameter.grad, eager_parameter.grad)
