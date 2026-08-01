from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.transformer import (
    AtomCrossAttentionTransformerEncoderBlock,
    ConditionedTransformerEncoderBlock,
    DenoisingTransformerEncoder,
    DenoisingTransformerEncoderBlock,
    NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
)


class TestConditionedTransformerEncoderBlock:
    def test_equation(self) -> None:
        torch.manual_seed(0)
        block = ConditionedTransformerEncoderBlock(
            hidden_size=4,
            conditioning_size=3,
            num_attention_heads=1,
            post_layer_norm=True,
            backend="eager",
        )
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        conditioning = torch.randn(2, 3, 3)
        bias = torch.randn(2, 3, 3, 1)

        output = block(hidden, conditioning, bias)

        normalized = block.conditioned_norm(hidden, conditioning)
        attention = block.attention(
            normalized,
            biases=(bias.movedim(-1, -3),),
        )
        updated = hidden + block.output_gate_projection(conditioning).sigmoid() * attention
        transition = block.transition
        transition_input = transition.conditioned_norm(updated, conditioning)
        activated = (
            transition.silu(transition.gate_projection(transition_input))
            * transition.value_projection(transition_input)
            * transition.expand_projection(transition_input)
        )
        transition_update = transition.conditioning_gate_projection(conditioning).sigmoid() * (
            transition.output_projection(activated)
        )
        expected = block.post_norm(updated + transition_update)

        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        assert hidden.grad is not None


class TestAtomCrossAttentionTransformerEncoderBlock:
    @pytest.mark.parametrize(
        ("block_class", "normalize_keys_from_queries"),
        [
            (AtomCrossAttentionTransformerEncoderBlock, False),
            (NormalizedKeyAtomCrossAttentionTransformerEncoderBlock, True),
        ],
    )
    def test_key_normalization_equation(self, block_class, normalize_keys_from_queries: bool) -> None:
        torch.manual_seed(0)
        block = block_class(
            hidden_size=4,
            conditioning_size=3,
            num_attention_heads=1,
            backend="eager",
        )
        hidden = torch.randn(2, 3, 4)
        conditioning = torch.randn(2, 3, 3)
        bias = torch.randn(2, 3, 3, 1)

        output = block(hidden, conditioning, bias)

        normalized = block.conditioned_norm(hidden, conditioning)
        key_source = normalized if normalize_keys_from_queries else hidden
        keys = block.key_value_norm(key_source, conditioning)
        attention = block.attention(
            normalized,
            key=keys,
            biases=(bias.movedim(-1, -3),),
        )
        updated = hidden + block.output_gate_projection(conditioning).sigmoid() * attention
        expected = updated + block.transition(updated, conditioning)

        torch.testing.assert_close(output, expected)


class TestDenoisingTransformerEncoderBlock:
    def test_accepts_key_mask(self) -> None:
        block = DenoisingTransformerEncoderBlock(
            hidden_size=4,
            conditioning_size=3,
            num_attention_heads=1,
            backend="eager",
        ).eval()

        output = block(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=torch.tensor([[True, False, True, True], [True, True, False, True]]),
        )

        assert output.shape == (2, 4, 4)
        assert torch.isfinite(output).all()


class TestDenoisingTransformerEncoder:
    def test_gradient_checkpointing(self) -> None:
        eager = DenoisingTransformerEncoder(
            num_blocks=2,
            num_attention_heads=1,
            hidden_size=4,
            conditioning_size=3,
            backend="eager",
        )
        checkpointed = DenoisingTransformerEncoder(
            num_blocks=2,
            num_attention_heads=1,
            hidden_size=4,
            conditioning_size=3,
            backend="eager",
        )
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_hidden = torch.randn(2, 4, 4, requires_grad=True)
        checkpointed_hidden = eager_hidden.detach().clone().requires_grad_()
        conditioning = torch.randn(2, 4, 3)
        bias = torch.zeros(2, 4, 4, 2)

        eager_output = eager(eager_hidden, conditioning, bias)
        checkpointed_output = checkpointed(checkpointed_hidden, conditioning, bias)
        eager_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_hidden.grad, eager_hidden.grad)
        for checkpointed_parameter, eager_parameter in zip(
            checkpointed.parameters(),
            eager.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(checkpointed_parameter.grad, eager_parameter.grad)
