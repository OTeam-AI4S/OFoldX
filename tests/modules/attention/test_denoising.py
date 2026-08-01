from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.attention import DenoisingTokenEncoder


class TestDenoisingTokenEncoder:
    def test_identity(self) -> None:
        encoder = DenoisingTokenEncoder(
            num_blocks=0,
            token_hidden_size=4,
            conditioning_size=3,
            pair_size=2,
            num_attention_heads=1,
            backend="eager",
        )
        token_hidden = torch.randn(2, 5, 4)

        output = encoder(
            token_hidden,
            conditioning=torch.randn(2, 5, 3),
            pair=torch.randn(2, 5, 5, 2),
        )

        torch.testing.assert_close(output, token_hidden)

    @pytest.mark.parametrize(
        "mask",
        [
            torch.tensor([[True, True, False, True, True], [True, False, True, True, True]]),
            torch.tensor(
                [
                    [[True, True, False, True, True]] * 5,
                    [[True, False, True, True, True]] * 5,
                ]
            ),
        ],
        ids=("key", "pair"),
    )
    def test_masks(self, mask: torch.Tensor) -> None:
        torch.manual_seed(0)
        encoder = DenoisingTokenEncoder(
            num_blocks=1,
            token_hidden_size=4,
            conditioning_size=3,
            pair_size=2,
            num_attention_heads=1,
            backend="eager",
        ).eval()

        with torch.no_grad():
            output = encoder(
                token_hidden=torch.randn(2, 5, 4),
                conditioning=torch.randn(2, 5, 3),
                pair=torch.randn(2, 5, 5, 2),
                pair_mask=mask,
            )

        assert output.shape == (2, 5, 4)
        assert torch.isfinite(output).all()

    def test_checkpointing(self) -> None:
        torch.manual_seed(1)
        eager = DenoisingTokenEncoder(
            num_blocks=2,
            token_hidden_size=4,
            conditioning_size=3,
            pair_size=2,
            num_attention_heads=1,
            backend="eager",
        )
        checkpointed = deepcopy(eager)
        checkpointed.transformer.gradient_checkpointing = True
        checkpointed.transformer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        eager_inputs = (
            torch.randn(2, 5, 4, requires_grad=True),
            torch.randn(2, 5, 3, requires_grad=True),
            torch.randn(2, 5, 5, 2, requires_grad=True),
        )
        checkpointed_inputs = tuple(value.detach().clone().requires_grad_() for value in eager_inputs)
        pair_mask = torch.ones(2, 5, 5, dtype=torch.bool)
        pair_mask[:, :, -1] = False

        eager_output = eager(*eager_inputs, pair_mask)
        checkpointed_output = checkpointed(*checkpointed_inputs, pair_mask)
        eager_output.square().sum().backward()
        checkpointed_output.square().sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        for actual, expected in zip(checkpointed_inputs, eager_inputs, strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
