from __future__ import annotations

import torch

from ofoldx.modules.transformer import ConditionedTransformerBlock


class TestConditionedTransformerBlock:
    def test_accepts_key_mask(self) -> None:
        block = ConditionedTransformerBlock(
            hidden_size=4,
            conditioning_size=3,
            num_attention_heads=1,
            operator_backend="eager",
            transition="double",
        ).eval()

        output = block(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=torch.tensor([[True, False, True, True], [True, True, False, True]]),
        )

        assert output.shape == (2, 4, 4)
        assert torch.isfinite(output).all()
