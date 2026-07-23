from __future__ import annotations

import pytest
import torch

from ofoldx.modules.transformer import ConditionedTransformerBlock


@pytest.fixture
def block() -> ConditionedTransformerBlock:
    torch.manual_seed(0)
    return ConditionedTransformerBlock(
        hidden_size=4,
        conditioning_size=3,
        num_attention_heads=1,
        operator_backend="torch",
        transition="double",
    ).eval()


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, False, True, True], [True, True, False, True]]),
        torch.tensor(
            [
                [
                    [True, False, True, True],
                    [True, False, True, True],
                    [True, False, True, True],
                    [True, False, True, True],
                ],
                [
                    [True, True, False, True],
                    [True, True, False, True],
                    [True, True, False, True],
                    [True, True, False, True],
                ],
            ]
        ),
    ],
)
def test_conditioned_transformer_block_accepts_key_or_pair_masks(
    block: ConditionedTransformerBlock,
    mask: torch.Tensor,
) -> None:
    with torch.no_grad():
        output = block(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=mask,
        )

    assert output.shape == (2, 4, 4)
    assert torch.isfinite(output).all()


def test_conditioned_transformer_block_rejects_malformed_mask(block: ConditionedTransformerBlock) -> None:
    with pytest.raises(ValueError, match="transformer mask"):
        block(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=torch.ones(2, 3, dtype=torch.bool),
        )
