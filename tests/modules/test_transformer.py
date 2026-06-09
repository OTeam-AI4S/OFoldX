from __future__ import annotations

import pytest
import torch

from ofoldx.modules.transformer import ConditionedTransformerLayer


@pytest.fixture
def layer() -> ConditionedTransformerLayer:
    torch.manual_seed(0)
    return ConditionedTransformerLayer(
        hidden_size=4,
        conditioning_size=3,
        num_attention_heads=1,
        operator_backend="sdpa",
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
def test_conditioned_transformer_layer_accepts_key_or_pair_masks(
    layer: ConditionedTransformerLayer,
    mask: torch.Tensor,
) -> None:
    with torch.no_grad():
        output = layer(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=mask,
        )

    assert output.shape == (2, 4, 4)
    assert torch.isfinite(output).all()


def test_conditioned_transformer_layer_rejects_malformed_mask(layer: ConditionedTransformerLayer) -> None:
    with pytest.raises(ValueError, match="transformer mask"):
        layer(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            mask=torch.ones(2, 3, dtype=torch.bool),
        )
