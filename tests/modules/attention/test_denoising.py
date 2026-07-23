from __future__ import annotations

import pytest
import torch

from ofoldx.modules.attention import DenoisingTokenEncoder


def test_denoising_token_encoder_zero_blocks_is_identity() -> None:
    encoder = DenoisingTokenEncoder(
        num_blocks=0,
        token_hidden_size=4,
        conditioning_size=3,
        pair_size=2,
        num_attention_heads=1,
        operator_backend="torch",
    )
    token_hidden = torch.randn(2, 5, 4)

    output = encoder(
        token_hidden,
        conditioning=torch.randn(2, 5, 3),
        pair=torch.randn(2, 5, 5, 2),
    )

    assert output is token_hidden


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True, False, True, True], [True, False, True, True, True]]),
        torch.tensor(
            [
                [
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                ],
                [
                    [True, False, True, True, True],
                    [True, False, True, True, True],
                    [True, False, True, True, True],
                    [True, False, True, True, True],
                    [True, False, True, True, True],
                ],
            ]
        ),
    ],
)
def test_denoising_token_encoder_accepts_key_or_pair_masks(mask: torch.Tensor) -> None:
    torch.manual_seed(0)
    encoder = DenoisingTokenEncoder(
        num_blocks=1,
        token_hidden_size=4,
        conditioning_size=3,
        pair_size=2,
        num_attention_heads=1,
        operator_backend="torch",
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
