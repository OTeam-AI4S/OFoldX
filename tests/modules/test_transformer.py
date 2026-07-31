from __future__ import annotations

import torch

from ofoldx.modules.transformer import DenoisingTransformerEncoder, DenoisingTransformerEncoderBlock


class TestDenoisingTransformerEncoder:
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

    def test_encoder_gradient_checkpointing(self) -> None:
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
