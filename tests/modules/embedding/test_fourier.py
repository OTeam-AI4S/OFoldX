from __future__ import annotations

import torch
import torch.nn.functional as F

from ofoldx.modules.embedding.fourier import FrozenFourierEmbedding


class TestFrozenFourierEmbedding:
    def test_equation(self) -> None:
        embedding = FrozenFourierEmbedding(4)

        output = embedding(torch.ones(2))
        projected = F.linear(
            torch.ones(2, 1),
            embedding.projection.weight,
            embedding.projection.bias,
        )

        torch.testing.assert_close(output, torch.cos(2 * torch.pi * projected))
        assert output.shape == (2, 4)

    def test_frozen(self) -> None:
        embedding = FrozenFourierEmbedding(4)

        assert not embedding.projection.weight.requires_grad
        assert not embedding.projection.bias.requires_grad
