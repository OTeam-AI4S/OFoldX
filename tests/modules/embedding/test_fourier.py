from __future__ import annotations

import torch

from ofoldx.modules.embedding.fourier import FrozenFourierEmbedding


def test_fourier_embedding_is_deterministic_frozen_and_bounded() -> None:
    embedding = FrozenFourierEmbedding(4)

    first = embedding(torch.ones(2))
    second = embedding(torch.ones(2))

    torch.testing.assert_close(first, second)
    assert first.shape == (2, 4)
    assert first.min() >= -1.0 and first.max() <= 1.0
    assert not embedding.projection.weight.requires_grad
    assert not embedding.projection.bias.requires_grad
