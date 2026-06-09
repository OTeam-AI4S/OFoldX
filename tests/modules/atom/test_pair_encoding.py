from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from ofoldx.modules.atom import encode_atom_pair


class ZeroFeedForward(nn.Module):
    def forward(self, pair: Tensor) -> Tensor:
        return torch.zeros_like(pair)


class PairEncodingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distance_vector_projection = zero_linear(3, 2)
        self.inverse_distance_projection = zero_linear(1, 2)
        self.distance_mask_projection = zero_linear(1, 2)
        self.left_atom_projection = zero_linear(5, 2)
        self.right_atom_projection = zero_linear(5, 2)
        self.feed_forward = ZeroFeedForward()


def zero_linear(in_features: int, out_features: int) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=False)
    nn.init.zeros_(linear.weight)
    return linear


def encode_with_bias(pair_bias: Tensor, *, prefix_ndim: int | None) -> Tensor:
    return encode_atom_pair(
        PairEncodingModule(),
        distance_vectors=torch.randn(2, 3, 4, 3),
        query_single=torch.randn(2, 3, 1, 5),
        key_single=torch.randn(2, 1, 4, 5),
        distance_term_mask=1.0,
        inverse_term_mask=1.0,
        valid_mask=torch.ones(2, 3, 4, 1),
        pair_bias=pair_bias,
        prefix_ndim=prefix_ndim,
    )


def test_encode_atom_pair_broadcasts_extra_prefix_axes() -> None:
    pair_bias = torch.randn(2, 2, 3, 4, 2)

    output = encode_with_bias(pair_bias, prefix_ndim=1)

    torch.testing.assert_close(output, pair_bias)


@pytest.mark.parametrize(
    ("pair_bias", "prefix_ndim", "match"),
    [
        (torch.randn(3, 4, 2), None, "rank"),
        (torch.randn(2, 2, 3, 4, 2), 5, "prefix_ndim"),
        (torch.randn(2, 2, 9, 4, 2), 1, "incompatible"),
        (torch.randn(1, 2, 3, 4, 2), 1, "cannot receive"),
    ],
)
def test_encode_atom_pair_rejects_unbroadcastable_pair_bias(
    pair_bias: Tensor,
    prefix_ndim: int | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        encode_with_bias(pair_bias, prefix_ndim=prefix_ndim)
