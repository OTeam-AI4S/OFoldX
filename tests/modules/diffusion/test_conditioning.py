from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from ofoldx.modules.diffusion.conditioning import condition_pair, condition_single


class NoiseFeatures(nn.Module):
    def forward(self, noise: Tensor) -> Tensor:
        return torch.stack((noise, noise.square()), dim=-1)


class OnesTransition(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        return torch.ones_like(hidden)


def zero_linear(in_features: int, out_features: int) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=False)
    nn.init.zeros_(linear.weight)
    return linear


@pytest.mark.parametrize(
    ("scaled_noise", "sample_axis", "expected_shape", "expected_noise_shape"),
    [
        (torch.tensor([0.5, 1.5]), False, (2, 3, 4), (2, 2)),
        (torch.tensor([[0.5, 1.5], [2.0, 3.0]]), True, (2, 2, 3, 4), (2, 2, 2)),
    ],
)
def test_condition_single_applies_noise_and_transitions(
    scaled_noise: Tensor,
    sample_axis: bool,
    expected_shape: tuple[int, ...],
    expected_noise_shape: tuple[int, ...],
) -> None:
    single_a = torch.randn(2, 3, 2)
    single_b = torch.randn(2, 3, 2)

    conditioned, normed_noise = condition_single(
        single_a,
        single_b,
        scaled_noise,
        norm=nn.Identity(),
        projection=zero_linear(4, 4),
        noise_fourier=NoiseFeatures(),
        noise_norm=nn.Identity(),
        noise_projection=zero_linear(2, 4),
        transitions=[OnesTransition()],
        sample_axis=sample_axis,
    )

    assert conditioned.shape == expected_shape
    assert normed_noise.shape == expected_noise_shape
    torch.testing.assert_close(conditioned, torch.ones(expected_shape))


def test_condition_pair_applies_relative_position_and_transitions() -> None:
    pair_trunk = torch.randn(2, 3, 3, 2)
    relative_position_encoding = torch.randn(2, 3, 3, 1)

    pair = condition_pair(
        pair_trunk,
        relative_position_encoding,
        norm=nn.Identity(),
        projection=zero_linear(3, 4),
        transitions=[OnesTransition()],
    )

    assert pair.shape == (2, 3, 3, 4)
    torch.testing.assert_close(pair, torch.ones(2, 3, 3, 4))
