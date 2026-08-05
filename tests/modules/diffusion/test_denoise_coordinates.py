from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from ofoldx.modules.diffusion import denoise_coordinates


@dataclass
class _Context:
    batch_shape: torch.Size
    num_atoms: int

    def to_denoising_input(self, *, noisy_coords: Tensor, noise_level: Tensor) -> Any:
        return SimpleNamespace(noisy_coords=noisy_coords, noise_level=noise_level)


def _context() -> _Context:
    return _Context(batch_shape=torch.Size([2]), num_atoms=3)


def _run(
    noisy_coords: Tensor,
    noise_level: Tensor,
    sample_chunk_size: int | None,
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    coordinates = noisy_coords.detach().clone().requires_grad_()
    scale = torch.tensor(0.75, dtype=coordinates.dtype, requires_grad=True)
    calls: list[int] = []

    def denoise(inputs: Any) -> Tensor:
        calls.append(inputs.noisy_coords.shape[-3])
        return scale * inputs.noisy_coords + inputs.noise_level[..., None, None]

    output = denoise_coordinates(
        denoise,
        context=_context(),  # type: ignore[arg-type]
        noisy_coords=coordinates,
        noise_level=noise_level,
        sample_chunk_size=sample_chunk_size,
    )
    input_gradient, parameter_gradient = torch.autograd.grad(output.square().sum(), (coordinates, scale))
    return output, input_gradient, parameter_gradient, calls


def test_chunks_preserve_values_input_gradients_and_parameter_gradients() -> None:
    generator = torch.Generator().manual_seed(17)
    noisy_coords = torch.randn(2, 5, 3, 3, generator=generator)
    noise_level = torch.randn(2, 5, generator=generator)

    unchunked = _run(noisy_coords, noise_level, None)
    chunked = _run(noisy_coords, noise_level, 2)

    for actual, expected in zip(chunked[:3], unchunked[:3]):
        torch.testing.assert_close(actual, expected)
    assert unchunked[3] == [5]
    assert chunked[3] == [2, 2, 1]


@pytest.mark.parametrize(
    ("sample_chunk_size", "expected_calls"),
    [(None, [5]), (1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (5, [5]), (8, [5])],
)
def test_chunk_boundaries_and_remainder(sample_chunk_size: int | None, expected_calls: list[int]) -> None:
    noisy_coords = torch.zeros(2, 5, 3, 3)
    noise_level = torch.ones(2, 5)

    output, _, _, calls = _run(noisy_coords, noise_level, sample_chunk_size)

    assert output.shape == (2, 5, 3, 3)
    assert calls == expected_calls


@pytest.mark.parametrize("sample_chunk_size", [True, False, 0, -1, 1.5])
def test_invalid_chunk_size_is_rejected(sample_chunk_size: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        denoise_coordinates(
            lambda inputs: inputs.noisy_coords,
            context=_context(),  # type: ignore[arg-type]
            noisy_coords=torch.zeros(2, 5, 3, 3),
            noise_level=torch.ones(2, 5),
            sample_chunk_size=sample_chunk_size,
        )


@pytest.mark.parametrize(
    ("noisy_shape", "noise_shape"),
    [
        ((2, 5, 3, 3), (2, 4)),
        ((2, 5, 4, 3), (2, 5)),
        ((3, 5, 3, 3), (3, 5)),
        ((2, 5, 3, 2), (2, 5)),
        ((2, 3, 3), (2, 5)),
    ],
)
def test_coordinate_and_noise_shape_mismatch_is_rejected(
    noisy_shape: tuple[int, ...], noise_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="shape|batch|atom|coordinate|noise"):
        denoise_coordinates(
            lambda inputs: inputs.noisy_coords,
            context=_context(),  # type: ignore[arg-type]
            noisy_coords=torch.zeros(noisy_shape),
            noise_level=torch.ones(noise_shape),
            sample_chunk_size=2,
        )
