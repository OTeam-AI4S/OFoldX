from __future__ import annotations

from typing import Literal

import pytest
import torch

from ofoldx.modules.atom import LocalAtomGeometryBuilder
from ofoldx.modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoisingInput,
    CoordinateEdmSampler,
    EdmScheduler,
)
from ofoldx.modules.embedding.position import generate_relative_position_features


def _make_context(batch_size: int = 1) -> CoordinateDenoiseContext:
    reference_positions = torch.zeros(batch_size, 5, 3)
    single = torch.zeros(batch_size, 3, 6)
    pair = torch.zeros(batch_size, 3, 3, 4)
    return CoordinateDenoiseContext(
        atom_features=torch.zeros(batch_size, 5, 5),
        atom_to_token=torch.tensor([[0, 0, 1, 2, 2]]).expand(batch_size, -1),
        reference_positions=reference_positions,
        reference_charges=torch.zeros(batch_size, 5, 1),
        reference_mask=torch.ones(batch_size, 5, dtype=torch.bool),
        atom_mask=torch.ones(batch_size, 5, dtype=torch.bool),
        token_embedding=torch.zeros(batch_size, 3, 8),
        single=single,
        conditioning_single=single,
        conditioning_pair=pair,
        relative_features=generate_relative_position_features(
            residue_index=torch.arange(3).expand(batch_size, -1),
            chain_index=torch.zeros(batch_size, 3, dtype=torch.long),
            entity_index=torch.zeros(batch_size, 3, dtype=torch.long),
            symmetry_index=torch.zeros(batch_size, 3, dtype=torch.long),
            token_index=torch.arange(3).expand(batch_size, -1),
            dtype=torch.float32,
        ),
        local_geometry=LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)(
            reference_positions,
            torch.zeros(batch_size, 5, dtype=torch.long),
        ),
    )


def _make_sampler(*, solver: Literal["euler", "heun"] = "euler") -> CoordinateEdmSampler:
    # churn_rate=0 + step_scale=1 makes an identity denoiser a fixed point, so a sampling run
    # with initial_coords returns them unchanged -- a cheap closed form to assert against.
    return CoordinateEdmSampler(
        scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1),
        solver=solver,
        churn_rate=0.0,
        step_scale=1.0,
        center_random_augmentation=False,
    )


def _zeros(inputs: CoordinateDenoisingInput) -> torch.Tensor:
    return torch.zeros_like(inputs.noisy_coords)


def _identity(inputs: CoordinateDenoisingInput) -> torch.Tensor:
    return inputs.noisy_coords


@pytest.fixture
def context() -> CoordinateDenoiseContext:
    return _make_context()


def test_sample_shapes_fixed_point_and_determinism(context: CoordinateDenoiseContext) -> None:
    sampler = _make_sampler()

    output = sampler(_zeros, context=context, num_samples=2, num_sampling_steps=1)
    assert output.shape == (1, 2, 5, 3)
    step_input = context.to_denoising_input(output, torch.ones(1, 2))
    assert isinstance(step_input, CoordinateDenoisingInput)
    assert step_input.noisy_coords is output

    # an identity denoiser is a fixed point of the (churn=0, step=1) update: coords come back exactly.
    sampled = torch.ones(1, 2, 5, 3)
    assert torch.equal(
        sampler(_identity, context=context, num_samples=2, num_sampling_steps=1, initial_coords=sampled), sampled
    )
    # an unsampled [..., N_atom, 3] initial is broadcast across the sample axis.
    unsampled = torch.ones(1, 5, 3)
    expanded = sampler(_identity, context=context, num_samples=2, num_sampling_steps=1, initial_coords=unsampled)
    assert expanded.shape == (1, 2, 5, 3)
    assert torch.equal(expanded[:, 0], unsampled)
    assert torch.equal(expanded[:, 1], unsampled)

    first = sampler(
        _zeros, context=context, num_samples=2, num_sampling_steps=1, generator=torch.Generator().manual_seed(7)
    )
    second = sampler(
        _zeros, context=context, num_samples=2, num_sampling_steps=1, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(first, second)


def test_chunking(context: CoordinateDenoiseContext) -> None:
    sampler = _make_sampler()

    # chunked sampling must reproduce the unchunked result bit-for-bit.
    sampled = torch.arange(1 * 4 * 5 * 3, dtype=torch.float32).reshape(1, 4, 5, 3)
    unchunked = sampler(_identity, context=context, num_samples=4, num_sampling_steps=1, initial_coords=sampled)
    chunked = sampler(
        _identity, context=context, num_samples=4, num_sampling_steps=1, sample_chunk_size=2, initial_coords=sampled
    )
    assert torch.equal(chunked, unchunked)

    unsampled = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    output = sampler(
        _identity, context=context, num_samples=4, num_sampling_steps=1, sample_chunk_size=2, initial_coords=unsampled
    )
    assert output.shape == (1, 4, 5, 3)
    assert torch.equal(output[:, 0], unsampled)
    assert torch.equal(output[:, -1], unsampled)


def test_unsampled_condition_coords_when_batch_equals_num_samples() -> None:
    context = _make_context(batch_size=2)
    sampler = _make_sampler()
    condition_coords = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)

    output = sampler(
        _identity,
        context=context,
        num_samples=2,
        num_sampling_steps=1,
        sample_chunk_size=1,
        initial_coords=torch.zeros(2, 5, 3),
        condition_coords=condition_coords,
        condition_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert torch.equal(output[:, 0], condition_coords)
    assert torch.equal(output[:, 1], condition_coords)


def test_augment_and_heun(context: CoordinateDenoiseContext) -> None:
    # regression: masked augmentation must broadcast the atom mask over the sample axis, not collide
    # with a real batch dimension (errored for batch>1, n_sample>1); padded atoms stay zeroed.
    sampler = CoordinateEdmSampler(
        scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1),
        center_random_augmentation=True,
    )
    batch, num_atoms = 2, 4
    coords = torch.randn(batch, num_atoms, 3)
    atom_mask = torch.ones(batch, num_atoms, dtype=torch.bool)
    atom_mask[:, -1] = False
    augmented = sampler.augment_target(
        coords, num_samples=3, atom_mask=atom_mask, generator=torch.Generator().manual_seed(0)
    )
    assert augmented.shape == (batch, 3, num_atoms, 3)
    torch.testing.assert_close(augmented[:, :, -1], torch.zeros(batch, 3, 3))
    assert (augmented[:, :, :-1] != 0).any()

    # the heun solver runs a second denoise on every non-terminal step: 2 steps -> first non-terminal
    # gets 2 calls, terminal gets 1 -> 3 total.
    heun = _make_sampler(solver="heun")
    calls = 0

    def counting(inputs: CoordinateDenoisingInput) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros_like(inputs.noisy_coords)

    heun(counting, context=context, num_samples=1, num_sampling_steps=2, initial_coords=torch.ones(1, 1, 5, 3))
    assert calls == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_samples": 2, "num_sampling_steps": 1, "initial_coords": torch.zeros(1, 3, 3)}, "initial_coords"),
        ({"num_samples": 0}, "num_samples"),
        ({"sample_chunk_size": 0}, "sample_chunk_size"),
    ],
)
def test_invalid_inputs(context: CoordinateDenoiseContext, kwargs: dict, match: str) -> None:
    sampler = CoordinateEdmSampler()
    with pytest.raises(ValueError, match=match):
        sampler(_identity, context=context, **kwargs)
