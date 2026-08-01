from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard

from ofoldx.modules.atom import LocalAtomGeometryBuilder
from ofoldx.modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoisingInput,
    CoordinateEdmSampler,
    EdmScheduler,
    denoise_coordinates,
)
from ofoldx.modules.embedding.position import generate_relative_position_features


def _make_context(
    batch_size: int = 1,
    *,
    device: torch.device | str = "cpu",
) -> CoordinateDenoiseContext:
    reference_positions = torch.zeros(batch_size, 5, 3, device=device)
    single = torch.zeros(batch_size, 3, 6, device=device)
    pair = torch.zeros(batch_size, 3, 3, 4, device=device)
    return CoordinateDenoiseContext(
        atom_features=torch.zeros(batch_size, 5, 5, device=device),
        atom_to_token=torch.tensor([[0, 0, 1, 2, 2]], device=device).expand(batch_size, -1),
        reference_positions=reference_positions,
        reference_charges=torch.zeros(batch_size, 5, 1, device=device),
        reference_mask=torch.ones(batch_size, 5, dtype=torch.bool, device=device),
        atom_mask=torch.ones(batch_size, 5, dtype=torch.bool, device=device),
        token_embedding=torch.zeros(batch_size, 3, 8, device=device),
        single=single,
        conditioning_single=single,
        conditioning_pair=pair,
        relative_features=generate_relative_position_features(
            residue_index=torch.arange(3, device=device).expand(batch_size, -1),
            chain_index=torch.zeros(batch_size, 3, dtype=torch.long, device=device),
            entity_index=torch.zeros(batch_size, 3, dtype=torch.long, device=device),
            symmetry_index=torch.zeros(batch_size, 3, dtype=torch.long, device=device),
            token_index=torch.arange(3, device=device).expand(batch_size, -1),
            dtype=torch.float32,
        ),
        local_geometry=LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)(
            reference_positions,
            torch.zeros(batch_size, 5, dtype=torch.long, device=device),
        ),
    )


def _make_ragged_context(atom_lengths: tuple[int, ...] = (4, 6)) -> CoordinateDenoiseContext:
    context = _make_context(batch_size=len(atom_lengths))
    reference_positions = NestedTensor(
        [torch.zeros(num_atoms, 3) for num_atoms in atom_lengths],
        ragged_dims=(0,),
    )
    atom_mask = NestedTensor(
        [torch.ones(num_atoms, dtype=torch.bool) for num_atoms in atom_lengths],
        ragged_dims=(0,),
    )
    return replace(
        context,
        atom_features=NestedTensor(
            [torch.zeros(num_atoms, 5) for num_atoms in atom_lengths],
            ragged_dims=(0,),
        ),
        atom_to_token=NestedTensor(
            [torch.zeros(num_atoms, dtype=torch.long) for num_atoms in atom_lengths],
            ragged_dims=(0,),
        ),
        reference_positions=reference_positions,
        reference_charges=NestedTensor(
            [torch.zeros(num_atoms, 1) for num_atoms in atom_lengths],
            ragged_dims=(0,),
        ),
        reference_mask=atom_mask,
        atom_mask=atom_mask,
    )


def _forbid_ragged_materialization():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
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


@pytest.fixture(autouse=True)
def reset_compiler_cache() -> None:
    # Independent sampler configurations must not share Dynamo's recompile budget.
    torch.compiler.reset()


class TestCoordinateDenoiseContext:
    def test_step_input(self, context: CoordinateDenoiseContext) -> None:
        coords = torch.randn(1, 2, 5, 3)
        step_input = context.to_denoising_input(coords, torch.ones(1, 2))

        assert isinstance(step_input, CoordinateDenoisingInput)
        torch.testing.assert_close(step_input.noisy_coords, coords)


class TestCoordinateEdmSampler:
    def test_shape(self, context: CoordinateDenoiseContext) -> None:
        output = _make_sampler()(_zeros, context=context, num_samples=2, num_sampling_steps=1)

        assert output.shape == (1, 2, 5, 3)
        assert output.dtype == torch.float32

    def test_fixed_point(self, context: CoordinateDenoiseContext) -> None:
        initial = torch.ones(1, 2, 5, 3)

        output = _make_sampler()(
            _identity,
            context=context,
            num_samples=2,
            num_sampling_steps=1,
            initial_coords=initial,
        )

        assert torch.equal(output, initial)

    def test_initial_broadcast(self, context: CoordinateDenoiseContext) -> None:
        initial = torch.ones(1, 5, 3)

        output = _make_sampler()(
            _identity,
            context=context,
            num_samples=2,
            num_sampling_steps=1,
            initial_coords=initial,
        )

        assert output.shape == (1, 2, 5, 3)
        assert torch.equal(output[:, 0], initial)
        assert torch.equal(output[:, 1], initial)

    def test_seed(self, context: CoordinateDenoiseContext) -> None:
        first = _make_sampler()(
            _zeros,
            context=context,
            num_samples=2,
            num_sampling_steps=1,
            generator=torch.Generator().manual_seed(7),
        )
        second = _make_sampler()(
            _zeros,
            context=context,
            num_samples=2,
            num_sampling_steps=1,
            generator=torch.Generator().manual_seed(7),
        )

        assert torch.equal(first, second)

    def test_dynamic_compile(self, context: CoordinateDenoiseContext) -> None:
        compiled = torch.compile(_make_sampler(), backend="eager", fullgraph=True, dynamic=True)

        output = compiled(_zeros, context=context, num_samples=2, num_sampling_steps=1)

        assert output.shape == (1, 2, 5, 3)
        assert output.isfinite().all()

    def test_ragged_initialization_stays_packed(self) -> None:
        context = _make_ragged_context()

        with _forbid_ragged_materialization():
            output = _make_sampler()(
                _identity,
                context=context,
                num_samples=2,
                num_sampling_steps=1,
                sample_chunk_size=1,
                generator=torch.Generator().manual_seed(7),
            )

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[2, 4, 3], [2, 6, 3]]
        assert output.ragged_dims == (1,)
        assert output.concat.isfinite().all()
        for element in output:
            assert not torch.equal(element[0], element[1])

    def test_ragged_initialization_fullgraph(self) -> None:
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(_make_sampler(), backend=counter, fullgraph=True, dynamic=True)
        for atom_lengths in ((4, 6), (5, 7)):
            context = _make_ragged_context(atom_lengths)
            with _forbid_ragged_materialization():
                output = compiled(
                    _zeros,
                    context=context,
                    num_samples=2,
                    num_sampling_steps=1,
                    sample_chunk_size=1,
                )

            assert isinstance(output, NestedTensor)
            assert output.element_sizes().tolist() == [
                [2, atom_lengths[0], 3],
                [2, atom_lengths[1], 3],
            ]
            assert output.ragged_dims == (1,)
            assert output.concat.isfinite().all()
        assert counter.frame_count == 1

    def test_ragged_default_augmentation_fullgraph(self) -> None:
        context = _make_ragged_context()
        sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1),
            churn_rate=0.0,
            step_scale=1.0,
            center_random_augmentation=True,
        )
        compiled = torch.compile(sampler, backend="eager", fullgraph=True, dynamic=True)

        with _forbid_ragged_materialization():
            output = compiled(
                _zeros,
                context=context,
                num_samples=2,
                num_sampling_steps=1,
                sample_chunk_size=1,
            )

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[2, 4, 3], [2, 6, 3]]
        assert output.ragged_dims == (1,)
        assert output.concat.isfinite().all()

    @pytest.mark.parametrize("sampled_condition", [True, False])
    @pytest.mark.parametrize("augmentation", [False, True])
    def test_ragged_initial_and_sampled_condition_fullgraph(self, sampled_condition: bool, augmentation: bool) -> None:
        atom_lengths = (4, 6)
        context = _make_ragged_context(atom_lengths)
        initial_coords = NestedTensor(
            [torch.randn(num_atoms, 3) for num_atoms in atom_lengths],
            ragged_dims=(0,),
        )
        condition_coords = NestedTensor(
            [torch.randn((2, num_atoms, 3) if sampled_condition else (num_atoms, 3)) for num_atoms in atom_lengths],
            ragged_dims=(1,) if sampled_condition else (0,),
        )
        sampler = _make_sampler()
        sampler.center_random_augmentation = augmentation
        compiled = torch.compile(sampler, backend="eager", fullgraph=True, dynamic=True)

        with _forbid_ragged_materialization():
            output = compiled(
                _identity,
                context=context,
                num_samples=2,
                num_sampling_steps=1,
                sample_chunk_size=1,
                initial_coords=initial_coords,
                condition_coords=condition_coords,
                condition_mask=context.atom_mask,
            )

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[2, 4, 3], [2, 6, 3]]
        assert output.ragged_dims == (1,)
        expected = condition_coords.concat
        if not sampled_condition:
            expected = expected.unsqueeze(1).expand(-1, 2, -1)
        torch.testing.assert_close(output.concat, expected)

    @pytest.mark.parametrize("sampled_coords", [False, True])
    def test_ragged_augmentation_roundtrip(self, sampled_coords: bool) -> None:
        atom_lengths = (4, 6)
        context = _make_ragged_context(atom_lengths)
        initial = [torch.randn((3, num_atoms, 3) if sampled_coords else (num_atoms, 3)) for num_atoms in atom_lengths]
        condition = [coords + 2 for coords in initial]
        masks = [torch.arange(num_atoms) % 2 == 0 for num_atoms in atom_lengths]
        sampler = _make_sampler(solver="heun")
        sampler.center_random_augmentation = True
        sampler.reverse_augmentation = True
        compiled = torch.compile(sampler, backend="eager", fullgraph=True, dynamic=True)
        ragged_dims = (1,) if sampled_coords else (0,)
        initial_coords = NestedTensor(initial, ragged_dims=ragged_dims)
        condition_coords = NestedTensor(condition, ragged_dims=ragged_dims)
        condition_mask = NestedTensor(masks, ragged_dims=(0,))

        with _forbid_ragged_materialization():
            output = compiled(
                _identity,
                context=context,
                num_samples=3,
                num_sampling_steps=2,
                sample_chunk_size=2,
                initial_coords=initial_coords,
                condition_coords=condition_coords,
                condition_mask=condition_mask,
            )

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[3, 4, 3], [3, 6, 3]]
        assert output.ragged_dims == (1,)
        for actual, start, fixed, mask in zip(output, initial, condition, masks, strict=True):
            expected = torch.where(mask[..., None], fixed, start)
            if not sampled_coords:
                expected = expected.unsqueeze(0).expand(3, -1, -1)
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("augmentation", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_ragged_augmentation_geometry(self, augmentation: bool, masked: bool) -> None:
        parts = [torch.randn(num_atoms, 3, dtype=torch.float64, requires_grad=True) for num_atoms in (4, 6)]
        masks = [
            torch.arange(len(part)) % 2 == 0 if masked else torch.ones(len(part), dtype=torch.bool) for part in parts
        ]
        coords = NestedTensor(parts, ragged_dims=(0,))
        atom_mask = NestedTensor(masks, ragged_dims=(0,))
        compiled = torch.compile(_make_sampler().augment_target, backend="eager", fullgraph=True, dynamic=True)

        with _forbid_ragged_materialization():
            output = compiled(
                coords,
                num_samples=3,
                atom_mask=atom_mask if masked else None,
                augmentation=augmentation,
                translation_scale=0.0,
            )
            loss = (output * atom_mask.unsqueeze(-2).unsqueeze(-1)).concat.square().sum()
            gradients = torch.autograd.grad(loss, parts)

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[3, 4, 3], [3, 6, 3]]
        assert output.ragged_dims == (1,)
        for actual, gradient, part, mask in zip(output, gradients, parts, masks, strict=True):
            observed = actual[:, mask]
            centered = part[mask] - part[mask].mean(dim=0)
            torch.testing.assert_close(observed.mean(dim=-2), torch.zeros(3, 3, dtype=part.dtype))
            distances = torch.cdist(part[mask], part[mask]).expand(3, -1, -1)
            torch.testing.assert_close(torch.cdist(observed, observed), distances)
            if not augmentation:
                torch.testing.assert_close(observed, centered.expand(3, -1, -1))
            elif masked:
                torch.testing.assert_close(actual[:, ~mask], torch.zeros_like(actual[:, ~mask]))
            expected_gradient = torch.zeros_like(part)
            expected_gradient[mask] = 6 * centered
            torch.testing.assert_close(gradient, expected_gradient)

    def test_chunking(self, context: CoordinateDenoiseContext) -> None:
        sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1),
            churn_rate=0.5,
            center_random_augmentation=True,
        )
        kwargs = {
            "context": context,
            "num_samples": 4,
            "num_sampling_steps": 2,
        }

        unchunked = sampler(_zeros, **kwargs, generator=torch.Generator().manual_seed(11))
        chunked = sampler(
            _zeros,
            **kwargs,
            sample_chunk_size=2,
            generator=torch.Generator().manual_seed(11),
        )

        assert torch.equal(chunked, unchunked)

    def test_conditioning(self) -> None:
        context = _make_context(batch_size=2)
        condition_coords = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)

        output = _make_sampler()(
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

    def test_augmentation_mask(self) -> None:
        sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1),
            center_random_augmentation=True,
        )
        coords = torch.randn(2, 4, 3)
        atom_mask = torch.ones(2, 4, dtype=torch.bool)
        atom_mask[:, -1] = False

        output = sampler.augment_target(
            coords,
            num_samples=3,
            atom_mask=atom_mask,
            generator=torch.Generator().manual_seed(0),
        )

        assert output.shape == (2, 3, 4, 3)
        torch.testing.assert_close(output[:, :, -1], torch.zeros(2, 3, 3))

    def test_heun_equation(self, context: CoordinateDenoiseContext) -> None:
        sampler = _make_sampler(solver="heun")
        initial = torch.ones(1, 1, 5, 3)

        def half(inputs: CoordinateDenoisingInput) -> torch.Tensor:
            return 0.5 * inputs.noisy_coords

        output = sampler(half, context=context, num_samples=1, num_sampling_steps=2, initial_coords=initial)
        sigma_0, sigma_1, sigma_2 = sampler.scheduler(
            num_sampling_steps=2,
            device=initial.device,
            dtype=initial.dtype,
        )
        delta_0 = 0.5 * initial / sigma_0
        euler = initial + (sigma_1 - sigma_0) * delta_0
        delta_1 = 0.5 * euler / sigma_1
        heun = initial + (sigma_1 - sigma_0) * 0.5 * (delta_0 + delta_1)
        expected = heun + (sigma_2 - sigma_1) * 0.5 * heun / sigma_1

        torch.testing.assert_close(output, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_generator_device(self) -> None:
        context = _make_context(device="cuda")
        output = _make_sampler()(
            _zeros,
            context=context,
            num_samples=2,
            num_sampling_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(7),
        )
        augmented = CoordinateEdmSampler().augment_target(
            torch.zeros(1, 3, 3),
            num_samples=1,
            generator=torch.Generator(device="cuda").manual_seed(7),
        )

        assert output.is_cuda
        assert augmented.device.type == "cpu"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"num_samples": 2, "num_sampling_steps": 1, "initial_coords": torch.zeros(1, 3, 3)},
                "initial_coords",
                id="coordinates",
            ),
            pytest.param({"num_samples": 0}, "num_samples", id="samples"),
            pytest.param({"sample_chunk_size": 0}, "sample_chunk_size", id="chunk"),
        ],
    )
    def test_invalid_input(self, context: CoordinateDenoiseContext, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            CoordinateEdmSampler()(_identity, context=context, **kwargs)


class TestDenoiseCoordinates:
    def test_chunking(self) -> None:
        context = _make_context(batch_size=2)
        noisy_coords = torch.randn(2, 5, 5, 3)
        noise_level = torch.randn(2, 5)

        def run(sample_chunk_size: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            coords = noisy_coords.clone().requires_grad_()
            scale = torch.tensor(0.5, requires_grad=True)

            def denoise(inputs: CoordinateDenoisingInput) -> torch.Tensor:
                return scale * inputs.noisy_coords + inputs.noise_level[..., None, None]

            output = denoise_coordinates(
                denoise,
                context=context,
                noisy_coords=coords,
                noise_level=noise_level,
                sample_chunk_size=sample_chunk_size,
            )
            coords_grad, scale_grad = torch.autograd.grad(output.square().sum(), (coords, scale))
            return output, coords_grad, scale_grad

        expected = run(None)
        actual = run(2)

        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_invalid_chunk(self) -> None:
        with pytest.raises(ValueError, match="sample_chunk_size must be positive"):
            denoise_coordinates(
                _identity,
                context=_make_context(),
                noisy_coords=torch.randn(1, 2, 5, 3),
                noise_level=torch.randn(1, 2),
                sample_chunk_size=0,
            )
