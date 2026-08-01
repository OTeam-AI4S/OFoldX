# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import cast

import pytest
import torch
from danling.tensors import NestedTensor
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.denoiser import CoordinateDenoiser, DenoisingTokenRefiner
from ofoldx.modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateDenoisingInput,
    CoordinateEdmSampler,
    EdmScheduler,
)
from ofoldx.modules.embedding.position import RelativePositionBuckets, generate_relative_position_features


def _coordinate_denoiser(backend: str = "eager") -> CoordinateDenoiser:
    return CoordinateDenoiser(
        num_encoder_blocks=1,
        num_encoder_attention_heads=2,
        sigma_data=1.0,
        atom_hidden_size=8,
        atom_pair_hidden_size=4,
        token_hidden_size=8,
        single_size=4,
        pair_size=3,
        token_feature_size=2,
        atom_feature_size=5,
        num_atom_encoder_blocks=1,
        num_atom_encoder_attention_heads=2,
        num_atom_decoder_blocks=1,
        num_atom_decoder_attention_heads=2,
        num_query_atoms=2,
        num_key_atoms=4,
        backend=backend,
    )


def _coordinate_context(
    model: CoordinateDenoiser,
    *,
    batch: int,
    atoms: int = 4,
    tokens: int = 2,
    requires_grad: bool = False,
) -> CoordinateDenoiseContext:
    reference_positions = torch.randn(batch, atoms, 3)
    atom_mask = torch.ones(batch, atoms, dtype=torch.bool)
    local_geometry = model.windowing(
        reference_positions,
        torch.zeros(batch, atoms, dtype=torch.long),
        atom_mask,
    )
    token_index = torch.arange(tokens).expand(batch, tokens)
    relative_features = generate_relative_position_features(
        residue_index=token_index,
        chain_index=torch.zeros_like(token_index),
        entity_index=torch.zeros_like(token_index),
        symmetry_index=torch.zeros_like(token_index),
        token_index=token_index,
        dtype=torch.float32,
    )
    return CoordinateDenoiseContext(
        atom_features=torch.randn(batch, atoms, 5, requires_grad=requires_grad),
        atom_to_token=torch.div(
            torch.arange(atoms) * tokens,
            atoms,
            rounding_mode="floor",
        ).expand(batch, atoms),
        token_embedding=torch.randn(batch, tokens, 2, requires_grad=requires_grad),
        single=torch.randn(batch, tokens, 4, requires_grad=requires_grad),
        conditioning_single=torch.randn(batch, tokens, 4, requires_grad=requires_grad),
        conditioning_pair=torch.randn(batch, tokens, tokens, 3, requires_grad=requires_grad),
        relative_features=relative_features,
        reference_positions=reference_positions,
        reference_charges=torch.randn(batch, atoms, 1),
        reference_mask=atom_mask,
        atom_mask=atom_mask,
        local_geometry=local_geometry,
    )


def _coordinate_inputs(
    model: CoordinateDenoiser,
    *,
    batch: int,
    samples: int,
    atoms: int = 4,
    tokens: int = 2,
    requires_grad: bool = False,
) -> CoordinateDenoisingInput:
    context = _coordinate_context(model, batch=batch, atoms=atoms, tokens=tokens, requires_grad=requires_grad)
    return context.to_denoising_input(
        noisy_coords=torch.randn(batch, samples, atoms, 3, requires_grad=requires_grad),
        noise_level=torch.linspace(0.5, 1.0, samples).expand(batch, samples),
    )


class TestCoordinateDenoiser:
    @pytest.mark.parametrize(
        "backend,device",
        [
            ("eager", "cpu"),
            ("eager", "cuda"),
            ("oops", "cuda"),
        ],
        ids=["shared-cpu", "shared-cuda", "shared-oops"],
    )
    def test_prepared_compile(self, backend: str, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Compiled CUDA denoising requires CUDA")
        if backend == "oops":
            pytest.importorskip("oops")
        torch.manual_seed(23)
        with torch.device(device):
            model = _coordinate_denoiser(backend).eval()
            context = _coordinate_context(model, batch=1)
            inputs = context.to_denoising_input(torch.randn(1, 2, 4, 3), torch.tensor([[0.5, 1.0]]))
        reference = deepcopy(model)
        sampler = CoordinateEdmSampler(
            scheduler=EdmScheduler(max_noise_level=1.0, min_noise_level=0.1), backend=backend
        )
        counter = CompileCounterWithBackend("aot_eager")
        if device == "cpu":
            model.compile(backend=counter, fullgraph=True)
        else:
            model.compile(fullgraph=True, mode="reduce-overhead")
        changed = replace(
            inputs,
            noisy_coords=inputs.noisy_coords + 0.25,
            noise_level=inputs.noise_level * 1.5,
        )
        next_context = replace(inputs, conditioning_single=inputs.conditioning_single.flip(-1))
        with torch.no_grad():
            get_rng_state = torch.get_rng_state if device == "cpu" else torch.cuda.get_rng_state
            rng_state = get_rng_state()
            denoise = model.prepare(inputs)
            first = denoise(inputs)
            if device == "cpu":
                assert counter.frame_count > 0
            retained = first.clone()
            torch.testing.assert_close(first, reference(inputs))
            torch.testing.assert_close(denoise(changed), reference(changed))
            torch.testing.assert_close(first, retained, rtol=0, atol=0)
            next_denoise = model.prepare(next_context)
            torch.testing.assert_close(next_denoise(next_context), reference(next_context))
            torch.testing.assert_close(denoise(inputs), retained)
            assert torch.equal(get_rng_state(), rng_state)
            eager_generator = torch.Generator(device=device).manual_seed(27)
            compiled_generator = torch.Generator(device=device).manual_seed(27)
            expected = sampler(
                reference.prepare(context),
                context=context,
                num_samples=2,
                num_sampling_steps=3,
                generator=eager_generator,
            )
            actual = sampler(
                model.prepare(context),
                context=context,
                num_samples=2,
                num_sampling_steps=3,
                generator=compiled_generator,
            )
            torch.testing.assert_close(actual, expected)
            assert torch.equal(compiled_generator.get_state(), eager_generator.get_state())

    def test_unavailable_noisy_gradients(self) -> None:
        model = _coordinate_denoiser()
        inputs = _coordinate_inputs(model, batch=1, samples=2)
        mask = torch.tensor([[True, True, False, True]])
        coordinates = inputs.noisy_coords.detach().clone()
        coordinates[..., 2, :] = float("nan")
        coordinates.requires_grad_()
        inputs = replace(
            inputs,
            atom_mask=mask,
            noisy_coords=coordinates,
            local_geometry=model.windowing(
                inputs.reference_positions, torch.zeros_like(mask, dtype=torch.long), mask, inputs.reference_mask
            ),
        )
        output = model(inputs)
        assert torch.isfinite(output).all()
        gradient = torch.autograd.grad(output.square().sum(), coordinates)[0]
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[..., 2, :], torch.zeros(1, 2, 3))

    def test_forward(self) -> None:
        torch.manual_seed(0)
        model = _coordinate_denoiser()
        inputs = _coordinate_inputs(model, batch=1, samples=2, requires_grad=True)

        output = model(inputs)
        output.square().mean().backward()

        assert output.shape == (1, 2, 4, 3)
        assert torch.isfinite(output).all()
        assert inputs.noisy_coords.grad is not None
        assert torch.isfinite(inputs.noisy_coords.grad).all()

    def test_fullgraph(self) -> None:
        torch.manual_seed(1)
        eager = _coordinate_denoiser()
        checkpointed = deepcopy(eager)
        eager_inputs = _coordinate_inputs(eager, batch=2, samples=4, requires_grad=True)
        checkpointed_inputs = replace(
            eager_inputs,
            atom_features=eager_inputs.atom_features.detach().clone().requires_grad_(),
            token_embedding=eager_inputs.token_embedding.detach().clone().requires_grad_(),
            single=eager_inputs.single.detach().clone().requires_grad_(),
            conditioning_single=eager_inputs.conditioning_single.detach().clone().requires_grad_(),
            conditioning_pair=eager_inputs.conditioning_pair.detach().clone().requires_grad_(),
            noisy_coords=eager_inputs.noisy_coords.detach().clone().requires_grad_(),
        )
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        eager_output = eager(eager_inputs)
        compiled = torch.compile(
            checkpointed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=True,
        )
        compiled_output = compiled(checkpointed_inputs)
        compiled_output.square().sum().backward()
        eager_output.square().sum().backward()

        torch.testing.assert_close(compiled_output, eager_output)
        for name in (
            "atom_features",
            "token_embedding",
            "single",
            "conditioning_single",
            "conditioning_pair",
            "noisy_coords",
        ):
            torch.testing.assert_close(getattr(checkpointed_inputs, name).grad, getattr(eager_inputs, name).grad)
        for actual, expected in zip(checkpointed.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps atom kernels require CUDA")
    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
    def test_sampled_ragged_forward_and_backward(self, compiled: bool) -> None:
        pytest.importorskip("oops")
        torch.manual_seed(17)
        device = torch.device("cuda")
        atom_lengths = (4, 6)
        token_lengths = (2, 3)
        samples = 2
        model = _coordinate_denoiser(backend="oops").to(device)
        reference_positions = NestedTensor(
            [torch.randn(atoms, 3, device=device) for atoms in atom_lengths],
        )
        atom_mask = NestedTensor(
            [torch.ones(atoms, dtype=torch.bool, device=device) for atoms in atom_lengths],
        )
        reference_space_id = NestedTensor(
            [torch.zeros(atoms, dtype=torch.long, device=device) for atoms in atom_lengths],
        )
        local_geometry = model.windowing(reference_positions, reference_space_id, atom_mask)
        atom_to_token = NestedTensor(
            [
                torch.arange(atoms, device=device).mul(tokens).div(atoms, rounding_mode="floor")
                for atoms, tokens in zip(atom_lengths, token_lengths, strict=True)
            ],
        )
        relative_features = RelativePositionBuckets(
            residue_bucket=NestedTensor(
                [torch.full((tokens, tokens), 32, dtype=torch.long, device=device) for tokens in token_lengths],
                ragged_dims=(0, 1),
            ),
            token_bucket=NestedTensor(
                [torch.full((tokens, tokens), 32, dtype=torch.long, device=device) for tokens in token_lengths],
                ragged_dims=(0, 1),
            ),
            same_entity=NestedTensor(
                [torch.ones(tokens, tokens, dtype=torch.bool, device=device) for tokens in token_lengths],
                ragged_dims=(0, 1),
            ),
            chain_bucket=NestedTensor(
                [torch.full((tokens, tokens), 2, dtype=torch.long, device=device) for tokens in token_lengths],
                ragged_dims=(0, 1),
            ),
            residue_size=66,
            chain_size=6,
            dtype=torch.float32,
        )
        noisy_leaves = [torch.randn(samples, atoms, 3, device=device, requires_grad=True) for atoms in atom_lengths]
        inputs = CoordinateDenoisingInput(
            atom_features=NestedTensor(
                [torch.randn(atoms, 5, device=device) for atoms in atom_lengths],
            ),
            atom_to_token=atom_to_token,
            token_embedding=NestedTensor(
                [torch.randn(tokens, 2, device=device) for tokens in token_lengths],
            ),
            single=NestedTensor(
                [torch.randn(tokens, 4, device=device) for tokens in token_lengths],
            ),
            conditioning_single=NestedTensor(
                [torch.randn(tokens, 4, device=device) for tokens in token_lengths],
            ),
            conditioning_pair=NestedTensor(
                [torch.randn(tokens, tokens, 3, device=device) for tokens in token_lengths],
                ragged_dims=(0, 1),
            ),
            relative_features=relative_features,
            noise_level=torch.tensor([[0.5, 1.0], [0.6, 1.1]], device=device),
            reference_positions=reference_positions,
            reference_charges=NestedTensor(
                [torch.randn(atoms, 1, device=device) for atoms in atom_lengths],
            ),
            reference_mask=atom_mask,
            atom_mask=atom_mask,
            noisy_coords=NestedTensor(noisy_leaves, ragged_dims=(1,)),
            local_geometry=local_geometry,
        )
        noisy_values = cast(NestedTensor, inputs.noisy_coords).concat.detach().requires_grad_()

        def run(denoising_input, values):
            denoising_input = replace(
                denoising_input,
                noisy_coords=denoising_input.noisy_coords.packed_like(values),
            )
            return model(denoising_input)

        if compiled:
            model.gradient_checkpointing = True
            model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        compiled_run = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True) if compiled else None
        if compiled_run is None:
            output = model(inputs)
        else:
            output = compiled_run(inputs, noisy_values)
        output_values = output.concat
        output_values.square().mean().backward()

        assert isinstance(output, NestedTensor)
        assert output.element_sizes().tolist() == [[2, 4, 3], [2, 6, 3]]
        assert output_values.shape == (10, 2, 3)
        assert torch.isfinite(output_values).all()
        if compiled:
            assert noisy_values.grad is not None and torch.isfinite(noisy_values.grad).all()
        else:
            assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in noisy_leaves)


class TestDenoisingTokenRefiner:
    def test_pair_broadcast(self) -> None:
        torch.manual_seed(2)
        shared = DenoisingTokenRefiner(
            num_blocks=1,
            token_hidden_size=8,
            single_size=4,
            pair_size=3,
            num_attention_heads=2,
            backend="eager",
        )
        explicit = deepcopy(shared)
        token_hidden = torch.randn(2, 3, 4, 8)
        single = torch.randn(2, 3, 4, 4)
        pair = torch.randn(2, 4, 4, 3, requires_grad=True)
        explicit_pair = pair.detach().clone().requires_grad_()

        shared_output = shared(token_hidden, single, pair)
        explicit_output = explicit(
            token_hidden,
            single,
            explicit_pair.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous(),
        )
        shared_output.square().sum().backward()
        explicit_output.square().sum().backward()

        torch.testing.assert_close(shared_output, explicit_output)
        torch.testing.assert_close(pair.grad, explicit_pair.grad)
