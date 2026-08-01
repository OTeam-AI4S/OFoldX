from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import cast

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.denoiser import CoordinateDenoiser, DenoisingTokenRefiner
from ofoldx.modules.diffusion import CoordinateDenoisingInput
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


def _coordinate_inputs(
    model: CoordinateDenoiser,
    *,
    batch: int,
    samples: int,
    atoms: int = 4,
    tokens: int = 2,
    requires_grad: bool = False,
) -> CoordinateDenoisingInput:
    reference_positions = torch.randn(batch, atoms, 3)
    atom_mask = torch.ones(batch, atoms, dtype=torch.bool)
    local_geometry = model.build_local_geometry(
        reference_positions,
        reference_space_id=torch.zeros(batch, atoms, dtype=torch.long),
        atom_mask=atom_mask,
    )
    noisy_coords = torch.randn(batch, samples, atoms, 3, requires_grad=requires_grad)
    token_index = torch.arange(tokens).expand(batch, tokens)
    relative_features = generate_relative_position_features(
        residue_index=token_index,
        chain_index=torch.zeros_like(token_index),
        entity_index=torch.zeros_like(token_index),
        symmetry_index=torch.zeros_like(token_index),
        token_index=token_index,
        dtype=torch.float32,
    )
    return CoordinateDenoisingInput(
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
        noise_level=torch.linspace(0.5, 1.0, samples).expand(batch, samples),
        reference_positions=reference_positions,
        reference_charges=torch.randn(batch, atoms, 1),
        reference_mask=atom_mask,
        atom_mask=atom_mask,
        noisy_coords=noisy_coords,
        local_geometry=local_geometry,
    )


class TestCoordinateDenoiser:
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
        torch.testing.assert_close(checkpointed_inputs.noisy_coords.grad, eager_inputs.noisy_coords.grad)

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
        local_geometry = model.build_local_geometry(reference_positions, reference_space_id, atom_mask)
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
        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
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
