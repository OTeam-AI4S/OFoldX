from __future__ import annotations

from typing import Any

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenEncoder,
    AtomToTokenOutput,
    CoordinateAtomToTokenEncoder,
    TokenToAtomDecoder,
)


class TestAtomConditioner:
    def test_mask(self) -> None:
        conditioner = AtomConditioner(atom_feature_size=5, atom_hidden_size=8)
        output = conditioner(
            torch.randn(1, 3, 5),
            reference_positions=torch.randn(1, 3, 3),
            reference_charges=torch.randn(1, 3, 1),
            reference_mask=torch.tensor([[True, False, True]]),
        )

        assert output.shape == (1, 3, 8)
        assert torch.count_nonzero(output[:, 1]) == 0

    def test_superposition(self) -> None:
        conditioner = AtomConditioner(atom_feature_size=5, atom_hidden_size=8)
        features = torch.randn(1, 3, 5)
        positions = torch.randn(1, 3, 3)
        charges = torch.randn(1, 3, 1)
        mask = torch.ones(1, 3, dtype=torch.bool)
        zeros = torch.zeros(1, 3, 1)

        output = conditioner(features, reference_positions=positions, reference_charges=charges, reference_mask=mask)
        features_only = conditioner(
            features,
            reference_positions=zeros.expand(-1, -1, 3),
            reference_charges=zeros,
            reference_mask=mask,
        )
        positions_only = conditioner(
            torch.zeros_like(features),
            reference_positions=positions,
            reference_charges=zeros,
            reference_mask=mask,
        )
        charges_only = conditioner(
            torch.zeros_like(features),
            reference_positions=zeros.expand(-1, -1, 3),
            reference_charges=charges,
            reference_mask=mask,
        )

        torch.testing.assert_close(output, features_only + positions_only + charges_only)


class TestAtomPairEncoder:
    def test_pair_bias(self) -> None:
        encoder = AtomPairEncoder(atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4)
        geometry = encoder.build_local_geometry(torch.randn(2, 5, 3), torch.zeros(2, 5, dtype=torch.long))
        conditioning = torch.zeros(2, 5, 8)
        pair_bias = torch.zeros(2, 3, 3, 2, 4, 4)
        pair_bias[:, 1:] = torch.randn(2, 2, 3, 2, 4, 4)

        output = encoder(conditioning, local_geometry=geometry, pair=pair_bias)
        without_bias = encoder(conditioning, local_geometry=geometry)

        assert output.shape == pair_bias.shape
        torch.testing.assert_close(output[:, 0], without_bias)
        assert not torch.allclose(output[:, 0], output[:, 1])


class TestCoordinateAtomToTokenEncoder:
    def test_forward_and_gradients(self) -> None:
        batch_size, num_samples, num_atoms, num_tokens = 2, 3, 5, 3
        atom_features = torch.randn(batch_size, num_atoms, 6, requires_grad=True)
        reference_positions = torch.randn(batch_size, num_atoms, 3)
        atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        single = torch.randn(batch_size, 1, num_tokens, 6)
        encoder = CoordinateAtomToTokenEncoder(
            token_hidden_size=8,
            atom_hidden_size=8,
            atom_pair_hidden_size=4,
            single_size=6,
            pair_size=5,
            num_blocks=1,
            num_attention_heads=2,
            num_query_atoms=2,
            num_key_atoms=4,
            atom_feature_size=6,
            backend="eager",
        )
        local_geometry = encoder.build_local_geometry(
            reference_positions,
            torch.zeros(batch_size, num_atoms, dtype=torch.long),
            atom_mask,
        )

        output = encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=torch.randn(batch_size, num_atoms, 1),
            reference_mask=atom_mask,
            single=single,
            pair=torch.randn(batch_size, 1, num_tokens, num_tokens, 5),
            noisy_coords=torch.randn(batch_size, num_samples, num_atoms, 3),
            local_geometry=local_geometry,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            token_reference=single,
        )
        output.token_hidden.square().mean().backward()

        assert output.token_hidden.shape == (batch_size, num_samples, num_tokens, 8)
        assert output.atom_query.shape == (batch_size, num_samples, num_atoms, 8)
        assert atom_features.grad is not None
        assert torch.isfinite(atom_features.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps NestedTensor kernels require CUDA")
class TestNestedAtomToTokenEncoder:
    def test_vertical_parity_vjp_and_dynamic_fullgraph(self) -> None:
        torch.manual_seed(20260828)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        encoder = AtomToTokenEncoder(
            token_hidden_size=32,
            atom_hidden_size=32,
            atom_pair_hidden_size=16,
            num_blocks=1,
            num_attention_heads=4,
            num_query_atoms=4,
            num_key_atoms=8,
            atom_feature_size=6,
            backend="oops",
        ).to(device=device, dtype=dtype)

        first = _nested_atom_encoder_case((7, 11), (4, 6), device=device, dtype=dtype)
        with _nested_guard():
            actual = _run_nested_atom_encoder(encoder, first)
        expected = _run_independent_atom_encoders(encoder, first)
        expected_token = torch.cat(tuple(output.token_hidden.squeeze(0) for output in expected))
        expected_query = torch.cat(tuple(output.atom_query.squeeze(0) for output in expected))
        expected_pair = torch.cat(tuple(output.atom_pair.squeeze(0) for output in expected))
        torch.testing.assert_close(actual.token_hidden.concat, expected_token, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual.atom_query.concat, expected_query, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual.atom_pair.concat, expected_pair, rtol=5e-2, atol=5e-2)

        leaves = (*first["leaves"], *tuple(encoder.parameters()))
        grad_token = torch.randn_like(actual.token_hidden.concat)
        with _nested_guard():
            actual_gradients = torch.autograd.grad(
                actual.token_hidden.concat,
                leaves,
                grad_token,
                retain_graph=True,
                allow_unused=True,
            )
        expected_gradients = torch.autograd.grad(
            expected_token,
            leaves,
            grad_token,
            allow_unused=True,
        )
        _assert_vertical_gradients(actual_gradients, expected_gradients, activation_count=3)

        counter = CompileCounter()

        def execute(atom_features, positions, charges, mask, spaces, mapping, token_reference):
            case = {
                "atom_features": atom_features,
                "positions": positions,
                "charges": charges,
                "mask": mask,
                "spaces": spaces,
                "mapping": mapping,
                "token_reference": token_reference,
            }
            return _run_nested_atom_encoder(encoder, case).token_hidden.concat

        compiled = torch.compile(execute, backend=counter, fullgraph=True, dynamic=True)
        for atom_lengths, token_lengths in (((7, 11), (4, 6)), ((9, 14), (5, 7))):
            case = _nested_atom_encoder_case(atom_lengths, token_lengths, device=device, dtype=dtype)
            operands = tuple(
                case[name]
                for name in (
                    "atom_features",
                    "positions",
                    "charges",
                    "mask",
                    "spaces",
                    "mapping",
                    "token_reference",
                )
            )
            with _nested_guard():
                output = compiled(*operands)
                gradients = torch.autograd.grad(
                    output.float().square().sum(),
                    (*case["leaves"], *tuple(encoder.parameters())),
                    allow_unused=True,
                )
            assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
        assert counter.frame_count == 1

    def test_coordinate_vertical_masked_windows_and_dynamic_fullgraph(self) -> None:
        torch.manual_seed(20260830)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        encoder = CoordinateAtomToTokenEncoder(
            token_hidden_size=32,
            atom_hidden_size=32,
            atom_pair_hidden_size=16,
            single_size=32,
            pair_size=16,
            num_blocks=1,
            num_attention_heads=4,
            num_query_atoms=2,
            num_key_atoms=4,
            atom_feature_size=6,
            backend="oops",
        ).to(device=device, dtype=dtype)

        def make_case(atom_lengths, token_lengths, masks):
            def nested(parts, ragged_dims, *, requires_grad=False):
                device_parts = tuple(part.to(device=device).requires_grad_(requires_grad) for part in parts)
                return NestedTensor(device_parts, ragged_dims=ragged_dims)

            atom_features = nested(
                tuple(torch.randn(length, 6, dtype=dtype) for length in atom_lengths),
                (0,),
                requires_grad=True,
            )
            positions = nested(
                tuple(torch.randn(length, 3, dtype=dtype) for length in atom_lengths),
                (0,),
                requires_grad=True,
            )
            charges = nested(
                tuple(torch.randn(length, 1, dtype=dtype) for length in atom_lengths),
                (0,),
                requires_grad=True,
            )
            atom_mask = nested(masks, (0,))
            spaces = nested(tuple(torch.arange(length) // 3 for length in atom_lengths), (0,))
            mapping = nested(
                tuple(
                    (torch.arange(atoms) * tokens // atoms).clamp_max(tokens - 1)
                    for atoms, tokens in zip(atom_lengths, token_lengths, strict=True)
                ),
                (0,),
            )
            single = nested(
                tuple(torch.randn(length, 32, dtype=dtype) for length in token_lengths),
                (0,),
                requires_grad=True,
            )
            pair = nested(
                tuple(torch.randn(length, length, 16, dtype=dtype) for length in token_lengths),
                (0, 1),
                requires_grad=True,
            )
            noisy = nested(
                tuple(torch.randn(length, 3, dtype=dtype) for length in atom_lengths),
                (0,),
                requires_grad=True,
            )
            return atom_features, positions, charges, atom_mask, spaces, mapping, single, pair, noisy

        def run(case):
            atom_features, positions, charges, atom_mask, spaces, mapping, single, pair, noisy = case
            geometry = encoder.build_local_geometry(positions, spaces, atom_mask)
            return encoder(
                atom_features,
                reference_positions=positions,
                reference_charges=charges,
                reference_mask=atom_mask,
                single=single,
                pair=pair,
                noisy_coords=noisy,
                local_geometry=geometry,
                atom_mask=atom_mask,
                atom_to_token=mapping,
                token_reference=single,
            )

        first = make_case(
            (5, 7),
            (3, 4),
            (
                torch.tensor([True, False, True, True, False]),
                torch.zeros(7, dtype=torch.bool),
            ),
        )
        with _nested_guard():
            actual = run(first)
        expected = []
        for elements in zip(*(torch.unbind(tensor) for tensor in first), strict=True):
            atom_features, positions, charges, atom_mask, spaces, mapping, single, pair, noisy = elements
            geometry = encoder.build_local_geometry(
                positions.unsqueeze(0),
                spaces.unsqueeze(0),
                atom_mask.unsqueeze(0),
            )
            expected.append(
                encoder(
                    atom_features.unsqueeze(0),
                    reference_positions=positions.unsqueeze(0),
                    reference_charges=charges.unsqueeze(0),
                    reference_mask=atom_mask.unsqueeze(0),
                    single=single.unsqueeze(0),
                    pair=pair.unsqueeze(0),
                    noisy_coords=noisy.unsqueeze(0),
                    local_geometry=geometry,
                    atom_mask=atom_mask.unsqueeze(0),
                    atom_to_token=mapping.unsqueeze(0),
                    token_reference=single.unsqueeze(0),
                )
            )
        expected_token = torch.cat(tuple(output.token_hidden.squeeze(0) for output in expected))
        expected_query = torch.cat(tuple(output.atom_query.squeeze(0) for output in expected))
        torch.testing.assert_close(actual.token_hidden.concat, expected_token, rtol=5e-2, atol=5e-2)
        valid_atoms = first[3].concat
        torch.testing.assert_close(
            actual.atom_query.concat[valid_atoms],
            expected_query[valid_atoms],
            rtol=5e-2,
            atol=5e-2,
        )
        activation_leaves = tuple(first[index].concat for index in (0, 1, 2, 6, 7, 8))
        leaves = (*activation_leaves, *tuple(encoder.parameters()))
        grad_token = torch.randn_like(actual.token_hidden.concat)
        with _nested_guard():
            actual_gradients = torch.autograd.grad(
                actual.token_hidden.concat,
                leaves,
                grad_token,
                retain_graph=True,
                allow_unused=True,
            )
        expected_gradients = torch.autograd.grad(
            expected_token,
            leaves,
            grad_token,
            allow_unused=True,
        )
        _assert_vertical_gradients(actual_gradients, expected_gradients, activation_count=6)

        counter = CompileCounter()

        def compiled_run(*case):
            return run(case).token_hidden.concat

        compiled = torch.compile(compiled_run, backend=counter, fullgraph=True, dynamic=True)
        cases = (
            first,
            make_case(
                (6, 9),
                (4, 5),
                (
                    torch.tensor([True, True, False, True, False, False]),
                    torch.tensor([True, False, True, True, True, False, False, False, False]),
                ),
            ),
        )
        for case in cases:
            with _nested_guard():
                output = compiled(*case)
            assert torch.isfinite(output).all()
        assert counter.frame_count == 1


class TestTokenToAtomDecoder:
    def test_forward_and_gradients(self) -> None:
        batch_size, num_samples, num_atoms, num_tokens = 2, 3, 5, 3
        atom_query = torch.randn(batch_size, num_samples, num_atoms, 8, requires_grad=True)
        atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        atom_output = AtomToTokenOutput(
            token_hidden=torch.empty(0),
            atom_query=atom_query,
            atom_pair=torch.randn(batch_size, 1, 3, 2, 4, 4),
            atom_conditioning=torch.randn(batch_size, 1, num_atoms, 8),
        )
        decoder = TokenToAtomDecoder(
            token_hidden_size=8,
            atom_hidden_size=8,
            atom_pair_hidden_size=4,
            num_blocks=1,
            num_attention_heads=2,
            num_query_atoms=2,
            num_key_atoms=4,
            backend="eager",
        )

        output = decoder(
            atom_output,
            torch.randn(batch_size, num_samples, num_tokens, 8),
            atom_to_token=atom_to_token,
            atom_mask=atom_mask,
            key_index=None,
        )
        output.square().mean().backward()

        assert output.shape == (batch_size, num_samples, num_atoms, 3)
        assert atom_query.grad is not None
        assert torch.isfinite(atom_query.grad).all()


def _nested_atom_encoder_case(
    atom_lengths: tuple[int, int],
    token_lengths: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    reference = NestedTensor([torch.empty(1, 6), torch.empty(1, 6)], ragged_dims=(0,))
    lengths = torch.tensor(atom_lengths, dtype=torch.int64)
    total_atoms = sum(atom_lengths)

    def atoms(values: torch.Tensor) -> NestedTensor:
        return reference.packed_with_lengths(values, lengths)

    feature_values = torch.randn(total_atoms, 6, device=device, dtype=dtype, requires_grad=True)
    position_values = torch.randn(total_atoms, 3, device=device, dtype=dtype, requires_grad=True)
    charge_values = torch.randn(total_atoms, 1, device=device, dtype=dtype, requires_grad=True)
    mask_values = torch.ones(total_atoms, device=device, dtype=torch.bool)
    space_values = torch.cat(
        tuple(
            torch.arange(atoms_count, device=device, dtype=torch.long) // max(atoms_count // 2, 1)
            for atoms_count in atom_lengths
        )
    )
    mapping_values = torch.cat(
        tuple(
            torch.div(
                torch.arange(atoms_count, device=device, dtype=torch.long) * tokens,
                atoms_count,
                rounding_mode="floor",
            ).clamp_max(tokens - 1)
            for atoms_count, tokens in zip(atom_lengths, token_lengths, strict=True)
        )
    )
    token_reference = reference.packed_with_lengths(
        torch.empty(sum(token_lengths), 3, device=device, dtype=dtype),
        torch.tensor(token_lengths, dtype=torch.int64),
    )
    return {
        "atom_lengths": atom_lengths,
        "token_lengths": token_lengths,
        "atom_features": atoms(feature_values),
        "positions": atoms(position_values),
        "charges": atoms(charge_values),
        "mask": atoms(mask_values),
        "spaces": atoms(space_values),
        "mapping": atoms(mapping_values),
        "token_reference": token_reference,
        "leaves": (feature_values, position_values, charge_values),
    }


def _run_nested_atom_encoder(encoder: AtomToTokenEncoder, case: dict[str, Any]) -> AtomToTokenOutput:
    geometry = encoder.build_local_geometry(case["positions"], case["spaces"], case["mask"])
    return encoder(
        case["atom_features"],
        reference_positions=case["positions"],
        reference_charges=case["charges"],
        reference_mask=case["mask"],
        local_geometry=geometry,
        atom_mask=case["mask"],
        atom_to_token=case["mapping"],
        token_reference=case["token_reference"],
    )


def _run_independent_atom_encoders(
    encoder: AtomToTokenEncoder,
    case: dict[str, Any],
) -> tuple[AtomToTokenOutput, ...]:
    atom_lengths = case["atom_lengths"]
    token_lengths = case["token_lengths"]
    parts = tuple(
        case[name].concat.split(atom_lengths)
        for name in ("atom_features", "positions", "charges", "mask", "spaces", "mapping")
    )
    token_parts = case["token_reference"].concat.split(token_lengths)
    outputs = []
    for features, positions, charges, mask, spaces, mapping, token_reference in zip(
        *parts,
        token_parts,
        strict=True,
    ):
        geometry = encoder.build_local_geometry(
            positions.unsqueeze(0),
            spaces.unsqueeze(0),
            mask.unsqueeze(0),
        )
        outputs.append(
            encoder(
                features.unsqueeze(0),
                reference_positions=positions.unsqueeze(0),
                reference_charges=charges.unsqueeze(0),
                reference_mask=mask.unsqueeze(0),
                local_geometry=geometry,
                atom_mask=mask.unsqueeze(0),
                atom_to_token=mapping.unsqueeze(0),
                token_reference=token_reference.unsqueeze(0),
            )
        )
    return tuple(outputs)


def _assert_vertical_gradients(
    actual: tuple[torch.Tensor | None, ...],
    expected: tuple[torch.Tensor | None, ...],
    *,
    activation_count: int,
) -> None:
    parameter_difference_square = 0.0
    parameter_reference_square = 0.0
    for index, (actual_gradient, expected_gradient) in enumerate(zip(actual, expected, strict=True)):
        assert (actual_gradient is None) is (expected_gradient is None)
        if actual_gradient is None or expected_gradient is None:
            continue
        difference = (actual_gradient.float() - expected_gradient.float()).norm()
        reference = torch.maximum(actual_gradient.float().norm(), expected_gradient.float().norm()).clamp_min(1e-7)
        if index < activation_count:
            assert float(difference / reference) < 0.08
        else:
            parameter_difference_square += float(difference.square())
            parameter_reference_square += float(reference.square())
    assert (parameter_difference_square / max(parameter_reference_square, 1e-14)) ** 0.5 < 0.08


def _nested_guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )
