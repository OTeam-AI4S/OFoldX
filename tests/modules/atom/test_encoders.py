from __future__ import annotations

import torch

from ofoldx.modules.atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenOutput,
    CoordinateAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedTokenToAtomDecoder,
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
            single=torch.randn(batch_size, 1, num_tokens, 6),
            pair=torch.randn(batch_size, 1, num_tokens, num_tokens, 5),
            noisy_coords=torch.randn(batch_size, num_samples, num_atoms, 3),
            local_geometry=local_geometry,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
        )
        output.token_hidden.square().mean().backward()

        assert output.token_hidden.shape == (batch_size, num_samples, num_tokens, 8)
        assert output.atom_query.shape == (batch_size, num_samples, num_atoms, 8)
        assert atom_features.grad is not None
        assert torch.isfinite(atom_features.grad).all()


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


class TestPackedDenoisingAtomToTokenEncoder:
    def test_multiplicity(self) -> None:
        encoder = PackedDenoisingAtomToTokenEncoder(
            atom_hidden_size=4,
            token_hidden_size=2,
            num_blocks=1,
            num_attention_heads=1,
            num_query_atoms=2,
            num_key_atoms=2,
        )
        atom_single = torch.randn(2, 2, 4)
        atom_conditioning = torch.randn(2, 2, 4)
        atom_pair_bias = torch.randn(2, 1, 2, 2, 1)
        atom_mask = torch.tensor([[True, True], [True, False]])
        atom_to_token = torch.tensor([[0, 1], [1, 0]])
        noisy_positions = torch.randn(4, 2, 3)

        batched = encoder(
            atom_single,
            atom_conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            2,
            noisy_positions,
            multiplicity=2,
        )
        separate = [
            encoder(
                atom_single[index : index + 1],
                atom_conditioning[index : index + 1],
                atom_pair_bias[index : index + 1],
                atom_mask[index : index + 1],
                atom_to_token[index : index + 1],
                2,
                noisy_positions[2 * index : 2 * index + 2],
                multiplicity=2,
            )
            for index in range(2)
        ]

        assert batched[0].shape == (4, 2, 4)
        for value_index, value in enumerate(batched):
            torch.testing.assert_close(value, torch.cat([part[value_index] for part in separate]))


class TestPackedTokenToAtomDecoder:
    def test_multiplicity(self) -> None:
        decoder = PackedTokenToAtomDecoder(
            token_hidden_size=2,
            atom_hidden_size=4,
            num_blocks=1,
            num_attention_heads=1,
            num_query_atoms=2,
            num_key_atoms=2,
        )
        token_repr = torch.randn(4, 2, 4)
        atom_skip = torch.randn(4, 2, 4, requires_grad=True)
        conditioning = torch.randn(4, 2, 4)
        atom_pair_bias = torch.randn(2, 1, 2, 2, 1)
        atom_mask = torch.tensor([[True, True], [True, False]])
        atom_to_token = torch.tensor([[0, 1], [1, 0]])

        batched = decoder(
            token_repr,
            atom_skip,
            conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            multiplicity=2,
        )
        separate = torch.cat(
            [
                decoder(
                    token_repr[2 * index : 2 * index + 2],
                    atom_skip[2 * index : 2 * index + 2],
                    conditioning[2 * index : 2 * index + 2],
                    atom_pair_bias[index : index + 1],
                    atom_mask[index : index + 1],
                    atom_to_token[index : index + 1],
                    multiplicity=2,
                )
                for index in range(2)
            ]
        )
        batched.square().mean().backward()

        assert batched.shape == (4, 2, 3)
        torch.testing.assert_close(batched, separate)
        assert atom_skip.grad is not None
