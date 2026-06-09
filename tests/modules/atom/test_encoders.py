from __future__ import annotations

import pytest
import torch

from ofoldx.modules.atom import AtomConditioner, AtomPairEncoder, PackedAtomAttention


@pytest.fixture
def pair_encoder() -> AtomPairEncoder:
    return AtomPairEncoder(atom_hidden_size=8, atom_pair_hidden_size=4, num_query_atoms=2, num_key_atoms=4)


def test_conditioner() -> None:
    conditioner = AtomConditioner(atom_feature_size=5, atom_hidden_size=8)
    # distinct per-element features/positions/charges so no projection is degenerate
    features = torch.randn(1, 3, 5)
    positions = torch.randn(1, 3, 3)
    charges = torch.randn(1, 3, 1)
    mask = torch.tensor([[True, False, True]])
    output = conditioner(features, reference_positions=positions, reference_charges=charges, reference_mask=mask)

    # conditioning is multiplied by reference_mask, so a masked atom is exactly zero
    assert output.shape == (1, 3, 8)
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))

    # the conditioner sums three independent bias-free linear projections, so superposition must hold
    zero_features, zero_positions, zero_charges = torch.zeros(1, 3, 5), torch.zeros(1, 3, 3), torch.zeros(1, 3, 1)
    only_features = conditioner(
        features, reference_positions=zero_positions, reference_charges=zero_charges, reference_mask=mask
    )
    only_positions = conditioner(
        zero_features, reference_positions=positions, reference_charges=zero_charges, reference_mask=mask
    )
    only_charges = conditioner(
        zero_features, reference_positions=zero_positions, reference_charges=charges, reference_mask=mask
    )
    torch.testing.assert_close(output, only_features + only_positions + only_charges)

    # per-atom conditioning is independent across atoms: perturbing the (masked) atom 1 leaves atoms 0 and 2 exact
    perturbed_features = features.clone()
    perturbed_features[:, 1] = torch.randn(5)
    perturbed_positions = positions.clone()
    perturbed_positions[:, 1] = torch.randn(3)
    perturbed = conditioner(
        perturbed_features, reference_positions=perturbed_positions, reference_charges=charges, reference_mask=mask
    )
    torch.testing.assert_close(perturbed[:, [0, 2]], output[:, [0, 2]])


@pytest.mark.parametrize("num_bias_samples", [1, 2, 3])
def test_pair_encoder(pair_encoder: AtomPairEncoder, num_bias_samples: int) -> None:
    positions = torch.randn(2, 5, 3)
    local_geometry = pair_encoder.build_local_geometry(positions[:1], torch.zeros(1, 5, dtype=torch.long))

    # dense layout is (batch, atom, atom, pair); local layout windows query/key into (batch, windows, q, k, pair)
    assert pair_encoder(torch.zeros(1, 5, 8), reference_positions=positions[:1]).shape == (1, 5, 5, 4)
    assert pair_encoder(torch.zeros(1, 5, 8), local_geometry=local_geometry).shape == (1, 3, 2, 4, 4)

    # the dense path ReLUs the atom-single term, so all-negative conditioning contributes identically to zeros
    negative = -torch.abs(torch.randn(2, 5, 8))
    torch.testing.assert_close(
        pair_encoder(negative, reference_positions=positions),
        pair_encoder(torch.zeros(2, 5, 8), reference_positions=positions),
    )

    # geometry is shared across samples, so only the added pair bias varies: zero-bias sample 0 matches no-bias
    pair_bias = torch.zeros(2, num_bias_samples, 5, 5, 4)
    for sample in range(1, num_bias_samples):
        pair_bias[:, sample] = torch.randn(2, 5, 5, 4)
    output = pair_encoder(torch.zeros(2, 5, 8), reference_positions=positions, pair=pair_bias)
    assert output.shape == pair_bias.shape
    no_bias = pair_encoder(torch.zeros(2, 5, 8), reference_positions=positions)
    torch.testing.assert_close(output[:, 0], no_bias)
    for sample in range(1, num_bias_samples):
        assert not torch.allclose(output[:, 0], output[:, sample])


def test_invalid_inputs(pair_encoder: AtomPairEncoder) -> None:
    with pytest.raises(ValueError, match="reference_positions is required"):
        pair_encoder(torch.zeros(1, 5, 8))

    attention = PackedAtomAttention(
        atom_hidden_size=8,
        conditioning_size=8,
        num_blocks=1,
        num_attention_heads=2,
        num_query_atoms=2,
        num_key_atoms=4,
        operator_backend="sdpa",
    )
    with pytest.raises(ValueError, match="unsupported attention_layout"):
        attention(torch.zeros(1, 5, 8), torch.zeros(1, 5, 8), torch.zeros(1, 5, 5, 1), attention_layout="bogus")
