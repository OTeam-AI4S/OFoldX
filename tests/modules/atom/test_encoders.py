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

from math import prod
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenEncoder,
    ClampedLocalAtomWindowing,
    CoordinateAtomToTokenEncoder,
    LocalAtomAttention,
    LocalAtomWindowing,
    TokenToAtomDecoder,
    TrunkToAtomConditioning,
)
from ofoldx.modules.attention import PerBlockPairBiasProjection
from ofoldx.modules.transformer import (
    AtomCrossAttentionTransformerEncoderBlock,
    ConditionedTransformerEncoderBlock,
    NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
)
from tests.modules import require_oops
from tests.modules.test_transformer import (
    _compare_complete_vjps,
    _conditioned_block_reference,
    _learned_parameters,
)

_WINDOWINGS = pytest.mark.parametrize(
    "windowing_type",
    [LocalAtomWindowing, ClampedLocalAtomWindowing],
    ids=["centred", "clamped"],
)


@pytest.mark.parametrize("prefix", [(), (2,), (2, 3), (2, 2)])
def test_zero_block_atom_attention_preserves_sampled_identity_and_input_vjp(prefix: tuple[int, ...]) -> None:
    attention = LocalAtomAttention(0, 2, 8, 4, backend="eager", block=ConditionedTransformerEncoderBlock)
    hidden = torch.arange(5 * 8 * max(1, prod(prefix)), dtype=torch.float64).reshape(*prefix, 5, 8)
    hidden = hidden.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    shared_prefix = prefix[:1]
    mask = torch.ones(*shared_prefix, 5, dtype=torch.bool)
    window = LocalAtomWindowing(2, 4).window_info(mask)
    conditioning = torch.zeros(*shared_prefix, 5, 4, dtype=hidden.dtype, requires_grad=True)
    bias = torch.empty(*shared_prefix, 3, 2, 4, 0, dtype=hidden.dtype, requires_grad=True)
    output = attention(hidden, conditioning, bias, mask, window)
    torch.testing.assert_close(output, hidden)
    coefficients = torch.linspace(0.2, 1.8, output.numel(), dtype=output.dtype).reshape_as(output)
    gradients = torch.autograd.grad((output * coefficients).sum(), (hidden, conditioning, bias), allow_unused=True)
    torch.testing.assert_close(gradients[0], coefficients)
    for gradient, operand in zip(gradients[1:], (conditioning, bias), strict=True):
        actual = torch.zeros_like(operand) if gradient is None else gradient
        torch.testing.assert_close(actual, torch.zeros_like(operand), rtol=0, atol=0)


class TestAtomConditioner:
    def test_unavailable_reference_gradients(self) -> None:
        model = AtomConditioner(atom_feature_size=2, atom_hidden_size=4)
        positions = torch.tensor([[[1.0, 2.0, 3.0], [float("nan"), float("inf"), 0.0]]], requires_grad=True)
        output = model(
            torch.ones(1, 2, 2),
            reference_positions=positions,
            reference_charges=torch.zeros(1, 2, 1),
            reference_mask=torch.tensor([[True, False]]),
        )
        assert torch.isfinite(output).all()
        gradient = torch.autograd.grad(output.square().sum(), positions)[0]
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[:, 1], torch.zeros(1, 3))

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
    def test_equation(self) -> None:
        encoder = AtomPairEncoder(atom_hidden_size=1, atom_pair_hidden_size=1)
        with torch.no_grad():
            encoder.distance_vector_projection.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
            encoder.inverse_distance_projection.weight.fill_(3.0)
            encoder.distance_mask_projection.weight.fill_(4.0)
            encoder.left_atom_projection.weight.fill_(5.0)
            encoder.right_atom_projection.weight.fill_(-2.0)
            encoder.feed_forward.input_projection.weight.fill_(0.5)
            encoder.feed_forward.intermediate_projections[0].weight.fill_(2.0)
            encoder.feed_forward.output_projection.weight.fill_(0.25)
        geometry = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(
            torch.randn(1, 5, 3),
            torch.tensor([[0, 0, 1, 1, 1]]),
            torch.tensor([[True, True, False, True, True]]),
        )
        states = torch.randn(1, 5, 1)
        pair = torch.randn(1, 3, 2, 4, 1)

        output = encoder(states, geometry, pair)

        query, key = geometry.window_info.gather(states)
        distance_vectors = geometry.distance_vectors
        valid = geometry.distance_mask
        inverse_distance = 1 / (1 + distance_vectors.square().sum(dim=-1, keepdim=True))
        state = (distance_vectors * torch.tensor([1.0, 2.0, -1.0])).sum(dim=-1, keepdim=True) * valid
        state = state + 3 * inverse_distance * valid + 4 * valid
        state = state + 5 * F.relu(query).unsqueeze(-2) - 2 * F.relu(key).unsqueeze(-3) + pair
        update = 2 * F.relu(0.5 * F.relu(state))
        expected = state + 0.25 * F.relu(update)

        torch.testing.assert_close(output, expected)

    def test_pair_input_prefix(self) -> None:
        encoder = AtomPairEncoder(atom_hidden_size=8, atom_pair_hidden_size=4)
        geometry = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(
            torch.randn(2, 5, 3),
            torch.zeros(2, 5, dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        )
        states = torch.zeros(2, 5, 8)
        pair = torch.zeros(2, 3, 3, 2, 4, 4)
        pair[:, 1:] = torch.randn(2, 2, 3, 2, 4, 4)

        output = encoder(states, geometry, pair)
        without_pair = encoder(states, geometry)

        assert output.shape == pair.shape
        torch.testing.assert_close(output[:, 0], without_pair)
        assert not torch.allclose(output[:, 0], output[:, 1])

    def test_sampled_states_share_geometry(self) -> None:
        encoder = AtomPairEncoder(atom_hidden_size=8, atom_pair_hidden_size=4)
        geometry = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(
            torch.randn(2, 5, 3),
            torch.zeros(2, 5, dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        )
        states = torch.randn(2, 3, 5, 8)

        output = encoder(states, geometry)

        assert output.shape == (2, 3, 3, 2, 4, 4)
        for sample in range(3):
            torch.testing.assert_close(output[:, sample], encoder(states[:, sample], geometry))


class TestTrunkToAtomConditioning:
    def test_conditioning(self) -> None:
        conditioning = TrunkToAtomConditioning(6, 5, 8, 4, backend="eager")
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        window_info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(
            torch.ones(2, 5, dtype=torch.bool)
        )
        atom_states = torch.randn(2, 5, 8)
        single = torch.randn(2, 3, 6)
        pair = torch.randn(2, 3, 3, 5)

        atom_conditioning, local_pair = conditioning(atom_states, single, pair, atom_to_token, window_info)

        projected = conditioning.single_projection(conditioning.single_norm(single))
        expected = atom_states + projected.gather(1, atom_to_token.unsqueeze(-1).expand(-1, -1, 8))
        torch.testing.assert_close(atom_conditioning, expected)
        assert local_pair.shape == (2, 3, 2, 4, 4)

    def test_sampled_single_inserts_sample_axis(self) -> None:
        conditioning = TrunkToAtomConditioning(6, 5, 8, 4, backend="eager")
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        window_info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(
            torch.ones(2, 5, dtype=torch.bool)
        )
        atom_states = torch.randn(2, 5, 8)
        single = torch.randn(2, 3, 3, 6)
        pair = torch.randn(2, 3, 3, 5)

        atom_conditioning, _ = conditioning(atom_states, single, pair, atom_to_token, window_info)

        assert atom_conditioning.shape == (2, 3, 5, 8)
        for sample in range(3):
            expected, _ = conditioning(atom_states, single[:, sample], pair, atom_to_token, window_info)
            torch.testing.assert_close(atom_conditioning[:, sample], expected)


class TestLocalAtomAttention:
    @_WINDOWINGS
    @pytest.mark.parametrize("num_queries,num_keys,num_atoms", [(4, 4, 16), (4, 4, 7), (4, 8, 11)])
    @pytest.mark.parametrize("sampled", [False, True])
    @pytest.mark.parametrize(
        "block_type,key_mode",
        [(ConditionedTransformerEncoderBlock, "self"), (AtomCrossAttentionTransformerEncoderBlock, "raw")],
    )
    def test_complete_learned_window_equation_and_all_named_vjps(
        self, windowing_type, num_queries, num_keys, num_atoms, sampled, block_type, key_mode
    ) -> None:
        generator = torch.Generator().manual_seed(932)
        attention = LocalAtomAttention(1, 2, 8, 6, "eager", block=block_type).double().eval()
        parameters = _learned_parameters(attention, generator)
        prefix = (1, 3) if sampled else (1,)
        num_windows = (num_atoms + num_queries - 1) // num_queries
        inputs = (
            torch.randn(*prefix, num_atoms, 8, dtype=torch.float64, generator=generator).requires_grad_(),
            torch.randn(*prefix, num_atoms, 6, dtype=torch.float64, generator=generator).requires_grad_(),
            torch.randn(
                *prefix, num_windows, num_queries, num_keys, 2, dtype=torch.float64, generator=generator
            ).requires_grad_(),
        )
        expected_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        mask = torch.ones(1, num_atoms, dtype=torch.bool)
        mask[:, 2::4] = False
        window = windowing_type(num_queries, num_keys).window_info(mask)
        actual = attention(*inputs, mask, window)
        expected = _local_atom_attention_reference(
            *expected_inputs,
            mask,
            parameters,
            num_queries=num_queries,
            num_keys=num_keys,
            clamped=windowing_type is ClampedLocalAtomWindowing,
            key_mode=key_mode,
        )
        _compare_complete_vjps(actual, expected, inputs, expected_inputs, attention, parameters, generator)

    @_WINDOWINGS
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual packed atom-window kernels require CUDA")
    def test_true_ragged_equal_width_windows_complete_learned_values_and_vjps(self, windowing_type) -> None:
        generator = torch.Generator(device="cuda").manual_seed(949)
        attention = (
            LocalAtomAttention(1, 2, 8, 6, "eager", block=ConditionedTransformerEncoderBlock).cuda().double().eval()
        )
        parameters = _learned_parameters(attention, generator)
        # Two actual ragged batch elements produce four windows. They do not
        # introduce a Monte Carlo sample axis into either element.
        lengths = (5, 7)
        hidden = [
            torch.randn(length, 8, device="cuda", dtype=torch.float64, generator=generator).requires_grad_()
            for length in lengths
        ]
        conditioning = [
            torch.randn(length, 6, device="cuda", dtype=torch.float64, generator=generator).requires_grad_()
            for length in lengths
        ]
        biases = [
            torch.randn(2, 4, 4, 2, device="cuda", dtype=torch.float64, generator=generator).requires_grad_()
            for _ in lengths
        ]
        masks = [torch.arange(length, device="cuda") % 4 != 2 for length in lengths]
        operands = (*hidden, *conditioning, *biases)
        expected_operands = tuple(value.detach().clone().requires_grad_() for value in operands)
        packed_hidden = NestedTensor(hidden, ragged_dims=(0,))
        packed_conditioning = NestedTensor(conditioning, ragged_dims=(0,))
        packed_bias = NestedTensor(biases, ragged_dims=(0,))
        packed_mask = NestedTensor(masks, ragged_dims=(0,))
        window = windowing_type(4, 4).window_info(packed_mask)
        actual = attention(packed_hidden, packed_conditioning, packed_bias, packed_mask, window).concat
        expected = torch.cat(
            [
                _local_atom_attention_reference(
                    expected_operands[index].unsqueeze(0),
                    expected_operands[index + 2].unsqueeze(0),
                    expected_operands[index + 4].unsqueeze(0),
                    masks[index].unsqueeze(0),
                    parameters,
                    num_queries=4,
                    num_keys=4,
                    clamped=windowing_type is ClampedLocalAtomWindowing,
                ).squeeze(0)
                for index in range(2)
            ]
        )
        _compare_complete_vjps(actual, expected, operands, expected_operands, attention, parameters, generator)

    @_WINDOWINGS
    def test_masked_atoms_are_not_keys(self, windowing_type: type[LocalAtomWindowing]) -> None:
        torch.manual_seed(0)
        attention = LocalAtomAttention(1, 2, 8, 8, "eager", block=AtomCrossAttentionTransformerEncoderBlock)
        atom_mask = torch.tensor([[True, True, False, True, True, True]])
        window_info = windowing_type(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
        hidden = torch.randn(1, 6, 8)
        conditioning = torch.randn(1, 6, 8)
        bias = torch.randn(1, 3, 2, 4, 2)
        perturbed_hidden = hidden.clone()
        perturbed_hidden[:, 2] = torch.randn(8)
        perturbed_conditioning = conditioning.clone()
        perturbed_conditioning[:, 2] = torch.randn(8)

        output = attention(hidden, conditioning, bias, atom_mask, window_info)
        perturbed = attention(perturbed_hidden, perturbed_conditioning, bias, atom_mask, window_info)

        torch.testing.assert_close(output[:, atom_mask[0]], perturbed[:, atom_mask[0]])

    def test_sample_axis_matches_samples(self) -> None:
        torch.manual_seed(1)
        attention = LocalAtomAttention(1, 2, 8, 8, "eager", block=AtomCrossAttentionTransformerEncoderBlock)
        atom_mask = torch.ones(2, 5, dtype=torch.bool)
        window_info = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
        hidden = torch.randn(2, 3, 5, 8)
        conditioning = torch.randn(2, 5, 8)
        bias = torch.randn(2, 3, 2, 4, 2)

        output = attention(hidden, conditioning, bias, atom_mask, window_info)

        assert output.shape == hidden.shape
        for sample in range(3):
            torch.testing.assert_close(
                output[:, sample],
                attention(hidden[:, sample], conditioning, bias, atom_mask, window_info),
            )


def _local_atom_attention_reference(
    hidden, conditioning, bias, mask, parameters, *, num_queries, num_keys, clamped, key_mode="self"
):
    """Independent atom-index windows followed by the complete learned block equation."""
    prefix = hidden.shape[:-2]
    num_atoms = hidden.shape[-2]
    num_windows = (num_atoms + num_queries - 1) // num_queries
    padded_atoms = num_windows * num_queries
    hidden = F.pad(hidden, (0, 0, 0, padded_atoms - num_atoms)).reshape(*prefix, num_windows, num_queries, 8)
    conditioning = F.pad(conditioning, (0, 0, 0, padded_atoms - num_atoms)).reshape(
        *prefix, num_windows, num_queries, 6
    )
    validity = mask.reshape(*mask.shape[:-1], *((1,) * (len(prefix) - mask.ndim + 1)), num_atoms).expand(
        *prefix, num_atoms
    )
    validity = F.pad(validity, (0, padded_atoms - num_atoms))
    starts = torch.arange(num_windows, device=mask.device) * num_queries - (num_keys - num_queries) // 2
    if clamped:
        extent = torch.where(validity, torch.arange(padded_atoms, device=mask.device) + 1, 0).amax(dim=-1)
        starts = torch.minimum(starts.clamp_min(0), (extent - num_keys).clamp_min(0).unsqueeze(-1))
    else:
        starts = starts.expand(*prefix, num_windows)
    indices = starts.unsqueeze(-1) + torch.arange(num_keys, device=mask.device)
    present = (indices >= 0) & (indices < padded_atoms)
    safe = indices.clamp(0, padded_atoms - 1).reshape(-1, num_windows * num_keys)
    key_validity = (
        torch.gather(validity.reshape(-1, padded_atoms), 1, safe).reshape(*prefix, num_windows, num_keys) & present
    )

    def gather(value):
        values = value.reshape(-1, padded_atoms, value.shape[-1])
        selected = torch.gather(values, 1, safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        return selected.reshape(*prefix, num_windows, num_keys, values.shape[-1]) * present.unsqueeze(-1)

    result = _conditioned_block_reference(
        hidden,
        conditioning,
        bias,
        key_validity.unsqueeze(-2),
        parameters,
        prefix="transformer.blocks.0",
        gather=gather,
        key_mode=key_mode,
    )
    return result.flatten(-3, -2)[..., :num_atoms, :]


class TestAtomToTokenEncoder:
    def test_forward_and_gradients(self) -> None:
        torch.manual_seed(2)
        batch_size, num_atoms, num_tokens = 2, 5, 3
        encoder = AtomToTokenEncoder(
            8,
            6,
            1,
            2,
            "eager",
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
        )
        atom_mask = torch.tensor([[True, True, True, True, False], [True] * 5])
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        window_info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
        atom_states = torch.randn(batch_size, num_atoms, 8, requires_grad=True)
        bias = torch.randn(batch_size, 3, 2, 4, 2, requires_grad=True)
        token_reference = torch.empty(batch_size, num_tokens, 1)

        token = encoder(atom_states, bias, atom_mask, atom_to_token, token_reference, window_info)
        token.square().mean().backward()

        assert token.shape == (batch_size, num_tokens, 6)
        assert atom_states.grad is not None and torch.isfinite(atom_states.grad).all()
        assert bias.grad is not None and torch.isfinite(bias.grad).all()

        # A masked atom contributes to no token: neither to the mean of its own token nor as a key.
        perturbed = atom_states.detach().clone()
        perturbed[0, 4] = torch.randn(8)
        torch.testing.assert_close(
            encoder(perturbed, bias.detach(), atom_mask, atom_to_token, token_reference, window_info),
            token.detach(),
        )


class TestCoordinateAtomToTokenEncoder:
    def test_forward_and_gradients(self) -> None:
        torch.manual_seed(3)
        batch_size, num_samples, num_atoms, num_tokens = 2, 3, 5, 3
        encoder = CoordinateAtomToTokenEncoder(8, 8, 1, 2, "eager", block=AtomCrossAttentionTransformerEncoderBlock)
        atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        window_info = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
        atom_conditioning = torch.randn(batch_size, num_atoms, 8, requires_grad=True)
        bias = torch.randn(batch_size, 3, 2, 4, 2, requires_grad=True)
        noisy = torch.randn(batch_size, num_samples, num_atoms, 3, requires_grad=True)
        token_reference = torch.empty(batch_size, num_tokens, 1)

        token, atom_query = encoder(
            noisy,
            atom_conditioning,
            atom_conditioning,
            bias,
            atom_mask,
            atom_to_token,
            token_reference,
            window_info,
        )
        token.square().mean().backward()

        assert token.shape == (batch_size, num_samples, num_tokens, 8)
        assert atom_query.shape == (batch_size, num_samples, num_atoms, 8)
        for leaf in (noisy, atom_conditioning, bias):
            assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
        for sample in range(num_samples):
            sample_token, sample_query = encoder(
                noisy[:, sample : sample + 1],
                atom_conditioning,
                atom_conditioning,
                bias,
                atom_mask,
                atom_to_token,
                token_reference,
                window_info,
            )
            torch.testing.assert_close(sample_token[:, 0], token[:, sample])
            torch.testing.assert_close(sample_query[:, 0], atom_query[:, sample])


class TestTokenToAtomDecoder:
    def test_forward_and_gradients(self) -> None:
        torch.manual_seed(4)
        batch_size, num_samples, num_atoms, num_tokens = 2, 3, 5, 3
        decoder = TokenToAtomDecoder(
            8,
            8,
            1,
            2,
            "eager",
            block=AtomCrossAttentionTransformerEncoderBlock,
            bias=False,
        )
        atom_query = torch.randn(batch_size, num_samples, num_atoms, 8, requires_grad=True)
        atom_mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
        atom_to_token = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 2, 2]])
        window_info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)

        output = decoder(
            torch.randn(batch_size, num_samples, num_tokens, 8),
            atom_query,
            torch.randn(batch_size, 1, num_atoms, 8),
            torch.randn(batch_size, 1, 3, 2, 4, 2),
            atom_mask,
            atom_to_token,
            window_info,
        )
        output.square().mean().backward()

        assert output.shape == (batch_size, num_samples, num_atoms, 3)
        assert atom_query.grad is not None
        assert torch.isfinite(atom_query.grad).all()


class _ReferenceAtomStack(nn.Module):
    r"""The reference-atom stack of the input embeddings: windowing, conditioning, pair bias, and pooling."""

    def __init__(self, windowing: LocalAtomWindowing, backend: str) -> None:
        super().__init__()
        self.windowing = windowing
        self.atom_conditioner = AtomConditioner(6, 32)
        self.atom_pair_encoder = AtomPairEncoder(32, 16)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=16,
            num_blocks=1,
            num_attention_heads=4,
            bias=False,
            backend=backend,
        )
        self.atom_encoder = AtomToTokenEncoder(32, 32, 1, 4, backend, block=AtomCrossAttentionTransformerEncoderBlock)

    def forward(
        self,
        atom_features: Tensor | NestedTensor,
        positions: Tensor | NestedTensor,
        charges: Tensor | NestedTensor,
        mask: Tensor | NestedTensor,
        spaces: Tensor | NestedTensor,
        mapping: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
    ) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor]:
        geometry = self.windowing(positions, spaces, mask)
        conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=positions,
            reference_charges=charges,
            reference_mask=mask,
        )
        atom_pair = self.atom_pair_encoder(conditioning, geometry)
        token = self.atom_encoder(
            conditioning,
            self.atom_pair_bias_projection(atom_pair),
            mask,
            mapping,
            token_reference,
            geometry.window_info,
        )
        return token, atom_pair


class _CoordinateAtomStack(nn.Module):
    r"""The atom encoder path of a coordinate denoiser: windowing, trunk conditioning, pair bias, and pooling."""

    def __init__(self, windowing: LocalAtomWindowing, backend: str) -> None:
        super().__init__()
        self.windowing = windowing
        self.atom_conditioner = AtomConditioner(6, 32)
        self.trunk_conditioning = TrunkToAtomConditioning(32, 16, 32, 16, bias=False, backend=backend)
        self.atom_pair_encoder = AtomPairEncoder(32, 16)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=16,
            num_blocks=1,
            num_attention_heads=4,
            bias=False,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            32,
            32,
            1,
            4,
            backend,
            block=AtomCrossAttentionTransformerEncoderBlock,
        )

    def forward(
        self,
        atom_features: Tensor | NestedTensor,
        positions: Tensor | NestedTensor,
        charges: Tensor | NestedTensor,
        mask: Tensor | NestedTensor,
        spaces: Tensor | NestedTensor,
        mapping: Tensor | NestedTensor,
        single: Tensor | NestedTensor,
        pair: Tensor | NestedTensor,
        noisy: Tensor | NestedTensor,
    ) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor]:
        geometry = self.windowing(positions, spaces, mask)
        conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=positions,
            reference_charges=charges,
            reference_mask=mask,
        )
        conditioning, local_pair = self.trunk_conditioning(conditioning, single, pair, mapping, geometry.window_info)
        atom_pair = self.atom_pair_encoder(conditioning, geometry, local_pair)
        return self.atom_encoder(
            noisy,
            conditioning,
            conditioning,
            self.atom_pair_bias_projection(atom_pair),
            mask,
            mapping,
            single,
            geometry.window_info,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps NestedTensor kernels require CUDA")
class TestNestedAtomStacks:
    @_WINDOWINGS
    @pytest.mark.parametrize("backend", ["auto", "oops"])
    def test_reference_vertical_parity_vjp_and_dynamic_fullgraph(
        self, windowing_type: type[LocalAtomWindowing], backend: str
    ) -> None:
        pytest.importorskip("oops")
        if backend == "oops":
            require_oops("adaptive_layer_norm", "norm_projection")
        torch.manual_seed(20260828)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        stack = _ReferenceAtomStack(windowing_type(num_query_atoms=4, num_key_atoms=8), backend).to(
            device=device, dtype=dtype
        )

        first = _nested_atom_encoder_case((7, 11), (4, 6), device=device, dtype=dtype)
        actual_token, actual_pair = _run_nested(stack, first)
        expected = _run_independent(stack, first)
        expected_token = torch.cat(tuple(token.squeeze(0) for token, _ in expected))
        expected_pair = torch.cat(tuple(pair.squeeze(0) for _, pair in expected))
        torch.testing.assert_close(actual_token.concat, expected_token, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual_pair.concat, expected_pair, rtol=5e-2, atol=5e-2)

        leaves = (*first["leaves"], *tuple(stack.parameters()))
        grad_token = torch.randn_like(actual_token.concat)
        actual_gradients = torch.autograd.grad(
            actual_token.concat,
            leaves,
            grad_token,
            retain_graph=True,
            allow_unused=True,
        )
        expected_gradients = torch.autograd.grad(expected_token, leaves, grad_token, allow_unused=True)
        _assert_vertical_gradients(actual_gradients, expected_gradients, activation_count=3)

        counter = CompileCounter()

        def execute(*operands: NestedTensor) -> Tensor:
            return stack(*operands)[0].concat

        compiled = torch.compile(execute, backend=counter, fullgraph=True, dynamic=True)
        for atom_lengths, token_lengths in (((7, 11), (4, 6)), ((9, 14), (5, 7))):
            case = _nested_atom_encoder_case(atom_lengths, token_lengths, device=device, dtype=dtype)
            output = compiled(*(case[name] for name in _STACK_OPERANDS))
            gradients = torch.autograd.grad(
                output.float().square().sum(),
                (*case["leaves"], *tuple(stack.parameters())),
                allow_unused=True,
            )
            assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
        assert counter.frame_count == 1

    @_WINDOWINGS
    @pytest.mark.parametrize("backend", ["auto", "oops"])
    def test_coordinate_vertical_masked_windows_and_dynamic_fullgraph(
        self,
        windowing_type: type[LocalAtomWindowing],
        backend: str,
    ) -> None:
        pytest.importorskip("oops")
        if backend == "oops":
            require_oops("adaptive_layer_norm", "norm_projection")
        torch.manual_seed(20260830)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        stack = _CoordinateAtomStack(windowing_type(num_query_atoms=2, num_key_atoms=4), backend).to(
            device=device, dtype=dtype
        )

        def make_case(atom_lengths, token_lengths, masks):
            activation_leaves = []

            def nested(parts, ragged_dims, *, requires_grad=False):
                carrier = NestedTensor(tuple(part.to(device=device) for part in parts), ragged_dims=ragged_dims)
                if not requires_grad:
                    return carrier
                values = carrier.concat.detach().requires_grad_()
                activation_leaves.append(values)
                return carrier.packed_like(values)

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
            return (atom_features, positions, charges, atom_mask, spaces, mapping, single, pair, noisy), tuple(
                activation_leaves
            )

        first, activation_leaves = make_case(
            (5, 7),
            (3, 4),
            (
                torch.tensor([True, False, True, True, False]),
                torch.zeros(7, dtype=torch.bool),
            ),
        )
        actual_token, actual_query = stack(*first)
        expected = [
            stack(*(element.unsqueeze(0) for element in elements))
            for elements in zip(*(torch.unbind(tensor) for tensor in first), strict=True)
        ]
        expected_token = torch.cat(tuple(token.squeeze(0) for token, _ in expected))
        expected_query = torch.cat(tuple(query.squeeze(0) for _, query in expected))
        torch.testing.assert_close(actual_token.concat, expected_token, rtol=5e-2, atol=5e-2)
        valid_atoms = first[3].concat
        torch.testing.assert_close(
            actual_query.concat[valid_atoms],
            expected_query[valid_atoms],
            rtol=5e-2,
            atol=5e-2,
        )
        leaves = (*activation_leaves, *tuple(stack.parameters()))
        grad_token = torch.randn_like(actual_token.concat)
        actual_gradients = torch.autograd.grad(
            actual_token.concat,
            leaves,
            grad_token,
            retain_graph=True,
            allow_unused=True,
        )
        expected_gradients = torch.autograd.grad(expected_token, leaves, grad_token, allow_unused=True)
        _assert_vertical_gradients(actual_gradients, expected_gradients, activation_count=6)

        counter = CompileCounter()

        def compiled_run(*case: NestedTensor) -> Tensor:
            return stack(*case)[0].concat

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
            )[0],
        )
        for case in cases:
            output = compiled(*case)
            assert torch.isfinite(output).all()
        assert counter.frame_count == 1


_STACK_OPERANDS = ("atom_features", "positions", "charges", "mask", "spaces", "mapping", "token_reference")


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


def _run_nested(
    stack: _ReferenceAtomStack,
    case: dict[str, Any],
) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor]:
    return stack(*(case[name] for name in _STACK_OPERANDS))


def _run_independent(
    stack: _ReferenceAtomStack,
    case: dict[str, Any],
) -> tuple[tuple[Tensor | NestedTensor, Tensor | NestedTensor], ...]:
    atom_lengths = case["atom_lengths"]
    token_lengths = case["token_lengths"]
    parts = tuple(case[name].concat.split(atom_lengths) for name in _STACK_OPERANDS[:-1])
    token_parts = case["token_reference"].concat.split(token_lengths)
    return tuple(
        stack(*(part.unsqueeze(0) for part in row), token_reference.unsqueeze(0))
        for *row, token_reference in zip(*parts, token_parts, strict=True)
    )


def _assert_vertical_gradients(
    actual: tuple[torch.Tensor | None, ...],
    expected: tuple[torch.Tensor | None, ...],
    *,
    activation_count: int,
) -> None:
    parameter_difference_square = 0.0
    parameter_reference_square = 0.0
    for index, (actual_gradient, expected_gradient) in enumerate(zip(actual, expected, strict=True)):
        if actual_gradient is None and expected_gradient is None:
            continue
        if actual_gradient is None:
            actual_gradient = torch.zeros_like(expected_gradient)
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(actual_gradient)
        difference = (actual_gradient.float() - expected_gradient.float()).norm()
        reference = torch.maximum(actual_gradient.float().norm(), expected_gradient.float().norm()).clamp_min(1e-7)
        if index < activation_count:
            assert float(difference / reference) < 0.08
        else:
            parameter_difference_square += float(difference.square())
            parameter_reference_square += float(reference.square())
    assert (parameter_difference_square / max(parameter_reference_square, 1e-14)) ** 0.5 < 0.08
