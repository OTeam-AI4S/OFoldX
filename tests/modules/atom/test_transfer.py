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

from typing import TYPE_CHECKING

import pytest
import torch
from danling.tensors import NestedTensor
from torch._dynamo.testing import CompileCounter
from torch.nn import functional as F

from ofoldx.modules.atom import (
    AtomToTokenMean,
    ClampedLocalAtomWindowing,
    LocalAtomWindowing,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)

if TYPE_CHECKING:
    from torch import Tensor


def test_atom_to_token_mean() -> None:
    # Masked-out atom 1 drops from both sum and count; every token takes the exact mean of its valid atoms.
    atom_hidden = (
        torch.tensor(
            [[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]]],
        )
        .unsqueeze(1)
        .repeat(1, 2, 1, 1)
        .requires_grad_()
    )
    atom_to_token = torch.tensor([[0, 0, 1, 1]])
    atom_mask = torch.tensor([[True, False, True, True]])
    token_reference = torch.empty(1, 3, 1)

    expected = torch.tensor([[[[1.0, 2.0], [6.0, 12.0], [0.0, 0.0]]]]).repeat(1, 2, 1, 1)

    token_hidden = AtomToTokenMean()(
        atom_hidden,
        atom_to_token,
        token_reference,
        atom_mask,
    )
    torch.testing.assert_close(token_hidden, expected)
    token_hidden.sum().backward()
    torch.testing.assert_close(
        atom_hidden.grad,
        torch.tensor([[[[1.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.5]]]]).repeat(1, 2, 1, 1),
    )


def test_atom_to_token_mean_fp16_empty_token_is_finite() -> None:
    atom = torch.tensor([[[2.0], [4.0]]], dtype=torch.float16)
    mapping = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, False]])
    token_reference = torch.empty(1, 2, 1, dtype=torch.float16)

    output = AtomToTokenMean()(atom, mapping, token_reference, mask)

    torch.testing.assert_close(output, torch.tensor([[[2.0], [0.0]]], dtype=torch.float16))
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps atom transfer requires CUDA")
def test_sampled_oops_atom_to_token_mean_accepts_shared_mask() -> None:
    pytest.importorskip("oops")
    atom = torch.randn(2, 3, 5, 4, device="cuda", requires_grad=True)
    mapping = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], device="cuda")
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]], device="cuda")
    token_reference = torch.empty(2, 3, 1, device="cuda")

    actual = AtomToTokenMean(backend="oops")(atom, mapping, token_reference, mask)
    weights = F.one_hot(mapping, num_classes=3).to(atom) * mask.unsqueeze(-1)
    expected = torch.einsum("bat,bxac->bxtc", weights, atom) / weights.sum(dim=-2).unsqueeze(1).unsqueeze(-1)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert atom.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps NestedTensor kernels require CUDA")
def test_nested_atom_to_token_mean_uses_token_reference() -> None:
    pytest.importorskip("oops")
    device = torch.device("cuda")
    module = AtomToTokenMean(backend="oops")
    counter = CompileCounter()
    compiled = torch.compile(
        lambda atom, mapping, token_reference, atom_mask: module(
            atom,
            mapping,
            token_reference,
            atom_mask,
        ).concat,
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    cases = (((3, 5), (2, 3)), ((6, 4), (4, 3)))
    for atom_lengths_tuple, token_lengths_tuple in cases:
        atom_lengths = torch.tensor(atom_lengths_tuple, dtype=torch.int64)
        token_lengths = torch.tensor(token_lengths_tuple, dtype=torch.int64)
        reference = NestedTensor([torch.empty(1, 2) for _ in atom_lengths_tuple], ragged_dims=(0,))
        atom_values = torch.randn(sum(atom_lengths_tuple), 4, device=device, requires_grad=True)
        mappings = tuple(
            torch.arange(atoms, device=device) % tokens
            for atoms, tokens in zip(atom_lengths_tuple, token_lengths_tuple, strict=True)
        )
        masks = tuple(torch.arange(atoms, device=device) % 4 != 1 for atoms in atom_lengths_tuple)
        mapping_values = torch.cat(mappings)
        mask_values = torch.cat(masks)
        token_values = torch.empty(sum(token_lengths_tuple), 2, device=device)
        atom = reference.packed_with_lengths(atom_values, atom_lengths)
        mapping = reference.packed_with_lengths(mapping_values, atom_lengths)
        token_reference = reference.packed_with_lengths(token_values, token_lengths)
        atom_mask = reference.packed_with_lengths(mask_values, atom_lengths)

        expected_parts = []
        atom_parts = atom_values.split(atom_lengths_tuple)
        for atom_part, mapping_part, mask_part, tokens in zip(
            atom_parts,
            mappings,
            masks,
            token_lengths_tuple,
            strict=True,
        ):
            weights = F.one_hot(mapping_part, num_classes=tokens).to(atom_part)
            weights = weights * mask_part.unsqueeze(-1)
            # A token without a valid atom sums to zero; dividing it by one keeps it zero.
            expected_parts.append(weights.transpose(0, 1) @ atom_part / weights.sum(0).clamp_min(1).unsqueeze(-1))
        expected = torch.cat(expected_parts)
        grad = torch.randn_like(expected)

        actual = compiled(atom, mapping, token_reference, atom_mask)
        actual_grad = torch.autograd.grad(actual, atom_values, grad, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, atom_values, grad)[0]

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
    assert counter.frame_count == 1


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_nested_local_pair_broadcast_reconstructs_row_extents(device) -> None:
    device = torch.device(device)
    query_atoms, key_atoms = 2, 4
    windowing = ClampedLocalAtomWindowing(num_query_atoms=query_atoms, num_key_atoms=key_atoms)
    module = TokenPairToLocalAtomPairBroadcast(backend="oops")
    counter = CompileCounter()
    compiled = torch.compile(
        lambda pair, mapping, atom_mask: module(pair, mapping, windowing.window_info(atom_mask)).concat,
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    cases = (((2, 4), (3, 8), (0, 5)), ((3, 5), (5, 9), (3, 6)))
    for token_lengths, atom_lengths_tuple, valid_extents in cases:
        mappings = tuple(
            torch.arange(atoms, device=device) % tokens
            for atoms, tokens in zip(atom_lengths_tuple, token_lengths, strict=True)
        )
        masks = tuple(
            torch.arange(atoms, device=device) < valid_atoms
            for atoms, valid_atoms in zip(atom_lengths_tuple, valid_extents, strict=True)
        )
        atom_lengths = torch.tensor(atom_lengths_tuple, dtype=torch.int64)
        pair_reference = NestedTensor(
            [torch.empty(tokens, tokens, 1) for tokens in token_lengths],
            ragged_dims=(0, 1),
        )
        atom_reference = NestedTensor([torch.empty(1, dtype=torch.bool) for _ in token_lengths], ragged_dims=(0,))
        total_pairs = sum(tokens * tokens for tokens in token_lengths)
        pair_values = torch.randn(total_pairs, 1, device=device, requires_grad=True)
        pair = pair_reference.packed_like(pair_values)
        mapping = pair_reference.packed_with_lengths(torch.cat(mappings), atom_lengths)
        atom_mask = atom_reference.packed_with_lengths(torch.cat(masks), atom_lengths)

        # The clamped rule: every centred start moves into [0, max(V - key_atoms, 0)]; keys outside the
        # valid extent V read token zero.
        key_indices = []
        for atoms, valid_atoms in zip(atom_lengths_tuple, valid_extents, strict=True):
            blocks = (atoms + query_atoms - 1) // query_atoms
            centered = torch.arange(blocks, device=device) * query_atoms - (key_atoms - query_atoms) // 2
            starts = centered.clamp(0, max(valid_atoms - key_atoms, 0))
            indices = starts[:, None] + torch.arange(key_atoms, device=device)
            key_indices.append(torch.where((indices >= 0) & (indices < valid_atoms), indices, -1))

        expected_parts = []
        pair_parts = pair_values.split(tuple(tokens * tokens for tokens in token_lengths))
        for packed_pair, atom_to_token, indices, tokens in zip(
            pair_parts,
            mappings,
            key_indices,
            token_lengths,
            strict=True,
        ):
            blocks = indices.shape[0]
            query_index = F.pad(
                atom_to_token,
                (0, blocks * query_atoms - atom_to_token.numel()),
            ).reshape(blocks, query_atoms)
            safe_key = indices.clamp_min(0)
            key_index_for_pair = torch.where(indices >= 0, atom_to_token[safe_key], 0)
            packed_pair = packed_pair.reshape(tokens, tokens, 1)
            expected_parts.append(packed_pair[query_index[:, :, None], key_index_for_pair[:, None, :]])
        expected = torch.cat(expected_parts)
        grad = torch.randn_like(expected)

        actual = compiled(pair, mapping, atom_mask)
        actual_grad = torch.autograd.grad(actual, pair_values, grad, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, pair_values, grad)[0]

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
    assert counter.frame_count == 1


@pytest.mark.parametrize("extra_prefix", [(), (1,)])
def test_token_to_atom_broadcast(extra_prefix: tuple[int, ...]) -> None:
    # token -> atom broadcast gathers each atom's token row, here through an arbitrary leading dim
    token_hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).reshape(1, *extra_prefix, 3, 2)
    atom_to_token = torch.tensor([[2, 0, 1, 1]])
    atom_hidden = TokenToAtomBroadcast()(token_hidden, atom_to_token)
    broadcast_expected = torch.tensor([[[5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]]).reshape(
        1, *extra_prefix, 4, 2
    )
    assert torch.equal(atom_hidden, broadcast_expected)


@pytest.mark.parametrize("num_samples", [1, 3])
def test_token_to_atom_broadcast_ragged_lengths_and_gradients(num_samples: int) -> None:
    token_parts = [torch.randn(num_samples, length, 2) for length in (3, 5)]
    token = NestedTensor(token_parts, ragged_dims=(1,))
    values = token.concat.detach().requires_grad_()
    token = token.packed_like(values)
    indices = [torch.tensor([2, 0, 1, 1, 2]), torch.tensor([4, 0, 1, 1, 3, 2, 4])]
    atom_to_token = NestedTensor(indices, ragged_dims=(0,))
    actual = TokenToAtomBroadcast(backend="eager")(token, atom_to_token)
    expected = [part[:, index] for part, index in zip(torch.unbind(token), indices, strict=True)]

    for part, reference in zip(torch.unbind(actual), expected, strict=True):
        torch.testing.assert_close(part, reference)
    actual_loss = sum(part.square().sum() for part in torch.unbind(actual))
    expected_loss = sum(part.square().sum() for part in expected)
    actual_gradient = torch.autograd.grad(actual_loss, values, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_loss, values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    ("atom_to_token", "expected"),
    [
        # Shared-across-batch layout. q=2,k=4 over 4 atoms gives 2 windows; the centred key offset
        # is (4-2)//2=1, so window w spans atoms [2w-1 .. 2w+2] with token id 0 in padded slots.
        (
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            torch.tensor(
                [
                    [
                        [[0.0, 0.0, 1.0, 2.0], [10.0, 10.0, 11.0, 12.0]],
                        [[21.0, 22.0, 23.0, 20.0], [31.0, 32.0, 33.0, 30.0]],
                    ],
                    [
                        [[100.0, 100.0, 101.0, 102.0], [110.0, 110.0, 111.0, 112.0]],
                        [[121.0, 122.0, 123.0, 120.0], [131.0, 132.0, 133.0, 130.0]],
                    ],
                ]
            ),
        ),
        # Distinct per-batch layout; batch 1 reverses its assignment to tokens [3, 2, 1, 0], using the
        # same centred window geometry over different gathered ids.
        (
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            torch.tensor(
                [
                    [
                        [[0.0, 0.0, 1.0, 2.0], [10.0, 10.0, 11.0, 12.0]],
                        [[21.0, 22.0, 23.0, 20.0], [31.0, 32.0, 33.0, 30.0]],
                    ],
                    [
                        [[130.0, 133.0, 132.0, 131.0], [120.0, 123.0, 122.0, 121.0]],
                        [[112.0, 111.0, 110.0, 110.0], [102.0, 101.0, 100.0, 100.0]],
                    ],
                ]
            ),
        ),
    ],
    ids=["shared-layout", "per-batch-layout"],
)
def test_local_pair_broadcast(atom_to_token: Tensor, expected: Tensor) -> None:
    # token_pair[b, i, j] = 100*b + 10*i + j uniquely encodes each (row, col) token pair, so a gathered value
    # reads back directly as the (query_token, key_token) the window selected (expected traced in parametrize).
    num_tokens = 4
    token_pair = torch.zeros(2, num_tokens, num_tokens, 1)
    rows = torch.arange(num_tokens).view(num_tokens, 1)
    cols = torch.arange(num_tokens).view(1, num_tokens)
    token_pair[0, :, :, 0] = (10 * rows + cols).float()
    token_pair[1, :, :, 0] = (100 + 10 * rows + cols).float()
    window_info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(torch.ones(2, 4, dtype=torch.bool))

    atom_pair = TokenPairToLocalAtomPairBroadcast()(token_pair, atom_to_token, window_info)

    assert atom_pair.shape == (2, 2, 2, 4, 1)
    torch.testing.assert_close(atom_pair, expected.unsqueeze(-1))


def test_local_pair_broadcast_follows_clamped_key_index() -> None:
    token_pair = torch.randn(2, 3, 3, 2)
    atom_to_token = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]])
    atom_mask = torch.tensor([[True] * 6, [True, True, False, True, False, False]])
    window_info = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
    assert window_info.key_index is not None

    atom_pair = TokenPairToLocalAtomPairBroadcast()(token_pair, atom_to_token, window_info)

    # Query slots read their atom's token; key slots read the token of their key_index atom, and token
    # zero where the slot holds no valid atom.
    key_index = window_info.key_index
    key_token = atom_to_token.unsqueeze(-2).expand(-1, key_index.shape[-2], -1).gather(-1, key_index.clamp_min(0))
    key_token = torch.where(key_index >= 0, key_token, 0)
    query_token = atom_to_token.reshape(2, 3, 2)
    batch = torch.arange(2).view(2, 1, 1, 1)
    expected = token_pair[batch, query_token[..., :, None], key_token[..., None, :]]
    assert torch.equal(key_index[1], torch.tensor([[0, 1, -1, 3]]).expand(3, -1))
    torch.testing.assert_close(atom_pair, expected, atol=0, rtol=0)
