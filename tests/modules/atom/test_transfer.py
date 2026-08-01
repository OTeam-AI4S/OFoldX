from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter
from torch.nn import functional as F

from ofoldx.modules.atom import (
    AtomToTokenMean,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)

if TYPE_CHECKING:
    from torch import Tensor


def test_atom_to_token_mean() -> None:
    # Masked-out atom 1 drops from both sum and count; the denominator retains the model's 1e-6 offset.
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

    expected = torch.tensor(
        [[[[1.0 / 1.000001, 2.0 / 1.000001], [12.0 / 2.000001, 24.0 / 2.000001], [0.0, 0.0]]]]
    ).repeat(1, 2, 1, 1)

    token_hidden = AtomToTokenMean(denominator_offset=1e-6)(
        atom_hidden,
        atom_to_token,
        token_reference,
        atom_mask,
    )
    torch.testing.assert_close(token_hidden, expected)
    token_hidden.sum().backward()
    torch.testing.assert_close(
        atom_hidden.grad,
        torch.tensor(
            [
                [
                    [
                        [1 / 1.000001, 1 / 1.000001],
                        [0.0, 0.0],
                        [1 / 2.000001, 1 / 2.000001],
                        [1 / 2.000001, 1 / 2.000001],
                    ]
                ]
            ]
        ).repeat(1, 2, 1, 1),
    )


def test_atom_to_token_mean_fp16_empty_token_is_finite() -> None:
    atom = torch.tensor([[[2.0], [4.0]]], dtype=torch.float16)
    mapping = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, False]])
    token_reference = torch.empty(1, 2, 1, dtype=torch.float16)

    output = AtomToTokenMean(denominator_offset=None)(atom, mapping, token_reference, mask)

    torch.testing.assert_close(output, torch.tensor([[[2.0], [0.0]]], dtype=torch.float16))
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps atom transfer requires CUDA")
def test_sampled_oops_atom_to_token_mean_accepts_shared_mask() -> None:
    pytest.importorskip("oops")
    atom = torch.randn(2, 3, 5, 4, device="cuda", requires_grad=True)
    mapping = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], device="cuda")
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]], device="cuda")
    token_reference = torch.empty(2, 3, 1, device="cuda")

    actual = AtomToTokenMean(backend="oops", denominator_offset=1e-6)(atom, mapping, token_reference, mask)
    weights = F.one_hot(mapping, num_classes=3).to(atom) * mask.unsqueeze(-1)
    expected = torch.einsum("bat,bxac->bxtc", weights, atom) / (weights.sum(dim=-2) + 1e-6).unsqueeze(1).unsqueeze(-1)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert atom.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps NestedTensor kernels require CUDA")
def test_nested_atom_to_token_mean_uses_token_reference() -> None:
    pytest.importorskip("oops")
    device = torch.device("cuda")
    module = AtomToTokenMean(backend="oops", denominator_offset=1e-6)
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
            expected_parts.append(weights.transpose(0, 1) @ atom_part / (weights.sum(0) + 1e-6).unsqueeze(-1))
        expected = torch.cat(expected_parts)
        grad = torch.randn_like(expected)

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            actual = compiled(atom, mapping, token_reference, atom_mask)
            actual_grad = torch.autograd.grad(actual, atom_values, grad, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, atom_values, grad)[0]

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps NestedTensor kernels require CUDA")
def test_nested_local_pair_broadcast_reconstructs_row_extents() -> None:
    pytest.importorskip("oops")
    device = torch.device("cuda")
    query_atoms, key_atoms = 2, 4
    module = TokenPairToLocalAtomPairBroadcast(
        num_query_atoms=query_atoms,
        num_key_atoms=key_atoms,
        backend="oops",
    )
    counter = CompileCounter()
    compiled = torch.compile(
        lambda pair, mapping, key_index, valid_extent: module(
            pair,
            mapping,
            key_index=key_index,
            valid_extent=valid_extent,
        ).concat,
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
        atom_lengths = torch.tensor(atom_lengths_tuple, dtype=torch.int64)
        pair_reference = NestedTensor(
            [torch.empty(tokens, tokens, 1) for tokens in token_lengths],
            ragged_dims=(0, 1),
        )
        total_pairs = sum(tokens * tokens for tokens in token_lengths)
        pair_values = torch.randn(total_pairs, 1, device=device, requires_grad=True)
        pair = pair_reference.packed_like(pair_values)
        mapping = pair_reference.packed_with_lengths(torch.cat(mappings), atom_lengths)

        key_indices = []
        block_lengths = []
        for atoms, valid_atoms in zip(atom_lengths_tuple, valid_extents, strict=True):
            blocks = (atoms + query_atoms - 1) // query_atoms
            block_lengths.append(blocks)
            centered = torch.arange(blocks, device=device) * query_atoms - (key_atoms - query_atoms) // 2
            starts = centered.clamp(0, valid_atoms - key_atoms) if valid_atoms >= key_atoms else centered
            indices = starts[:, None] + torch.arange(key_atoms, device=device)
            key_indices.append(torch.where((indices >= 0) & (indices < valid_atoms), indices, -1))
        key_index = pair_reference.packed_with_lengths(
            torch.cat(key_indices),
            torch.tensor(block_lengths, dtype=torch.int64),
        )

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

        with nested_execution_guard(
            forbid_iteration=True,
            forbid_storage_map=True,
            forbid_eager_fallback=True,
            forbid_padded_materialization=True,
            forbid_dense_repack=True,
        ):
            actual = compiled(pair, mapping, key_index, torch.tensor(valid_extents, device=device))
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

    atom_pair = TokenPairToLocalAtomPairBroadcast(num_query_atoms=2, num_key_atoms=4)(token_pair, atom_to_token)

    assert atom_pair.shape == (2, 2, 2, 4, 1)
    torch.testing.assert_close(atom_pair, expected.unsqueeze(-1))
