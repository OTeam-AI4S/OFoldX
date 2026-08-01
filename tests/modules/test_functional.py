from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ofoldx.modules.functional import (
    pair_mask_from_msa_mask,
    pair_mask_from_token_mask,
)


def test_token_pair_mask() -> None:
    token_mask = torch.tensor([[True, False, True]])
    expected = torch.tensor([[[True, False, True], [False, False, False], [True, False, True]]])
    assert torch.equal(pair_mask_from_token_mask(token_mask), expected)


def test_msa_pair_mask() -> None:
    # an MSA mask collapses its rows (logical-or) before the outer mask, so two rows == token [F, T, T]
    msa_mask = torch.tensor([[[False, True, False], [False, False, True]]])
    pair_mask = pair_mask_from_msa_mask(msa_mask)
    assert pair_mask.shape == (1, 3, 3)
    assert torch.equal(pair_mask, pair_mask_from_token_mask(torch.tensor([[False, True, True]])))


def _guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def test_nested_token_pair_mask_stays_packed() -> None:
    parts = (
        torch.tensor([True, False]),
        torch.tensor([False, True, True]),
    )
    token_mask = NestedTensor(parts, ragged_dims=(0,))

    with _guard():
        pair_mask = pair_mask_from_token_mask(token_mask)

    assert isinstance(pair_mask, NestedTensor)
    assert pair_mask.ragged_dims == (0, 1)
    assert pair_mask.packed_dim_order == (0, 1)
    assert pair_mask.packed_offsets().tolist() == [0, 4, 13]
    expected = torch.cat(tuple((part[:, None] & part[None, :]).reshape(-1) for part in parts))
    assert torch.equal(pair_mask.concat, expected)


def test_nested_token_pair_mask_external_fake() -> None:
    template = NestedTensor([torch.empty(2, dtype=torch.bool), torch.empty(3, dtype=torch.bool)], ragged_dims=(0,))
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    mode = FakeTensorMode(shape_env=shape_env)
    with mode:
        token_mask = template.packed_like(torch.empty(5, device="cuda", dtype=torch.bool))
        pair_mask = pair_mask_from_token_mask(token_mask)

    assert isinstance(pair_mask, NestedTensor)
    assert is_fake(pair_mask.concat)
    assert pair_mask.concat.dim() == 1
    assert pair_mask.ragged_dims == (0, 1)
    assert pair_mask.packed_dim_order == (0, 1)


def test_nested_token_pair_mask_rejects_noncanonical_layout() -> None:
    token_mask = NestedTensor(
        [torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)],
        ragged_dims=(1,),
    )
    with pytest.raises(ValueError, match="explicit canonical"):
        pair_mask_from_token_mask(token_mask)


def test_nested_token_pair_mask_preserves_empty_elements() -> None:
    token_mask = NestedTensor(
        [torch.empty(0, dtype=torch.bool), torch.tensor([True, False])],
        ragged_dims=(0,),
    )
    with _guard():
        pair_mask = pair_mask_from_token_mask(token_mask)
    assert pair_mask.packed_offsets().tolist() == [0, 0, 4]
    assert torch.equal(pair_mask.concat, torch.tensor([True, False, False, False]))


def test_nested_token_pair_mask_reuses_one_dynamic_fullgraph() -> None:
    counter = CompileCounter()

    def run(token_mask):
        pair_mask = pair_mask_from_token_mask(token_mask)
        return pair_mask.concat, pair_mask.packed_offsets()

    compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)
    for parts in (
        (torch.tensor([True, False]), torch.tensor([False, True, True])),
        (torch.tensor([True, False, True]), torch.tensor([True, True, False, True, False])),
    ):
        token_mask = NestedTensor(parts, ragged_dims=(0,))
        with _guard():
            values, offsets = compiled(token_mask)
        expected = torch.cat(tuple((part[:, None] & part[None, :]).reshape(-1) for part in parts))
        assert torch.equal(values, expected)
        assert offsets.tolist() == [0, *(sum(part.numel() ** 2 for part in parts[:index]) for index in range(1, 3))]
    assert counter.frame_count == 1
