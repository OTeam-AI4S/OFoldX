from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard

from ofoldx.modules.batch import repeat_batch


def _packed_leaf(lengths: tuple[int, ...], channels: int = 4) -> tuple[NestedTensor, torch.Tensor]:
    template = NestedTensor(
        [torch.empty(length, channels) for length in lengths],
        ragged_dims=(0,),
    )
    values = torch.randn_like(template.concat, requires_grad=True)
    return template.packed_like(values), values


def _strict_guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def test_repeat_batch_preserves_dense_semantics_bitwise():
    dense = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)

    assert repeat_batch(dense, 1) is dense
    assert torch.equal(
        repeat_batch(dense, 3),
        dense.repeat_interleave(3, dim=0),
    )


def test_repeat_batch_preserves_nested_order_and_topology():
    source, values = _packed_leaf((2, 5))

    assert repeat_batch(source, 1) is source
    with _strict_guard():
        output = repeat_batch(source, 3)

    expected = torch.cat((values[:2].repeat(3, 1), values[2:].repeat(3, 1)))
    assert torch.equal(output.concat, expected)
    assert torch.equal(output.packed_offsets(), torch.tensor([0, 2, 4, 6, 11, 16, 21]))
    assert output.ragged_dims == (0,)
    assert output.packed_dim_order == (0, 1)
    output._validate_metadata()


def test_repeat_batch_supports_external_fake_tensor():
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    source, _ = _packed_leaf((2, 5))

    with fake_tensor_mod.FakeTensorMode() as mode:
        output = repeat_batch(mode.from_tensor(source), 3)

    assert fake_tensor_mod.is_fake(output.concat)
    assert fake_tensor_mod.is_fake(output.packed_offsets())
    assert output.ragged_dims == (0,)


def test_repeat_batch_aot_backward_reuses_one_dynamic_graph():
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("aot_eager")

    def consume(source):
        output = repeat_batch(source, 3)
        return output.concat.square().sum(), output.packed_offsets(), output.element_sizes()

    compiled = torch.compile(consume, backend=counter, fullgraph=True, dynamic=True)
    compiled_frame_count = None
    for lengths in ((2, 5), (4, 1), (3, 6)):
        source, values = _packed_leaf(lengths)
        with _strict_guard():
            loss, offsets, sizes = compiled(source)
        loss.backward()
        assert torch.equal(values.grad, 6 * values)
        repeated_lengths = torch.tensor(lengths).repeat_interleave(3)
        assert torch.equal(offsets, torch.nn.functional.pad(repeated_lengths.cumsum(0), (1, 0)))
        assert torch.equal(sizes[:, 0], repeated_lengths)
        if compiled_frame_count is None:
            compiled_frame_count = counter.frame_count
            assert compiled_frame_count == 2  # one AOTAutograd forward/backward graph pair
        else:
            assert counter.frame_count == compiled_frame_count
