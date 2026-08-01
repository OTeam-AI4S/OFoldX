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

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.batch import flatten_batch, repeat_batch, unflatten_batch


def _packed_leaf(lengths: tuple[int, ...], channels: int | tuple[int, ...] = 4) -> tuple[NestedTensor, torch.Tensor]:
    tail_shape = (channels,) if isinstance(channels, int) else channels
    template = NestedTensor(
        [torch.empty(length, *tail_shape) for length in lengths],
        ragged_dims=(0,),
    )
    values = torch.randn_like(template.concat, requires_grad=True)
    return template.packed_like(values), values


def test_repeat_batch_preserves_dense_semantics_bitwise():
    dense = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)

    assert torch.equal(repeat_batch(dense, 1), dense)
    assert torch.equal(
        repeat_batch(dense, 3),
        dense.repeat_interleave(3, dim=0),
    )


def test_repeat_batch_preserves_nested_order_and_topology():
    source, values = _packed_leaf((2, 5))

    assert torch.equal(repeat_batch(source, 1).concat, source.concat)
    output = repeat_batch(source, 3)

    expected = torch.cat((values[:2].repeat(3, 1), values[2:].repeat(3, 1)))
    assert torch.equal(output.concat, expected)
    assert torch.equal(output.packed_offsets(), torch.tensor([0, 2, 4, 6, 11, 16, 21]))
    assert output.ragged_dims == (0,)
    assert output.packed_dim_order == (0, 1)


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
        loss, offsets, sizes = compiled(source)
        loss.backward()
        assert torch.equal(values.grad, 6 * values)
        repeated_lengths = torch.tensor(lengths).repeat_interleave(3)
        assert torch.equal(offsets, torch.nn.functional.pad(repeated_lengths.cumsum(0), (1, 0)))
        assert torch.equal(sizes[:, 0], repeated_lengths)
        if compiled_frame_count is None:
            compiled_frame_count = counter.frame_count
        else:
            assert counter.frame_count == compiled_frame_count


@pytest.mark.parametrize("samples", (1, 2, 3))
def test_sample_batch_helpers_preserve_dense_semantics(samples):
    source = torch.arange(3 * samples * 5 * 4).reshape(3 * samples, 5, 4)

    grouped = unflatten_batch(source, (3, samples))

    assert torch.equal(grouped, source.unflatten(0, (3, samples)))
    assert torch.equal(flatten_batch(grouped), source)


@pytest.mark.parametrize("samples", (1, 2, 3))
@pytest.mark.parametrize("lengths", ((7, 5, 3), (4, 4, 4)))
@pytest.mark.parametrize("tail_shape", ((), (3,), (2, 4)))
def test_sample_batch_helpers_preserve_nested_topology_and_autograd(samples, lengths, tail_shape):
    flat_lengths = tuple(length for length in lengths for _ in range(samples))
    source, values = _packed_leaf(flat_lengths, tail_shape)
    offsets = source.packed_offsets()

    grouped = unflatten_batch(source, (len(lengths), samples))
    flattened = flatten_batch(grouped)

    assert grouped.shape == (len(lengths), samples, max(lengths), *tail_shape)
    assert grouped.ragged_dims == (0, 1)
    assert torch.equal(grouped.packed_offsets(), offsets[::samples])
    assert torch.equal(grouped.ragged_level_offsets(0), torch.arange(len(lengths) + 1) * samples)
    assert torch.equal(grouped.ragged_level_offsets(1), offsets)
    assert torch.equal(flattened.element_sizes(), source.element_sizes())
    assert torch.equal(flattened.packed_offsets(), offsets)
    assert flattened.ragged_dims == source.ragged_dims
    assert flattened.packed_dim_order == source.packed_dim_order

    for batch_index, length in enumerate(lengths):
        for sample_index in range(samples):
            flat_index = batch_index * samples + sample_index
            selected = grouped[batch_index, sample_index]
            assert selected.shape == (length, *tail_shape)
            assert torch.equal(selected, values[offsets[flat_index] : offsets[flat_index + 1]])

    flattened.concat.square().sum().backward()
    assert torch.equal(values.grad, 2 * values)


def test_unflatten_batch_rejects_inconsistent_sample_lengths():
    source, _ = _packed_leaf((7, 5, 3, 3))

    with pytest.raises(RuntimeError, match="identical element sizes"):
        unflatten_batch(source, (2, 2))


def test_unflatten_batch_rejects_mismatched_batch_size():
    source, _ = _packed_leaf((7, 5, 3))

    with pytest.raises(ValueError, match="matching the batch"):
        unflatten_batch(source, (3, 2))


@pytest.mark.parametrize("samples", (1, 2, 3))
@pytest.mark.parametrize("lengths", ((7, 5, 3), (4, 4, 4)))
def test_flatten_batch_accepts_static_sample_axis(samples, lengths):
    template = NestedTensor(
        [torch.empty(samples, length, 3) for length in lengths],
        ragged_dims=(1,),
    )
    values = torch.randn_like(template.concat, requires_grad=True)
    source = template.packed_like(values)
    expected = torch.cat([source[batch, sample] for batch in range(len(lengths)) for sample in range(samples)])

    flattened = flatten_batch(source)
    regrouped = unflatten_batch(flattened, (len(lengths), samples))

    assert torch.equal(flattened.concat, expected)
    assert flattened.ragged_dims == (0,)
    assert flattened.element_sizes()[:, 0].tolist() == [length for length in lengths for _ in range(samples)]
    for batch, length in enumerate(lengths):
        for sample in range(samples):
            assert regrouped[batch, sample].shape == (length, 3)
            assert torch.equal(regrouped[batch, sample], source[batch, sample])
    regrouped.concat.square().sum().backward()
    assert torch.equal(values.grad, 2 * values)


@pytest.mark.parametrize("static_sample_axis", (False, True))
def test_sample_batch_helpers_support_external_fake_tensor(static_sample_axis):
    from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

    flat, _ = _packed_leaf((2, 2, 5, 5))
    grouped = (
        NestedTensor([torch.empty(2, length, 4) for length in (2, 5)], ragged_dims=(1,))
        if static_sample_axis
        else unflatten_batch(flat, (2, 2))
    )
    with FakeTensorMode() as mode:
        fake_grouped = unflatten_batch(mode.from_tensor(flat), (2, 2))
        fake_flat = flatten_batch(mode.from_tensor(grouped))

    assert is_fake(fake_grouped.concat)
    assert is_fake(fake_grouped.element_sizes())
    assert is_fake(fake_flat.concat)
    assert is_fake(fake_flat.packed_offsets())
    assert fake_grouped.ragged_dims == (0, 1)
    assert fake_flat.ragged_dims == (0,)


@pytest.mark.parametrize("static_sample_axis", (False, True))
def test_sample_batch_helpers_fullgraph_preserve_topology_and_vjp(static_sample_axis):
    def consume(source):
        flat = flatten_batch(source)
        grouped = unflatten_batch(flat, (2, 2))
        return grouped.concat.square().sum(), grouped.element_sizes(), flat.packed_offsets()

    compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
    for lengths in ((2, 5), (4, 1), (3, 6)):
        if static_sample_axis:
            template = NestedTensor([torch.empty(2, length, 4) for length in lengths], ragged_dims=(1,))
            values = torch.randn_like(template.concat, requires_grad=True)
            source = template.packed_like(values)
        else:
            flat, values = _packed_leaf(tuple(length for length in lengths for _ in range(2)))
            source = unflatten_batch(flat, (2, 2))

        loss, sizes, offsets = compiled(source)
        loss.backward()

        torch.testing.assert_close(values.grad, 2 * values, atol=0, rtol=0)
        assert torch.equal(sizes, torch.tensor([[2, lengths[0], 4], [2, lengths[1], 4]]))
        assert torch.equal(
            offsets, torch.nn.functional.pad(torch.tensor(lengths).repeat_interleave(2).cumsum(0), (1, 0))
        )
