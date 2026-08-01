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


@pytest.mark.parametrize("nested, compiled", [(False, False), (True, False), (True, True)])
def test_repeat_values(nested, compiled):
    parts = [torch.randn(n, 3, requires_grad=True) for n in ((2, 5) if nested else (3, 3))]
    source = NestedTensor(parts, ragged_dims=(0,)) if nested else torch.stack(parts)

    def operation(value):
        return repeat_batch(value, 3)

    if compiled:
        operation = torch.compile(operation, backend="aot_eager", fullgraph=True, dynamic=True)
    actual = operation(source)
    for index in range(6):
        torch.testing.assert_close(actual[index], parts[index // 3])
    for gradient, part in zip(torch.autograd.grad(actual.square().sum(), parts), parts, strict=True):
        torch.testing.assert_close(gradient, 6 * part)


def test_dense_samples():
    source = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    grouped = unflatten_batch(source, (2, 2))
    torch.testing.assert_close(grouped, torch.stack((source[:2], source[2:])))
    torch.testing.assert_close(flatten_batch(grouped), source)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("lengths", [(2, 5), (3, 3), (0, 3)])
def test_sample_values(pair, lengths):
    parts = [torch.randn(n, *((n,) if pair else ()), 3, requires_grad=True) for n in lengths for _ in range(2)]
    source = NestedTensor(parts, ragged_dims=(0, 1) if pair else (0,))
    grouped = unflatten_batch(source, (2, 2))
    for index in range(2):
        torch.testing.assert_close(grouped[index], torch.stack(parts[2 * index : 2 * index + 2]))
    flattened = flatten_batch(grouped)
    torch.testing.assert_close(flattened.unbind(0), tuple(parts))
    for gradient, part in zip(torch.autograd.grad(grouped.square().sum(), parts), parts, strict=True):
        torch.testing.assert_close(gradient, 2 * part)


@pytest.mark.parametrize("pair", [False, True])
def test_static_samples(pair):
    parts = [torch.randn(2, n, *((n,) if pair else ()), 3, requires_grad=True) for n in (2, 5)]
    source = NestedTensor(parts, ragged_dims=(1, 2) if pair else (1,))
    flat = flatten_batch(source)
    expected = tuple(sample for part in parts for sample in part.unbind(0))
    torch.testing.assert_close(flat.unbind(0), expected)
    grouped = unflatten_batch(flat, (2, 2))
    torch.testing.assert_close(grouped.unbind(0), tuple(parts))
    for gradient, part in zip(torch.autograd.grad(grouped.square().sum(), parts), parts, strict=True):
        torch.testing.assert_close(gradient, 2 * part)


def test_fake_samples():
    from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

    source = NestedTensor([torch.randn(2, n, 3) for n in (2, 5)], ragged_dims=(1,))
    with FakeTensorMode() as mode:
        actual = unflatten_batch(flatten_batch(mode.from_tensor(source)), (2, 2))
    assert isinstance(actual, NestedTensor)
    assert is_fake(actual.concat)


@pytest.mark.parametrize("static", [False, True])
def test_compiled_samples(static):
    def consume(source):
        return unflatten_batch(flatten_batch(source), (2, 2))

    compiled = torch.compile(consume, backend="aot_eager", fullgraph=True, dynamic=True)
    for lengths in ((2, 5), (4, 1)):
        parts = [torch.randn(2, n, 3, requires_grad=True) for n in lengths]
        source = NestedTensor(parts, ragged_dims=(1,) if static else (0, 1))
        actual = compiled(source)
        torch.testing.assert_close(actual.unbind(0), tuple(parts))
        for gradient, part in zip(torch.autograd.grad(actual.square().sum(), parts), parts, strict=True):
            torch.testing.assert_close(gradient, 2 * part)
