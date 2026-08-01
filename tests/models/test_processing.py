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

"""Behavior of ofoldx.models.processing."""

from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.processing import (
    augment_reference_positions,
    model_frame_rotation,
    model_log_probabilities,
    project_model_features,
)
from ofoldx.modules.batch import repeat_batch, unflatten_batch


class TestReferenceAugmentation:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rigid_geometry(self, dtype):
        points = torch.arange(12, dtype=dtype).reshape(4, 3) / 7
        before = torch.pdist(points)
        augment_reference_positions(points, torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(torch.pdist(points), before)

    @pytest.mark.parametrize("payload", [91.0, float("nan"), float("inf")])
    def test_unavailable_payload(self, payload):
        known = torch.tensor([True, False, True])
        source = torch.tensor([[1.0, 2.0, 3.0], [payload] * 3, [3.0, 1.0, 2.0]], requires_grad=True)
        actual = source.clone()
        expected = source.detach().masked_fill(~known[:, None], 0)
        spaces = torch.zeros(3, dtype=torch.long)
        augment_reference_positions(actual, spaces, known, generator=torch.Generator().manual_seed(174))
        augment_reference_positions(expected, spaces, known, generator=torch.Generator().manual_seed(174))
        torch.testing.assert_close(actual, expected)
        assert torch.equal(actual[~known], torch.zeros_like(actual[~known]))
        gradient = torch.autograd.grad(actual.square().sum(), source)[0]
        assert torch.isfinite(gradient).all()
        assert torch.equal(gradient[~known], torch.zeros_like(gradient[~known]))

    @pytest.mark.parametrize("size", [0, 2])
    def test_empty_observation_domain(self, size):
        positions = torch.full((size, 3), float("nan"))
        augment_reference_positions(positions, torch.zeros(size, dtype=torch.long), torch.zeros(size, dtype=torch.bool))
        assert torch.equal(positions, torch.zeros_like(positions))


class TestFeatureProjection:
    @pytest.mark.parametrize("nested", [False, True])
    def test_unavailable_values_and_gradients(self, nested):
        source = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]], requires_grad=True)
        mask = torch.tensor([[True], [False]])
        values = NestedTensor([source], ragged_dims=(0,)) if nested else source
        observed = NestedTensor([mask], ragged_dims=(0,)) if nested else mask
        output = project_model_features(values, observed, dtype=torch.float64)
        actual = output.concat if nested else output
        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64))
        gradient = torch.autograd.grad(actual.sum(), source)[0]
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))

    @pytest.mark.parametrize(("value", "error"), [(float("nan"), InvalidRequestError), (1e100, UnsupportedError)])
    def test_distinct_numeric_failures(self, value, error):
        with pytest.raises(error):
            project_model_features(
                torch.tensor([value], dtype=torch.float64), torch.tensor([True]), dtype=torch.float32
            )


class TestCategoricalProjection:
    @pytest.mark.parametrize("nested", [False, True])
    def test_unavailable_rows_and_gradients(self, nested):
        source = torch.tensor(
            [[1.0, 0.0, -torch.inf], [float("nan"), float("inf"), -torch.inf]],
            dtype=torch.float64,
            requires_grad=True,
        )
        values = NestedTensor([source], ragged_dims=(0,)) if nested else source
        mask = torch.tensor([True, False])
        available = NestedTensor([mask], ragged_dims=(0,)) if nested else mask
        output = model_log_probabilities(values, available)
        actual = output.concat if nested else output
        torch.testing.assert_close(actual[0], source[0].log_softmax(-1))
        torch.testing.assert_close(actual[1], source.new_tensor(3.0).log().neg().expand(3))
        expected = torch.zeros_like(source)
        expected[0] = source[0].softmax(-1)
        expected[0, 0] -= 1
        torch.testing.assert_close(torch.autograd.grad(-actual[:, 0].sum(), source)[0], expected)

    @pytest.mark.parametrize("payload", [float("nan"), float("inf"), -float("inf")])
    def test_invalid_available_row(self, payload):
        with pytest.raises(InvalidRequestError, match="one finite class"):
            model_log_probabilities(torch.full((1, 3), payload), torch.tensor([True]))

    def test_normalization_range(self):
        maximum = torch.finfo(torch.float32).max
        with pytest.raises(UnsupportedError, match="dtype range"):
            model_log_probabilities(torch.tensor([[maximum, -maximum]]), torch.tensor([True]))


class TestPhysicalFrames:
    @pytest.mark.parametrize("nested", [False, True])
    def test_unavailable_geometry_and_gradients(self, nested):
        first = torch.tensor([[1.0, 1.0, 0.0], [float("nan")] * 3], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([[0.0, 1.0, 1.0], [float("inf")] * 3], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([True, False])
        x = NestedTensor([first], ragged_dims=(0,)) if nested else first
        xy = NestedTensor([second], ragged_dims=(0,)) if nested else second
        available = NestedTensor([mask], ragged_dims=(0,)) if nested else mask
        frames = model_frame_rotation(x, xy, available)
        actual = frames.concat if nested else frames
        expected = torch.stack(
            (
                first.new_tensor([1.0, 1.0, 0.0]) / 2**0.5,
                first.new_tensor([-1.0, 1.0, 2.0]) / 6**0.5,
                first.new_tensor([1.0, -1.0, 1.0]) / 3**0.5,
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], torch.eye(3, dtype=first.dtype))
        x_axis = first[0] / first[0].norm()
        y_vector = second[0] - (second[0] * x_axis).sum() * x_axis
        y_axis = y_vector / y_vector.norm()
        reference = torch.stack((x_axis, y_axis, torch.cross(x_axis, y_axis, dim=-1)), dim=-1)
        weights = torch.arange(9, dtype=first.dtype).reshape(3, 3)
        expected_gradients = torch.autograd.grad((reference * weights).sum(), (first, second))
        actual_gradients = torch.autograd.grad((actual * weights).sum(), (first, second))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient)
            assert not bool(actual_gradient[1].any())

    @pytest.mark.parametrize("first,second", [([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), ([1.0] * 3, [2.0] * 3)])
    def test_degenerate_geometry(self, first, second):
        with pytest.raises(UnsupportedError):
            model_frame_rotation(torch.tensor([first]), torch.tensor([second]), torch.tensor([True]))

    def test_static_sample_axis(self):
        first = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
        second = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
        x = NestedTensor([first.expand(2, 1, 3), torch.empty(2, 0, 3, dtype=first.dtype)])
        xy = NestedTensor([second.expand(2, 1, 3), torch.empty(2, 0, 3, dtype=first.dtype)])
        available = NestedTensor([torch.ones(2, 1, dtype=torch.bool), torch.empty(2, 0, dtype=torch.bool)])
        frames = model_frame_rotation(x, xy, available)
        reference = model_frame_rotation(first, second, torch.tensor([True]))
        torch.testing.assert_close(frames[0][1], reference)

    @pytest.mark.parametrize("magnitude", [2.0**-600, 2.0**900])
    def test_magnitude_does_not_define_geometry(self, magnitude):
        first = torch.tensor([[magnitude, 0.0, 0.0]], dtype=torch.float64)
        second = torch.tensor([[0.0, magnitude, 0.0]], dtype=torch.float64)
        rotation = model_frame_rotation(first, second, torch.tensor([True]))
        torch.testing.assert_close(rotation, torch.eye(3, dtype=torch.float64).unsqueeze(0))

    def test_sampled_empty_atom_layout(self):
        first = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
        x = NestedTensor([first, torch.empty(0, 3, dtype=first.dtype)], ragged_dims=(0,))
        xy = NestedTensor([second, torch.empty(0, 3, dtype=first.dtype)], ragged_dims=(0,))
        available = NestedTensor([torch.tensor([True]), torch.empty(0, dtype=torch.bool)], ragged_dims=(0,))
        sampled_x = unflatten_batch(repeat_batch(x, 2), (2, 2))
        sampled_xy = unflatten_batch(repeat_batch(xy, 2), (2, 2))
        sampled_available = unflatten_batch(repeat_batch(available, 2), (2, 2))
        frames = model_frame_rotation(sampled_x, sampled_xy, sampled_available)
        assert isinstance(frames, NestedTensor)
        assert frames.shape[:2] == (2, 2)
        assert frames[0].shape == (2, 1, 3, 3)
        assert frames[1].shape == (2, 0, 3, 3)
        weights = torch.arange(9, dtype=first.dtype).reshape(3, 3)
        reference = model_frame_rotation(first, second, torch.tensor([True]))
        expected = torch.autograd.grad(2 * (reference * weights).sum(), (first, second))
        actual = torch.autograd.grad((frames.concat * weights).sum(), (first, second))
        for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient)


def test_reference_half_centroid():
    source = torch.full((1000, 3), 100.0, dtype=torch.float16, requires_grad=True)
    positions = source.clone()
    augment_reference_positions(
        positions,
        torch.zeros(1000, dtype=torch.long),
        torch.ones(1000, dtype=torch.bool),
        generator=torch.Generator().manual_seed(17),
    )
    assert torch.isfinite(positions).all()
    torch.testing.assert_close(positions, positions[:1].expand_as(positions))
    gradient = torch.autograd.grad(positions.float().square().mean(), source)[0]
    assert torch.isfinite(gradient).all()
