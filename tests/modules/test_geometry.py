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

import math

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.geometry import (
    apply_inverse_rigid_transform,
    apply_rigid_transform,
    kabsch_transform,
    neighbor_atom_distances,
    normalize_vectors,
    quaternion_to_rotation,
    rigid_alignment_transform,
    rotate_vectors,
    rotation_from_two_vectors,
    rotation_to_quaternion,
    smoothed_vector_norm,
    uniform_random_rotation,
    unit_quaternion_to_rotation,
    virtual_beta_carbon,
    weighted_rigid_align,
)


def _neighbor_distance_reference(coordinates, neighbor_indices, eps):
    neighbors = torch.stack([coordinates[batch][indices] for batch, indices in enumerate(neighbor_indices)])
    displacement = coordinates[:, :, None, :, None, :] - neighbors[:, :, :, None, :, :]
    return (displacement.square().sum(-1) + eps).sqrt()


@pytest.mark.parametrize(("num_residues", "num_atoms", "num_neighbors"), [(5, 3, 2), (2, 2, 2), (5, 3, 0)])
def test_neighbor_atom_distances_preserve_ordered_indexes_and_coordinate_vjps(num_residues, num_atoms, num_neighbors):
    values = torch.randn(num_residues, 2, num_atoms, 3, dtype=torch.float64).transpose(0, 1)
    indices = (
        torch.arange(2 * num_residues * num_neighbors).reshape(2, num_residues, num_neighbors) * 2 + 1
    ) % num_residues
    actual_input = values.detach().clone().requires_grad_()
    reference_input = values.detach().clone().requires_grad_()
    actual = neighbor_atom_distances(actual_input, indices, eps=1e-6)
    expected = _neighbor_distance_reference(reference_input, indices, 1e-6)
    cotangent = torch.linspace(-0.7, 0.9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)

    assert actual.shape == (2, num_residues, num_neighbors, num_atoms, num_atoms)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad((actual * cotangent).sum(), actual_input)[0],
        torch.autograd.grad((expected * cotangent).sum(), reference_input)[0],
    )


def test_neighbor_atom_distances_preserve_sample_local_packed_residue_axis():
    parts = (torch.randn(3, 4, 3, dtype=torch.float64), torch.randn(5, 4, 3, dtype=torch.float64))
    indexes = (torch.tensor([[2, 0], [0, 2], [1, 1]]), torch.tensor([[4, 0], [3, 2], [0, 4], [1, 2], [2, 3]]))
    coordinates = NestedTensor(parts, ragged_dims=(0,))
    neighbors = NestedTensor(indexes, ragged_dims=(0,))
    actual = neighbor_atom_distances(coordinates, neighbors, eps=1e-6)
    expected = torch.cat(
        [
            _neighbor_distance_reference(value[None], index[None], 1e-6).squeeze(0)
            for value, index in zip(parts, indexes, strict=True)
        ]
    )

    assert isinstance(actual, NestedTensor)
    assert actual.ragged_dims == (0,)
    assert actual.element_sizes().tolist() == [[3, 2, 4, 4], [5, 2, 4, 4]]
    torch.testing.assert_close(actual.concat, expected)


@pytest.mark.parametrize(
    ("dtype", "magnitude", "tolerance"),
    [
        (torch.float64, 1e200, 3e-14),
        (torch.float64, 1e-200, 3e-14),
        (torch.float32, 1e20, 3e-6),
        (torch.float32, 1e-30, 3e-6),
        (torch.float16, 1000.0, 4e-3),
        (torch.float16, 1e-3, 4e-3),
        (torch.bfloat16, 1e20, 4e-2),
        (torch.bfloat16, 1e-30, 4e-2),
    ],
)
@pytest.mark.parametrize("eps", [0.0, 1e-6])
@pytest.mark.parametrize("keepdim", [False, True])
def test_smoothed_vector_norm_extreme_values_and_gradients(dtype, magnitude, tolerance, eps, keepdim):
    vectors = torch.tensor([[magnitude, -magnitude / 2, magnitude / 4]], dtype=dtype, requires_grad=True)
    coordinates = vectors.detach().double().tolist()[0]
    reference = math.hypot(*coordinates, math.sqrt(eps))
    expected = vectors.new_tensor([reference])
    if keepdim:
        expected = expected.unsqueeze(-1)
    actual = smoothed_vector_norm(vectors, eps=eps, keepdim=keepdim)
    gradient = torch.autograd.grad(actual.sum(), vectors)[0]
    expected_gradient = vectors.new_tensor([[value / reference for value in coordinates]])

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all() and torch.isfinite(gradient).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=tolerance, atol=0)


@pytest.mark.parametrize("eps", [0.0, 1e-6])
def test_smoothed_vector_norm_zero_subgradient_and_empty_points(eps):
    vectors = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    actual = smoothed_vector_norm(vectors, eps=eps)
    gradient = torch.autograd.grad(actual.sum(), vectors)[0]
    torch.testing.assert_close(actual, torch.full((2,), math.sqrt(eps), dtype=vectors.dtype), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(vectors), rtol=0, atol=0)
    assert smoothed_vector_norm(torch.empty(2, 0, 3), eps=eps).shape == (2, 0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_quaternion_normalization_covers_nonzero_magnitudes(dtype):
    direction = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype)
    magnitudes = torch.tensor([torch.finfo(dtype).tiny / 8, 1.0, torch.finfo(dtype).max * 0.75], dtype=dtype)
    quaternion = magnitudes[:, None] * direction
    normalized = normalize_vectors(quaternion)
    expected = direction.expand_as(quaternion) / 2
    rotation = quaternion_to_rotation(quaternion)

    assert normalized.dtype == dtype and rotation.dtype == dtype
    assert torch.isfinite(normalized).all() and torch.isfinite(rotation).all()
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    torch.testing.assert_close(rotation, _quaternion_action_rotation(expected), rtol=0, atol=0)
    torch.testing.assert_close(quaternion_to_rotation(-quaternion), rotation, rtol=0, atol=0)
    torch.testing.assert_close(
        rotation.double() @ rotation.double().mT, torch.eye(3).double().expand_as(rotation), rtol=0, atol=0
    )


@pytest.mark.parametrize("magnitude", [1e-100, 1.0, 1e100])
def test_quaternion_normalization_all_input_vjps(magnitude):
    quaternion = (torch.tensor([[0.7, -0.2, 0.3, 0.6]], dtype=torch.float64) * magnitude).requires_grad_()
    cotangent = torch.tensor([[0.3, -0.2, 0.1, 0.4]], dtype=torch.float64)
    normalized = normalize_vectors(quaternion)
    gradient = torch.autograd.grad((normalized * cotangent).sum(), quaternion)[0]
    norm = quaternion.detach().norm(dim=-1, keepdim=True)
    expected_unit = quaternion.detach() / norm
    expected = (cotangent - expected_unit * (expected_unit * cotangent).sum(-1, keepdim=True)) / norm

    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(normalized, expected_unit, rtol=3e-14, atol=3e-14)
    torch.testing.assert_close(gradient, expected, rtol=3e-14, atol=1e-114)
    torch.testing.assert_close(
        (gradient * quaternion.detach()).sum(-1), torch.zeros(1, dtype=gradient.dtype), rtol=0, atol=2e-15
    )


def test_unit_quaternion_kernel_preserves_unit_input_values_and_vjps():
    value = torch.tensor([[0.7, -0.2, 0.3, 0.6]], dtype=torch.float64)
    value = value / value.norm(dim=-1, keepdim=True)
    actual_input = value.clone().requires_grad_()
    reference_input = value.clone().requires_grad_()
    actual = unit_quaternion_to_rotation(actual_input)
    scalar, vector = reference_input[..., :1], reference_input[..., 1:]
    points = torch.eye(3, dtype=value.dtype).expand(*value.shape[:-1], 3, 3)
    vector = vector[..., None, :].expand_as(points)
    cross = torch.cross(vector, points, dim=-1)
    expected = (points + 2 * scalar[..., None] * cross + 2 * torch.cross(vector, cross, dim=-1)).mT
    cotangent = torch.linspace(-0.4, 0.8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad((actual * cotangent).sum(), actual_input)[0],
        torch.autograd.grad((expected * cotangent).sum(), reference_input)[0],
        rtol=2e-13,
        atol=2e-13,
    )


def _rigid_component_reference(points, rotation, translation, inverse):
    shifted = points - translation.unsqueeze(-2) if inverse else points
    components = []
    for output in range(3):
        components.append(
            sum(
                shifted[..., source]
                * (rotation[..., source, output] if inverse else rotation[..., output, source]).unsqueeze(-1)
                for source in range(3)
            )
        )
    result = torch.stack(components, dim=-1)
    return result if inverse else result + translation.unsqueeze(-2)


@pytest.mark.parametrize(
    ("point_shape", "frame_shape"),
    [((2, 5, 7, 3), (2, 5)), ((2, 5, 4, 7, 3), (2, 5, 1)), ((2, 2, 2, 2, 3), (2, 2, 1)), ((2, 5, 0, 3), (2, 5))],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_rigid_transform_values_and_all_input_vjps(point_shape, frame_shape, inverse):
    points = torch.randn(point_shape, dtype=torch.float64)
    rotation = torch.randn(*frame_shape, 3, 3, dtype=torch.float64)
    translation = torch.randn(*frame_shape, 3, dtype=torch.float64)
    actual_inputs = [value.clone().requires_grad_() for value in (points, rotation, translation)]
    reference_inputs = [value.clone().requires_grad_() for value in (points, rotation, translation)]
    operation = apply_inverse_rigid_transform if inverse else apply_rigid_transform
    actual = operation(*actual_inputs)
    expected = _rigid_component_reference(*reference_inputs, inverse)
    cotangent = torch.linspace(-0.6, 0.8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)

    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad((actual * cotangent).sum(), actual_inputs)
    expected_gradients = torch.autograd.grad((expected * cotangent).sum(), reference_inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_rigid_transform_inverse_and_noncontiguous_points():
    points = torch.randn(5, 2, 7, 3, dtype=torch.float64).transpose(0, 1)
    quaternion = torch.randn(5, 2, 4, dtype=torch.float64).transpose(0, 1)
    rotation = quaternion_to_rotation(quaternion)
    translation = torch.randn(5, 2, 3, dtype=torch.float64).transpose(0, 1)
    transformed = apply_rigid_transform(points, rotation, translation)

    torch.testing.assert_close(transformed, _rigid_component_reference(points, rotation, translation, False))
    torch.testing.assert_close(apply_inverse_rigid_transform(transformed, rotation, translation), points)


def _quaternion_action_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    scalar, vector = quaternion[..., :1], quaternion[..., 1:]
    points = torch.eye(3, device=quaternion.device, dtype=quaternion.dtype).expand(*quaternion.shape[:-1], 3, 3)
    vector = vector.unsqueeze(-2).expand_as(points)
    first_cross = torch.cross(vector, points, dim=-1)
    rotated = points + 2 * scalar.unsqueeze(-1) * first_cross + 2 * torch.cross(vector, first_cross, dim=-1)
    return rotated.mT


def _axis_angle_rotation(parameters: torch.Tensor) -> torch.Tensor:
    axis = parameters[..., :3]
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angle = parameters[..., 3]
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).unflatten(-1, (3, 3))
    identity = torch.eye(3, device=parameters.device, dtype=parameters.dtype)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    cosine = angle.cos()[..., None, None]
    return cosine * identity + (1 - cosine) * outer + angle.sin()[..., None, None] * cross


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_quaternion_rotation_half_turns_and_gauge(dtype) -> None:
    quaternions = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -3, 2, 1], [0.3, -1, 2, -3]],
        dtype=dtype,
    )
    expected = _quaternion_action_rotation(quaternions)
    rotations = quaternion_to_rotation(quaternions)
    recovered = rotation_to_quaternion(rotations)
    tolerance = {torch.float64: 2e-14, torch.float32: 2e-6, torch.float16: 0.008, torch.bfloat16: 0.04}[dtype]

    torch.testing.assert_close(rotations, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(quaternion_to_rotation(-7 * quaternions), rotations, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(quaternion_to_rotation(recovered), rotations, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(recovered.norm(dim=-1), torch.ones(6, dtype=dtype), rtol=tolerance, atol=tolerance)
    assert (recovered[..., 0] >= 0).all()
    torch.testing.assert_close(rotations[1], torch.diag(torch.tensor([1, -1, -1], dtype=dtype)), rtol=0, atol=0)
    torch.testing.assert_close(recovered[:4], torch.eye(4, dtype=dtype), rtol=0, atol=0)
    assert recovered[4, 1] > 0


def test_quaternion_rotation_values_and_all_input_vjps() -> None:
    quaternion = torch.tensor([[0.7, -0.2, 0.3, 0.6], [-0.4, 0.8, 0.2, -0.1]], dtype=torch.float64)
    actual_input = quaternion.clone().requires_grad_()
    reference_input = quaternion.clone().requires_grad_()
    actual = quaternion_to_rotation(actual_input)
    expected = _quaternion_action_rotation(reference_input)
    cotangent = torch.linspace(-0.4, 0.8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual_gradient = torch.autograd.grad((actual * cotangent).sum(), actual_input)[0]
    expected_gradient = torch.autograd.grad((expected * cotangent).sum(), reference_input)[0]

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(
        (actual_gradient * actual_input).sum(-1), torch.zeros(2, dtype=quaternion.dtype), atol=2e-14, rtol=0
    )


def test_rotation_quaternion_reconstruction_near_pi_and_tangent_vjps() -> None:
    parameters = torch.tensor(
        [
            [1, 2, -3, 0],
            [1, 2, -3, 0.8],
            [1, 2, -3, torch.pi - 1e-8],
            [1, 2, -3, torch.pi],
            [1, 2, -3, torch.pi + 1e-8],
        ],
        dtype=torch.float64,
    )
    actual_input = parameters.clone().requires_grad_()
    reference_input = parameters.clone().requires_grad_()
    rotations = _axis_angle_rotation(actual_input)
    quaternion = rotation_to_quaternion(rotations)
    actual = quaternion_to_rotation(quaternion)
    expected = _axis_angle_rotation(reference_input)
    cotangent = torch.linspace(-0.4, 0.8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual_gradient = torch.autograd.grad((actual * cotangent).sum(), actual_input)[0]
    expected_gradient = torch.autograd.grad((expected * cotangent).sum(), reference_input)[0]

    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual, expected, rtol=3e-14, atol=3e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(5, dtype=parameters.dtype), rtol=2e-14, atol=2e-14)
    assert (quaternion[..., 0] >= 0).all()


def test_rotation_quaternion_reconstructs_arbitrary_exact_half_turn() -> None:
    axis = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    axis = axis / axis.norm()
    rotation = 2 * axis[:, None] * axis[None, :] - torch.eye(3, dtype=axis.dtype)
    quaternion = rotation_to_quaternion(rotation)

    torch.testing.assert_close(quaternion, torch.cat((axis.new_zeros(1), axis)), rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(quaternion_to_rotation(quaternion), rotation, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [0.0, 1e-8, 0.125])
@pytest.mark.parametrize("keepdim", [False, True])
def test_smoothed_norm_complete_values_and_derivatives(dtype, eps, keepdim) -> None:
    vector = torch.tensor([[0, 0, 0], [2e-5, -3e-5, 0], [500, 300, 100]], dtype=dtype, requires_grad=True)
    result = smoothed_vector_norm(vector, eps=eps, keepdim=keepdim)
    reference = vector.detach().double()
    expected = (reference.square().sum(-1, keepdim=keepdim) + eps).sqrt()
    tolerance = {torch.float64: 1e-11, torch.float32: 1e-5, torch.float16: 0.006, torch.bfloat16: 0.03}[dtype]
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.double(), expected, rtol=tolerance, atol=1e-7)
    gradient = torch.autograd.grad(result.sum(), vector)[0]
    denominator = (reference.square().sum(-1, keepdim=True) + eps).sqrt()
    expected_gradient = torch.where(denominator > 0, reference / denominator.clamp_min(1e-300), 0.0)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient.double(), expected_gradient, rtol=tolerance, atol=tolerance * 1e-3)
    torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_smoothed_norm_preserves_native_double_ragged_topology(dtype) -> None:
    parts = [torch.zeros(0, 2, 3, dtype=dtype), torch.zeros(2, 2, 3, dtype=dtype), torch.randn(3, 3, 3, dtype=dtype)]
    parts = [value.requires_grad_() for value in parts]
    source = NestedTensor(parts, ragged_dims=(0, 1))
    output = smoothed_vector_norm(source)
    expected = torch.cat([(part.detach().double().square().sum(-1) + 1e-8).sqrt().flatten() for part in parts])
    tolerance = {torch.float64: 1e-11, torch.float32: 1e-5, torch.float16: 0.006, torch.bfloat16: 0.03}[dtype]
    torch.testing.assert_close(output.concat.double(), expected, rtol=tolerance, atol=1e-7)
    assert output.ragged_dims == (0, 1)
    torch.testing.assert_close(output.element_sizes(), torch.tensor([[0, 2], [2, 2], [3, 3]]))
    gradients = torch.autograd.grad(output.concat.sum(), parts)
    for part, gradient in zip(parts, gradients, strict=True):
        vector = part.detach().double()
        expected_gradient = vector / (vector.square().sum(-1, keepdim=True) + 1e-8).sqrt()
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient.double(), expected_gradient, rtol=tolerance, atol=tolerance * 1e-3)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_smoothed_frame_zero_axes_have_finite_gradients(dtype) -> None:
    first = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    second = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    rotation = rotation_from_two_vectors(first, second)
    torch.testing.assert_close(rotation, torch.zeros_like(rotation), rtol=0, atol=0)
    gradients = torch.autograd.grad(rotation.sum(), (first, second))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_rotation_from_two_vectors() -> None:
    rotation = rotation_from_two_vectors(torch.tensor([2.0, 0.0, 0.0]), torch.tensor([1.0, 3.0, 0.0]))

    torch.testing.assert_close(rotation, torch.eye(3))


def test_virtual_beta_carbon() -> None:
    nitrogen = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    alpha_carbon = nitrogen + torch.tensor([1.0, 0.0, 0.0])
    carbon = alpha_carbon + torch.tensor([0.0, 1.0, 0.0])

    beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)

    expected_offset = torch.tensor([1.56802827, -0.54067466, -0.58273431])
    torch.testing.assert_close(beta_carbon, nitrogen + expected_offset)


def test_rotate_vectors() -> None:
    vectors = torch.randn(2, 5, 3)
    rotation = uniform_random_rotation(2)

    torch.testing.assert_close(rotate_vectors(vectors, rotation), vectors @ rotation)


def test_weighted_rigid_align() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target = (source.detach() @ rotation.mT + source.new_tensor([2.0, -1.0, 4.0])).requires_grad_()
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

    aligned = weighted_rigid_align(source=source, target=target, weight=weight)

    # A generous, dtype-appropriate tolerance: this asserts the fit recovers a known noiseless
    # rigid transform, not that some particular SVD path rounds identically to any other.
    torch.testing.assert_close(aligned, target, rtol=1e-4, atol=1e-4)
    assert not aligned.requires_grad


def test_rigid_alignment_transform_column_convention() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
    )
    known_rotation = source.new_tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = source @ known_rotation.mT + source.new_tensor([2.0, -1.0, 4.0])
    rotation, source_centroid, target_centroid = rigid_alignment_transform(
        source=source,
        target=target,
        weight=source.new_tensor([1.0, 2.0, 3.0, 4.0]),
    )

    torch.testing.assert_close(rotation, known_rotation, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close((source - source_centroid) @ rotation.mT + target_centroid, target, rtol=1e-4, atol=1e-4)


def test_uniform_random_rotation() -> None:
    device = torch.device("cpu")
    rotations = uniform_random_rotation(
        4,
        device=device,
        dtype=torch.float64,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    repeated = uniform_random_rotation(
        4,
        device=device,
        dtype=torch.float64,
        generator=torch.Generator(device=device).manual_seed(7),
    )

    assert rotations.shape == (4, 3, 3)
    assert rotations.dtype == torch.float64
    assert rotations.device == device
    torch.testing.assert_close(rotations, repeated)
    torch.testing.assert_close(rotations @ rotations.mT, torch.eye(3, dtype=rotations.dtype).expand_as(rotations))
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(4, dtype=rotations.dtype))


def test_random_rotation_compile() -> None:
    def dynamic_rotation(reference: torch.Tensor) -> torch.Tensor:
        return uniform_random_rotation(
            torch.numel(reference[..., 0, 0]),
            device=reference.device,
            dtype=reference.dtype,
        )

    compiled = torch.compile(dynamic_rotation, backend="eager", dynamic=True, fullgraph=True)
    assert compiled(torch.ones(2, 4, 5, 3)).shape == (8, 3, 3)
    assert compiled(torch.ones(3, 2, 4, 5, 3)).shape == (24, 3, 3)


def test_kabsch_fit_is_invariant_to_small_positive_weight_scale() -> None:
    points = [[0, 0, 0], [1, 2, 0], [2, -1, 3], [4, 0, -2], [-1, 5, 1], [3, 2, 4]]
    for dtype, scales, tolerance in (
        (torch.float64, (1.0, 1e-12, 1e307), 1e-11),
        (torch.float32, (1.0, 1e-12, 1e-40, 5e37), 2e-5),
    ):
        source = torch.tensor(points, dtype=dtype)
        rotation = source.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = source @ rotation.mT + source.new_tensor([7, -4, 2])
        for scale in scales:
            fitted, first, second = kabsch_transform(
                source=source, target=target, weight=torch.arange(1, 7, dtype=dtype) * scale
            )
            torch.testing.assert_close(fitted, rotation, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close((source - first) @ fitted.mT + second, target, rtol=tolerance, atol=tolerance)


def test_nested_leaf_norm_gradients():
    parts = [torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True), torch.empty(0, 3, requires_grad=True)]
    result = smoothed_vector_norm(NestedTensor(parts, ragged_dims=(0,)), eps=0)
    gradients = torch.autograd.grad(result.concat.sum(), parts)
    torch.testing.assert_close(result.concat, torch.tensor([5.0]))
    torch.testing.assert_close(gradients[0], torch.tensor([[0.6, 0.8, 0.0]]))
    torch.testing.assert_close(gradients[1], torch.empty(0, 3))


def test_sampled_norm_preserves_axes():
    parts = [torch.randn(3, count, 3, dtype=torch.float64, requires_grad=True) for count in (0, 2)]
    source = NestedTensor(parts, ragged_dims=(1,))
    result = smoothed_vector_norm(source, eps=0.125)
    expected = [(part.square().sum(-1) + 0.125).sqrt() for part in parts]
    assert isinstance(result, NestedTensor)
    for actual, reference in zip(result.unbind(0), expected, strict=True):
        torch.testing.assert_close(actual, reference)
    actual_gradients = torch.autograd.grad(result.concat.sum(), parts, retain_graph=True)
    expected_gradients = torch.autograd.grad(sum(part.sum() for part in expected), parts)
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
