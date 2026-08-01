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

r"""Prepare numerical features shared by model families.

Functions project masked coordinates, categories, counts, and continuous features,
normalize log probabilities, and handle frames and reference augmentation.
Canonical target construction and atom correspondences are provided by
`ofoldx.models.preparation` and its related modules; these functions consume tensors.
"""

from __future__ import annotations

from typing import NoReturn, cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor

from ..errors import RequestRef, fail
from ..modules.atom.coordinates import gather_observed_coordinates, project_coordinates
from ..modules.batch import flatten_batch, unflatten_batch
from ..modules.functional import project_categories, project_counts, weighted_mean
from ..modules.geometry import quaternion_to_rotation, rotate_vectors


def _invalid(feature: str, problem: str) -> NoReturn:
    fail("value_invalid", RequestRef(feature, problem))


def _numeric(feature: str, problem: str) -> NoReturn:
    fail("model_numeric", RequestRef(feature, problem))


def project_model_coordinates(
    coordinates: Tensor,
    observed: Tensor,
    *,
    dtype: torch.dtype | None = None,
    feature: str = "coordinates",
) -> Tensor:
    r"""Project coordinates with contextual validation errors for a model feature.

    Args:
        coordinates: Floating-point XYZ tensor.
        observed: Boolean availability mask over its coordinate positions.
        dtype: Optional model dtype; None retains the input dtype.
        feature: Feature name included in validation errors.

    Returns:
        Projected coordinates with the input's layout and optional requested dtype.
        Canonical observations and the supplied mask remain unchanged.

    Raises:
        InvalidRequestError: Available values are nonfinite or input types, axes,
            layouts, or devices are invalid.
        UnsupportedError: Finite available values cannot be represented in the model dtype.

    Notes:
        Unavailable payload and its derivatives are zeroed before arithmetic or
        conversion. See `ofoldx.modules.atom.coordinates.project_coordinates` for the
        tensor and compiled-execution behavior.
    """
    try:
        return project_coordinates(coordinates, observed, dtype=dtype)
    except OverflowError as error:
        _numeric(feature, f"not representable in the requested dtype: {error}")
    except (TypeError, ValueError) as error:
        _invalid(feature, str(error))


def gather_model_coordinates(
    coordinates: Tensor,
    coordinate_mask: Tensor,
    atom_index: Tensor,
    atom_mask: Tensor,
    *,
    feature: str = "coordinates",
) -> tuple[Tensor, Tensor]:
    """Gather model coordinates through a validated observation correspondence.

    Unmatched (-1) or unavailable atoms return zero and false availability.
    Caller map, axis and finite-value violations receive contextual request
    errors; unexpected implementation exceptions retain their identity.
    """
    try:
        return gather_observed_coordinates(coordinates, coordinate_mask, atom_index, atom_mask)
    except (TypeError, ValueError) as error:
        _invalid(feature, str(error))


def project_model_features(
    values: Tensor,
    observed: Tensor,
    *,
    dtype: torch.dtype | None = None,
    feature: str = "features",
) -> Tensor:
    """Project an explicitly masked finite continuous model feature.

    The caller supplies a boolean mask broadcastable to the feature axes.
    Unavailable values have zero output and derivatives. This operation is for
    continuous features; categorical indices and logit exclusions have their
    own contracts. Canonical inputs and the supplied mask are not changed.

    Raises:
        InvalidRequestError: If active values are nonfinite or the mask, dtype
            or broadcast axes are invalid.
        UnsupportedError: If finite values overflow the model dtype.
    """
    if not values.is_floating_point() or (dtype is not None and not dtype.is_floating_point):
        _invalid(feature, "a floating-point dtype is required")
    if observed.dtype != torch.bool or observed.device != values.device:
        _invalid(feature, "a boolean mask on the feature device is required")
    try:
        shape = torch.broadcast_shapes(observed.shape, values.shape)
    except RuntimeError:
        _invalid(feature, "the mask does not match the feature axes")
    if shape != values.shape:
        _invalid(feature, "the mask introduces feature axes")
    result = torch.where(observed, values, 0)
    active = result.concat if isinstance(result, NestedTensor) else result
    finite = torch.isfinite(active).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite, f"active {feature} must be finite")
    elif not bool(finite):
        _invalid(feature, "active values are not finite")
    if dtype is None or dtype == result.dtype:
        return result
    result = result.to(dtype=dtype)
    lowered = result.concat if isinstance(result, NestedTensor) else result
    finite = torch.isfinite(lowered).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite, f"{feature} exceed the requested dtype range")
    elif not bool(finite):
        _numeric(feature, "not representable in the requested dtype")
    return result


def project_model_categories(
    values: Tensor, available: Tensor, *, num_classes: int, feature: str = "categorical features"
) -> Tensor:
    """Validate one model's declared categorical domain before lookup."""
    try:
        return project_categories(values, available, num_classes=num_classes)
    except (TypeError, ValueError) as error:
        _invalid(feature, str(error))


def project_model_counts(values: Tensor, available: Tensor, *, feature: str = "raw counts") -> Tensor:
    """Select and validate raw count facts before any model feature transform."""
    try:
        return project_counts(values, available)
    except (TypeError, ValueError) as error:
        _invalid(feature, str(error))


def model_log_probabilities(logits: Tensor, available: Tensor, *, feature: str = "classification logits") -> Tensor:
    """Normalize available categorical rows, retaining explicit class exclusions.

    ``available`` is a boolean row mask broadcastable over the leading axes.
    Unavailable rows have uniform fill probabilities and zero logit derivatives;
    that fill asserts no categorical observation. Available rows may exclude
    classes with -Inf, but require one finite class and contain no NaN or +Inf.
    The consumer separately checks that its teacher or sampled class is admitted.
    """
    if not logits.is_floating_point() or logits.dim() == 0 or logits.shape[-1] == 0:
        _invalid(feature, "a nonempty floating-point vocabulary axis is required")
    if available.dtype != torch.bool or available.device != logits.device:
        _invalid(feature, "a boolean row mask on the logit device is required")
    try:
        shape = torch.broadcast_shapes(available.shape, logits.shape[:-1])
    except RuntimeError:
        _invalid(feature, "the availability does not match the row axes")
    if shape != logits.shape[:-1]:
        _invalid(feature, "the availability introduces row axes")
    selected = torch.where(available.unsqueeze(-1), logits, 0.0)
    values = selected.concat if isinstance(selected, NestedTensor) else selected
    finite = torch.isfinite(values)
    valid = ((finite | torch.isneginf(values)).all(dim=-1) & finite.any(dim=-1)).all()
    if torch.compiler.is_compiling():
        torch._assert_async(valid, f"available {feature} require one finite class and no NaN/+Inf")
    elif not bool(valid):
        _invalid(feature, "an available row needs one finite class and no NaN/+Inf")
    result = selected.log_softmax(dim=-1)
    normalized = result.concat if isinstance(result, NestedTensor) else result
    represented = (torch.isfinite(normalized) | torch.isneginf(values)).all()
    if torch.compiler.is_compiling():
        torch._assert_async(represented, f"{feature} normalization exceeds the model dtype range")
    elif not bool(represented):
        _numeric(feature, "the normalisation exceeds the dtype range")
    return result


def model_frame_rotation(
    x_vector: Tensor, xy_vector: Tensor, available: Tensor, *, feature: str = "physical frames"
) -> Tensor:
    """Construct available right-handed physical frames with exact normalization.

    The x-axis follows ``x_vector``; the xy-plane contains ``xy_vector``.
    Callers select unavailable point coordinates out before deriving these
    vectors. No regularization epsilon supplies a frame for degenerate geometry.
    Unavailable rows return identity fill with zero vector derivatives. The
    operation supports canonical leading ragged axes and static sample batches.
    """
    if x_vector.shape != xy_vector.shape or x_vector.dtype != xy_vector.dtype or x_vector.device != xy_vector.device:
        _invalid(feature, "the vectors differ in XYZ axes, dtype or device")
    if isinstance(x_vector, NestedTensor) != isinstance(xy_vector, NestedTensor):
        _invalid(feature, "the vectors differ in dense or nested layout")
    x = project_model_coordinates(x_vector, available, feature=f"{feature} x vectors")
    xy = project_model_coordinates(xy_vector, available, feature=f"{feature} xy vectors")
    if isinstance(x, NestedTensor):
        other = cast(NestedTensor, xy)
        if x.ragged_dims != other.ragged_dims or x.packed_dim_order != other.packed_dim_order:
            _invalid(feature, "the vectors differ in nested layout")
        sizes_match = (x.element_sizes() == other.element_sizes()).all()
        if torch.compiler.is_compiling():
            torch._assert_async(sizes_match, f"{feature} vectors must have matching nested cardinality")
        elif not bool(sizes_match):
            _invalid(feature, "the vectors differ in nested cardinality")
        if x.ragged_dims == (1,) and x.batch_first and x.dim() > 3:
            batch_size, num_samples = x.shape[:2]
            mask = available if isinstance(available, NestedTensor) else x[..., 0].nested_like(available)
            frames = model_frame_rotation(flatten_batch(x), flatten_batch(other), flatten_batch(mask), feature=feature)
            return unflatten_batch(frames, (batch_size, num_samples))
        rank = x.dim() - 1
        if x.ragged_dims != tuple(range(len(x.ragged_dims))) or x.packed_dim_order != tuple(range(rank)):
            fail("model_capacity", RequestRef(feature, "frame axes are not the leading ragged axes"))
        mask = available if isinstance(available, NestedTensor) else x[..., 0].nested_like(available)
        frames = model_frame_rotation(x.concat, other.concat, mask.concat, feature=feature)
        return x.packed_with_static_tail(frames)

    # Scale before norm/cross arithmetic: only directions define a frame, and
    # representable large or small vector magnitudes must not overflow a norm.
    work_x = x.double()
    work_xy = xy.double()
    axis_x = torch.tensor((1.0, 0.0, 0.0), dtype=work_x.dtype, device=work_x.device)
    axis_y = torch.tensor((0.0, 1.0, 0.0), dtype=work_x.dtype, device=work_x.device)
    work_x = torch.where(available.unsqueeze(-1), work_x, axis_x)
    work_xy = torch.where(available.unsqueeze(-1), work_xy, axis_y)
    scale_x = work_x.abs().amax(dim=-1, keepdim=True)
    scale_xy = work_xy.abs().amax(dim=-1, keepdim=True)
    nonzero = ((scale_x > 0) & (scale_xy > 0)).all()
    if torch.compiler.is_compiling():
        torch._assert_async(nonzero, f"{feature} requires nonzero defining vectors")
    elif not bool(nonzero):
        _numeric(feature, "a defining vector is zero")
    scaled_x, scaled_xy = work_x / scale_x, work_xy / scale_xy
    x_axis = scaled_x / scaled_x.square().sum(dim=-1, keepdim=True).sqrt()
    normal = torch.cross(scaled_x, scaled_xy, dim=-1)
    scale_normal = normal.abs().amax(dim=-1, keepdim=True)
    independent = (scale_normal > 0).all()
    if torch.compiler.is_compiling():
        torch._assert_async(independent, f"{feature} requires noncollinear representable defining vectors")
    elif not bool(independent):
        _numeric(feature, "the defining vectors are collinear")
    normal = normal / scale_normal
    z_axis = normal / normal.square().sum(dim=-1, keepdim=True).sqrt()
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1).to(x_vector.dtype)
    finite = torch.isfinite(rotation).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite, f"{feature} exceeds the model numeric range")
    elif not bool(finite):
        _numeric(feature, "the frames exceed the numeric range")
    return rotation


def augment_reference_positions(
    positions: Tensor,
    space_ids: Tensor,
    available: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> None:
    r"""Randomly rotate and translate each space with available reference atoms.

    Active spaces move about the available-atom centroid. Unavailable rows are
    zero-filled and cannot influence available coordinates or their gradients.
    """

    positions.copy_(project_model_coordinates(positions, available, feature="reference positions"))
    if not len(space_ids):
        return
    active_space_ids = torch.unique(space_ids[available & (space_ids >= 0)], sorted=True)
    num_active_spaces = active_space_ids.numel()
    if not num_active_spaces:
        return

    random_device = positions.device if generator is None else generator.device
    random_values = torch.randn(
        num_active_spaces,
        7,
        dtype=positions.dtype,
        device=random_device,
        generator=generator,
    ).to(positions.device)
    rotations = quaternion_to_rotation(random_values[:, :4])
    translations = random_values[:, 4:]

    for space_id, rotation, translation in zip(active_space_ids, rotations, translations):
        included = space_ids == space_id
        mask = available[included]
        values = positions[included]
        center = weighted_mean(values.transpose(-1, -2), mask.unsqueeze(-2)).unsqueeze(-2)
        transformed = rotate_vectors(values - center, rotation) + translation
        positions[included] = transformed.masked_fill(~mask[:, None], 0)
