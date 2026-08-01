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

r"""Atom-coordinate gathers and observation masks."""

from __future__ import annotations

from typing import cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor


def _finite_coordinates(values: Tensor, observed: Tensor, dtype: torch.dtype | None) -> Tensor:
    values = torch.where(observed.unsqueeze(-1), values, 0)
    finite = torch.isfinite(values).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite, "observed coordinates must be finite")
    elif not bool(finite):
        raise ValueError("observed coordinates must be finite")
    if dtype is None or dtype == values.dtype:
        return values
    lowered = values.to(dtype=dtype)
    finite = torch.isfinite(lowered).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite, "observed coordinates exceed the requested dtype range")
    elif not bool(finite):
        raise OverflowError("observed coordinates exceed the requested dtype range")
    return lowered


def project_coordinates(coordinates: Tensor, observed: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    r"""Project available XYZ values into a finite numerical coordinate field.

    Args:
        coordinates: Floating-point XYZ tensor with a final axis of length three.
        observed: Boolean availability mask with shape `coordinates.shape[:-1]`.
        dtype: Optional floating-point output dtype; None retains the input dtype.

    Returns:
        Coordinates with the requested dtype and the input's shape and nested layout.
        Unavailable positions have zero values and zero coordinate derivatives.

    Raises:
        TypeError: Coordinates or the requested dtype are not floating, or the
            observation mask is not boolean.
        ValueError: Axes, layouts, or devices disagree, or available values are nonfinite.
        OverflowError: Finite available values overflow the requested dtype.

    Notes:
        Availability is applied before arithmetic or dtype conversion, including when
        excluded payload contains NaN or infinity. The supplied mask is unchanged.
        Nested masks can share the coordinate layout or use the exact dense logical
        shape; padding is ignored. Compiled calls retain device assertions for numeric
        checks, whose failures are reported by the selected compilation backend.

    Examples:
        >>> import torch
        >>> coordinates = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0]])
        >>> observed = torch.tensor([True, False])
        >>> project_coordinates(coordinates, observed).tolist()
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
    """
    if not coordinates.is_floating_point() or (dtype is not None and not dtype.is_floating_point):
        raise TypeError("coordinates and their requested dtype must be floating point")
    if observed.dtype != torch.bool:
        raise TypeError("coordinate observation mask must be boolean")
    if coordinates.dim() == 0 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have a final XYZ axis")
    if observed.shape != coordinates.shape[:-1] or observed.device != coordinates.device:
        raise ValueError("coordinate observation axes and device must match coordinates")
    if not isinstance(coordinates, NestedTensor):
        if isinstance(observed, NestedTensor):
            raise ValueError("dense coordinates require a dense observation mask")
        return _finite_coordinates(coordinates, observed, dtype)

    xyz = coordinates.dim() - 2
    if xyz in coordinates.ragged_dims:
        raise ValueError("the XYZ axis must be static in nested coordinates")
    mask = (
        observed
        if isinstance(observed, NestedTensor)
        else cast(NestedTensor, coordinates[..., 0].nested_like(observed))
    )
    if mask.ragged_dims != coordinates.ragged_dims:
        raise ValueError("nested coordinate observation layouts must match")
    same_shapes = (mask.element_sizes() == coordinates.element_sizes()[:, :-1]).all()
    if torch.compiler.is_compiling():
        torch._assert_async(same_shapes, "nested coordinate observation shapes must match")
    elif not bool(same_shapes):
        raise ValueError("nested coordinate observation shapes must match")
    ragged_rank = len(coordinates.ragged_dims)
    static_order = coordinates.packed_dim_order[ragged_rank:]
    xyz_axis = 1 + static_order.index(xyz)
    point_order = tuple(dim for dim in static_order if dim != xyz)
    mask_order = mask.packed_dim_order[ragged_rank:]
    mask_values = mask.concat.permute(0, *(1 + mask_order.index(dim) for dim in point_order))
    values = coordinates.concat.movedim(xyz_axis, -1)
    projected = _finite_coordinates(values, mask_values, dtype).movedim(-1, xyz_axis)
    return coordinates.packed_like(projected)


def gather_atoms_by_index(coords: Tensor, atom_index: Tensor) -> Tensor:
    r"""Gather atom coordinates through a per-position global atom index."""

    index = atom_index[..., None].expand(*([-1] * atom_index.dim()), coords.shape[-1])
    return torch.gather(coords, dim=-2, index=index)


def gather_observed_coordinates(
    coordinates: Tensor,
    coordinate_mask: Tensor,
    atom_index: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Gather observed coordinates, excluding unknown or masked model atoms.

    The source atom axis is the penultimate coordinate axis and the final mask
    axis. Model atom indices may be ``-1`` for unavailable correspondences.
    Excluded coordinates return zero and carry zero coordinate gradients.
    """

    if coordinates.dim() < 2 or coordinates.shape[-1] != 3 or not coordinates.is_floating_point():
        raise ValueError("source coordinates must be floating XYZ values")
    if coordinates.shape[:-2] != atom_index.shape[:-1]:
        raise ValueError("source and model atom batch axes must match")
    if atom_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("atom correspondence indices must be integers")
    if coordinate_mask.dtype != torch.bool or atom_mask.dtype != torch.bool:
        raise TypeError("coordinate and model atom masks must be boolean")
    if coordinate_mask.shape != coordinates.shape[:-1] or atom_mask.shape != atom_index.shape:
        raise ValueError("coordinate and model atom masks must match their axes")
    if any(value.device != coordinates.device for value in (coordinate_mask, atom_index, atom_mask)):
        raise ValueError("coordinates, masks and correspondence indices must share a device")
    atom_index = atom_index.long()
    in_range = ((atom_index >= -1) & (atom_index < coordinates.shape[-2])).all()
    if torch.compiler.is_compiling():
        torch._assert_async(in_range, "atom correspondence indices must be -1 or valid source rows")
    elif not bool(in_range):
        raise ValueError("atom correspondence indices must be -1 or valid source rows")
    valid = atom_index >= 0
    if coordinates.shape[-2] == 0:
        empty = coordinates.sum(dim=-2, keepdim=True).expand(*atom_index.shape, 3)
        return empty, torch.zeros_like(atom_index, dtype=torch.bool)
    gather_index = atom_index.clamp_min(0)
    gathered = coordinates.gather(
        -2,
        gather_index.unsqueeze(-1).expand(*([-1] * gather_index.dim()), coordinates.shape[-1]),
    )
    observed = coordinate_mask.gather(-1, gather_index) & valid & atom_mask
    return project_coordinates(gathered, observed), observed
