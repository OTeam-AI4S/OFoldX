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

r"""Topology-dispatched masked k-nearest neighbours."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ..._oops import oops_kernel
from .resolve import resolve_tensor_backend

_OopsMaskedKnn = Callable[
    [NestedTensor, NestedTensor, int, float],
    tuple[NestedTensor, NestedTensor, NestedTensor],
]
_oops_masked_knn = cast(_OopsMaskedKnn, oops_kernel("masked_knn"))


def masked_knn(
    coordinates: Tensor | NestedTensor,
    mask: Tensor | NestedTensor,
    num_neighbors: int,
    eps: float = 1e-6,
    *,
    backend: str = "auto",
) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor, Tensor | NestedTensor]:
    r"""Return fixed-width sample-local neighbours under a validity mask.

    Dense tensors use the public PyTorch equation. Automatic acceleration
    selects OOps for its canonical CUDA nested contract. The native backend
    applies the same equation independently over actual nested samples.
    """

    nested = isinstance(coordinates, NestedTensor)
    if nested != isinstance(mask, NestedTensor):
        raise TypeError("masked_knn coordinates and mask must both be dense or both be NestedTensor")
    supported = ("oops", "eager") if nested else ("eager",)
    if nested and backend == "auto":
        reference = cast(NestedTensor, coordinates)
        nested_mask = cast(NestedTensor, mask)
        if not (
            reference.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and nested_mask.dtype
            in (
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
            and reference.batch_first
            and nested_mask.batch_first
            and reference.dim() == 3
            and nested_mask.dim() == 2
            and reference.ragged_dims == nested_mask.ragged_dims == (0,)
            and reference.packed_dim_order == (0, 1)
            and nested_mask.packed_dim_order == (0,)
            and 1 <= num_neighbors <= 128
            and eps <= torch.finfo(torch.float32).max
        ):
            supported = ("eager",)
    backend = resolve_tensor_backend(
        backend,
        tensors=(coordinates, mask),
        supported=supported,
        public_default="eager",
        oops=_oops_masked_knn,
    )
    if nested:
        if backend != "oops":
            reference = cast(NestedTensor, coordinates)
            samples = tuple(
                _masked_knn_eager(sample, sample_mask, num_neighbors, eps)
                for sample, sample_mask in zip(reference, cast(NestedTensor, mask), strict=True)
            )
            outputs = [
                NestedTensor(
                    [sample[axis] for sample in samples],
                    ragged_dims=reference.ragged_dims,
                    batch_first=reference.batch_first,
                )
                for axis in range(3)
            ]
            return outputs[0], outputs[1], outputs[2]
        return _oops_masked_knn(
            cast(NestedTensor, coordinates),
            cast(NestedTensor, mask),
            num_neighbors,
            eps,
        )
    return _masked_knn_eager(
        cast(Tensor, coordinates),
        cast(Tensor, mask),
        num_neighbors,
        eps,
    )


def gather_nodes(
    nodes: Tensor | NestedTensor,
    neighbor_indices: Tensor | NestedTensor,
) -> Tensor | NestedTensor:
    r"""Gather sample-local node rows for a fixed-width neighbor list."""

    nested = isinstance(nodes, NestedTensor)
    if nested != isinstance(neighbor_indices, NestedTensor):
        raise TypeError("gather_nodes operands must both be dense or both be NestedTensor")
    if nested and not cast(NestedTensor, nodes).batch_first:
        reference = cast(NestedTensor, nodes)
        return NestedTensor(
            [
                gather_nodes(sample.unsqueeze(0), indices.unsqueeze(0))[0]
                for sample, indices in zip(reference, cast(NestedTensor, neighbor_indices), strict=True)
            ],
            ragged_dims=reference.ragged_dims,
            batch_first=False,
        )
    feature_size = nodes.shape[-1]
    num_neighbors = neighbor_indices.shape[-1]
    return torch.gather(
        nodes.unsqueeze(-2).expand(*((-1,) * (nodes.dim() - 1)), num_neighbors, feature_size),
        -3,
        neighbor_indices.unsqueeze(-1).expand(*((-1,) * neighbor_indices.dim()), feature_size),
    )


def _masked_knn_eager(
    coordinates: Tensor,
    mask: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""PyTorch reference for the packed OOps masked-kNN contract."""

    from ..functional import _scaled_power
    from ..geometry import smoothed_vector_norm

    if coordinates.shape[:-1] != mask.shape or coordinates.shape[-1] != 3:
        raise ValueError("coordinates and mask must have shapes (..., N, 3) and (..., N)")
    if num_neighbors < 0:
        raise ValueError("num_neighbors must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    num_points = coordinates.shape[-2]
    output_shape = (*mask.shape, num_neighbors)
    if num_points == 0:
        return (
            coordinates.sum(dim=-1, keepdim=True).expand(output_shape),
            torch.empty(output_shape, device=coordinates.device, dtype=torch.long),
            torch.empty(output_shape, device=coordinates.device, dtype=torch.bool),
        )

    truth = mask.bool()
    physical_coordinates = torch.where(truth.unsqueeze(-1), coordinates, 0.0).double()
    displacement = physical_coordinates.unsqueeze(-2) - physical_coordinates.unsqueeze(-3)
    candidate_valid = truth.unsqueeze(-1) & truth.unsqueeze(-2)
    num_selected_neighbors = min(num_points, num_neighbors)
    selection_displacement = displacement.detach()
    if coordinates.dtype == torch.float64:
        # A squared Float64 distance can overflow or underflow while its norm
        # remains representable. Sort its binary exponent and mantissa instead.
        _, exponent = torch.frexp(selection_displacement.abs().amax(dim=-1))
        scaled = _scaled_power(selection_displacement, -exponent.unsqueeze(-1))
        mantissa, squared_exponent = torch.frexp(scaled.square().sum(dim=-1))
        squared_exponent = squared_exponent + 2 * exponent
        squared_exponent = torch.where(mantissa == 0, torch.iinfo(exponent.dtype).min, squared_exponent)
        squared_exponent = torch.where(
            candidate_valid & torch.isfinite(mantissa), squared_exponent, torch.iinfo(exponent.dtype).max
        )
        mantissa = torch.where(candidate_valid, mantissa, torch.inf)
        order = torch.argsort(mantissa, dim=-1, stable=True)
        exponent_order = torch.argsort(squared_exponent.gather(-1, order), dim=-1, stable=True)
        order = order.gather(-1, exponent_order)[..., :num_selected_neighbors]
    else:
        squared_distance = selection_displacement.square().sum(dim=-1)
        squared_distance = torch.where(candidate_valid, squared_distance, torch.inf)
        order = torch.argsort(squared_distance, dim=-1, stable=True)[..., :num_selected_neighbors]
    distance = smoothed_vector_norm(displacement, eps=eps, backend="eager")
    valid = candidate_valid.gather(-1, order)
    selected = distance.gather(-1, order)
    indices = torch.where(valid, order, torch.zeros_like(order))
    row_max = torch.where(candidate_valid, distance, torch.zeros_like(distance)).max(dim=-1).values
    filler = row_max.detach().add(1).unsqueeze(-1)
    selected = torch.where(valid, selected, filler)

    if num_selected_neighbors < num_neighbors:
        padding = num_neighbors - num_selected_neighbors
        selected = torch.cat((selected, filler.expand(*filler.shape[:-1], padding)), dim=-1)
        indices = F.pad(indices, (0, padding), value=0)
        valid = F.pad(valid, (0, padding), value=False)
    return selected.to(dtype=coordinates.dtype), indices, valid
