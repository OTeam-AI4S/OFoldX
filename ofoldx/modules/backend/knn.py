r"""Topology-dispatched masked k-nearest neighbours."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ._oops import oops_kernel

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
) -> tuple[Tensor | NestedTensor, Tensor | NestedTensor, Tensor | NestedTensor]:
    r"""Return fixed-width sample-local neighbours under a validity mask.

    Dense tensors use the public PyTorch equation. DanLing ``NestedTensor``
    inputs are already the physical packed batch and dispatch directly to OOps;
    the function never converts between dense and nested layouts.
    """

    nested = isinstance(coordinates, NestedTensor)
    if nested != isinstance(mask, NestedTensor):
        raise TypeError("masked_knn coordinates and mask must both be dense or both be NestedTensor")
    if nested:
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
    channels = nodes.shape[-1]
    neighbors = neighbor_indices.shape[-1]
    return torch.gather(
        nodes.unsqueeze(-2).expand(-1, -1, neighbors, channels),
        -3,
        neighbor_indices.unsqueeze(-1).expand(-1, -1, neighbors, channels),
    )


def _masked_knn_eager(
    coordinates: Tensor,
    mask: Tensor,
    num_neighbors: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""PyTorch reference for the packed OOps masked-kNN contract."""

    if coordinates.shape[:-1] != mask.shape or coordinates.shape[-1] != 3:
        raise ValueError("coordinates and mask must have shapes (..., N, 3) and (..., N)")
    if not 1 <= num_neighbors <= 128:
        raise ValueError("num_neighbors must lie in [1, 128]")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    length = coordinates.shape[-2]
    output_shape = (*mask.shape, num_neighbors)
    if length == 0:
        return (
            coordinates.new_empty(output_shape),
            torch.empty(output_shape, device=coordinates.device, dtype=torch.long),
            torch.empty(output_shape, device=coordinates.device, dtype=torch.bool),
        )

    truth = mask.bool()
    floating = coordinates.float()
    distance = torch.sqrt((floating.unsqueeze(-2) - floating.unsqueeze(-3)).square().sum(dim=-1) + eps)
    candidate_valid = truth.unsqueeze(-1) & truth.unsqueeze(-2)
    adjusted = torch.where(candidate_valid, distance, torch.full_like(distance, torch.inf))
    count = min(length, num_neighbors)
    order = torch.argsort(adjusted, dim=-1, stable=True)[..., :count]
    valid = candidate_valid.gather(-1, order)
    selected = distance.gather(-1, order)
    indices = torch.where(valid, order, torch.zeros_like(order))
    row_max = torch.where(candidate_valid, distance, torch.zeros_like(distance)).max(dim=-1).values
    filler = row_max.detach().add(1).unsqueeze(-1)
    selected = torch.where(valid, selected, filler)

    if count < num_neighbors:
        padding = num_neighbors - count
        selected = torch.cat((selected, filler.expand(*filler.shape[:-1], padding)), dim=-1)
        indices = F.pad(indices, (0, padding), value=0)
        valid = F.pad(valid, (0, padding), value=False)
    return selected.to(dtype=coordinates.dtype), indices, valid


__all__ = ["gather_nodes", "masked_knn"]
