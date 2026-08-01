r"""
Frame geometry for frame-based confidence losses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def insert_sample_axis(index: Tensor, num_samples: int, *, dim: int = -2) -> Tensor:
    r"""Insert and expand the sample axis of a token or frame gather index."""

    expanded = index.unsqueeze(dim)
    sizes = [-1] * expanded.dim()
    sizes[dim] = num_samples
    return expanded.expand(*sizes)


def gather_atoms_by_index(coords: Tensor, atom_index: Tensor) -> Tensor:
    r"""Gather atom coordinates through a per-position global atom index."""

    index = atom_index[..., None].expand(*([-1] * atom_index.dim()), coords.shape[-1])
    return torch.gather(coords, dim=-2, index=index)
