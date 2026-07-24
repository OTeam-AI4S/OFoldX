r"""
Frame geometry for frame-based confidence losses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def insert_sample_axis(index: Tensor, num_samples: int, *, dim: int = -2) -> Tensor:
    r"""
    Insert a sample axis into a gather index and expand it to ``num_samples``.

    ``dim`` is the (negative) position of the new sample axis, placed immediately before the index's
    token axis: ``-2`` for a per-token index ``[..., N_token]`` (token at ``-1``) and ``-3`` for a
    frame index ``[..., N_token, 3]`` (token at ``-2``). Every other axis is expanded with ``-1`` (left
    as-is); only the ``num_samples`` axis is materialised.

    Args:
        index: Gather index tensor of shape `[..., N_token]` or `[..., N_token, 3]`.
        num_samples: Number of samples to broadcast the new axis over.
        dim: Negative axis where the sample dimension should be inserted.

    Returns:
        Expanded index tensor with a materialized sample axis of size ``num_samples``.
    """

    expanded = index.unsqueeze(dim)
    sizes = [-1] * expanded.dim()
    sizes[dim] = num_samples
    return expanded.expand(*sizes)


def gather_atoms_by_index(coords: Tensor, atom_index: Tensor) -> Tensor:
    r"""
    Gather coords of the atoms named by a per-position global atom index.

    The gather index expands the trailing coordinate axis with ``-1`` on every existing dim, so the
    leading batch/atom/token axes are preserved as-is: the only explicit size is the static channel
    count ``coords.shape[-1]``.

    Args:
        coords: `[..., N_atom, C]` atom coords.
        atom_index: `[..., M]` global atom indices (same leading dims as ``coords``).

    Returns:
        `[..., M, C]` coords gathered at the requested indices.
    """

    index = atom_index[..., None].expand(*([-1] * atom_index.dim()), coords.shape[-1])
    return torch.gather(coords, dim=-2, index=index)
