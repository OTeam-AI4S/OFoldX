r"""
Frame geometry for frame-based confidence losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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

    Examples:
        >>> import torch
        >>> index = torch.tensor([[0, 1, 2]])
        >>> insert_sample_axis(index, 4).shape
        torch.Size([1, 4, 3])
        >>> frame_index = torch.tensor([[[0, 1, 2]]])
        >>> insert_sample_axis(frame_index, 4, dim=-3).shape
        torch.Size([1, 4, 1, 3])
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

    Examples:
        >>> import torch
        >>> coords = torch.arange(12.0).reshape(1, 4, 3)
        >>> atom_index = torch.tensor([[0, 2]])
        >>> gather_atoms_by_index(coords, atom_index).tolist()
        [[[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]]
    """

    index = atom_index[..., None].expand(*([-1] * atom_index.dim()), coords.shape[-1])
    return torch.gather(coords, dim=-2, index=index)


def gather_frame_atoms(coords: Tensor, frame_atom_index: Tensor) -> Tensor:
    r"""
    Gather each token's three frame atom coords.

    Args:
        coords: `[..., N_atom, 3]` atom coords.
        frame_atom_index: `[..., N_token, 3]` global atom indices of each token's frame atoms.

    Returns:
        `[..., N_token, 3, 3]` frame atom coords (three atoms x xyz). Leading dims of
        ``coords`` and ``frame_atom_index`` must match.

    Examples:
        >>> import torch
        >>> coords = torch.arange(15.0).reshape(1, 5, 3)
        >>> frame_atom_index = torch.tensor([[[0, 1, 2]]])
        >>> frames = gather_frame_atoms(coords, frame_atom_index)
        >>> frames.shape
        torch.Size([1, 1, 3, 3])
        >>> frames.tolist()
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]
    """

    # Merge the (N_token, 3) frame axes into one gather axis with ``-1`` (token count inferred), gather the
    # three frame atoms, then split the static trailing 3 back out.
    flat_index = frame_atom_index.reshape(*frame_atom_index.shape[:-2], -1)  # [..., N_token * 3]
    gathered = gather_atoms_by_index(coords, flat_index)  # [..., N_token * 3, 3]
    return gathered.reshape(*gathered.shape[:-2], -1, 3, 3)


def express_coords_in_frame(coords: Tensor, frames: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Express coords in each frame's local basis.

    Each frame's orthonormal basis is built from its three atoms: ``b`` is the origin, and the axes
    come from a Gram-Schmidt-style combination of ``a - b`` and ``c - b`` with a right-handed third
    axis. Coords are then projected onto that basis.

    Args:
        coords: `[..., N_atom, 3]` coords to express.
        frames: `[..., N_frame, 3, 3]` three frame atoms ``(a, b, c)`` per frame; ``b`` is the
            origin and the basis is built from ``a - b`` and ``c - b``.
        eps: Floor for the basis-vector normalization denominators.

    Returns:
        `[..., N_frame, N_atom, 3]` coords in each frame's local basis.

    Examples:
        >>> import torch
        >>> frames = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        >>> coords = torch.tensor([[[1.0, 1.0, 0.0]]])
        >>> local = express_coords_in_frame(coords, frames)
        >>> local.shape
        torch.Size([1, 1, 1, 3])
        >>> round(float(local[0, 0, 0, 0]), 4)
        1.4142
    """

    # Static slices over the size-3 frame-atom axis (a/b/c) + a ``cat`` of the three unsqueezed
    # projections assemble the trailing basis axis (equivalent to torch.unbind + torch.stack).
    a = frames[..., 0, :]  # each [..., N_frame, 3]
    b = frames[..., 1, :]
    c = frames[..., 2, :]
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)  # [..., N_frame, 3]
    offset = coords[..., None, :, :] - b[..., None, :]  # [..., N_frame, N_atom, 3]
    return torch.cat(
        [
            (offset * e1[..., None, :]).sum(dim=-1, keepdim=True),
            (offset * e2[..., None, :]).sum(dim=-1, keepdim=True),
            (offset * e3[..., None, :]).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )


def frame_aligned_error_squared(
    pred_coords: Tensor,
    true_coords: Tensor,
    pred_frames: Tensor,
    true_frames: Tensor,
) -> Tensor:
    r"""
    Per-frame, per-token squared aligned error.

    Each token's frame-center coordinate is expressed in every frame's local basis for both the
    prediction and the ground truth; the squared difference between the two, summed over xyz, is the
    per-frame/per-token aligned error that the PAE head is trained against.

    Args:
        pred_coords: `[..., N_sample, N_token, 3]` predicted per-token frame-center coords.
        true_coords: `[..., N_token, 3]` ground-truth per-token frame-center coords.
        pred_frames: `[..., N_sample, N_frame, 3, 3]` predicted frames.
        true_frames: `[..., N_frame, 3, 3]` ground-truth frames.

    Returns:
        `[..., N_sample, N_frame, N_token]` squared aligned error.

    Examples:
        >>> import torch
        >>> frame = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        >>> pred_coords = torch.zeros(1, 2, 1, 3)
        >>> true_coords = torch.zeros(1, 1, 3)
        >>> pred_frames = frame.unsqueeze(-4).expand(1, 2, 1, 3, 3).contiguous()
        >>> error = frame_aligned_error_squared(pred_coords, true_coords, pred_frames, frame)
        >>> error.shape
        torch.Size([1, 2, 1, 1])
        >>> error.round(decimals=4).tolist()
        [[[[0.0]], [[0.0]]]]
    """

    transformed_pred = express_coords_in_frame(pred_coords, pred_frames)
    transformed_true = express_coords_in_frame(true_coords, true_frames)
    return ((transformed_pred - transformed_true.unsqueeze(dim=-4)) ** 2).sum(dim=-1)
