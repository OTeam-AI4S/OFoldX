r"""
Local atom windowing modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def gather_key_windows(single: Tensor, num_query_atoms: int, num_key_atoms: int) -> Tensor:
    r"""
    Gather each query window's overlapping key-atom neighbourhood from a flat ``(B, N, D)`` atom tensor.

    Uses a strided ``unfold`` view (O(N), no matmul) instead of a one-hot indexing-matrix einsum (O(num_windows²)),
    producing exactly the same result.

    Args:
        single: ``[..., N_atom, C]`` atom features.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.

    Returns:
        ``[..., N_window, num_key_atoms, C]`` key windows.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import gather_key_windows
        >>> single = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        >>> gather_key_windows(single, num_query_atoms=2, num_key_atoms=4).shape
        torch.Size([1, 3, 4, 1])
    """
    num_atoms = single.shape[-2]
    num_windows = (num_atoms + num_query_atoms - 1) // num_query_atoms
    key_padding_left = (num_key_atoms - num_query_atoms) // 2
    key_padding_right = int((num_windows - 0.5) * num_query_atoms + num_key_atoms / 2 - num_atoms + 0.5)
    padded = F.pad(single, (0, 0, key_padding_left, key_padding_right))
    return padded.unfold(-2, num_key_atoms, num_query_atoms).movedim(-1, -2)


@dataclass(slots=True)
class LocalAtomWindowInfo:
    r"""
    Padding and masking metadata describing one local-window partition of an atom axis.

    Records how the atom axis was padded to fill whole windows and which query/key positions are
    real, so downstream attention can mask out the padded slots without re-deriving them.

    Attributes:
        attention_mask: `[..., N_window, num_query_atoms, num_key_atoms]` boolean mask; ``True``
            where both the query and key atom of a window slot are real (not padding).
        query_padding: Number of padded query atoms appended to reach a whole number of windows.
        key_padding_left: Number of padded key atoms before the first real atom.
        key_padding_right: Number of padded key atoms after the last real atom.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import LocalAtomWindowInfo
        >>> info = LocalAtomWindowInfo(torch.ones(1, 3, 2, 4, dtype=torch.bool), 0, 1, 1)
        >>> info.query_padding, info.key_padding_left, info.key_padding_right
        (0, 1, 1)
    """

    attention_mask: Tensor
    query_padding: int
    key_padding_left: int
    key_padding_right: int


@dataclass(slots=True)
class LocalAtomWindows:
    r"""
    Query and key windows carved from a single atom-axis tensor, plus their padding metadata.

    Attributes:
        query: `[..., N_window, num_query_atoms, C]` non-overlapping query windows.
        key: `[..., N_window, num_key_atoms, C]` overlapping key neighbourhoods, one per query window.
        info: Padding and mask metadata for the windows.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import LocalAtomWindowing
        >>> windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(torch.zeros(1, 6, 1))
        >>> windows.query.shape, windows.key.shape
        (torch.Size([1, 3, 2, 1]), torch.Size([1, 3, 4, 1]))
    """

    query: Tensor
    key: Tensor
    info: LocalAtomWindowInfo


@dataclass(slots=True)
class LocalAtomGeometry:
    r"""
    Windowed atom-pair geometry (displacements and same-space mask) for atom attention blocks.

    Attributes:
        distance_vectors: `[..., N_window, num_query_atoms, num_key_atoms, 3]` canonical
            query-minus-key displacement vectors.
        distance_mask: `[..., N_window, num_query_atoms, num_key_atoms, 1]` mask that is one where
            the query and key atom share a reference space.
        window_info: Padding and mask metadata for the local windows.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import LocalAtomGeometryBuilder
        >>> builder = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)
        >>> geometry = builder(torch.zeros(1, 6, 3), torch.zeros(1, 6, dtype=torch.long))
        >>> geometry.distance_vectors.shape
        torch.Size([1, 3, 2, 4, 3])
    """

    distance_vectors: Tensor
    distance_mask: Tensor
    window_info: LocalAtomWindowInfo


class LocalAtomGeometryBuilder(nn.Module):
    r"""
    Build windowed atom-pair geometry from reference coords and reference-space ids.

    Windows the atoms once, then emits the per-window query-minus-key displacement vectors and a
    same-space mask, feeding the geometric terms of the atom-pair encoder without materialising the
    dense ``N_atom x N_atom`` pair grid.

    Args:
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import LocalAtomGeometryBuilder
        >>> builder = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)
        >>> geometry = builder(torch.zeros(1, 6, 3), torch.zeros(1, 6, dtype=torch.long))
        >>> geometry.distance_vectors.shape, geometry.distance_mask.shape
        (torch.Size([1, 3, 2, 4, 3]), torch.Size([1, 3, 2, 4, 1]))
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128) -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms)

    def forward(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        r"""
        Build local displacement vectors and same-space masks.

        Args:
            reference_positions: ``[..., N_atom, 3]`` reference coords.
            reference_space_id: ``[..., N_atom]`` reference-space ids.

        Returns:
            Local atom geometry for atom-pair encoding.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.atom.window import LocalAtomGeometryBuilder
            >>> builder = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)
            >>> geometry = builder(torch.zeros(1, 6, 3), torch.zeros(1, 6, dtype=torch.long))
            >>> geometry.distance_vectors.shape
            torch.Size([1, 3, 2, 4, 3])
        """

        reference_space_id = reference_space_id.to(device=reference_positions.device)
        position_windows = self.windowing(reference_positions)
        space_windows = self.windowing(reference_space_id.unsqueeze(-1).to(dtype=reference_positions.dtype))
        distance_vectors = position_windows.query[..., None, :] - position_windows.key[..., None, :, :]
        distance_mask = (space_windows.query.unsqueeze(-2).int() == space_windows.key.unsqueeze(-3).int()).to(
            dtype=reference_positions.dtype
        )
        return LocalAtomGeometry(
            distance_vectors=distance_vectors,
            distance_mask=distance_mask,
            window_info=position_windows.info,
        )


class LocalAtomWindowing(nn.Module):
    r"""
    Partition an atom axis into non-overlapping query windows and overlapping key neighbourhoods.

    The layout that makes atom attention local: query atoms are chunked into fixed windows while
    each window's key window reaches ``(num_key_atoms - num_query_atoms) / 2`` atoms to either side,
    so neighbouring windows overlap. The atom axis is padded to fill whole windows, and the returned
    metadata records that padding plus a mask of the real slots.

    Args:
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.

    Raises:
        ValueError: If window sizes are odd or ``num_key_atoms < num_query_atoms``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.window import LocalAtomWindowing
        >>> windowing = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)
        >>> windows = windowing(torch.zeros(1, 6, 1))
        >>> windows.query.shape, windows.info.attention_mask.shape
        (torch.Size([1, 3, 2, 1]), torch.Size([1, 3, 2, 4]))
        >>> LocalAtomWindowing(num_query_atoms=3, num_key_atoms=4)
        Traceback (most recent call last):
            ...
        ValueError: num_key_atoms and num_query_atoms must be even and num_key_atoms >= num_query_atoms
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128) -> None:
        super().__init__()
        if num_key_atoms < num_query_atoms or num_query_atoms % 2 or num_key_atoms % 2:
            raise ValueError("num_key_atoms and num_query_atoms must be even and num_key_atoms >= num_query_atoms")
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms

    def forward(self, tensor: Tensor) -> LocalAtomWindows:
        r"""
        Split an atom-axis tensor into query and key windows.

        Args:
            tensor: ``[..., N_atom, C]`` atom-axis tensor.

        Returns:
            Local query/key windows and padding metadata.

        Examples:
            >>> import torch
            >>> from ofoldx.modules.atom.window import LocalAtomWindowing
            >>> windowing = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)
            >>> windows = windowing(torch.zeros(1, 6, 1))
            >>> windows.query.shape, windows.key.shape
            (torch.Size([1, 3, 2, 1]), torch.Size([1, 3, 4, 1]))
        """

        num_atoms = tensor.shape[-2]
        num_windows = (num_atoms + self.num_query_atoms - 1) // self.num_query_atoms
        query_padding = num_windows * self.num_query_atoms - num_atoms

        query_windows = F.pad(tensor, (0, 0, 0, query_padding))
        query_windows = query_windows.reshape(*tensor.shape[:-2], num_windows, self.num_query_atoms, tensor.shape[-1])

        key_padding_left = (self.num_key_atoms - self.num_query_atoms) // 2
        key_padding_right = int((num_windows - 0.5) * self.num_query_atoms + self.num_key_atoms / 2 - num_atoms + 0.5)
        key_windows = gather_key_windows(tensor, self.num_query_atoms, self.num_key_atoms)
        # Build the window attention mask directly in windowed (num_windows, query, key) layout. Deriving it from
        # a dense (num_atoms, num_atoms) pad mask is quadratic in the atom count (costly memory at large N); this
        # form is linear in num_windows and is exactly equal to the dense-derived mask. A pair is valid iff the
        # query atom is real and the key atom (the query window shifted left by the key padding) lies in
        # [0, num_atoms).
        device = tensor.device
        window_start = torch.arange(num_windows, device=device).unsqueeze(-1) * self.num_query_atoms
        query_index = window_start + torch.arange(self.num_query_atoms, device=device)
        key_index = window_start - key_padding_left + torch.arange(self.num_key_atoms, device=device)
        window_mask = (
            (query_index[:, :, None] < num_atoms) & (key_index[:, None, :] >= 0) & (key_index[:, None, :] < num_atoms)
        ).reshape(*((1,) * len(tensor.shape[:-2])), num_windows, self.num_query_atoms, self.num_key_atoms)
        return LocalAtomWindows(
            query=query_windows,
            key=key_windows,
            info=LocalAtomWindowInfo(
                attention_mask=window_mask,
                query_padding=query_padding,
                key_padding_left=key_padding_left,
                key_padding_right=key_padding_right,
            ),
        )
