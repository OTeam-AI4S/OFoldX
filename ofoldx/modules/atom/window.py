r"""
Local atom windowing modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def key_window_starts(
    num_windows: int, num_query_atoms: int, num_key_atoms: int, num_valid_atoms: int, device: torch.device
) -> Tensor:
    r"""
    Return per-window start offsets for clamped key neighborhoods.

    Each key window is centred on its query window (``num_query_atoms`` wide) and reaches
    ``(num_key_atoms - num_query_atoms) / 2`` atoms to either side, then is **shifted inwards** so it
    lies fully inside the ``[0, num_valid_atoms)`` real-atom range. Boundary windows therefore keep a full
    ``num_key_atoms`` of real keys instead of padding. Requires ``num_valid_atoms >= num_key_atoms``.

    Args:
        num_windows: Number of query windows.
        num_query_atoms: Query atoms per window.
        num_key_atoms: Key atoms gathered per window.
        num_valid_atoms: Number of real (non-padding) atoms, packed contiguously at the front.
        device: Device for the returned index tensor.

    Returns:
        ``[num_windows]`` integer start offsets, each in ``[0, num_valid_atoms - num_key_atoms]``.
    """

    key_padding_left = (num_key_atoms - num_query_atoms) // 2
    base = torch.arange(num_windows, device=device) * num_query_atoms - key_padding_left
    return base.clamp(0, num_valid_atoms - num_key_atoms)


def gather_key_windows(
    single: Tensor, num_query_atoms: int, num_key_atoms: int, num_valid_atoms: int | None = None
) -> Tensor:
    r"""
    Gather each query window's overlapping key-atom neighbourhood from a flat ``(B, N, D)`` atom tensor.

    With ``num_valid_atoms`` unset (or below ``num_key_atoms``) the fixed zero-padded window is used:
    a strided ``unfold`` view (O(N), no matmul) equivalent to a one-hot indexing-matrix einsum, with slots
    past the atom range read as zero. With ``num_valid_atoms`` given (and ``>= num_key_atoms``), a
    **clamped/shifted** window is used instead: boundary windows are slid inwards so every key is a real
    atom (see [`key_window_starts`][ofoldx.modules.atom.window.key_window_starts]).

    Args:
        single: ``[..., N_atom, C]`` atom features.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        num_valid_atoms: Number of real atoms (contiguous at the front); enables the clamped layout.

    Returns:
        ``[..., N_window, num_key_atoms, C]`` key windows.
    """

    num_atoms = single.shape[-2]
    num_windows = (num_atoms + num_query_atoms - 1) // num_query_atoms
    if num_valid_atoms is not None and num_valid_atoms >= num_key_atoms:
        starts = key_window_starts(num_windows, num_query_atoms, num_key_atoms, num_valid_atoms, single.device)
        key_index = starts[:, None] + torch.arange(num_key_atoms, device=single.device)[None, :]
        return single[..., key_index, :]
    key_padding_left = (num_key_atoms - num_query_atoms) // 2
    key_padding_right = int((num_windows - 0.5) * num_query_atoms + num_key_atoms / 2 - num_atoms + 0.5)
    padded = F.pad(single, (0, 0, key_padding_left, key_padding_right))
    return padded.unfold(-2, num_key_atoms, num_query_atoms).movedim(-1, -2)


@dataclass(eq=False)
class LocalAtomWindowInfo:
    r"""
    Padding and masking metadata describing one local-window partition of an atom axis.

    Records how the atom axis was padded to fill whole windows and which query/key positions are
    real, so downstream attention can mask out the padded slots without re-deriving them.

    Args:
        attention_mask: `[..., N_window, num_query_atoms, num_key_atoms]` boolean mask; ``True``
            where both the query and key atom of a window slot are real (not padding).
        query_padding: Number of padded query atoms appended to reach a whole number of windows.
        key_padding_left: Number of padded key atoms before the first real atom.
        key_padding_right: Number of padded key atoms after the last real atom.
        num_valid_atoms: Number of contiguous valid atoms at the start of the atom
            axis, or ``None`` when all atom rows participate in window placement.
    """

    attention_mask: Tensor
    query_padding: int
    key_padding_left: int
    key_padding_right: int
    num_valid_atoms: int | None = None


@dataclass(eq=False)
class LocalAtomWindows:
    r"""
    Query and key windows carved from a single atom-axis tensor, plus their padding metadata.

    Args:
        query: `[..., N_window, num_query_atoms, C]` non-overlapping query windows.
        key: `[..., N_window, num_key_atoms, C]` overlapping key neighbourhoods, one per query window.
        info: Padding and mask metadata for the windows.
    """

    query: Tensor
    key: Tensor
    info: LocalAtomWindowInfo


@dataclass(eq=False)
class LocalAtomGeometry:
    r"""
    Windowed atom-pair geometry (displacements and same-space mask) for atom attention blocks.

    Args:
        distance_vectors: `[..., N_window, num_query_atoms, num_key_atoms, 3]`
            query-minus-key displacement vectors.
        distance_mask: `[..., N_window, num_query_atoms, num_key_atoms, 1]` mask that is one where
            the query and key atom share a reference space.
        window_info: Padding and mask metadata for the local windows.
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
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128) -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms)

    def forward(
        self, reference_positions: Tensor, reference_space_id: Tensor, num_valid_atoms: int | None = None
    ) -> LocalAtomGeometry:
        r"""
        Build local displacement vectors and same-space masks.

        Args:
            reference_positions: ``[..., N_atom, 3]`` reference coords.
            reference_space_id: ``[..., N_atom]`` reference-space ids.
            num_valid_atoms: Number of real atoms; enables the clamped key window (see
                [`gather_key_windows`][ofoldx.modules.atom.window.gather_key_windows]).

        Returns:
            Local atom geometry for atom-pair encoding.
        """

        position_windows = self.windowing(reference_positions, num_valid_atoms)
        space_windows = self.windowing(reference_space_id.unsqueeze(-1), num_valid_atoms)
        distance_vectors = position_windows.query[..., None, :] - position_windows.key[..., None, :, :]
        distance_mask = (space_windows.query.unsqueeze(-2) == space_windows.key.unsqueeze(-3)).to(
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
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128) -> None:
        super().__init__()
        if num_key_atoms < num_query_atoms or num_query_atoms % 2 or num_key_atoms % 2:
            raise ValueError("num_key_atoms and num_query_atoms must be even and num_key_atoms >= num_query_atoms")
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms

    def forward(self, tensor: Tensor, num_valid_atoms: int | None = None) -> LocalAtomWindows:
        r"""
        Split an atom-axis tensor into query and key windows.

        Args:
            tensor: ``[..., N_atom, C]`` atom-axis tensor.
            num_valid_atoms: Number of real atoms (contiguous at the front). When given (and
                ``>= num_key_atoms``), a clamped/shifted key window is used, so boundary
                windows carry a full ``num_key_atoms`` of real keys; otherwise the fixed zero-padded window
                is used. See [`gather_key_windows`][ofoldx.modules.atom.window.gather_key_windows].

        Returns:
            Local query/key windows and padding metadata.
        """

        num_atoms = tensor.shape[-2]
        num_windows = (num_atoms + self.num_query_atoms - 1) // self.num_query_atoms
        query_padding = num_windows * self.num_query_atoms - num_atoms

        query_windows = F.pad(tensor, (0, 0, 0, query_padding))
        query_windows = query_windows.reshape(*tensor.shape[:-2], num_windows, self.num_query_atoms, tensor.shape[-1])

        key_padding_left = (self.num_key_atoms - self.num_query_atoms) // 2
        key_padding_right = int((num_windows - 0.5) * self.num_query_atoms + self.num_key_atoms / 2 - num_atoms + 0.5)
        key_windows = gather_key_windows(tensor, self.num_query_atoms, self.num_key_atoms, num_valid_atoms)
        # Build the window attention mask directly in windowed (num_windows, query, key) layout. Deriving it from
        # a dense (num_atoms, num_atoms) pad mask is quadratic in the atom count (costly memory at large N); this
        # form is linear in num_windows and is exactly equal to the dense-derived mask.
        device = tensor.device
        window_start = torch.arange(num_windows, device=device).unsqueeze(-1) * self.num_query_atoms
        query_index = window_start + torch.arange(self.num_query_atoms, device=device)
        clamped = num_valid_atoms is not None and num_valid_atoms >= self.num_key_atoms
        if clamped:
            # Clamped/shifted keys are all real, so a pair is valid iff its query atom is real.
            query_valid = query_index < num_valid_atoms
            window_mask = query_valid[:, :, None].expand(-1, -1, self.num_key_atoms)
        else:
            # A pair is valid iff the query atom is real and the key atom (the query window shifted left by the
            # key padding) lies in [0, num_atoms).
            key_index = window_start - key_padding_left + torch.arange(self.num_key_atoms, device=device)
            window_mask = (
                (query_index[:, :, None] < num_atoms)
                & (key_index[:, None, :] >= 0)
                & (key_index[:, None, :] < num_atoms)
            )
        window_mask = window_mask.reshape(
            *((1,) * len(tensor.shape[:-2])), num_windows, self.num_query_atoms, self.num_key_atoms
        )
        return LocalAtomWindows(
            query=query_windows,
            key=key_windows,
            info=LocalAtomWindowInfo(
                attention_mask=window_mask,
                query_padding=query_padding,
                key_padding_left=key_padding_left,
                key_padding_right=key_padding_right,
                num_valid_atoms=num_valid_atoms,
            ),
        )
