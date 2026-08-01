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

r"""
Local atom windowing modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from .coordinates import project_coordinates
from .nested import PackedLocalAtomLayout, gather_packed_atom_windows, packed_local_atom_layout


def gather_key_windows(
    single: Tensor,
    num_query_atoms: int,
    num_key_atoms: int,
    key_index: Tensor | None = None,
) -> Tensor:
    r"""
    Gather each query window's overlapping key-atom neighbourhood from a flat ``(B, N, D)`` atom tensor.

    ``key_index`` carries the exact layout as atom indices with ``-1`` marking invalid
    slots. When omitted, the ordinary centred zero-padded layout is constructed.

    Args:
        single: ``[..., N_atom, C]`` atom features.
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        key_index: Optional ``[..., N_window, num_key_atoms]`` atom indices.

    Returns:
        ``[..., N_window, num_key_atoms, C]`` key windows.
    """

    num_atoms = single.shape[-2]
    num_windows = (num_atoms + num_query_atoms - 1) // num_query_atoms
    if key_index is None:
        key_padding_left = (num_key_atoms - num_query_atoms) // 2
        key_padding_right = int((num_windows - 0.5) * num_query_atoms + num_key_atoms / 2 - num_atoms + 0.5)
        padded = F.pad(single, (0, 0, key_padding_left, key_padding_right))
        return padded.unfold(-2, num_key_atoms, num_query_atoms).movedim(-1, -2)

    source_prefix = tuple(key_index.shape[:-2])
    target_prefix = tuple(single.shape[:-2])
    missing = len(target_prefix) - len(source_prefix)
    if missing:
        key_index = key_index.reshape(*source_prefix, *((1,) * missing), *key_index.shape[-2:])
    key_index = key_index.expand(*target_prefix, *key_index.shape[-2:])

    gather_index = key_index.clamp_min(0).unsqueeze(-1).expand(*key_index.shape, single.shape[-1])
    source = single.unsqueeze(-3).expand(*target_prefix, num_windows, num_atoms, single.shape[-1])
    gathered = source.gather(-2, gather_index)
    return gathered * (key_index >= 0).unsqueeze(-1).to(dtype=single.dtype)


@dataclass(eq=False)
class LocalAtomWindowInfo:
    r"""
    One local-window partition of an atom axis, shared by every gather over that axis.

    Attributes:
        attention_mask: `[..., N_window, num_query_atoms, num_key_atoms]` boolean mask; ``True``
            where the query and key slot of a window both hold a valid atom.
        key_index: Optional `[..., N_window, num_key_atoms]` row-specific atom indices of a dense
            layout; ``-1`` marks an invalid key and ``None`` selects the centred layout.
        valid_extent: Optional ``(B,)`` exclusive valid-atom extent of every sample of a packed clamped
            layout; ``None`` for the centred layout.
        packed_layout: Index metadata of a packed NestedTensor partition.
    """

    attention_mask: Tensor
    key_index: Tensor | None = None
    valid_extent: Tensor | None = None
    packed_layout: PackedLocalAtomLayout | None = None

    @property
    def num_query_atoms(self) -> int:
        r"""Query atoms per window."""

        return self.attention_mask.shape[-2]

    @property
    def num_key_atoms(self) -> int:
        r"""Key atoms gathered around each query window."""

        return self.attention_mask.shape[-1]

    def gather(self, tensor: Tensor | NestedTensor) -> tuple[Tensor, Tensor]:
        r"""
        Gather the query and key windows of an atom-axis tensor.

        Args:
            tensor: ``[..., N_atom, C]`` atom tensor, or its packed NestedTensor.

        Returns:
            query: ``[..., N_window, num_query_atoms, C]`` query windows.
            key: ``[..., N_window, num_key_atoms, C]`` key windows.
        """

        if self.packed_layout is not None:
            return gather_packed_atom_windows(self.values(tensor), self.packed_layout)
        if isinstance(tensor, NestedTensor):
            raise ValueError("dense local atom windows require dense values")
        num_atoms = tensor.shape[-2]
        num_windows = self.attention_mask.shape[-3]
        query = F.pad(tensor, (0, 0, 0, num_windows * self.num_query_atoms - num_atoms)).reshape(
            *tensor.shape[:-2],
            num_windows,
            self.num_query_atoms,
            tensor.shape[-1],
        )
        key = gather_key_windows(tensor, self.num_query_atoms, self.num_key_atoms, self.key_index)
        return query, key

    def values(self, tensor: Tensor | NestedTensor) -> Tensor:
        r"""Return local-window values from their dense or packed carrier."""

        if self.packed_layout is None:
            if isinstance(tensor, NestedTensor):
                raise ValueError("dense local atom windows require dense values")
            return tensor
        if not isinstance(tensor, NestedTensor):
            raise ValueError("packed local atom windows require packed values")
        return tensor.concat

    def pack_like(self, reference: Tensor | NestedTensor, values: Tensor) -> Tensor | NestedTensor:
        r"""Wrap local-window values with the packed batch topology of ``reference`` when needed."""

        if self.packed_layout is None:
            if isinstance(reference, NestedTensor):
                raise ValueError("dense local atom windows require a dense reference")
            return values
        if not isinstance(reference, NestedTensor):
            raise ValueError("packed local atom windows require a packed reference")
        return reference.packed_with_lengths(values, self.packed_layout.block_lengths)


@dataclass(eq=False)
class LocalAtomGeometry:
    r"""
    Windowed reference geometry of the local atom pairs.

    Attributes:
        distance_vectors: `[..., N_window, num_query_atoms, num_key_atoms, 3]`
            query-minus-key reference displacements.
        distance_mask: `[..., N_window, num_query_atoms, num_key_atoms, 1]` mask that is one where
            both atoms are valid, have available reference positions, and share a reference space.
        window_info: The local window partition of the atom axis.
    """

    distance_vectors: Tensor
    distance_mask: Tensor
    window_info: LocalAtomWindowInfo


class LocalAtomWindowing(nn.Module):
    r"""
    Partition an atom axis into non-overlapping query windows and centred overlapping key neighbourhoods.

    The layout that makes atom attention local: query atoms are chunked into fixed windows while
    each window's key neighbourhood reaches ``(num_key_atoms - num_query_atoms) / 2`` atoms to either
    side of it, so neighbouring windows overlap. Slots outside the atom axis are padding. Calling the
    module windows the reference geometry once; the returned window partition then serves every
    other gather over the same atoms.

    Args:
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.

    Raises:
        ValueError: If window sizes are odd or ``num_key_atoms < num_query_atoms``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import LocalAtomWindowing
        >>> geometry = LocalAtomWindowing(4, 8)(
        ...     torch.randn(1, 10, 3), torch.zeros(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.bool)
        ... )
        >>> geometry.distance_vectors.shape, geometry.window_info.key_index is None
        (torch.Size([1, 3, 4, 8, 3]), True)
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128) -> None:
        super().__init__()
        if num_key_atoms < num_query_atoms or num_query_atoms % 2 or num_key_atoms % 2:
            raise ValueError("num_key_atoms and num_query_atoms must be even and num_key_atoms >= num_query_atoms")
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms

    def forward(
        self,
        reference_positions: Tensor | NestedTensor,
        reference_space_id: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor,
        reference_mask: Tensor | NestedTensor | None = None,
    ) -> LocalAtomGeometry:
        r"""
        Window the reference geometry of the local atom pairs.

        Args:
            reference_positions: ``[..., N_atom, 3]`` reference coordinates.
            reference_space_id: ``[..., N_atom]`` reference-space ids; atoms pair geometrically only
                within one space. A shorter dense prefix broadcasts over the coordinates' trailing
                prefix axes.
            atom_mask: ``[..., N_atom]`` valid-atom mask, broadcast like ``reference_space_id``.
            reference_mask: Optional reference-position availability, broadcast like ``atom_mask``.
                Atom existence still defines the windows and attention; omitted means all references available.

        Returns:
            The windowed geometry and its window partition.
        """

        if not isinstance(reference_positions, NestedTensor):
            prefix = tuple(reference_positions.shape[:-2])
            reference_space_id = _expand_atom_prefix(reference_space_id, prefix)
            atom_mask = _expand_atom_prefix(atom_mask, prefix)
            if reference_mask is not None:
                reference_mask = _expand_atom_prefix(reference_mask, prefix)
        info = self.window_info(atom_mask)
        available = atom_mask.bool()
        if reference_mask is not None:
            available = available & reference_mask.bool()
        reference_positions = project_coordinates(reference_positions, available)
        position_query, position_key = info.gather(reference_positions)
        space_query, space_key = info.gather(reference_space_id.unsqueeze(-1))
        distance_vectors = position_query.unsqueeze(-2) - position_key.unsqueeze(-3)
        same_space = space_query.unsqueeze(-2) == space_key.unsqueeze(-3)
        distance_mask = same_space & info.attention_mask.unsqueeze(-1)
        if reference_mask is not None:
            available_query, available_key = info.gather(available.unsqueeze(-1))
            distance_mask = distance_mask & available_query.unsqueeze(-2) & available_key.unsqueeze(-3)
        distance_vectors = distance_vectors.masked_fill(~distance_mask, 0)
        return LocalAtomGeometry(distance_vectors, distance_mask.to(distance_vectors.dtype), info)

    def window_info(self, atom_mask: Tensor | NestedTensor) -> LocalAtomWindowInfo:
        r"""
        Partition the atom axis of ``atom_mask`` into local windows.

        The dense centred layout is the implicit zero-padded unfold, so it carries no key indices.

        Args:
            atom_mask: ``[..., N_atom]`` valid-atom mask, or its packed NestedTensor.

        Returns:
            The window partition; masked and out-of-range slots are invalid.
        """

        if isinstance(atom_mask, NestedTensor):
            layout = packed_local_atom_layout(atom_mask, self.num_query_atoms, self.num_key_atoms, self.key_starts)
            return LocalAtomWindowInfo(
                layout.query_valid.unsqueeze(-1) & layout.key_valid.unsqueeze(-2),
                packed_layout=layout,
            )
        atom_mask = atom_mask.bool()
        num_atoms = atom_mask.shape[-1]
        num_windows = (num_atoms + self.num_query_atoms - 1) // self.num_query_atoms
        query_valid = F.pad(atom_mask, (0, num_windows * self.num_query_atoms - num_atoms)).reshape(
            *atom_mask.shape[:-1],
            num_windows,
            self.num_query_atoms,
        )
        key_valid = gather_key_windows(
            atom_mask.unsqueeze(-1).to(torch.uint8), self.num_query_atoms, self.num_key_atoms
        )
        return LocalAtomWindowInfo(query_valid.unsqueeze(-1) & key_valid.squeeze(-1).bool().unsqueeze(-2))

    def key_starts(self, centred: Tensor, valid_extent: Tensor) -> Tensor:
        r"""
        Return the start of each key neighbourhood: the centred layout keeps the centred starts.

        Args:
            centred: Centred key starts of the query windows.
            valid_extent: Exclusive valid-atom extent of each window's row, broadcastable to ``centred``.

        Returns:
            Key starts with the shape of ``centred``.
        """

        return centred


class ClampedLocalAtomWindowing(LocalAtomWindowing):
    r"""
    Local atom windows whose key neighbourhoods are shifted inward to hold only valid atoms.

    Each centred key start is clamped into ``[0, max(V - num_key_atoms, 0)]`` for the row's exclusive
    valid-atom extent ``V``, the AlphaFold 3 layout: a row with at least ``num_key_atoms`` valid atoms
    keeps every key neighbourhood inside them, and a shorter row starts every neighbourhood at zero,
    so each one holds all of the row's valid atoms. Dense rows carry explicit key indices; packed
    rows carry their valid extents describing the same clamped layout.

    Args:
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom import ClampedLocalAtomWindowing
        >>> info = ClampedLocalAtomWindowing(4, 8).window_info(torch.arange(12) < 10)
        >>> info.key_index
        tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [2, 3, 4, 5, 6, 7, 8, 9]])
    """

    def window_info(self, atom_mask: Tensor | NestedTensor) -> LocalAtomWindowInfo:
        if isinstance(atom_mask, NestedTensor):
            info = super().window_info(atom_mask)
            layout = info.packed_layout
            assert layout is not None
            return LocalAtomWindowInfo(info.attention_mask, valid_extent=layout.valid_extent, packed_layout=layout)
        atom_mask = atom_mask.bool()
        num_atoms = atom_mask.shape[-1]
        num_windows = (num_atoms + self.num_query_atoms - 1) // self.num_query_atoms
        device = atom_mask.device
        atom_index = torch.arange(num_atoms, device=device)
        valid_extent = torch.where(atom_mask, atom_index + 1, 0).amax(dim=-1)
        centred = torch.arange(num_windows, device=device) * self.num_query_atoms
        starts = self.key_starts(centred - (self.num_key_atoms - self.num_query_atoms) // 2, valid_extent.unsqueeze(-1))
        key_index = starts.unsqueeze(-1) + torch.arange(self.num_key_atoms, device=device)
        in_range = (key_index >= 0) & (key_index < num_atoms)
        safe_index = key_index.clamp(0, num_atoms - 1)
        key_valid = in_range & torch.take_along_dim(atom_mask.unsqueeze(-2), safe_index, dim=-1)
        query_valid = F.pad(atom_mask, (0, num_windows * self.num_query_atoms - num_atoms)).reshape(
            *atom_mask.shape[:-1],
            num_windows,
            self.num_query_atoms,
        )
        return LocalAtomWindowInfo(
            query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2),
            key_index=torch.where(key_valid, safe_index, -1),
        )

    def key_starts(self, centred: Tensor, valid_extent: Tensor) -> Tensor:
        r"""
        Clamp each centred key start into ``[0, max(valid_extent - num_key_atoms, 0)]``.

        Args:
            centred: Centred key starts of the query windows.
            valid_extent: Exclusive valid-atom extent of each window's row, broadcastable to ``centred``.

        Returns:
            Key starts with the broadcast shape of ``centred`` and ``valid_extent``.
        """

        return torch.minimum(centred.clamp_min(0), (valid_extent - self.num_key_atoms).clamp_min(0))


def _expand_atom_prefix(tensor: Tensor | NestedTensor, prefix: tuple[int, ...]) -> Tensor:
    r"""Broadcast a dense ``[..., N_atom]`` tensor whose prefix leads ``prefix`` to ``(*prefix, N_atom)``."""

    if isinstance(tensor, NestedTensor):
        raise ValueError("dense local atom windows require dense reference_space_id and atom_mask tensors")
    missing = len(prefix) - (tensor.dim() - 1)
    if missing > 0:
        tensor = tensor.reshape(*tensor.shape[:-1], *((1,) * missing), tensor.shape[-1])
    return tensor.expand(*prefix, tensor.shape[-1])
