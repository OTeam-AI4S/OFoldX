r"""
Local atom windowing modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from .nested import PackedLocalAtomLayout, gather_packed_atom_windows, packed_local_atom_layout


def key_window_starts(
    num_windows: int,
    num_query_atoms: int,
    num_key_atoms: int,
    valid_extent: Tensor,
) -> Tensor:
    r"""
    Return row-specific start offsets for local key neighborhoods.

    Each key window is centred on its query window (``num_query_atoms`` wide) and reaches
    ``(num_key_atoms - num_query_atoms) / 2`` atoms to either side, then is **shifted inwards** so it
    lies fully inside the row's real-atom extent when that extent is at least ``num_key_atoms``.
    Shorter rows retain the centred layout and use invalid key indices for padding.

    Args:
        num_windows: Number of query windows.
        num_query_atoms: Query atoms per window.
        num_key_atoms: Key atoms gathered per window.
        valid_extent: ``[...]`` exclusive upper bound of the last valid atom in each row.

    Returns:
        ``[..., num_windows]`` integer start offsets.
    """

    key_padding_left = (num_key_atoms - num_query_atoms) // 2
    base = torch.arange(num_windows, device=valid_extent.device) * num_query_atoms - key_padding_left
    base = base.reshape(*((1,) * valid_extent.dim()), num_windows).expand(*valid_extent.shape, num_windows)
    shifted = torch.minimum(base.clamp_min(0), (valid_extent - num_key_atoms).clamp_min(0).unsqueeze(-1))
    return torch.where((valid_extent >= num_key_atoms).unsqueeze(-1), shifted, base)


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
    Padding and masking metadata describing one local-window partition of an atom axis.

    Carries an optional row-specific key layout and the corresponding pair mask so
    every downstream gather uses the same windows.

    Attributes:
        attention_mask: `[..., N_window, num_query_atoms, num_key_atoms]` boolean mask; ``True``
            where both the query and key atom of a window slot are real (not padding).
        key_index: Optional `[..., N_window, num_key_atoms]` row-specific atom
            indices; ``-1`` marks an invalid key and ``None`` selects the
            reconstructible centred layout.
        valid_extent: Optional exclusive valid-atom extent for each leading row.
            This compact metadata reconstructs ``key_index`` without materializing
            it in packed OOps calls.
    """

    attention_mask: Tensor
    key_index: Tensor | None
    valid_extent: Tensor | None = None
    packed_layout: PackedLocalAtomLayout | None = None

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
class LocalAtomWindows:
    r"""
    Query and key windows carved from a single atom-axis tensor, plus their padding metadata.

    Attributes:
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

    Attributes:
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
        self,
        reference_positions: Tensor | NestedTensor,
        reference_space_id: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
    ) -> LocalAtomGeometry:
        r"""
        Build local displacement vectors and same-space masks.

        Args:
            reference_positions: ``[..., N_atom, 3]`` reference coords.
            reference_space_id: ``[..., N_atom]`` reference-space ids.
            atom_mask: Optional ``[..., N_atom]`` valid-atom mask selecting
                row-specific clamped key windows.

        Returns:
            Local atom geometry for atom-pair encoding.
        """

        position_windows = self.windowing(reference_positions, atom_mask)
        if isinstance(reference_positions, NestedTensor):
            if not isinstance(reference_space_id, NestedTensor):
                raise ValueError("packed local atom geometry requires a NestedTensor reference_space_id")
            packed_layout = position_windows.info.packed_layout
            if packed_layout is None:
                raise RuntimeError("packed local atom geometry is missing its packed window layout")
            space_queries, space_keys = gather_packed_atom_windows(
                reference_space_id.concat.unsqueeze(-1),
                packed_layout,
            )
            distance_vectors = position_windows.query[..., None, :] - position_windows.key[..., None, :, :]
            distance_mask = (space_queries.unsqueeze(-2) == space_keys.unsqueeze(-3)).to(
                reference_positions.dtype
            ) * position_windows.info.attention_mask.unsqueeze(-1)
            return LocalAtomGeometry(
                distance_vectors=distance_vectors,
                distance_mask=distance_mask,
                window_info=position_windows.info,
            )

        if isinstance(reference_space_id, NestedTensor) or isinstance(atom_mask, NestedTensor):
            raise ValueError("dense local atom geometry requires dense reference_space_id and atom_mask tensors")
        target_prefix = tuple(reference_positions.shape[:-2])
        source_prefix = tuple(reference_space_id.shape[:-1])
        missing = len(target_prefix) - len(source_prefix)
        if missing:
            reference_space_id = reference_space_id.reshape(
                *source_prefix,
                *((1,) * missing),
                reference_space_id.shape[-1],
            )
        reference_space_id = reference_space_id.expand(*target_prefix, reference_space_id.shape[-1])
        num_windows = position_windows.query.shape[-3]
        query_padding = num_windows * self.windowing.num_query_atoms - reference_space_id.shape[-1]
        space_queries = F.pad(reference_space_id.unsqueeze(-1), (0, 0, 0, query_padding))
        space_queries = space_queries.reshape(
            *reference_space_id.shape[:-1],
            num_windows,
            self.windowing.num_query_atoms,
            1,
        )
        space_keys = gather_key_windows(
            reference_space_id.unsqueeze(-1),
            self.windowing.num_query_atoms,
            self.windowing.num_key_atoms,
            position_windows.info.key_index,
        )
        distance_vectors = position_windows.query[..., None, :] - position_windows.key[..., None, :, :]
        distance_mask = (space_queries.unsqueeze(-2) == space_keys.unsqueeze(-3)).to(
            reference_positions.dtype
        ) * position_windows.info.attention_mask.unsqueeze(-1)
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

    def forward(
        self,
        tensor: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
    ) -> LocalAtomWindows:
        r"""
        Split an atom-axis tensor into query and key windows.

        Args:
            tensor: ``[..., N_atom, C]`` atom-axis tensor.
            atom_mask: Optional ``[..., N_atom]`` valid-atom mask. Each row uses
                its own clamped key layout; omitted selects the fixed centred layout.

        Returns:
            Local query/key windows and padding metadata.
        """

        if isinstance(tensor, NestedTensor):
            if atom_mask is not None and not isinstance(atom_mask, NestedTensor):
                raise ValueError("packed local atom windows require a NestedTensor atom_mask")
            layout = packed_local_atom_layout(
                tensor,
                atom_mask,
                self.num_query_atoms,
                self.num_key_atoms,
            )
            query_windows, key_windows = gather_packed_atom_windows(tensor.concat, layout)
            if atom_mask is None:
                query_valid = layout.query_valid
                key_valid = layout.key_valid
            else:
                query_valid = layout.query_valid
                key_valid = layout.key_valid
            window_mask = query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2)
            return LocalAtomWindows(
                query=query_windows,
                key=key_windows,
                info=LocalAtomWindowInfo(
                    attention_mask=window_mask,
                    key_index=None,
                    valid_extent=layout.valid_extent,
                    packed_layout=layout,
                ),
            )
        if isinstance(atom_mask, NestedTensor):
            raise ValueError("dense local atom windows require a dense atom_mask")

        num_atoms = tensor.shape[-2]
        num_windows = (num_atoms + self.num_query_atoms - 1) // self.num_query_atoms
        query_padding = num_windows * self.num_query_atoms - num_atoms

        query_windows = F.pad(tensor, (0, 0, 0, query_padding))
        query_windows = query_windows.reshape(*tensor.shape[:-2], num_windows, self.num_query_atoms, tensor.shape[-1])

        device = tensor.device
        window_start = torch.arange(num_windows, device=device) * self.num_query_atoms
        query_index = window_start[:, None] + torch.arange(self.num_query_atoms, device=device)
        key_padding_left = (self.num_key_atoms - self.num_query_atoms) // 2
        if atom_mask is None:
            valid_extent = None
            starts = window_start - key_padding_left
            centered_index = starts[:, None] + torch.arange(self.num_key_atoms, device=device)[None, :]
            key_valid = (centered_index >= 0) & (centered_index < num_atoms)
            query_valid = query_index < num_atoms
            singleton_prefix = (1,) * len(tensor.shape[:-2])
            key_valid = key_valid.reshape(*singleton_prefix, num_windows, self.num_key_atoms)
            query_valid = query_valid.reshape(*singleton_prefix, num_windows, self.num_query_atoms)
            key_index = None
        else:
            atom_mask = atom_mask.to(torch.bool)
            target_prefix = tuple(tensor.shape[:-2])
            source_prefix = tuple(atom_mask.shape[:-1])
            missing = len(target_prefix) - len(source_prefix)
            if missing:
                atom_mask = atom_mask.reshape(*source_prefix, *((1,) * missing), atom_mask.shape[-1])
            atom_mask = atom_mask.expand(*target_prefix, num_atoms)
            atom_index = torch.arange(num_atoms, device=device)
            valid_extent = torch.where(atom_mask, atom_index + 1, 0).amax(dim=-1)
            starts = key_window_starts(
                num_windows,
                self.num_query_atoms,
                self.num_key_atoms,
                valid_extent,
            )
            key_index = starts.unsqueeze(-1) + torch.arange(self.num_key_atoms, device=device)
            in_bounds = (key_index >= 0) & (key_index < num_atoms)
            safe_index = key_index.clamp(0, num_atoms - 1)
            key_valid = in_bounds & torch.take_along_dim(atom_mask.unsqueeze(-2), safe_index, dim=-1)
            key_index = torch.where(key_valid, safe_index, -1)
            query_valid = F.pad(atom_mask, (0, query_padding)).reshape(
                *atom_mask.shape[:-1], num_windows, self.num_query_atoms
            )

        key_windows = gather_key_windows(
            tensor,
            self.num_query_atoms,
            self.num_key_atoms,
            key_index,
        )
        window_mask = query_valid.unsqueeze(-1) & key_valid.unsqueeze(-2)
        return LocalAtomWindows(
            query=query_windows,
            key=key_windows,
            info=LocalAtomWindowInfo(
                attention_mask=window_mask,
                key_index=key_index,
                valid_extent=valid_extent,
            ),
        )

    def gather(
        self,
        tensor: Tensor | NestedTensor,
        info: LocalAtomWindowInfo,
    ) -> LocalAtomWindows:
        r"""Gather another atom tensor with an existing local-window layout."""

        if isinstance(tensor, NestedTensor):
            if info.packed_layout is None:
                raise ValueError("packed local atom gather requires packed window metadata")
            query, key = gather_packed_atom_windows(tensor.concat, info.packed_layout)
            return LocalAtomWindows(query=query, key=key, info=info)
        if info.packed_layout is not None:
            raise ValueError("dense local atom gather requires dense window metadata")

        num_atoms = tensor.shape[-2]
        num_windows = info.attention_mask.shape[-3]
        query_padding = num_windows * self.num_query_atoms - num_atoms
        query = F.pad(tensor, (0, 0, 0, query_padding)).reshape(
            *tensor.shape[:-2],
            num_windows,
            self.num_query_atoms,
            tensor.shape[-1],
        )
        key = gather_key_windows(
            tensor,
            self.num_query_atoms,
            self.num_key_atoms,
            info.key_index,
        )
        return LocalAtomWindows(query=query, key=key, info=info)
