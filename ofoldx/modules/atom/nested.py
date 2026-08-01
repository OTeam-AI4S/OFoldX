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

r"""Packed local-window metadata for NestedTensor atom batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from danling.tensors import NestedTensor
from torch import Tensor

_PACKED_LOCAL_ATOM_LAYOUT_SCHEMA = (
    "(Tensor atom_offsets, SymInt num_atoms, int num_query_atoms) -> (Tensor, Tensor, Tensor)"
)


@torch.library.custom_op(
    "ofoldx::_canonicalize_packed_local_atom_layout",
    mutates_args=(),
    device_types="cpu",
    schema=_PACKED_LOCAL_ATOM_LAYOUT_SCHEMA,
)
def canonicalize_packed_local_atom_layout(
    atom_offsets: Tensor,
    num_atoms: int,
    num_query_atoms: int,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Validate packed atom topology and build one tensorized block map."""

    _validate_offsets(atom_offsets, num_atoms)
    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    if bool(torch.any(atom_lengths <= 0)):
        raise ValueError("packed local atom windows require every batch element to contain at least one atom")
    block_lengths = torch.div(
        atom_lengths + num_query_atoms - 1,
        num_query_atoms,
        rounding_mode="floor",
    )
    block_offsets = torch.cat((block_lengths.new_zeros(1), block_lengths.cumsum(0)))
    num_windows = int(block_offsets[-1])
    block_to_sample = torch.repeat_interleave(
        torch.arange(block_lengths.numel(), dtype=torch.long),
        block_lengths,
        output_size=num_windows,
    )
    sample_metadata = torch.cat((atom_offsets, block_offsets)).contiguous()
    return sample_metadata, block_lengths.contiguous(), block_to_sample.contiguous()


@torch.library.register_fake(canonicalize_packed_local_atom_layout)
def _canonicalize_packed_local_atom_layout_fake(
    atom_offsets: Tensor,
    num_atoms: int,
    num_query_atoms: int,
) -> tuple[Tensor, Tensor, Tensor]:
    del num_atoms, num_query_atoms
    batch_size = atom_offsets.shape[0] - 1
    num_windows = torch.library.get_ctx().new_dynamic_size(min=0)
    return (
        torch.empty((2 * (batch_size + 1),), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((batch_size,), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((num_windows,), dtype=torch.int64, device=atom_offsets.device),
    )


def _validate_offsets(offsets: Tensor, num_atoms: int) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError("packed local atom windows require one-dimensional CPU int64 atom offsets")
    if offsets.numel() == 0:
        raise ValueError("packed local atom windows require atom offsets to include the initial zero")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError("packed local atom windows require monotonic atom offsets starting at zero")
    if int(offsets[-1]) != num_atoms:
        raise ValueError(
            f"packed local atom windows atom offsets cover {int(offsets[-1])} values, "
            f"but packed storage contains {num_atoms}"
        )


@dataclass(eq=False)
class PackedLocalAtomLayout:
    r"""
    Device indices and CPU lengths for one packed local-window partition.

    Attributes:
        block_lengths: CPU ``(B,)`` number of query windows of each sample.
        valid_extent: ``(B,)`` exclusive extent of each sample's valid atoms.
        query_atom_index: ``(sum blocks, Q)`` packed atom index of every query slot.
        query_storage_valid: ``(sum blocks, Q)`` whether a query slot holds a stored atom.
        query_valid: ``(sum blocks, Q)`` whether a query slot holds a valid atom.
        key_atom_index: ``(sum blocks, K)`` packed atom index of every key slot.
        key_window_index: ``(sum blocks, K)`` index of every key slot into the flattened query windows.
        key_valid: ``(sum blocks, K)`` whether a key slot holds a valid atom.
        atom_window_index: ``(sum A,)`` flattened query-window slot of every packed atom.
    """

    block_lengths: Tensor
    valid_extent: Tensor
    query_atom_index: Tensor
    query_storage_valid: Tensor
    query_valid: Tensor
    key_atom_index: Tensor
    key_window_index: Tensor
    key_valid: Tensor
    atom_window_index: Tensor


def packed_local_atom_layout(
    atom_mask: NestedTensor,
    num_query_atoms: int,
    num_key_atoms: int,
    key_starts: Callable[[Tensor, Tensor], Tensor],
) -> PackedLocalAtomLayout:
    r"""
    Build a no-padding local-window layout from a canonical packed atom mask.

    Args:
        atom_mask: Valid-atom mask with canonical ``(A_i,)`` elements. Its topology fixes the packed
            windows, and its masked atoms are invalid query and key slots.
        num_query_atoms: Query atoms per window.
        num_key_atoms: Key atoms gathered around each query window.
        key_starts: Maps the centred key start of every window and the exclusive valid-atom extent of
            its sample, both ``(sum blocks,)``, to the key start of the layout.

    Returns:
        The packed window layout.
    """

    _check_nested_atom_mask(atom_mask)
    mask_values = atom_mask.concat.to(torch.bool)
    atom_offsets = atom_mask.ragged_level_offsets(0, dtype=torch.int64)
    sample_metadata, block_lengths, block_to_sample = canonicalize_packed_local_atom_layout(
        atom_offsets,
        mask_values.shape[0],
        num_query_atoms,
    )

    device = mask_values.device
    batch_size = atom_offsets.shape[0] - 1
    metadata = sample_metadata.to(device=device)
    block_to_sample = block_to_sample.to(device=device)
    device_atom_offsets = metadata[: batch_size + 1]
    block_offsets = metadata[batch_size + 1 :]
    atom_lengths = device_atom_offsets[1:] - device_atom_offsets[:-1]

    num_atoms = mask_values.shape[0]
    atom_to_sample = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long),
        atom_lengths,
        output_size=num_atoms,
    )
    atom_index = torch.arange(num_atoms, device=device, dtype=torch.long)
    atom_local = atom_index - device_atom_offsets.index_select(0, atom_to_sample)
    valid_extent = torch.zeros(batch_size, device=device, dtype=torch.long).scatter_reduce(
        0,
        atom_to_sample,
        torch.where(mask_values, atom_local + 1, 0),
        reduce="amax",
        include_self=True,
    )

    num_windows = block_to_sample.shape[0]
    block_index = torch.arange(num_windows, device=device, dtype=torch.long)
    sample_block_start = block_offsets.index_select(0, block_to_sample)
    local_block = block_index - sample_block_start
    sample_atom_start = device_atom_offsets.index_select(0, block_to_sample)
    sample_atom_length = atom_lengths.index_select(0, block_to_sample)

    query_slot = torch.arange(num_query_atoms, device=device, dtype=torch.long)
    query_local = local_block.unsqueeze(-1) * num_query_atoms + query_slot
    query_storage_valid = query_local < sample_atom_length.unsqueeze(-1)
    query_safe = torch.minimum(query_local, (sample_atom_length - 1).clamp_min(0).unsqueeze(-1))
    query_atom_index = sample_atom_start.unsqueeze(-1) + query_safe
    query_valid = query_storage_valid & mask_values.index_select(0, query_atom_index.reshape(-1)).reshape(
        query_atom_index.shape
    )

    centred_key_start = local_block * num_query_atoms - (num_key_atoms - num_query_atoms) // 2
    key_start = key_starts(centred_key_start, valid_extent.index_select(0, block_to_sample))
    key_slot = torch.arange(num_key_atoms, device=device, dtype=torch.long)
    key_local = key_start.unsqueeze(-1) + key_slot
    key_in_range = (key_local >= 0) & (key_local < sample_atom_length.unsqueeze(-1))
    key_safe = torch.minimum(key_local.clamp_min(0), (sample_atom_length - 1).clamp_min(0).unsqueeze(-1))
    key_atom_index = sample_atom_start.unsqueeze(-1) + key_safe
    key_window_index = sample_block_start.unsqueeze(-1) * num_query_atoms + key_safe
    key_valid = key_in_range & mask_values.index_select(0, key_atom_index.reshape(-1)).reshape(key_atom_index.shape)
    atom_window_index = block_offsets.index_select(0, atom_to_sample) * num_query_atoms + atom_local

    return PackedLocalAtomLayout(
        block_lengths=block_lengths,
        valid_extent=valid_extent,
        query_atom_index=query_atom_index,
        query_storage_valid=query_storage_valid,
        query_valid=query_valid,
        key_atom_index=key_atom_index,
        key_window_index=key_window_index,
        key_valid=key_valid,
        atom_window_index=atom_window_index,
    )


def gather_packed_atom_windows(
    values: Tensor,
    layout: PackedLocalAtomLayout,
) -> tuple[Tensor, Tensor]:
    r"""Gather query and key panes and zero only mathematically padded local slots."""

    query = _gather_and_mask(values, layout.query_atom_index, layout.query_storage_valid)
    key = _gather_and_mask(values, layout.key_atom_index, layout.key_valid)
    return query, key


def gather_packed_window_keys(values: Tensor, layout: PackedLocalAtomLayout) -> Tensor:
    r"""Gather keys from a dense ``(sum blocks, Q, ...)`` query-window tensor."""

    flat_values = values.flatten(0, 1)
    return _gather_and_mask(flat_values, layout.key_window_index, layout.key_valid)


def _gather_and_mask(values: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    tail = tuple(values.shape[1:])
    gathered = values.index_select(0, index.reshape(-1)).reshape(*index.shape, *tail)
    mask = valid.reshape(*valid.shape, *((1,) * len(tail)))
    return torch.where(mask, gathered, torch.zeros((), dtype=values.dtype, device=values.device))


def _check_nested_atom_mask(atom_mask: NestedTensor) -> None:
    if not atom_mask.batch_first:
        raise ValueError("packed local atom windows require a leading NestedTensor batch for atom_mask")
    if len(atom_mask) == 0:
        raise ValueError("packed local atom windows require a non-empty NestedTensor wrapper for atom_mask")
    if atom_mask.dim() != 2 or atom_mask.ragged_dims != (0,) or atom_mask.packed_dim_order != (0,):
        raise ValueError("packed local atom windows require canonical (A_i,) atom_mask elements")
