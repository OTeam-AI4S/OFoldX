r"""Packed local-window metadata for NestedTensor atom batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from danling.tensors import NestedTensor
from torch import Tensor

_PACKED_LOCAL_ATOM_LAYOUT_SCHEMA = (
    "(Tensor atom_offsets, Tensor? mask_offsets, SymInt atom_total, SymInt mask_total, "
    "int num_query_atoms, int num_key_atoms) -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.custom_op(
    "ofoldx::_canonicalize_packed_local_atom_layout",
    mutates_args=(),
    device_types="cpu",
    schema=_PACKED_LOCAL_ATOM_LAYOUT_SCHEMA,
)
def canonicalize_packed_local_atom_layout(
    atom_offsets: Tensor,
    mask_offsets: Tensor | None,
    atom_total: int,
    mask_total: int,
    num_query_atoms: int,
    num_key_atoms: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Validate packed atom topology and build one tensorized block map."""

    _validate_offsets(atom_offsets, atom_total, operand="atom")
    if mask_offsets is not None:
        _validate_offsets(mask_offsets, mask_total, operand="atom_mask")
        if not torch.equal(atom_offsets, mask_offsets):
            raise ValueError("packed local atom windows require matching atom and atom_mask layouts")

    atom_lengths = atom_offsets[1:] - atom_offsets[:-1]
    if bool(torch.any(atom_lengths <= 0)):
        raise ValueError("packed local atom windows require every batch element to contain at least one atom")
    block_lengths = torch.div(
        atom_lengths + num_query_atoms - 1,
        num_query_atoms,
        rounding_mode="floor",
    )
    block_offsets = torch.cat((block_lengths.new_zeros(1), block_lengths.cumsum(0)))
    total_blocks = int(block_offsets[-1])
    block_to_sample = torch.repeat_interleave(
        torch.arange(block_lengths.numel(), dtype=torch.long),
        block_lengths,
        output_size=total_blocks,
    )
    sample_metadata = torch.cat((atom_offsets, block_offsets)).contiguous()
    validation = torch.ones((), dtype=torch.bool, device=atom_offsets.device)
    return sample_metadata, block_lengths.contiguous(), block_to_sample.contiguous(), validation


@torch.library.register_fake(canonicalize_packed_local_atom_layout)
def _canonicalize_packed_local_atom_layout_fake(
    atom_offsets: Tensor,
    mask_offsets: Tensor | None,
    atom_total: int,
    mask_total: int,
    num_query_atoms: int,
    num_key_atoms: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del mask_offsets, atom_total, mask_total, num_query_atoms, num_key_atoms
    samples = atom_offsets.shape[0] - 1
    total_blocks = torch.library.get_ctx().new_dynamic_size(min=0)
    return (
        torch.empty((2 * (samples + 1),), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((samples,), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((total_blocks,), dtype=torch.int64, device=atom_offsets.device),
        torch.empty((), dtype=torch.bool, device=atom_offsets.device),
    )


def _validate_offsets(offsets: Tensor, total: int, *, operand: str) -> None:
    if offsets.device.type != "cpu" or offsets.dtype is not torch.int64 or offsets.dim() != 1:
        raise ValueError(f"packed local atom windows require one-dimensional CPU int64 {operand} offsets")
    if offsets.numel() == 0:
        raise ValueError(f"packed local atom windows require {operand} offsets to include the initial zero")
    if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ValueError(f"packed local atom windows require monotonic {operand} offsets starting at zero")
    if int(offsets[-1]) != total:
        raise ValueError(
            f"packed local atom windows {operand} offsets cover {int(offsets[-1])} values, "
            f"but packed storage contains {total}"
        )


@dataclass(eq=False)
class PackedLocalAtomLayout:
    r"""Device indices and CPU lengths for one packed local-window partition."""

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
    atom: NestedTensor,
    atom_mask: NestedTensor | None,
    num_query_atoms: int,
    num_key_atoms: int,
) -> PackedLocalAtomLayout:
    r"""Build a no-padding local-window layout from canonical packed atom storage."""

    _check_nested_atom(atom, operand="atom", feature_axis=True)
    if atom_mask is not None:
        _check_nested_atom(atom_mask, operand="atom_mask", feature_axis=False)

    atom_values = atom.concat
    mask_values = None if atom_mask is None else atom_mask.concat
    atom_offsets = atom.ragged_level_offsets(0, dtype=torch.int64)
    mask_offsets = None if atom_mask is None else atom_mask.ragged_level_offsets(0, dtype=torch.int64)
    sample_metadata, block_lengths, block_to_sample, validation = canonicalize_packed_local_atom_layout(
        atom_offsets,
        mask_offsets,
        atom_values.shape[0],
        0 if mask_values is None else mask_values.shape[0],
        num_query_atoms,
        num_key_atoms,
    )
    torch._check_tensor_all(validation)

    device = atom_values.device
    samples = atom_offsets.shape[0] - 1
    metadata = sample_metadata.to(device=device)
    block_to_sample = block_to_sample.to(device=device)
    device_atom_offsets = metadata[: samples + 1]
    block_offsets = metadata[samples + 1 :]
    atom_lengths = device_atom_offsets[1:] - device_atom_offsets[:-1]

    total_atoms = atom_values.shape[0]
    device_atom_lengths = atom_lengths.to(device=device)
    atom_to_sample = torch.repeat_interleave(
        torch.arange(samples, device=device, dtype=torch.long),
        device_atom_lengths,
        output_size=total_atoms,
    )
    atom_index = torch.arange(total_atoms, device=device, dtype=torch.long)
    atom_local = atom_index - device_atom_offsets.index_select(0, atom_to_sample)
    if mask_values is None:
        valid_extent = device_atom_lengths
    else:
        valid_positions = torch.where(mask_values.to(torch.bool), atom_local + 1, 0)
        valid_extent = torch.zeros(samples, device=device, dtype=torch.long).scatter_reduce(
            0,
            atom_to_sample,
            valid_positions,
            reduce="amax",
            include_self=True,
        )

    total_blocks = block_to_sample.shape[0]
    block_index = torch.arange(total_blocks, device=device, dtype=torch.long)
    sample_block_start = block_offsets.index_select(0, block_to_sample)
    local_block = block_index - sample_block_start
    sample_atom_start = device_atom_offsets.index_select(0, block_to_sample)
    sample_atom_length = atom_lengths.index_select(0, block_to_sample)
    sample_valid_extent = valid_extent.index_select(0, block_to_sample)

    query_slot = torch.arange(num_query_atoms, device=device, dtype=torch.long)
    query_local = local_block.unsqueeze(-1) * num_query_atoms + query_slot
    query_storage_valid = query_local < sample_atom_length.unsqueeze(-1)
    query_valid = query_storage_valid
    query_safe = torch.minimum(query_local, (sample_atom_length - 1).clamp_min(0).unsqueeze(-1))
    query_atom_index = sample_atom_start.unsqueeze(-1) + query_safe

    key_padding_left = (num_key_atoms - num_query_atoms) // 2
    centered_key_start = local_block * num_query_atoms - key_padding_left
    if mask_values is None:
        key_start = centered_key_start
    else:
        shifted_key_start = torch.minimum(
            centered_key_start.clamp_min(0),
            (sample_valid_extent - num_key_atoms).clamp_min(0),
        )
        key_start = torch.where(sample_valid_extent >= num_key_atoms, shifted_key_start, centered_key_start)
    key_slot = torch.arange(num_key_atoms, device=device, dtype=torch.long)
    key_local = key_start.unsqueeze(-1) + key_slot
    key_valid = (key_local >= 0) & (key_local < sample_atom_length.unsqueeze(-1))
    key_safe = torch.minimum(key_local.clamp_min(0), (sample_atom_length - 1).clamp_min(0).unsqueeze(-1))
    key_atom_index = sample_atom_start.unsqueeze(-1) + key_safe
    key_window_index = sample_block_start.unsqueeze(-1) * num_query_atoms + key_safe

    if mask_values is not None:
        mask_bool = mask_values.to(torch.bool)
        query_valid = query_valid & mask_bool.index_select(0, query_atom_index.reshape(-1)).reshape(
            query_atom_index.shape
        )
        key_valid = key_valid & mask_bool.index_select(0, key_atom_index.reshape(-1)).reshape(key_atom_index.shape)
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


def _check_nested_atom(atom: NestedTensor, *, operand: str, feature_axis: bool) -> None:
    expected_dim = 3 if feature_axis else 2
    expected_order = (0, 1) if feature_axis else (0,)
    if not atom.batch_first:
        raise ValueError(f"packed local atom windows require a leading NestedTensor batch for {operand}")
    if len(atom) == 0:
        raise ValueError(f"packed local atom windows require a non-empty NestedTensor wrapper for {operand}")
    if atom.dim() != expected_dim or atom.ragged_dims != (0,) or atom.packed_dim_order != expected_order:
        suffix = "(A_i, C)" if feature_axis else "(A_i,)"
        raise ValueError(f"packed local atom windows require canonical {suffix} {operand} elements")


__all__ = [
    "PackedLocalAtomLayout",
    "canonicalize_packed_local_atom_layout",
    "gather_packed_atom_windows",
    "gather_packed_window_keys",
    "packed_local_atom_layout",
]
