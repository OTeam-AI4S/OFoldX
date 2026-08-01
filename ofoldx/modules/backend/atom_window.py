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
Backend-dispatched atom/token transfer operators.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ..._oops import oops_kernel
from .resolve import resolve_tensor_backend

if TYPE_CHECKING:
    from ..atom.nested import PackedLocalAtomLayout

_oops_atom_to_token_mean = cast(Callable[..., Tensor], oops_kernel("atom_to_token_mean"))
_oops_token_pair_to_atom_window_gather = cast(Callable[..., Tensor], oops_kernel("token_pair_to_atom_window_gather"))
_oops_token_to_atom_broadcast = cast(Callable[..., Tensor], oops_kernel("token_to_atom_broadcast"))

_SUPPORTED = ("oops", "eager")


def _broadcast_atom_axis(tensor: Tensor, target_prefix: tuple[int, ...]) -> Tensor:
    r"""Broadcast a sample-independent atom axis across missing logical prefix dimensions."""

    for _ in range(len(target_prefix) - (tensor.dim() - 1)):
        tensor = tensor.unsqueeze(-2)
    return tensor.expand(*target_prefix, -1)


def _sample_tensor(
    value: Tensor | NestedTensor | None, sample: int, tail_dims: int, batch_first: bool
) -> Tensor | None:
    r"""Select one logical batch row, retaining shared tensors without a batch prefix."""

    if value is None:
        return None
    if isinstance(value, NestedTensor):
        return cast(Tensor, value[sample])
    if value.dim() > tail_dims:
        batch_axis = 0 if batch_first else 1
        return value.select(batch_axis, 0 if value.shape[batch_axis] == 1 else sample)
    return value


def atom_to_token_mean(
    atom: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor,
    *,
    token_reference: Tensor | NestedTensor,
    atom_mask: Tensor | NestedTensor | None,
    backend: str,
) -> Tensor | NestedTensor:
    r"""
    Exact masked mean-pool of atom features onto tokens on the selected backend.

    Args:
        atom: ``[..., N_atom, C]`` atom features.
        atom_to_token: ``[..., N_atom]`` long token index for each atom.
        token_reference: Token-axis tensor whose ``N_token`` topology defines the
            output. Its values are not read.
        atom_mask: Optional ``[..., N_atom]`` valid-atom mask; padded atoms drop from sum and count.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., N_token, C]`` token features; tokens without a valid atom are zero.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(atom, atom_to_token, atom_mask),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_atom_to_token_mean,
    )
    if backend == "oops":
        if isinstance(atom, NestedTensor):
            if not isinstance(token_reference, NestedTensor):
                raise ValueError("nested atom_to_token_mean requires a NestedTensor token_reference")
            return _oops_atom_to_token_mean(
                atom,
                atom_to_token,
                token_reference=token_reference,
                atom_mask=atom_mask,
            )
        if isinstance(token_reference, NestedTensor):
            raise ValueError("dense atom_to_token_mean requires a dense token_reference")
        if atom_mask is not None:
            atom_mask = _broadcast_atom_axis(atom_mask, tuple(atom.shape[:-2]))
        return _oops_atom_to_token_mean(
            atom,
            atom_to_token,
            num_tokens=token_reference.shape[-2],
            atom_mask=atom_mask,
        )
    nested_reference = next(
        (value for value in (atom, atom_to_token, token_reference, atom_mask) if isinstance(value, NestedTensor)), None
    )
    if nested_reference is not None:
        outputs = [
            atom_to_token_mean(
                cast(Tensor, _sample_tensor(atom, sample, 2, nested_reference.batch_first)),
                cast(Tensor, _sample_tensor(atom_to_token, sample, 1, nested_reference.batch_first)),
                token_reference=cast(Tensor, _sample_tensor(token_reference, sample, 2, nested_reference.batch_first)),
                atom_mask=_sample_tensor(atom_mask, sample, 1, nested_reference.batch_first),
                backend="eager",
            )
            for sample in range(len(nested_reference))
        ]
        prefix_axes = ()
        if isinstance(atom, NestedTensor):
            prefix_axes = tuple(axis for axis in atom.ragged_dims if axis < outputs[0].dim() - 2)
        if isinstance(token_reference, NestedTensor):
            extra_axes = outputs[0].dim() - token_reference[0].dim()
            token_axes = tuple(axis + extra_axes for axis in token_reference.ragged_dims)
        else:
            token_axes = ()
        if prefix_axes or isinstance(token_reference, NestedTensor):
            return NestedTensor(
                outputs,
                ragged_dims=tuple(sorted({*prefix_axes, *token_axes})),
                batch_first=nested_reference.batch_first,
            )
        return torch.stack(outputs, dim=0 if nested_reference.batch_first else 1)
    num_tokens = token_reference.shape[-2]
    # Scatter-add values into token slots, then divide by the masked per-token count.
    # Empty tokens stay zero.
    index = _broadcast_atom_axis(atom_to_token, tuple(atom.shape[:-2]))
    if atom_mask is not None:
        atom_mask = _broadcast_atom_axis(atom_mask, tuple(atom.shape[:-2]))
    values = atom
    if atom_mask is not None:
        values = values.masked_fill((atom_mask == 0).unsqueeze(-1) & ~torch.isfinite(values), 0)
        values = values * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    scatter_index = index.unsqueeze(-1).expand(*index.shape, atom.shape[-1])
    out = atom.new_zeros(*atom.shape[:-2], num_tokens, atom.shape[-1])
    out.scatter_add_(-2, scatter_index, values)

    ones = torch.ones(*index.shape, 1, device=atom.device, dtype=atom.dtype)
    if atom_mask is not None:
        ones = ones * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    denom = atom.new_zeros(*atom.shape[:-2], num_tokens, 1)
    denom.scatter_add_(-2, index.unsqueeze(-1), ones)
    return out / torch.where(denom > 0, denom, torch.ones_like(denom))


def token_to_atom_broadcast(token: Tensor, atom_to_token: Tensor, *, backend: str) -> Tensor:
    r"""
    Broadcast token features to atoms by gathering each atom's token row on the selected backend.

    Args:
        token: ``[..., N_token, C]`` token features.
        atom_to_token: ``[..., N_atom]`` long token index for each atom.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., N_atom, C]`` atom features.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(token, atom_to_token),
        supported=_SUPPORTED,
        public_default="eager",
        oops=_oops_token_to_atom_broadcast,
    )
    if backend == "oops":
        return _oops_token_to_atom_broadcast(token, atom_to_token)
    reference = next((value for value in (token, atom_to_token) if isinstance(value, NestedTensor)), None)
    if reference is not None and not reference.batch_first:
        outputs = [
            token_to_atom_broadcast(
                cast(Tensor, _sample_tensor(token, sample, 2, False)),
                cast(Tensor, _sample_tensor(atom_to_token, sample, 1, False)),
                backend="eager",
            )
            for sample in range(len(reference))
        ]
        prefix_axes = (
            tuple(axis for axis in token.ragged_dims if axis < token.dim() - 3)
            if isinstance(token, NestedTensor)
            else ()
        )
        atom_axes = ()
        if isinstance(atom_to_token, NestedTensor):
            extra_axes = outputs[0].dim() - atom_to_token.dim()
            atom_axes = tuple(axis + extra_axes for axis in atom_to_token.ragged_dims)
        if prefix_axes or isinstance(atom_to_token, NestedTensor):
            return NestedTensor(
                outputs,
                ragged_dims=tuple(sorted({*prefix_axes, *atom_axes})),
                batch_first=atom_to_token.batch_first if isinstance(atom_to_token, NestedTensor) else False,
            )
        return torch.stack(outputs, dim=1)
    # Gather the token row for every atom along the token axis.
    atom_to_token = _broadcast_atom_axis(atom_to_token, tuple(token.shape[:-2]))
    gather_idx = atom_to_token.unsqueeze(-1).expand(*((-1,) * atom_to_token.dim()), token.shape[-1])
    return token.gather(-2, gather_idx)


def token_pair_to_atom_window_gather(
    token_pair: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor,
    *,
    num_query_atoms: int,
    num_key_atoms: int,
    backend: str,
    key_index: Tensor | NestedTensor | None = None,
    packed_layout: PackedLocalAtomLayout | None = None,
) -> Tensor | NestedTensor:
    r"""
    Gather token-pair features into atom-local pair windows on the selected backend.

    Args:
        token_pair: ``[..., N_token, N_token, C]`` token-pair features.
        atom_to_token: ``[..., N_atom]`` long token indices. Shared 1-D or
            broadcasted maps are supported where compatible with the selected backend.
        num_query_atoms: Query atoms per window.
        num_key_atoms: Key atoms gathered around each query window.
        backend: Operator provider; see [ofoldx.modules.backend][].
        key_index: Optional ``[..., N_block, num_key_atoms]`` atom indices carrying an
            explicit row-specific key layout. ``-1`` marks invalid keys.
        packed_layout: Exact prepared packed query/key indices. When present, the
            indexed PyTorch composition consumes those choices directly.

    Returns:
        ``[..., N_blocks, num_query_atoms, num_key_atoms, C]`` local atom-pair features.
    """

    if packed_layout is not None:
        resolve_tensor_backend(
            backend, tensors=(token_pair, atom_to_token), supported=("eager",), public_default="eager"
        )
        if not isinstance(token_pair, NestedTensor) or not isinstance(atom_to_token, NestedTensor):
            raise ValueError("a prepared packed window layout requires nested pair and atom mappings")
        return _prepared_token_pair_windows(token_pair, atom_to_token, packed_layout)
    nested = isinstance(token_pair, NestedTensor)
    shared_mapping = (
        isinstance(atom_to_token, NestedTensor)
        or atom_to_token.dim() == 1
        or all(
            size == 1 or stride == 0
            for size, stride in zip(atom_to_token.shape[:-1], atom_to_token.stride()[:-1], strict=True)
        )
    )
    oops_can_reconstruct = key_index is None and shared_mapping
    backend = resolve_tensor_backend(
        backend,
        tensors=(token_pair, atom_to_token, key_index),
        supported=_SUPPORTED if oops_can_reconstruct else ("eager",),
        public_default="eager",
        oops=_oops_token_pair_to_atom_window_gather,
    )
    if backend == "oops":
        if nested and not isinstance(atom_to_token, NestedTensor):
            raise ValueError("nested token_pair_to_atom_window_gather requires a NestedTensor atom_to_token")
        return _oops_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            n_queries=num_query_atoms,
            n_keys=num_key_atoms,
        )
    nested_reference = next(
        (value for value in (token_pair, atom_to_token, key_index) if isinstance(value, NestedTensor)), None
    )
    if nested_reference is not None:
        outputs = [
            token_pair_to_atom_window_gather(
                cast(Tensor, _sample_tensor(token_pair, sample, 3, nested_reference.batch_first)),
                cast(Tensor, _sample_tensor(atom_to_token, sample, 1, nested_reference.batch_first)),
                num_query_atoms=num_query_atoms,
                num_key_atoms=num_key_atoms,
                backend="eager",
                key_index=_sample_tensor(key_index, sample, 2, nested_reference.batch_first),
            )
            for sample in range(len(nested_reference))
        ]
        window_axis = outputs[0].dim() - 4
        prefix_axes = tuple(axis for axis in nested_reference.ragged_dims if axis < window_axis)
        return NestedTensor(outputs, ragged_dims=(*prefix_axes, window_axis), batch_first=nested_reference.batch_first)
    atom_to_token = _broadcast_atom_axis(atom_to_token, tuple(token_pair.shape[:-3]))
    if atom_to_token.shape[-1] == 0:
        return (
            token_pair.sum(dim=(-3, -2))
            .reshape(*token_pair.shape[:-3], 1, 1, 1, token_pair.shape[-1])
            .expand(*token_pair.shape[:-3], 0, num_query_atoms, num_key_atoms, token_pair.shape[-1])
        )
    query_indices, key_indices = _local_indices(atom_to_token, num_query_atoms, num_key_atoms, key_index)
    prefix = tuple(token_pair.shape[:-3])
    num_windows = query_indices.shape[-2]
    pair_size = token_pair.shape[-1]
    flat_token_pair = token_pair.reshape(-1, token_pair.shape[-3], token_pair.shape[-2], pair_size)
    flat_query_indices = query_indices.expand(*prefix, num_windows, num_query_atoms).reshape(
        -1, num_windows, num_query_atoms
    )
    flat_key_indices = key_indices.expand(*prefix, num_windows, num_key_atoms).reshape(-1, num_windows, num_key_atoms)
    batch_index = torch.arange(flat_token_pair.shape[0], device=token_pair.device).view(-1, 1, 1, 1)
    row_index = flat_query_indices.unsqueeze(-1).expand(-1, -1, -1, num_key_atoms)
    column_index = flat_key_indices.unsqueeze(-2).expand(-1, -1, num_query_atoms, -1)
    gathered = flat_token_pair[batch_index, row_index, column_index]
    return gathered.reshape(*prefix, num_windows, num_query_atoms, num_key_atoms, pair_size)


def _prepared_token_pair_windows(
    token_pair: NestedTensor, atom_to_token: NestedTensor, layout: PackedLocalAtomLayout
) -> NestedTensor:
    """Gather the exact prepared windows from canonical packed token pairs."""
    if (
        token_pair.ragged_dims != (0, 1)
        or token_pair.packed_dim_order != (0, 1, 2)
        or atom_to_token.ragged_dims != (0,)
        or atom_to_token.packed_dim_order != (0,)
    ):
        raise ValueError("prepared pair windows require canonical (N,N,C) pairs and (A,) atom mappings")
    if len(token_pair) != len(atom_to_token) or layout.block_lengths.shape != (len(token_pair),):
        raise ValueError("prepared pair windows must share the pair and atom batch axis")
    device = token_pair.device
    sizes = token_pair.element_sizes().to(device=device)
    block_to_sample = torch.repeat_interleave(
        torch.arange(len(token_pair), device=device),
        layout.block_lengths.to(device=device),
        output_size=layout.query_atom_index.shape[0],
    )
    token_count = sizes[:, 0].index_select(0, block_to_sample)
    mapping = atom_to_token.concat
    queries = mapping.index_select(0, layout.query_atom_index.reshape(-1)).reshape(layout.query_atom_index.shape)
    keys = mapping.index_select(0, layout.key_atom_index.reshape(-1)).reshape(layout.key_atom_index.shape)
    queries = torch.where(layout.query_storage_valid, queries, 0)
    keys = torch.where(layout.key_valid, keys, 0)
    valid = (
        ((queries >= 0) & (queries < token_count[:, None])).all()
        & ((keys >= 0) & (keys < token_count[:, None])).all()
        & (sizes[:, 0] == sizes[:, 1]).all()
    )
    if torch.compiler.is_compiling():
        torch._assert_async(valid, "prepared window token indices must address their square pair sample")
    elif not bool(valid):
        raise ValueError("prepared window token indices must address their square pair sample")
    starts = token_pair.packed_offsets(device=device, dtype=torch.long)[:-1].index_select(0, block_to_sample)
    indices = starts[:, None, None] + queries[:, :, None] * token_count[:, None, None] + keys[:, None, :]
    values = token_pair.concat.index_select(0, indices.reshape(-1)).reshape(*indices.shape, token_pair.shape[-1])
    return atom_to_token.packed_with_lengths(values, layout.block_lengths)


def _local_indices(
    atom_to_token: Tensor,
    num_query_atoms: int,
    num_key_atoms: int,
    key_index: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the per-window query/key token indices for the atom-local pair windows.

    Splits the atoms into ``ceil(N_atom / num_query_atoms)`` blocks and, for each block, records the token
    index of every query atom and of the ``num_key_atoms`` atoms in the key window.
    ``key_index`` supplies the exact atom indices used by geometry and attention;
    otherwise the fixed centred layout is constructed.

    Args:
        atom_to_token: ``[..., N_atom]`` long token indices.
        num_query_atoms: Query atoms per window.
        num_key_atoms: Key atoms gathered around each query window.
        key_index: Optional ``[..., N_block, num_key_atoms]`` atom indices. ``-1`` marks
            invalid keys.

    Returns:
        A ``(query_indices, key_indices)`` pair of shapes ``[N_blocks, num_query_atoms]`` and
        ``[N_blocks, num_key_atoms]`` holding token indices into the pair axes.
    """
    num_atoms = atom_to_token.shape[-1]
    num_windows = math.ceil(num_atoms / num_query_atoms)
    q_pad = num_windows * num_query_atoms - num_atoms
    pad_left = (num_key_atoms - num_query_atoms) // 2
    query_indices = F.pad(atom_to_token, (0, q_pad), value=0).reshape(
        *atom_to_token.shape[:-1], num_windows, num_query_atoms
    )
    if key_index is not None:
        source_prefix = tuple(key_index.shape[:-2])
        target_prefix = tuple(atom_to_token.shape[:-1])
        missing = len(target_prefix) - len(source_prefix)
        if missing:
            key_index = key_index.reshape(*source_prefix, *((1,) * missing), *key_index.shape[-2:])
        key_index = key_index.expand(*target_prefix, num_windows, num_key_atoms)
        key_indices = (
            atom_to_token.unsqueeze(-2)
            .expand(*target_prefix, num_windows, num_atoms)
            .gather(-1, key_index.clamp_min(0))
        )
        key_indices = torch.where(key_index >= 0, key_indices, 0)
        return query_indices, key_indices
    pad_right = int((num_windows - 1 / 2) * num_query_atoms + num_key_atoms / 2 - num_atoms + 1 / 2)
    key_indices = F.pad(atom_to_token, (pad_left, pad_right), value=0).unfold(-1, num_key_atoms, num_query_atoms)
    return query_indices, key_indices
