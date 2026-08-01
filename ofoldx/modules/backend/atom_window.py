r"""
Backend-dispatched atom/token transfer operators.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_backend

_oops_atom_to_token_mean = cast(Callable[..., Tensor], oops_kernel("atom_to_token_mean"))
_oops_token_pair_to_atom_window_gather = cast(Callable[..., Tensor], oops_kernel("token_pair_to_atom_window_gather"))
_oops_token_to_atom_broadcast = cast(Callable[..., Tensor], oops_kernel("token_to_atom_broadcast"))

_SUPPORTED = ("oops", "eager")


def _broadcast_atom_axis(tensor: Tensor, target_prefix: tuple[int, ...]) -> Tensor:
    r"""Broadcast a sample-independent atom axis across missing logical prefix dimensions."""

    atoms = tensor.shape[-1]
    source_prefix = tuple(tensor.shape[:-1])
    if len(source_prefix) > len(target_prefix) or source_prefix != target_prefix[: len(source_prefix)]:
        raise ValueError(f"atom-axis prefix {source_prefix} is not broadcastable to {target_prefix}")
    if len(source_prefix) < len(target_prefix):
        tensor = tensor.reshape(*source_prefix, *((1,) * (len(target_prefix) - len(source_prefix))), atoms)
    return tensor.expand(*target_prefix, atoms)


def atom_to_token_mean(
    atom: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor,
    *,
    token_reference: Tensor | NestedTensor,
    atom_mask: Tensor | NestedTensor | None,
    denominator_offset: float | None,
    backend: str,
) -> Tensor | NestedTensor:
    r"""
    Masked mean-pool atom features onto tokens on the selected backend.

    Args:
        atom: ``[..., N_atom, C]`` atom features.
        atom_to_token: ``[..., N_atom]`` long token index for each atom.
        token_reference: Token-axis tensor whose ``N_token`` topology defines the
            output. Its values are not read.
        atom_mask: Optional ``[..., N_atom]`` valid-atom mask; padded atoms drop from sum and count.
        denominator_offset: Additive offset for non-empty token counts. ``None``
            selects the input dtype's smallest positive value.
        backend: Operator provider; see [ofoldx.modules.backend][].

    Returns:
        ``[..., N_token, C]`` token features.
    """

    backend = resolve_tensor_backend(
        backend,
        tensors=(atom, atom_to_token, atom_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    offset = torch.finfo(atom.dtype).tiny if denominator_offset is None else denominator_offset
    if backend == "oops":
        if isinstance(atom, NestedTensor):
            if not isinstance(token_reference, NestedTensor):
                raise ValueError("nested atom_to_token_mean requires a NestedTensor token_reference")
            return _oops_atom_to_token_mean(
                atom,
                atom_to_token,
                token_reference=token_reference,
                atom_mask=atom_mask,
                denominator_offset=offset,
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
            denominator_offset=offset,
        )
    if isinstance(token_reference, NestedTensor):
        raise ValueError("the eager atom_to_token_mean backend does not serve NestedTensor inputs")
    num_tokens = token_reference.shape[-2]
    # Scatter-add values into token slots, then divide by the masked per-token count plus its offset.
    # Empty tokens stay zero.
    index = _broadcast_atom_axis(atom_to_token, tuple(atom.shape[:-2]))
    if atom_mask is not None:
        atom_mask = _broadcast_atom_axis(atom_mask, tuple(atom.shape[:-2]))
    values = atom if atom_mask is None else atom * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    scatter_index = index.unsqueeze(-1).expand(*index.shape, atom.shape[-1])
    out = atom.new_zeros(*atom.shape[:-2], num_tokens, atom.shape[-1])
    out.scatter_add_(-2, scatter_index, values)

    ones = torch.ones(*index.shape, 1, device=atom.device, dtype=atom.dtype)
    if atom_mask is not None:
        ones = ones * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    denom = atom.new_zeros(*atom.shape[:-2], num_tokens, 1)
    denom.scatter_add_(-2, index.unsqueeze(-1), ones)
    denominator = torch.where(denom > 0, denom + offset, torch.ones_like(denom))
    return out / denominator


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
    )
    if backend == "oops":
        return _oops_token_to_atom_broadcast(token, atom_to_token)
    # Gather the token row for every atom along the token axis.
    atom_to_token = _broadcast_atom_axis(atom_to_token, tuple(token.shape[:-2]))
    gather_idx = atom_to_token.unsqueeze(-1).expand(*atom_to_token.shape, token.shape[-1])
    return token.gather(-2, gather_idx)


def token_pair_to_atom_window_gather(
    token_pair: Tensor | NestedTensor,
    atom_to_token: Tensor | NestedTensor,
    *,
    n_queries: int,
    n_keys: int,
    backend: str,
    key_index: Tensor | NestedTensor | None = None,
    valid_extent: Tensor | None = None,
) -> Tensor | NestedTensor:
    r"""
    Gather token-pair features into atom-local pair windows on the selected backend.

    Args:
        token_pair: ``[..., N_token, N_token, C]`` token-pair features.
        atom_to_token: ``[..., N_atom]`` long token indices. A shared 1-D or
            broadcasted map lets OOps select its Triton gather when supported.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        backend: Operator provider; see [ofoldx.modules.backend][].
        key_index: Optional ``[..., N_block, n_keys]`` atom indices carrying an
            explicit row-specific key layout. ``-1`` marks invalid keys.
        valid_extent: Optional one-dimensional CPU or CUDA integer tensor carrying
            one exclusive valid-atom extent per NestedTensor sample. CUDA extents
            remain device-resident while OOps reconstructs the row-specific
            centred/clamped layout from this compact metadata.

    Returns:
        ``[..., N_blocks, n_queries, n_keys, C]`` local atom-pair features.
    """

    nested = isinstance(token_pair, NestedTensor)
    oops_can_reconstruct = key_index is None or (nested and valid_extent is not None)
    backend = resolve_tensor_backend(
        backend,
        tensors=(token_pair, atom_to_token, key_index),
        supported=_SUPPORTED if oops_can_reconstruct else ("eager",),
        public_default="eager",
    )
    if backend == "oops":
        if nested and not isinstance(atom_to_token, NestedTensor):
            raise ValueError("nested token_pair_to_atom_window_gather requires a NestedTensor atom_to_token")
        return _oops_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            n_queries=n_queries,
            n_keys=n_keys,
            num_valid_atoms=valid_extent,
        )
    atom_to_token = _broadcast_atom_axis(atom_to_token, tuple(token_pair.shape[:-3]))
    query_indices, key_indices = _local_indices(atom_to_token, n_queries, n_keys, key_index)
    prefix = tuple(token_pair.shape[:-3])
    blocks = query_indices.shape[-2]
    channels = token_pair.shape[-1]
    flat_token_pair = token_pair.reshape(-1, token_pair.shape[-3], token_pair.shape[-2], channels)
    flat_query_indices = query_indices.expand(*prefix, blocks, n_queries).reshape(-1, blocks, n_queries)
    flat_key_indices = key_indices.expand(*prefix, blocks, n_keys).reshape(-1, blocks, n_keys)
    batch_index = torch.arange(flat_token_pair.shape[0], device=token_pair.device).view(-1, 1, 1, 1)
    row_index = flat_query_indices.unsqueeze(-1).expand(-1, -1, -1, n_keys)
    column_index = flat_key_indices.unsqueeze(-2).expand(-1, -1, n_queries, -1)
    gathered = flat_token_pair[batch_index, row_index, column_index]
    return gathered.reshape(*prefix, blocks, n_queries, n_keys, channels)


def _local_indices(
    atom_to_token: Tensor,
    n_queries: int,
    n_keys: int,
    key_index: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the per-window query/key token indices for the atom-local pair windows.

    Splits the atoms into ``ceil(N_atom / n_queries)`` blocks and, for each block, records the token
    index of every query atom and of the ``n_keys`` atoms in the key window.
    ``key_index`` supplies the exact atom indices used by geometry and attention;
    otherwise the fixed centred layout is constructed.

    Args:
        atom_to_token: ``[..., N_atom]`` long token indices.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        key_index: Optional ``[..., N_block, n_keys]`` atom indices. ``-1`` marks
            invalid keys.

    Returns:
        A ``(query_indices, key_indices)`` pair of shapes ``[N_blocks, n_queries]`` and
        ``[N_blocks, n_keys]`` holding token indices into the pair axes.
    """
    atoms = atom_to_token.shape[-1]
    n_blocks = math.ceil(atoms / n_queries)
    q_pad = n_blocks * n_queries - atoms
    pad_left = (n_keys - n_queries) // 2
    query_indices = F.pad(atom_to_token, (0, q_pad), value=0).reshape(*atom_to_token.shape[:-1], n_blocks, n_queries)
    if key_index is not None:
        source_prefix = tuple(key_index.shape[:-2])
        target_prefix = tuple(atom_to_token.shape[:-1])
        missing = len(target_prefix) - len(source_prefix)
        if missing:
            key_index = key_index.reshape(*source_prefix, *((1,) * missing), *key_index.shape[-2:])
        key_index = key_index.expand(*target_prefix, n_blocks, n_keys)
        key_indices = (
            atom_to_token.unsqueeze(-2).expand(*target_prefix, n_blocks, atoms).gather(-1, key_index.clamp_min(0))
        )
        key_indices = torch.where(key_index >= 0, key_indices, 0)
        return query_indices, key_indices
    pad_right = int((n_blocks - 1 / 2) * n_queries + n_keys / 2 - atoms + 1 / 2)
    key_indices = F.pad(atom_to_token, (pad_left, pad_right), value=0).unfold(-1, n_keys, n_queries)
    return query_indices, key_indices
