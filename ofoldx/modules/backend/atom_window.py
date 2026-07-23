r"""
Backend dispatch for the atom/token transfer operators.

Three closely related index-driven operators move features between the atom axis and the token axis:
``atom_to_token_mean`` (masked segmented mean of atom features onto tokens), ``token_to_atom_broadcast``
(gather each atom's token row), and ``token_pair_to_atom_window_gather`` (gather token-pair features into
atom-local pair windows). Each dispatcher owns the OOps call and a pure-torch eager
implementation with the same tensor contract. Modules pass normalized inputs plus
their requested ``operator_backend`` and never branch on the backend themselves.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import oops_kernel
from .resolve import resolve_tensor_operator_backend

_oops_atom_to_token_mean = oops_kernel("atom_to_token_mean")
_oops_token_pair_to_atom_window_gather = oops_kernel("token_pair_to_atom_window_gather")
_oops_token_to_atom_broadcast = oops_kernel("token_to_atom_broadcast")

_SUPPORTED = ("oops", "eager")


def atom_to_token_mean(
    atom: Tensor,
    atom_to_token: Tensor,
    *,
    num_tokens: int,
    atom_mask: Tensor | None,
    eps: float,
    operator_backend: str,
) -> Tensor:
    r"""
    Masked mean-pool atom features onto tokens on the selected backend.

    Args:
        atom: ``[..., N_atom, C]`` atom features.
        atom_to_token: ``[..., N_atom]`` long token index for each atom.
        num_tokens: Number of output tokens.
        atom_mask: Optional ``[..., N_atom]`` valid-atom mask; padded atoms drop from sum and count.
        eps: Denominator floor (``denom.clamp(min=eps)``).
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., N_token, C]`` token features.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.atom_window import atom_to_token_mean
        >>> atom = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> atom_to_token = torch.tensor([0, 0, 1])
        >>> out = atom_to_token_mean(
        ...     atom, atom_to_token, num_tokens=2, atom_mask=None, eps=1e-6, operator_backend="eager"
        ... )
        >>> out.tolist()
        [[2.0, 3.0], [5.0, 6.0]]
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(atom, atom_to_token, atom_mask),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_atom_to_token_mean(
            atom,
            atom_to_token,
            num_tokens=num_tokens,
            atom_mask=atom_mask,
            eps=eps,
        )
    # Scatter-add values into token slots, then divide by the masked per-token count floored at eps.
    # Empty tokens stay zero.
    index = atom_to_token
    values = atom if atom_mask is None else atom * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    scatter_index = index.unsqueeze(-1).expand(*index.shape, atom.shape[-1])
    out = atom.new_zeros(*atom.shape[:-2], num_tokens, atom.shape[-1])
    out.scatter_add_(-2, scatter_index, values)

    ones = torch.ones(*index.shape, 1, device=atom.device, dtype=atom.dtype)
    if atom_mask is not None:
        ones = ones * atom_mask.to(dtype=atom.dtype).unsqueeze(-1)
    denom = atom.new_zeros(*atom.shape[:-2], num_tokens, 1)
    denom.scatter_add_(-2, index.unsqueeze(-1), ones)
    return out / denom.clamp(min=eps)


def token_to_atom_broadcast(token: Tensor, atom_to_token: Tensor, *, operator_backend: str) -> Tensor:
    r"""
    Broadcast token features to atoms by gathering each atom's token row on the selected backend.

    Args:
        token: ``[..., N_token, C]`` token features.
        atom_to_token: ``[..., N_atom]`` long token index for each atom.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., N_atom, C]`` atom features.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.atom_window import token_to_atom_broadcast
        >>> token = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        >>> atom_to_token = torch.tensor([0, 1, 1])
        >>> token_to_atom_broadcast(token, atom_to_token, operator_backend="eager").tolist()
        [[10.0, 11.0], [20.0, 21.0], [20.0, 21.0]]
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(token, atom_to_token),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_token_to_atom_broadcast(token, atom_to_token)
    # Gather the token row for every atom along the token axis.
    gather_idx = atom_to_token.unsqueeze(-1).expand(*atom_to_token.shape, token.shape[-1])
    return token.gather(-2, gather_idx)


def token_pair_to_atom_window_gather(
    token_pair: Tensor,
    atom_to_token: Tensor,
    *,
    n_queries: int,
    n_keys: int,
    operator_backend: str,
    num_valid_atoms: int | None = None,
) -> Tensor:
    r"""
    Gather token-pair features into atom-local pair windows on the selected backend.

    Args:
        token_pair: ``[..., N_token, N_token, C]`` token-pair features.
        atom_to_token: ``[..., N_atom]`` long token indices. A shared 1-D or
            broadcasted map lets OOps select its Triton gather when supported.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., N_blocks, n_queries, n_keys, C]`` local atom-pair features.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.backend.atom_window import token_pair_to_atom_window_gather
        >>> token_pair = torch.arange(16).reshape(4, 4, 1).float()
        >>> atom_to_token = torch.tensor([0, 1, 2, 3])
        >>> out = token_pair_to_atom_window_gather(
        ...     token_pair, atom_to_token, n_queries=2, n_keys=4, operator_backend="eager"
        ... )
        >>> out.shape
        torch.Size([2, 2, 4, 1])
    """

    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(token_pair, atom_to_token),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops":
        return _oops_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            n_queries=n_queries,
            n_keys=n_keys,
            num_valid_atoms=num_valid_atoms,
        )
    # Build per-window query/key token indices (zero-padded past the atom range, or clamped and shifted
    # inward when num_valid_atoms is set), then gather token_pair[..., query, key, :].
    query_indices, key_indices = _local_indices(atom_to_token, n_queries, n_keys, num_valid_atoms)
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
    atom_to_token: Tensor, n_queries: int, n_keys: int, num_valid_atoms: int | None = None
) -> tuple[Tensor, Tensor]:
    r"""
    Build the per-window query/key token indices for the atom-local pair windows.

    Splits the atoms into ``ceil(N_atom / n_queries)`` blocks and, for each block, records the token
    index of every query atom and of the ``n_keys`` atoms in the key window. With ``num_valid_atoms`` unset
    the key window is fixed and centered, with zeros beyond the atom range. With ``num_valid_atoms`` given
    (and ``>= n_keys``), the window is clamped and shifted inward so
    every key is a real atom and the pair-bias windows match the attention key gather.

    Args:
        atom_to_token: ``[..., N_atom]`` long token indices.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        num_valid_atoms: Number of real atoms (contiguous at the front); enables the clamped layout.

    Returns:
        A ``(query_indices, key_indices)`` pair of shapes ``[N_blocks, n_queries]`` and
        ``[N_blocks, n_keys]`` holding token indices into the pair axes.
    """
    atoms = atom_to_token.shape[-1]
    n_blocks = math.ceil(atoms / n_queries)
    q_pad = n_blocks * n_queries - atoms
    pad_left = (n_keys - n_queries) // 2
    query_indices = F.pad(atom_to_token, (0, q_pad), value=0).reshape(*atom_to_token.shape[:-1], n_blocks, n_queries)
    if num_valid_atoms is not None and num_valid_atoms >= n_keys:
        starts = (torch.arange(n_blocks, device=atom_to_token.device) * n_queries - pad_left).clamp(
            0, num_valid_atoms - n_keys
        )
        key_atom_index = starts[:, None] + torch.arange(n_keys, device=atom_to_token.device)[None, :]
        key_indices = atom_to_token[..., key_atom_index]
        return query_indices, key_indices
    pad_right = int((n_blocks - 1 / 2) * n_queries + n_keys / 2 - atoms + 1 / 2)
    key_indices = F.pad(atom_to_token, (pad_left, pad_right), value=0).unfold(-1, n_keys, n_queries)
    return query_indices, key_indices
