r"""
Backend dispatch for the atom/token transfer operators.

Three closely related index-driven operators move features between the atom axis and the token axis:
``atom_to_token_mean`` (masked segmented mean of atom features onto tokens), ``token_to_atom_broadcast``
(gather each atom's token row), and ``token_pair_to_atom_window_gather`` (gather token-pair features into
atom-local pair windows). Each dispatcher owns its CUDA guard, the OOps call, and a pure-torch eager
reference numerically equivalent to the OOps kernel. Modules pass normalized inputs plus their requested
``operator_backend`` and never branch on the backend themselves.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ._oops import OOpsUnsupportedError, oops_kernel
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

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

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
        if not atom.is_cuda:
            raise OOpsUnsupportedError("operator_backend 'oops' for atom_to_token_mean requires CUDA tensors")
        return _oops_atom_to_token_mean(
            atom,
            atom_to_token,
            num_tokens=num_tokens,
            atom_mask=atom_mask,
            eps=eps,
        )
    # Eager reference mirrors oops._atom_to_token_mean_index: masked scatter-add of values into token slots,
    # divided by the masked per-token count floored at eps (empty tokens stay zero).
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

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

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
        if not token.is_cuda:
            raise OOpsUnsupportedError("operator_backend 'oops' for token_to_atom_broadcast requires CUDA tensors")
        return _oops_token_to_atom_broadcast(token, atom_to_token)
    # Eager reference mirrors oops._token_to_atom_broadcast_index (batched branch): gather the token row for
    # every atom along the token axis.
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
        atom_to_token: 1-D ``[N_atom]`` long token index shared across the batch.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        operator_backend: ``"auto"`` / ``"oops"`` / ``"eager"``.

    Returns:
        ``[..., N_blocks, n_queries, n_keys, C]`` local atom-pair features.

    Raises:
        OOpsUnsupportedError: If the resolved backend is ``"oops"`` on non-CUDA tensors.

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

    # The DeepMind AF3 clamped/shifted key window is not expressible to the OOps kernel (which takes only
    # n_queries/n_keys), so when it is requested fall back to the eager gather that implements it.
    clamped = num_valid_atoms is not None and num_valid_atoms >= n_keys
    backend = resolve_tensor_operator_backend(
        operator_backend,
        tensors=(token_pair, atom_to_token),
        supported=_SUPPORTED,
        public_default="eager",
    )
    if backend == "oops" and not clamped:
        if not token_pair.is_cuda:
            raise OOpsUnsupportedError(
                "operator_backend 'oops' for token_pair_to_atom_window_gather requires CUDA tensors"
            )
        return _oops_token_pair_to_atom_window_gather(token_pair, atom_to_token, n_queries=n_queries, n_keys=n_keys)
    # Eager reference mirrors oops._token_pair_to_atom_window_gather_torch: build the per-window query/key token
    # indices (zero-padded past the atom range, or clamped/shifted inwards when num_valid_atoms is set) and
    # gather token_pair[..., query, key, :].
    query_indices, key_indices = _local_indices(atom_to_token, n_queries, n_keys, num_valid_atoms)
    query_indices = query_indices.unsqueeze(-1).expand(-1, -1, n_keys)
    key_indices = key_indices.unsqueeze(1).expand(-1, n_queries, -1)
    return token_pair[..., query_indices, key_indices, :]


def _local_indices(
    atom_to_token: Tensor, n_queries: int, n_keys: int, num_valid_atoms: int | None = None
) -> tuple[Tensor, Tensor]:
    r"""
    Build the per-window query/key token indices for the atom-local pair windows.

    Splits the atoms into ``ceil(N_atom / n_queries)`` blocks and, for each block, records the token
    index of every query atom and of the ``n_keys`` atoms in the key window. With ``num_valid_atoms`` unset
    the key window is the legacy centered window, zero-padded past the atom range (matching the OOps kernel).
    With ``num_valid_atoms`` given (and ``>= n_keys``) the DeepMind AF3 clamped/shifted window is used, so
    every key is a real atom and the pair-bias windows match the attention key gather.

    Args:
        atom_to_token: 1-D ``[N_atom]`` long token index shared across the batch.
        n_queries: Query atoms per window.
        n_keys: Key atoms gathered around each query window.
        num_valid_atoms: Number of real atoms (contiguous at the front); enables the clamped layout.

    Returns:
        A ``(query_indices, key_indices)`` pair of shapes ``[N_blocks, n_queries]`` and
        ``[N_blocks, n_keys]`` holding token indices into the pair axes.
    """
    atoms = atom_to_token.numel()
    n_blocks = math.ceil(atoms / n_queries)
    q_pad = n_blocks * n_queries - atoms
    pad_left = (n_keys - n_queries) // 2
    query_indices = F.pad(atom_to_token, (0, q_pad), value=0).reshape(n_blocks, n_queries)
    if num_valid_atoms is not None and num_valid_atoms >= n_keys:
        starts = (torch.arange(n_blocks, device=atom_to_token.device) * n_queries - pad_left).clamp(
            0, num_valid_atoms - n_keys
        )
        key_atom_index = starts[:, None] + torch.arange(n_keys, device=atom_to_token.device)[None, :]
        key_indices = atom_to_token[key_atom_index]
        return query_indices, key_indices
    pad_right = int((n_blocks - 1 / 2) * n_queries + n_keys / 2 - atoms + 1 / 2)
    key_indices = F.pad(atom_to_token, (pad_left, pad_right), value=0).unfold(0, n_keys, n_queries)
    return query_indices, key_indices
