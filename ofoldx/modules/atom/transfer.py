r"""
Atom/token transfer modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import atom_to_token_mean as dispatch_atom_to_token_mean
from ..backend import resolve_config_backend
from ..backend import token_pair_to_atom_window_gather as dispatch_token_pair_to_atom_window_gather
from ..backend import token_to_atom_broadcast as dispatch_token_to_atom_broadcast
from .window import LocalAtomWindowing


class AtomToTokenMean(nn.Module):
    r"""
    Mean-pool per-atom features to tokens by scattering on the atom->token index.

    Empty tokens stay zero. An optional ``atom_mask`` zeroes padded atoms before the mean;
    arbitrary leading dimensions are supported by ``_expand_atom_axis``.

    Args:
        backend: Which backend to run; see [`backend`][ofoldx.modules.backend].
    """

    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "eager"),
            public_default="eager",
        )

    def forward(
        self,
        atom_hidden: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Aggregate atom features onto their tokens.

        Args:
            atom_hidden: ``[..., N_atom, C]`` atom features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.
            num_tokens: Number of output tokens.
            atom_mask: Optional ``[..., N_atom]`` valid-atom mask.

        Returns:
            ``[..., N_token, C]`` token features; tokens with no atoms stay zero.
        """

        atom_to_token = _expand_atom_axis(atom_to_token, tuple(atom_hidden.shape[:-2]))
        if atom_mask is not None:
            atom_mask = _expand_atom_axis(atom_mask, tuple(atom_hidden.shape[:-2]))

        return dispatch_atom_to_token_mean(
            atom_hidden,
            atom_to_token,
            num_tokens=num_tokens,
            atom_mask=atom_mask,
            eps=1.0,
            backend=self.backend,
        )


def dense_mean_to_tokens(
    atom_hidden: Tensor,
    atom_to_token: Tensor,
    num_tokens: int,
    atom_mask: Tensor | None = None,
    count_eps: float = 1e-6,
) -> Tensor:
    r"""
    Mean-pool atoms to tokens through a dense one-hot matrix.

    Forms a dense atom-to-token one-hot matrix, normalizes it by per-token counts plus
    ``count_eps``, and applies ``torch.bmm``. This path provides a deterministic dense
    reduction order distinct from scatter pooling.

    Args:
        atom_hidden: ``[..., N_atom, C]`` atom features.
        atom_to_token: ``[..., N_atom]`` token index for each atom.
        num_tokens: Number of output tokens.
        atom_mask: Optional ``[..., N_atom]`` valid-atom mask.
        count_eps: Additive denominator epsilon for empty and sparse tokens.

    Returns:
        ``[..., N_token, C]`` token features.
    """

    prefix = tuple(atom_hidden.shape[:-2])
    num_atoms, hidden_size = atom_hidden.shape[-2:]
    atom_to_token = _expand_atom_axis(atom_to_token, prefix)
    weights = F.one_hot(atom_to_token, num_classes=num_tokens).to(dtype=atom_hidden.dtype)
    if atom_mask is not None:
        atom_mask = _expand_atom_axis(atom_mask.to(dtype=atom_hidden.dtype), prefix)
        weights = weights * atom_mask.unsqueeze(-1)

    weights = weights / (weights.sum(dim=-2, keepdim=True) + count_eps)
    flat_weights = weights.reshape(-1, num_atoms, num_tokens)
    flat_hidden = atom_hidden.reshape(-1, num_atoms, hidden_size)
    pooled = torch.bmm(flat_weights.transpose(1, 2), flat_hidden)
    return pooled.reshape(*prefix, num_tokens, hidden_size)


class TokenToAtomBroadcast(nn.Module):
    r"""
    Scatter each token's hidden state back onto every atom that belongs to it.

    Inverse of the atom->token pooling: gathering along the ``atom_to_token`` index copies a token
    vector to all of its atoms, so a per-token signal can condition atom-level layers.

    Args:
        backend: Which backend to run; see [`backend`][ofoldx.modules.backend].
    """

    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")

    def forward(self, token_hidden: Tensor, atom_to_token: Tensor) -> Tensor:
        r"""
        Broadcast token features to atom features.

        Args:
            token_hidden: ``[..., N_token, C]`` token features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.

        Returns:
            ``[..., N_atom, C]`` atom features.
        """

        atom_to_token = _expand_atom_axis(atom_to_token, tuple(token_hidden.shape[:-2]))
        return dispatch_token_to_atom_broadcast(token_hidden, atom_to_token, backend=self.backend)


class TokenPairToLocalAtomPairBroadcast(nn.Module):
    r"""
    Index a token-pair map into local atom-pair windows for windowed atom attention.

    Emits, per query window, only the token pairs the local attention reads, keeping memory linear
    in the atom count. When every batch
    element shares one ``atom_to_token`` map, OOps dispatch may select its Triton gather when supported;
    distinct per-example maps are handled by the same backend API.

    Args:
        num_query_atoms: Number of query atoms per local window.
        num_key_atoms: Number of key atoms gathered around each query window.
        backend: Which backend to run; see [`backend`][ofoldx.modules.backend].
    """

    def __init__(self, num_query_atoms: int = 32, num_key_atoms: int = 128, backend: str = "auto") -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms)
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")

    def forward(self, token_pair: Tensor, atom_to_token: Tensor, key_index: Tensor | None = None) -> Tensor:
        r"""
        Broadcast token-pair features to local atom-pair windows.

        Args:
            token_pair: ``[..., N_token, N_token, C]`` token-pair features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.
            key_index: Optional ``[..., N_window, num_key_atoms]`` atom indices
                shared with geometry and attention gathering.

        Returns:
            ``[..., N_window, num_query_atoms, num_key_atoms, C]`` local atom-pair features.
        """

        atom_to_token = _expand_atom_axis(atom_to_token, tuple(token_pair.shape[:-3]))
        return dispatch_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            n_queries=self.windowing.num_query_atoms,
            n_keys=self.windowing.num_key_atoms,
            backend=self.backend,
            key_index=key_index,
        )


def _expand_atom_axis(tensor: Tensor, target_prefix: tuple[int, ...]) -> Tensor:
    r"""
    Broadcast an atom-axis tensor's leading dims up to ``target_prefix``, keeping the atom axis.

    Args:
        tensor: ``[..., N_atom]`` tensor whose leading prefix may be shorter than the target.
        target_prefix: The desired leading (batch/sample) shape before the atom axis.

    Returns:
        ``tensor`` broadcast to ``(*target_prefix, N_atom)``.
    """

    num_atoms = tensor.shape[-1]
    target_shape = (*target_prefix, num_atoms)
    if tuple(tensor.shape) == target_shape:
        return tensor

    source_prefix = tuple(tensor.shape[:-1])
    if len(source_prefix) <= len(target_prefix) and source_prefix == target_prefix[: len(source_prefix)]:
        view_shape = (*source_prefix, *((1,) * (len(target_prefix) - len(source_prefix))), num_atoms)
        tensor = tensor.reshape(view_shape)
    return torch.broadcast_to(tensor, target_shape)
