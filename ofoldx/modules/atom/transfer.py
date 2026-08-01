r"""
Atom/token transfer modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import atom_to_token_mean as dispatch_atom_to_token_mean
from ..backend import resolve_config_backend
from ..backend import token_pair_to_atom_window_gather as dispatch_token_pair_to_atom_window_gather
from ..backend import token_to_atom_broadcast as dispatch_token_to_atom_broadcast
from .window import LocalAtomWindowing


class AtomToTokenMean(nn.Module):
    r"""
    Mean-pool per-atom features to tokens by scattering on the atom->token index.

    Non-empty token counts receive ``denominator_offset``; empty tokens remain zero.
    An optional ``atom_mask`` zeroes invalid atoms before aggregation.

    Args:
        backend: Which backend to run; see [`backend`][ofoldx.modules.backend].
        denominator_offset: Additive offset for non-empty token counts. ``None``
            uses the smallest positive value representable by the input dtype.
    """

    def __init__(self, *, denominator_offset: float | None, backend: str = "auto") -> None:
        super().__init__()
        self.denominator_offset = denominator_offset
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "eager"),
            public_default="eager",
        )

    def forward(
        self,
        atom_hidden: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
        atom_mask: Tensor | NestedTensor | None = None,
    ) -> Tensor | NestedTensor:
        r"""
        Aggregate atom features onto their tokens.

        Args:
            atom_hidden: ``[..., N_atom, C]`` atom features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.
            token_reference: Token-axis tensor whose topology defines the output.
                Its values are not read.
            atom_mask: Optional ``[..., N_atom]`` valid-atom mask.

        Returns:
            ``[..., N_token, C]`` token features; tokens with no atoms stay zero.
        """

        return dispatch_atom_to_token_mean(
            atom_hidden,
            atom_to_token,
            token_reference=token_reference,
            atom_mask=atom_mask,
            denominator_offset=self.denominator_offset,
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

    def forward(
        self,
        token_hidden: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
    ) -> Tensor | NestedTensor:
        r"""
        Broadcast token features to atom features.

        Args:
            token_hidden: ``[..., N_token, C]`` token features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.

        Returns:
            ``[..., N_atom, C]`` atom features.
        """

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

    def forward(
        self,
        token_pair: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        key_index: Tensor | NestedTensor | None = None,
        valid_extent: Tensor | None = None,
    ) -> Tensor | NestedTensor:
        r"""
        Broadcast token-pair features to local atom-pair windows.

        Args:
            token_pair: ``[..., N_token, N_token, C]`` token-pair features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.
            key_index: Optional ``[..., N_window, num_key_atoms]`` atom indices
                shared with geometry and attention gathering.
            valid_extent: Optional compact per-sample valid-atom extents used to
                reconstruct row-specific NestedTensor key windows on OOps. CUDA
                tensors remain device-resident through compiled execution.

        Returns:
            ``[..., N_window, num_query_atoms, num_key_atoms, C]`` local atom-pair features.
        """

        return dispatch_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            n_queries=self.windowing.num_query_atoms,
            n_keys=self.windowing.num_key_atoms,
            backend=self.backend,
            key_index=key_index,
            valid_extent=valid_extent,
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
