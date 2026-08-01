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
Atom/token transfer modules.
"""

from __future__ import annotations

from danling.tensors import NestedTensor
from torch import Tensor, nn

from ..backend import atom_to_token_mean as dispatch_atom_to_token_mean
from ..backend import resolve_config_backend
from ..backend import token_pair_to_atom_window_gather as dispatch_token_pair_to_atom_window_gather
from ..backend import token_to_atom_broadcast as dispatch_token_to_atom_broadcast
from .window import LocalAtomWindowInfo


class AtomToTokenMean(nn.Module):
    r"""
    Mean-pool per-atom features to tokens by scattering on the atom->token index.

    Each token receives the exact mean of its valid atoms; tokens without a valid atom remain zero.
    An optional ``atom_mask`` drops invalid atoms from the sum and the count.

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
            ``[..., N_token, C]`` token features; tokens with no valid atom stay zero.
        """

        return dispatch_atom_to_token_mean(
            atom_hidden,
            atom_to_token,
            token_reference=token_reference,
            atom_mask=atom_mask,
            backend=self.backend,
        )


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
        backend: Which backend to run; see [`backend`][ofoldx.modules.backend].
    """

    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")

    def forward(
        self,
        token_pair: Tensor | NestedTensor,
        atom_to_token: Tensor | NestedTensor,
        window_info: LocalAtomWindowInfo,
    ) -> Tensor | NestedTensor:
        r"""
        Broadcast token-pair features to local atom-pair windows.

        Args:
            token_pair: ``[..., N_token, N_token, C]`` token-pair features.
            atom_to_token: ``[..., N_atom]`` token index for each atom.
            window_info: The local window partition of the atom axis.

        Returns:
            ``[..., N_window, num_query_atoms, num_key_atoms, C]`` local atom-pair features.
        """

        return dispatch_token_pair_to_atom_window_gather(
            token_pair,
            atom_to_token,
            num_query_atoms=window_info.num_query_atoms,
            num_key_atoms=window_info.num_key_atoms,
            backend=self.backend,
            key_index=window_info.key_index,
            packed_layout=window_info.packed_layout,
        )
