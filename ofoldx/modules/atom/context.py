r"""
Typed atom attention context objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(eq=False)
class AtomContext:
    r"""
    Precomputed atom-pair and per-atom conditioning tensors.

    Bundles the two atom-level tensors that are computed once per structure and reused across
    every atom attention block, so downstream layers receive them together instead of threading
    two loose tensors through each call.

    Args:
        pair: `[..., N_atom, N_atom, C]` (dense) or `[..., N_window, num_query_atoms, num_key_atoms, C]`
            (local) atom-pair bias / representation tensor.
        conditioning: `[..., N_atom, C]` per-atom conditioning tensor.
    """

    pair: Tensor
    conditioning: Tensor


@dataclass(eq=False)
class AtomToTokenOutput:
    r"""
    Atom encoder output consumed by token embedders and coordinate decoders.

    Carries the token-level hidden states produced by pooling atoms, together with the atom-level
    tensors a coordinate decoder later needs so it can reuse them without recomputing.

    Args:
        token_hidden: `[..., N_token, C]` token-level hidden states aggregated from atoms.
        atom_query: `[..., N_atom, C]` atom query hidden states.
        atom_context: Precomputed atom-pair and per-atom conditioning tensors.
    """

    token_hidden: Tensor
    atom_query: Tensor
    atom_context: AtomContext
