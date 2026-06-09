r"""
Typed atom attention context objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class AtomContext:
    r"""
    Precomputed atom-pair and per-atom conditioning tensors.

    Bundles the two atom-level tensors that are computed once per structure and reused across
    every atom attention block, so downstream layers receive them together instead of threading
    two loose tensors through each call.

    Attributes:
        pair: `[..., N_atom, N_atom, C]` (dense) or `[..., N_window, num_query_atoms, num_key_atoms, C]`
            (local) atom-pair bias / representation tensor.
        conditioning: `[..., N_atom, C]` per-atom conditioning tensor.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.context import AtomContext
        >>> ctx = AtomContext(pair=torch.zeros(2, 2, 4), conditioning=torch.zeros(2, 4))
        >>> ctx.pair.shape, ctx.conditioning.shape
        (torch.Size([2, 2, 4]), torch.Size([2, 4]))
    """

    pair: Tensor
    conditioning: Tensor


@dataclass(slots=True)
class AtomToTokenOutput:
    r"""
    Atom encoder output consumed by token embedders and coordinate decoders.

    Carries the token-level hidden states produced by pooling atoms, together with the atom-level
    tensors a coordinate decoder later needs so it can reuse them without recomputing.

    Attributes:
        token_hidden: `[..., N_token, C]` token-level hidden states aggregated from atoms.
        atom_query: `[..., N_atom, C]` atom query hidden states.
        atom_context: Precomputed atom-pair and per-atom conditioning tensors.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.atom.context import AtomContext, AtomToTokenOutput
        >>> ctx = AtomContext(pair=torch.zeros(2, 2, 4), conditioning=torch.zeros(2, 4))
        >>> out = AtomToTokenOutput(torch.zeros(3, 8), torch.zeros(2, 8), ctx)
        >>> out.token_hidden.shape, out.atom_query.shape
        (torch.Size([3, 8]), torch.Size([2, 8]))
        >>> out.atom_context.pair.shape
        torch.Size([2, 2, 4])
    """

    token_hidden: Tensor
    atom_query: Tensor
    atom_context: AtomContext
