r"""
Typed atom attention context objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(eq=False)
class AtomToTokenOutput:
    r"""
    Atom encoder output consumed by token embedders and coordinate decoders.

    Carries the token-level hidden states produced by pooling atoms, together with the atom-level
    tensors a coordinate decoder later needs so it can reuse them without recomputing.

    Attributes:
        token_hidden: `[..., N_token, C]` token-level hidden states aggregated from atoms.
        atom_query: `[..., N_atom, C]` atom query hidden states.
        atom_pair: Precomputed dense or local atom-pair states.
        atom_conditioning: Per-atom conditioning states.
    """

    token_hidden: Tensor
    atom_query: Tensor
    atom_pair: Tensor
    atom_conditioning: Tensor
