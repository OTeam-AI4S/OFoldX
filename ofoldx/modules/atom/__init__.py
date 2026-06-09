r"""
Shared atom-level building blocks (windowing, pair encoding, transfer ops, contexts).
"""

from __future__ import annotations

from .context import AtomContext, AtomToTokenOutput
from .encoders import (
    AtomConditioner,
    AtomPairEncoder,
    AtomSelfAttentionStack,
    AtomToTokenEncoder,
    AtomToTokenEncoderBase,
    AttentionLayout,
    CoordinateAtomToTokenEncoder,
    PackedAtomAttention,
    TokenToAtomDecoder,
)
from .pair_encoding import encode_atom_pair
from .transfer import (
    TokenPairToAtomPairBroadcast,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from .window import (
    LocalAtomGeometry,
    LocalAtomGeometryBuilder,
    LocalAtomWindowInfo,
    LocalAtomWindowing,
    LocalAtomWindows,
)

__all__ = [
    "AtomConditioner",
    "AtomContext",
    "AtomPairEncoder",
    "AtomSelfAttentionStack",
    "AtomToTokenEncoder",
    "AtomToTokenEncoderBase",
    "AtomToTokenOutput",
    "AttentionLayout",
    "CoordinateAtomToTokenEncoder",
    "LocalAtomGeometry",
    "LocalAtomGeometryBuilder",
    "LocalAtomWindowInfo",
    "LocalAtomWindowing",
    "LocalAtomWindows",
    "PackedAtomAttention",
    "TokenPairToAtomPairBroadcast",
    "TokenPairToLocalAtomPairBroadcast",
    "TokenToAtomBroadcast",
    "TokenToAtomDecoder",
    "encode_atom_pair",
]
