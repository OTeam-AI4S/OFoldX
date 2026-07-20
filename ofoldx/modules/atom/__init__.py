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
