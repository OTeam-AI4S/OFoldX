r"""
Reusable atom-level encoding and transfer components.
"""

from __future__ import annotations

from .context import AtomContext, AtomToTokenOutput
from .encoders import (
    AtomConditioner,
    AtomPairEncoder,
    AtomSelfAttention,
    AtomToTokenEncoder,
    AtomToTokenEncoderBase,
    AttentionLayout,
    CoordinateAtomToTokenEncoder,
    NormalizedKeyAtomToTokenEncoder,
    NormalizedKeyCoordinateAtomToTokenEncoder,
    NormalizedKeyTokenToAtomDecoder,
    PackedAtomAttention,
    PackedAtomToTokenEncoder,
    PackedDenoisingAtomToTokenEncoder,
    PackedReferenceAtomEncoder,
    PackedTokenToAtomDecoder,
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
