r"""
Per-architecture decoders for the Tier-0 result objects.

A neutral registry that lets a family teach [`DesignResult`][ofoldx.results.DesignResult] /
[`StructurePredictionResult`][ofoldx.results.StructurePredictionResult] how to decode *its own* output
(e.g. a token → sequence decoder)
**without** [`ofoldx.results`][ofoldx.results] importing any model family. Each family self-registers at
import time under its architecture key -- the same key it stamps into ``metadata["architecture"]`` -- and the
result objects look the callable up by that key. Families depend on this registry; the registry depends on
nothing; ``results.py`` depends only on the registry. The waterline holds.

Examples:
    >>> from ofoldx.results_registry import register_sequence_decoder, sequence_decoder
    >>> register_sequence_decoder("demo", lambda tokens: "".join("ACDE"[int(t)] for t in tokens))
    >>> sequence_decoder("demo")([0, 2, 3])
    'ADE'
    >>> sequence_decoder("unregistered") is None
    True
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

SequenceDecoder = Callable[[Iterable[int]], str]

_SEQUENCE_DECODERS: dict[str, SequenceDecoder] = {}


def register_sequence_decoder(architecture: str, decoder: SequenceDecoder) -> None:
    r"""
    Register a token → sequence decoder for an architecture key.

    Args:
        architecture: The architecture key a family stamps into ``metadata["architecture"]``.
        decoder: Callable turning an iterable of token ids into a residue-letter string.
    """

    _SEQUENCE_DECODERS[architecture] = decoder


def sequence_decoder(architecture: str | None) -> SequenceDecoder | None:
    r"""
    Return the decoder registered for ``architecture``, or ``None`` if unset.

    Args:
        architecture: The architecture key, or ``None``.

    Returns:
        The registered decoder, or ``None``.
    """

    if architecture is None:
        return None
    return _SEQUENCE_DECODERS.get(architecture)


def _decode(architecture: str | None, tokens: Any) -> str | None:
    r"""Decode a 1-D ``tokens`` tensor/sequence for ``architecture``, or ``None`` when no decoder is registered."""

    decoder = sequence_decoder(architecture)
    if decoder is None:
        return None
    row = tokens[0] if getattr(tokens, "ndim", 1) >= 2 else tokens
    return decoder(row.tolist() if hasattr(row, "tolist") else list(row))
