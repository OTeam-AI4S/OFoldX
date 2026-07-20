r"""Prediction heads shared by structure-model families."""

from __future__ import annotations

from .confidence import (
    AtomConfidenceProjector,
    ConfidenceHead,
    ConfidencePairEmbedder,
    PairConfidenceProjector,
)
from .distogram import DistogramHead
from .tm import tm_bin_weights, tm_length_scale, tm_weight_from_pae
