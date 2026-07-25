r"""
Reusable prediction heads for structure-model implementations.
"""

from __future__ import annotations

from .confidence import AtomConfidenceProjector, expected_plddt
from .distogram import DistogramHead, PairSymmetrizedDistogramHead
from .tm import tm_bin_weights, tm_length_scale, tm_weight_from_pae
