r"""
Attention architecture blocks shared across the trunk, atom, and diffusion stacks.

Re-exports the multi-head attention primitives ([`GatedMultiheadAttention`]
[ofoldx.modules.attention.multihead.GatedMultiheadAttention],
[`MultiheadAttentionProjection`][ofoldx.modules.attention.multihead.MultiheadAttentionProjection])
and the higher-level blocks built on them: pair-biased self-attention, the denoising token encoder,
and the triangle attention / multiplication updates.
"""

from __future__ import annotations

from .denoising import DenoisingTokenEncoder
from .encoder import PairBiasedSelfAttention
from .multihead import GatedMultiheadAttention, MultiheadAttentionProjection
from .triangle import TriangleAttention, TriangleMultiplication

__all__ = [
    "DenoisingTokenEncoder",
    "GatedMultiheadAttention",
    "MultiheadAttentionProjection",
    "PairBiasedSelfAttention",
    "TriangleAttention",
    "TriangleMultiplication",
]
