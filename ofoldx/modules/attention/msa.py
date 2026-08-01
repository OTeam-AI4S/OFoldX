# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
MSA row, column, and global-column attention updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..backend import LayerNorm
from .multihead import GatedMultiheadAttentionProjection, GlobalAttention, MultiheadAttention


class MsaRowAttentionWithPairBias(nn.Module):
    r"""
    Attend over residues within each MSA row using pair-derived head biases.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of row-attention heads.
        backend: Backend used by gated attention.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        num_attention_heads: int,
        backend: str,
    ) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(
                msa_size,
                num_attention_heads=num_attention_heads,
                query_bias=False,
                output_bias=True,
                gate_bias=True,
                gate_bias_init=1.0,
            ),
            backend=backend,
        )

    def forward(self, msa: Tensor, pair: Tensor, *, msa_mask: Tensor) -> Tensor:
        r"""
        Return an additive MSA row-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            pair: Pair representation with shape ``(..., N_residue, N_residue, C_pair)``.
            msa_mask: MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Row-attention update with the same shape as ``msa``.
        """

        msa = self.query_norm(msa)
        pair = self.pair_norm(pair)
        pair_bias = self.pair_bias_projection(pair).movedim(-1, -3).unsqueeze(-4)
        return self.attention(msa, biases=(pair_bias,), attention_mask=msa_mask.unsqueeze(-2).bool())


class MsaColumnAttention(nn.Module):
    r"""
    Attend over MSA rows independently at each residue position.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of column-attention heads.
        head_size: Per-head width; defaults to ``msa_size // num_attention_heads``.
        backend: Backend used by gated attention.
    """

    def __init__(
        self, *, msa_size: int, num_attention_heads: int, head_size: int | None = None, backend: str = "auto"
    ) -> None:
        super().__init__()
        self.query_norm = LayerNorm(msa_size, backend=backend)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(
                msa_size,
                num_attention_heads=num_attention_heads,
                head_size=head_size,
                query_bias=False,
                output_bias=True,
                gate_bias=True,
                gate_bias_init=1.0,
            ),
            backend=backend,
        )

    def forward(self, msa: Tensor, *, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Return an additive MSA column-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            msa_mask: Optional MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Column-attention update with the same shape as ``msa``.
        """

        msa = msa.transpose(-3, -2)
        attention_mask = None if msa_mask is None else msa_mask.transpose(-2, -1).unsqueeze(-2).bool()
        output = self.attention(self.query_norm(msa), attention_mask=attention_mask)
        return output.transpose(-3, -2)


class MsaColumnGlobalAttention(nn.Module):
    r"""
    Attend over MSA rows from one mask-weighted global query per residue.

    Args:
        msa_size: Width of the MSA representation.
        num_attention_heads: Number of global-attention heads.
        head_size: Per-head width; defaults to ``msa_size // num_attention_heads``.
        backend: Backend used by query normalization and global attention.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        num_attention_heads: int,
        head_size: int | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if head_size is None:
            head_size = msa_size // num_attention_heads
        self.query_norm = LayerNorm(msa_size, backend=backend)
        self.attention = GlobalAttention(
            hidden_size=msa_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            backend=backend,
        )

    def forward(self, msa: Tensor, *, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Return an additive global column-attention update.

        Args:
            msa: MSA representation with shape ``(..., N_msa, N_residue, C_msa)``.
            msa_mask: Optional MSA validity mask with shape ``(..., N_msa, N_residue)``.

        Returns:
            Global column-attention update with the same shape as ``msa``.
        """

        msa = msa.transpose(-3, -2)
        attention_mask = None if msa_mask is None else msa_mask.transpose(-2, -1).unsqueeze(-1)
        if attention_mask is not None:
            keep = (
                attention_mask
                if attention_mask.dtype == torch.bool
                else ((attention_mask != 0) | torch.isfinite(msa).all(-1, keepdim=True))
            )
            msa = torch.where(keep, msa, 0)
        output = self.attention(self.query_norm(msa), attention_mask)
        return output.transpose(-3, -2)
