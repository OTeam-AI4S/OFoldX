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
MSA encoding, joint MSA/pair updates, and outer-product projections.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, is_grad_enabled, nn

from ..compilation import HomogeneousBlockRegions
from .backend import LayerNorm
from .backend import categorical_msa_projection as dispatch_categorical_msa_projection
from .backend import outer_product_mean as dispatch_outer_product_mean
from .backend import pair_weighted_average as dispatch_pair_weighted_average
from .backend import resolve_config_backend
from .dropout import StructuredDropout
from .functional import pair_mask_from_msa_mask, project_categories
from .pair import LineDropoutTrianglePairUpdateBlock, TrianglePairUpdateBlock
from .transition import Transition


class MsaEncoder(HomogeneousBlockRegions, nn.Module):
    r"""
    MSA/pair encoder with homogeneous joint blocks and one final pair-only update.

    The encoder linearly projects the raw MSA features and an optional per-token embedding to width
    ``msa_size``, runs ``num_stages - 1`` homogeneous joint update blocks, then applies a
    [`FinalMsaPairUpdate`][ofoldx.modules.msa.FinalMsaPairUpdate]. ``num_stages=0`` skips both
    stages, while ``num_stages=1`` runs only the final pair update.

    Args:
        msa_size: Width of the projected MSA representation.
        pair_size: Width of the pair representation.
        msa_vocab_size: Number of categorical MSA tokens.
        token_feature_size: Width of the optional per-token embedding.
        num_stages: Total number of MSA/pair stages, including the final pair-only update.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        msa_attention_head_size: Per-head value width in the MSA pair-weighted averaging. ``None`` derives it from
            ``msa_size`` and ``num_msa_attention_heads``.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of every transition.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        msa_vocab_size: int = 32,
        token_feature_size: int = 449,
        num_stages: int = 4,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int | None = None,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(msa_vocab_size + 2, msa_size, bias=False)
        self.backend = resolve_config_backend(
            backend,
            supported=("oops", "eager"),
            public_default="eager",
        )
        self.token_projection = nn.Linear(token_feature_size, msa_size, bias=False)
        self.num_stages = num_stages
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            OuterProductFirstMsaPairBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_attention_head_size=msa_attention_head_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            for _ in range(num_stages - 1)
        )
        self.final_pair_update = (
            FinalMsaPairUpdate(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_head_size=triangle_attention_head_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                pair_dropout=pair_dropout,
                backend=backend,
            )
            if num_stages
            else None
        )

    _gradient_checkpointing_func: Callable[..., Any]

    def forward(
        self,
        msa_tokens: Tensor,
        msa_continuous_features: Tensor,
        pair: Tensor,
        token_embedding: Tensor | None = None,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Project the MSA features and optional token embedding, then run the block stack.

        The MSA features are projected to width ``msa_size``; when ``token_embedding`` is given its
        projection is broadcast over the sequence axis and added. When ``pair_mask`` is omitted but
        ``msa_mask`` is present, the pair mask is derived from the MSA mask.

        Args:
            msa_tokens: Compact MSA token indices `[B, S, N]`.
            msa_continuous_features: Continuous MSA features `[B, S, N, 2]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            token_embedding: Optional per-token embedding `[B, N, token_feature_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`; derived from ``msa_mask`` if omitted.

        Returns:
            msa: Projected-and-refined MSA `[B, S, N, msa_size]`. The final update
                leaves the MSA untouched.
            pair: Refined pair `[B, N, N, pair_size]`.
        """

        available = torch.ones_like(msa_tokens, dtype=torch.bool) if msa_mask is None else msa_mask.bool()
        msa_tokens = project_categories(msa_tokens, available, num_classes=self.msa_projection.in_features - 2)
        msa_continuous_features = torch.where(available[..., None], msa_continuous_features, 0)
        msa = dispatch_categorical_msa_projection(
            msa_tokens,
            msa_continuous_features,
            self.msa_projection.weight,
            backend=self.backend,
        )
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask, backend=self.backend)
        checkpoint = (
            self._gradient_checkpointing_func
            if self.gradient_checkpointing and self.training and is_grad_enabled()
            else None
        )
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask, checkpoint=checkpoint)
        if self.final_pair_update is not None:
            pair = self.final_pair_update(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask, checkpoint=checkpoint)
        return msa, pair


class MsaPairBlock(nn.Module):
    r"""
    Update the MSA, then refine its pair representation with line-wise dropout.

    Pair-weighted averaging and a transition update the MSA before its
    pre-projection-masked outer-product mean updates the pair. A line-dropout
    actual pair-update block then refines the pair. The MSA residual mask is shared over
    residue and feature axes within each MSA row.

    ``pair_block`` receives ``pair_size``, ``expansion``, ``dropout`` and
    ``backend``; its other architectural arguments are bound by the owner.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        expansion: int = 4,
        backend: str = "auto",
        *,
        pair_block: Callable[..., nn.Module] = LineDropoutTrianglePairUpdateBlock,
    ) -> None:
        super().__init__()
        self.pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_size=msa_attention_head_size,
            backend=backend,
        )
        self.msa_dropout = StructuredDropout(msa_dropout, broadcast_dims=(-2, -1))
        self.msa_transition = Transition(msa_size, expansion=expansion, backend=backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = pair_block(
            pair_size=pair_size,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Return updated ``(msa, pair)`` with the input axes preserved."""

        msa = msa + self.msa_dropout(self.pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        msa = msa + self.msa_transition(msa)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return msa, pair


class OuterProductFirstMsaPairBlock(nn.Module):
    r"""
    Joint MSA and pair update block: outer-product mean, MSA update, then triangle pair update.

    Adds the outer-product mean of the MSA to the pair representation, updates the MSA against the new
    pair via an [`MsaUpdateBlock`][ofoldx.modules.msa.MsaUpdateBlock], then refines the pair with
    a [`TrianglePairUpdateBlock`][ofoldx.modules.pair.TrianglePairUpdateBlock].

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        msa_attention_head_size: Per-head value width in the MSA pair-weighted averaging. ``None`` derives it from
            ``msa_size`` and ``num_msa_attention_heads``.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Feed-forward expansion factor of every transition.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int | None = None,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.msa_update = MsaUpdateBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_size=msa_attention_head_size,
            expansion=expansion,
            dropout=msa_dropout,
            backend=backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
        checkpoint: Callable[..., Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Add the outer-product mean of the MSA to the pair, update the MSA against the new pair, then run
        the triangle pair update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.
            checkpoint: Optional checkpoint callable supplied by the containing
                encoder. Each update branch owns its recomputation boundary.

        Returns:
            Updated ``(msa, pair)`` of shapes `[B, S, N, msa_size]` and `[B, N, N, pair_size]`.
        """

        if checkpoint is None:
            pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
            msa = self.msa_update(msa, pair, pair_mask=pair_mask)
            return msa, self.triangle_pair_update(pair, pair_mask=pair_mask)

        pair = pair + checkpoint(self.outer_product_mean, msa, msa_mask=msa_mask)
        msa = self.msa_update(msa, pair, pair_mask=pair_mask, checkpoint=checkpoint)
        pair = checkpoint(self.triangle_pair_update, pair, pair_mask=pair_mask)
        return msa, pair


class FinalMsaPairUpdate(nn.Module):
    r"""
    Final MSA-stage update that refines only the pair representation.

    Adds the outer-product mean of the MSA to the pair representation and runs the triangle pair update.
    The MSA passes through untouched, since the encoder's final stage emits no MSA update.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_head_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of every transition.
        pair_dropout: Row-wise dropout probability on each pair residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        pair_dropout: float = 0.25,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = PostProjectionOuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            backend=backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
        checkpoint: Callable[..., Any] | None = None,
    ) -> Tensor:
        r"""
        Add the outer-product mean of the MSA to the pair and run the triangle pair update.

        The MSA passes through unchanged, since the encoder's final stage emits no MSA update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.
            checkpoint: Optional checkpoint callable supplied by the containing
                encoder, applied separately to the outer-product and triangle updates.

        Returns:
            The refined pair representation `[B, N, N, pair_size]`.
        """

        if checkpoint is None:
            pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
            return self.triangle_pair_update(pair, pair_mask=pair_mask)
        pair = pair + checkpoint(self.outer_product_mean, msa, msa_mask=msa_mask)
        return checkpoint(self.triangle_pair_update, pair, pair_mask=pair_mask)


class MsaUpdateBlock(nn.Module):
    r"""
    MSA update block: pair-weighted averaging residual followed by a transition residual.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of heads in the MSA pair-weighted averaging.
        head_size: Per-head value width. ``None`` derives it from ``msa_size`` and ``num_attention_heads``.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability on the pair-weighted averaging residual.
        backend: Backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        head_size: int | None = None,
        expansion: int = 4,
        dropout: float = 0.15,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            backend=backend,
        )
        self.rowwise_dropout: nn.Module = StructuredDropout(dropout, broadcast_dims=-3)
        self.transition = Transition(msa_size, expansion=expansion, backend=backend)

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        pair_mask: Tensor | None = None,
        *,
        checkpoint: Callable[..., Any] | None = None,
    ) -> Tensor:
        r"""
        Update the MSA with a row-dropout pair-weighted averaging residual, then a transition residual.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.
            checkpoint: Optional checkpoint callable supplied by the containing
                encoder. Dropout is recomputed with the weighted-average branch.

        Returns:
            The updated MSA representation `[B, S, N, msa_size]`.
        """

        if checkpoint is None:
            msa = msa + self._weighted_update(msa, pair, pair_mask)
            return msa + self.transition(msa)
        msa = msa + checkpoint(self._weighted_update, msa, pair, pair_mask)
        return msa + checkpoint(self.transition, msa)

    def _weighted_update(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None) -> Tensor:
        return self.rowwise_dropout(self.msa_pair_weighted_averaging(msa, pair, pair_mask=pair_mask))


class MsaPairWeightedAveraging(nn.Module):
    r"""
    Pair-conditioned averaging for MSA representations.

    Averages MSA values along the query-residue axis with per-head weights softmaxed from the pair
    representation, then sigmoid-gates and projects back. The inner width is
    ``num_attention_heads * head_size``; ``head_size`` defaults to ``msa_size // num_attention_heads``, and a
    fixed larger ``head_size`` widens the value/gate/output projections.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation used to compute the per-head weights.
        num_attention_heads: Number of averaging heads.
        head_size: Per-head width; defaults to ``msa_size // num_attention_heads`` when ``None``.
        backend: Backend selector; resolves to ``"oops"`` or ``"eager"``.

    Raises:
        ValueError: If ``head_size`` is ``None`` and ``msa_size`` is not divisible by
            ``num_attention_heads``.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        head_size: int | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        if head_size is None:
            if msa_size % num_attention_heads != 0:
                raise ValueError(
                    f"msa_size ({msa_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                )
            head_size = msa_size // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        inner_size = num_attention_heads * head_size
        self.msa_norm = LayerNorm(msa_size, backend=backend)
        self.value_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.gate_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, msa_size, bias=False)

    def forward(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Average MSA values along the query-residue axis with per-head weights softmaxed from the pair
        representation, then sigmoid-gate and project back.

        The ``oops`` backend delegates to the accelerated implementation; the ``eager`` path uses
        ordinary PyTorch operations.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]` masking invalid keys.

        Returns:
            The MSA update `[B, S, N, msa_size]`.
        """

        return dispatch_pair_weighted_average(
            msa,
            pair,
            pair_mask,
            msa_norm_weight=self.msa_norm.weight,
            msa_norm_bias=self.msa_norm.bias,
            pair_norm_weight=self.pair_norm.weight,
            pair_norm_bias=self.pair_norm.bias,
            value_weight=self.value_projection.weight,
            gate_weight=self.gate_projection.weight,
            pair_logits_weight=self.pair_projection.weight,
            output_weight=self.output_projection.weight,
            num_attention_heads=self.num_attention_heads,
            head_size=self.head_size,
            msa_norm_eps=self.msa_norm.eps,
            pair_norm_eps=self.pair_norm.eps,
            backend=self.backend,
        )


class OuterProductMean(nn.Module):
    r"""
    Outer-product mean projection from MSA representations to pair representations.

    Forms the mask-weighted mean over sequence rows of the outer product of the left/right MSA projections,
    then applies the affine output projection. The valid-row count is clamped to one before the projection,
    so the output bias is added once and is never divided by the count.

    Args:
        msa_size: Width of the MSA representation.
        hidden_size: Hidden width of each of the left/right MSA projections.
        pair_size: Width of the produced pair update.
        backend: Backend selector; resolves to ``"oops"`` or ``"eager"``.
    """

    def __init__(
        self,
        msa_size: int = 64,
        hidden_size: int = 32,
        pair_size: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.msa_norm = LayerNorm(msa_size, backend=backend)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
        projections and project it to a pair update.

        The ``oops`` backend delegates to the accelerated implementation; ``eager`` uses ordinary PyTorch
        operations.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]` zeroing invalid rows and counting valid pairs.

        Returns:
            The pair contribution `[B, N, N, pair_size]`.
        """

        return dispatch_outer_product_mean(
            msa,
            msa_mask,
            msa_norm_weight=self.msa_norm.weight,
            msa_norm_bias=self.msa_norm.bias,
            left_weight=self.left_projection.weight,
            right_weight=self.right_projection.weight,
            left_bias=self.left_projection.bias,
            right_bias=self.right_projection.bias,
            output_weight=self.output_projection.weight,
            output_bias=self.output_projection.bias,
            msa_norm_eps=self.msa_norm.eps,
            denominator_eps=0.0,
            divide_before_projection=True,
            clamp_denominator=False,
            backend=self.backend,
        )


class PostProjectionOuterProductMean(nn.Module):
    r"""
    Project an MSA outer-product sum, then normalize the projected pair update.

    The affine output projection is applied to the masked outer-product sum before division by
    ``count + eps``, optionally clamped to at least one. Consequently its output bias is also divided
    by the configured count denominator. This is the
    exact post-projection-normalization equation used by AlphaFold 3-derived MSA trunks and Evoformer.

    Args:
        msa_size: Width of the MSA representation.
        hidden_size: Hidden width of each of the left/right MSA projections.
        pair_size: Width of the produced pair update.
        eps: Epsilon added to the per-pair valid-row count after projection.
        backend: Backend selector; resolves to ``"oops"`` or ``"eager"``.
        projection_bias: Whether the left and right MSA projections include biases.
        clamp_denominator: Whether to clamp the count-plus-epsilon denominator to at least one.
    """

    def __init__(
        self,
        msa_size: int = 64,
        hidden_size: int = 32,
        pair_size: int = 128,
        eps: float = 1e-3,
        backend: str = "auto",
        *,
        projection_bias: bool = False,
        clamp_denominator: bool = False,
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.eps = eps
        self.clamp_denominator = clamp_denominator
        self.msa_norm = LayerNorm(msa_size, backend=backend)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=projection_bias)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=projection_bias)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Project the masked outer-product sum and divide by the configured count denominator.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]` zeroing invalid rows and counting valid pairs.

        Returns:
            The pair contribution `[B, N, N, pair_size]`.
        """

        return dispatch_outer_product_mean(
            msa,
            msa_mask,
            msa_norm_weight=self.msa_norm.weight,
            msa_norm_bias=self.msa_norm.bias,
            left_weight=self.left_projection.weight,
            right_weight=self.right_projection.weight,
            left_bias=self.left_projection.bias,
            right_bias=self.right_projection.bias,
            output_weight=self.output_projection.weight,
            output_bias=self.output_projection.bias,
            msa_norm_eps=self.msa_norm.eps,
            denominator_eps=self.eps,
            divide_before_projection=False,
            clamp_denominator=self.clamp_denominator,
            backend=self.backend,
        )
