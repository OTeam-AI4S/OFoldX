r"""
Multiple-sequence-alignment trunk blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attention import TriangleAttention, TriangleMultiplication
from ..backend import LayerNorm
from ..backend import outer_product_mean as dispatch_outer_product_mean
from ..backend import pair_weighted_average as dispatch_pair_weighted_average
from ..backend import resolve_config_operator_backend
from ..dropout import RowwiseDropout
from ..functional import pair_mask_from_msa_mask, pair_mask_from_token_mask
from ..transition import Transition


class MsaTrunk(nn.Module):
    r"""
    Stack of MSA and pair update blocks with the final block emitting only a pair update.

    The trunk linearly projects the raw MSA features and an optional per-token embedding to width
    ``msa_size``, then runs ``num_blocks`` joint update blocks. The last block is swapped for a
    [`FinalMsaPairBlock`][ofoldx.modules.trunk.msa.FinalMsaPairBlock], which refines the pair
    representation without touching the MSA.

    Args:
        msa_size: Width of the projected MSA representation.
        pair_size: Width of the pair representation.
        msa_feature_size: Width of the raw per-residue MSA feature vector.
        token_feature_size: Width of the optional per-token embedding.
        num_blocks: Number of blocks in the stack (the last is a final pair-only block).
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of every transition.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        msa_feature_size: int = 34,
        token_feature_size: int = 449,
        num_blocks: int = 4,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.token_projection = nn.Linear(token_feature_size, msa_size, bias=False)
        blocks: list[nn.Module] = [
            MsaPairBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                num_msa_attention_heads=num_msa_attention_heads,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks - 1)
        ]
        if num_blocks:
            blocks.append(
                FinalMsaPairBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    expansion=expansion,
                    pair_dropout=pair_dropout,
                    operator_backend=operator_backend,
                )
            )
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        msa: Tensor,
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
            msa: Raw MSA features `[B, S, N, msa_feature_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            token_embedding: Optional per-token embedding `[B, N, token_feature_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`; derived from ``msa_mask`` if omitted.

        Returns:
            Tuple of the projected-and-refined MSA `[B, S, N, msa_size]` and the refined pair
            `[B, N, N, pair_size]`. The final block leaves the MSA untouched.
        """

        msa = self.msa_projection(msa)
        if token_embedding is not None:
            msa = msa + self.token_projection(token_embedding).unsqueeze(-3)
        if pair_mask is None and msa_mask is not None:
            pair_mask = pair_mask_from_msa_mask(msa_mask)
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return msa, pair


class CategoricalMsaTrunk(nn.Module):
    r"""
    MSA trunk that builds the MSA state from raw features and refines the pair representation.

    Unlike [`MsaTrunk`][ofoldx.modules.trunk.msa.MsaTrunk], this trunk builds the MSA state inline
    from integer token ids and deletion features, adds the projected single input embedding, and returns
    only the refined pair representation. The one-hot vocabulary plus the deletion and paired features must
    match ``msa_feature_size`` (``vocab_size + 3`` when ``use_paired_feature`` else ``vocab_size + 2``).

    Args:
        single_size: Width of the single input embedding.
        pair_size: Width of the pair representation.
        msa_size: Width of the projected MSA state.
        msa_feature_size: Expected width of the concatenated MSA feature vector.
        num_blocks: Number of [`MsaBlock`][ofoldx.modules.trunk.msa.MsaBlock] blocks.
        vocab_size: Token vocabulary size used to one-hot the MSA ids.
        use_paired_feature: Whether to append the ``is_paired`` feature.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        msa_head_dim: Per-head width of the MSA pair-weighted averaging.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Feed-forward expansion factor of every transition.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        msa_feature_size: int = 36,
        num_blocks: int = 4,
        vocab_size: int = 33,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.use_paired_feature = use_paired_feature
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            MsaBlock(
                msa_size=msa_size,
                pair_size=pair_size,
                outer_product_hidden_size=outer_product_hidden_size,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                num_msa_attention_heads=num_msa_attention_heads,
                msa_head_dim=msa_head_dim,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        *,
        msa_tokens: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        r"""
        Build the MSA state from raw features, then refine the pair representation through every block.

        Concatenates the one-hot MSA token ids (``vocab_size`` classes) with ``has_deletion`` and
        ``deletion_value`` (and ``is_paired`` when ``use_paired_feature``), projects the result to width
        ``msa_size``, and adds the projected single inputs broadcast over the sequence axis. The pair mask
        is derived from ``token_mask``.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            single_inputs: Per-token single input embedding `[B, N, single_size]`.
            msa_tokens: Integer MSA token ids `[B, S, N]`.
            has_deletion: Binary deletion-presence feature `[B, S, N]`.
            deletion_value: Continuous deletion-count feature `[B, S, N]`.
            is_paired: Binary paired-sequence feature `[B, S, N]`; used only when
                ``use_paired_feature`` is set.
            msa_mask: MSA validity mask `[B, S, N]`.
            token_mask: Per-token validity mask `[B, N]`.

        Returns:
            The refined pair representation `[B, N, N, pair_size]`; the internal MSA state is not returned.
        """

        features = [
            F.one_hot(msa_tokens, num_classes=self.vocab_size).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa_representation = self.msa_projection(torch.cat(features, dim=-1))
        msa_representation = msa_representation + self.token_projection(single_inputs).unsqueeze(1)

        pair_mask = pair_mask_from_token_mask(token_mask)
        for block in self.blocks:
            pair, msa_representation = block(pair, msa_representation, pair_mask, msa_mask)
        return pair


class MsaBlock(nn.Module):
    r"""
    One MSA block: pair-weighted averaging, MSA transition, outer-product mean, then a triangle pair update.

    The pair-weighted averaging widens to ``msa_head_dim`` (32, inner 256 for 8 heads) and the
    outer-product mean can divide before projecting (see
    [`OuterProductMean`][ofoldx.modules.trunk.msa.OuterProductMean]).

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        msa_head_dim: Per-head width of the MSA pair-weighted averaging.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Feed-forward expansion factor of every transition.
        divide_before_projection: Whether the outer-product mean divides before its projection.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        divide_before_projection: bool = True,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_dim=msa_head_dim,
            operator_backend=operator_backend,
        )
        self.msa_transition = Transition(msa_size, expansion=expansion, operator_backend=operator_backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            divide_before_projection=divide_before_projection,
            operator_backend=operator_backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            operator_backend=operator_backend,
        )
        self.msa_dropout = RowwiseDropout(msa_dropout)

    def forward(self, pair: Tensor, msa: Tensor, pair_mask: Tensor, msa_mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Apply the four residual updates: pair-weighted averaging and transition on the MSA, then outer-
        product mean and triangle pair update on the pair representation.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            msa: MSA representation `[B, S, N, msa_size]`.
            pair_mask: Pair validity mask `[B, N, N]`.
            msa_mask: MSA validity mask `[B, S, N]`.

        Returns:
            Updated ``(pair, msa)`` of shapes `[B, N, N, pair_size]` and `[B, S, N, msa_size]`.
        """

        msa = msa + self.msa_dropout(self.pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        msa = msa + self.msa_transition(msa)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return pair, msa


class MsaPairBlock(nn.Module):
    r"""
    Joint MSA and pair update block: outer-product mean, MSA update, then triangle pair update.

    Adds the outer-product mean of the MSA to the pair representation, updates the MSA against the new
    pair via an [`MsaUpdateBlock`][ofoldx.modules.trunk.msa.MsaUpdateBlock], then refines the pair with
    a [`TrianglePairUpdateBlock`][ofoldx.modules.trunk.msa.TrianglePairUpdateBlock].

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        num_msa_attention_heads: Number of heads in the MSA pair-weighted averaging.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Feed-forward expansion factor of every transition.
        msa_dropout: Row-wise dropout probability on the MSA residual.
        pair_dropout: Row-wise dropout probability on each pair residual.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.msa_update = MsaUpdateBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            expansion=expansion,
            dropout=msa_dropout,
            operator_backend=operator_backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Add the outer-product mean of the MSA to the pair, update the MSA against the new pair, then run
        the triangle pair update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated ``(msa, pair)`` of shapes `[B, S, N, msa_size]` and `[B, N, N, pair_size]`.
        """

        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        msa = self.msa_update(msa, pair, pair_mask=pair_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return msa, pair


class FinalMsaPairBlock(nn.Module):
    r"""
    Final MSA block that refines the pair representation only, leaving the MSA unchanged.

    Adds the outer-product mean of the MSA to the pair representation and runs the triangle pair update.
    The MSA passes through untouched, since the last block of the trunk emits no MSA update.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        outer_product_hidden_size: Hidden width of the outer-product mean.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of every transition.
        pair_dropout: Row-wise dropout probability on each pair residual.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        pair_dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.triangle_pair_update = TrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            expansion=expansion,
            dropout=pair_dropout,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        msa_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Add the outer-product mean of the MSA to the pair and run the triangle pair update.

        The MSA passes through unchanged, since the last block of the trunk emits no MSA update.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Tuple of the unchanged ``msa`` `[B, S, N, msa_size]` and the refined ``pair``
            `[B, N, N, pair_size]`.
        """

        pair = pair + self.outer_product_mean(msa, msa_mask=msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return msa, pair


class MsaUpdateBlock(nn.Module):
    r"""
    MSA update block: pair-weighted averaging residual followed by a transition residual.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation.
        num_attention_heads: Number of heads in the MSA pair-weighted averaging.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability on the pair-weighted averaging residual.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.15,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            operator_backend=operator_backend,
        )
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.transition = Transition(msa_size, expansion=expansion, operator_backend=operator_backend)

    def forward(self, msa: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Update the MSA with a row-dropout pair-weighted averaging residual, then a transition residual.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            The updated MSA representation `[B, S, N, msa_size]`.
        """

        msa = msa + self.rowwise_dropout(self.msa_pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        return msa + self.transition(msa)


class TrianglePairUpdateStack(nn.Module):
    r"""
    Apply a homogeneous sequence of pair-only triangle update blocks.

    Args:
        num_blocks: Number of triangle pair-update blocks.
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability applied to each residual.
        operator_backend: Operator backend selector forwarded to every block.
    """

    def __init__(
        self,
        num_blocks: int,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=pair_size,
                triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                expansion=expansion,
                dropout=dropout,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Move a pair representation through every homogeneous update block.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Updated pair representation `[B, N, N, pair_size]`.
        """

        for block in self.blocks:
            pair = block(pair, pair_mask=pair_mask)
        return pair


class TrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only triangle update block: triangle multiplication, triangle attention, then a transition.

    Runs outgoing then incoming triangle multiplication, starting- then ending-node triangle attention
    (each with row dropout), and a transition, all as residual updates on the pair representation.

    Args:
        pair_size: Width of the pair representation.
        triangle_multiplication_hidden_size: Hidden width of each triangle-multiplication update.
        triangle_attention_hidden_size: Per-head width of each triangle-attention update.
        num_triangle_attention_heads: Number of heads in triangle attention.
        expansion: Expansion factor of the transition.
        dropout: Row-wise dropout probability applied to each residual.
        operator_backend: Operator backend selector forwarded to every submodule.
    """

    def __init__(
        self,
        pair_size: int = 128,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=True,
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            outgoing=False,
            operator_backend=operator_backend,
        )
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            operator_backend=operator_backend,
        )
        self.rowwise_dropout = RowwiseDropout(dropout)
        self.transition = Transition(pair_size, expansion=expansion, operator_backend=operator_backend)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the five residual updates: outgoing then incoming triangle multiplication, starting- then
        ending-node triangle attention (each with row dropout), and a transition.

        The ending-node update transposes the residue axes ``(-3, -2)`` (and the pair mask's last two)
        around the attention call.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            The updated pair representation `[B, N, N, pair_size]`.
        """

        pair = self.triangle_update(pair, pair_mask=pair_mask)
        return pair + self.transition(pair)

    def triangle_update(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the four triangle residual branches before the final pair transition.

        Args:
            pair: Pair representation `[B, N, N, pair_size]`.
            pair_mask: Optional pair validity mask `[B, N, N]`.

        Returns:
            Pair representation after outgoing/incoming multiplication and starting/ending attention.
        """

        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask=pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask=pair_mask))
        end_pair_mask = None if pair_mask is None else pair_mask.transpose(-1, -2)
        pair = pair + self.rowwise_dropout(
            self.triangle_attention_ending_node(pair.transpose(-3, -2), pair_mask=end_pair_mask).transpose(
                -3,
                -2,
            ),
        )
        return pair


class MsaPairWeightedAveraging(nn.Module):
    r"""
    Pair-conditioned averaging for MSA representations.

    Averages MSA values along the query-residue axis with per-head weights softmaxed from the pair
    representation, then sigmoid-gates and projects back. The inner width is
    ``num_attention_heads * head_dim``; ``head_dim`` defaults to ``msa_size // num_attention_heads``, and a
    fixed larger ``head_dim`` widens the value/gate/output projections.

    Args:
        msa_size: Width of the MSA representation.
        pair_size: Width of the pair representation used to compute the per-head weights.
        num_attention_heads: Number of averaging heads.
        head_dim: Per-head width; defaults to ``msa_size // num_attention_heads`` when ``None``.
        operator_backend: Operator backend selector; resolves to ``"oops"`` or ``"eager"``.

    Raises:
        ValueError: If ``head_dim`` is ``None`` and ``msa_size`` is not divisible by
            ``num_attention_heads``.
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        head_dim: int | None = None,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        if head_dim is None:
            if msa_size % num_attention_heads != 0:
                raise ValueError(
                    f"msa_size ({msa_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                )
            head_dim = msa_size // num_attention_heads

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        inner_size = num_attention_heads * head_dim
        self.msa_norm = LayerNorm(msa_size, operator_backend=operator_backend)
        self.value_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, operator_backend=operator_backend)
        self.pair_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.gate_projection = nn.Linear(msa_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, msa_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

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
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            msa_norm_eps=self.msa_norm.eps,
            pair_norm_eps=self.pair_norm.eps,
            operator_backend=self.operator_backend,
        )


class OuterProductMean(nn.Module):
    r"""
    Outer-product mean projection from MSA representations to pair representations.

    Forms the mask-weighted mean over sequence rows of the outer product of the left/right MSA projections
    and projects it to a pair update. ``divide_before_projection`` selects when the per-pair sequence count
    normalises the outer product: ``False`` projects then divides by ``count + eps``; ``True`` divides by
    ``count.clamp(min=1)`` first, then projects once. Because ``output_projection`` is affine, the two place
    the linear bias differently.

    Args:
        msa_size: Width of the MSA representation.
        hidden_size: Hidden width of each of the left/right MSA projections.
        pair_size: Width of the produced pair update.
        eps: Floor added to (or applied via ``clamp`` on) the per-pair sequence count.
        divide_before_projection: Whether to divide by the sequence count before the output projection.
        operator_backend: Operator backend selector; resolves to ``"oops"`` or ``"eager"``.
    """

    def __init__(
        self,
        msa_size: int = 64,
        hidden_size: int = 32,
        pair_size: int = 128,
        eps: float = 1e-3,
        divide_before_projection: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.operator_backend = resolve_config_operator_backend(
            operator_backend, supported=("oops", "eager"), public_default="eager"
        )
        self.eps = eps
        self.divide_before_projection = divide_before_projection
        self.msa_norm = LayerNorm(msa_size, operator_backend=operator_backend)
        self.left_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.right_projection = nn.Linear(msa_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size * hidden_size, pair_size)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Form the mask-weighted mean over sequence rows of the outer product of the left/right MSA
        projections and project it to a pair update.

        ``divide_before_projection`` and ``eps`` set the normalisation as documented on the class. The
        ``oops`` delegates to the accelerated implementation; ``eager`` uses ordinary PyTorch operations.

        Args:
            msa: MSA representation `[B, S, N, msa_size]`.
            msa_mask: Optional MSA validity mask `[B, S, N]` zeroing invalid rows and counting valid pairs.

        Returns:
            The pair contribution `[B, N, N, pair_size]`.
        """

        return dispatch_outer_product_mean(
            msa,
            msa_mask,
            norm_weight=self.msa_norm.weight,
            norm_bias=self.msa_norm.bias,
            proj_left_weight=self.left_projection.weight,
            proj_right_weight=self.right_projection.weight,
            proj_left_bias=self.left_projection.bias,
            proj_right_bias=self.right_projection.bias,
            proj_out_weight=self.output_projection.weight,
            proj_out_bias=self.output_projection.bias,
            norm_eps=self.msa_norm.eps,
            denom_eps=self.eps,
            divide_before_projection=self.divide_before_projection,
            operator_backend=self.operator_backend,
        )
